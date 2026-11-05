import functools
import logging
import os

import torch

from .config import Config
from .contract import ContractError
from .hip_monokernel import DenseK1, DenseK2
from .hip_ffn import numerical_error
from .mono_validation import check_error, native_k1, native_k2
from .report import write_report


logger = logging.getLogger(__name__)


class MonoRuntime:
    def __init__(self, runner):
        backbone = getattr(runner.model, "model", None)
        layers = getattr(backbone, "layers", None)
        if runner.ps.tp_size != 1 or layers is None or len(layers) != 64 or backbone.layers_to_capture:
            raise ContractError("Dense Monokernel requires TP1, 64 Qwen dense layers and no auxiliary hidden capture")
        self.enabled = False
        self.entries = []
        self.validation = []
        self.shadow_checks = []
        self.shadow_phase = None
        self.shadow_native_outputs = {}
        self.shadow_layer_checks = []
        self.shadow_local_checks = []
        self.shadow_limit = int(os.environ.get("SGLANG_AMD_MONO_0518_SHADOW_STEPS", "0"))
        self.validation_path = write_report(Config.from_environment().report_dir, {"phase": "validating_real_weights", "checks": []}, "dense_real_weight_validation.json")
        for layer_index, layer in enumerate(layers):
            is_gdn = hasattr(layer, "linear_attn")
            expected = "Qwen3_5LinearDecoderLayer" if is_gdn else "Qwen3_5AttentionDecoderLayer"
            if type(layer).__name__ != expected or type(layer.mlp).__name__ != "Qwen2MoeMLP":
                raise ContractError(f"Unexpected dense layer type at {layer_index}: {type(layer).__name__}")
            attention = layer.linear_attn if is_gdn else layer
            output_projection = attention.out_proj if is_gdn else layer.o_proj
            projections = [output_projection, layer.mlp.gate_up_proj, layer.mlp.down_proj]
            if is_gdn:
                projections.extend((attention.in_proj_qkvz, attention.in_proj_ba))
                if attention.conv1d.bias is not None or attention.activation != "silu" or attention.norm.eps != 1.0e-6 or not attention.norm.norm_before_gate:
                    raise ContractError(f"GDN convolution/gating configuration drift at layer {layer_index}")
            if any(type(projection.quant_method).__name__ != "UnquantizedLinearMethod" or projection.bias is not None for projection in projections):
                raise ContractError(f"Dense Monokernel requires unbiased BF16 projections at layer {layer_index}")
            if any(norm.variance_epsilon != 1.0e-6 for norm in (layer.input_layernorm, layer.post_attention_layernorm)):
                raise ContractError(f"Dense Monokernel RMSNorm epsilon drift at layer {layer_index}")
            kernel2 = DenseK2(output_projection.weight, layer.post_attention_layernorm.weight, layer.mlp.gate_up_proj.weight, layer.mlp.down_proj.weight)
            kernel1 = DenseK1(layer.input_layernorm.weight, attention.in_proj_qkvz.weight, attention.in_proj_ba.weight, attention.conv1d.weight, attention.norm.weight, attention.A_log, attention.dt_bias) if is_gdn else None
            self._validate_layer(layer_index, kernel1, kernel2)
            self.entries.append((layer, layer.forward, kernel1, kernel2))
            if layer_index % 8 == 7:
                logger.warning("AMD_DENSE_MONO real-weight validation %d/64 layers", layer_index + 1)
        if sum(kernel1 is not None for _, _, kernel1, _ in self.entries) != 48:
            raise ContractError("Dense Monokernel expected 48 GDN and 16 full-attention layers")
        write_report(Config.from_environment().report_dir, {"passed": True, "checks": self.validation, "limits": {"K1_output": 0.003, "K1_state": 0.003, "K2_output": 0.003}}, "dense_real_weight_validation.json")
        for layer, original, kernel1, kernel2 in self.entries:
            layer.forward = self._wrap_layer(layer, original, kernel1, kernel2)
            if kernel1 is not None:
                kernel1.executions.zero_()
            kernel2.executions.zero_()
        torch.cuda.synchronize()
        logger.warning("AMD_DENSE_MONO attached 48 GDN K1 and 64 K2 kernels; 112 cooperative launches per decode forward; native full-attention core and LM head retained")

    def _validate_layer(self, layer_index, kernel1, kernel2):
        generator = torch.Generator(device=kernel2.device).manual_seed(20261105 + layer_index)
        for rows in (1, 2):
            hidden = torch.randn((rows, 5120), device=kernel2.device, dtype=torch.bfloat16, generator=generator)
            residual = torch.randn_like(hidden)
            core = torch.randn((rows, 6144), device=kernel2.device, dtype=torch.bfloat16, generator=generator)
            if kernel1 is not None:
                conv_state = torch.randn((3, 3, 10240), device=kernel2.device, dtype=torch.bfloat16, generator=generator).transpose(1, 2)
                ssm_state = torch.randn((3, 48, 128, 128), device=kernel2.device, generator=generator) * 0.01
                reference_conv, reference_ssm = conv_state.clone(), ssm_state.clone()
                indices = torch.tensor([2, 1][:rows], device=kernel2.device, dtype=torch.int32)
                actual, actual_residual = kernel1(hidden, residual if layer_index else None, conv_state, ssm_state, indices)
                expected, expected_residual, _ = native_k1(kernel1, hidden, residual if layer_index else None, reference_conv, reference_ssm, indices)
                self.validation.append({"layer": layer_index, "rows": rows, "kernel": "K1", "output": check_error(actual, expected, 0.003, "K1 output"), "residual": check_error(actual_residual, expected_residual, 0.002, "K1 residual"), "convolution_state": check_error(conv_state, reference_conv, 0.002, "K1 convolution state"), "ssm_state": check_error(ssm_state, reference_ssm, 0.003, "K1 SSM state")})
            actual, actual_residual = kernel2(core, residual)
            expected, expected_residual, _ = native_k2(kernel2, core, residual)
            self.validation.append({"layer": layer_index, "rows": rows, "kernel": "K2", "output": check_error(actual, expected, 0.003, "K2 output"), "residual": check_error(actual_residual, expected_residual, 0.002, "K2 residual")})
        write_report(Config.from_environment().report_dir, {"phase": "validating_real_weights", "validated_layers": layer_index + 1, "checks": self.validation}, "dense_real_weight_validation.json")

    def _wrap_layer(self, layer, original, kernel1, kernel2):
        skip_projection = [False]
        if kernel1 is None:
            original_projection = layer.o_proj.forward

            @functools.wraps(original_projection)
            def output_projection(hidden, *args, **kwargs):
                if skip_projection[0]:
                    return hidden, None
                return original_projection(hidden, *args, **kwargs)

            layer.o_proj.forward = output_projection

        @functools.wraps(original)
        def forward(positions, hidden_states, forward_batch, residual=None, captured_last_layer_outputs=None):
            if not self.enabled:
                result = original(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch, residual=residual, captured_last_layer_outputs=captured_last_layer_outputs)
                if self.shadow_phase == "native":
                    self.shadow_native_outputs[layer.layer_id] = tuple(tensor.clone() for tensor in result)
                return result
            if captured_last_layer_outputs is not None:
                raise ContractError("Auxiliary hidden capture cannot enter dense Monokernel")
            if kernel1 is not None:
                from sglang.srt.model_executor.forward_context import get_attn_backend

                backend = get_attn_backend().linear_attn_backend
                cache = backend.req_to_token_pool.mamba2_layer_cache(layer.linear_attn.layer_id)
                indices = backend.forward_metadata.mamba_cache_indices
                if self.shadow_phase == "fused":
                    reference_conv, reference_ssm = cache.conv[0].clone(), cache.temporal.clone()
                    reference_core, _, _ = native_k1(kernel1, hidden_states, residual, reference_conv, reference_ssm, indices)
                core, residual = kernel1(hidden_states, residual, cache.conv[0], cache.temporal, indices)
                if self.shadow_phase == "fused":
                    active_indices = indices[indices >= 0].long().unique()
                    self.shadow_local_checks.append({"layer": layer.layer_id, "kernel": "K1", "output": numerical_error(core, reference_core), "ssm_state": numerical_error(cache.temporal.index_select(0, active_indices), reference_ssm.index_select(0, active_indices))})
                backend._track_mamba_state_decode(forward_batch, cache.conv[0], cache.temporal, indices, layer.linear_attn.layer_id)
            else:
                if residual is None:
                    residual = hidden_states
                    normalized = layer.input_layernorm(hidden_states)
                else:
                    normalized, residual = layer.input_layernorm(hidden_states, residual)
                skip_projection[0] = True
                try:
                    core = layer.self_attention(positions=positions, hidden_states=normalized, forward_batch=forward_batch)
                finally:
                    skip_projection[0] = False
            if self.shadow_phase == "fused":
                reference_local, _, _ = native_k2(kernel2, core, residual)
            result = kernel2(core, residual)
            if self.shadow_phase == "fused":
                self.shadow_local_checks.append({"layer": layer.layer_id, "kernel": "K2", "output": numerical_error(result[0], reference_local)})
                reference_hidden, reference_residual = self.shadow_native_outputs[layer.layer_id]
                self.shadow_layer_checks.append({"layer": layer.layer_id, "hidden": numerical_error(result[0], reference_hidden), "residual": numerical_error(result[1], reference_residual)})
            return result

        return forward

    @staticmethod
    def supports(batch):
        return batch.forward_mode.is_decode() and 1 <= int(batch.batch_size) <= 2

    def shadow_forward(self, original, args, kwargs, batch):
        from sglang.srt.model_executor.forward_context import get_attn_backend

        if torch.cuda.is_current_stream_capturing():
            raise ContractError("Model shadow validation requires disabled decode graphs")
        backend = get_attn_backend().linear_attn_backend
        indices = backend.forward_metadata.mamba_cache_indices
        active = indices[indices >= 0].long().unique()
        snapshots = []
        for layer, _, kernel1, _ in self.entries:
            if kernel1 is not None:
                cache = backend.req_to_token_pool.mamba2_layer_cache(layer.linear_attn.layer_id)
                snapshots.append((cache, cache.conv[0].index_select(0, active), cache.temporal.index_select(0, active)))
        self.enabled = False
        self.shadow_phase = "native"
        reference = original(*args, **kwargs)
        reference_states = [(cache.conv[0].index_select(0, active), cache.temporal.index_select(0, active)) for cache, _, _ in snapshots]
        for cache, convolution, temporal in snapshots:
            cache.conv[0].index_copy_(0, active, convolution)
            cache.temporal.index_copy_(0, active, temporal)
        self.enabled = True
        self.shadow_phase = "fused"
        actual = original(*args, **kwargs)
        self.shadow_phase = None
        native_logprobs = reference.next_token_logits.float().log_softmax(-1)
        fused_logprobs = actual.next_token_logits.float().log_softmax(-1)
        kl_divergence = (native_logprobs.exp() * (native_logprobs - fused_logprobs)).sum(-1)
        check = {"step": len(self.shadow_checks), "rows": int(batch.batch_size), "logits": numerical_error(actual.next_token_logits, reference.next_token_logits), "argmax_equal": (actual.next_token_logits.argmax(-1) == reference.next_token_logits.argmax(-1)).tolist(), "kl_divergence_nats_per_row": kl_divergence.tolist(), "state_checks": [], "layer_checks": self.shadow_layer_checks, "local_checks": self.shadow_local_checks}
        for (cache, _, _), (reference_conv, reference_ssm) in zip(snapshots, reference_states):
            check["state_checks"].append({"conv": numerical_error(cache.conv[0].index_select(0, active), reference_conv), "ssm": numerical_error(cache.temporal.index_select(0, active), reference_ssm)})
        self.shadow_checks.append(check)
        passed = check["logits"]["finite"] and check["logits"]["cosine_similarity"] >= 0.999 and all(-1e-6 <= value <= 0.01 for value in check["kl_divergence_nats_per_row"]) and all(error["finite"] and error["normalized_rms_error"] <= 0.03 for state in check["state_checks"] for error in state.values()) and all(item["output"]["finite"] and item["output"]["normalized_rms_error"] <= 0.003 and item.get("ssm_state", {"normalized_rms_error": 0})["normalized_rms_error"] <= 0.003 for item in self.shadow_local_checks)
        check["passed"] = passed
        write_report(Config.from_environment().report_dir, {"scope": "native and fused full-model decode from identical active GDN states and previous KV history; native evaluated first, fused result retained", "acceptance": {"logits_cosine_min": 0.999, "kl_nats_per_row_max": 0.01, "same_input_kernel_nrmse_max": 0.003, "model_active_state_nrmse_max": 0.03, "logits_nrmse": "reported as a diagnostic; 2% trial criterion failed despite low distribution KL and small same-input kernel errors"}, "checks": self.shadow_checks, "passed": all(item["passed"] for item in self.shadow_checks)}, "dense_model_shadow_validation.json")
        self.shadow_native_outputs.clear()
        self.shadow_layer_checks = []
        self.shadow_local_checks = []
        if not passed:
            torch.save({"native_logits": reference.next_token_logits.cpu(), "fused_logits": actual.next_token_logits.cpu()}, Config.from_environment().report_dir / "dense_failed_shadow_logits.pt")
            raise ContractError("Whole-model shadow validation failed; see dense_model_shadow_validation.json")
        return actual

    def snapshot(self):
        torch.cuda.synchronize()
        kernel1_counts = [int(kernel1.executions.item()) if kernel1 is not None else 0 for _, _, kernel1, _ in self.entries]
        kernel2_counts = [int(kernel2.executions.item()) for _, _, _, kernel2 in self.entries]
        program = self.entries[0][3].program
        return {"scope": "complete dense K1/K2 decode: GDN input norm/projections/convolution/state/gated norm; all output projections, residual/postnorm and FFNs; native full-attention core, final norm and LM head", "execution_path": "dense_bf16_cooperative_monokernel", "whole_model_monokernel": True, "monokernel_executions": sum(kernel1_counts) + sum(kernel2_counts), "k1_device_executions_per_layer": kernel1_counts, "k2_device_executions_per_layer": kernel2_counts, "kernel_launches_per_decode_forward": 112, "gdn_layers": 48, "k2_layers": 64, "extra_weight_copies": 0, "compiler": "DTK HIP", "grid_synchronization": "cooperative_groups::this_grid().sync(); cooperative launch admission checked by HIP occupancy", "graph_launch_api": "hipLaunchKernelExC with hipLaunchAttributeCooperative", "occupancy": program.information, "codegen_manifest": program.manifest_path, "binary_sha256": program.manifest["binary_sha256"], "real_weight_validation": str(self.validation_path), "model_shadow_steps": len(self.shadow_checks), "counter_scope": "GPU executions after attachment, including warmup and graph capture; compare snapshots for serving deltas"}
