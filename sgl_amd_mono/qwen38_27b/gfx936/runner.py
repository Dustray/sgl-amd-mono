import functools
import logging
import os

import torch

from .config import accepted_features, decode_model_enabled, experimental_dense_enabled, full_attention_enabled, report_directory
from ..errors import ContractError, UnsupportedConfiguration
from .program import DenseK1, DenseK2
from .ffn import numerical_error
from .check import SHADOW_LIMITS, check_error, native_k1, native_k2, shadow_accepted
from ..report import implementation_digest, write_report


logger = logging.getLogger(__name__)


SGLANG_READS = {
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/attention/flashattention_backend.py": ("FlashAttentionBackend", "page_table", "cache_seqlens_int32"),
}


class MonoRuntime:
    def __init__(self, runner):
        self.accepted_features = accepted_features()
        if "dense_eager" not in self.accepted_features and not experimental_dense_enabled():
            raise UnsupportedConfiguration("Dense BF16 prototype requires explicit experimental opt-in")
        backbone = getattr(runner.model, "model", None)
        layers = getattr(backbone, "layers", None)
        if runner.ps.tp_size != 1 or layers is None or len(layers) != 64 or backbone.layers_to_capture:
            raise ContractError("Dense Monokernel requires TP1, 64 Qwen dense layers and no auxiliary hidden capture")
        self.enabled = False
        self.implementation_sha256 = implementation_digest()
        self.entries = []
        self.attention_entries = {}
        self.use_decode_io = decode_model_enabled()
        self.use_full_attention = full_attention_enabled()
        self.tail = None
        self.tail_pending = None
        self.decode_input_ids = None
        self.embedding = backbone.embed_tokens if self.use_decode_io else None
        self.context_limit = int(runner.server_args.context_length)
        if self.use_full_attention and (self.context_limit > 4096 or runner.server_args.page_size != 64):
            raise ContractError("Full attention fusion requires context <=4096 and page size 64")
        if self.use_full_attention:
            if runner.server_args.attention_backend != "fa3" or runner.server_args.kv_cache_dtype != "fp8_e5m2":
                raise ContractError("Full attention serving requires the validated FA3 backend and unscaled FP8 E5M2 KV")
        self.validation = []
        self.shadow_checks = []
        self.shadow_phase = None
        self.shadow_native_outputs = {}
        self.shadow_native_inputs = {}
        self.shadow_attention_inputs = {}
        self.capture_shadow_inputs = os.environ.get("SGLANG_AMD_MONO_0518_CAPTURE_SHADOW_INPUTS", "0") == "1"
        self.shadow_layer_checks = []
        self.shadow_local_checks = []
        self.shadow_k2_inputs = {}
        self.shadow_limit = int(os.environ.get("SGLANG_AMD_MONO_0518_SHADOW_STEPS", "0"))
        self.validation_path = write_report(report_directory(), {"phase": "validating_real_weights", "checks": []}, "dense_real_weight_validation.json")
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
            kernel1_type = DenseK1
            if layer_index == 0 and self.use_decode_io:
                from .decode_io import DenseEntryK1

                kernel1_type = DenseEntryK1
            kernel1 = kernel1_type(layer.input_layernorm.weight, attention.in_proj_qkvz.weight, attention.in_proj_ba.weight, attention.conv1d.weight, attention.norm.weight, attention.A_log, attention.dt_bias) if is_gdn else None
            if not is_gdn and self.use_full_attention:
                from .attention import DenseAttentionK1

                if layer.qkv_proj.bias is not None or type(layer.qkv_proj.quant_method).__name__ != "UnquantizedLinearMethod" or not layer.attn_output_gate or layer.attn.k_scale is not None or layer.attn.v_scale is not None:
                    raise ContractError("Full attention fusion requires unbiased BF16 QKV, output gating and unscaled KV")
                rotary = layer.rotary_emb
                if not rotary.is_neox_style or getattr(rotary, "mrope_interleaved_glm", False):
                    raise ContractError("Full attention fusion requires Qwen NeoX MRoPE")
                self.attention_entries[layer.layer_id] = DenseAttentionK1(layer.input_layernorm.weight, layer.qkv_proj.weight, layer.q_norm.weight, layer.k_norm.weight, rotary.rotary_dim, rotary.mrope_section, rotary.mrope_interleaved)
            try:
                self._validate_layer(layer_index, kernel1, kernel2)
                if layer.layer_id in self.attention_entries:
                    self._validate_attention_layer(layer_index, layer.rotary_emb)
            except ContractError as error:
                write_report(report_directory(), {"passed": False, "failed_layer": layer_index, "error": str(error), "checks": self.validation}, "dense_real_weight_validation.json")
                raise
            self.entries.append((layer, layer.forward, kernel1, kernel2))
            if layer_index % 8 == 7:
                logger.warning("AMD_DENSE_MONO real-weight validation %d/64 layers", layer_index + 1)
        if sum(kernel1 is not None for _, _, kernel1, _ in self.entries) != 48:
            raise ContractError("Dense Monokernel expected 48 GDN and 16 full-attention layers")
        self.synchronization_errors = [(layer.layer_id, kind, kernel.synchronization[2]) for layer, _, kernel1, kernel2 in self.entries for kind, kernel in (("K1", kernel1), ("K2", kernel2)) if kernel is not None]
        self.synchronization_errors.extend((layer, "AttentionK1", kernel.synchronization[2]) for layer, kernel in self.attention_entries.items())
        if self.use_decode_io:
            from .decode_io import DenseTail

            processor = runner.model.logits_processor
            if processor.use_fp32_lm_head or runner.model.lm_head.weight.shape != (248320, 5120) or backbone.norm.variance_epsilon != 1.0e-6:
                raise ContractError("Complete decode fusion requires the validated BF16 LM head and final Gemma norm")
            self.tail = DenseTail(backbone.norm.weight, runner.model.lm_head.weight)
            self._validate_tail()
            self.synchronization_errors.append((64, "Tail", self.tail.synchronization[2]))
        write_report(report_directory(), {"passed": True, "checks": self.validation, "limits": {"K1_output": 0.003, "K1_state": 0.003, "K2_output": 0.003}}, "dense_real_weight_validation.json")
        for layer, original, kernel1, kernel2 in self.entries:
            layer.forward = self._wrap_layer(layer, original, kernel1, kernel2)
            if kernel1 is not None:
                kernel1.executions.zero_()
            kernel2.executions.zero_()
        for kernel in self.attention_entries.values():
            kernel.executions.zero_()
        if self.use_decode_io:
            self._wrap_decode_io(backbone, runner.model)
            self.tail.executions.zero_()
        torch.cuda.synchronize()
        logger.warning("AMD_DENSE_MONO attached 48 GDN K1, %d attention K1, 64 K2 and %d tail kernels; %d cooperative launches per decode forward; complete_decode_model=%s", len(self.attention_entries), int(self.tail is not None), 112 + len(self.attention_entries) + int(self.tail is not None), self.use_decode_io)

    def _validate_tail(self):
        from .check import native_norm

        generator = torch.Generator(device=self.tail.device).manual_seed(20261105)
        for rows in (1, 2):
            hidden = torch.randn((rows, 5120), device=self.tail.device, dtype=torch.bfloat16, generator=generator)
            residual = torch.randn(hidden.shape, device=self.tail.device, dtype=hidden.dtype, generator=generator)
            normalized, _ = native_norm(hidden, residual, self.tail.weights[0])
            expected = torch.nn.functional.linear(normalized, self.tail.weights[1])
            actual = self.tail(hidden, residual)
            self.validation.append({"layer": 64, "rows": rows, "kernel": "Tail", "output": check_error(actual, expected, 0.003, "final norm and LM head"), "normalized": check_error(self.tail.normalized[:rows], normalized, 0.0, "final norm")})

    def _wrap_decode_io(self, backbone, model):
        original_embedding = backbone.embed_tokens.forward
        original_norm = backbone.norm.forward
        original_head = model.logits_processor._compute_lm_head

        @functools.wraps(original_embedding)
        def embedding(input_ids, *args, **kwargs):
            if not self.enabled:
                return original_embedding(input_ids, *args, **kwargs)
            self.decode_input_ids = input_ids
            return torch.empty((input_ids.shape[0], 5120), device=input_ids.device, dtype=torch.bfloat16)

        @functools.wraps(original_norm)
        def norm(hidden, residual=None, *args, **kwargs):
            if not self.enabled:
                return original_norm(hidden, residual, *args, **kwargs)
            self.tail_pending = hidden, residual
            return hidden if residual is None else (hidden, residual)

        @functools.wraps(original_head)
        def head(hidden, lm_head, embedding_bias=None):
            if not self.enabled:
                return original_head(hidden, lm_head, embedding_bias)
            if self.tail_pending is None or lm_head is not model.lm_head or embedding_bias is not None:
                raise ContractError("Complete decode tail contract drift before LM head")
            pending_hidden, residual = self.tail_pending
            if hidden.shape != pending_hidden.shape:
                raise ContractError("Complete decode tail cannot reorder final hidden rows")
            if self.shadow_phase == "fused":
                from .check import native_norm

                normalized, _ = native_norm(hidden, residual, self.tail.weights[0])
                reference = torch.nn.functional.linear(normalized, self.tail.weights[1])
            output = self.tail(hidden, residual)
            self.tail_pending = None
            if self.shadow_phase == "fused":
                self.shadow_local_checks.append({"layer": 64, "kernel": "Tail", "output": numerical_error(output, reference), "normalized": numerical_error(self.tail.normalized[:hidden.shape[0]], normalized)})
            return output

        backbone.embed_tokens.forward = embedding
        backbone.norm.forward = norm
        model.logits_processor._compute_lm_head = head

    def _validate_layer(self, layer_index, kernel1, kernel2):
        generator = torch.Generator(device=kernel2.device).manual_seed(20261105 + layer_index)
        for rows in (1, 2):
            hidden = torch.randn((rows, 5120), device=kernel2.device, dtype=torch.bfloat16, generator=generator)
            residual = torch.randn(hidden.shape, device=kernel2.device, dtype=hidden.dtype, generator=generator)
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
        write_report(report_directory(), {"phase": "validating_real_weights", "validated_layers": layer_index + 1, "checks": self.validation}, "dense_real_weight_validation.json")

    def _validate_attention_layer(self, layer_index, rotary):
        from .attention_check import ATTENTION_LIMITS, check_fp32_reference, fp32_attention, native_attention, precision_attention

        kernel = self.attention_entries[layer_index]
        cache = rotary.cos_sin_cache.to(dtype=torch.bfloat16, device=kernel.device)
        generator = torch.Generator(device=kernel.device).manual_seed(20261105 + layer_index)
        for rows in (1, 2):
            hidden = torch.randn((rows, 5120), device=kernel.device, dtype=torch.bfloat16, generator=generator)
            residual = torch.randn(hidden.shape, device=kernel.device, dtype=hidden.dtype, generator=generator)
            key = torch.randn((4, 64, 4, 256), device=kernel.device, dtype=hidden.dtype, generator=generator).to(torch.float8_e5m2)
            value = torch.randn((4, 64, 4, 256), device=kernel.device, dtype=hidden.dtype, generator=generator).to(torch.float8_e5m2)
            reference_key, reference_value = key.clone(), value.clone()
            table = torch.tensor([[1, 2], [3, 0]][:rows], device=kernel.device, dtype=torch.int32)
            lengths = torch.tensor([65, 47][:rows], device=kernel.device, dtype=torch.int32)
            positions = (lengths - 1).long().repeat(3, 1)
            positions[1].add_(3)
            positions[2].add_(7)
            locations = table[torch.arange(rows, device=kernel.device), (lengths.long() - 1) // 64].long() * 64 + (lengths.long() - 1) % 64
            expected, expected_residual, _ = native_attention(kernel, hidden, residual, positions, cache, reference_key, reference_value, locations, table, lengths)
            from .attention_check import native_attention_prepare

            stages, _ = native_attention_prepare(kernel, hidden, residual, positions, cache)
            fp32_reference = fp32_attention(stages, reference_key, reference_value, table, lengths)
            precision_reference = precision_attention(stages, reference_key, reference_value, table, lengths)
            actual, actual_residual = kernel(hidden, residual, positions, cache, key, value, locations, table, lengths, 128)
            selected_pages, selected_offsets = locations // 64, locations % 64
            fp32_error, native_fp32_error = check_fp32_reference(actual, expected, fp32_reference)
            self.validation.append({"layer": layer_index, "rows": rows, "kernel": "AttentionK1", "output": check_error(actual, expected, ATTENTION_LIMITS["native_fa3_nrmse_max"], "full attention FA3 output"), "precision_output": check_error(actual, precision_reference, ATTENTION_LIMITS["precision_reference_nrmse_max"], "full attention precision reference"), "fp32_output": fp32_error, "fa3_fp32_output": native_fp32_error, "residual": check_error(actual_residual, expected_residual, 0.0, "full attention residual"), "key_cache": check_error(key[selected_pages, selected_offsets].float(), reference_key[selected_pages, selected_offsets].float(), 0.003, "full attention key write"), "value_cache": check_error(value[selected_pages, selected_offsets].float(), reference_value[selected_pages, selected_offsets].float(), 0.003, "full attention value write")})

    def _wrap_layer(self, layer, original, kernel1, kernel2):
        skip_projection = [False]
        projection = layer.o_proj if kernel1 is None else layer.linear_attn.out_proj
        original_projection = projection.forward

        @functools.wraps(original_projection)
        def output_projection(hidden, *args, **kwargs):
            if self.shadow_phase == "native" and self.capture_shadow_inputs:
                self.shadow_native_inputs[layer.layer_id]["core"] = hidden.clone()
            if skip_projection[0]:
                return hidden, None
            return original_projection(hidden, *args, **kwargs)

        projection.forward = output_projection

        @functools.wraps(original)
        def forward(positions, hidden_states, forward_batch, residual=None, captured_last_layer_outputs=None):
            if not self.enabled:
                if self.shadow_phase == "native" and self.capture_shadow_inputs:
                    captured = {"hidden": hidden_states.clone(), "residual": residual.clone() if residual is not None else None}
                    if kernel1 is not None:
                        from sglang.srt.model_executor.forward_context import get_attn_backend

                        backend = get_attn_backend().linear_attn_backend
                        cache = backend.req_to_token_pool.mamba2_layer_cache(layer.linear_attn.layer_id)
                        indices = backend.forward_metadata.mamba_cache_indices
                        active = indices[indices >= 0].long().unique()
                        captured.update(conv_state=cache.conv[0].index_select(0, active), ssm_state=cache.temporal.index_select(0, active), indices=torch.where(indices >= 0, torch.searchsorted(active, indices.long()), -1))
                    self.shadow_native_inputs[layer.layer_id] = captured
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
                    reference_hidden = self.embedding.weight[self.decode_input_ids] if layer.layer_id == 0 and getattr(self, "use_decode_io", False) else hidden_states
                    reference_core, _, reference_stages = native_k1(kernel1, reference_hidden, residual, reference_conv, reference_ssm, indices)
                if layer.layer_id == 0 and getattr(self, "use_decode_io", False):
                    core, residual = kernel1(hidden_states, residual, cache.conv[0], cache.temporal, indices, self.embedding.weight, self.decode_input_ids)
                else:
                    core, residual = kernel1(hidden_states, residual, cache.conv[0], cache.temporal, indices)
                if self.shadow_phase == "fused":
                    active_indices = indices[indices >= 0].long().unique()
                    local_check = {"layer": layer.layer_id, "kernel": "K1", "output": numerical_error(core, reference_core), "ssm_state": numerical_error(cache.temporal.index_select(0, active_indices), reference_ssm.index_select(0, active_indices)), "stages": {name: numerical_error(getattr(kernel1, name)[:hidden_states.shape[0]], expected) for name, expected in reference_stages.items()}}
                    if layer.layer_id == 0 and getattr(self, "use_decode_io", False):
                        local_check["embedding"] = numerical_error(hidden_states, reference_hidden)
                    self.shadow_local_checks.append(local_check)
                backend._track_mamba_state_decode(forward_batch, cache.conv[0], cache.temporal, indices, layer.linear_attn.layer_id)
            elif layer.layer_id in getattr(self, "attention_entries", {}):
                from sglang.srt.model_executor.forward_context import get_attn_backend
                from .attention import cache_views

                attention_backend = get_attn_backend().full_attn_backend
                if type(attention_backend).__name__ != "FlashAttentionBackend" or attention_backend.page_size != 64 or attention_backend.kv_cache_dtype not in (torch.bfloat16, torch.float8_e5m2):
                    raise ContractError("Full attention fusion requires the admitted FA3 paged KV backend")
                metadata = attention_backend.forward_metadata
                key, value = attention_backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
                layout = "legacy" if attention_backend._use_hcu_legacy_layout else "hnd" if attention_backend._use_hcu_bhsd else "nhd"
                key, value = cache_views(key, value, 64, layout)
                attention_kernel = self.attention_entries[layer.layer_id]
                if self.shadow_phase == "fused":
                    from .attention_check import fp32_attention, native_attention_prepare, precision_attention

                    normalized, _ = layer.input_layernorm(hidden_states.clone(), residual.clone())
                    skip_projection[0] = True
                    try:
                        reference_core = layer.self_attention(positions=positions, hidden_states=normalized, forward_batch=forward_batch)
                    finally:
                        skip_projection[0] = False
                    reference_stages, _ = native_attention_prepare(attention_kernel, hidden_states, residual, positions, layer.rotary_emb.cos_sin_cache)
                    fp32_reference = fp32_attention(reference_stages, key, value, metadata.page_table, metadata.cache_seqlens_int32)
                    precision_reference = precision_attention(reference_stages, key, value, metadata.page_table, metadata.cache_seqlens_int32)
                    active = metadata.cache_seqlens_int32 > 0
                    locations = forward_batch.out_cache_loc[active].long()
                    reference_key = key[locations // 64, locations % 64].float().clone()
                    reference_value = value[locations // 64, locations % 64].float().clone()
                attention_residual = residual
                core, residual = attention_kernel(hidden_states, residual, positions, layer.rotary_emb.cos_sin_cache, key, value, forward_batch.out_cache_loc, metadata.page_table, metadata.cache_seqlens_int32, self.context_limit, request_indices=forward_batch.req_pool_indices)
                if self.shadow_phase == "fused":
                    self.shadow_local_checks.append({"layer": layer.layer_id, "kernel": "AttentionK1", "output": numerical_error(core[active], reference_core[active]), "precision_output": numerical_error(core[active], precision_reference[active]), "fp32_output": numerical_error(core[active], fp32_reference[active]), "fa3_fp32_output": numerical_error(reference_core[active], fp32_reference[active]), "key_cache": numerical_error(key[locations // 64, locations % 64].float(), reference_key), "value_cache": numerical_error(value[locations // 64, locations % 64].float(), reference_value), "stages": {name: numerical_error(getattr(attention_kernel, name)[:hidden_states.shape[0]], expected) for name, expected in reference_stages.items()}})
                    if self.capture_shadow_inputs:
                        self.shadow_attention_inputs[layer.layer_id] = {"hidden": hidden_states, "residual": attention_residual, "positions": positions, "rotary_cache": layer.rotary_emb.cos_sin_cache, "key_cache": key, "value_cache": value, "locations": forward_batch.out_cache_loc, "page_table": metadata.page_table, "lengths": metadata.cache_seqlens_int32, "request_indices": forward_batch.req_pool_indices, "expected": reference_core, "actual": core, "stages": reference_stages}
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
                self.shadow_k2_inputs[layer.layer_id] = (core.clone(), residual.clone())
                reference_local, _, reference_stages = native_k2(kernel2, core, residual)
            result = kernel2(core, residual)
            if self.shadow_phase == "fused":
                self.shadow_local_checks.append({"layer": layer.layer_id, "kernel": "K2", "output": numerical_error(result[0], reference_local), "stages": {name: numerical_error(getattr(kernel2, name)[:core.shape[0]], expected) for name, expected in reference_stages.items()}})
                reference_hidden, reference_residual = self.shadow_native_outputs[layer.layer_id]
                self.shadow_layer_checks.append({"layer": layer.layer_id, "hidden": numerical_error(result[0], reference_hidden), "residual": numerical_error(result[1], reference_residual)})
            return result

        return forward

    def supports(self, batch):
        eligible = batch.forward_mode.is_decode() and 1 <= int(batch.batch_size) <= 2
        if eligible and self.use_decode_io and getattr(batch, "input_embeds", None) is not None:
            return False
        if eligible and self.use_full_attention:
            from sglang.srt.model_executor.forward_context import get_attn_backend

            backend = get_attn_backend().full_attn_backend
            if type(backend).__name__ != "FlashAttentionBackend" or backend.page_size != 64 or backend.kv_cache_dtype != torch.float8_e5m2:
                raise ContractError("Full attention backend drift detected before model state updates")
        return eligible

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
        reference_logits = reference.next_token_logits.clone()
        reference_states = [(cache.conv[0].index_select(0, active), cache.temporal.index_select(0, active)) for cache, _, _ in snapshots]
        for cache, convolution, temporal in snapshots:
            cache.conv[0].index_copy_(0, active, convolution)
            cache.temporal.index_copy_(0, active, temporal)
        self.enabled = True
        self.shadow_phase = "fused"
        actual = original(*args, **kwargs)
        actual_logits = actual.next_token_logits.clone()
        self.shadow_phase = None
        native_logprobs = reference_logits.float().log_softmax(-1)
        fused_logprobs = actual.next_token_logits.float().log_softmax(-1)
        kl_divergence = (native_logprobs.exp() * (native_logprobs - fused_logprobs)).sum(-1)
        check = {"step": len(self.shadow_checks), "rows": int(batch.batch_size), "logits": numerical_error(actual.next_token_logits, reference_logits), "argmax_equal": (actual.next_token_logits.argmax(-1) == reference_logits.argmax(-1)).tolist(), "kl_divergence_nats_per_row": kl_divergence.tolist(), "state_checks": [], "layer_checks": self.shadow_layer_checks, "local_checks": self.shadow_local_checks, "full_attention_fused": bool(self.attention_entries), "complete_decode_model": self.use_decode_io}
        check["logits_per_row"] = [numerical_error(actual_row, reference_row) for actual_row, reference_row in zip(actual.next_token_logits, reference_logits)]
        for (cache, _, _), (reference_conv, reference_ssm) in zip(snapshots, reference_states):
            check["state_checks"].append({"conv": numerical_error(cache.conv[0].index_select(0, active), reference_conv), "ssm": numerical_error(cache.temporal.index_select(0, active), reference_ssm)})
        self.shadow_checks.append(check)
        passed = shadow_accepted(check)
        check["passed"] = passed
        if not passed:
            if self.capture_shadow_inputs:
                self._save_attention_replays(check)
            fused_states = [(cache.conv[0].index_select(0, active), cache.temporal.index_select(0, active)) for cache, _, _ in snapshots]
            for cache, convolution, temporal in snapshots:
                cache.conv[0].index_copy_(0, active, convolution)
                cache.temporal.index_copy_(0, active, temporal)
            self.enabled = False
            repeated = original(*args, **kwargs)
            check["native_repeat_logits"] = numerical_error(repeated.next_token_logits, reference_logits)
            for (cache, _, _), (convolution, temporal) in zip(snapshots, fused_states):
                cache.conv[0].index_copy_(0, active, convolution)
                cache.temporal.index_copy_(0, active, temporal)
            self.enabled = True
            torch.save({"native_logits": reference_logits.cpu(), "fused_logits": actual_logits.cpu(), "k2_inputs": {layer: tuple(tensor.cpu() for tensor in tensors) for layer, tensors in self.shadow_k2_inputs.items()}, "input_ids": batch.input_ids.cpu(), "positions": batch.positions.cpu()}, report_directory() / "dense_failed_shadow_logits.pt")
            if self.capture_shadow_inputs:
                torch.save({"implementation_sha256": self.implementation_sha256, "step": check["step"], "layers": {layer: {**{name: tensor.cpu() if tensor is not None else None for name, tensor in inputs.items()}, "output": self.shadow_native_outputs[layer][0].cpu(), "residual_output": self.shadow_native_outputs[layer][1].cpu()} for layer, inputs in self.shadow_native_inputs.items()}}, report_directory() / "dense_failed_shadow_layers.pt")
        write_report(report_directory(), {"scope": "native and fused full-model decode from identical active GDN states and previous KV history; native evaluated first, fused result retained; logits accepted separately for each request", "implementation_sha256": self.implementation_sha256, "kernel_source_sha256": self.entries[0][3].program.manifest["source_sha256"], "acceptance": SHADOW_LIMITS, "checks": self.shadow_checks, "passed": all(item["passed"] for item in self.shadow_checks)}, "dense_model_shadow_validation.json")
        self.shadow_native_outputs.clear()
        self.shadow_native_inputs.clear()
        self.shadow_attention_inputs.clear()
        self.shadow_layer_checks = []
        self.shadow_local_checks = []
        self.shadow_k2_inputs.clear()
        if not passed:
            raise ContractError("Whole-model shadow validation failed; see dense_model_shadow_validation.json")
        return actual

    def _save_attention_replays(self, check):
        errors = sorted((item for item in check["local_checks"] if item["kernel"] == "AttentionK1"), key=lambda item: item["output"]["normalized_rms_error"], reverse=True)
        for error in errors[:4]:
            layer = error["layer"]
            inputs = self.shadow_attention_inputs[layer]
            page_count = (int(inputs["lengths"].max().item()) + 63) // 64
            table = inputs["page_table"][:, :page_count].long()
            selected = torch.cat((torch.zeros(1, device=table.device, dtype=torch.int64), table.flatten())).unique(sorted=True)
            locations = inputs["locations"].long()
            replay = {name: tensor.cpu() if tensor is not None else None for name, tensor in inputs.items() if name not in {"key_cache", "value_cache", "page_table", "locations", "stages"}}
            replay.update(layer=layer, implementation_sha256=self.implementation_sha256, step=check["step"], key_cache=inputs["key_cache"].index_select(0, selected).cpu(), value_cache=inputs["value_cache"].index_select(0, selected).cpu(), page_table=torch.searchsorted(selected, table).int().cpu(), locations=(torch.searchsorted(selected, locations // 64) * 64 + locations % 64).cpu(), stages={name: tensor.cpu() for name, tensor in inputs["stages"].items()}, error=error)
            torch.save(replay, report_directory() / f"dense_failed_attention_layer_{layer}.pt")

    def check_synchronization(self):
        errors = torch.stack([state for _, _, state in self.synchronization_errors])
        if int(errors.max().item()):
            failures = [{"layer": layer, "kernel": kind, "barrier_stage": value} for (layer, kind, _), value in zip(self.synchronization_errors, errors.tolist()) if value]
            write_report(report_directory(), {"passed": False, "failures": failures, "stateful_fallback_allowed": False}, "dense_synchronization_failure.json")
            raise ContractError(f"Dense Monokernel grid synchronization failed; refusing partially updated state: {failures}")

    def snapshot(self):
        torch.cuda.synchronize()
        self.check_synchronization()
        kernel1_counts = [int(kernel1.executions.item()) if kernel1 is not None else 0 for _, _, kernel1, _ in self.entries]
        kernel2_counts = [int(kernel2.executions.item()) for _, _, _, kernel2 in self.entries]
        attention_counts = {str(layer): int(kernel.executions.item()) for layer, kernel in self.attention_entries.items()}
        tail_count = int(self.tail.executions.item()) if self.tail is not None else 0
        program = self.entries[0][3].program
        return {
            "scope": "dense decode: GDN K1 and all K2, plus full-attention QKV/QK norm/MRoPE/paged KV/softmax/gate when full_attention_fused=true",
            "execution_path": "dense_bf16_cooperative_monokernel",
            "implementation_sha256": self.implementation_sha256,
            "accepted_features": sorted(self.accepted_features),
            "dense_decode_monokernel": True,
            "whole_model_monokernel": self.use_decode_io and len(self.attention_entries) == 16,
            "complete_decode_model": self.use_decode_io and len(self.attention_entries) == 16,
            "model_forward_modes": ["decode"],
            "decode_io_codegen": self.tail.program.manifest if self.tail is not None else None,
            "attention_codegen": next(iter(self.attention_entries.values())).program.manifest if self.attention_entries else None,
            "native_operations": ["prefill", "sampling", "scheduler_and_cache_state_tracking"] + ([] if self.use_decode_io else ["embedding", "final_norm", "lm_head"]) + ([] if self.attention_entries else ["full_attention_core"]),
            "full_attention_fused": bool(self.attention_entries),
            "all_decoder_layers_fused": len(self.attention_entries) == 16,
            "attention_k1_device_executions_per_layer": attention_counts,
            "tail_device_executions": tail_count,
            "monokernel_executions": sum(kernel1_counts) + sum(kernel2_counts) + sum(attention_counts.values()) + tail_count,
            "k1_device_executions_per_layer": kernel1_counts,
            "k2_device_executions_per_layer": kernel2_counts,
            "kernel_launches_per_decode_forward": 112 + len(self.attention_entries) + int(self.tail is not None),
            "gdn_layers": 48,
            "k2_layers": 64,
            "extra_weight_copies": 0,
            "compiler": "DTK HIP",
            "grid_synchronization": "device atomic epoch barrier with one-second deadline and persistent failure latch; cooperative launch admission checked by HIP occupancy",
            "eager_launch_api": program.manifest["launch_api"],
            "graph_launch_api": program.manifest["graph_capture_api"],
            "occupancy": program.information,
            "codegen_manifest": program.manifest_path,
            "binary_sha256": program.manifest["binary_sha256"],
            "real_weight_validation": str(self.validation_path),
            "model_shadow_steps": len(self.shadow_checks),
            "counter_scope": "GPU executions after attachment, including warmup and graph capture; compare snapshots for serving deltas",
        }
