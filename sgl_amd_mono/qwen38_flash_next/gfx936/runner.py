import functools
import json
import logging
import os
from pathlib import Path

import torch

from .program import FlashContractError, HCMix, PeerGroup, PostAttention
from .gdn import GDNAttention
from .qsa import QSAAttention
from .ple import PLE
from .decode_io import DecodeIO


logger = logging.getLogger(__name__)


def error_metrics(actual, expected):
    actual = actual.float()
    expected = expected.float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    delta = actual - expected
    return {"finite": finite, "nrmse": float(delta.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12)), "max_absolute_error": float(delta.abs().max())}


def logits_metrics(actual, expected):
    actual = actual.float()
    expected = expected.float()
    reference_log_probability = expected.log_softmax(-1)
    actual_log_probability = actual.log_softmax(-1)
    reference_probability = reference_log_probability.exp()
    kl = (reference_probability * (reference_log_probability - actual_log_probability)).sum(-1).clamp_min(0)
    variation = (reference_probability - actual_log_probability.exp()).abs().sum(-1) * 0.5
    actual_top = actual.topk(2, dim=-1)
    expected_top = expected.topk(2, dim=-1)
    return {"finite": bool(torch.isfinite(kl).all() and torch.isfinite(variation).all()), "kl_reference_actual": kl.cpu().tolist(), "total_variation": variation.cpu().tolist(), "argmax_equal": (actual.argmax(-1) == expected.argmax(-1)).cpu().tolist(), "actual_token_ids": actual.argmax(-1).cpu().tolist(), "reference_token_ids": expected.argmax(-1).cpu().tolist(), "actual_top2_margin": (actual_top.values[:, 0] - actual_top.values[:, 1]).cpu().tolist(), "reference_top2_margin": (expected_top.values[:, 0] - expected_top.values[:, 1]).cpu().tolist()}


def logits_accepted(metrics):
    return metrics["finite"] and all(metrics["argmax_equal"]) and max(metrics["kl_reference_actual"]) <= 0.01 and max(metrics["total_variation"]) <= 0.05


class FlashRuntime:
    def __init__(self, runner):
        from sglang.srt.distributed import get_tp_group

        self.runner = runner
        self.model = runner.model.model
        self.device = self.model.layers[0].mlp.gate.weight.device
        self.rank = runner.ps.tp_rank
        self.peers = PeerGroup(self.device, self.rank, get_tp_group().cpu_group)
        self.enabled = False
        self.report_dir = Path(os.environ.get("SGLANG_AMD_FLASH_REPORT_DIR", "monokernel_reports/flash_next_tp8")).resolve()
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.validation = []
        self.communication_validation = []
        self.forward_keys = set()
        self.mixers = []
        self.posts = []
        self.gdn = []
        self.qsa = []
        self.ple = []
        self.gdn_validation = []
        self.shadow_checks = []
        self.full_model_shadow_checks = []
        self.serving_shadow_failure = None
        self.serving_shadow_failures = []
        self.reference_layer_checks = None
        self.layer_outputs = [None] * 48
        self.completed_decode_steps = 0
        self.counter_offsets = None
        self.tail_pending = None
        self.shadow_remaining = int(os.environ.get("SGLANG_AMD_FLASH_SHADOW_STEPS", "1"))
        if self.shadow_remaining < 1:
            raise FlashContractError("Flash-Next requires at least one serving shadow decode step")
        self.shadow_steps = self.shadow_remaining
        self.shadow_rearm_on_prefill = os.environ.get("SGLANG_AMD_FLASH_SHADOW_REARM_ON_PREFILL", "1") != "0"
        self.shadow_record_failures = os.environ.get("SGLANG_AMD_FLASH_SHADOW_RECORD_FAILURES", "0") == "1"
        self.failure_captures = set()
        self.shadow_remaining_by_batch = {}
        self.validated_batch_sizes = set()
        if self.model.start_layer != 0 or self.model.end_layer != 48 or len(self.model.layers) != 48:
            raise FlashContractError("Flash-Next runtime requires all 48 target layers")
        for layer in self.model.layers:
            hc = layer.attn_hyper_connection
            self.mixers.append(HCMix(hc.hc_norm.weight, hc.input_mix_weight_down.weight, hc.input_mix_weight_up.weight))
            self.mixers[-1].state = self.peers.state
            self.posts.append(PostAttention(layer, self.peers))
            self.gdn.append(GDNAttention(layer.linear_attn, self.mixers[-1]) if hasattr(layer, "linear_attn") else None)
            self.qsa.append(QSAAttention(layer, self.mixers[-1]) if hasattr(layer, "indexer") else None)
            self.ple.append(PLE(layer.ple, self.peers) if layer.ple is not None else None)
        hc = self.model.hyper_connection_mixer
        self.tail = HCMix(hc.hc_norm.weight, hc.input_mix_weight_down.weight, hc.input_mix_weight_up.weight)
        self.tail.state = self.peers.state
        self.io = DecodeIO(runner.model, self.tail, self.peers)
        self._validate_communication()
        self._validate_weights()
        self.counter_offsets = self.execution_counts()
        for layer_id, layer in enumerate(self.model.layers):
            self._wrap_layer(layer_id, layer)
        self._wrap_decode_io(runner.model)
        self._wrap_forward(runner.model)
        self.save()

    @staticmethod
    def native_post(layer, attention, residuals, batch=None):
        from sglang.srt.layers.dp_attention import attn_tp_all_reduce

        attention = attn_tp_all_reduce(attention.clone()).clone()
        after_attention = layer.attn_hyper_connection.combine(attention, residuals)
        mixed, mlp_residuals = layer.mlp_hyper_connection.mix(after_attention)
        moe = layer.mlp(mixed, batch).clone()
        output = layer.mlp_hyper_connection.combine(moe, mlp_residuals)
        return output, {"reduced_attention": attention, "post_attention": after_attention, "mixed": mixed, "reduced_moe": moe}

    def _validate_communication(self):
        import torch.distributed as distributed
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.layers.dp_attention import attn_tp_all_reduce

        generator = torch.Generator(device=self.device).manual_seed(38080 + self.rank)
        for rows in (1, 2):
            for width in (2560, 10240):
                for iteration in range(3):
                    values = torch.randn((rows, width), generator=generator, device=self.device)
                    values *= torch.pow(2.0, torch.randint(-8, 9, values.shape, generator=generator, device=self.device))
                    values = values.bfloat16()
                    expected = attn_tp_all_reduce(values.clone()).clone()
                    torch.cuda.synchronize(self.device)
                    distributed.barrier(group=get_tp_group().cpu_group)
                    actual = self.peers.reduce(values)
                    check = error_metrics(actual, expected)
                    self.communication_validation.append({"rows": rows, "width": width, "iteration": iteration, **check})
                    self.peers.check()
                    if not check["finite"] or check["max_absolute_error"]:
                        self.save()
                        raise FlashContractError(f"Flash-Next TP reduction differs from native AITER: {check}")
        logger.info("AMD_FLASH_TP8 native AITER reduction matched exactly on rank %s", self.rank)

    def _validate_weights(self):
        import torch.distributed as distributed
        from sglang.srt.distributed import get_tp_group

        generator = torch.Generator(device=self.device).manual_seed(38027)
        for layer_id, layer in enumerate(self.model.layers):
            for rows in (1, 2):
                hidden = torch.randn((rows, 10240), generator=generator, device=self.device, dtype=torch.bfloat16)
                attention = torch.randn((rows, 2560), generator=generator, device=self.device, dtype=torch.bfloat16) * 0.01
                native_mixed, residuals = layer.attn_hyper_connection.mix(hidden)
                torch.cuda.synchronize(self.device)
                distributed.barrier(group=get_tp_group().cpu_group)
                fused_mixed, fused_residuals = self.mixers[layer_id](hidden)
                if self.gdn[layer_id] is not None:
                    gdn = self.gdn[layer_id]
                    conv = torch.randn((4, 3, 1280), generator=generator, device=self.device, dtype=torch.bfloat16).transpose(1, 2)
                    ssm = torch.randn((4, 6, 128, 128), generator=generator, device=self.device, dtype=torch.float32) * 0.01
                    reference_conv, reference_ssm = conv.clone(), ssm.clone()
                    indices = torch.tensor([2, 1][:rows], device=self.device, dtype=torch.int32)
                    actual_gdn, _ = gdn(hidden, conv, ssm, indices)
                    expected_gdn = gdn.native(native_mixed, reference_conv, reference_ssm, indices)
                    gdn_checks = {"output": error_metrics(actual_gdn, expected_gdn), "conv": error_metrics(conv, reference_conv), "ssm": error_metrics(ssm, reference_ssm)}
                    self.gdn_validation.append({"layer": layer_id, "rows": rows, "checks": gdn_checks})
                    if any(not check["finite"] or check["nrmse"] > 0.006 for check in gdn_checks.values()):
                        self.save()
                        raise FlashContractError(f"Flash-Next GDN validation failed at layer {layer_id}: {gdn_checks}")
                    torch.cuda.synchronize(self.device)
                    distributed.barrier(group=get_tp_group().cpu_group)
                output = self.posts[layer_id](attention, residuals)
                expected, intermediates = self.native_post(layer, attention, residuals)
                native_moe_for_fused_input = layer.mlp(self.posts[layer_id].mixed[:rows])
                checks = {"attention_mix": error_metrics(fused_mixed, native_mixed), "attention_norm": error_metrics(fused_residuals[1], residuals[1]), "output": error_metrics(output, expected), "moe_same_input": error_metrics(self.posts[layer_id].reduced_moe[:rows], native_moe_for_fused_input)}
                for name, reference in intermediates.items():
                    checks[name] = error_metrics(getattr(self.posts[layer_id], name)[:rows], reference)
                self.peers.check()
                entry = {"layer": layer_id, "rows": rows, "checks": checks}
                self.validation.append(entry)
                self.save()
                failed = {name: check for name, check in checks.items() if not check["finite"] or check["nrmse"] > (0.02 if name in {"moe_same_input", "reduced_moe"} else 0.006)}
                if failed:
                    torch.save({"hidden": hidden, "attention": attention, "actual": output, "expected": expected, "fused_mixed": self.posts[layer_id].mixed[:rows], "reference_mixed": intermediates["mixed"], "actual_moe": self.posts[layer_id].reduced_moe[:rows], "expected_moe": native_moe_for_fused_input}, self.report_dir / f"failed-layer-{layer_id}-rows-{rows}-rank-{self.rank}.pt")
                    raise FlashContractError(f"Flash-Next real-weight validation failed: layer={layer_id} rows={rows}: {failed}")
            if layer_id % 6 == 0:
                logger.info("AMD_FLASH_TP8 validated layer %s/48 on rank %s", layer_id + 1, self.rank)
        logger.info("AMD_FLASH_TP8 all 48 layers passed real BF16 weight validation on rank %s", self.rank)

    def _wrap_decode_io(self, model):
        original_embedding = self.model.embed_tokens.forward
        original_mix = self.model.hyper_connection_mixer.mix
        original_head = model.logits_processor._compute_lm_head
        original_gather = model.logits_processor._logits_gatherer
        original_attn_gather = model.logits_processor._gather_attn_tp_logits

        @functools.wraps(original_embedding)
        def embedding(input_ids, *args, **kwargs):
            if not self.enabled:
                return original_embedding(input_ids, *args, **kwargs)
            result = self.io.entry(input_ids)
            if self.shadow_remaining:
                expected = original_embedding(input_ids, *args, **kwargs).repeat(1, 4)
                check = error_metrics(result, expected)
                self.shadow_checks.append({"kind": "embedding", "checks": check})
                if not check["finite"] or check["max_absolute_error"] != 0:
                    self.save()
                    raise FlashContractError(f"Flash-Next embedding serving shadow failed: {check}")
            return result

        @functools.wraps(original_mix)
        def mix(hidden):
            if not self.enabled:
                return original_mix(hidden)
            self.tail_pending = hidden
            return torch.empty((hidden.shape[0], 2560), dtype=torch.bfloat16, device=self.device), (hidden, hidden)

        @functools.wraps(original_head)
        def head(hidden, lm_head, embedding_bias=None):
            if not self.enabled:
                return original_head(hidden, lm_head, embedding_bias)
            if self.tail_pending is None or lm_head is not model.lm_head or embedding_bias is not None or hidden.shape != (self.tail_pending.shape[0], 2560):
                raise FlashContractError("Flash-Next fused tail cannot reorder rows or change LM head")
            result, mixed = self.io.tail(self.tail_pending)
            if self.shadow_remaining:
                native_mixed, _ = original_mix(self.tail_pending)
                native_local = original_head(native_mixed, lm_head, embedding_bias)
                expected = (original_attn_gather if model.logits_processor.use_attn_tp_group else original_gather)(native_local)
                checks = {"mixed": error_metrics(mixed, native_mixed), "logits": error_metrics(result, expected)}
                self.shadow_checks.append({"kind": "tail", "checks": checks, "argmax_equal": (result.argmax(-1) == expected.argmax(-1)).cpu().tolist()})
                if any(not check["finite"] or check["nrmse"] > 0.004 for check in checks.values()):
                    self._shadow_failure(f"Flash-Next tail serving shadow failed: {checks}")
            self.tail_pending = None
            return result

        def gather(original):
            @functools.wraps(original)
            def wrapped(logits):
                if not self.enabled:
                    return original(logits)
                if logits.shape != (logits.shape[0], 248320):
                    raise FlashContractError("Flash-Next cooperative logits gather returned the wrong vocabulary")
                return logits
            return wrapped

        self.model.embed_tokens.forward = embedding
        self.model.hyper_connection_mixer.mix = mix
        model.logits_processor._compute_lm_head = head
        model.logits_processor._logits_gatherer = gather(original_gather)
        model.logits_processor._gather_attn_tp_logits = gather(original_attn_gather)

    def _wrap_layer(self, layer_id, layer):
        original = layer.forward

        @functools.wraps(original)
        def forward(*args, **kwargs):
            if not self.enabled:
                result = original(*args, **kwargs)
                if self.reference_layer_checks is not None:
                    expected_hidden = result[0]
                    self.reference_layer_checks.append({"layer": layer_id, **error_metrics(self.layer_outputs[layer_id], expected_hidden)})
                return result
            batch = kwargs["forward_batch"]
            hidden = kwargs["hidden_states"]
            if hidden.shape[-1] == 2560:
                hidden = torch.cat([hidden] * 4, dim=-1)
            if layer.ple is not None:
                from sglang.srt.model_executor.forward_context import get_req_to_token_pool
                from sglang.srt.models.qwen4_exp import _ple_track_targets

                ple_batch = kwargs["ple_batch"]
                if not ple_batch.use_decode_fast_path or kwargs.get("residual") is not None:
                    raise FlashContractError("PLE fused decode requires one token per row and no pending residual")
                conv = get_req_to_token_pool().short_conv_layer_cache(layer.ple.layer_id)
                if self.shadow_remaining:
                    before_conv = conv.clone()
                output = self.ple[layer_id](hidden, ple_batch.ngram_context, ple_batch.state_indices, conv)
                if self.shadow_remaining:
                    actual_conv = conv.clone()
                    conv.copy_(before_conv)
                    expected = hidden + layer.ple(hidden, batch, ple_batch)
                    checks = {"output": error_metrics(output, expected), "conv": error_metrics(actual_conv, conv)}
                    conv.copy_(actual_conv)
                    self.shadow_checks.append({"kind": "ple", "layer": layer_id, "checks": checks})
                    if any(not check["finite"] or check["nrmse"] > 0.008 for check in checks.values()):
                        self._shadow_failure(f"Flash-Next PLE serving shadow failed at layer {layer_id}: {checks}")
                track = _ple_track_targets(batch, ple_batch)
                if track is not None:
                    track_indices, _ = track
                    conv[track_indices] = conv.index_select(0, ple_batch.state_indices)
                hidden = output
            if self.gdn[layer_id] is not None:
                from sglang.srt.model_executor.forward_context import get_attn_backend

                backend = get_attn_backend().linear_attn_backend
                cache = backend.req_to_token_pool.mamba2_layer_cache(layer.linear_attn.layer_id)
                indices = backend.forward_metadata.mamba_cache_indices
                if self.shadow_remaining:
                    reference_conv, reference_ssm = cache.conv[0].clone(), cache.temporal.clone()
                attention, residuals = self.gdn[layer_id](hidden, cache.conv[0], cache.temporal, indices)
                backend._track_mamba_state_decode(batch, cache.conv[0], cache.temporal, indices, layer.linear_attn.layer_id)
                if self.shadow_remaining:
                    actual_conv, actual_ssm = cache.conv[0].clone(), cache.temporal.clone()
                    cache.conv[0].copy_(reference_conv)
                    cache.temporal.copy_(reference_ssm)
                    mixed, _ = layer.attn_hyper_connection.mix(hidden)
                    fused_mixed = self.gdn[layer_id].mixed[:hidden.shape[0]]
                    reference_attention = layer.linear_attn(fused_mixed, batch)
                    checks = {"mixed": error_metrics(fused_mixed, mixed), "output": error_metrics(attention, reference_attention), "conv": error_metrics(actual_conv, cache.conv[0]), "ssm": error_metrics(actual_ssm, cache.temporal)}
                    cache.conv[0].copy_(actual_conv)
                    cache.temporal.copy_(actual_ssm)
                    self.shadow_checks.append({"kind": "gdn", "layer": layer_id, "checks": checks})
                    if any(not check["finite"] or check["nrmse"] > 0.006 for check in checks.values()):
                        torch.save({"hidden": hidden, "mixed": fused_mixed, "native_mixed": mixed, "actual": attention, "expected": reference_attention, "before_conv": reference_conv, "before_ssm": reference_ssm, "indices": indices, "projected": self.gdn[layer_id].projected, "projected_ba": self.gdn[layer_id].projected_ba, "recurrent": self.gdn[layer_id].recurrent, "core": self.gdn[layer_id].core}, self.report_dir / f"failed-gdn-shadow-layer-{layer_id}-rank-{self.rank}.pt")
                        self._shadow_failure(f"Flash-Next GDN serving shadow failed at layer {layer_id}: {checks}")
            else:
                from sglang.srt.model_executor.forward_context import get_attn_backend

                backend = get_attn_backend().full_attn_backend
                pool = backend.token_to_kv_pool
                key, value = pool.get_key_buffer(layer_id), pool.get_value_buffer(layer_id)
                ring, ring_positions, compressed = pool.get_qsa_key_state_buffer(layer_id), pool.qsa_rope_position_buffer, pool.get_qsa_compressed_k_buffer(layer_id)
                cache_tensors = (key, value, ring, ring_positions, compressed)
                if self.shadow_remaining:
                    before = [tensor.clone() for tensor in cache_tensors]
                attention, residuals = self.qsa[layer_id](hidden, kwargs["positions"], batch.seq_lens.to(torch.int32), batch.req_pool_indices, batch.out_cache_loc, backend.req_to_token_pool.req_to_token, key, value, ring, ring_positions, compressed)
                if self.shadow_remaining:
                    actual_cache = [tensor.clone() for tensor in cache_tensors]
                    for tensor, saved in zip(cache_tensors, before):
                        tensor.copy_(saved)
                    mixed, _ = layer.attn_hyper_connection.mix(hidden)
                    reference_attention = layer.self_attention(kwargs["positions"], mixed, batch)
                    checks = {"output": error_metrics(attention, reference_attention)}
                    for name, tensor, actual in zip(("key", "value", "ring", "ring_positions", "compressed"), cache_tensors, actual_cache):
                        checks[name] = error_metrics(actual[1:], tensor[1:]) if name == "compressed" else error_metrics(actual, tensor)
                        tensor.copy_(actual)
                    self.shadow_checks.append({"kind": "qsa", "layer": layer_id, "checks": checks})
                    failed = any(not check["finite"] or check["nrmse"] > 0.008 for check in checks.values())
                    capture_layers = {int(value) for value in os.environ.get("SGLANG_AMD_FLASH_CAPTURE_QSA_LAYERS", "47").split(",")}
                    capture = self.rank == 2 and layer_id in capture_layers and self.completed_decode_steps == int(os.environ.get("SGLANG_AMD_FLASH_CAPTURE_QSA_STEP", "-1"))
                    if capture or (failed and ("qsa", layer_id) not in self.failure_captures):
                        self._capture_qsa(layer_id, layer, hidden, kwargs["positions"], batch, backend, before, actual_cache, attention, mixed, reference_attention, checks)
                        self.failure_captures.add(("qsa", layer_id))
                    if failed:
                        self._shadow_failure(f"Flash-Next QSA serving shadow failed at layer {layer_id}: {checks}")
            output = self.posts[layer_id](attention, residuals)
            self.layer_outputs[layer_id] = output
            if self.shadow_remaining:
                expected, intermediates = self.native_post(layer, attention, residuals, batch)
                check = error_metrics(output, expected)
                checks = {"output": check}
                for name, reference in intermediates.items():
                    checks[name] = error_metrics(getattr(self.posts[layer_id], name)[:hidden.shape[0]], reference)
                self.shadow_checks.append({"kind": "post_attention", "layer": layer_id, "checks": checks})
                if not check["finite"] or check["nrmse"] > 0.006:
                    post = self.posts[layer_id]
                    native_logits, _ = layer.mlp.gate(post.mixed[:hidden.shape[0]])
                    native_topk = layer.mlp.topk(post.mixed[:hidden.shape[0]], native_logits)
                    torch.save({"hidden": hidden, "attention": attention, "residuals": residuals, "actual": output, "expected": expected, "intermediates": intermediates, "actual_mixed": post.mixed[:hidden.shape[0]], "actual_moe": post.reduced_moe[:hidden.shape[0]], "router_logits": post.router_logits[:hidden.shape[0]], "top_ids": post.top_ids[:hidden.shape[0]], "top_weights": post.top_weights[:hidden.shape[0]], "native_logits": native_logits, "native_top_ids": native_topk.topk_ids, "native_top_weights": native_topk.topk_weights}, self.report_dir / f"failed-shadow-layer-{layer_id}-rank-{self.rank}.pt")
                    self._shadow_failure(f"Flash-Next serving shadow failed at layer {layer_id}: {check}")
            return output, None

        layer.forward = forward

    def _shadow_failure(self, message):
        self.serving_shadow_failure = message
        self.serving_shadow_failures.append({"step": self.completed_decode_steps, "reason": message})
        self.save()
        if not self.shadow_record_failures:
            raise FlashContractError(message)
        logger.error("AMD_FLASH_CORRECTNESS_DIAGNOSTIC %s", message)

    def _capture_qsa(self, layer_id, layer, hidden, positions, batch, backend, before, actual_cache, attention, native_mixed, native_output, checks):
        kernel = self.qsa[layer_id]
        pool = backend.token_to_kv_pool
        cache_tensors = (pool.get_key_buffer(layer_id), pool.get_value_buffer(layer_id), pool.get_qsa_key_state_buffer(layer_id), pool.qsa_rope_position_buffer, pool.get_qsa_compressed_k_buffer(layer_id))
        captured = {"hidden": hidden.clone(), "positions": positions.clone(), "lengths": batch.seq_lens.to(torch.int32).clone(), "requests": batch.req_pool_indices.clone(), "locations": batch.out_cache_loc.clone(), "request_tokens": backend.req_to_token_pool.req_to_token.clone(), "cache_before": before, "cache_actual": actual_cache, "actual": attention.clone(), "mixed": kernel.mixed[:hidden.shape[0]].clone(), "native_mixed": native_mixed.clone(), "native_output": native_output.clone(), "checks": checks, "step": self.completed_decode_steps, "tp_rank": self.rank, "layer": layer_id, "program": kernel.program.manifest}
        for name in ("projected", "index_projected", "queries", "index_queries", "index_scores", "selected", "attention_scores", "core"):
            captured[name] = getattr(kernel, name)[:hidden.shape[0]].clone()
        originals = []

        def record(target, name, label, inputs=False):
            original = getattr(target, name)
            originals.append((target, name, original))

            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                if inputs:
                    captured[label] = args[0].clone()
                result = original(*args, **kwargs)
                if not inputs:
                    captured[label] = tuple(value.clone() if isinstance(value, torch.Tensor) else value for value in result) if isinstance(result, tuple) else result.clone()
                return result

            setattr(target, name, wrapped)

        try:
            for tensor, saved in zip(cache_tensors, before):
                tensor.copy_(saved)
            record(layer.qkv_proj, "forward", "native_projected")
            record(layer, "_prepare_qkv_gate", "native_qkv_gate")
            record(layer, "_compute_qsa_topk_indices", "native_selected")
            record(layer.o_proj, "forward", "native_core", inputs=True)
            expected = layer.self_attention(positions, captured["mixed"], batch)
            captured["matched_input_output"] = expected.clone()
            captured["matched_input_check"] = error_metrics(attention, expected)
            captured["matched_input_cache"] = [tensor.clone() for tensor in cache_tensors]
        finally:
            for target, name, original in reversed(originals):
                setattr(target, name, original)
            for tensor, saved in zip(cache_tensors, actual_cache):
                tensor.copy_(saved)
        weights = self.report_dir / f"qsa-weights-layer-{layer_id}-rank-{self.rank}.pt"
        if not weights.exists():
            torch.save({"mixer": tuple(weight.detach().clone() for weight in kernel.mixer.weights), "attention": tuple(weight.detach().clone() for weight in kernel.weights), "rotary_cache": kernel.rotary_cache.detach().clone()}, weights)
        captured["weights_file"] = weights.name
        destination = self.report_dir / f"qsa-step-{self.completed_decode_steps}-layer-{layer_id}-rank-{self.rank}.pt"
        torch.save(captured, destination)
        self.shadow_checks[-1]["matched_input_output"] = captured["matched_input_check"]
        self.shadow_checks[-1]["capture"] = str(destination)
        logger.info("AMD_FLASH_QSA_CAPTURE file=%s matched_input=%s", destination, captured["matched_input_check"])

    def supports(self, batch):
        return batch.forward_mode.is_decode() and batch.batch_size in (1, 2) and batch.input_ids.numel() == batch.batch_size and getattr(batch, "input_embeds", None) is None

    def _decode_states(self):
        from sglang.srt.model_executor.forward_context import get_attn_backend, get_req_to_token_pool

        backend = get_attn_backend()
        pool = get_req_to_token_pool()
        kv_pool = backend.full_attn_backend.token_to_kv_pool
        tensors = [("ngram_context", pool.ngram_pool.context), ("qsa_positions", kv_pool.qsa_rope_position_buffer)]
        for layer_id, layer in enumerate(self.model.layers):
            if self.gdn[layer_id] is not None:
                cache = backend.linear_attn_backend.req_to_token_pool.mamba2_layer_cache(layer.linear_attn.layer_id)
                tensors.extend(((f"conv.{layer_id}", cache.conv[0]), (f"ssm.{layer_id}", cache.temporal)))
            else:
                tensors.extend(((f"key.{layer_id}", kv_pool.get_key_buffer(layer_id)), (f"value.{layer_id}", kv_pool.get_value_buffer(layer_id)), (f"qsa_ring.{layer_id}", kv_pool.get_qsa_key_state_buffer(layer_id)), (f"qsa_compressed.{layer_id}", kv_pool.get_qsa_compressed_k_buffer(layer_id))))
            if layer.ple is not None:
                tensors.append((f"ple_conv.{layer_id}", pool.short_conv_layer_cache(layer.ple.layer_id)))
        return tensors

    def _full_model_shadow(self, original, args, kwargs, result, states, before):
        actual_states = [tensor.clone() for _, tensor in states]
        actual_logits = result.next_token_logits.clone()
        actual_hc = self.model.last_hc_hidden_states
        for (_, tensor), saved in zip(states, before):
            tensor.copy_(saved)
        self.enabled = False
        self.reference_layer_checks = []
        try:
            expected = original(*args, **kwargs)
            expected_logits = expected.next_token_logits.clone()
            check = error_metrics(actual_logits, expected_logits)
            distribution = logits_metrics(actual_logits, expected_logits)
            accepted = check["finite"] and logits_accepted(distribution)
            entry = {"step": self.completed_decode_steps, "rows": actual_logits.shape[0], "logits": check, "distribution": distribution, "argmax_equal": distribution["argmax_equal"], "accepted": accepted, "restored_state_tensors": len(states), "layers": self.reference_layer_checks, "native_repeats": []}
            batch = kwargs.get("forward_batch")
            if batch is None and len(args) > 2:
                batch = args[2]
            if batch is not None:
                entry["input_token_ids"] = batch.input_ids.cpu().tolist()
                entry["sequence_lengths"] = batch.seq_lens.cpu().tolist()
            self.full_model_shadow_checks.append(entry)
            self.reference_layer_checks = None
            for repeat in range(3):
                for (_, tensor), saved in zip(states, before):
                    tensor.copy_(saved)
                repeated = original(*args, **kwargs)
                entry["native_repeats"].append({"repeat": repeat, "logits": error_metrics(repeated.next_token_logits, expected_logits), "distribution": logits_metrics(repeated.next_token_logits, expected_logits)})
            if not accepted:
                if ("full_model", 0) not in self.failure_captures:
                    torch.save({"actual_logits": actual_logits, "expected_logits": expected_logits, "native_repeat_logits": repeated.next_token_logits.clone()}, self.report_dir / f"failed-full-model-step-{self.completed_decode_steps}-rank-{self.rank}.pt")
                    self.failure_captures.add(("full_model", 0))
                message = f"Flash-Next full-model same-state shadow failed: {check}; distribution={distribution}"
                self._shadow_failure(message)
        finally:
            self.enabled = True
            self.reference_layer_checks = None
            for (_, tensor), saved in zip(states, actual_states):
                tensor.copy_(saved)
            result.next_token_logits.copy_(actual_logits)
            self.model.last_hc_hidden_states = actual_hc

    def _wrap_forward(self, model):
        original = model.forward

        @functools.wraps(original)
        def forward(*args, **kwargs):
            batch = kwargs.get("forward_batch")
            if batch is None and len(args) > 2:
                batch = args[2]
            if batch is None:
                raise FlashContractError("Flash-Next forward batch is missing")
            previous = self.enabled
            self.enabled = self.supports(batch)
            if not self.enabled and batch.forward_mode.is_extend() and self.shadow_rearm_on_prefill:
                self.shadow_remaining_by_batch.clear()
                self.shadow_remaining = self.shadow_steps
            if self.enabled:
                self.shadow_remaining = self.shadow_remaining_by_batch.setdefault(int(batch.batch_size), self.shadow_steps)
            key = (batch.forward_mode.name, int(batch.batch_size), self.enabled)
            if key not in self.forward_keys:
                self.forward_keys.add(key)
                logger.info("AMD_FLASH_TP8_ROUTE key=%s route=%s", key, "gfx936_tp8_bf16_decode" if self.enabled else "sglang_original")
            try:
                states = self._decode_states() if self.enabled and self.shadow_remaining else []
                before = [tensor.clone() for _, tensor in states]
                result = original(*args, **kwargs)
                if self.enabled and not torch.cuda.is_current_stream_capturing():
                    self.peers.check()
                    if states:
                        self._full_model_shadow(original, args, kwargs, result, states, before)
                    self.completed_decode_steps += 1
                    if self.shadow_remaining:
                        self.shadow_remaining -= 1
                        self.shadow_remaining_by_batch[int(batch.batch_size)] = self.shadow_remaining
                        if not self.shadow_remaining and not any(check["rows"] == int(batch.batch_size) and not check.get("accepted", True) for check in self.full_model_shadow_checks):
                            self.validated_batch_sizes.add(int(batch.batch_size))
                        self.save()
                return result
            except FlashContractError as error:
                self.serving_shadow_failure = str(error)
                self.save()
                raise
            finally:
                self.enabled = previous

        model.forward = forward

    def execution_counts(self):
        counters = [(kind, layer_id, module.executions) for kind, modules in (("hc", self.mixers), ("gdn", self.gdn), ("qsa", self.qsa), ("ple", self.ple), ("post", self.posts)) for layer_id, module in enumerate(modules) if module is not None]
        counters.append(("tail_hc", 48, self.tail.executions))
        counters.extend((("embedding", 0, self.io.entry_executions), ("tail", 48, self.io.tail_executions)))
        values = torch.cat([counter.reshape(1) for _, _, counter in counters]).cpu().tolist()
        counts = {f"{kind}.{layer_id}": value for (kind, layer_id, _), value in zip(counters, values)}
        if self.counter_offsets is not None:
            return {key: value - self.counter_offsets[key] for key, value in counts.items()}
        return counts

    def snapshot(self):
        status = "startup_validating" if len(self.validation) != 96 or len(self.gdn_validation) != 72 or len(self.communication_validation) != 12 else ("serving_shadow_pending" if not self.completed_decode_steps or self.shadow_remaining else "serving_shadow_passed")
        if getattr(self, "serving_shadow_failure", None) or any(not check.get("accepted", True) for check in self.full_model_shadow_checks):
            status = "serving_shadow_failed"
        return {
            "backend": "flash_next_gfx936_tp8_bf16",
            "stage": "complete_decode",
            "whole_decoder_monokernel": status == "serving_shadow_passed",
            "acceptance_status": status,
            "serving_shadow_failure": getattr(self, "serving_shadow_failure", None),
            "serving_shadow_failures": self.serving_shadow_failures,
            "mathematical_launches_per_decode": 99,
            "coverage": "TP embedding and HC expansion, HC mix, GDN projections/conv/recurrent state/gated norm/output projection, QSA projection/norm/RoPE/KV cache/index compression/top512/sparse attention/gate/output projection, PLE hash/embedding/TP reduction/projections/gate/short convolution, attention TP reduction, HC combine, MLP HC mix, router, top10, routed/shared BF16 experts, MoE TP reduction, HC combine, final HC mix/LM head/cooperative global logits gather",
            "native_components": ["prefill", "PLE history metadata", "state checkpoint copies", "sampling", "scheduler and cache management"],
            "tp_rank": self.rank,
            "tp_size": 8,
            "validation": self.validation,
            "communication_validation": self.communication_validation,
            "reduction_rounding": "fp32_rank_order_sum_bf16_rtz",
            "gdn_validation": self.gdn_validation,
            "shadow_checks": self.shadow_checks,
            "full_model_shadow_checks": self.full_model_shadow_checks,
            "full_model_logit_limits": {"kl_reference_actual": 0.01, "total_variation": 0.05, "greedy_argmax_equal": True, "raw_nrmse": "reported_only_native_repeat_is_not_bitwise_deterministic"},
            "shadow_steps_per_batch_size": self.shadow_steps,
            "shadow_rearmed_on_prefill": self.shadow_rearm_on_prefill,
            "shadow_record_failures": self.shadow_record_failures,
            "validated_decode_batch_sizes": sorted(self.validated_batch_sizes),
            "completed_decode_steps": self.completed_decode_steps,
            "executions": self.execution_counts(),
            "forward_keys": sorted(self.forward_keys),
            "program": self.peers.program.manifest,
            "gdn_program": self.gdn[0].program.manifest,
            "qsa_program": self.qsa[3].program.manifest,
            "ple_program": self.ple[1].program.manifest,
            "io_program": self.io.program.manifest,
        }

    def save(self):
        destination = self.report_dir / f"rank-{self.rank}.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2) + "\n")
        temporary.replace(destination)
