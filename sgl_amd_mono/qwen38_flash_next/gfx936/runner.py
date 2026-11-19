import functools
import json
import logging
import math
import os
from pathlib import Path

import torch

from .program import FlashContractError, HCMix, PeerGroup, PostAttention
from .geometry import DRAFT_TOKENS, MAX_ROWS
from .gdn import GDNAttention
from .qsa import QSAAttention, cache_views
from .ple import PLE
from .decode_io import DecodeIO


logger = logging.getLogger(__name__)


def bf16_ulp_error(actual, expected):
    """Worst absolute difference, in bf16 ULPs taken at the reference tensor's peak magnitude.

    Why the peak and not the RMS or each element: measured on the real layer-16 verify dump
    (2026-11-19, rank 6), 70% of the 10240 elements differ and the elementwise relative error
    is bimodal -- elements at |v| >= 0.05 sit at 0.5 ULP (correct rounding) while the 1110
    elements below 1e-3 reach 15843 ULP. That is cancellation, not arithmetic: the absolute
    error stays capped at 4.9e-4 across the whole tensor while the values themselves approach
    zero. An elementwise ULP therefore explodes on the cancelled elements, and an RMS-scaled
    one is dragged by the same near-zero mass (that dump's RMS is 0.0116 against a peak of
    0.136). The peak is the stable scale: 4.9e-4 is exactly half a ULP at 0.136, the
    correct-rounding bound, while nrmse reads 0.00605 for the same tensors.

    A real divergence -- a wrong expert, a mis-indexed row -- moves an element by something of
    order the tensor's own scale, which is hundreds of ULPs here, so this stays discriminative.
    """
    difference = float((actual.detach().float() - expected.detach().float()).abs().max())
    peak = float(expected.detach().float().abs().max())
    if not math.isfinite(peak) or not math.isfinite(difference):
        return float("inf")
    if peak == 0.0:
        return 0.0 if difference == 0.0 else float("inf")
    # bf16 keeps 8 significand bits, so the step at |v| is 2**(floor(log2|v|) - 7).
    return difference / 2.0 ** (math.floor(math.log2(peak)) - 7)


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
        self.fusion_enabled = True
        self.graph_capture_active = False
        self.graph = None
        from .contract import spec_verify_enabled

        # Routes TARGET_VERIFY batches (4 fused rows per request) through the mono
        # kernels; without it a speculative server never installs this runtime.
        self.spec_verify = spec_verify_enabled(runner.server_args)

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
        # Sizes the QSA index top-k (one 4096-token sort chunk per power-of-two list).
        self.context_length = int(getattr(runner.server_args, "context_length", None) or runner.model_config.context_len)
        for layer in self.model.layers:
            hc = layer.attn_hyper_connection
            self.mixers.append(HCMix(hc.hc_norm.weight, hc.input_mix_weight_down.weight, hc.input_mix_weight_up.weight))
            self.mixers[-1].state = self.peers.state
            self.posts.append(PostAttention(layer, self.peers))
            self.gdn.append(GDNAttention(layer.linear_attn, self.mixers[-1]) if hasattr(layer, "linear_attn") else None)
            self.qsa.append(QSAAttention(layer, self.mixers[-1], self.context_length) if hasattr(layer, "indexer") else None)
            self.ple.append(PLE(layer.ple, self.peers) if layer.ple is not None else None)
        hc = self.model.hyper_connection_mixer
        self.tail = HCMix(hc.hc_norm.weight, hc.input_mix_weight_down.weight, hc.input_mix_weight_up.weight)
        self.tail.state = self.peers.state
        self.io = DecodeIO(runner.model, self.tail, self.peers)
        self.stage_timing = None
        self.cu_timing = None
        self._validate_communication()
        self._validate_weights()
        if os.environ.get("SGLANG_AMD_FLASH_STAGE_TIMING", "0") == "1":
            self._attach_stage_timing()
        if os.environ.get("SGLANG_AMD_FLASH_CU_TIMING", "0") == "1":
            self._attach_cu_timing()
        self.counter_offsets = self.execution_counts()
        for layer_id, layer in enumerate(self.model.layers):
            self._wrap_layer(layer_id, layer)
        self._wrap_decode_io(runner.model)
        self._wrap_forward(runner.model)
        self.save()

    def _attach_stage_timing(self):
        from .stage_timing import StageTiming

        records = int(os.environ.get("SGLANG_AMD_FLASH_STAGE_TIMING_RECORDS", str(1 << 18)))
        timing = StageTiming(self.device, self.rank, records)
        for program in self._timed_programs().values():
            timing.attach(program)
        self.stage_timing = timing
        logger.info("AMD_FLASH_STAGE_TIMING attached to %s libraries on rank %s, %s records", len(timing.libraries), self.rank, records)

    def _attach_cu_timing(self):
        from .cu_timing import CUTiming

        records = int(os.environ.get("SGLANG_AMD_FLASH_CU_TIMING_RECORDS", str(1 << 13)))
        blocks = int(self.peers.program.manifest.get("blocks", 80))
        timing = CUTiming(self.device, self.rank, records, blocks)
        for program in self._timed_programs().values():
            timing.attach(program)
        self.cu_timing = timing
        logger.info("AMD_FLASH_CU_TIMING attached to %s libraries on rank %s, %s blocks x %s records (%.1f MiB)",
                    len(timing.libraries), self.rank, blocks, records, timing.capacity * 8 / 2 ** 20)

    def _timed_programs(self):
        """Every distinct compiled library: each includes kernel.cpp and so has its own symbols."""
        programs = {id(self.peers.program): self.peers.program, id(self.io.program): self.io.program}
        for module in (*self.mixers, *self.posts, *self.ple, *self.gdn, *self.qsa):
            if module is not None:
                programs[id(module.program)] = module.program
        return programs

    def export_stage_timing(self, label="decode"):
        if self.stage_timing is None:
            raise FlashContractError("Flash-Next stage timing is not enabled")
        clock_khz = self.peers.program.information.get("clock_khz")
        return self.stage_timing.export(self.report_dir / f"stage-timing-{label}-rank-{self.rank}.json", clock_khz)

    def export_cu_timing(self, label="decode", base=None):
        if self.cu_timing is None:
            raise FlashContractError("Flash-Next CU timing is not enabled")
        clock_khz = self.peers.program.information.get("clock_khz")
        return self.cu_timing.export(self.report_dir / f"cu-timing-{label}-rank-{self.rank}.json", clock_khz, base)

    def begin_stage_timing_step(self):
        step = int(os.environ.get("SGLANG_AMD_FLASH_STAGE_TIMING_STEP", "0"))
        if step <= 0 or self.completed_decode_steps + 1 != step:
            return
        if self.stage_timing is not None:
            self.stage_timing.reset()
        if self.cu_timing is not None:
            self.cu_timing.reset()

    def finish_stage_timing_step(self):
        if self.stage_timing is None and self.cu_timing is None:
            return
        first_step = int(os.environ.get("SGLANG_AMD_FLASH_STAGE_TIMING_STEP", "0"))
        step_count = int(os.environ.get("SGLANG_AMD_FLASH_STAGE_TIMING_STEPS", "1"))
        last_step = first_step + step_count - 1
        if first_step > 0 and step_count > 0 and self.completed_decode_steps == last_step:
            torch.cuda.synchronize(self.device)
            base = None
            if self.stage_timing is not None:
                result = self.export_stage_timing()
                base = result.get("base_ticks")
                logger.info("AMD_FLASH_STAGE_TIMING exported rank %s %s", self.rank, result)
            if self.cu_timing is not None:
                result = self.export_cu_timing(base=base)
                logger.info("AMD_FLASH_CU_TIMING exported rank %s trace=%s compute_units=%s dropped=%s shared=%s migrated=%s",
                            self.rank, result["trace"], result["compute_units"], result["dropped"],
                            result["shared_units"], result["migrated_launch_count"])

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
            verify = bool(batch.forward_mode.is_target_verify())
            if hidden.shape[-1] == 2560:
                hidden = torch.cat([hidden] * 4, dim=-1)
            if layer.ple is not None:
                from sglang.srt.model_executor.forward_context import get_req_to_token_pool
                from sglang.srt.models.qwen4_exp import _ple_track_targets

                ple_batch = kwargs["ple_batch"]
                if verify:
                    # Fused PLE verify comes later; the native verify branch fills the
                    # short-conv intermediates that the worker's own commit replays into
                    # the working state, and does not track, so skip the fused fast path
                    # and the decode-side track entirely.
                    hidden = hidden + layer.ple(hidden, batch, ple_batch)
                else:
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
                if verify:
                    # One fused launch over requests * DRAFT_TOKENS rows. The kernel
                    # replays the draft chain per request: conv advances the working
                    # window and snapshots every step into the scratch (exactly like
                    # native causal_conv1d verify), while the recurrent state stays in
                    # registers with per-step snapshots — native likewise never touches
                    # the working ssm during verify (disable_state_update). The worker's
                    # own commit scatters the accepted-prefix snapshots back, so the
                    # decode-side track must not run here.
                    attention, residuals = self.gdn[layer_id].verify(hidden, cache.conv[0], cache.temporal, indices, backend.verify_intermediate_state_indices, cache.intermediate_conv_window[0], cache.intermediate_ssm, batch.batch_size)
                else:
                    attention, residuals = self.gdn[layer_id](hidden, cache.conv[0], cache.temporal, indices)
                    backend._track_mamba_state_decode(batch, cache.conv[0], cache.temporal, indices, layer.linear_attn.layer_id)
                if self.shadow_remaining:
                    actual_conv, actual_ssm = cache.conv[0].clone(), cache.temporal.clone()
                    # Verify's real product is the per-step scratch (the worker's commit scatters it
                    # back), and the native rerun below overwrites it, so save mono's copy, compare it,
                    # and put it back; otherwise the shadow would check and keep native's snapshots.
                    scratch = self._verify_scratch_views(cache, backend, batch.batch_size) if verify else []
                    actual_scratch = [tensor.clone() for _, tensor in scratch]
                    cache.conv[0].copy_(reference_conv)
                    cache.temporal.copy_(reference_ssm)
                    mixed, _ = layer.attn_hyper_connection.mix(hidden)
                    fused_mixed = self.gdn[layer_id].mixed[:hidden.shape[0]]
                    reference_attention = layer.linear_attn(fused_mixed, batch)
                    checks = {"mixed": error_metrics(fused_mixed, mixed), "output": error_metrics(attention, reference_attention), "conv": error_metrics(actual_conv, cache.conv[0]), "ssm": error_metrics(actual_ssm, cache.temporal)}
                    if verify:
                        # A verify row is one draft step, so this output is the end of a
                        # DRAFT_TOKENS-long chain, not the single step the nrmse gate was
                        # calibrated on. Report how far apart the two sides are in bf16 ULPs as
                        # well; see `_gdn_shadow_failures` for how the two are combined.
                        checks["output"]["bf16_ulp"] = bf16_ulp_error(attention, reference_attention)
                    for (name, tensor), actual in zip(scratch, actual_scratch):
                        checks[name] = error_metrics(actual, tensor)
                        tensor.copy_(actual)
                    cache.conv[0].copy_(actual_conv)
                    cache.temporal.copy_(actual_ssm)
                    self.shadow_checks.append({"kind": "gdn", "layer": layer_id, "checks": checks})
                    failed = self._gdn_shadow_failures(checks)
                    if failed:
                        torch.save({"hidden": hidden, "mixed": fused_mixed, "native_mixed": mixed, "actual": attention, "expected": reference_attention, "before_conv": reference_conv, "before_ssm": reference_ssm, "indices": indices, "projected": self.gdn[layer_id].projected, "projected_ba": self.gdn[layer_id].projected_ba, "recurrent": self.gdn[layer_id].recurrent, "core": self.gdn[layer_id].core}, self.report_dir / f"failed-gdn-shadow-layer-{layer_id}-rank-{self.rank}.pt")
                        self._shadow_failure(f"Flash-Next GDN serving shadow failed at layer {layer_id} on {failed}: {checks}")
            else:
                from sglang.srt.model_executor.forward_context import get_attn_backend

                backend = get_attn_backend().full_attn_backend
                pool = backend.token_to_kv_pool
                key, value = pool.get_key_buffer(layer_id), pool.get_value_buffer(layer_id)
                ring, ring_positions, compressed = pool.get_qsa_key_state_buffer(layer_id), pool.qsa_rope_position_buffer, pool.get_qsa_compressed_k_buffer(layer_id)
                cache_tensors = (key, value, ring, ring_positions, compressed)
                if self.shadow_remaining:
                    before = [tensor.clone() for tensor in cache_tensors]
                if verify:
                    # One row per draft token: request repeated per step; batch.seq_lens is the
                    # pre-verify length, so row j of a group sees prefix + j + 1 tokens, exactly
                    # the per-row lengths native QwenSparseAttnBackend._graph_speculative_layout
                    # builds. The scheduler pre-wrote req_to_token and out_cache_loc for every
                    # row, which is what the kernel validates.
                    requests = batch.req_pool_indices.repeat_interleave(DRAFT_TOKENS)
                    steps = torch.arange(1, DRAFT_TOKENS + 1, dtype=batch.seq_lens.dtype, device=batch.seq_lens.device)
                    lengths = (batch.seq_lens.repeat_interleave(DRAFT_TOKENS) + steps.repeat(batch.batch_size)).to(torch.int32)
                else:
                    requests = batch.req_pool_indices
                    lengths = batch.seq_lens.to(torch.int32)
                attention, residuals = self.qsa[layer_id](hidden, kwargs["positions"], lengths, requests, batch.out_cache_loc, backend.req_to_token_pool.req_to_token, key, value, ring, ring_positions, compressed, verify=verify)
                if self.shadow_remaining:
                    actual_cache = [tensor.clone() for tensor in cache_tensors]
                    for tensor, saved in zip(cache_tensors, before):
                        tensor.copy_(saved)
                    mixed, _ = layer.attn_hyper_connection.mix(hidden)
                    reference_attention = layer.self_attention(kwargs["positions"], mixed, batch)
                    checks = {"output": error_metrics(attention, reference_attention)}
                    for name, tensor, actual in zip(("key", "value", "ring", "ring_positions", "compressed"), cache_tensors, actual_cache):
                        checks[name] = error_metrics(actual[1:], tensor[1:]) if name == "compressed" else error_metrics(actual, tensor)
                    # Whole-buffer metrics dilute one wrong row over the pool (a fully wrong compressed
                    # row at context 200k is nrmse ~0.006, under the gate), so also compare exactly the
                    # rows this step wrote. Compressed keys only decide the index top-k above 512 groups,
                    # so a wrong decode-time write is invisible at short context.
                    checks.update(self._qsa_written_checks(requests, lengths, batch.out_cache_loc, backend, cache_tensors, actual_cache))
                    for tensor, actual in zip(cache_tensors, actual_cache):
                        tensor.copy_(actual)
                    self.shadow_checks.append({"kind": "qsa", "layer": layer_id, "checks": checks})
                    failed = any(not check["finite"] or check["nrmse"] > 0.008 or check.get("count", 0) or check.get("read_slot_matches") is False for check in checks.values())
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

    def _shadow_failure(self, message, discriminating=True):
        """Record a shadow mismatch, and raise unless the check cannot discriminate.

        ``discriminating=False`` marks a comparison whose reference is noisier than its own
        gate, so a failure says nothing about mono and must never stop the server. Only the
        full-model logits check is in that position today (see ``_full_model_shadow``). Every
        per-module check stays fatal by default -- those are the ones that catch kernel bugs --
        and ``SGLANG_AMD_FLASH_SHADOW_RECORD_FAILURES=1`` still downgrades all of them, so the
        flag keeps its documented meaning.
        """
        self.serving_shadow_failure = message
        self.serving_shadow_failures.append({"step": self.completed_decode_steps, "reason": message, "discriminating": discriminating})
        self.save()
        if discriminating and not self.shadow_record_failures:
            raise FlashContractError(message)
        logger.error("AMD_FLASH_CORRECTNESS_DIAGNOSTIC %s", message)

    @staticmethod
    def _qsa_written_checks(row_requests, row_lengths, row_locations, backend, expected_cache, actual_cache):
        lengths = row_lengths.long()
        requests = row_requests.long()
        locations = row_locations.long()
        key, value, ring, ring_positions, compressed = expected_cache
        actual_key, actual_value, actual_ring, actual_ring_positions, actual_compressed = actual_cache
        (key, value), (actual_key, actual_value) = cache_views(key, value), cache_views(actual_key, actual_value)
        pages, offsets = locations // 64, locations % 64
        ring_rows = requests * 4 + (lengths - 1) % 4
        checks = {
            "written_key": error_metrics(actual_key[pages, offsets], key[pages, offsets]),
            "written_value": error_metrics(actual_value[pages, offsets], value[pages, offsets]),
            "written_ring": error_metrics(actual_ring[ring_rows], ring[ring_rows]),
            "written_ring_positions": error_metrics(actual_ring_positions[ring_rows], ring_positions[ring_rows]),
        }
        completed = lengths % 4 == 0
        if bool(completed.any()):
            rows = locations[completed] // 4
            checks["written_compressed"] = error_metrics(actual_compressed[rows], compressed[rows])
            # Index scores read the group's key at its first token's slot; both must name one row.
            table = backend.req_to_token_pool.req_to_token
            first = table[requests[completed], lengths[completed] - 4].long() // 4
            checks["written_compressed"]["read_slot_matches"] = bool(torch.equal(first, rows))
        # Anything the step wrote outside these rows (e.g. a compressed key at the wrong slot).
        for name, expected, actual, written in (("key", key, actual_key, None), ("compressed", compressed, actual_compressed, (locations[completed] // 4) if bool(completed.any()) else None)):
            if name == "key":
                mask = torch.ones(expected.shape[:2], dtype=torch.bool, device=expected.device)
                mask[pages, offsets] = False
                differs = (expected.view(torch.int16) != actual.view(torch.int16)).flatten(2).any(-1) & mask
            else:
                mask = torch.ones(expected.shape[0], dtype=torch.bool, device=expected.device)
                mask[0] = False
                if written is not None:
                    mask[written] = False
                differs = (expected.view(torch.int16) != actual.view(torch.int16)).flatten(1).any(-1) & mask
            checks[f"unwritten_{name}_rows_changed"] = {"finite": True, "nrmse": 0.0, "count": int(differs.sum())}
        return checks

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
        if getattr(batch, "input_embeds", None) is not None:
            return False
        if batch.batch_size not in (1, 2):
            return False
        if batch.forward_mode.is_decode():
            return batch.input_ids.numel() == batch.batch_size
        # Fused MTP verify: the native worker front-packs DRAFT_TOKENS rows per
        # request, matching the kernel row model (MAX_ROWS capacity).
        return batch.forward_mode.is_target_verify() and self.spec_verify and batch.input_ids.numel() == batch.batch_size * DRAFT_TOKENS

    @staticmethod
    def _gdn_shadow_failures(checks):
        """Names of the GDN shadow checks that fail; empty means the layer passed.

        nrmse at 0.006 is the decode gate and stays the primary test. A check that also
        carries ``bf16_ulp`` (only the fused verify output) may clear it on that instead,
        because nrmse normalises by an RMS that near-zero, cancellation-dominated elements
        drag down: measured 2026-11-19 on a real verify step, GDN layer 16 read nrmse 0.00605
        while its worst element was half a ULP from native at the tensor's peak -- the
        correct-rounding bound. The per-row split of that same dump showed no growth along the
        draft chain (row 0 0.0061, row 3 0.0050), so this is not chain accumulation and the
        tolerance is not about chain length: it is the slack between two implementations that
        round independently and anything a real divergence would produce (hundreds).

        The tolerance is 16, not ``DRAFT_TOKENS``: a second dump the same day (layer 32,
        rank 3, serving verify, nrmse 0.0113 / 6 ULP at a peak of 0.0615) was cleared by
        rebuilding the whole verify chain off the checkpoint on CPU -- bf16 in_proj (both
        buffers bit-identical), conv+silu, exp/softplus gating, l2norm, the delta recurrence
        in fp32 and the sigmoid(z) norm -- which reproduced mono's ``recurrent`` buffer
        bit for bit and left <= 1.3 ULP in ``core``. So mono is the exact chain and the
        6 ULP is the reference side's independent rounding: native runs verify through the
        aiter Triton kernel (its own fp32 accumulation orders), with native in_proj/o_proj
        GEMMs and an 8-way bf16 all-reduce summed in a different order on top. That noise
        scales with the layer's activation scale, not the peak, so a small-peak layer reads
        many ULPs at the same absolute error (layer 16: peak 0.136 -> 0.5 ULP; layer 32:
        peak 0.0615 -> 6 ULP at max_abs 1.5e-3 for both). Everything a real divergence
        produces stays orders above this: a flipped step scores nrmse ~1.4, a flipped
        expert ~0.2, and a mis-indexed row moves whole 128-wide head blocks.
        """
        failures = []
        for name, check in checks.items():
            if not check["finite"]:
                failures.append(name)
            elif check["nrmse"] > 0.006 and check.get("bf16_ulp", float("inf")) > 16.0:
                failures.append(name)
        return failures

    @staticmethod
    def _verify_scratch_views(cache, backend, requests):
        """Non-overlapping views of the speculative scratch rows one verify pass writes.

        ``intermediate_conv_window`` uses SGLang's deduplicated layout: an ``as_strided`` view
        whose step stride equals its window stride, so consecutive steps share columns and
        ``copy_`` into it is rejected as internal overlap. For that layout the row is
        re-viewed as its physical ``[dim, steps + width - 1]`` envelope, which covers exactly
        the same memory without overlap. The dense layout is used as is.
        """
        views = []
        for row in backend.verify_intermediate_state_indices[:requests].tolist():
            window = cache.intermediate_conv_window[0][row]
            steps, dim, width = window.shape
            if window.stride(0) == window.stride(2):
                window = window.as_strided((dim, steps + width - 1), (window.stride(1), window.stride(2)), window.storage_offset())
            views.append((f"conv_window.{row}", window))
            views.append((f"intermediate_ssm.{row}", cache.intermediate_ssm[row]))
        return views

    def _decode_states(self, batch=None):
        from sglang.srt.model_executor.forward_context import get_attn_backend, get_req_to_token_pool

        backend = get_attn_backend()
        pool = get_req_to_token_pool()
        kv_pool = backend.full_attn_backend.token_to_kv_pool
        verify = batch is not None and batch.forward_mode.is_target_verify()
        tensors = [("ngram_context", pool.ngram_pool.context), ("qsa_positions", kv_pool.qsa_rope_position_buffer)]
        if verify:
            # Native verify writes the PLE per-step side states too; the commit reads them.
            for name, owner, attribute in (("ngram_intermediate", "ngram_pool", "intermediate_context"), ("ple_conv_intermediate", "short_conv_pool", "intermediate_conv_state")):
                tensor = getattr(getattr(pool, owner, None), attribute, None)
                if tensor is not None:
                    tensors.append((name, tensor))
        for layer_id, layer in enumerate(self.model.layers):
            if self.gdn[layer_id] is not None:
                cache = backend.linear_attn_backend.req_to_token_pool.mamba2_layer_cache(layer.linear_attn.layer_id)
                tensors.extend(((f"conv.{layer_id}", cache.conv[0]), (f"ssm.{layer_id}", cache.temporal)))
                if verify:
                    tensors.extend((f"{name}.{layer_id}", view) for name, view in self._verify_scratch_views(cache, backend.linear_attn_backend, batch.batch_size))
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
                # Score native's own re-runs through the same gate. Native HC mix sums its
                # split-K with FP32 atomics, so the same input can flip the 10th/11th MoE
                # expert and the reference is not reproducible: measured 2026-11-19 on a verify
                # batch, native against itself scored KL 0.02118 / TV 0.085 while the gate is
                # KL 0.01 / TV 0.05. When the reference misses its own gate this comparison
                # cannot tell a mono defect from reference noise, so it records instead of
                # killing the server. The per-module shadows stay fatal and keep that job.
                native_accepted = all(logits_accepted(repeat["distribution"]) for repeat in entry["native_repeats"])
                entry["reference_reproducible"] = native_accepted
                message = f"Flash-Next full-model same-state shadow failed: {check}; distribution={distribution}; reference_reproducible={native_accepted}"
                self._shadow_failure(message, discriminating=native_accepted)
        finally:
            self.enabled = True
            self.reference_layer_checks = None
            for (_, tensor), saved in zip(states, actual_states):
                tensor.copy_(saved)
            result.next_token_logits.copy_(actual_logits)
            self.model.last_hc_hidden_states = actual_hc

    def _wrap_forward(self, model):
        original = model.forward
        self.original_forward = original

        @functools.wraps(original)
        def forward(*args, **kwargs):
            batch = kwargs.get("forward_batch")
            if batch is None and len(args) > 2:
                batch = args[2]
            if batch is None:
                raise FlashContractError("Flash-Next forward batch is missing")
            previous = self.enabled
            self.enabled = getattr(self, "fusion_enabled", True) and self.supports(batch)
            if not self.enabled and batch.forward_mode.is_extend() and self.shadow_rearm_on_prefill:
                self.shadow_remaining_by_batch.clear()
                self.shadow_remaining = self.shadow_steps
            # Verify and decode count their shadow steps separately (decode keeps the plain
            # batch-size key that graph.py also uses), so one cannot use up the other's checks.
            verify = bool(self.enabled) and bool(getattr(batch.forward_mode, "is_target_verify", lambda: False)())
            shadow_key = ("verify", int(batch.batch_size)) if verify else int(batch.batch_size)
            if self.enabled and not getattr(self, "graph_capture_active", False):
                self.shadow_remaining = self.shadow_remaining_by_batch.setdefault(shadow_key, self.shadow_steps)
            key = (batch.forward_mode.name, int(batch.batch_size), self.enabled)
            if key not in self.forward_keys:
                self.forward_keys.add(key)
                logger.info("AMD_FLASH_TP8_ROUTE key=%s route=%s", key, "gfx936_tp8_bf16_decode" if self.enabled else "sglang_original")
            try:
                if self.enabled and not getattr(self, "graph_capture_active", False):
                    self.begin_stage_timing_step()
                states = self._decode_states(batch) if self.enabled and self.shadow_remaining else []
                before = [tensor.clone() for _, tensor in states]
                result = original(*args, **kwargs)
                if self.enabled and not getattr(self, "graph_capture_active", False) and not torch.cuda.is_current_stream_capturing():
                    self.peers.check()
                    if states:
                        self._full_model_shadow(original, args, kwargs, result, states, before)
                    self.completed_decode_steps += 1
                    self.finish_stage_timing_step()
                    if self.shadow_remaining:
                        self.shadow_remaining -= 1
                        self.shadow_remaining_by_batch[shadow_key] = self.shadow_remaining
                        if not verify and not self.shadow_remaining and not any(check["rows"] == int(batch.batch_size) and not check.get("accepted", True) for check in self.full_model_shadow_checks):
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
            "execution_mode": "monokernel" if getattr(self, "fusion_enabled", True) else "native",
            "graph": self.graph.snapshot() if getattr(self, "graph", None) is not None else None,
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
