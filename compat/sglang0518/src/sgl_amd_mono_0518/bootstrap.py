import functools
import inspect
import logging
import os

from .config import Config
from .contract import ContractError, HOOK_TARGETS, probe, validate
from .observer import WeightObservation
from .report import write_report


logger = logging.getLogger(__name__)
_registered = False
_process_report = {}
_ffn_runtime = None
_mono_runtime = None


def _save():
    write_report(Config.from_environment().report_dir, _process_report)


def before_resolve(server_args):
    if getattr(server_args, "_mono_0518_checked", None) == os.getpid():
        return
    config = Config.from_environment()
    report = probe(server_args.model_path, server_args.tp_size)
    report.update(mode=config.mode, pid=os.getpid(), phase="configuration_checked")
    _process_report.update(report)
    _save()
    validate(report, config.mode)
    if config.mode in {"fused_ffn", "monokernel", "required"}:
        unsupported = [
            name
            for name in ("enable_lora", "enable_torch_compile", "enable_dp_attention", "enable_dp_lm_head", "enable_deterministic_inference", "speculative_algorithm", "enable_linear_replayssm", "enable_linear_replayssm_spec", "enable_unified_memory")
            if getattr(server_args, name, None)
        ]
        if server_args.pp_size != 1 or unsupported:
            raise ContractError(f"Dense fusion supports PP1 without additional execution features: {unsupported}")
        if int(os.environ.get("SGLANG_AMD_MONO_0518_SHADOW_STEPS", "0")) and getattr(server_args, "cuda_graph_backend_decode", None) != "disabled":
            raise ContractError("Model shadow validation requires --cuda-graph-backend-decode disabled")
    server_args._mono_0518_checked = os.getpid()
    if config.mode in {"monokernel", "required"}:
        if os.environ.get("SGLANG_USE_AITER_LINEAR_ATTN") != "1":
            raise ContractError("Dense Monokernel requires SGLANG_USE_AITER_LINEAR_ATTN=1 for matching FP32 GDN beta semantics")
        logger.warning("AMD_DENSE_MONO configuration eligible; awaiting all-layer real-weight validation before attachment")
    elif config.mode == "fused_ffn":
        logger.warning("AMD_FUSED_FFN configuration eligible; awaiting real-weight validation and decode routing; whole_model_monokernel=false")
    else:
        logger.warning(
            "AMD_MONO_AUDIT mode=%s execution_path=sglang_original monokernel_executions=0; %s",
            config.mode,
            "; ".join(report["disable_reasons"]),
        )


def around_load_weights(original, model, weights, target_device):
    observation = WeightObservation()
    result = original(model, observation.iterate(weights), target_device)
    if not observation.completed or not observation.count:
        raise ContractError("Checkpoint observer was bypassed or not exhausted")
    model._mono_0518_observation = observation.summary()
    _process_report.update(phase="weights_observed", weights=observation.summary())
    _save()
    logger.info("AMD_MONO_AUDIT observed %d checkpoint tensors; retained copies=0", observation.count)
    return result


def after_initialize(result, runner):
    global _ffn_runtime, _mono_runtime
    observation = getattr(runner.model, "_mono_0518_observation", None)
    if observation is None or not observation["completed"]:
        raise ContractError("ModelRunner initialized without a completed checkpoint observer")
    if not isinstance(runner.capture_tail_hooks, list):
        raise ContractError("ModelRunner.capture_tail_hooks contract changed")
    original = runner.model.forward
    signature = inspect.signature(original)
    seen = set()
    _process_report.update(
        phase="runner_initialized",
        pid=os.getpid(),
        tp_rank=runner.ps.tp_rank,
        tp_size=runner.ps.tp_size,
        weights=observation,
        forward_signature=str(signature),
        observed_forward_keys=[],
    )
    if Config.from_environment().mode == "fused_ffn":
        from sgl_amd_mono.qwen38_27b.spec import build

        _ffn_runtime = build(runner, "fused_ffn")
        _process_report.update(_ffn_runtime.snapshot())
        _process_report["ffn_real_weight_validation"] = _ffn_runtime.validation
    elif Config.from_environment().mode in {"monokernel", "required"}:
        from sgl_amd_mono.qwen38_27b.spec import build

        _mono_runtime = build(runner, Config.from_environment().mode)
        _process_report.update(_mono_runtime.snapshot())
    _save()

    @functools.wraps(original)
    def forward(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        batch = bound.arguments.get("forward_batch")
        if batch is None:
            raise ContractError("Model forward did not provide forward_batch")
        key = (batch.forward_mode.name, int(batch.batch_size))
        runtime = _mono_runtime if _mono_runtime is not None else _ffn_runtime
        fused = runtime is not None and runtime.supports(batch)
        route = ("dense_bf16_cooperative_monokernel" if _mono_runtime is not None else "fused_bf16_ffn_and_sglang_attention") if fused else "sglang_original"
        if key not in seen:
            seen.add(key)
            _process_report["observed_forward_keys"].append(
                {"forward_mode": key[0], "batch_size": key[1], "route": route}
            )
            _save()
            logger.info("AMD_MONO_ROUTE forward_key=%s route=%s", key, route)
        previous = runtime.enabled if runtime is not None else False
        if runtime is not None:
            runtime.enabled = fused
        try:
            if fused and _mono_runtime is not None and len(_mono_runtime.shadow_checks) < _mono_runtime.shadow_limit:
                return _mono_runtime.shadow_forward(original, args, kwargs, batch)
            return original(*args, **kwargs)
        finally:
            if runtime is not None:
                runtime.enabled = previous

    runner.model.forward = forward


def after_internal_state(result, scheduler, request):
    if _mono_runtime is not None:
        snapshot = _mono_runtime.snapshot()
        result.internal_state["amd_dense_monokernel"] = snapshot
        _process_report.update(snapshot)
        _save()
    if _ffn_runtime is not None:
        snapshot = _ffn_runtime.snapshot()
        result.internal_state["amd_fused_ffn"] = snapshot
        _process_report.update(snapshot)
        _save()
    return result


def register():
    global _registered
    if _registered or os.environ.get("SGLANG_AMD_MONO_0518", "0") == "0":
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(HOOK_TARGETS[0], before_resolve, HookType.BEFORE)
    HookRegistry.register(HOOK_TARGETS[1], around_load_weights, HookType.AROUND)
    HookRegistry.register(HOOK_TARGETS[2], after_initialize, HookType.AFTER)
    HookRegistry.register(HOOK_TARGETS[3], after_internal_state, HookType.AFTER)
    _registered = True


def verify_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry

    missing = set(HOOK_TARGETS) - HookRegistry._patched
    if not _registered or missing:
        raise ContractError(f"Monokernel audit hooks were not installed: {sorted(missing)}")
