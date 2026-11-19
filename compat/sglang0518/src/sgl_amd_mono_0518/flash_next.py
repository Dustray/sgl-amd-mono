import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)
_registered = False
_runtime = None
_accuracy_trace = None
_profile_ranges = None


def before_resolve(server_args):
    if getattr(server_args, "_flash_next_mono_checked", None) == os.getpid():
        return
    from sgl_amd_mono.qwen38_flash_next.gfx936.contract import validate_server

    validate_server(server_args)
    server_args._flash_next_mono_checked = os.getpid()
    logger.warning("AMD_FLASH_TP8 eligible; fusion=%s accuracy_reference=%s", os.environ.get("SGLANG_AMD_FLASH_NEXT_0518", "0"), os.environ.get("SGLANG_AMD_FLASH_ACCURACY_REFERENCE", "0"))


def after_initialize(result, runner):
    global _runtime, _accuracy_trace, _profile_ranges

    if os.environ.get("SGLANG_AMD_FLASH_ACCURACY_REFERENCE", "0") == "1":
        from sgl_amd_mono.qwen38_flash_next.gfx936.reference import install

        logger.warning("AMD_FLASH_ACCURACY_REFERENCE %s", install())
    if os.environ.get("SGLANG_AMD_FLASH_NEXT_0518", "0") == "1":
        from sgl_amd_mono.qwen38_flash_next.gfx936 import contract

        spec = contract.spec_enabled(runner.server_args)
        # This hook also runs on the draft worker's ModelRunner. Without the fused
        # verify env the whole server (target and draft) stays native (P0); with it,
        # only the target runner installs the runtime and its verify batches (4 fused
        # rows per request) go through the mono kernels while the draft stays native.
        fused_target = spec and contract.spec_fused_requested() and not getattr(runner, "is_draft_worker", False)
        if spec:
            logger.warning(
                "AMD_FLASH_TP8 speculative=%s fused_verify=%s draft=%s route=%s verify_graph=%s",
                contract.spec_algorithm_name(runner.server_args.speculative_algorithm),
                contract.spec_fused_requested(),
                bool(getattr(runner, "is_draft_worker", False)),
                "gfx936_tp8_bf16_verify" if fused_target else "sglang_original",
                contract.spec_verify_graph_policy(getattr(runner.server_args, "cuda_graph_backend_decode", None), contract.spec_verify_graph_requested()) if fused_target else "n/a",
            )
        if not spec or fused_target:
            from sgl_amd_mono.qwen38_flash_next.gfx936.runner import FlashRuntime

            _runtime = FlashRuntime(runner)
            from sgl_amd_mono.qwen38_flash_next.gfx936.graph import GraphExecution

            if not spec and getattr(runner.server_args, "cuda_graph_backend_decode", None) == "full":
                _runtime.graph = GraphExecution(_runtime)
            elif fused_target and contract.spec_verify_graph_enabled(runner.server_args):
                # Opt-in (SGLANG_AMD_FLASH_SPEC_VERIFY_GRAPH=1): capture the fused
                # TARGET_VERIFY batch into the target decode graph. SGLang's own
                # DecodeCudaGraphRunner captures TARGET_VERIFY (bs x DRAFT_TOKENS rows);
                # the mono kernels run because the target's fused model.forward executes
                # under capture. GraphExecution.execute validates the first replay
                # bit-for-bit against eager mono. With the flag off the target captures
                # nothing (see around_graph_capture) and verify stays eager.
                _runtime.graph = GraphExecution(_runtime)
            apply_benchmark_mode(_runtime, None)
    if os.environ.get("SGLANG_AMD_FLASH_ACCURACY_TRACE_DIR"):
        from sgl_amd_mono.qwen38_flash_next.gfx936.accuracy_trace import AccuracyTrace

        _accuracy_trace = AccuracyTrace(runner)
    if os.environ.get("SGLANG_AMD_PROFILE_RANGES", "0") == "1":
        from sgl_amd_mono.common.profiling import install_model_ranges

        _profile_ranges = install_model_ranges(runner, _runtime)


def apply_benchmark_mode(runtime, scheduler):
    control_file = os.environ.get("SGLANG_AMD_FLASH_BENCHMARK_CONTROL_FILE")
    if not control_file:
        return
    mode = Path(control_file).read_text().strip()
    if mode not in ("native", "monokernel"):
        raise ValueError(f"Invalid Flash-Next benchmark execution mode: {mode!r}")
    enabled = mode == "monokernel"
    if runtime.fusion_enabled == enabled:
        return
    if getattr(runtime, "graph", None) is not None:
        raise RuntimeError("Flash-Next benchmark mode cannot change after selecting Graph capture")
    running = getattr(scheduler, "running_batch", None)
    if getattr(running, "reqs", None) or getattr(scheduler, "waiting_queue", None) or getattr(scheduler, "cur_batch", None):
        raise RuntimeError("Flash-Next benchmark mode can only change while the scheduler is idle")
    runtime.fusion_enabled = enabled
    logger.warning("AMD_FLASH_BENCHMARK execution_mode=%s", mode)


def after_internal_state(result, scheduler, request):
    if _profile_ranges is not None:
        result.internal_state["amd_profile_ranges"] = _profile_ranges
    if os.environ.get("SGLANG_AMD_FLASH_ACCURACY_REFERENCE", "0") == "1":
        from sgl_amd_mono.qwen38_flash_next.gfx936.reference import snapshot

        result.internal_state["amd_flash_accuracy_reference"] = snapshot()
    if _accuracy_trace is not None:
        result.internal_state["amd_flash_accuracy_trace"] = _accuracy_trace.snapshot()
    if _runtime is not None:
        apply_benchmark_mode(_runtime, scheduler)
        result.internal_state["amd_flash_next_monokernel"] = _runtime.snapshot()
        _runtime.save()
    return result


def around_graph_capture(original, runner, *args, **kwargs):
    # EAGLEDraftCudaGraphRunner inherits this method, so the hook also fires for the
    # draft worker's runners; they carry their own model_runner and must capture
    # normally -- the three draft forwards of an MTP round are where the graph pays off.
    own = _runtime is not None and getattr(runner, "model_runner", None) is _runtime.runner
    if own and _runtime.graph is not None:
        # Non-spec full decode, or a fused-verify target that opted into verify-graph
        # capture: run SGLang's capture (the fused model.forward provides the mono kernels)
        # and install the graph/eager check in GraphExecution.capture.
        return _runtime.graph.capture(original, runner, *args, **kwargs)
    if own and _runtime.spec_verify:
        # Fused verify without the verify-graph opt-in: capture nothing on the target.
        # FullCudaGraphBackend.can_run is `shape_key in self._graphs`, so an empty graph
        # set sends every verify batch down the eager path, which is where the mono
        # kernels run. The draft worker (own=False) still captures its graphs below.
        logger.warning("AMD_FLASH_TP8 skipping target decode-graph capture; TARGET_VERIFY runs eager through the fused kernels")
        return None
    return original(runner, *args, **kwargs)


def around_graph_execute(original, runner, batch, *args, **kwargs):
    if _profile_ranges is not None:
        from sgl_amd_mono.common.profiling import inference_range

        # Graph replay bypasses model.forward, so mark the step around the replay.
        with inference_range(runner.model_runner, batch):
            return _graph_execute(original, runner, batch, *args, **kwargs)
    return _graph_execute(original, runner, batch, *args, **kwargs)


def _graph_execute(original, runner, batch, *args, **kwargs):
    if _runtime is not None and _runtime.graph is not None and runner.model_runner is _runtime.runner:
        return _runtime.graph.execute(original, runner, batch, *args, **kwargs)
    return original(runner, batch, *args, **kwargs)


class _NoBeamCoordinator:
    """Stand-in for the beam coordinator that this SGLang build references but never creates.

    0.5.18+dtk2604 `Scheduler.abort_request` calls `self.beam_coordinator.retire_group(req)` for
    every request it pops from the waiting queue, but nothing ever sets `beam_coordinator`, so
    aborting a queued request (client timeout/disconnect) crashes all TP schedulers with
    AttributeError. Beam search is not used here, so there is no beam group to retire.
    """

    def retire_group(self, req):
        return None


def _make_abort_req(req, finished_reason=None):
    """Missing helper of 0.5.18+dtk2604 `scheduler.py`, called on queued and chunked-prefill aborts.

    Builds the same `AbortReq` as the other abort sites in that file (rid plus optional
    `finished_reason` dict); without a reason the tokenizer reports "Abort in waiting queue".
    """
    from sglang.srt.managers.io_struct import AbortReq

    return AbortReq(rid=req.rid, finished_reason=finished_reason)


def install_scheduler_fixes(scheduler, *args, **kwargs):
    """Fill in names that this SGLang build references but never defines; no-op once SGLang has them."""
    if not hasattr(scheduler, "beam_coordinator"):
        scheduler.beam_coordinator = _NoBeamCoordinator()
    import sglang.srt.managers.scheduler as scheduler_module

    if not hasattr(scheduler_module, "_make_abort_req"):
        scheduler_module._make_abort_req = _make_abort_req
    return None


_scheduler_fix_registered = False

# run_event_loop installs the fixes before any request (the chunked-prefill abort path can be
# reached without abort_request); abort_request keeps them in place for direct callers.
_SCHEDULER_FIX_TARGETS = (
    "sglang.srt.managers.scheduler.Scheduler.run_event_loop",
    "sglang.srt.managers.scheduler.Scheduler.abort_request",
)


def _register_scheduler_fixes(HookRegistry, HookType):
    global _scheduler_fix_registered

    if _scheduler_fix_registered:
        return
    for target in _SCHEDULER_FIX_TARGETS:
        HookRegistry.register(target, install_scheduler_fixes, HookType.BEFORE)
    _scheduler_fix_registered = True


def register():
    global _registered

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    # Needed by native runs too, so it sits before the Flash-Next opt-in gate.
    _register_scheduler_fixes(HookRegistry, HookType)
    if _registered or not any(os.environ.get(name, "0") == "1" for name in ("SGLANG_AMD_FLASH_NEXT_0518", "SGLANG_AMD_FLASH_ACCURACY_REFERENCE", "SGLANG_AMD_PROFILE_RANGES")):
        return
    if os.environ.get("SGLANG_AMD_MONO_0518", "0") != "0":
        raise RuntimeError("Select either Flash-Next or the dense 27B compatibility plugin")
    if os.environ.get("SGLANG_AMD_PROFILE_EVENT_DIR"):
        from sgl_amd_mono.common.profiling import register_frontend_ranges

        register_frontend_ranges()
    HookRegistry.register("sglang.srt.server_args.ServerArgs.resolve_once", before_resolve, HookType.BEFORE)
    HookRegistry.register("sglang.srt.model_executor.model_runner.ModelRunner.initialize", after_initialize, HookType.AFTER)
    HookRegistry.register("sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.capture", around_graph_capture, HookType.AROUND)
    HookRegistry.register("sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.execute", around_graph_execute, HookType.AROUND)
    HookRegistry.register("sglang.srt.managers.scheduler.Scheduler.get_internal_state", after_internal_state, HookType.AFTER)
    _registered = True
