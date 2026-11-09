import logging
import os


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
        from sgl_amd_mono.qwen38_flash_next.gfx936.runner import FlashRuntime

        _runtime = FlashRuntime(runner)
    if os.environ.get("SGLANG_AMD_FLASH_ACCURACY_TRACE_DIR"):
        from sgl_amd_mono.qwen38_flash_next.gfx936.accuracy_trace import AccuracyTrace

        _accuracy_trace = AccuracyTrace(runner)
    if os.environ.get("SGLANG_AMD_PROFILE_RANGES", "0") == "1":
        from sgl_amd_mono.common.profiling import install_model_ranges

        _profile_ranges = install_model_ranges(runner, _runtime)


def after_internal_state(result, scheduler, request):
    if _profile_ranges is not None:
        result.internal_state["amd_profile_ranges"] = _profile_ranges
    if os.environ.get("SGLANG_AMD_FLASH_ACCURACY_REFERENCE", "0") == "1":
        from sgl_amd_mono.qwen38_flash_next.gfx936.reference import snapshot

        result.internal_state["amd_flash_accuracy_reference"] = snapshot()
    if _accuracy_trace is not None:
        result.internal_state["amd_flash_accuracy_trace"] = _accuracy_trace.snapshot()
    if _runtime is not None:
        result.internal_state["amd_flash_next_monokernel"] = _runtime.snapshot()
        _runtime.save()
    return result


def register():
    global _registered

    if _registered or not any(os.environ.get(name, "0") == "1" for name in ("SGLANG_AMD_FLASH_NEXT_0518", "SGLANG_AMD_FLASH_ACCURACY_REFERENCE", "SGLANG_AMD_PROFILE_RANGES")):
        return
    if os.environ.get("SGLANG_AMD_MONO_0518", "0") != "0":
        raise RuntimeError("Select either Flash-Next or the dense 27B compatibility plugin")
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if os.environ.get("SGLANG_AMD_PROFILE_EVENT_DIR"):
        from sgl_amd_mono.common.profiling import register_frontend_ranges

        register_frontend_ranges()
    HookRegistry.register("sglang.srt.server_args.ServerArgs.resolve_once", before_resolve, HookType.BEFORE)
    HookRegistry.register("sglang.srt.model_executor.model_runner.ModelRunner.initialize", after_initialize, HookType.AFTER)
    HookRegistry.register("sglang.srt.managers.scheduler.Scheduler.get_internal_state", after_internal_state, HookType.AFTER)
    _registered = True
