import functools
import json
import os
from pathlib import Path
import threading
import time


def _host_event(name, started, failed=False):
    directory = Path(os.environ["SGLANG_AMD_PROFILE_EVENT_DIR"])
    event = {"ph": "X", "cat": "frontend", "name": name, "pid": os.getpid(), "tid": threading.get_native_id(), "ts": started / 1000, "dur": (time.time_ns() - started) / 1000, "args": {"failed": failed}}
    with (directory / f"host-{os.getpid()}.jsonl").open("a") as output:
        output.write(json.dumps(event) + "\n")


def _host_started():
    directory = os.environ.get("SGLANG_AMD_PROFILE_EVENT_DIR")
    return time.time_ns() if directory and (Path(directory) / "capture.active").exists() else None


def host_range(original, *args, **kwargs):
    started = _host_started()
    try:
        result = original(*args, **kwargs)
    except BaseException:
        if started is not None:
            _host_event(original.__qualname__, started, True)
        raise
    if started is not None:
        _host_event(original.__qualname__, started)
    return result


async def async_host_range(original, *args, **kwargs):
    started = _host_started()
    try:
        result = await original(*args, **kwargs)
    except BaseException:
        if started is not None:
            _host_event(original.__qualname__, started, True)
        raise
    if started is not None:
        _host_event(original.__qualname__, started)
    return result


def register_frontend_ranges():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for target in ("tokenizer_manager.TokenizerManager._tokenize_one_request", "tokenizer_manager.TokenizerManager._tokenize_texts", "tokenizer_manager.TokenizerManager._send_one_request"):
        hook = async_host_range if "_tokenize_" in target else host_range
        HookRegistry.register(f"sglang.srt.managers.{target}", hook, HookType.AROUND)
    HookRegistry.register("sglang.srt.entrypoints.engine.Engine._launch_detokenizer_subprocesses", launch_profiled_detokenizer, HookType.AROUND)


def profiled_detokenizer_process(original_entry, *args, **kwargs):
    from sglang.srt.managers.detokenizer_manager import DetokenizerManager

    original = DetokenizerManager._decode_batch_token_id_output

    @functools.wraps(original)
    def decode(*decode_args, **decode_kwargs):
        return host_range(original, *decode_args, **decode_kwargs)

    DetokenizerManager._decode_batch_token_id_output = decode
    return original_entry(*args, **kwargs)


def launch_profiled_detokenizer(original, *args, **kwargs):
    kwargs["run_detokenizer_process_func"] = functools.partial(profiled_detokenizer_process, kwargs["run_detokenizer_process_func"])
    return original(*args, **kwargs)


def wrap_range(original, name):
    import torch

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(name):
            return original(*args, **kwargs)

    return wrapped


def mark_callable(instance, name):
    instance._mono_profile_label = name
    instance_type = type(instance)
    if instance_type.__dict__.get("_mono_profile_wrapped", False):
        return
    original = instance_type.__call__

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        import torch

        with torch.profiler.record_function(self._mono_profile_label):
            return original(self, *args, **kwargs)

    instance_type.__call__ = wrapped
    instance_type._mono_profile_wrapped = True


def inference_range(runner, batch):
    import torch

    mode = getattr(getattr(batch, "forward_mode", None), "name", "UNKNOWN")
    rows = getattr(batch, "batch_size", 0)
    runner._mono_profile_sequence = getattr(runner, "_mono_profile_sequence", 0) + 1
    return torch.profiler.record_function(f"inference/{mode}/step={runner._mono_profile_sequence}/batch={rows}")


def install_model_ranges(runner, runtime=None):
    import torch

    if getattr(runner, "_mono_profile_ranges", None) is not None:
        return runner._mono_profile_ranges
    modules = 0
    for name, module in runner.model.named_modules():
        if not name:
            continue
        module.forward = wrap_range(module.forward, f"module/{name}:{type(module).__name__}")
        modules += 1
        if "HyperConnection" in type(module).__name__:
            for method in ("mix", "combine", "inject"):
                original = getattr(module, method, None)
                if callable(original):
                    setattr(module, method, wrap_range(original, f"module/{name}.{method}"))
    original_forward = runner.model.forward

    @functools.wraps(original_forward)
    def forward(*args, **kwargs):
        batch = kwargs.get("forward_batch")
        if batch is None and len(args) > 2:
            batch = args[2]
        with inference_range(runner, batch):
            return original_forward(*args, **kwargs)

    runner.model.forward = forward
    if runtime is not None:
        for kind in ("mixers", "gdn", "qsa", "ple", "posts"):
            for layer, instance in enumerate(getattr(runtime, kind)):
                if instance is not None:
                    mark_callable(instance, f"mono/layer.{layer:02d}/{kind}")
        mark_callable(runtime.tail, "mono/final_hc_mix")
        for method in ("entry", "tail"):
            setattr(runtime.io, method, wrap_range(getattr(runtime.io, method), f"mono/{method}"))
        runtime.peers.check = wrap_range(runtime.peers.check, "mono/peer_error_check")
    runner._mono_profile_ranges = {"modules": modules, "fused_ranges": runtime is not None, "rank": runner.ps.tp_rank}
    return runner._mono_profile_ranges
