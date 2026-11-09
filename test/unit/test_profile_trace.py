import asyncio
import gzip
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import pytest
import torch

from sgl_amd_mono.common.profile_trace import merge_profiles, read_trace
from sgl_amd_mono.common.profiling import async_host_range, host_range, install_model_ranges, mark_callable


def rank_trace(directory, rank, base=1000000000, kernels=True):
    offset = rank * 10
    events = [
        {"ph": "M", "name": "process_name", "pid": 1, "tid": 0, "args": {"name": "GPU"}},
        {"ph": "X", "cat": "user_annotation", "name": "inference/EXTEND/step=1/batch=1", "pid": 2, "tid": 3, "ts": 100 - offset, "dur": 50},
        {"ph": "X", "cat": "user_annotation", "name": "inference/DECODE/step=2/batch=1", "pid": 2, "tid": 3, "ts": 200 - offset, "dur": 50},
        {"ph": "X", "cat": "cuda_runtime", "name": "hipLaunchKernel", "pid": 2, "tid": 3, "ts": 205 - offset, "dur": 2, "args": {"correlation": 42}},
        {"ph": "s", "cat": "ac2g", "name": "flow", "id": 7, "pid": 2, "tid": 3, "ts": 206 - offset},
        {"ph": "f", "cat": "ac2g", "name": "flow", "id": 7, "pid": 1, "tid": 0, "ts": 210 - offset},
        {"ph": "X", "cat": "python_function", "name": "python_call", "pid": 2, "tid": 3, "ts": 200 - offset, "dur": 50},
    ]
    if kernels:
        events.append({"ph": "X", "cat": "kernel", "name": "actual_gpu_kernel", "pid": 1, "tid": 0, "ts": 210 - offset, "dur": 20 + rank, "args": {"correlation": 42}})
    path = directory / f"full-TP-{rank}.trace.json.gz"
    with gzip.open(path, "wt") as output:
        json.dump({"baseTimeNanoseconds": base + rank * 10000, "traceEvents": events}, output)
    return path


@pytest.mark.parametrize("standard_json", (False, True))
def test_merge_preserves_cross_rank_clock_and_separates_processes_and_flows(tmp_path, monkeypatch, standard_json):
    if standard_json:
        monkeypatch.setattr("sgl_amd_mono.common.profile_trace.orjson", None)
    paths = [rank_trace(tmp_path, rank) for rank in range(2)]
    client = [{"ph": "X", "cat": "client", "name": "HTTP", "pid": 1, "tid": 0, "ts": 1000210, "dur": 20}]
    summary = merge_profiles(paths, tmp_path / "merged.json.gz", client_events=client, expected_ranks=2, expected_decode_steps=1)
    events = read_trace(tmp_path / "merged.json.gz")["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    assert [event["ts"] for event in kernels] == [210, 210]
    assert len({event["pid"] for event in kernels}) == 2
    names = {event["pid"]: event["args"]["name"] for event in events if event.get("name") == "process_name"}
    assert all(names[event["pid"]].startswith("TP") for event in kernels)
    flows = [event for event in events if event.get("cat") == "ac2g"]
    assert flows[0]["id"] == flows[1]["id"]
    assert flows[2]["id"] == flows[3]["id"]
    assert flows[0]["id"] != flows[2]["id"]
    host = next(event for event in events if event.get("cat") == "client")
    assert host["ts"] == 210 and host["pid"] not in {event["pid"] for event in kernels}
    assert summary["ranks"][1]["phases"]["DECODE"]["gpu_kernel_us"] == 21
    assert summary["ranks"][1]["unattributed_gpu_kernels"] == 0
    assert summary["ranks"][1]["python_function_events_filtered"] == 1
    assert not any(event.get("cat") == "python_function" for event in events)
    assert (tmp_path / "operators.csv").exists()


@pytest.mark.parametrize("failure", ("missing_rank", "cpu_only", "decode_coverage", "decode_token_coverage", "missing_clock", "duplicate_rank"))
def test_merge_rejects_incomplete_capture(tmp_path, failure):
    path = rank_trace(tmp_path, 0, kernels=failure != "cpu_only")
    if failure == "missing_clock":
        trace = read_trace(path)
        del trace["baseTimeNanoseconds"]
        with gzip.open(path, "wt") as output:
            json.dump(trace, output)
    with pytest.raises(ValueError):
        merge_profiles([path, path] if failure == "duplicate_rank" else [path], tmp_path / "merged.json.gz", expected_ranks=2 if failure == "missing_rank" else 1, expected_decode_steps=2 if failure == "decode_coverage" else 1, expected_decode_token_rows=2 if failure == "decode_token_coverage" else 1)
    assert not (tmp_path / "merged.json.gz").exists()


def test_host_ranges_only_record_capture_window_and_preserve_async_exceptions(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_PROFILE_EVENT_DIR", str(tmp_path))
    assert host_range(lambda: 7) == 7
    assert not list(tmp_path.glob("host-*.jsonl"))
    (tmp_path / "capture.active").touch()

    async def fails():
        raise RuntimeError("tokenization failed")

    with pytest.raises(RuntimeError, match="tokenization failed"):
        asyncio.run(async_host_range(fails))
    event = json.loads(next(tmp_path.glob("host-*.jsonl")).read_text())
    assert event["args"]["failed"] and event["dur"] >= 0


def test_named_ranges_preserve_fused_object_and_model_behavior():
    class Kernel:
        def __call__(self, value):
            return value + 2

    first, second = Kernel(), Kernel()
    mark_callable(first, "mono/first")
    mark_callable(second, "mono/second")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        assert first(3) == second(3) == 5
    names = {event.name for event in profile.events()}
    assert {"mono/first", "mono/second"} <= names
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    runner = SimpleNamespace(model=model, ps=SimpleNamespace(tp_rank=0))
    inputs = torch.ones(1, 2)
    before = model(inputs).detach().clone()
    state = install_model_ranges(runner)
    assert state["modules"] == 1
    assert torch.equal(model(inputs), before)
    assert install_model_ranges(runner) is state


def test_resume_refuses_to_overwrite_existing_capture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools"))
    from profile_flash_next_serving import main

    (tmp_path / "native").mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"original": true}')
    monkeypatch.setattr(sys, "argv", ["profile", "--mode", "native", "--resume", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert manifest.read_text() == '{"original": true}'


def test_attach_requires_explicit_frontend_directory_and_single_mode(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools"))
    from profile_flash_next_serving import main

    monkeypatch.setattr(sys, "argv", ["profile", "--url", "http://127.0.0.1:30160", "--output-dir", str(tmp_path / "run")])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


def test_detokenizer_entry_installs_ranges_inside_a_fresh_process(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_PROFILE_EVENT_DIR", str(tmp_path))
    (tmp_path / "capture.active").touch()
    code = """
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sgl_amd_mono.common.profiling import profiled_detokenizer_process
def decode(self, value):
    return value + 1
DetokenizerManager._decode_batch_token_id_output = decode
assert profiled_detokenizer_process(lambda: DetokenizerManager._decode_batch_token_id_output(None, 41)) == 42
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=90, capture_output=True)
    events = [json.loads(line) for path in tmp_path.glob("host-*.jsonl") for line in path.read_text().splitlines()]
    assert len(events) == 1 and events[0]["name"] == "decode"
