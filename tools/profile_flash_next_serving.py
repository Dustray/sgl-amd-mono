import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import aiohttp
import requests

from sgl_amd_mono.common.gpus import check_gpus_idle
from sgl_amd_mono.common.profile_trace import merge_profiles


ROOT = Path(__file__).resolve().parents[1]
MODEL = "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master"


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prompts(model_path, length, batch_size):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    passage = "A distributed inference system processes requests, maintains its key value cache, and generates one token at a time. Explain the process with concrete examples. "
    results = []
    for prefix in ("Alpha: ", "Beta: ")[:batch_size]:
        token_ids = tokenizer.encode(prefix + passage * (length // 20 + 20), add_special_tokens=False)[:length]
        text = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
        if tokenizer.encode(text, add_special_tokens=False) != token_ids:
            raise ValueError("Prompt does not round-trip to the exact requested token count")
        results.append({"text": text, "input_ids": token_ids, "sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()})
    return results


def request_json(url, endpoint, payload=None):
    timeout = 3600 if endpoint in ("/start_profile", "/stop_profile") else 600
    response = requests.get(url + endpoint, timeout=timeout) if payload is None else requests.post(url + endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json() if "json" in response.headers.get("content-type", "") else response.text


async def generate(session, url, prompt, output_tokens, index):
    payload = {"text": prompt["text"], "stream": True, "sampling_params": {"temperature": 0, "max_new_tokens": output_tokens, "ignore_eos": True}}
    started = time.time_ns()
    monotonic = time.perf_counter()
    arrivals = []
    events = []
    final = None
    async with session.post(url + "/generate", json=payload) as response:
        response.raise_for_status()
        async for raw in response.content:
            if not raw.startswith(b"data: ") or raw.strip() == b"data: [DONE]":
                continue
            final = json.loads(raw[6:])
            if "error" in final:
                raise RuntimeError(final)
            count = len(final.get("output_ids", []))
            if count > len(arrivals):
                received = time.time_ns()
                for token in range(len(arrivals), count):
                    arrivals.append((received - started) / 1e9)
                    events.append({"ph": "i", "s": "t", "cat": "client", "name": f"response/token.{token + 1}", "pid": os.getpid(), "tid": index, "ts": received / 1000})
    elapsed = time.perf_counter() - monotonic
    if final is None or len(final.get("output_ids", [])) != output_tokens:
        raise ValueError("Generation did not return the requested number of output tokens")
    metadata = final["meta_info"]
    if metadata["prompt_tokens"] != len(prompt["input_ids"]) or metadata.get("cached_tokens", 0) or metadata.get("num_retractions", 0):
        raise ValueError(f"Request token/cache/retraction contract failed: {metadata}")
    events.append({"ph": "X", "cat": "client", "name": f"request/{index}/HTTP_generate", "pid": os.getpid(), "tid": index, "ts": started / 1000, "dur": (time.time_ns() - started) / 1000, "args": {"input_tokens": len(prompt["input_ids"]), "output_tokens": output_tokens}})
    return {"e2e_s": elapsed, "ttft_s": arrivals[0], "decode_s": arrivals[-1] - arrivals[0], "output_ids": final["output_ids"], "text": final.get("text"), "meta_info": metadata, "token_arrivals_s": arrivals}, events


async def generate_batch(url, inputs, output_tokens):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as session:
        return await asyncio.gather(*(generate(session, url, prompt, output_tokens, index) for index, prompt in enumerate(inputs)))


def shadow_state(info):
    states = [state["amd_flash_next_monokernel"] for state in info.get("internal_states", []) if "amd_flash_next_monokernel" in state]
    return [{"rank": state["tp_rank"], "acceptance_status": state["acceptance_status"], "whole_decoder_monokernel": state["whole_decoder_monokernel"], "completed_decode_steps": state["completed_decode_steps"], "shadow_checks": len(state["shadow_checks"]), "full_model_shadow_checks": len(state["full_model_shadow_checks"]), "executions": state["executions"]} for state in states]


def capture(arguments, mode, directory, url, inputs):
    raw_directory = directory / "ranks"
    raw_directory.mkdir()
    host_directory = directory / "host"
    host_directory.mkdir(exist_ok=True)
    event_directory = Path(arguments.host_event_dir).resolve() if arguments.host_event_dir else host_directory
    if not event_directory.is_dir():
        raise ValueError(f"Missing frontend event directory: {event_directory}")
    for iteration in range(arguments.warmup):
        request_json(url, "/flush_cache", {})
        asyncio.run(generate_batch(url, inputs, 4))
        print(f"{mode}: warmup {iteration + 1}/{arguments.warmup} complete", flush=True)
    request_json(url, "/flush_cache", {})
    before = request_json(url, "/server_info")
    if (mode == "monokernel") != bool(shadow_state(before)):
        raise RuntimeError("Live server route does not match the requested profiling mode")
    if any("amd_flash_accuracy_trace" in state for state in before.get("internal_states", [])):
        raise RuntimeError("Accuracy-logit recording must be disabled while profiling")
    if not any("amd_profile_ranges" in state for state in before.get("internal_states", [])):
        raise RuntimeError("Model profiling ranges were not installed")
    configuration = {"output_dir": str(raw_directory), "activities": ["CPU", "GPU"], "with_stack": arguments.with_stack, "record_shapes": True, "profile_by_stage": False, "merge_profiles": False, "profile_id": f"{mode}-full", "detailed_annotations": True}
    save(directory / "capture.json", {"mode": mode, "configuration": configuration, "inputs": inputs, "output_tokens": arguments.output_tokens, "server_info_before": before, "profile_is_accuracy_acceptance": False})
    active = event_directory / "capture.active"
    window_start = time.time_ns() / 1000
    with active.open("x") as marker:
        marker.write(str(window_start))
    started = False
    try:
        request_json(url, "/start_profile", configuration)
        started = True
        print(f"{mode}: capturing full {arguments.input_tokens}->{arguments.output_tokens}, batch {arguments.batch_size}", flush=True)
        results = asyncio.run(generate_batch(url, inputs, arguments.output_tokens))
    finally:
        try:
            if started:
                request_json(url, "/stop_profile", {})
        finally:
            active.unlink(missing_ok=True)
    window_end = time.time_ns() / 1000
    if event_directory != host_directory:
        for path in event_directory.glob("host-*.jsonl"):
            events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            events = [event for event in events if window_start <= event["ts"] <= window_end]
            (host_directory / path.name).write_text("".join(json.dumps(event) + "\n" for event in events))
    save(directory / "requests.json", [result for result, _ in results])
    after = request_json(url, "/server_info")
    save(directory / "server_info_after.json", after)
    before_shadow, after_shadow = shadow_state(before), shadow_state(after)
    if [(state["shadow_checks"], state["full_model_shadow_checks"]) for state in before_shadow] != [(state["shadow_checks"], state["full_model_shadow_checks"]) for state in after_shadow]:
        raise RuntimeError("Accuracy shadow work contaminated the measured request")
    if mode == "monokernel" and (not before_shadow or after_shadow[0]["completed_decode_steps"] <= before_shadow[0]["completed_decode_steps"]):
        raise RuntimeError("Captured decode did not exclusively use the requested fused route")
    trace_paths = list(raw_directory.glob("*TP-*.trace.json.gz"))
    client_events = [event for _, events in results for event in events]
    save(host_directory / "client.json", client_events)
    stage_directory = directory / "runtime" if arguments.kernel_stages and mode == "monokernel" else None
    summary = merge_profiles(trace_paths, directory / "inference.perfetto.json.gz", host_directory, client_events, expected_ranks=8, expected_decode_steps=arguments.output_tokens - 1 if arguments.batch_size == 1 else None, include_python_functions=arguments.keep_python_functions, expected_decode_token_rows=arguments.batch_size * (arguments.output_tokens - 1), device_stage_directory=stage_directory)
    for before_state, after_state in zip(before_shadow, after_shadow):
        expected = next(state["phases"]["DECODE"]["forwards"] for state in summary["ranks"] if state["rank"] == after_state["rank"])
        if after_state["completed_decode_steps"] - before_state["completed_decode_steps"] != expected:
            raise RuntimeError("Fused route counters do not match captured decode forwards")
    summary["shadow_state_before"] = before_shadow
    summary["shadow_state_after"] = after_shadow
    summary["profile_is_accuracy_acceptance"] = False
    summary["prompt_tokens"] = arguments.input_tokens
    summary["output_tokens"] = arguments.output_tokens
    summary["batch_size"] = arguments.batch_size
    save(directory / "summary.json", summary)
    if not any("_tokenize_texts" in name for name in summary["host_ranges"]):
        raise RuntimeError("Actual text tokenization range was not captured")
    if not any("_decode_batch_token_id_output" in name for name in summary["host_ranges"]):
        raise RuntimeError("Detokenization range was not captured")
    print(f"{mode}: trace exported: {summary['trace']} ({summary['total_events']} events)", flush=True)
    return summary


def stop_under_hipprof(server, port):
    """hipprof writes the trace when the process it wraps exits, so signal the traced server itself:
    a signal to hipprof would kill the exporter before it runs."""
    traced = [int(path.name) for path in Path("/proc").iterdir() if path.name.isdigit()
              and b"sglang.launch_server" in (path / "cmdline").read_bytes() if f"--port\x00{port}".encode() in (path / "cmdline").read_bytes()]
    if not traced:
        print("no traced sglang process found; signalling the hipprof group instead", flush=True)
        os.killpg(server.pid, signal.SIGTERM)
    for pid in traced:
        os.kill(pid, signal.SIGTERM)
    print(f"stopped the server ({traced}); hipprof is exporting its trace, this can take minutes", flush=True)
    try:
        server.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=60)
        raise RuntimeError("hipprof did not finish exporting within an hour")


def hipprof_control(session, instruction):
    command = ["hipprof", "--session-client", session, instruction]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)} failed: {result.stdout}{result.stderr}")
    return result.stdout.strip()


def hipprof_capture(arguments, mode, directory, url, inputs, session):
    """Trace only the measured request: the server runs under `hipprof --trace-off`, and the
    session client turns tracing on around this one generate call."""
    for iteration in range(arguments.warmup):
        request_json(url, "/flush_cache", {})
        asyncio.run(generate_batch(url, inputs, 4))
        print(f"{mode}: warmup {iteration + 1}/{arguments.warmup} complete", flush=True)
    request_json(url, "/flush_cache", {})
    before = request_json(url, "/server_info")
    if (mode == "monokernel") != bool(shadow_state(before)):
        raise RuntimeError("Live server route does not match the requested profiling mode")
    save(directory / "capture.json", {"mode": mode, "tool": "hipprof", "session": session, "inputs": inputs,
                                      "output_tokens": arguments.output_tokens, "server_info_before": before, "profile_is_accuracy_acceptance": False})
    hipprof_control(session, "--start")
    try:
        print(f"{mode}: hipprof tracing {arguments.input_tokens}->{arguments.output_tokens}, batch {arguments.batch_size}", flush=True)
        results = asyncio.run(generate_batch(url, inputs, arguments.output_tokens))
    finally:
        hipprof_control(session, "--stop")
        hipprof_control(session, "--flush")
    save(directory / "requests.json", [result for result, _ in results])
    after = request_json(url, "/server_info")
    save(directory / "server_info_after.json", after)
    before_shadow, after_shadow = shadow_state(before), shadow_state(after)
    if [(state["shadow_checks"], state["full_model_shadow_checks"]) for state in before_shadow] != [(state["shadow_checks"], state["full_model_shadow_checks"]) for state in after_shadow]:
        raise RuntimeError("Accuracy shadow work contaminated the traced request")
    steps = None
    if mode == "monokernel":
        steps = after_shadow[0]["completed_decode_steps"] - before_shadow[0]["completed_decode_steps"]
        if steps != arguments.output_tokens - 1:
            raise RuntimeError(f"Traced request ran {steps} fused decode steps, expected {arguments.output_tokens - 1}")
    summary = {"tool": "hipprof", "mode": mode, "prompt_tokens": arguments.input_tokens, "output_tokens": arguments.output_tokens,
               "batch_size": arguments.batch_size, "mono_decode_steps": steps, "shadow_state_before": before_shadow,
               "shadow_state_after": after_shadow, "profile_is_accuracy_acceptance": False,
               "client": {key: results[0][0][key] for key in ("e2e_s", "ttft_s", "decode_s")}}
    save(directory / "summary.json", summary)
    return summary


def run_service(arguments, mode, directory, inputs):
    directory.mkdir()
    (directory / "host").mkdir()
    with socket.socket() as connection:
        if connection.connect_ex(("127.0.0.1", arguments.port)) == 0:
            raise RuntimeError(f"Port {arguments.port} is already in use")
    environment = os.environ.copy()
    overrides = {"SGLANG_AMD_FLASH_NEXT_0518": "1" if mode == "monokernel" else "0", "SGLANG_AMD_MONO_0518": "0", "SGLANG_AMD_PROFILE_RANGES": "0" if arguments.hipprof else "1", "SGLANG_AMD_PROFILE_EVENT_DIR": str(directory / "host"), "SGLANG_AMD_FLASH_REPORT_DIR": str(directory / "runtime"), "SGLANG_AMD_FLASH_ACCURACY_REFERENCE": "0", "SGLANG_AMD_FLASH_SHADOW_STEPS": "1", "SGLANG_AMD_FLASH_SHADOW_REARM_ON_PREFILL": "0", "SGLANG_AMD_FLASH_SHADOW_RECORD_FAILURES": "1", "SGLANG_PROFILE_V2": "0", "QWEN_FLASH_NEXT_MODEL_PATH": arguments.model_path}
    if arguments.kernel_stages and mode == "monokernel":
        overrides["SGLANG_AMD_FLASH_STAGE_TIMING"] = "1"
        overrides["SGLANG_AMD_FLASH_STAGE_TIMING_STEP"] = str(arguments.warmup * 3 + 1)
        overrides["SGLANG_AMD_FLASH_STAGE_TIMING_STEPS"] = str(arguments.output_tokens - 1)
    if arguments.decode_graph:
        overrides["HIP_GRAPH_ACCUMULATE_DISPATCH"] = "0"
    environment.update(overrides)
    environment.pop("SGLANG_AMD_FLASH_ACCURACY_TRACE_DIR", None)
    launcher = "launch_qwen_flash_next_monokernel.sh" if mode == "monokernel" else "launch_qwen_flash_next.sh"
    command = ["bash", str(ROOT / launcher), "--port", str(arguments.port), "--skip-server-warmup", "--enable-metrics", "--enable-cache-report", "--stream-interval", "1", "--max-running-requests", str(arguments.batch_size)]
    session = None
    if arguments.hipprof:
        session = f"flashnext-{mode}-{os.getpid()}"
        (directory / "hipprof").mkdir()
        command = ["hipprof", "--hip-trace", "--follow-fork", "--trace-off", "--session", session,
                   "--output-type", "2", "-o", str(directory / "hipprof" / f"{mode}.pftrace"),
                   "-d", str(directory / "hipprof"), *command]
    if arguments.decode_graph:
        batch_sizes = [str(size) for size in range(1, arguments.batch_size + 1)]
        command += ["--cuda-graph-backend-decode", "full", "--cuda-graph-max-bs-decode", str(arguments.batch_size), "--cuda-graph-bs-decode", *batch_sizes, "--disable-cuda-graph-padding"]
    with (directory / "server.log").open("w") as log:
        server = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch = {"pid": server.pid, "command": command, "environment_overrides": overrides, "started_at": datetime.now(timezone.utc).isoformat(), "stopped": False}
    save(directory / "launch.json", launch)
    print(f"{mode}: loading model, PID {server.pid}; log: {directory / 'server.log'}", flush=True)
    url = f"http://127.0.0.1:{arguments.port}"
    try:
        deadline = time.monotonic() + arguments.startup_timeout
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"{mode} server exited with {server.returncode}; see {directory / 'server.log'}")
            try:
                if requests.get(url + "/health", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise TimeoutError(f"{mode} startup exceeded {arguments.startup_timeout} seconds")
        if arguments.hipprof:
            return hipprof_capture(arguments, mode, directory, url, inputs, session)
        return capture(arguments, mode, directory, url, inputs)
    finally:
        if server.poll() is None:
            if arguments.hipprof:
                stop_under_hipprof(server, arguments.port)
            else:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=20)
        launch.update(stopped=True, returncode=server.returncode, stopped_at=datetime.now(timezone.utc).isoformat())
        save(directory / "launch.json", launch)


def report(directory, summaries):
    lines = ["# Flash-Next TP8 / gfx936 全推理 Profile", "", "范围：文本分词、调度器 CPU / ATen / HIP API、全部 prefill 与 decode forward、八卡 GPU kernels / memcpy、反分词及客户端流式响应。文件为 Perfetto 可导入的 Chrome Trace JSON gzip。", ""]
    for mode, summary in summaries.items():
        if mode == "monokernel" and summary.get("device_kernel_stages"):
            lines.append("核内阶段时间已按 rank 和 kernel 发射顺序对齐到 GPU kernel，并嵌入同一份 Perfetto trace；阶段时间来自设备端 wall clock。")
            break
    else:
        lines.append("核内阶段计时未启用；TP peer 通信等待包含在对应融合 kernel 内。")
    lines += ["", "Profile 开销包含在本轮请求耗时中，不能替代独立性能基准。融合路径的精度验收状态单独保留；本采集不宣称精度通过。", "", "| 模式 | 输入→输出 / batch | 事件数 | 每卡 Prefill / Decode forward | 文件 |", "| --- | --- | ---: | --- | --- |"]
    for mode, summary in summaries.items():
        phases = summary["ranks"][0]["phases"]
        prefills = sum(value["forwards"] for phase, value in phases.items() if phase != "DECODE")
        lines.append(f"| {mode} | {summary['prompt_tokens']}→{summary['output_tokens']} / {summary['batch_size']} | {summary['total_events']} | {prefills} / {phases['DECODE']['forwards']} | [{mode} trace]({mode}/inference.perfetto.json.gz) |")
    for mode, summary in summaries.items():
        lines += ["", f"## {mode}：rank 0 GPU kernel 累计耗时前十", "", "累计耗时按 kernel duration 相加，存在并行时不等于请求耗时。完整八卡 CPU / GPU / 标记统计见同目录 operators.csv，CPU 与模块时间为 inclusive。", "", "| Kernel | 次数 | 累计 ms | 平均 μs |", "| --- | ---: | ---: | ---: |"]
        for row in summary["top_gpu_kernels_rank0"][:10]:
            lines.append(f"| `{row['name'].replace('|', '/')}` | {row['count']} | {row['total_us'] / 1000:.3f} | {row['mean_us']:.3f} |")
    (directory / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Capture full native/fused TP8 inference as a Perfetto-compatible trace")
    parser.add_argument("--mode", choices=("native", "monokernel", "both"), default="both")
    parser.add_argument("--model-path", default=MODEL)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--decode-graph", action="store_true", help="Capture decode CUDA graphs for exactly the profiled batch sizes")
    parser.add_argument("--hipprof", action="store_true", help="Trace with hipprof (HIP API/kernel/copy) instead of the torch profiler; writes a .pftrace per mode")
    parser.add_argument("--keep-python-functions", action="store_true")
    parser.add_argument("--kernel-stages", action="store_true", help="embed every measured Monokernel decode step's device-stage spans in the torch-profiler Perfetto trace")
    parser.add_argument("--port", type=int, default=30160)
    parser.add_argument("--url", help="Attach to an existing server with profiling ranges already enabled")
    parser.add_argument("--host-event-dir", help="Frontend event directory configured on an attached server")
    parser.add_argument("--allow-busy-gpus", action="store_true", help="only warn, instead of refusing to start, when another job is using the GPUs")
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Append missing modes to an existing run directory without replacing captures")
    arguments = parser.parse_args()
    if arguments.kernel_stages and (arguments.mode != "monokernel" or arguments.hipprof or arguments.url):
        parser.error("--kernel-stages requires a launched --mode monokernel torch-profiler capture")
    if arguments.url and (arguments.mode == "both" or not arguments.host_event_dir):
        parser.error("Attaching requires one explicit --mode and --host-event-dir")
    if arguments.host_event_dir and not arguments.url:
        parser.error("--host-event-dir is only used with --url")
    if arguments.input_tokens < 1 or arguments.output_tokens < 2 or arguments.warmup < 1 or arguments.input_tokens + arguments.output_tokens > 4096:
        parser.error("Require input>=1, output>=2, warmup>=1, input+output<=4096")
    directory = arguments.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=arguments.resume)
    previous_manifest = json.loads((directory / "manifest.json").read_text()) if arguments.resume and (directory / "manifest.json").exists() else None
    selected_modes = ("native", "monokernel") if arguments.mode == "both" else (arguments.mode,)
    for mode in selected_modes:
        if (directory / mode).exists():
            parser.error(f"Capture already exists, refusing to overwrite: {directory / mode}")
    paths = [ROOT / "sgl_amd_mono/common/profiling.py", ROOT / "sgl_amd_mono/common/profile_trace.py", Path(__file__).resolve(), ROOT / "compat/sglang0518/src/sgl_amd_mono_0518/flash_next.py"]
    paths.extend((ROOT / "sgl_amd_mono/qwen38_flash_next/gfx936").glob("*.py"))
    paths.extend((ROOT / "sgl_amd_mono/qwen38_flash_next/gfx936").glob("*.cpp"))
    manifest = {"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}, "started_at": datetime.now(timezone.utc).isoformat(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}, "complete": False}
    if previous_manifest is not None:
        manifest["history"] = [previous_manifest]
    save(directory / "manifest.json", manifest)
    summaries = {}
    try:
        if arguments.hipprof and arguments.url:
            parser.error("--hipprof must launch the server itself so it can wrap it")
        manifest["gpu_usage_before"] = check_gpus_idle(require=not (arguments.allow_busy_gpus or arguments.url))
        save(directory / "manifest.json", manifest)
        inputs = prompts(arguments.model_path, arguments.input_tokens, arguments.batch_size)
        for mode in selected_modes:
            if arguments.url:
                (directory / mode).mkdir()
                summaries[mode] = capture(arguments, mode, directory / mode, arguments.url.rstrip("/"), inputs)
            else:
                summaries[mode] = run_service(arguments, mode, directory / mode, inputs)
            if not arguments.hipprof:
                report(directory, summaries)
        if not arguments.hipprof:
            for mode in ("native", "monokernel"):
                path = directory / mode / "summary.json"
                if mode not in summaries and path.exists():
                    summaries[mode] = json.loads(path.read_text())
            report(directory, summaries)
        manifest["captured_modes"] = sorted(summaries)
        manifest["complete"] = True
    except BaseException as error:
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(directory / "manifest.json", manifest)


if __name__ == "__main__":
    main()
