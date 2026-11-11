#!/usr/bin/env python3
"""Qwen3.8-Flash-Next TP8 / gfx936 BF16 serving benchmark.

Launches (or attaches to) an SGLang server in native or Monokernel decode
mode, with or without decode CUDA graphs, runs every batch size x input
length x output length case, and reports TTFT, TPOT, E2E, prefill and
throughput. Optionally captures one torch-profiler trace per case and merges
it into a Perfetto file.

Examples:
    python3 bench_flash_next.py --mode native --graph --batch-size 1 --input-len 512 --output-len 64
    python3 bench_flash_next.py --mode mono --batch-size 1 2 --input-len 64 512 2048 --output-len 16 64 128
    python3 bench_flash_next.py --mode mono --graph --profile --input-len 512 --output-len 64
    python3 bench_flash_next.py --url http://127.0.0.1:30000 --mode mono --input-len 512 --output-len 64
"""

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import statistics
import subprocess
import sys
import time

import aiohttp
import requests

from sgl_amd_mono.common.gpus import check_gpus_idle, gpu_usage


ROOT = Path(__file__).resolve().parent
MODEL = "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master"
CONTEXT = 4096
PAGE = 64
MONO_BATCH_SIZES = (1, 2)
NUMA_MIN_FREE_GIB = 60


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def make_inputs(model_path, lengths, rows, seed):
    """Fixed random token ids: exact lengths, no tokenizer round trip, reproducible by seed."""
    config = json.loads((Path(model_path) / "config.json").read_text())
    vocab = config.get("text_config", config)["vocab_size"]
    generator = random.Random(seed)
    return {length: [[generator.randrange(1000, vocab - 1000) for _ in range(length)] for _ in range(rows)] for length in sorted(lengths)}


def mono_states(info):
    return [state["amd_flash_next_monokernel"] for state in info.get("internal_states", []) if "amd_flash_next_monokernel" in state]


def server_info(url):
    response = requests.get(url + "/server_info", timeout=120)
    response.raise_for_status()
    return response.json()


def post(url, endpoint, payload=None, timeout=600):
    response = requests.post(url + endpoint, json=payload or {}, timeout=timeout)
    response.raise_for_status()
    return response


# ---------------------------------------------------------------- server


def numa_free_gib():
    free = {}
    for node in range(8):
        path = Path(f"/sys/devices/system/node/node{node}/meminfo")
        if not path.exists():
            return None
        line = next(line for line in path.read_text().splitlines() if "MemFree" in line)
        free[node] = int(line.split()[-2]) / 2**20
    return free


def use_numa(policy):
    """Rank i loads ~45 GiB through host memory; binding it to node i OOMs if that node is full."""
    if policy != "auto":
        return policy == "on"
    free = numa_free_gib()
    if free is None:
        return False
    if min(free.values()) < NUMA_MIN_FREE_GIB:
        print(f"NUMA binding skipped: free GiB per node {{{', '.join(f'{node}: {value:.0f}' for node, value in free.items())}}} "
              f"(< {NUMA_MIN_FREE_GIB} on some node)", flush=True)
        return False
    return True


def server_command(arguments):
    batch = max(arguments.batch_size)
    # A full context per request: an exact batch * (input + output) budget leaves no room for page rounding and
    # reserved slots, and the scheduler then serializes the requests of a wave.
    total_tokens = batch * CONTEXT
    command = ["bash", str(ROOT / "launch_qwen_flash_next.sh"),
               "--port", str(arguments.port), "--random-seed", str(arguments.seed),
               "--max-running-requests", str(batch), "--max-total-tokens", str(total_tokens),
               "--max-mamba-cache-size", str(max(4, 4 * batch)), "--kv-cache-dtype", "bfloat16",
               "--enable-metrics", "--enable-cache-report", "--stream-interval", "1", "--watchdog-timeout", "1200"]
    if use_numa(arguments.numa):
        command += ["--numa-node", *[str(node) for node in range(8)]]
    if arguments.graph:
        command += ["--cuda-graph-backend-decode", "full", "--cuda-graph-max-bs-decode", str(batch),
                    "--cuda-graph-bs-decode", *[str(size) for size in sorted(arguments.batch_size)], "--disable-cuda-graph-padding"]
    return command + arguments.server_args


def server_environment(arguments, directory):
    overrides = {
        "QWEN_FLASH_NEXT_MODEL_PATH": arguments.model_path,
        "SGLANG_USE_FUSED_TOPK_SOFTMAX": "0",  # LightOp fused top-k picks wrong experts on real routing inputs
        "SGLANG_AMD_FLASH_NEXT_0518": "1" if arguments.mode == "mono" else "0",
        "SGLANG_AMD_MONO_0518": "0",
        "SGLANG_AMD_FLASH_REPORT_DIR": str(directory / "runtime"),
        "SGLANG_AMD_FLASH_ACCURACY_REFERENCE": "0",
        # One shadow check per batch size during warmup, never inside timed requests.
        "SGLANG_AMD_FLASH_SHADOW_STEPS": "1",
        "SGLANG_AMD_FLASH_SHADOW_REARM_ON_PREFILL": "0",
        "SGLANG_AMD_FLASH_SHADOW_RECORD_FAILURES": "0" if arguments.strict_accuracy else "1",
        "SGLANG_AMD_PROFILE_RANGES": "1" if arguments.profile else "0",
        "SGLANG_PROFILE_V2": "0",
    }
    if arguments.graph:
        overrides["HIP_GRAPH_ACCUMULATE_DISPATCH"] = "0"
    if arguments.profile:
        overrides["SGLANG_AMD_PROFILE_EVENT_DIR"] = str(directory / "profile_events")
        (directory / "profile_events").mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.update(overrides)
    environment.pop("SGLANG_AMD_FLASH_ACCURACY_TRACE_DIR", None)
    return environment, overrides


def server_exit_detail(log_path):
    """Extract the last traceback / fatal line(s) from server.log so the real cause
    (e.g. FlashContractError) is visible even when SGLang's kill_process_tree makes
    the wrapper exit with -9."""
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    starts = [i for i, line in enumerate(lines) if "Traceback (most recent call last)" in line]
    if not starts:
        fatal = [line for line in lines if any(word in line for word in ("Error:", "Exception", "SIGKILL", "Killed"))]
        return "\n".join(fatal[-3:])
    tail = lines[starts[-1]:]
    # The final exception line is the first non-indented, non-empty line at the end of the traceback.
    exception_line = ""
    for line in reversed(tail):
        if line and not line.startswith((" ", "\t")):
            exception_line = line
            break
    # Keep the traceback header, "  File ..." frames and the final exception line.
    frames = [line for line in tail if not line.lstrip().startswith(("^", "~"))]
    if len(frames) > 12:
        frames = frames[:2] + ["  ..."] + frames[-4:]
    if exception_line and exception_line not in frames:
        frames.append(exception_line)
    return "\n".join(frames)


def start_server(arguments, directory):
    with socket.socket() as connection:
        if connection.connect_ex(("127.0.0.1", arguments.port)) == 0:
            raise RuntimeError(f"Port {arguments.port} is already in use; pass --url to attach or --port to choose another")
    command = server_command(arguments)
    environment, overrides = server_environment(arguments, directory)
    log = (directory / "server.log").open("w")
    server = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(directory / "launch.json", {"pid": server.pid, "command": command, "environment_overrides": overrides, "started_at": datetime.now(timezone.utc).isoformat()})
    print(f"server PID {server.pid} loading; log {directory / 'server.log'}", flush=True)
    url = f"http://127.0.0.1:{arguments.port}"
    deadline = time.monotonic() + arguments.startup_timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            detail = server_exit_detail(directory / "server.log")
            suffix = f"\n{detail}" if detail else ""
            raise RuntimeError(f"server exited with {server.returncode}; see {directory / 'server.log'}{suffix}")
        try:
            if requests.get(url + "/health", timeout=3).ok:
                print(f"server ready after {time.monotonic() - deadline + arguments.startup_timeout:.0f}s", flush=True)
                return server, url
        except requests.RequestException:
            pass
        time.sleep(3)
    stop_server(server)
    raise TimeoutError(f"server not ready within {arguments.startup_timeout}s")


def stop_server(server):
    if server is None or server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=60)


def check_server(arguments, info):
    states = mono_states(info)
    if arguments.mode == "mono" and not states:
        raise RuntimeError("Server has no Flash-Next Monokernel runtime; start it with SGLANG_AMD_FLASH_NEXT_0518=1")
    if arguments.mode == "native" and states:
        raise RuntimeError("Server has the Monokernel runtime attached but --mode native was requested")
    backend = info.get("cuda_graph_backend_decode")
    if (backend == "full") != arguments.graph:
        raise RuntimeError(f"Server decode graph backend is {backend!r} but --graph={arguments.graph}")
    if info.get("kv_cache_dtype") != "bfloat16":
        raise RuntimeError(f"Expected BF16 KV cache, server uses {info.get('kv_cache_dtype')}")
    if max(arguments.batch_size) > info["max_running_requests"]:
        raise RuntimeError(f"Server max_running_requests={info['max_running_requests']} is below batch size {max(arguments.batch_size)}")
    if arguments.profile and not any("amd_profile_ranges" in state for state in info.get("internal_states", [])):
        raise RuntimeError("--profile needs a server started with SGLANG_AMD_PROFILE_RANGES=1")


# ---------------------------------------------------------------- requests


async def generate(session, url, token_ids, output_len):
    payload = {"input_ids": token_ids, "stream": True, "sampling_params": {"temperature": 0, "ignore_eos": True, "max_new_tokens": output_len}}
    arrivals = []
    final = None
    started = time.perf_counter()
    async with session.post(url + "/generate", json=payload) as response:
        response.raise_for_status()
        async for raw in response.content:
            if not raw.startswith(b"data:") or raw[5:].strip() == b"[DONE]":
                continue
            event = json.loads(raw[5:])
            if "error" in event:
                raise RuntimeError(event)
            count = event["meta_info"]["completion_tokens"]
            now = time.perf_counter() - started
            arrivals.extend([now] * (count - len(arrivals)))
            final = event
    e2e = time.perf_counter() - started
    metadata = final["meta_info"]
    if metadata["completion_tokens"] != output_len or len(arrivals) != output_len:
        raise RuntimeError(f"expected {output_len} output tokens, got {metadata['completion_tokens']}")
    if metadata["prompt_tokens"] != len(token_ids) or metadata.get("cached_tokens", 0) or metadata.get("num_retractions", 0):
        raise RuntimeError(f"prompt/cache/retraction check failed: {metadata}")
    prefill = metadata["prefill_finished_time"] - metadata["forward_entry_time"] if "prefill_finished_time" in metadata and "forward_entry_time" in metadata else None
    return {"e2e_s": e2e, "ttft_s": arrivals[0], "tpot_s": (arrivals[-1] - arrivals[0]) / (output_len - 1) if output_len > 1 else None,
            "prefill_s": prefill, "arrivals_s": arrivals, "output_ids": final.get("output_ids")}


async def wave_async(url, inputs, output_len):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
        started = time.perf_counter()
        results = await asyncio.gather(*(generate(session, url, tokens, output_len) for tokens in inputs))
        return results, time.perf_counter() - started


def wave(url, inputs, output_len):
    post(url, "/flush_cache")
    results, wall = asyncio.run(wave_async(url, inputs, output_len))
    batch = len(inputs)
    first = min(result["ttft_s"] for result in results)
    last = max(result["arrivals_s"][-1] for result in results)
    decode = last - first
    if batch > 1 and max(result["ttft_s"] for result in results) > min(result["arrivals_s"][-1] for result in results):
        print(f"WARNING: batch {batch} wave was serialized (a request started after another finished)", flush=True)
    return {"batch_size": batch, "wall_s": wall, "decode_window_s": decode,
            "decode_tps": batch * (output_len - 1) / decode if output_len > 1 and decode > 0 else None,
            "output_tps": batch * output_len / wall, "requests": results}


# ---------------------------------------------------------------- profile


def profile_case(url, directory, event_directory, inputs, output_len, expected_decode_steps):
    from sgl_amd_mono.common.profile_trace import merge_profiles

    directory.mkdir(parents=True)
    (directory / "ranks").mkdir()
    (directory / "host").mkdir()
    configuration = {"output_dir": str(directory / "ranks"), "activities": ["CPU", "GPU"], "with_stack": False, "record_shapes": True,
                     "profile_by_stage": False, "merge_profiles": False, "profile_id": directory.name}
    post(url, "/flush_cache")
    marker = event_directory / "capture.active"
    window_start = time.time_ns() / 1000
    marker.write_text(str(window_start))
    try:
        post(url, "/start_profile", configuration, timeout=3600)
        try:
            measured = wave(url, inputs, output_len)
        finally:
            post(url, "/stop_profile", timeout=3600)
    finally:
        marker.unlink(missing_ok=True)
    window_end = time.time_ns() / 1000
    for path in event_directory.glob("host-*.jsonl"):
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        (directory / "host" / path.name).write_text("".join(json.dumps(event) + "\n" for event in events if window_start <= event["ts"] <= window_end))
    traces = sorted((directory / "ranks").glob("*TP-*.trace.json.gz"))
    summary = merge_profiles(traces, directory / "inference.perfetto.json.gz", directory / "host", (), expected_ranks=8,
                             expected_decode_steps=expected_decode_steps, expected_decode_token_rows=len(inputs) * (output_len - 1))
    save(directory / "summary.json", summary)
    return {"trace": str(directory / "inference.perfetto.json.gz"), "events": summary["total_events"], "wall_s": measured["wall_s"]}


# ---------------------------------------------------------------- statistics / report


def describe(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0, "min": min(values), "max": max(values)}


def summarize(case, waves):
    requests_ = [request for item in waves for request in item["requests"]]
    return {
        "batch_size": case[0], "input_len": case[1], "output_len": case[2], "waves": len(waves),
        "ttft_s": describe([request["ttft_s"] for request in requests_]),
        "tpot_s": describe([request["tpot_s"] for request in requests_]),
        "e2e_s": describe([request["e2e_s"] for request in requests_]),
        "prefill_s": describe([request["prefill_s"] for request in requests_]),
        "wave_wall_s": describe([item["wall_s"] for item in waves]),
        "decode_tps": describe([item["decode_tps"] for item in waves]),
        "output_tps": describe([item["output_tps"] for item in waves]),
    }


def mean(summary, key, scale=1.0):
    value = summary[key]
    return "-" if value is None else f"{value['mean'] * scale:.2f}"


def render(arguments, summaries):
    title = f"Flash-Next TP8 BF16 | mode={arguments.mode} | graph={'on' if arguments.graph else 'off'} | repeats={arguments.repeats}"
    lines = [f"# {title}", "",
             "| batch | in→out | TTFT ms | TPOT ms | E2E s | prefill ms | decode tok/s | output tok/s |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for summary in summaries:
        lines.append(f"| {summary['batch_size']} | {summary['input_len']}→{summary['output_len']} | {mean(summary, 'ttft_s', 1000)} | {mean(summary, 'tpot_s', 1000)} | "
                     f"{mean(summary, 'e2e_s')} | {mean(summary, 'prefill_s', 1000)} | {mean(summary, 'decode_tps')} | {mean(summary, 'output_tps')} |")
    lines += ["", "TTFT/TPOT/E2E/prefill: mean over all requests of all repeats. decode tok/s = batch*(out-1)/(last token - first token) of a wave; "
              "output tok/s = batch*out/wave wall time. Timed requests contain no shadow-accuracy work."]
    if arguments.profile:
        lines += ["", "With --profile the server wraps every module forward in a profiler range, which slows eager steps "
                  "(prefill, and decode without --graph). Rerun without --profile for clean TTFT/prefill numbers."]
    return "\n".join(lines) + "\n"


def write_csv(path, summaries):
    keys = ("ttft_s", "tpot_s", "e2e_s", "prefill_s", "wave_wall_s", "decode_tps", "output_tps")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["batch_size", "input_len", "output_len", "waves"] + [f"{key}_{stat}" for key in keys for stat in ("mean", "std", "min", "max")])
        for summary in summaries:
            row = [summary["batch_size"], summary["input_len"], summary["output_len"], summary["waves"]]
            for key in keys:
                row += [None if summary[key] is None else summary[key][stat] for stat in ("mean", "std", "min", "max")]
            writer.writerow(row)


# ---------------------------------------------------------------- main


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("native", "mono"), required=True, help="decode path: native SGLang or Flash-Next Monokernel")
    parser.add_argument("--graph", action="store_true", help="capture decode CUDA graphs for exactly the requested batch sizes (no padding)")
    parser.add_argument("--profile", action="store_true", help="after the timed repeats, capture one torch-profiler trace per case (Perfetto .json.gz); slows eager steps of the timed runs too")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1], help="concurrent requests per wave (mono supports 1 and 2)")
    parser.add_argument("--input-len", type=int, nargs="+", default=[512], help="prompt tokens per request")
    parser.add_argument("--output-len", type=int, nargs="+", default=[64], help="generated tokens per request (ignore_eos)")
    parser.add_argument("--repeats", type=int, default=3, help="timed waves per case")
    parser.add_argument("--warmups", type=int, default=1, help="untimed waves per case before timing")
    parser.add_argument("--seed", type=int, default=42, help="seed for the random prompt token ids and the server")
    parser.add_argument("--url", help="attach to a running server instead of launching one (it must match --mode/--graph)")
    parser.add_argument("--port", type=int, default=30160)
    parser.add_argument("--model-path", default=MODEL)
    parser.add_argument("--output-dir", type=Path, help="default: bench_results/<timestamp>_<mode>_<graph>")
    parser.add_argument("--keep-server", action="store_true", help="leave the launched server running at the end")
    parser.add_argument("--numa", choices=("auto", "on", "off"), default="off", help="bind rank i to NUMA node i (on), or only if every node has enough free memory (auto); other jobs on this host fill NUMA 0, so off by default")
    parser.add_argument("--strict-accuracy", action="store_true", help="mono: refuse requests when the shadow accuracy check fails instead of recording it")
    parser.add_argument("--allow-busy-gpus", action="store_true", help="only warn, instead of refusing to start, when another job is using the GPUs")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("server_args", nargs=argparse.REMAINDER, help="extra sglang.launch_server arguments after '--'")
    arguments = parser.parse_args()
    arguments.server_args = arguments.server_args[1:] if arguments.server_args[:1] == ["--"] else arguments.server_args
    if arguments.mode == "mono" and not set(arguments.batch_size) <= set(MONO_BATCH_SIZES):
        parser.error("mono decode supports batch sizes 1 and 2")
    if min(arguments.batch_size) < 1 or min(arguments.input_len) < 1 or min(arguments.output_len) < 2:
        parser.error("batch >= 1, input >= 1 and output >= 2 are required")
    if max(arguments.input_len) + max(arguments.output_len) > CONTEXT:
        parser.error(f"input + output must fit the {CONTEXT}-token context")
    if arguments.repeats < 1 or arguments.warmups < 1:
        parser.error("--repeats and --warmups must be >= 1 (the first wave per batch size runs the mono shadow check)")
    if arguments.url and arguments.server_args:
        parser.error("extra server arguments only apply when this script launches the server")
    return arguments


def main():
    arguments = parse_arguments()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = (arguments.output_dir or ROOT / "bench_results" / f"{stamp}_{arguments.mode}_{'graph' if arguments.graph else 'eager'}").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    cases = [(batch, input_len, output_len) for batch in sorted(arguments.batch_size) for input_len in arguments.input_len for output_len in arguments.output_len]
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "argv": sys.argv, "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()},
              "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(), "cases": [], "complete": False}
    server = None
    try:
        report["gpu_usage_before"] = check_gpus_idle(require=not (arguments.allow_busy_gpus or arguments.url))
        save(directory / "results.json", report)
        if arguments.url:
            url = arguments.url.rstrip("/")
        else:
            server, url = start_server(arguments, directory)
        info = server_info(url)
        save(directory / "server_info.json", info)
        check_server(arguments, info)
        report["server"] = {key: info.get(key) for key in ("model_path", "tp_size", "dtype", "kv_cache_dtype", "attention_backend", "moe_runner_backend", "cuda_graph_backend_decode",
                                                            "cuda_graph_bs_decode", "disable_overlap_schedule", "disable_radix_cache", "max_running_requests", "context_length", "random_seed", "version")}
        inputs = make_inputs(info["model_path"], arguments.input_len, max(arguments.batch_size), arguments.seed)
        report["input_sha256"] = {str(length): hashlib.sha256(json.dumps(rows).encode()).hexdigest() for length, rows in inputs.items()}
        summaries = []
        for batch, input_len, output_len in cases:
            prompts = inputs[input_len][:batch]
            label = f"B{batch} {input_len}->{output_len}"
            for _ in range(arguments.warmups):
                wave(url, prompts, min(output_len, 128))
            before = mono_states(server_info(url))
            waves = []
            for repeat in range(arguments.repeats):
                measured = wave(url, prompts, output_len)
                waves.append(measured)
                first = measured["requests"][0]
                print(f"{label} #{repeat + 1}: TTFT={first['ttft_s'] * 1000:.1f}ms TPOT={first['tpot_s'] * 1000:.2f}ms E2E={first['e2e_s']:.3f}s decode={measured['decode_tps']:.2f}tok/s", flush=True)
            after = mono_states(server_info(url))
            audit = None
            if arguments.mode == "mono":
                steps = after[0]["completed_decode_steps"] - before[0]["completed_decode_steps"]
                shadow = [len(state["full_model_shadow_checks"]) for state in after] != [len(state["full_model_shadow_checks"]) for state in before]
                if steps <= 0 or shadow:
                    raise RuntimeError(f"{label}: timed waves did not run purely on the mono path (decode steps {steps}, shadow work {shadow})")
                audit = {"mono_decode_steps": steps, "acceptance_status": sorted({state["acceptance_status"] for state in after})}
            summary = summarize((batch, input_len, output_len), waves)
            summary["mono_audit"] = audit
            if arguments.profile:
                summary["profile"] = profile_case(url, directory / "profile" / f"b{batch}_{input_len}-{output_len}", directory / "profile_events", prompts, output_len, output_len - 1 if batch == 1 else None)
                print(f"{label}: trace {summary['profile']['trace']}", flush=True)
            summaries.append(summary)
            report["cases"].append({**summary, "samples": [{key: value for key, value in item.items() if key != "requests"} | {"requests": [{key: value for key, value in request.items() if key != "output_ids"} for request in item["requests"]]} for item in waves]})
            save(directory / "results.json", report)
        report["gpu_usage_after"] = gpu_usage(samples=1)
        if arguments.mode == "mono":
            report["mono_acceptance"] = sorted({state["acceptance_status"] for state in mono_states(server_info(url))})
        report["complete"] = True
        table = render(arguments, summaries)
        (directory / "summary.md").write_text(table)
        write_csv(directory / "summary.csv", summaries)
        print("\n" + table)
        if arguments.mode == "mono" and report["mono_acceptance"] != ["serving_shadow_passed"]:
            print(f"NOTE: mono shadow accuracy status is {report['mono_acceptance']}; these are performance numbers only, not an accuracy pass.")
        print(f"results: {directory}")
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(directory / "results.json", report)
        if server is not None and not arguments.keep_server:
            stop_server(server)
        elif server is not None:
            print(f"server left running: PID {server.pid}, {url}")


if __name__ == "__main__":
    main()
