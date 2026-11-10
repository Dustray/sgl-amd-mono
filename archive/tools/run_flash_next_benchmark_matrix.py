import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import requests


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run(arguments):
    directory = arguments.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    results = directory / "results.jsonl"
    if results.exists():
        raise FileExistsError(f"Use a new output directory: {results}")
    response = requests.get(arguments.url + "/server_info", timeout=30)
    response.raise_for_status()
    info = response.json()
    save(directory / "server_info_before.json", info)
    assert info["tp_size"] == 8
    assert info["max_running_requests"] >= 2
    if arguments.mode == "native_graph":
        assert not info.get("speculative_algorithm")
        assert info["cuda_graph_config"]["decode"]["backend"] == "full"
    if arguments.mode == "mtp":
        assert info.get("speculative_algorithm") == "EAGLE"
    report = {
        "mode": arguments.mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "repeats": arguments.repeats,
        "batch_sizes": [1, 2],
        "workloads": [[input_len, output_len] for input_len in (64, 512, 2048) for output_len in (16, 64, 128)] + [[16, 1024]],
        "dataset": "random-ids",
        "seed": 42,
        "warmup": "standard SGLang script: 1024->16 for each batch size before each invocation",
        "definitions": {
            "output_throughput": "batch_size * output_len / (latency - last_ttft), as reported by SGLang",
            "decode_tps": "batch_size * (output_len - 1) / (latency - last_ttft)",
            "decode_s": "client stream EOF minus last request first-token arrival; includes final stream delivery",
        },
        "invocations": [],
    }
    save(directory / "matrix.json", report)
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    base = [
        sys.executable, "-u", "-m", "sglang.benchmark.one_batch_server",
        "--model-path", info["model_path"],
        "--local-tokenizer-path", info["tokenizer_path"],
        "--base-url", arguments.url,
        "--dataset-name", "random-ids",
        "--batch-size", "1", "2",
        "--seed", "42",
        "--result-filename", str(results),
    ]
    groups = [
        ("grid", ["--input-len", "64", "512", "2048", "--output-len", "16", "64", "128"]),
        ("long_output", ["--input-len", "16", "--output-len", "1024"]),
    ]
    try:
        for group, lengths in groups:
            for repeat in range(1, arguments.repeats + 1):
                name = f"{arguments.mode}-{group}-{repeat}"
                command = base + lengths + ["--run-name", name]
                invocation = {"name": name, "command": command, "started_at": datetime.now(timezone.utc).isoformat()}
                report["invocations"].append(invocation)
                save(directory / "matrix.json", report)
                print(f"Starting {name}; service reused", flush=True)
                with (directory / f"{name}.log").open("w") as logfile:
                    client = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    for line in client.stdout:
                        print(line, end="", flush=True)
                        logfile.write(line)
                        logfile.flush()
                    code = client.wait()
                invocation.update(returncode=code, finished_at=datetime.now(timezone.utc).isoformat())
                save(directory / "matrix.json", report)
                if code:
                    raise RuntimeError(f"{name} failed with code {code}")
        rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
        expected = {(batch, input_len, output_len) for batch in (1, 2) for input_len, output_len in report["workloads"]}
        counts = {key: 0 for key in expected}
        for row in rows:
            key = (row["batch_size"], row["input_len"], row["output_len"])
            counts[key] += 1
            assert row["latency"] > row["last_ttft"] > 0
        assert len(rows) == 20 * arguments.repeats
        assert set(counts.values()) == {arguments.repeats}, counts
        response = requests.get(arguments.url + "/server_info", timeout=30)
        response.raise_for_status()
        save(directory / "server_info_after.json", response.json())
        report.update(passed=True, measured_requests=len(rows))
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(directory / "matrix.json", report)
    print(f"{arguments.mode}: all 20 workloads complete; service still running", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("native_graph", "native_bf16", "monokernel", "mtp"), required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    assert arguments.repeats >= 2
    run(arguments)


if __name__ == "__main__":
    main()
