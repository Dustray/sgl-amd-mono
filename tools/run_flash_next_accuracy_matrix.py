import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import requests


ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run_service(mode, directory, manifest):
    service_dir = directory / mode
    service_dir.mkdir()
    environment = os.environ.copy()
    overrides = {
        "SGLANG_AMD_FLASH_NEXT_0518": "1" if mode == "monokernel" else "0",
        "SGLANG_AMD_MONO_0518": "0",
        "SGLANG_AMD_FLASH_ACCURACY_REFERENCE": "1",
        "SGLANG_AMD_FLASH_ACCURACY_TRACE_DIR": str(service_dir / "trace"),
        "SGLANG_AMD_FLASH_REPORT_DIR": str(service_dir),
        "SGLANG_AMD_FLASH_SHADOW_STEPS": "63",
        "SGLANG_AMD_FLASH_SHADOW_REARM_ON_PREFILL": "1",
        "SGLANG_AMD_FLASH_SHADOW_RECORD_FAILURES": "1",
    }
    environment.update(overrides)
    launcher = "launch_qwen_flash_next_monokernel.sh" if mode == "monokernel" else "launch_qwen_flash_next.sh"
    command = ["bash", launcher, "--skip-server-warmup", "--enable-metrics", "--enable-cache-report", "--stream-interval", "1", "--max-running-requests", "2"]
    with (service_dir / "server.log").open("w") as log:
        server = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    manifest[mode] = {"pid": server.pid, "command": command, "environment": overrides, "started_at": datetime.now(timezone.utc).isoformat()}
    save(directory / "launch.json", manifest)
    try:
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"{mode} service exited: {server.returncode}")
            try:
                if requests.get("http://127.0.0.1:30160/health", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(3)
        else:
            raise TimeoutError(f"{mode} service startup timed out")
        print(f"{mode} service ready", flush=True)
        for batch in (1, 2):
            script = "verify_flash_next_correctness.py" if batch == 1 else "verify_flash_next_batch_accuracy.py"
            command = [sys.executable, "-u", str(ROOT / "tools" / script)]
            if batch == 1:
                command += ["run"]
            command += ["--mode", mode, "--trace-dir", str(service_dir / "trace"), "--lengths", *(str(length) for length in ([64, 512, 2048, 3072] if batch == 1 else [64, 512, 1536])), "--output", str(directory / f"{mode}-batch{batch}.json")]
            if mode == "monokernel":
                command += ["--diagnostic", "--rank-report-dir", str(service_dir)]
            with (directory / f"{mode}-batch{batch}.log").open("w") as log:
                client = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                while client.poll() is None:
                    if server.poll() is not None or "Scheduler hit an exception" in (service_dir / "server.log").read_text(errors="replace"):
                        raise RuntimeError(f"{mode} service failed during batch {batch}")
                    time.sleep(3)
                if client.returncode:
                    raise RuntimeError(f"{mode} batch {batch} client exited: {client.returncode}")
            finally:
                if client.poll() is None:
                    client.terminate()
                    client.wait(timeout=10)
            print(f"{mode} batch {batch} collection complete", flush=True)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        manifest[mode]["stopped_at"] = datetime.now(timezone.utc).isoformat()
        save(directory / "launch.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    directory = arguments.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"scope": "accuracy matrix with explicit deterministic reference; diagnostic failures remain failures"}
    for mode in ("monokernel", "native"):
        run_service(mode, directory, manifest)
    accepted = True
    for batch in (1, 2):
        command = [sys.executable, str(ROOT / "tools/compare_flash_next_accuracy_trace.py"), "--native", str(directory / f"native-batch{batch}.json"), "--fused", str(directory / f"monokernel-batch{batch}.json"), "--output", str(directory / f"comparison-batch{batch}.json")]
        result = subprocess.run(command, cwd=ROOT, check=False)
        accepted &= result.returncode == 0
    print(f"Accuracy collection complete; accepted={accepted}", flush=True)
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
