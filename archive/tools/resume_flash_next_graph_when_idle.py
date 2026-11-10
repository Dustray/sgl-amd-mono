import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_flash_next_matrix import save


def run(directory, timeout):
    directory = directory.resolve()
    launch = json.loads((directory / "launch.json").read_text())
    deadline = time.monotonic() + timeout
    stable = 0
    launch.update(stage="waiting_for_idle_gpus", passed=False, waiting_since=datetime.now(timezone.utc).isoformat())
    save(directory / "launch.json", launch)
    print("Waiting for all eight GPUs: VRAM used <1GiB and utilization <5%, three consecutive checks", flush=True)
    while stable < 3:
        output = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--json"], stderr=subprocess.DEVNULL, text=True)
        devices = json.loads(output)
        idle = len(devices) == 8 and all(float(device["vram Total Used Memory (MiB)"]) < 1024 and float(device["HCU use (%)"]) < 5 for device in devices.values())
        stable = stable + 1 if idle else 0
        save(directory / "gpu_wait_status.json", {"checked_at": datetime.now(timezone.utc).isoformat(), "devices": devices, "consecutive_idle_checks": stable})
        if time.monotonic() > deadline:
            raise TimeoutError("GPUs remain occupied; see gpu_wait_status.json")
        if stable < 3:
            time.sleep(10)
    process = Path(f"/proc/{launch['pid']}/cmdline")
    if process.exists() and process.read_bytes():
        raise RuntimeError("Recorded service still exists; refusing duplicate launch")
    attempt = launch.get("attempt", 1)
    save(directory / f"attempt-{attempt}-launch.json", launch)
    logfile = directory / "server.log"
    if logfile.exists():
        logfile.rename(directory / f"attempt-{attempt}-server.log")
    environment = os.environ.copy()
    environment.update(launch["environment_overrides"])
    for name in ("FLASH_NEXT_MEMORY_DIAGNOSTIC", "SGLANG_AMD_FLASH_BENCHMARK_CONTROL_FILE"):
        environment.pop(name, None)
    with logfile.open("w") as stream:
        server = subprocess.Popen(launch["command"], env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    launch.update(pid=server.pid, stage="loading", attempt=attempt + 1, started_at=datetime.now(timezone.utc).isoformat())
    for name in ("error", "failed_at", "diagnostic_environment"):
        launch.pop(name, None)
    save(directory / "launch.json", launch)
    (directory / "server.pid").write_text(str(server.pid) + "\n")
    print(f"GPUs idle; started Mono Graph service {server.pid}", flush=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("run_flash_next_graph_matrix.py")), "--run-dir", str(directory)], check=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("summarize_flash_next_graph.py")), "--run-dir", str(directory), "--update-testresult"], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    arguments = parser.parse_args()
    run(arguments.run_dir, arguments.timeout)
