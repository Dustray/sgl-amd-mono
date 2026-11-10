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

from benchmark_flash_next_matrix import save


ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def stop(pid):
    if not Path(f"/proc/{pid}/cmdline").exists():
        return
    assert os.getpgid(pid) == pid, pid
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(1)
    os.killpg(pid, signal.SIGKILL)
    time.sleep(3)


def ready(server, logfile):
    deadline = time.monotonic() + 2400
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Service exited {server.returncode}; see {logfile}")
        try:
            response = requests.get("http://127.0.0.1:30000/health", timeout=3)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    raise TimeoutError(f"Service startup timed out: {logfile}")


def benchmark(mode, directory, extra=()):
    output = directory / ("native_graph_detailed" if mode == "native_graph" else mode)
    command = [sys.executable, "-u", str(ROOT / "tools/benchmark_flash_next_matrix.py"), "--mode", mode, "--output-dir", str(output), "--fixture", str(directory / "fixture.json"), *extra]
    with (directory / f"{mode}-client.log").open("w") as logfile:
        result = subprocess.run(command, cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{mode} benchmark failed; see {directory / (mode + '-client.log')}")


def run(arguments):
    directory = arguments.run_dir.resolve()
    launch = json.loads((directory / "launch.json").read_text())
    pid = launch["pid"]
    assert pid == arguments.native_pid
    if arguments.reuse_shared_pid:
        environment = {entry.split(b"=", 1)[0].decode(): entry.split(b"=", 1)[1].decode() for entry in Path(f"/proc/{arguments.reuse_shared_pid}/environ").read_bytes().split(b"\0") if b"=" in entry}
    elif arguments.resume_shared:
        environment = os.environ.copy()
        environment.update(launch["environment_overrides"])
    else:
        process_args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        assert any(b"sglang" in value for value in process_args)
        environment = {entry.split(b"=", 1)[0].decode(): entry.split(b"=", 1)[1].decode() for entry in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0") if b"=" in entry}
    environment.update(SGLANG_AMD_MONO_0518="0", SGLANG_AMD_PROFILE_RANGES="0", SGLANG_AMD_FLASH_ACCURACY_REFERENCE="0", SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D="1")
    for name in ("SGLANG_AMD_FLASH_ACCURACY_TRACE_DIR", "SGLANG_AMD_PROFILE_EVENT_DIR"):
        environment.pop(name, None)
    manifest = json.loads((directory / "comparison.json").read_text()) if arguments.resume_shared or arguments.reuse_shared_pid else {"started_at": now(), "stage": "waiting_native_graph_matrix", "passed": False, "services": []}
    if arguments.resume_shared:
        manifest["previous_failure"] = manifest.pop("error", None)
        manifest["resumed_at"] = now()
    save(directory / "comparison.json", manifest)
    while True:
        matrix = directory / "native_graph_detailed/matrix.json"
        if matrix.exists():
            state = json.loads(matrix.read_text())
            if "finished_at" in state:
                assert state["passed"], state
                break
        time.sleep(3)
    if not arguments.resume_shared and not arguments.reuse_shared_pid:
        stop(pid)
    if not launch.get("stopped"):
        launch.update(stopped=True, stopped_at=now())
    save(directory / "launch.json", launch)
    control = directory / "execution_mode.txt"
    control.write_text("native\n")
    runtime = directory / "shared_bf16_runtime"
    runtime.mkdir(exist_ok=bool(arguments.reuse_shared_pid))
    mono_environment = environment.copy()
    mono_environment["SGLANG_USE_FUSED_TOPK_SOFTMAX"] = "0"
    mono_environment.update(SGLANG_AMD_FLASH_NEXT_0518="1", SGLANG_AMD_FLASH_REPORT_DIR=str(runtime), SGLANG_AMD_FLASH_BENCHMARK_CONTROL_FILE=str(control), SGLANG_AMD_FLASH_SHADOW_STEPS="1", SGLANG_AMD_FLASH_SHADOW_REARM_ON_PREFILL="0", SGLANG_AMD_FLASH_SHADOW_RECORD_FAILURES="1", HIP_GRAPH_ACCUMULATE_DISPATCH="0")
    command = ["bash", str(ROOT / "launch_qwen_flash_next_monokernel.sh"), "--port", "30000", "--max-running-requests", "2", "--max-total-tokens", "8192", "--max-mamba-cache-size", "8", "--random-seed", "42", "--enable-metrics", "--enable-cache-report", "--stream-interval", "1", "--watchdog-timeout", "1200"]
    if arguments.reuse_shared_pid:
        server = type("ExistingService", (), {"pid": arguments.reuse_shared_pid, "poll": lambda self: None if Path(f"/proc/{self.pid}").exists() else -1})()
        service = manifest["services"][-1]
        assert service["pid"] == server.pid
        service["adopted_at"] = now()
    else:
        manifest["stage"] = "loading_shared_bf16"
        with (directory / "shared_bf16_server.log").open("w") as logfile:
            server = subprocess.Popen(command, cwd=ROOT, env=mono_environment, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        service = {"mode": "shared_bf16_native_and_monokernel", "pid": server.pid, "command": command, "started_at": now()}
        manifest["services"].append(service)
    save(directory / "comparison.json", manifest)
    try:
        ready(server, directory / "shared_bf16_server.log")
        service.setdefault("ready_at", now())
        for mode in ("native_bf16", "monokernel"):
            manifest["stage"] = "benchmark_" + mode
            save(directory / "comparison.json", manifest)
            if arguments.reuse_shared_pid and mode == "native_bf16":
                while True:
                    matrix = json.loads((directory / mode / "matrix.json").read_text())
                    if "finished_at" in matrix:
                        assert matrix["passed"], matrix
                        break
                    time.sleep(3)
                if arguments.previous_controller_pid:
                    controller = arguments.previous_controller_pid
                    assert b"tools/run_flash_next_comparison.py" in Path(f"/proc/{controller}/cmdline").read_bytes()
                    os.kill(controller, signal.SIGKILL)
            else:
                benchmark(mode, directory, ["--control-file", str(control), "--rank-report-dir", str(runtime)])
    finally:
        stop(server.pid)
        service["stopped_at"] = now()
        save(directory / "comparison.json", manifest)
    run_native_stages(directory, launch, environment, manifest)


def run_native_stages(directory, launch, environment, manifest):
    mtp_environment = environment.copy()
    mtp_environment.update(SGLANG_AMD_FLASH_NEXT_0518="0", SGLANG_USE_FUSED_TOPK_SOFTMAX="0")
    mtp_environment.pop("SGLANG_AMD_FLASH_BENCHMARK_CONTROL_FILE", None)
    command = launch["command"][:]
    loader_index = command.index("--model-loader-extra-config")
    command[loader_index + 1] = '{"enable_multithread_load":true,"num_threads":8}'
    numa_index = command.index("--numa-node") + 1
    command[numa_index:numa_index + 8] = [str(rank) for rank in range(8)]
    if launch["environment_overrides"].get("SGLANG_USE_FUSED_TOPK_SOFTMAX") == "1":
        legacy = directory / "native_graph_fused_topk1"
        if not legacy.exists():
            (directory / "native_graph_detailed").rename(legacy)
        manifest["legacy_native_graph_dir"] = str(legacy)
        manifest["stage"] = "loading_native_graph_fixed"
        with (directory / "native_graph_fixed-server.log").open("w") as logfile:
            server = subprocess.Popen(command, cwd=ROOT, env=mtp_environment, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        fixed_service = {"mode": "native_graph_fixed_topk", "pid": server.pid, "command": command[:], "started_at": now(), "fused_topk_softmax": "0"}
        manifest["services"].append(fixed_service)
        save(directory / "comparison.json", manifest)
        try:
            ready(server, directory / "native_graph_fixed-server.log")
            fixed_service["ready_at"] = now()
            manifest["stage"] = "benchmark_native_graph_fixed"
            save(directory / "comparison.json", manifest)
            benchmark("native_graph", directory)
        finally:
            stop(server.pid)
            fixed_service["stopped_at"] = now()
            save(directory / "comparison.json", manifest)
    command += ["--speculative-moe-runner-backend", "aiter", "--speculative-algorithm", "EAGLE", "--speculative-num-steps", "3", "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4"]
    manifest["stage"] = "loading_mtp"
    with (directory / "mtp-server.log").open("w") as logfile:
        server = subprocess.Popen(command, cwd=ROOT, env=mtp_environment, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
    service = {"mode": "native_mtp_graph", "pid": server.pid, "command": command, "started_at": now(), "fused_topk_softmax": "0"}
    manifest["services"].append(service)
    save(directory / "comparison.json", manifest)
    (directory / "mtp-server.pid").write_text(str(server.pid) + "\n")
    ready(server, directory / "mtp-server.log")
    service["ready_at"] = now()
    manifest["stage"] = "benchmark_mtp"
    save(directory / "comparison.json", manifest)
    benchmark("mtp", directory)
    manifest.update(stage="complete", passed=True, finished_at=now(), final_service_pid=server.pid)
    save(directory / "comparison.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--native-pid", type=int, required=True)
    parser.add_argument("--resume-shared", action="store_true")
    parser.add_argument("--reuse-shared-pid", type=int)
    parser.add_argument("--previous-controller-pid", type=int)
    parser.add_argument("--resume-native-stages", action="store_true")
    arguments = parser.parse_args()
    try:
        if arguments.resume_native_stages:
            directory = arguments.run_dir.resolve()
            launch = json.loads((directory / "launch.json").read_text())
            manifest = json.loads((directory / "comparison.json").read_text())
            if "error" in manifest:
                manifest.setdefault("errors", []).append({"error": manifest.pop("error"), "failed_at": manifest.pop("failed_at", None)})
            manifest["native_stages_resumed_at"] = now()
            environment = os.environ.copy()
            environment.update(launch["environment_overrides"])
            environment.update(SGLANG_AMD_MONO_0518="0", SGLANG_AMD_PROFILE_RANGES="0", SGLANG_AMD_FLASH_ACCURACY_REFERENCE="0", HIP_GRAPH_ACCUMULATE_DISPATCH="0")
            run_native_stages(directory, launch, environment, manifest)
        else:
            run(arguments)
    except BaseException as error:
        path = arguments.run_dir / "comparison.json"
        manifest = json.loads(path.read_text()) if path.exists() else {}
        manifest.update(error=repr(error), failed_at=now())
        save(path, manifest)
        raise


if __name__ == "__main__":
    main()
