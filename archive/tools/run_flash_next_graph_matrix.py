import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import requests

from benchmark_flash_next_matrix import save, stream_case


def run(directory):
    directory = directory.resolve()
    launch = json.loads((directory / "launch.json").read_text())
    url = "http://127.0.0.1:30000"
    print(f"Waiting for Mono Graph service {launch['pid']}", flush=True)
    try:
        deadline = time.monotonic() + 1800
        while True:
            process = Path(f"/proc/{launch['pid']}/cmdline")
            if not process.exists() or not process.read_bytes():
                raise RuntimeError("Server exited; see server.log")
            try:
                if requests.get(url + "/health", timeout=3).ok:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Graph startup timed out")
            time.sleep(3)
        launch.update(stage="validating_graph", ready_at=datetime.now(timezone.utc).isoformat())
        save(directory / "launch.json", launch)
        fixture = Path(launch["baseline_run_dir"]) / "fixture.json"
        inputs = json.loads(fixture.read_text())
        with requests.Session() as session:
            info = session.get(url + "/server_info", timeout=30).json()
            assert info["cuda_graph_config"]["decode"]["backend"] == "full"
            assert not info.get("speculative_algorithm")
            smoke = [{"batch_size": batch, **stream_case(session, url, inputs["512"][:batch], 2)} for batch in (1, 2)]
            save(directory / "graph_smoke.json", {"passed": True, "requests": smoke})
        print("Graph first-request verification complete; benchmarking matrix", flush=True)
        launch["stage"] = "benchmark_monokernel_graph"
        save(directory / "launch.json", launch)
        command = [sys.executable, "-u", str(Path(__file__).with_name("benchmark_flash_next_matrix.py")), "--mode", "monokernel", "--output-dir", str(directory / "monokernel_graph"), "--fixture", str(fixture), "--rank-report-dir", str(directory / "runtime")]
        with (directory / "benchmark.log").open("w") as logfile:
            result = subprocess.run(command, stdout=logfile, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Graph benchmark failed; see benchmark.log")
        launch.update(stage="complete", passed=True, finished_at=datetime.now(timezone.utc).isoformat())
        save(directory / "launch.json", launch)
        print("Mono Graph matrix complete", flush=True)
    except BaseException as error:
        launch.update(stage="failed", error=repr(error), failed_at=datetime.now(timezone.utc).isoformat())
        save(directory / "launch.json", launch)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    run(parser.parse_args().run_dir)
