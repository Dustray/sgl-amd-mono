import argparse
import ctypes
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HIP_GRAPH_ACCUMULATE_DISPATCH", "0")

import torch

from sgl_amd_mono.qwen38_27b.gfx936.program import HIPMonoProgram
from sgl_amd_mono.qwen38_27b.report import implementation_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    program = HIPMonoProgram.get(torch.device("cuda", torch.cuda.current_device()))
    probe = program.library.mono_probe_synchronization
    probe.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]
    state = torch.zeros(3, device="cuda", dtype=torch.int32)
    report = {"scope": "bounded grid synchronization, missing-participant fault injection, latched failure and fresh-state recovery", "implementation_sha256": implementation_digest(), "program": program.manifest, "checks": [], "passed": False}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")

    def launch(blocks):
        program.check(probe(state.data_ptr(), program.barrier_timeout_ticks, blocks, torch.cuda.current_stream().cuda_stream))

    save()
    for mode in ("eager", "graph"):
        for blocks, clear_state in ((80, True), (79, True), (80, False), (80, True)):
            if clear_state:
                state.zero_()
            graph = None
            if mode == "graph":
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch(blocks)
            start = time.monotonic()
            if graph is None:
                launch(blocks)
            else:
                graph.replay()
            event = torch.cuda.Event()
            event.record()
            while not event.query():
                if time.monotonic() - start > 5:
                    report["error"] = "Synchronization fault injection exceeded its completion deadline"
                    save()
                    os._exit(124)
                time.sleep(0.01)
            actual = state.tolist()
            expected_failure = blocks == 79 or not clear_state
            check = {"mode": mode, "blocks": blocks, "fresh_state": clear_state, "state": actual, "elapsed_seconds": time.monotonic() - start, "passed": actual[2] == int(expected_failure)}
            report["checks"].append(check)
            save()
            print(json.dumps(check), flush=True)
            if not check["passed"]:
                raise RuntimeError("Incorrect synchronization failure state")
    report["passed"] = True
    save()


if __name__ == "__main__":
    main()
