import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import torch
import sgl_amd_mono_0518

from sgl_amd_mono_0518.hip_ffn import FusedBF16FFN, reference_ffn, validate_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-shape", action="store_true")
    arguments = parser.parse_args()
    torch.manual_seed(20261105)
    shapes = [(256, 512)]
    if arguments.full_shape:
        shapes.append((5120, 17408))
    report = {
        "flydsl": importlib.metadata.version("flydsl"),
        "torch": importlib.metadata.version("torch"),
        "architecture": torch.cuda.get_device_properties(0).gcnArchName,
        "kernel_launches_per_ffn": 2,
        "runtime": "HIP modules; FlyDSL compilation isolated in a subprocess",
        "activation_contract": "BF16 SiLU rounding before BF16 multiplication",
        "validation_nrmse_limit": 0.002,
        "kernel_source_sha256": hashlib.sha256((Path(sgl_amd_mono_0518.__file__).parent / "bf16_ffn.py").read_bytes()).hexdigest(),
        "scope": "FFN computation only; attention, residual and normalization remain outside these kernels",
        "checks": [],
        "passed": False,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    for hidden_size, intermediate_size in shapes:
        gate_up = torch.randn(2 * intermediate_size, hidden_size, device="cuda", dtype=torch.bfloat16) / math.sqrt(hidden_size)
        down = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=torch.bfloat16) / math.sqrt(intermediate_size)
        fused = FusedBF16FFN(gate_up, down)
        for rows in [1, 2]:
            hidden = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
            executions_before = fused.executions.item()
            started = time.perf_counter()
            actual = fused(hidden)
            torch.cuda.synchronize()
            check = {"hidden_size": hidden_size, "intermediate_size": intermediate_size, "rows": rows, "first_call_seconds": time.perf_counter() - started}
            expected = reference_ffn(hidden, gate_up, down)
            check["eager"] = validate_result(actual, expected)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = fused(hidden)
            check["graph"] = []
            for replay in range(3):
                hidden.normal_()
                expected = reference_ffn(hidden, gate_up, down)
                graph.replay()
                torch.cuda.synchronize()
                check["graph"].append(validate_result(captured, expected))
            check["device_execution_delta"] = fused.executions.item() - executions_before
            if check["device_execution_delta"] != 4:
                raise RuntimeError(f"Eager and graph executions were not all counted: {check}")
            report["checks"].append(check)
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(check), flush=True)
    report["passed"] = True
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
