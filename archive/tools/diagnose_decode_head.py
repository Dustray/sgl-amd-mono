import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path)
    arguments = parser.parse_args()
    source = Path(__file__).with_suffix(".cpp").resolve()
    binary = arguments.output.with_suffix(".so").resolve()
    subprocess.run(["/opt/dtk/bin/hipcc", "-O3", "-std=c++17", "-fPIC", "-shared", "--offload-arch=gfx936", "-ffp-contract=off", str(source), "-o", str(binary)], check=True, timeout=180)
    library = ctypes.CDLL(str(binary))
    library.probe_head.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    mapping = json.loads((arguments.model_path / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(arguments.model_path / mapping["lm_head.weight"], framework="pt", device="cpu") as checkpoint:
        weight = checkpoint.get_tensor("lm_head.weight").cuda()
    report = {"scope": "diagnostic native LM-head 1024-stride split with BF16 MMAC and exact FP32 reduction tree", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "checks": [], "passed": False}
    for rows in (1, 2):
        for seed in (0, 42, 20261105):
            torch.manual_seed(seed)
            hidden = torch.randn((rows, 5120), device="cuda", dtype=torch.bfloat16) if arguments.capture is None else torch.load(arguments.capture, weights_only=True)["normalized"][:1].cuda().repeat(rows, 1)
            channels = weight.shape[0]
            expected = torch.nn.functional.linear(hidden, weight)
            for order in range(4):
                actual = torch.empty_like(expected)
                status = library.probe_head(hidden.data_ptr(), weight.data_ptr(), actual.data_ptr(), rows, channels, order, torch.cuda.current_stream().cuda_stream)
                if status:
                    raise RuntimeError(f"HIP launch failed: {status}")
                torch.cuda.synchronize()
                check = {"rows": rows, "seed": seed, "order": order, "mismatches": int((actual != expected).sum()), "max_error": float((actual.float() - expected.float()).abs().max())}
                report["checks"].append(check)
                arguments.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(check), flush=True)
    report["matching_orders"] = [order for order in range(4) if all(check["mismatches"] == 0 for check in report["checks"] if check["order"] == order)]
    report["passed"] = bool(report["matching_orders"])
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
