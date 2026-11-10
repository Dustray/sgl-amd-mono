import argparse
import ctypes
import json
from pathlib import Path
import struct

import numpy as np
import torch

from sgl_amd_mono.qwen38_27b.gfx936.attention import AttentionProgram
from sgl_amd_mono.qwen38_27b.report import implementation_digest, write_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    scale = struct.unpack("f", bytes.fromhex("3baa383c"))[0]
    lower = np.arange(0, 0x7BFF, dtype=np.uint16).view(np.float16).astype(np.float64)
    upper = np.arange(1, 0x7C00, dtype=np.uint16).view(np.float16).astype(np.float64)
    midpoints = ((lower + upper) / (2 * scale)).astype(np.float32)
    bits = midpoints.view(np.int32)
    positive = np.concatenate([(bits + offset).view(np.float32) for offset in (-1, 0, 1)])
    values = np.concatenate([positive, -positive])
    reference = (values * np.float32(scale)).astype(np.float16).astype(np.float32) * 8
    single_rounded = (values.astype(np.float64) * scale).astype(np.float16).astype(np.float32) * 8
    distinguishing = int(np.count_nonzero(single_rounded != reference))
    if distinguishing == 0:
        raise RuntimeError("Rounding regression inputs cannot distinguish the original compiler contraction")
    inputs = torch.from_numpy(values.copy()).cuda()
    expected = torch.from_numpy(reference).cuda()
    output = torch.empty_like(inputs)
    program = AttentionProgram.get(inputs.device)
    function = program.library.mono_attention_score_rounding
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

    def run():
        status = function(inputs.data_ptr(), output.data_ptr(), inputs.numel(), torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f"Attention score rounding probe failed: {status}")

    run()
    torch.cuda.synchronize()
    eager_mismatches = int((output != expected).sum())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    replays = []
    for reverse in (True, False, True):
        inputs.copy_(torch.from_numpy(values[::-1].copy() if reverse else values).cuda())
        wanted = expected.flip(0) if reverse else expected
        graph.replay()
        torch.cuda.synchronize()
        replays.append(int((output != wanted).sum()))
    report = {"scope": "production attention_score GPU helper against independent CPU FP32 multiply then FP16 rounding at both signs of every finite FP16 midpoint; changing Graph inputs", "implementation_sha256": implementation_digest(), "inputs": inputs.numel(), "single_rounding_would_differ": distinguishing, "eager_mismatches": eager_mismatches, "graph_mismatches": replays, "passed": eager_mismatches == 0 and all(value == 0 for value in replays)}
    write_report(arguments.output.parent, report, arguments.output.name)
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Attention score double-rounding regression failed")


if __name__ == "__main__":
    main()
