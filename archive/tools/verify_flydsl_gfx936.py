import argparse
import importlib.metadata
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends.rocm import RocmBackend
from flydsl.compiler.external_llvm import use_hcu_clang_codegen
from flydsl.expr import math as fxmath
from flydsl.runtime.device import is_hcu_arch


@flyc.kernel
def silu_mul_kernel(gate: fx.Tensor, value: fx.Tensor, output: fx.Tensor, count: fx.Int32):
    position = fx.block_idx.x * 256 + fx.thread_idx.x
    if position < count:
        gate_value = fx.Float32(gate[position])
        up_value = fx.Float32(value[position])
        activated = gate_value / (fx.Float32(1.0) + fxmath.exp(-gate_value))
        output[position] = fx.BFloat16(activated * up_value)


@flyc.jit
def launch_silu_mul(
    gate: fx.Tensor,
    value: fx.Tensor,
    output: fx.Tensor,
    count: fx.Int32,
    stream: fx.Stream,
):
    silu_mul_kernel(gate, value, output, count).launch(
        grid=((count + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
    )


@flyc.kernel
def bf16_mma_kernel(left: fx.Tensor, right: fx.Tensor, output: fx.Tensor):
    thread_id = fx.thread_idx.x
    left_buffer = fx.rocdl.make_buffer_tensor(left)
    right_buffer = fx.rocdl.make_buffer_tensor(right)
    output_buffer = fx.rocdl.make_buffer_tensor(output)
    left_tile = fx.slice(fx.zipped_divide(left_buffer, (64, 16)), (None, 0))
    right_tile = fx.slice(fx.zipped_divide(right_buffer, (64, 16)), (None, 0))
    output_tile = fx.slice(fx.zipped_divide(output_buffer, (64, 64)), (None, 0))
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    thread_mma = tiled_mma.thr_slice(thread_id)
    input_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
    output_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    left_copy = fx.make_tiled_copy_A(input_copy, tiled_mma).get_slice(thread_id)
    right_copy = fx.make_tiled_copy_B(input_copy, tiled_mma).get_slice(thread_id)
    result_copy = fx.make_tiled_copy_C(output_copy, tiled_mma).get_slice(thread_id)
    left_fragment = thread_mma.make_fragment_A(left_tile)
    right_fragment = thread_mma.make_fragment_B(right_tile)
    result_fragment = thread_mma.make_fragment_C(output_tile)
    fx.copy(input_copy, left_copy.partition_S(left_tile), left_copy.retile(left_fragment))
    fx.copy(input_copy, right_copy.partition_S(right_tile), right_copy.retile(right_fragment))
    result_fragment.fill(0)
    fx.gemm(mma_atom, result_fragment, left_fragment, right_fragment, result_fragment)
    fx.copy(output_copy, result_copy.retile(result_fragment), result_copy.partition_S(output_tile))


@flyc.jit
def launch_bf16_mma(left: fx.Tensor, right: fx.Tensor, output: fx.Tensor, stream: fx.Stream):
    bf16_mma_kernel(left, right, output).launch(
        grid=(1, 1, 1), block=(256, 1, 1), stream=stream
    )


def compare(actual, expected, *, absolute, relative):
    torch.testing.assert_close(actual, expected, atol=absolute, rtol=relative)
    difference = (actual.float() - expected.float()).abs()
    return {"max_absolute_error": difference.max().item(), "mean_absolute_error": difference.mean().item()}


def check_silu_mul():
    checks = []
    for count in [127, 5120, 17408]:
        gate = torch.randn(count, dtype=torch.bfloat16, device="cuda")
        value = torch.randn_like(gate)
        output = torch.empty_like(gate)
        expected = (torch.nn.functional.silu(gate.float()) * value.float()).bfloat16()
        stream = torch.cuda.current_stream()
        launch_silu_mul(gate, value, output, count, stream)
        torch.cuda.synchronize()
        eager = compare(output, expected, absolute=0.03125, relative=0.015625)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch_silu_mul(gate, value, output, count, torch.cuda.current_stream())
        replay_errors = []
        for replay in range(3):
            gate.normal_()
            value.normal_()
            expected = (torch.nn.functional.silu(gate.float()) * value.float()).bfloat16()
            graph.replay()
            torch.cuda.synchronize()
            replay_errors.append(compare(output, expected, absolute=0.03125, relative=0.015625))
        checks.append({"count": count, "eager": eager, "graph_replays_with_changed_inputs": 3, "graph_errors": replay_errors})
    return checks


def check_bf16_mma():
    left = torch.randn(64, 16, dtype=torch.bfloat16, device="cuda")
    right = torch.randn_like(left)
    output = torch.empty(64, 64, dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream()
    launch_bf16_mma(left, right, output, stream)
    torch.cuda.synchronize()
    expected = left.float() @ right.float().T
    eager = compare(output, expected, absolute=0.0001, relative=0.0001)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_bf16_mma(left, right, output, torch.cuda.current_stream())
    replay_errors = []
    for replay in range(3):
        left.normal_()
        right.normal_()
        graph.replay()
        torch.cuda.synchronize()
        replay_errors.append(compare(output, left.float() @ right.float().T, absolute=0.0001, relative=0.0001))
    return {"left_shape": [64, 16], "right_shape": [64, 16], "eager": eager, "graph_replays_with_changed_inputs": 3, "graph_errors": replay_errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.manual_seed(20261104)
    properties = torch.cuda.get_device_properties(0)
    target = RocmBackend.detect_target()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "torch": importlib.metadata.version("torch"),
        "flydsl": importlib.metadata.version("flydsl"),
        "hip": torch.version.hip,
        "device": {"name": properties.name, "arch": properties.gcnArchName, "compute_units": properties.multi_processor_count},
        "flydsl_target": {"backend": target.backend, "arch": target.arch, "warp_size": target.warp_size},
        "hcu_arch": is_hcu_arch(target.arch),
        "hcu_clang_codegen": use_hcu_clang_codegen(),
        "scope": "FlyDSL primitive correctness and graph validation; no model Monokernel is attached",
        "checks": {},
    }
    failures = []
    for name, check in [("silu_mul", check_silu_mul), ("bf16_mma", check_bf16_mma)]:
        try:
            report["checks"][name] = {"passed": True, "result": check()}
            print(f"{name}: PASS", flush=True)
        except Exception:
            report["checks"][name] = {"passed": False, "traceback": traceback.format_exc()}
            failures.append(name)
            print(f"{name}: FAIL\n{report['checks'][name]['traceback']}", flush=True)
    report["passed"] = not failures
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Report: {arguments.output}", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
