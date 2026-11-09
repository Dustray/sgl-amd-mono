import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import triton
import triton.language as tl
from safetensors import safe_open

from sgl_amd_mono.qwen38_flash_next.gfx936.program import HCMix
from sgl_amd_mono.qwen38_flash_next.gfx936.reference import _down_partials, _reduce_partials, deterministic_hc_mix
from sgl_amd_mono.qwen38_flash_next.gfx936.runner import error_metrics


@triton.jit
def activation(reduced, activated, rows: tl.constexpr, rank: tl.constexpr):
    offset = tl.program_id(0) * 512 + tl.arange(0, 512)
    values = tl.load(reduced + offset, mask=offset < rows * rank, other=0) / 4
    tl.store(activated + offset, values * tl.sigmoid(values), mask=offset < rows * rank)


@triton.jit
def projection(activated, weights, gates, rows: tl.constexpr, rank: tl.constexpr, hidden: tl.constexpr):
    row = tl.arange(0, 16)
    column = tl.program_id(0) * 32 + tl.arange(0, 32)
    branch = tl.arange(0, 4)
    coordinates = tl.reshape(branch[:, None] * hidden + column[None, :], (128,))
    inner = tl.arange(0, 64)
    total = tl.full((16, 128), 0, tl.float32)
    for start in range(0, rank, 64):
        values = tl.load(activated + row[:, None] * rank + start + inner[None, :], mask=(row[:, None] < rows) & (start + inner[None, :] < rank), other=0)
        weight = tl.load(weights + coordinates[:, None] * rank + start + inner[None, :], mask=start + inner[None, :] < rank, other=0)
        total = tl.dot(values, tl.trans(weight), total)
    tl.store(gates + row[:, None] * (hidden * 4) + coordinates[None, :], total, mask=row[:, None] < rows)


@triton.jit
def combine(normalized, gates, output, rows: tl.constexpr, hidden: tl.constexpr):
    row = tl.arange(0, 16)
    column = tl.program_id(0) * 32 + tl.arange(0, 32)
    branch = tl.arange(0, 4)
    offset = row[:, None, None] * (hidden * 4) + branch[None, :, None] * hidden + column[None, None, :]
    raw = tl.load(gates + offset, mask=row[:, None, None] < rows, other=0)
    values = tl.load(normalized + offset, mask=row[:, None, None] < rows, other=0).to(tl.float32)
    mixed = tl.sum(values * tl.sigmoid(raw), axis=1) / 4
    tl.store(output + row[:, None] * hidden + column[None, :], mixed, mask=row[:, None] < rows)


def main():
    from sglang.srt.layers.hyperconnection import GroupedGemmaRMSNorm

    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master"))
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.cuda.set_device(0)
    captured = torch.load(arguments.capture, map_location="cuda:0", weights_only=True)
    mapping = json.loads((arguments.model_path / "model.safetensors.index.json").read_text())["weight_map"]
    weights = []
    for suffix in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up"):
        name = f"model.language_model.layers.{arguments.layer}.mlp_hyper_connection.{suffix}.weight"
        with safe_open(arguments.model_path / mapping[name], framework="pt", device="cpu") as checkpoint:
            weights.append(checkpoint.get_tensor(name).contiguous().cuda())
    mixer = HCMix(*weights)
    norm = GroupedGemmaRMSNorm(10240, group_size=2560).cuda().to(torch.bfloat16)
    norm.weight = torch.nn.Parameter(weights[0], requires_grad=False)
    hidden = captured["intermediates"]["post_attention"]
    normalized = norm(hidden)
    actual, _ = mixer(hidden)
    expected = deterministic_hc_mix(normalized, *weights[1:], 4, 2560)
    rows = hidden.shape[0]
    partials = torch.empty((40, 16, 320), dtype=torch.float32, device="cuda")
    reduced = torch.empty((16, 320), dtype=torch.float32, device="cuda")
    activated = torch.empty((16, 320), dtype=torch.bfloat16, device="cuda")
    gates = torch.empty((rows, 10240), dtype=torch.float32, device="cuda")
    _down_partials[(40, 10)](normalized, weights[1], partials, rows, 10240, 320, num_warps=8)
    _reduce_partials[(10,)](partials, reduced, 320, 40, num_warps=8)
    activation[(triton.cdiv(rows * 320, 512),)](reduced, activated, rows, 320)
    projection[(80,)](activated, weights[2], gates, rows, 320, 2560, num_warps=8)
    reconstructed = torch.empty_like(expected)
    fused_gates_mix = torch.empty_like(expected)
    combine[(80,)](normalized, gates, reconstructed, rows, 2560, num_warps=8)
    combine[(80,)](normalized, mixer.gates, fused_gates_mix, rows, 2560, num_warps=8)
    report = {"capture": str(arguments.capture), "capture_sha256": hashlib.sha256(arguments.capture.read_bytes()).hexdigest(), "layer": arguments.layer, "normalized": error_metrics(mixer.normalized[:rows], normalized), "lowrank": error_metrics(mixer.lowrank[:rows], activated[:rows]), "raw_gates": error_metrics(mixer.gates[:rows], gates), "reconstructed_native_mix": error_metrics(reconstructed, expected), "triton_combine_of_fused_gates": error_metrics(fused_gates_mix, expected), "fused_combine_vs_triton_same_gates": error_metrics(actual, fused_gates_mix), "full_mix": error_metrics(actual, expected)}
    source = Path(__file__).with_suffix(".cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    binary = arguments.output.parent / f"hc-mix-variants-{digest[:16]}.so"
    subprocess.run(["/opt/dtk/bin/hipcc", "-O3", "-std=c++17", "-shared", "-fPIC", "--offload-arch=gfx936", "-ffp-contract=off", str(source), "-o", str(binary)], check=True)
    library = ctypes.CDLL(str(binary.resolve()))
    library.launch_mix_variants.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    variants = torch.empty((9, rows, 2560), dtype=torch.bfloat16, device="cuda")
    mixer.program.check(library.launch_mix_variants(normalized.data_ptr(), gates.data_ptr(), variants.data_ptr(), rows, torch.cuda.current_stream().cuda_stream))
    report["variants"] = [{"sigmoid": ("expf_division", "exp2_division", "exp2_approximate_reciprocal")[variant // 3], "sum": ("sequential", "02_13", "01_23")[variant % 3], "check": error_metrics(variants[variant], expected)} for variant in range(9)]
    report["variant_source_sha256"] = digest
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
