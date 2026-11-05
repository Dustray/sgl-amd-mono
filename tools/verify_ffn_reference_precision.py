import argparse
import gc
import json
from pathlib import Path

import torch
from lightop import fuse_silu_and_mul
from safetensors import safe_open

from sgl_amd_mono_0518.hip_ffn import FusedBF16FFN, numerical_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    mapping = json.loads((arguments.model_path / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(layer_index, projection):
        name = f"model.language_model.layers.{layer_index}.mlp.{projection}_proj.weight"
        with safe_open(arguments.model_path / mapping[name], framework="pt", device="cpu") as checkpoint:
            return checkpoint.get_tensor(name).cuda()

    report = {"scope": "Native and fused BF16 FFN versus FP64 dot products with BF16 rounding between operations", "checks": []}
    for layer_index in (0, 61):
        gate_up = torch.cat((weight(layer_index, "gate"), weight(layer_index, "up")), dim=0)
        down = weight(layer_index, "down")
        fused = FusedBF16FFN(gate_up, down)
        generator = torch.Generator(device="cuda").manual_seed(20261105 + layer_index)
        for rows in (1, 2):
            hidden = torch.randn(rows, down.shape[0], device="cuda", dtype=torch.bfloat16, generator=generator)
            projected = torch.nn.functional.linear(hidden, gate_up)
            activated = torch.empty(rows, down.shape[1], device="cuda", dtype=torch.bfloat16)
            fuse_silu_and_mul(projected, activated)
            native = torch.nn.functional.linear(activated, down)
            actual = fused(hidden)
            accurate_projection = torch.nn.functional.linear(hidden.double(), gate_up.double()).bfloat16()
            gate, up = accurate_projection.chunk(2, dim=-1)
            accurate_activation = torch.nn.functional.silu(gate) * up
            accurate = torch.nn.functional.linear(accurate_activation.double(), down.double()).bfloat16()
            check = {
                "layer": layer_index,
                "rows": rows,
                "fused_vs_fp64": numerical_error(actual, accurate),
                "native_vs_fp64": numerical_error(native, accurate),
                "fused_vs_native": numerical_error(actual, native),
                "gate_up_bf16_mismatches_native_vs_fp64": int((projected != accurate_projection).sum()),
            }
            report["checks"].append(check)
            print(json.dumps(check), flush=True)
            arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        del fused, gate_up, down
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
