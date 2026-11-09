import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import functional

from sgl_amd_mono.qwen38_27b.gfx936.check import native_k2, native_norm
from sgl_amd_mono.qwen38_27b.gfx936.ffn import numerical_error
from sgl_amd_mono.qwen38_27b.gfx936.program import DenseK2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[2])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--captured-inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    from lightop import fuse_silu_and_mul

    mapping = json.loads((arguments.model_path / "model.safetensors.index.json").read_text())["weight_map"]
    captured = torch.load(arguments.captured_inputs, map_location="cpu", weights_only=True)["k2_inputs"] if arguments.captured_inputs else None
    report = {"scope": "K2 intermediate and same-input numerical diagnosis", "checks": []}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    def weight(layer_index, suffix):
        name = f"model.language_model.layers.{layer_index}.{suffix}"
        with safe_open(arguments.model_path / mapping[name], framework="pt", device="cpu") as checkpoint:
            return checkpoint.get_tensor(name).cuda()

    for layer_index in arguments.layers:
        prefix = "self_attn.o_proj" if layer_index % 4 == 3 else "linear_attn.out_proj"
        gate_up = torch.cat((weight(layer_index, "mlp.gate_proj.weight"), weight(layer_index, "mlp.up_proj.weight")))
        kernel = DenseK2(weight(layer_index, prefix + ".weight"), weight(layer_index, "post_attention_layernorm.weight"), gate_up, weight(layer_index, "mlp.down_proj.weight"))
        for seed in ([None] if captured else arguments.seeds):
            for rows in ((captured[layer_index][0].shape[0],) if captured else (1, 2)):
                if captured:
                    core, residual = (tensor.cuda() for tensor in captured[layer_index])
                else:
                    generator = torch.Generator(device="cuda").manual_seed(20261105 + 100 * layer_index + seed)
                    core = torch.randn((rows, 6144), generator=generator, device="cuda", dtype=torch.bfloat16)
                    residual = torch.randn((rows, 5120), generator=generator, device="cuda", dtype=torch.bfloat16)
                actual, actual_residual = kernel(core, residual)
                expected, expected_residual, stages = native_k2(kernel, core, residual)
                normalized_same_input, _ = native_norm(kernel.attention_output[:rows], residual, kernel.weights[1])
                projected_same_input = functional.linear(kernel.normalized[:rows], gate_up)
                activated_same_input = torch.empty_like(kernel.activated[:rows])
                fuse_silu_and_mul(kernel.projected[:rows], activated_same_input)
                output_same_input = functional.linear(kernel.activated[:rows], kernel.weights[-1])
                check = {
                    "layer": layer_index, "seed": seed, "rows": rows,
                    "output": numerical_error(actual, expected),
                    "residual": numerical_error(actual_residual, expected_residual),
                    "stages": {name: numerical_error(getattr(kernel, name)[:rows], reference) for name, reference in stages.items()},
                    "same_input": {
                        "normalized": numerical_error(kernel.normalized[:rows], normalized_same_input),
                        "projected": numerical_error(kernel.projected[:rows], projected_same_input),
                        "activated": numerical_error(kernel.activated[:rows], activated_same_input),
                        "output": numerical_error(actual, output_same_input),
                    },
                }
                report["checks"].append(check)
                arguments.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"layer": layer_index, "seed": seed, "rows": rows, "output_nrmse": check["output"]["normalized_rms_error"], "stages": {name: value["normalized_rms_error"] for name, value in check["stages"].items()}, "same_input": {name: value["normalized_rms_error"] for name, value in check["same_input"].items()}}), flush=True)
        del kernel, gate_up
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
