import argparse
import contextlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

os.environ.setdefault("SGLANG_USE_AITER_LINEAR_ATTN", "1")

from sgl_amd_mono.qwen38_27b.gfx936.check import native_k1, native_k2, native_norm
from sgl_amd_mono.qwen38_27b.gfx936.ffn import numerical_error
from sgl_amd_mono.qwen38_27b.gfx936.program import DenseK1, DenseK2
from sgl_amd_mono.qwen38_27b.report import implementation_digest


def compare(actual, expected):
    return {**numerical_error(actual, expected), "different_elements": int((actual != expected).sum()), "elements": actual.numel()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--captured-inputs", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(64)))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-native", action="store_true")
    arguments = parser.parse_args()
    mapping = json.loads((arguments.model_path / "model.safetensors.index.json").read_text())["weight_map"]
    captured = torch.load(arguments.captured_inputs, map_location="cpu", weights_only=True)
    report = {"scope": "isolated layers from original SGLang inputs, GDN states and outputs", "implementation_sha256": implementation_digest(), "captured_implementation_sha256": captured["implementation_sha256"], "step": captured["step"], "checks": []}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    def weight(layer_index, suffix):
        name = f"model.language_model.layers.{layer_index}.{suffix}"
        with safe_open(arguments.model_path / mapping[name], framework="pt", device="cpu") as checkpoint:
            return checkpoint.get_tensor(name).cuda()

    for layer_index in arguments.layers:
        inputs = {name: tensor.cuda() if tensor is not None else None for name, tensor in captured["layers"][layer_index].items()}
        is_gdn = layer_index % 4 != 3
        output_projection = "linear_attn.out_proj" if is_gdn else "self_attn.o_proj"
        kernel2 = DenseK2(weight(layer_index, output_projection + ".weight"), weight(layer_index, "post_attention_layernorm.weight"), torch.cat((weight(layer_index, "mlp.gate_proj.weight"), weight(layer_index, "mlp.up_proj.weight"))), weight(layer_index, "mlp.down_proj.weight"))
        check = {"layer": layer_index, "rows": inputs["hidden"].shape[0]}
        norm_weight = weight(layer_index, "input_layernorm.weight")
        _, residual = native_norm(inputs["hidden"], inputs["residual"], norm_weight)
        if is_gdn:
            kernel1 = DenseK1(norm_weight, torch.cat((weight(layer_index, "linear_attn.in_proj_qkv.weight"), weight(layer_index, "linear_attn.in_proj_z.weight"))), torch.cat((weight(layer_index, "linear_attn.in_proj_b.weight"), weight(layer_index, "linear_attn.in_proj_a.weight"))), weight(layer_index, "linear_attn.conv1d.weight"), weight(layer_index, "linear_attn.norm.weight"), weight(layer_index, "linear_attn.A_log").float(), weight(layer_index, "linear_attn.dt_bias"))
            actual_conv, actual_ssm = inputs["conv_state"].clone(), inputs["ssm_state"].clone()
            reference_conv, reference_ssm = inputs["conv_state"].clone(), inputs["ssm_state"].clone()
            actual_core, actual_residual = kernel1(inputs["hidden"], inputs["residual"], actual_conv, actual_ssm, inputs["indices"])
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) if arguments.profile_native else contextlib.nullcontext() as profile:
                reference_core, _, stages = native_k1(kernel1, inputs["hidden"], inputs["residual"], reference_conv, reference_ssm, inputs["indices"])
            if profile is not None:
                profile.export_chrome_trace(str(arguments.output.with_name(f"{arguments.output.stem}-layer-{layer_index}-trace.json")))
            check["K1"] = {"fused_vs_sglang": compare(actual_core, inputs["core"]), "reference_vs_sglang": compare(reference_core, inputs["core"]), "fused_vs_reference": compare(actual_core, reference_core), "conv_state": compare(actual_conv, reference_conv), "ssm_state": compare(actual_ssm, reference_ssm), "stages": {name: compare(getattr(kernel1, name)[:check["rows"]], value) for name, value in stages.items()}}
            layer_output, layer_residual = kernel2(actual_core, actual_residual)
            check["isolated_layer"] = {"output": compare(layer_output, inputs["output"]), "residual": compare(layer_residual, inputs["residual_output"])}
            del kernel1
        actual, actual_residual = kernel2(inputs["core"], residual)
        expected, expected_residual, stages = native_k2(kernel2, inputs["core"], residual)
        check["K2"] = {"fused_vs_sglang": compare(actual, inputs["output"]), "reference_vs_sglang": compare(expected, inputs["output"]), "residual_vs_sglang": compare(actual_residual, inputs["residual_output"]), "fused_vs_reference": compare(actual, expected), "stages": {name: compare(getattr(kernel2, name)[:check["rows"]], value) for name, value in stages.items()}}
        report["checks"].append(check)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"layer": layer_index, **{name: {metric: round(value["normalized_rms_error"], 8) for metric, value in check[name].items() if "normalized_rms_error" in value} for name in ("K1", "K2") if name in check}}), flush=True)
        del kernel2, inputs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
