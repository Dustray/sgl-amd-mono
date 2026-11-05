import argparse
import gc
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

os.environ.setdefault("SGLANG_USE_AITER_LINEAR_ATTN", "1")

from sgl_amd_mono_0518.hip_monokernel import DenseK1, DenseK2
from sgl_amd_mono_0518.hip_ffn import numerical_error
from sgl_amd_mono_0518.mono_validation import native_k1, native_k2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3, 61])
    arguments = parser.parse_args()
    mapping = json.loads((arguments.model_path / "model.safetensors.index.json").read_text())["weight_map"]
    report = {"scope": "real BF16 weights; cooperative K1/K2 eager and graph; recurrent state and cache slot mapping", "checks": []}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    def weight(layer_index, suffix):
        name = f"model.language_model.layers.{layer_index}.{suffix}"
        with safe_open(arguments.model_path / mapping[name], framework="pt", device="cpu") as checkpoint:
            return checkpoint.get_tensor(name).cuda()

    for layer_index in arguments.layers:
        gate_up = torch.cat((weight(layer_index, "mlp.gate_proj.weight"), weight(layer_index, "mlp.up_proj.weight")))
        attention_prefix = "self_attn" if layer_index % 4 == 3 else "linear_attn"
        output_name = "o_proj" if attention_prefix == "self_attn" else "out_proj"
        kernel2 = DenseK2(weight(layer_index, f"{attention_prefix}.{output_name}.weight"), weight(layer_index, "post_attention_layernorm.weight"), gate_up, weight(layer_index, "mlp.down_proj.weight"))
        kernel1 = None
        if attention_prefix == "linear_attn":
            qkvz = torch.cat((weight(layer_index, "linear_attn.in_proj_qkv.weight"), weight(layer_index, "linear_attn.in_proj_z.weight")))
            ba = torch.cat((weight(layer_index, "linear_attn.in_proj_b.weight"), weight(layer_index, "linear_attn.in_proj_a.weight")))
            kernel1 = DenseK1(weight(layer_index, "input_layernorm.weight"), qkvz, ba, weight(layer_index, "linear_attn.conv1d.weight"), weight(layer_index, "linear_attn.norm.weight"), weight(layer_index, "linear_attn.A_log").float(), weight(layer_index, "linear_attn.dt_bias"))
        for rows in (1, 2):
            generator = torch.Generator(device="cuda").manual_seed(20261105 + layer_index + rows)
            hidden = torch.randn((rows, 5120), device="cuda", dtype=torch.bfloat16, generator=generator)
            residual = torch.randn((rows, 5120), device="cuda", dtype=torch.bfloat16, generator=generator)
            core = torch.randn((rows, 6144), device="cuda", dtype=torch.bfloat16, generator=generator)
            if kernel1 is not None:
                conv_state = torch.randn((4, 3, 10240), device="cuda", dtype=torch.bfloat16, generator=generator).transpose(1, 2)
                ssm_state = torch.randn((4, 48, 128, 128), device="cuda", dtype=torch.float32, generator=generator) * 0.01
                reference_conv = conv_state.clone()
                reference_ssm = ssm_state.clone()
                indices = torch.tensor([2, 1][:rows], device="cuda", dtype=torch.int32)
                for step in range(5):
                    actual, actual_residual = kernel1(hidden, residual if step else None, conv_state, ssm_state, indices)
                    expected, expected_residual, intermediates = native_k1(kernel1, hidden, residual if step else None, reference_conv, reference_ssm, indices)
                    check = {"kernel": "K1", "layer": layer_index, "rows": rows, "step": step, "output": numerical_error(actual, expected), "residual": numerical_error(actual_residual, expected_residual), "conv_state": numerical_error(conv_state, reference_conv), "ssm_state": numerical_error(ssm_state, reference_ssm), "intermediates": {name: numerical_error(getattr(kernel1, name)[:rows], tensor) for name, tensor in intermediates.items()}}
                    report["checks"].append(check)
                    print(json.dumps(check), flush=True)
                    hidden.add_(0.03125)
                    indices.copy_(torch.tensor(([-1, 3] if step == 2 else [1, 3])[:rows], device="cuda", dtype=indices.dtype))
                reference_conv.copy_(conv_state)
                reference_ssm.copy_(ssm_state)
                indices = indices.long()
                graph1 = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph1):
                    graph_output1, graph_residual1 = kernel1(hidden, residual, conv_state, ssm_state, indices)
                before = int(kernel1.executions.item())
                for step in range(5):
                    hidden.add_(0.03125)
                    indices.copy_(torch.tensor(([-1, 2] if step == 4 else ([3, 2] if step % 2 else [1, 3]))[:rows], device="cuda", dtype=indices.dtype))
                    graph1.replay()
                    expected, expected_residual, _ = native_k1(kernel1, hidden, residual, reference_conv, reference_ssm, indices)
                    check = {"kernel": "K1", "mode": "graph", "layer": layer_index, "rows": rows, "step": step, "output": numerical_error(graph_output1, expected), "residual": numerical_error(graph_residual1, expected_residual), "conv_state": numerical_error(conv_state, reference_conv), "ssm_state": numerical_error(ssm_state, reference_ssm)}
                    report["checks"].append(check)
                    print(json.dumps(check), flush=True)
                assert int(kernel1.executions.item()) - before == 5
                del graph1
            for mode in ("eager", "graph"):
                graph2 = None
                if mode == "graph":
                    graph2 = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph2):
                        actual, actual_residual = kernel2(core, residual)
                before = int(kernel2.executions.item())
                for step in range(3 if graph2 is not None else 1):
                    core.add_(0.03125)
                    if graph2 is not None:
                        graph2.replay()
                    else:
                        actual, actual_residual = kernel2(core, residual)
                    expected, expected_residual, intermediates = native_k2(kernel2, core, residual)
                    check = {"kernel": "K2", "mode": mode, "layer": layer_index, "rows": rows, "step": step, "output": numerical_error(actual, expected), "residual": numerical_error(actual_residual, expected_residual), "intermediates": {name: numerical_error(getattr(kernel2, name)[:rows], tensor) for name, tensor in intermediates.items()}}
                    report["checks"].append(check)
                    print(json.dumps(check), flush=True)
                assert int(kernel2.executions.item()) - before == (3 if graph2 is not None else 1)
                del graph2
            arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        report["program"] = kernel2.program.manifest
        report["occupancy"] = kernel2.program.information
        del kernel1, kernel2, gate_up
        gc.collect()
        torch.cuda.empty_cache()
    report["passed"] = all(check["output"]["finite"] and check["output"]["normalized_rms_error"] < 0.003 and check["residual"]["finite"] and check["residual"]["normalized_rms_error"] < 0.002 and check.get("ssm_state", {"normalized_rms_error": 0})["normalized_rms_error"] < 0.003 and check.get("conv_state", {"normalized_rms_error": 0})["normalized_rms_error"] < 0.002 for check in report["checks"])
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError(f"Dense Monokernel verification failed: {arguments.output}")


if __name__ == "__main__":
    main()
