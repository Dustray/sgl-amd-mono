import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from sgl_amd_mono.qwen38_27b.gfx936.attention import DenseAttentionK1
from sgl_amd_mono.qwen38_27b.gfx936.attention_check import ATTENTION_LIMITS, check_fp32_reference, fp32_attention, native_attention, precision_attention
from sgl_amd_mono.qwen38_27b.gfx936.check import check_error
from sgl_amd_mono.qwen38_27b.gfx936.ffn import numerical_error
from sgl_amd_mono.qwen38_27b.report import implementation_digest, write_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--zero-gate", action="store_true")
    arguments = parser.parse_args()
    captured = torch.load(arguments.capture, map_location="cpu", weights_only=True)
    mapping = json.loads((arguments.model_path / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(suffix):
        name = f"model.language_model.layers.{captured['layer']}.{suffix}"
        with safe_open(arguments.model_path / mapping[name], framework="pt", device="cpu") as checkpoint:
            return checkpoint.get_tensor(name).cuda()

    kernel = DenseAttentionK1(weight("input_layernorm.weight"), torch.cat([weight(f"self_attn.{name}_proj.weight") for name in ("q", "k", "v")]), weight("self_attn.q_norm.weight"), weight("self_attn.k_norm.weight"))
    inputs = {name: captured[name].cuda() if captured[name] is not None else None for name in ("hidden", "residual", "positions", "rotary_cache", "key_cache", "value_cache", "locations", "page_table", "lengths", "request_indices")}
    reference_key, reference_value = inputs["key_cache"].clone(), inputs["value_cache"].clone()
    positional = [inputs[name] for name in ("hidden", "residual", "positions", "rotary_cache")]
    expected, expected_residual, stages = native_attention(kernel, *positional, reference_key, reference_value, inputs["locations"], inputs["page_table"], inputs["lengths"])
    native_replay = numerical_error(expected, captured["expected"].cuda())
    check_error(expected, captured["expected"].cuda(), 0.003, "captured native attention replay")
    if arguments.zero_gate:
        for head in range(24):
            kernel.weights[1][head * 512 + 256:(head + 1) * 512].zero_()
        expected, expected_residual, stages = native_attention(kernel, *positional, reference_key, reference_value, inputs["locations"], inputs["page_table"], inputs["lengths"])
    fp32_reference = fp32_attention(stages, reference_key, reference_value, inputs["page_table"], inputs["lengths"])
    precision_reference = precision_attention(stages, reference_key, reference_value, inputs["page_table"], inputs["lengths"])

    def run():
        return kernel(*positional, inputs["key_cache"], inputs["value_cache"], inputs["locations"], inputs["page_table"], inputs["lengths"], request_indices=inputs["request_indices"])

    actual, actual_residual = run()
    torch.cuda.synchronize()
    report = {"scope": "diagnostic replay of actual serving full-attention inputs and compacted paged KV history", "capture": str(arguments.capture), "capture_implementation_sha256": captured["implementation_sha256"], "implementation_sha256": implementation_digest(), "layer": captured["layer"], "step": captured["step"], "zero_gate": arguments.zero_gate, "passed": False, "native_replay": native_replay, "output": numerical_error(actual, expected), "precision_output": numerical_error(actual, precision_reference), "stages": {name: numerical_error(getattr(kernel, name)[:actual.shape[0]], reference) for name, reference in stages.items()}, "synchronization_failure": int(kernel.synchronization[2].item())}
    report["differing_outputs"] = [{"row": row, "head": column // 256, "dimension": column % 256, "native": float(expected[row, column]), "actual": float(actual[row, column]), "precision_reference": float(precision_reference[row, column])} for row, column in (actual != expected).nonzero().tolist()[:32]]
    report["fp32_output"], report["fa3_fp32_output"] = check_fp32_reference(actual, expected, fp32_reference)
    write_report(arguments.output.parent, report, arguments.output.name)
    check_error(actual, expected, ATTENTION_LIMITS["native_fa3_nrmse_max"], "serving attention replay")
    check_error(actual, precision_reference, ATTENTION_LIMITS["precision_reference_nrmse_max"], "serving attention precision reference")
    check_error(actual_residual, expected_residual, 0.0, "serving attention residual")
    if arguments.graphs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output, graph_residual = run()
        graph.replay()
        torch.cuda.synchronize()
        report["graph_output"] = check_error(graph_output, actual, 0.0, "serving attention graph output")
        check_error(graph_residual, actual_residual, 0.0, "serving attention graph residual")
    if int(kernel.synchronization[2].item()):
        raise RuntimeError("Attention replay left a device failure latch")
    report["passed"] = True
    write_report(arguments.output.parent, report, arguments.output.name)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
