import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors import safe_open

from sgl_amd_mono.qwen38_27b.gfx936.attention import DenseAttentionK1, cache_views
from sgl_amd_mono.qwen38_27b.gfx936.attention_check import ATTENTION_LIMITS, check_fp32_reference, fp32_attention, native_attention, precision_attention
from sgl_amd_mono.qwen38_27b.gfx936.check import check_error
from sgl_amd_mono.qwen38_27b.gfx936.ffn import numerical_error
from sgl_amd_mono.qwen38_27b.report import implementation_digest, write_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 31, 63])
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 63, 64, 65, 257, 4096])
    parser.add_argument("--layouts", nargs="+", default=["legacy", "nhd", "hnd"])
    parser.add_argument("--dtypes", nargs="+", default=["fp8", "bf16"])
    parser.add_argument("--graphs", action="store_true")
    arguments = parser.parse_args()
    mapping = json.loads((arguments.model_path / "model.safetensors.index.json").read_text())["weight_map"]
    report = {"scope": "real full-attention weights; complete K1 versus installed QK/RoPE/FA3/gate and independent FP32 attention; paged KV writes and graph replay", "limits": ATTENTION_LIMITS, "implementation_sha256": implementation_digest(), "passed": False, "checks": []}

    def save():
        write_report(arguments.output.parent, report, arguments.output.name)

    def weight(layer, suffix):
        name = f"model.language_model.layers.{layer}.{suffix}"
        with safe_open(arguments.model_path / mapping[name], framework="pt", device="cpu") as checkpoint:
            return checkpoint.get_tensor(name).cuda()

    positions_cache = torch.arange(8192, device="cuda", dtype=torch.float32)
    frequencies = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64))
    angles = positions_cache[:, None] * frequencies[None, :]
    rotary_cache = torch.cat((angles.cos(), angles.sin()), dim=1).to(torch.bfloat16)
    for layer in arguments.layers:
        kernel = DenseAttentionK1(weight(layer, "input_layernorm.weight"), torch.cat([weight(layer, f"self_attn.{name}_proj.weight") for name in ("q", "k", "v")]), weight(layer, "self_attn.q_norm.weight"), weight(layer, "self_attn.k_norm.weight"))
        for dtype_name in arguments.dtypes:
            dtype = torch.float8_e5m2 if dtype_name == "fp8" else torch.bfloat16
            for layout in arguments.layouts:
                for rows in (1, 2):
                    for length in arguments.lengths:
                        generator = torch.Generator(device="cuda").manual_seed(20261105 + layer + rows + length)
                        hidden = torch.randn((rows, 5120), device="cuda", dtype=torch.bfloat16, generator=generator)
                        residual = torch.randn(hidden.shape, device="cuda", dtype=hidden.dtype, generator=generator) if length != 1 else None
                        lengths = torch.tensor([length, max(1, length - 17)][:rows], device="cuda", dtype=torch.int32)
                        canonical_key = torch.randn((132, 64, 4, 256), device="cuda", dtype=torch.bfloat16, generator=generator).to(dtype)
                        canonical_value = torch.randn((132, 64, 4, 256), device="cuda", dtype=torch.bfloat16, generator=generator).to(dtype)
                        if layout == "legacy":
                            raw_key = canonical_key.permute(0, 2, 1, 3).contiguous()
                            raw_value = canonical_value.permute(0, 2, 3, 1).contiguous()
                        elif layout == "hnd":
                            raw_key = canonical_key.permute(0, 2, 1, 3).contiguous()
                            raw_value = canonical_value.permute(0, 2, 1, 3).contiguous()
                        else:
                            raw_key = canonical_key.flatten(0, 1)
                            raw_value = canonical_value.flatten(0, 1)
                        actual_key, actual_value = cache_views(raw_key, raw_value, 64, layout)
                        reference_key, reference_value = canonical_key.clone(), canonical_value.clone()
                        page_table = torch.randperm(128, device="cuda", generator=generator).view(2, 64)[:rows].int() + 1
                        locations = page_table[torch.arange(rows, device="cuda"), (lengths.long() - 1) // 64].long() * 64 + (lengths.long() - 1) % 64
                        positions = (lengths - 1).long()
                        if length % 2:
                            positions = torch.stack((positions, positions + 3, positions + 7))
                        expected, expected_residual, stages = native_attention(kernel, hidden, residual, positions, rotary_cache, reference_key, reference_value, locations, page_table, lengths)
                        fp32_reference = fp32_attention(stages, reference_key, reference_value, page_table, lengths)
                        precision_reference = precision_attention(stages, reference_key, reference_value, page_table, lengths)
                        actual, actual_residual = kernel(hidden, residual, positions, rotary_cache, actual_key, actual_value, locations, page_table, lengths)
                        torch.cuda.synchronize()
                        check = {"layer": layer, "rows": rows, "length": length, "layout": layout, "dtype": dtype_name, "positions_axes": positions.ndim, "synchronization_failure": int(kernel.synchronization[2].item()), "output": numerical_error(actual, expected), "residual": numerical_error(actual_residual, expected_residual), "stages": {name: numerical_error(getattr(kernel, name)[:rows], value) for name, value in stages.items()}, "key_cache": numerical_error(actual_key.float(), reference_key.float()), "value_cache": numerical_error(actual_value.float(), reference_value.float())}
                        selected_pages, selected_offsets = locations // 64, locations % 64
                        check["key_write"] = numerical_error(actual_key[selected_pages, selected_offsets].float(), reference_key[selected_pages, selected_offsets].float())
                        check["value_write"] = numerical_error(actual_value[selected_pages, selected_offsets].float(), reference_value[selected_pages, selected_offsets].float())
                        check["fp32_output"] = numerical_error(actual, fp32_reference)
                        check["precision_output"] = numerical_error(actual, precision_reference)
                        check["fa3_vs_fp32"] = numerical_error(expected, fp32_reference)
                        report["checks"].append(check)
                        save()
                        print(json.dumps({name: check[name] for name in ("layer", "rows", "length", "layout", "dtype", "synchronization_failure", "output")}), flush=True)
                        if check["output"]["normalized_rms_error"] > 0.003:
                            torch.save({"weights": [tensor.cpu() for tensor in kernel.weights], "hidden": hidden.cpu(), "residual": residual.cpu() if residual is not None else None, "positions": positions.cpu(), "rotary_cache": rotary_cache.cpu(), "stages": {name: tensor.cpu() for name, tensor in stages.items()}, "actual": {name: getattr(kernel, name)[:rows].cpu() for name in stages}, "actual_scores": kernel.scores[:rows].cpu(), "actual_output": actual.cpu(), "page_table": page_table.cpu(), "lengths": lengths.cpu(), "key_cache": reference_key.cpu(), "value_cache": reference_value.cpu()}, arguments.output.with_suffix(".pt"))
                        if check["synchronization_failure"]:
                            raise RuntimeError(f"Attention synchronization failed: {check}")
                        check_error(actual, expected, ATTENTION_LIMITS["native_fa3_nrmse_max"], "attention K1 FA3 output")
                        check_fp32_reference(actual, expected, fp32_reference)
                        check_error(actual, precision_reference, ATTENTION_LIMITS["precision_reference_nrmse_max"], "attention K1 native precision output")
                        check_error(actual_residual, expected_residual, 0.0, "attention residual")
                        for name in ("normalized", "projected", "query", "key", "value", "gate"):
                            check_error(getattr(kernel, name)[:rows], stages[name], 0.003, f"attention {name}")
                        check_error(actual_key.float(), reference_key.float(), 0.0001, "attention key cache")
                        check_error(actual_value.float(), reference_value.float(), 0.0001, "attention value cache")
                        check_error(actual_key[selected_pages, selected_offsets].float(), reference_key[selected_pages, selected_offsets].float(), 0.003, "attention key write")
                        check_error(actual_value[selected_pages, selected_offsets].float(), reference_value[selected_pages, selected_offsets].float(), 0.003, "attention value write")
                        if arguments.graphs:
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                graph_output, graph_residual = kernel(hidden, residual, positions, rotary_cache, actual_key, actual_value, locations, page_table, lengths)
                            graph.replay()
                            torch.cuda.synchronize()
                            check["graph_output"] = check_error(graph_output, actual, 0.0, "attention graph output")
                            check["graph_residual"] = check_error(graph_residual, actual_residual, 0.0, "attention graph residual")
                            del graph, graph_output, graph_residual
                        save()
                        del actual_key, actual_value, reference_key, reference_value, canonical_key, canonical_value, raw_key, raw_value
        report["codegen"] = kernel.program.manifest
        del kernel
        gc.collect()
        torch.cuda.empty_cache()
    report["passed"] = True
    save()


if __name__ == "__main__":
    main()
