import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from sgl_amd_mono.qwen38_27b.gfx936.check import check_error, native_k1, native_norm
from sgl_amd_mono.qwen38_27b.gfx936.decode_io import DenseEntryK1, DenseTail
from sgl_amd_mono.qwen38_27b.report import implementation_digest, write_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    mapping = json.loads((arguments.model_path / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(name):
        with safe_open(arguments.model_path / mapping[name], framework="pt", device="cpu") as checkpoint:
            return checkpoint.get_tensor(name).cuda()

    def layer_weight(suffix):
        return weight("model.language_model.layers.0." + suffix)

    embedding = weight("model.language_model.embed_tokens.weight")
    tail = DenseTail(weight("model.language_model.norm.weight"), weight("lm_head.weight"))
    entry = DenseEntryK1(layer_weight("input_layernorm.weight"), torch.cat((layer_weight("linear_attn.in_proj_qkv.weight"), layer_weight("linear_attn.in_proj_z.weight"))), torch.cat((layer_weight("linear_attn.in_proj_b.weight"), layer_weight("linear_attn.in_proj_a.weight"))), layer_weight("linear_attn.conv1d.weight"), layer_weight("linear_attn.norm.weight"), layer_weight("linear_attn.A_log").float(), layer_weight("linear_attn.dt_bias"))
    report = {"scope": "real embedding plus first GDN K1 in one launch; real final Gemma norm plus entire LM head in one launch", "implementation_sha256": implementation_digest(), "passed": False, "checks": []}
    for rows in (1, 2):
        generator = torch.Generator(device="cuda").manual_seed(20261105 + rows)
        for step, ids in enumerate(([0, 248319], [271, 198], [1024, 151645])):
            input_ids = torch.tensor(ids[:rows], device="cuda", dtype=torch.int64)
            hidden = torch.empty((rows, 5120), device="cuda", dtype=torch.bfloat16)
            convolution = torch.randn((4, 3, 10240), device="cuda", dtype=torch.bfloat16, generator=generator).transpose(1, 2)
            state = torch.randn((4, 48, 128, 128), device="cuda", generator=generator) * 0.01
            reference_convolution, reference_state = convolution.clone(), state.clone()
            indices = torch.tensor([2, 1][:rows], device="cuda", dtype=torch.int32)
            expected, expected_residual, _ = native_k1(entry, embedding[input_ids], None, reference_convolution, reference_state, indices)
            actual, actual_residual = entry(hidden, None, convolution, state, indices, embedding, input_ids)
            check = {"rows": rows, "step": step, "entry": check_error(actual, expected, 0.003, "embedding and first K1"), "embedding": check_error(hidden, embedding[input_ids], 0.0, "embedding gather"), "residual": check_error(actual_residual, expected_residual, 0.0, "embedding residual"), "state": check_error(state, reference_state, 0.003, "entry state")}
            residual = torch.randn(hidden.shape, device="cuda", dtype=hidden.dtype, generator=generator)
            normalized, _ = native_norm(hidden, residual, tail.weights[0])
            expected_logits = torch.nn.functional.linear(normalized, tail.weights[1])
            logits = tail(hidden, residual)
            if not torch.equal(logits, expected_logits):
                torch.save({"normalized": normalized.cpu(), "expected": expected_logits.cpu(), "actual": logits.cpu(), "implementation_sha256": implementation_digest()}, arguments.output.with_suffix(".failure.pt"))
            check["tail"] = check_error(logits, expected_logits, 0.0, "final norm and LM head")
            check["normalized"] = check_error(tail.normalized[:rows], normalized, 0.0, "final norm")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_core, graph_residual = entry(hidden, None, convolution, state, indices, embedding, input_ids)
                graph_logits = tail(hidden, residual)
            check["graph_replays"] = []
            for replay in range(3):
                changed_ids = [(value + 17 * (replay + 1)) % embedding.shape[0] for value in ids[:rows]]
                input_ids.copy_(torch.tensor(changed_ids, device="cuda", dtype=input_ids.dtype))
                reference_convolution, reference_state = convolution.clone(), state.clone()
                expected_core, expected_entry_residual, _ = native_k1(entry, embedding[input_ids], None, reference_convolution, reference_state, indices)
                normalized, _ = native_norm(embedding[input_ids], residual, tail.weights[0])
                expected_logits = torch.nn.functional.linear(normalized, tail.weights[1])
                graph.replay()
                torch.cuda.synchronize()
                check["graph_tail"] = check_error(graph_logits, expected_logits, 0.0, "tail graph replay")
                check["graph_embedding"] = check_error(hidden, embedding[input_ids], 0.0, "entry graph embedding")
                check["graph_replays"].append({"replay": replay, "input_ids": changed_ids, "core": check_error(graph_core, expected_core, 0.003, "entry graph output"), "residual": check_error(graph_residual, expected_entry_residual, 0.0, "entry graph residual"), "convolution": check_error(convolution, reference_convolution, 0.0, "entry graph convolution"), "state": check_error(state, reference_state, 0.003, "entry graph recurrent state"), "tail": check["graph_tail"]})
            check["synchronization"] = [int(entry.synchronization[2].item()), int(tail.synchronization[2].item())]
            if any(check["synchronization"]):
                raise RuntimeError(f"Decode input/output synchronization failed: {check}")
            report["checks"].append(check)
            write_report(arguments.output.parent, report, arguments.output.name)
            print(json.dumps(check), flush=True)
            del graph
    report["passed"] = True
    report["codegen"] = tail.program.manifest
    write_report(arguments.output.parent, report, arguments.output.name)


if __name__ == "__main__":
    main()
