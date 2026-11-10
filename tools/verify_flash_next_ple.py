import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as distributed
from torch.nn import functional
from safetensors import safe_open

from sgl_amd_mono.qwen38_flash_next.gfx936.ple import PLE
from sgl_amd_mono.qwen38_flash_next.gfx936.program import PeerGroup
from sgl_amd_mono.qwen38_flash_next.gfx936.runner import error_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master"))
    parser.add_argument("--report-dir", type=Path, default=Path("monokernel_reports/flash_next_tp8/ple"))
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    distributed.init_process_group("gloo")
    peers = PeerGroup(device, rank, distributed.group.WORLD)
    mapping = json.loads((args.model_path / "model.safetensors.index.json").read_text())["weight_map"]
    args.report_dir.mkdir(parents=True, exist_ok=True)
    destination = args.report_dir / f"rank-{rank}.json"
    destination.write_text(json.dumps({"passed": False, "checks": []}) + "\n")

    def weight(suffix):
        name = f"model.language_model.layers.1.ple.{suffix}"
        with safe_open(args.model_path / mapping[name], framework="pt", device="cpu") as checkpoint:
            return checkpoint.get_tensor(name).to(device)

    parameter = lambda value: SimpleNamespace(weight=value)
    generator = torch.Generator(device=device).manual_seed(3836)
    full_embedding = torch.randn((2048, 160), dtype=torch.bfloat16, device=device, generator=generator) * 0.03
    embedding = SimpleNamespace(layer_multipliers=weight("ple_embedding.layer_multipliers"), ngram_heads_vocab_sizes=torch.full((16,), 128, dtype=torch.int64, device=device), ngram_heads_offsets=torch.arange(16, dtype=torch.int64, device=device) * 128, ngram_embedding=SimpleNamespace(weight=full_embedding[rank * 256:(rank + 1) * 256].clone(), shard_indices=SimpleNamespace(org_vocab_start_index=rank * 256, org_vocab_end_index=(rank + 1) * 256)), eos_token_id=248044, ngram_size=3, heads_per_ngram=8, gather_dp_tokens=False)
    layer = SimpleNamespace(ple_embedding=embedding, key_proj=parameter(weight("key_proj.weight")), value_proj=parameter(weight("value_proj.weight")), norm_key=parameter(weight("norm_key.weight")), norm_query=parameter(weight("norm_query.weight")), norm_conv=parameter(weight("norm_conv.weight")), conv1d=parameter(weight("conv1d.weight")), short_conv_dilation=3, short_conv_state_len=9)
    kernel = PLE(layer, peers)

    def normalize(value, norm_weight):
        shaped = value.float().reshape(value.shape[0], 4, 2560)
        return (shaped * torch.rsqrt(shaped.square().mean(-1, keepdim=True) + 1e-6) * (1 + norm_weight.float().reshape(4, 2560))).bfloat16()

    def reference(hidden, contexts, indices, conv):
        token0, token1, token2 = contexts.unbind(-1)
        previous2 = torch.where((token0 == 248044) | (token1 == 248044), 248044, token0)
        mixed2 = (token2 * embedding.layer_multipliers[0]) ^ (token1 * embedding.layer_multipliers[1])
        mixed3 = mixed2 ^ (previous2 * embedding.layer_multipliers[2])
        lookup = torch.cat((mixed2[:, None].expand(-1, 8), mixed3[:, None].expand(-1, 8)), dim=-1) % embedding.ngram_heads_vocab_sizes + embedding.ngram_heads_offsets
        projected_embedding = full_embedding[lookup].reshape(hidden.shape[0], 2560)
        key = functional.linear(projected_embedding, layer.key_proj.weight)
        value = functional.linear(projected_embedding, layer.value_proj.weight)
        normalized_key, normalized_query = normalize(key, layer.norm_key.weight), normalize(hidden, layer.norm_query.weight)
        gate = (normalized_key * normalized_query).sum(-1, keepdim=True) / 2560 ** 0.5
        gate = torch.sigmoid(gate.abs().clamp_min(1e-6).sqrt() * gate.sign())
        gated = gate * value.unsqueeze(-2)
        normalized = normalize(gated.flatten(1), layer.norm_conv.weight).flatten(1)
        conv_input = torch.cat((conv[indices], normalized.unsqueeze(-1)), dim=-1)
        convolved = functional.conv1d(conv_input, layer.conv1d.weight, dilation=3, groups=10240).squeeze(-1)
        conv[indices] = conv_input[:, :, 1:]
        return hidden + (gated.flatten(1) + functional.silu(convolved)), projected_embedding

    checks = []
    for rows in (1, 2):
        hidden = torch.randn((rows, 10240), dtype=torch.bfloat16, device=device, generator=generator)
        contexts = torch.tensor([[123, 456, 789], [248044, 12, 34]][:rows], dtype=torch.int64, device=device)
        indices = torch.tensor([2, 1][:rows], dtype=torch.int64, device=device)
        conv = torch.randn((4, 10240, 9), dtype=torch.bfloat16, device=device, generator=generator) * 0.01
        reference_conv = conv.clone()
        for mode in ("eager", "graph"):
            graph = None
            if mode == "graph":
                real_indices = indices.clone()
                initial_conv = conv.clone()
                indices.zero_()
                dummy = kernel(hidden, contexts, indices, conv)
                assert torch.equal(dummy, hidden) and torch.equal(conv, initial_conv)
                peers.check()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = kernel(hidden, contexts, indices, conv)
                indices.copy_(real_indices)
            for step in range(6):
                if step == 3:
                    contexts[:, 1] = 248044
                contexts[:, 2].add_(1)
                hidden.add_(0.015625)
                if graph is None:
                    output = kernel(hidden, contexts, indices, conv)
                else:
                    graph.replay()
                expected, expected_embedding = reference(hidden, contexts, indices, reference_conv)
                result = {"rows": rows, "mode": mode, "step": step, "output": error_metrics(output, expected), "embedding": error_metrics(kernel.embedding[:rows], expected_embedding), "conv": error_metrics(conv, reference_conv), "state": peers.check()}
                checks.append(result)
                destination.write_text(json.dumps({"passed": False, "checks": checks}, indent=2) + "\n")
                assert all(result[name]["finite"] and result[name]["nrmse"] < 0.008 for name in ("output", "embedding", "conv")), result
    destination.write_text(json.dumps({"passed": True, "scope": "Real PLE projection/norm/convolution/multiplier weights; compact synthetic sharded embedding table; TP8 eager/Graph and EOS boundaries", "checks": checks, "program": kernel.program.manifest}, indent=2) + "\n")
    print(json.dumps({"rank": rank, "passed": True, "checks": len(checks)}), flush=True)
    peers.close(distributed.group.WORLD)
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
