import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from sgl_amd_mono.qwen38_flash_next.gfx936.gdn import GDNAttention
from sgl_amd_mono.qwen38_flash_next.gfx936.program import HCMix
from sgl_amd_mono.qwen38_flash_next.gfx936.runner import error_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master"))
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 46])
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 7])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("monokernel_reports/flash_next_tp8/gdn.json"))
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    mapping = json.loads((args.model_path / "model.safetensors.index.json").read_text())["weight_map"]
    checks = []

    def weight(layer, suffix, selection=None):
        name = f"model.language_model.layers.{layer}.{suffix}"
        with safe_open(args.model_path / mapping[name], framework="pt", device="cpu") as checkpoint:
            value = checkpoint.get_tensor(name)
            value = value if selection is None else selection(value)
            return value.contiguous().to(device)

    def split_qkv(value, rank):
        query, key, val = value.split((2048, 2048, 6144), dim=0)
        return torch.cat((query[rank * 256:(rank + 1) * 256], key[rank * 256:(rank + 1) * 256], val[rank * 768:(rank + 1) * 768]))

    for layer in args.layers:
        mixer = HCMix(weight(layer, "attn_hyper_connection.hc_norm.weight"), weight(layer, "attn_hyper_connection.input_mix_weight_down.weight"), weight(layer, "attn_hyper_connection.input_mix_weight_up.weight"))
        for rank in args.ranks:
            projected_qkv = weight(layer, "linear_attn.in_proj_qkv.weight", lambda value: split_qkv(value, rank))
            projected_z = weight(layer, "linear_attn.in_proj_z.weight", lambda value: value[rank * 768:(rank + 1) * 768])
            projected_b = weight(layer, "linear_attn.in_proj_b.weight", lambda value: value[rank * 6:(rank + 1) * 6])
            projected_a = weight(layer, "linear_attn.in_proj_a.weight", lambda value: value[rank * 6:(rank + 1) * 6])
            attention = SimpleNamespace(in_proj_qkvz=SimpleNamespace(weight=torch.cat((projected_qkv, projected_z))), in_proj_ba=SimpleNamespace(weight=torch.cat((projected_b, projected_a))), conv1d=SimpleNamespace(weight=weight(layer, "linear_attn.conv1d.weight", lambda value: split_qkv(value, rank))), norm=SimpleNamespace(weight=weight(layer, "linear_attn.norm.weight")), A_log=weight(layer, "linear_attn.A_log", lambda value: value[rank * 6:(rank + 1) * 6]).float(), dt_bias=weight(layer, "linear_attn.dt_bias", lambda value: value[rank * 6:(rank + 1) * 6]), out_proj=SimpleNamespace(weight=weight(layer, "linear_attn.out_proj.weight", lambda value: value[:, rank * 768:(rank + 1) * 768])))
            kernel = GDNAttention(attention, mixer)
            for rows in (1, 2):
                generator = torch.Generator(device=device).manual_seed(3836 + rank + layer * 8 + rows)
                hidden = torch.randn((rows, 10240), device=device, dtype=torch.bfloat16, generator=generator)
                indices = torch.tensor([2, 1][:rows], device=device, dtype=torch.int32)
                conv = torch.randn((4, 3, 1280), device=device, dtype=torch.bfloat16, generator=generator).transpose(1, 2)
                ssm = torch.randn((4, 6, 128, 128), device=device, dtype=torch.float32, generator=generator) * 0.01
                reference_conv, reference_ssm = conv.clone(), ssm.clone()
                for mode in ("eager", "graph"):
                    graph = None
                    if mode == "graph":
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            output, _ = kernel(hidden, conv, ssm, indices)
                    for step in range(5):
                        hidden.add_(0.015625)
                        if graph is None:
                            output, _ = kernel(hidden, conv, ssm, indices)
                        else:
                            graph.replay()
                        mixed, _ = mixer(hidden)
                        expected = kernel.native(mixed, reference_conv, reference_ssm, indices)
                        result = {"layer": layer, "tp_rank": rank, "rows": rows, "mode": mode, "step": step, "output": error_metrics(output, expected), "conv": error_metrics(conv, reference_conv), "ssm": error_metrics(ssm, reference_ssm), "guard_stage": int(mixer.state[2].item())}
                        checks.append(result)
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps({"checks": checks, "passed": False}, indent=2) + "\n")
                        assert result["guard_stage"] == 0 and all(result[name]["finite"] and result[name]["nrmse"] < 0.006 for name in ("output", "conv", "ssm")), result
                print(json.dumps({"layer": layer, "tp_rank": rank, "rows": rows, "passed": True}), flush=True)
    args.output.write_text(json.dumps({"scope": "Real BF16 TP8 shards, fused HC/GDN/output projection, native recurrent reference, eager and Graph replay", "checks": checks, "passed": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
