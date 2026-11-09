import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from sgl_amd_mono.qwen38_flash_next.gfx936.program import HCMix
from sgl_amd_mono.qwen38_flash_next.gfx936.runner import error_metrics


def main():
    from sglang.srt.layers.hc_mix_triton import fused_hc_mix
    from sglang.srt.layers.hyperconnection import GroupedGemmaRMSNorm

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master"))
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 6, 7, 47])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("monokernel_reports/flash_next_tp8/hc.json"))
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    mapping = json.loads((args.model_path / "model.safetensors.index.json").read_text())["weight_map"]
    generator = torch.Generator(device=device).manual_seed(3836)
    report = {"passed": False, "checks": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def weight(name, selection=None):
        with safe_open(args.model_path / mapping[name], framework="pt", device="cpu") as checkpoint:
            value = checkpoint.get_slice(name)
            return (value[:] if selection is None else value[selection]).contiguous().to(device)

    prefixes = [(layer, kind, f"model.language_model.layers.{layer}.{kind}_hyper_connection.") for layer in args.layers for kind in ("attn", "mlp")]
    prefixes.append((48, "tail", "model.language_model.hyper_connection_mixer."))
    for layer, kind, prefix in prefixes:
        norm_weight = weight(prefix + "hc_norm.weight")
        mixer = HCMix(norm_weight, weight(prefix + "input_mix_weight_down.weight"), weight(prefix + "input_mix_weight_up.weight"))
        norm = GroupedGemmaRMSNorm(10240, group_size=2560).to(device=device, dtype=torch.bfloat16)
        norm.weight = torch.nn.Parameter(norm_weight, requires_grad=False)
        for rows in (1, 2):
            for sample in ("random", "embedding"):
                if sample == "random":
                    hidden = torch.randn((rows, 10240), device=device, dtype=torch.bfloat16, generator=generator)
                else:
                    hidden = torch.stack([weight("model.language_model.embed_tokens.weight", token) for token in (20, 42)[:rows]]).repeat(1, 4)
                for mode in ("eager", "graph"):
                    if mode == "graph":
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            actual, _ = mixer(hidden)
                        graph.replay()
                    else:
                        actual, _ = mixer(hidden)
                    normalized = norm(hidden)
                    expected = fused_hc_mix(normalized, mixer.weights[1], mixer.weights[2], 4, 2560)
                    check = {"layer": layer, "kind": kind, "rows": rows, "sample": sample, "mode": mode, "normalized": error_metrics(mixer.normalized[:rows], normalized), "mixed": error_metrics(actual, expected)}
                    report["checks"].append(check)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    assert all(check[name]["finite"] and check[name]["nrmse"] < 0.002 for name in ("normalized", "mixed")), check
                    if layer == 0 and kind == "attn" and sample == "embedding" and rows == 1:
                        assert check["mixed"]["max_absolute_error"] == 0, check
        print(json.dumps({"layer": layer, "kind": kind, "passed": True}), flush=True)
    report.update(passed=True, scope="Real HC weights, native grouped norm and persistent split-K HC reference, embedding and random inputs, eager and Graph")
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
