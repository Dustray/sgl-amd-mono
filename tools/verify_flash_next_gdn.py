import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from sgl_amd_mono.qwen38_flash_next.gfx936.gdn import GDNAttention
from sgl_amd_mono.qwen38_flash_next.gfx936.geometry import DRAFT_TOKENS, MAX_REQUESTS
from sgl_amd_mono.qwen38_flash_next.gfx936.program import HCMix
from sgl_amd_mono.qwen38_flash_next.gfx936.runner import error_metrics


def scratch_pool(slots, device, generator):
    """SGLang's deduplicated conv-window scratch plus the per-step ssm snapshots.

    `intermediate_conv_window[0]` is an ``as_strided`` view over a physical
    ``[slots, dim, steps + width - 1]`` envelope whose step stride equals its window
    stride, so consecutive draft steps share columns (`_allocate_deduplicated_conv_window`
    in SGLang's memory_pool.py). Both buffers start as noise so an unwritten row is
    distinguishable from a correctly zeroed one.
    """
    physical = torch.randn((slots, 1280, DRAFT_TOKENS + 2), device=device, dtype=torch.bfloat16, generator=generator)
    window = physical.as_strided((slots, DRAFT_TOKENS, 1280, 3), (physical.stride(0), physical.stride(2), physical.stride(1), physical.stride(2)))
    intermediate = torch.randn((slots, DRAFT_TOKENS, 6, 128, 128), device=device, dtype=torch.float32, generator=generator) * 0.01
    return physical, window, intermediate


def verify_reference(kernel, mixer, hidden, conv_state, ssm_state, working, requests):
    """Replay the draft chain as `DRAFT_TOKENS` serial decode steps on a copy of the state.

    One verify row is one decode step, so the already-validated decode reference is the
    reference for verify too: after step j the working conv state of a slot *is* that
    step's window snapshot ([t-2, t-1, t]) and the working ssm state is that step's ssm
    snapshot. Returns the expected outputs, the per-(request, step) snapshots, and the
    expected final working conv state.
    """
    mixed, _ = mixer(hidden)
    conv_reference, ssm_reference = conv_state.clone(), ssm_state.clone()
    outputs = torch.empty((requests * DRAFT_TOKENS, 2560), device=hidden.device, dtype=torch.bfloat16)
    windows, states = {}, {}
    for step in range(DRAFT_TOKENS):
        rows = torch.tensor([request * DRAFT_TOKENS + step for request in range(requests)], device=hidden.device)
        outputs[rows] = kernel.native(mixed[rows].contiguous(), conv_reference, ssm_reference, working)
        for request, slot in enumerate(working.tolist()):
            windows[request, step] = conv_reference[slot].clone()
            states[request, step] = ssm_reference[slot].clone()
    return outputs, windows, states, conv_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master"))
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 46])
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 7])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--verify", action="store_true", help="also check the fused MTP TARGET_VERIFY branch (requests * DRAFT_TOKENS rows, per-step scratch snapshots)")
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
            for requests in (range(1, MAX_REQUESTS + 1) if args.verify else ()):
                rows = requests * DRAFT_TOKENS
                generator = torch.Generator(device=device).manual_seed(9181 + rank + layer * 8 + requests)
                hidden = torch.randn((rows, 10240), device=device, dtype=torch.bfloat16, generator=generator)
                # Neither table is the identity, so a kernel that indexed by request would fail.
                working = torch.tensor([2, 0][:requests], device=device, dtype=torch.int32)
                scratch = torch.tensor([1, 3][:requests], device=device, dtype=torch.int32)
                conv = torch.randn((4, 3, 1280), device=device, dtype=torch.bfloat16, generator=generator).transpose(1, 2)
                ssm = torch.randn((4, 6, 128, 128), device=device, dtype=torch.float32, generator=generator) * 0.01
                physical, window, intermediate = scratch_pool(5, device, generator)
                before = {"conv": conv.clone(), "ssm": ssm.clone(), "physical": physical.clone(), "intermediate": intermediate.clone()}
                output, _ = kernel.verify(hidden, conv, ssm, working, scratch, window, intermediate, requests)
                expected, windows, states, expected_conv = verify_reference(kernel, mixer, hidden, before["conv"], before["ssm"], working, requests)
                result = {"layer": layer, "tp_rank": rank, "requests": requests, "rows": rows, "mode": "verify",
                          "output": error_metrics(output, expected),
                          # The chain advances the working conv window but must leave the working ssm
                          # alone: native verify runs with disable_state_update and the worker's commit
                          # scatters the accepted prefix back out of the snapshots.
                          "conv": error_metrics(conv, expected_conv),
                          "ssm_untouched": error_metrics(ssm, before["ssm"]),
                          "guard_stage": int(mixer.state[2].item())}
                for request in range(requests):
                    row = int(scratch[request])
                    result[f"window.{request}"] = error_metrics(window[row], torch.stack([windows[request, step] for step in range(DRAFT_TOKENS)]))
                    result[f"snapshot.{request}"] = error_metrics(intermediate[row], torch.stack([states[request, step] for step in range(DRAFT_TOKENS)]))
                    # A kernel that skipped the snapshot would leave the starting noise, which no
                    # reference matches -- but assert the write happened so the match cannot be read
                    # as "both sides are untouched".
                    assert not torch.equal(physical[row], before["physical"][row]) and not torch.equal(intermediate[row], before["intermediate"][row]), f"scratch row {row} was never written"
                untouched = [row for row in range(5) if row not in scratch.tolist()]
                result["scratch_rows_untouched"] = error_metrics(physical[untouched], before["physical"][untouched])
                result["snapshot_rows_untouched"] = error_metrics(intermediate[untouched], before["intermediate"][untouched])
                checks.append(result)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({"checks": checks, "passed": False}, indent=2) + "\n")
                exact = ("ssm_untouched", "scratch_rows_untouched", "snapshot_rows_untouched")
                assert result["guard_stage"] == 0, result
                assert all(result[name]["finite"] and result[name]["max_absolute_error"] == 0 for name in exact), result
                assert all(value["finite"] and value["nrmse"] < 0.006 for name, value in result.items() if isinstance(value, dict) and name not in exact), result
                print(json.dumps({"layer": layer, "tp_rank": rank, "requests": requests, "rows": rows, "mode": "verify", "passed": True}), flush=True)
    scope = "Real BF16 TP8 shards, fused HC/GDN/output projection, native recurrent reference, eager and Graph replay"
    if args.verify:
        scope += "; fused MTP verify (eager only, graph capture is unsupported by contract.py) against DRAFT_TOKENS serial decode steps, checking outputs, both scratch snapshots, the advanced working conv window, and that the working ssm and unused scratch rows are untouched"
    args.output.write_text(json.dumps({"scope": scope, "checks": checks, "passed": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
