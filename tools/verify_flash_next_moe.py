import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as distributed
from safetensors import safe_open

from sgl_amd_mono.qwen38_flash_next.gfx936.program import HCMix, PeerGroup, PostAttention
from sgl_amd_mono.qwen38_flash_next.gfx936.runner import error_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--report-dir", type=Path, default=Path("monokernel_reports/flash_next_tp8/moe"))
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    distributed.init_process_group("gloo")
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path=str(args.model_path), tp_size=8, moe_runner_backend="triton", dtype="bfloat16"))
    from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl
    from sgl_kernel import silu_and_mul
    from lightop import fuse_silu_and_mul

    mapping = json.loads((args.model_path / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(suffix, selection=None):
        name = f"model.language_model.layers.{args.layer}.{suffix}"
        with safe_open(args.model_path / mapping[name], framework="pt", device="cpu") as checkpoint:
            value = checkpoint.get_tensor(name) if selection is None else checkpoint.get_slice(name)[selection]
            return value.contiguous().to(device)

    def parameter(value):
        return SimpleNamespace(weight=value)

    def connection(prefix):
        return SimpleNamespace(**{name: parameter(weight(f"{prefix}.{name}.weight")) for name in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")})

    gate_up = torch.cat([weight("mlp.experts.gate_up_proj", (slice(None), slice(offset + rank * 80, offset + (rank + 1) * 80), slice(None))) for offset in (0, 640)], dim=1)
    down = weight("mlp.experts.down_proj", (slice(None), slice(None), slice(rank * 80, (rank + 1) * 80)))
    shared_up = torch.cat([weight(f"mlp.shared_expert.{name}_proj.weight", (slice(rank * 80, (rank + 1) * 80), slice(None))) for name in ("gate", "up")])
    shared_down = weight("mlp.shared_expert.down_proj.weight", (slice(None), slice(rank * 80, (rank + 1) * 80)))
    mlp = SimpleNamespace(shared_expert=SimpleNamespace(gate_up_proj=parameter(shared_up), down_proj=parameter(shared_down)), num_fused_shared_experts=0, gate=parameter(weight("mlp.gate.weight")), shared_expert_gate=parameter(weight("mlp.shared_expert_gate.weight")), experts=SimpleNamespace(w13_weight=gate_up, w2_weight=down))
    layer = SimpleNamespace(attn_hyper_connection=connection("attn_hyper_connection"), mlp_hyper_connection=connection("mlp_hyper_connection"), mlp=mlp)
    peers = PeerGroup(device, rank, distributed.group.WORLD)
    mixer = HCMix(layer.attn_hyper_connection.hc_norm.weight, layer.attn_hyper_connection.input_mix_weight_down.weight, layer.attn_hyper_connection.input_mix_weight_up.weight)
    post = PostAttention(layer, peers)
    generator = torch.Generator(device=device).manual_seed(38027)
    checks = []
    for rows in (1, 2):
        for iteration in range(args.iterations):
            hidden = torch.randn((rows, 10240), device=device, dtype=torch.bfloat16, generator=generator)
            attention = torch.randn((rows, 2560), device=device, dtype=torch.bfloat16, generator=generator) * 0.01
            _, residuals = mixer(hidden)
            distributed.barrier()
            post(attention, residuals)
            mixed = post.mixed[:rows]
            native_route = fused_experts_impl(mixed.clone(), gate_up, down, post.top_weights[:rows], post.top_ids[:rows], inplace=True, filter_expert=False)
            projected = torch.nn.functional.linear(mixed, shared_up)
            activated = torch.empty((rows, 80), device=device, dtype=torch.bfloat16)
            fuse_silu_and_mul(projected, activated)
            shared = torch.nn.functional.linear(activated, shared_down)
            fused_gate_sigmoid_mul_add(mixed, mlp.shared_expert_gate.weight.squeeze(), shared, native_route)
            expected_activation = torch.empty((rows * 11, 80), device=device, dtype=torch.bfloat16)
            silu_and_mul(post.projected[:rows].reshape(rows * 11, 160), expected_activation)
            entries = {"activation": error_metrics(post.activated[:rows].reshape(rows * 11, 80), expected_activation), "shared_projection": error_metrics(post.expert_output[:rows, 10], shared), "local_moe": error_metrics(post.local_moe[:rows], native_route)}
            parts = [None] * 8
            distributed.all_gather_object(parts, native_route.detach().cpu())
            expected = torch.zeros((rows, 2560), dtype=torch.float32)
            for part in parts:
                expected.add_(part.float())
            expected = (expected.view(torch.int32) & -65536).view(torch.float32).bfloat16().to(device)
            entries["reduced_moe"] = error_metrics(post.reduced_moe[:rows], expected)
            peers.check()
            result = {"rows": rows, "iteration": iteration, "checks": entries}
            checks.append(result)
            args.report_dir.mkdir(parents=True, exist_ok=True)
            report = {"rank": rank, "layer": args.layer, "checks": checks, "passed": False, "program": peers.program.manifest}
            (args.report_dir / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
            assert all(check["finite"] and check["nrmse"] < 0.002 for check in entries.values()), result
    report["passed"] = True
    (args.report_dir / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rank": rank, "passed": True, "checks": len(checks)}), flush=True)
    peers.close(distributed.group.WORLD)
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
