import argparse
import datetime
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as distributed

from sgl_amd_mono.qwen38_flash_next.gfx936.program import HCMix, PeerGroup


def metrics(actual, expected):
    actual = actual.float()
    expected = expected.float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    difference = actual - expected
    return {"finite": finite, "nrmse": float(difference.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12)), "max_absolute_error": float(difference.abs().max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=Path("monokernel_reports/flash_next_tp8/primitive"))
    parser.add_argument("--iterations", type=int, default=16)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    distributed.init_process_group("gloo")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / f"rank-{rank}.json").write_text(json.dumps({"passed": False, "phase": "started"}) + "\n")
    peers = PeerGroup(device, rank, distributed.group.WORLD)
    generator = torch.Generator(device=device).manual_seed(3900 + rank)
    checks = []
    for rows in (1, 2):
        input_tensor = torch.empty((rows, 2560), device=device, dtype=torch.bfloat16)
        output = torch.empty_like(input_tensor)
        second = torch.empty_like(input_tensor)
        second_output = torch.empty_like(input_tensor)
        input_tensor.fill_(rank)
        second.fill_(-rank)
        torch.cuda.synchronize(device)
        distributed.barrier()
        for iteration in range(args.iterations):
            input_tensor.fill_(rank + iteration)
            peers.reduce(input_tensor, output)
            check = metrics(output, torch.full_like(output, 28 + iteration * 8))
            checks.append({"kind": "ipc_reduce_eager", "rows": rows, "iteration": iteration, **check})
            if check["max_absolute_error"]:
                print(json.dumps({"rank": rank, "iteration": iteration, "input": input_tensor[0, :8].cpu().float().tolist(), "output": output[0, :8].cpu().float().tolist(), "state": peers.snapshot()}), flush=True)
            assert check["finite"] and check["max_absolute_error"] == 0, checks[-1]
        print(json.dumps({"rank": rank, "rows": rows, "phase": "integer_graph_passed", "epoch": peers.check()["collective_epoch"]}), flush=True)
        for iteration in range(args.iterations):
            input_tensor.fill_(rank + iteration)
            second.fill_(-rank - iteration)
            peers.reduce_pair(input_tensor, second, (output, second_output))
            peers.check()
            for name, actual, expected in (("first", output, 28 + iteration * 8), ("second", second_output, -28 - iteration * 8)):
                check = metrics(actual, torch.full_like(actual, expected))
                checks.append({"kind": f"ipc_reduce_pair_{name}", "rows": rows, "iteration": iteration, **check})
                assert check["finite"] and check["max_absolute_error"] == 0, checks[-1]
        input_tensor.fill_(rank)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            peers.reduce(input_tensor, output)
        for iteration in range(args.iterations):
            input_tensor.fill_(rank + iteration)
            graph.replay()
            check = metrics(output, torch.full_like(output, 28 + iteration * 8))
            checks.append({"kind": "ipc_reduce_graph", "rows": rows, "iteration": iteration, **check})
            assert check["finite"] and check["max_absolute_error"] == 0, checks[-1]
        pair_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(pair_graph):
            peers.reduce_pair(input_tensor, second, (output, second_output))
        for mode in ("eager", "graph"):
            for iteration in range(args.iterations):
                reference_generator = torch.Generator().manual_seed(380000 + rows * 1000 + iteration)
                values = (torch.randn((8, rows, 2560), generator=reference_generator) * torch.pow(2.0, torch.randint(-8, 9, (8, rows, 2560), generator=reference_generator))).bfloat16()
                reference = torch.zeros((rows, 2560), dtype=torch.float32)
                for peer in range(8):
                    reference.add_(values[peer].float())
                reference = (reference.view(torch.int32) & -65536).view(torch.float32).bfloat16().to(device)
                input_tensor.copy_(values[rank])
                second.copy_(-values[rank].flip(-1))
                time.sleep((rank * 7 + iteration) % 5 * 0.0001)
                if mode == "eager":
                    peers.reduce_pair(input_tensor, second, (output, second_output))
                else:
                    pair_graph.replay()
                peers.check()
                for name, actual, expected in (("first", output, reference), ("second", second_output, -reference.flip(-1))):
                    check = metrics(actual, expected)
                    checks.append({"kind": f"random_pair_{mode}_{name}", "rows": rows, "iteration": iteration, **check})
                    assert check["finite"] and check["max_absolute_error"] == 0, checks[-1]
            print(json.dumps({"rank": rank, "rows": rows, "phase": f"random_pair_{mode}_passed", "epoch": peers.check()["collective_epoch"]}), flush=True)
        norm = torch.randn(10240, generator=generator, device=device, dtype=torch.bfloat16) * 0.05
        down = torch.randn(320, 10240, generator=generator, device=device, dtype=torch.bfloat16) * 0.01
        up = torch.randn(10240, 320, generator=generator, device=device, dtype=torch.bfloat16) * 0.03
        hidden = torch.randn(rows, 10240, generator=generator, device=device, dtype=torch.bfloat16)
        mix = HCMix(norm, down, up)
        actual, (_, normalized_actual) = mix(hidden)
        branch = hidden.float().reshape(rows, 4, 2560)
        normalized = (branch * torch.rsqrt(branch.square().mean(-1, keepdim=True) + 1e-6) * (1 + norm.float().reshape(4, 2560))).reshape(rows, 10240).bfloat16()
        lower = torch.nn.functional.linear(normalized.float(), down.float()) / 4
        lower = torch.nn.functional.silu(lower).bfloat16()
        gates = torch.nn.functional.linear(lower.float(), up.float())
        expected = (torch.sigmoid(gates.reshape(rows, 4, 2560)) * normalized.float().reshape(rows, 4, 2560)).mean(1).bfloat16()
        for name, result, reference in (("hc_mix", actual, expected), ("hc_norm", normalized_actual, normalized)):
            check = metrics(result, reference)
            checks.append({"kind": name, "rows": rows, **check})
            assert check["finite"] and check["nrmse"] < 0.004, checks[-1]
    state = peers.check()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = {"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "rank": rank, "device": str(device), "transport": peers.transport, "reduction_rounding": "fp32_rank_order_sum_bf16_rtz", "program": peers.program.information, "checks": checks, "state": state, "passed": True}
    (args.report_dir / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rank": rank, "passed": True, "checks": len(checks), "state": state}), flush=True)
    peers.close(distributed.group.WORLD)
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
