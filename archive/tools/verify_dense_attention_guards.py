import argparse
import json
from pathlib import Path

import torch

from sgl_amd_mono.qwen38_27b.gfx936.attention import DenseAttentionK1
from sgl_amd_mono.qwen38_27b.report import implementation_digest, write_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    weights = [tensor.cuda() for tensor in torch.load(arguments.weights_capture, weights_only=True)["weights"]]
    report = {"scope": "device metadata rejection before KV writes; persistent failure latch; graph padding; fresh workspace recovery", "implementation_sha256": implementation_digest(), "passed": False, "checks": []}
    hidden = torch.randn((2, 5120), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(hidden)
    angles = torch.arange(8192, device="cuda", dtype=torch.float32)[:, None] / (10000000.0 ** (torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64))
    cache = torch.cat((angles.cos(), angles.sin()), dim=1).bfloat16()
    for mode in ("eager", "graph"):
        for failure in ("negative_length", "long_length", "page_oob", "slot_mismatch", "position_oob", "duplicate_slot", "request_negative", "request_padding_length", "request_zero_slot"):
            kernel = DenseAttentionK1(*weights)
            key = torch.randn((4, 64, 4, 256), device="cuda", dtype=torch.bfloat16).to(torch.float8_e5m2)
            value = torch.randn_like(key.bfloat16()).to(torch.float8_e5m2)
            before_key, before_value = key.clone(), value.clone()
            lengths = torch.tensor([2, 3], device="cuda", dtype=torch.int32)
            table = torch.tensor([[1] * 64, [2] * 64], device="cuda", dtype=torch.int32)
            locations = torch.tensor([65, 130], device="cuda", dtype=torch.int64)
            positions = torch.tensor([1, 2], device="cuda", dtype=torch.int64)
            requests = torch.tensor([1, 2], device="cuda", dtype=torch.int64)
            if failure == "negative_length":
                lengths[0] = -1
            elif failure == "long_length":
                lengths[0] = 4097
            elif failure == "page_oob":
                table[0, 0] = 4
                locations[0] = 257
            elif failure == "slot_mismatch":
                locations[0] = 66
            elif failure == "position_oob":
                positions[0] = 8192
            elif failure == "duplicate_slot":
                table[1] = table[0]
                locations[1] = locations[0]
                lengths[1] = lengths[0]
            elif failure == "request_negative":
                requests[0] = -1
            elif failure == "request_padding_length":
                requests[0] = 0
                locations[0] = 0
            else:
                lengths[0] = 1
                locations[0] = 0
                table[0].zero_()
            if mode == "graph":
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    kernel(hidden, residual, positions, cache, key, value, locations, table, lengths, request_indices=requests)
                graph.replay()
            else:
                kernel(hidden, residual, positions, cache, key, value, locations, table, lengths, request_indices=requests)
            torch.cuda.synchronize()
            stage = int(kernel.synchronization[2].item())
            assert stage == (102 if failure == "duplicate_slot" else 101)
            lengths.copy_(torch.tensor([2, 3], device="cuda", dtype=torch.int32))
            locations.copy_(torch.tensor([65, 130], device="cuda", dtype=torch.int64))
            positions.copy_(torch.tensor([1, 2], device="cuda", dtype=torch.int64))
            table[0].fill_(1)
            table[1].fill_(2)
            requests.copy_(torch.tensor([1, 2], device="cuda", dtype=requests.dtype))
            kernel(hidden, residual, positions, cache, key, value, locations, table, lengths, request_indices=requests)
            torch.cuda.synchronize()
            assert int(kernel.synchronization[2].item()) == stage and int(kernel.executions.item()) == 0
            unchanged = torch.equal(key.view(torch.uint8), before_key.view(torch.uint8)) and torch.equal(value.view(torch.uint8), before_value.view(torch.uint8))
            assert unchanged
            fresh = DenseAttentionK1(*weights)
            output, _ = fresh(hidden, residual, positions, cache, key, value, locations, table, lengths, request_indices=requests)
            torch.cuda.synchronize()
            assert int(fresh.synchronization[2].item()) == 0 and bool(torch.isfinite(output).all())
            report["checks"].append({"mode": mode, "failure": failure, "stage": stage, "kv_unchanged": unchanged, "failure_latched": True, "fresh_workspace_recovered": True})
            write_report(arguments.output.parent, report, arguments.output.name)
            if mode == "graph":
                del graph
    for padded_rows in (1, 2):
        kernel = DenseAttentionK1(*weights)
        lengths = torch.tensor([2, 0] if padded_rows == 1 else [0, 0], device="cuda", dtype=torch.int32)
        locations = torch.tensor([65, 0] if padded_rows == 1 else [0, 0], device="cuda")
        positions = torch.zeros(2, device="cuda", dtype=torch.int64)
        before_key, before_value = key.clone(), value.clone()
        output, _ = kernel(hidden, residual, positions, cache, key, value, locations, table, lengths)
        torch.cuda.synchronize()
        assert int(kernel.synchronization[2].item()) == 0 and bool((output[lengths == 0] == 0).all())
        assert torch.equal(key[0].view(torch.uint8), before_key[0].view(torch.uint8)) and torch.equal(value[0].view(torch.uint8), before_value[0].view(torch.uint8))
        report["checks"].append({"padding_rows": padded_rows, "inactive_outputs_zero": True, "dummy_kv_unchanged": True})
    for index_dtype in (torch.int32, torch.int64):
        kernel = DenseAttentionK1(*weights)
        requests = torch.zeros(2, device="cuda", dtype=index_dtype)
        lengths = torch.ones(2, device="cuda", dtype=torch.int32)
        locations = torch.zeros(2, device="cuda", dtype=torch.int64)
        positions = torch.zeros_like(locations)
        before_key, before_value = key.clone(), value.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output, graph_residual = kernel(hidden, residual, positions, cache, key, value, locations, table, lengths, request_indices=requests)
        graph.replay()
        torch.cuda.synchronize()
        assert int(kernel.synchronization[2].item()) == 0 and bool((graph_output == 0).all())
        assert torch.equal(key.view(torch.uint8), before_key.view(torch.uint8)) and torch.equal(value.view(torch.uint8), before_value.view(torch.uint8))
        for active_rows in (2, 1):
            requests.copy_(torch.tensor([1, 2] if active_rows == 2 else [1, 0], device="cuda", dtype=index_dtype))
            lengths.copy_(torch.tensor([2, 3] if active_rows == 2 else [2, 1], device="cuda", dtype=torch.int32))
            locations.copy_(torch.tensor([65, 130] if active_rows == 2 else [65, 0], device="cuda", dtype=torch.int64))
            reference_key, reference_value = key.clone(), value.clone()
            fresh = DenseAttentionK1(*weights)
            expected, expected_residual = fresh(hidden, residual, positions, cache, reference_key, reference_value, locations, table, lengths, request_indices=requests)
            graph.replay()
            torch.cuda.synchronize()
            assert int(kernel.synchronization[2].item()) == 0 and torch.equal(graph_output, expected) and torch.equal(graph_residual, expected_residual)
            assert torch.equal(key.view(torch.uint8), reference_key.view(torch.uint8)) and torch.equal(value.view(torch.uint8), reference_value.view(torch.uint8))
        report["checks"].append({"graph_request_index_dtype": str(index_dtype), "dummy_length_one_accepted": True, "dummy_kv_unchanged": True, "replay_active_rows": [2, 1], "outputs_match_eager": True})
        del graph
    report["passed"] = True
    write_report(arguments.output.parent, report, arguments.output.name)
    print(json.dumps({"passed": True, "checks": len(report["checks"])}))


if __name__ == "__main__":
    main()
