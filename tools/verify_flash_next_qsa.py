import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from sgl_amd_mono.qwen38_flash_next.gfx936.program import HCMix
from sgl_amd_mono.qwen38_flash_next.gfx936.qsa import QSAAttention
from sgl_amd_mono.qwen38_flash_next.gfx936.runner import error_metrics


def exact_selection(scores, length):
    """Token order the kernel must produce from its own index scores: top 512 groups by
    (score descending, index ascending), each expanded to 4 tokens, then the partial group."""
    groups = length // 4
    if groups <= 512:
        tokens = torch.arange(length, device=scores.device)
    else:
        order = sorted(range(groups), key=lambda index, values=scores[:groups].tolist(): (-values[index], index))[:512]
        blocks = torch.tensor(order, device=scores.device)
        tokens = torch.cat(((blocks[:, None] * 4 + torch.arange(4, device=scores.device)).flatten(), torch.arange(groups * 4, length, device=scores.device)))
    selected = torch.full((2051,), -1, dtype=torch.int32, device=scores.device)
    selected[:tokens.numel()] = tokens.int()
    return selected


def boundary_margin(actual_scores, expected_scores, length):
    """How unambiguous the 512-group cutoff is, against the kernel-vs-native score noise.

    Above 512 groups the kept set is decided by index scores both sides compute in BF16, so two
    groups closer together than that noise can legitimately land on either side of the cutoff (the
    known native QSA tie non-determinism). Returns (gap, noise): `gap` is native's distance between
    the last kept and first dropped score, `noise` the largest score difference between the two
    implementations. Comparing the selected *set* with native only means something when gap > noise;
    the kernel's own ranking is checked unconditionally by `exact_selection`.
    """
    groups = length // 4
    if groups <= 512:
        return float("inf"), 0.0
    actual, expected = actual_scores[:groups].float(), expected_scores[:groups].float()
    ordered = expected.sort(descending=True).values
    return (ordered[511] - ordered[512]).item(), (actual - expected).abs().max().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master"))
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 47])
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 7])
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 3, 63, 2047, 4093])
    parser.add_argument("--context", type=int, default=4096, help="kernel context limit; > 4096 exercises the chunked index top-k merge")
    parser.add_argument("--device", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("monokernel_reports/flash_next_tp8/qsa.json"))
    args = parser.parse_args()
    if max(args.lengths) + 3 > args.context:
        parser.error("every length + 3 decode steps must fit --context")
    pages = -(-args.context // 64)
    # One row, then two equal rows, then two rows of different lengths: a short row only reaches the
    # multi-chunk index path when a longer row in the same wave forces more than one sort chunk.
    cases = [(length,) for length in args.lengths] + [(length, length) for length in args.lengths]
    if min(args.lengths) != max(args.lengths):
        cases += [(min(args.lengths), max(args.lengths)), (max(args.lengths), min(args.lengths))]
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    mapping = json.loads((args.model_path / "model.safetensors.index.json").read_text())["weight_map"]
    checks = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def weight(layer, suffix, selection=None):
        name = f"model.language_model.layers.{layer}.{suffix}"
        with safe_open(args.model_path / mapping[name], framework="pt", device="cpu") as checkpoint:
            value = checkpoint.get_tensor(name)
            value = value if selection is None else selection(value)
            return value.contiguous().to(device)

    positions_table = torch.arange(pages * 64, dtype=torch.float32, device=device)
    frequencies = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 64))
    angles = positions_table[:, None] * frequencies[None, :]
    rotary = SimpleNamespace(cos_sin_cache=torch.cat((angles.cos(), angles.sin()), dim=-1))
    for layer_id in args.layers:
        mixer = HCMix(weight(layer_id, "attn_hyper_connection.hc_norm.weight"), weight(layer_id, "attn_hyper_connection.input_mix_weight_down.weight"), weight(layer_id, "attn_hyper_connection.input_mix_weight_up.weight"))
        for rank in args.ranks:
            namespace = lambda value: SimpleNamespace(weight=value)
            query = weight(layer_id, "self_attn.q_proj.weight", lambda value: value[rank * 1536:(rank + 1) * 1536])
            key = weight(layer_id, "self_attn.k_proj.weight", lambda value: value[rank // 4 * 256:(rank // 4 + 1) * 256])
            value = weight(layer_id, "self_attn.v_proj.weight", lambda value: value[rank // 4 * 256:(rank // 4 + 1) * 256])
            layer = SimpleNamespace(qkv_proj=namespace(torch.cat((query, key, value))), q_norm=namespace(weight(layer_id, "self_attn.q_norm.weight")), k_norm=namespace(weight(layer_id, "self_attn.k_norm.weight")), o_proj=namespace(weight(layer_id, "self_attn.o_proj.weight", lambda value: value[:, rank * 768:(rank + 1) * 768])), indexer=SimpleNamespace(index_qk_proj=namespace(weight(layer_id, "self_attn.indexer.index_qk_proj.weight")), q_layernorm=namespace(weight(layer_id, "self_attn.indexer.q_layernorm.weight")), k_layernorm=namespace(weight(layer_id, "self_attn.indexer.k_layernorm.weight"))), rotary_emb=rotary)
            kernel = QSAAttention(layer, mixer, args.context)
            for case in cases:
                rows = len(case)
                generator = torch.Generator(device=device).manual_seed(3836 + layer_id * 8 + rank + rows + sum(case))
                hidden = torch.randn((rows, 10240), device=device, dtype=torch.bfloat16, generator=generator)
                requests = torch.tensor([2, 1][:rows], dtype=torch.int64, device=device)
                table = torch.zeros((4, pages * 64), dtype=torch.int32, device=device)
                for request in (1, 2):
                    request_pages = torch.randperm(pages, device=device, generator=generator) + 1 + (request - 1) * pages
                    table[request] = (request_pages[:, None] * 64 + torch.arange(64, device=device)).flatten().int()
                cache_pages = 2 * pages + 1
                key = torch.randn((cache_pages, 1, 64, 256), dtype=torch.bfloat16, device=device, generator=generator)
                value = torch.randn((cache_pages, 1, 256, 64), dtype=torch.bfloat16, device=device, generator=generator)
                ring = torch.randn((16, 1, 128), dtype=torch.bfloat16, device=device, generator=generator)
                ring_positions = torch.zeros((16, 3), dtype=torch.int64, device=device)
                compressed = torch.randn((cache_pages * 16, 1, 128), dtype=torch.bfloat16, device=device, generator=generator)
                reference_key, reference_value, reference_ring, reference_positions, reference_compressed = (tensor.clone() for tensor in (key, value, ring, ring_positions, compressed))
                lengths = torch.tensor(case, dtype=torch.int32, device=device)
                positions = lengths.long() - 1
                locations = table[requests, positions].long()
                for row in range(rows):
                    request = int(requests[row])
                    for offset in range(4):
                        base_position = ((case[row] - 1) // 4) * 4 + offset
                        ring_positions[request * 4 + offset] = base_position
                reference_positions.copy_(ring_positions)
                graph = None
                for mode in ("eager", "graph"):
                    if mode == "graph":
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            output, _ = kernel(hidden, positions, lengths, requests, locations, table, key, value, ring, ring_positions, compressed)
                    for step in range(2):
                        hidden.add_(0.015625)
                        if graph is None:
                            output, _ = kernel(hidden, positions, lengths, requests, locations, table, key, value, ring, ring_positions, compressed)
                        else:
                            graph.replay()
                        mixed, _ = mixer(hidden)
                        expected, intermediates = kernel.native(mixed, positions, lengths, requests, locations, table, reference_key, reference_value, reference_ring, reference_positions, reference_compressed)
                        result = {"layer": layer_id, "tp_rank": rank, "rows": rows, "length": [int(length) for length in lengths], "mode": mode, "step": step, "output": error_metrics(output, expected), "key": error_metrics(key, reference_key), "value": error_metrics(value, reference_value), "ring": error_metrics(ring, reference_ring), "compressed": error_metrics(compressed, reference_compressed), "guard_stage": int(mixer.state[2].item())}
                        for name in ("queries", "index_queries", "core"):
                            result[name] = error_metrics(getattr(kernel, name)[:rows], intermediates[name])
                        result["selected_set_matches"] = bool(torch.equal(kernel.selected[:rows].sort(-1).values, intermediates["selected"].sort(-1).values))
                        result["selected_order_matches"] = bool(torch.equal(kernel.selected[:rows], intermediates["selected"]))
                        result["ring_positions_match"] = bool(torch.equal(ring_positions, reference_positions))
                        result["selected_exact_topk"] = all(torch.equal(kernel.selected[row], exact_selection(kernel.index_scores[row], int(lengths[row]))) for row in range(rows))
                        margins = [boundary_margin(kernel.index_scores[row], intermediates["index_scores"][row], int(lengths[row])) for row in range(rows)]
                        result["selection_boundary"] = [{"gap": gap, "score_noise": noise} for gap, noise in margins]
                        # Only compare the selected set with native where native's own cutoff is wider than the
                        # BF16 score noise; otherwise the two sets may differ by groups tied to within that noise.
                        result["selection_unambiguous"] = all(gap > 2 * noise for gap, noise in margins)
                        if not result["selection_unambiguous"] and not result["selected_set_matches"]:
                            # Native kept a different, tied set, so its attention covers different tokens. Re-run the
                            # reference on the kernel's own token list: that still checks all the arithmetic, and it is
                            # the only comparison that is meaningful when the cutoff is below the BF16 score noise.
                            expected, intermediates = kernel.native(mixed, positions, lengths, requests, locations, table, reference_key, reference_value, reference_ring, reference_positions, reference_compressed, selection=kernel.selected[:rows])
                            result["reference_on_kernel_selection"] = True
                            result["output"] = error_metrics(output, expected)
                            result["core"] = error_metrics(kernel.core[:rows], intermediates["core"])
                        checks.append(result)
                        args.output.write_text(json.dumps({"checks": checks, "passed": False, "program": kernel.program.manifest}, indent=2) + "\n")
                        if result["guard_stage"] or not result["ring_positions_match"] or not result["selected_exact_topk"] or (result["selection_unambiguous"] and not result["selected_set_matches"]) or (max(int(length) for length in lengths) // 4 <= 512 and not result["selected_order_matches"]) or any(not result[name]["finite"] or result[name]["nrmse"] > 0.008 for name in ("output", "key", "value", "ring", "compressed", "queries", "index_queries", "core")):
                            torch.save({"result": result, "lengths": lengths, "actual_scores": kernel.index_scores[:rows], "expected_scores": intermediates["index_scores"], "actual_selected": kernel.selected[:rows], "expected_selected": intermediates["selected"]}, args.output.with_suffix(".failure.pt"))
                            raise AssertionError(result)
                        if mode != "graph" or step != 1:
                            lengths.add_(1)
                            positions.add_(1)
                            locations.copy_(table[requests, positions].long())
                print(json.dumps({"layer": layer_id, "rank": rank, "rows": rows, "initial_lengths": list(case), "passed": True}), flush=True)
    args.output.write_text(json.dumps({"scope": "Real BF16 TP8 shards; fused HC, QKV/index projection, Q/K norm/RoPE, caches, compression, index selection, sparse attention, gate, output projection; eager and Graph replay", "checks": checks, "passed": True, "program": kernel.program.manifest}, indent=2) + "\n")


if __name__ == "__main__":
    main()
