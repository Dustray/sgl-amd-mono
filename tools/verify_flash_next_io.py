import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as distributed
from safetensors import safe_open

from sgl_amd_mono.qwen38_flash_next.gfx936.decode_io import DecodeIO
from sgl_amd_mono.qwen38_flash_next.gfx936.program import HCMix, PeerGroup
from sgl_amd_mono.qwen38_flash_next.gfx936.runner import error_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master"))
    parser.add_argument("--report-dir", type=Path, default=Path("monokernel_reports/flash_next_tp8/io"))
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

    def weight(name, selection=None):
        with safe_open(args.model_path / mapping[name], framework="pt", device="cpu") as checkpoint:
            sliced = checkpoint.get_slice(name)
            return (sliced[:] if selection is None else sliced[selection]).contiguous().to(device)

    prefix = "model.language_model.hyper_connection_mixer."
    mixer = HCMix(weight(prefix + "hc_norm.weight"), weight(prefix + "input_mix_weight_down.weight"), weight(prefix + "input_mix_weight_up.weight"))
    mixer.state = peers.state
    shard = SimpleNamespace(org_vocab_start_index=rank * 31040, org_vocab_end_index=(rank + 1) * 31040)
    embedding_name = "model.language_model.embed_tokens.weight"
    head_name = "lm_head.weight"
    embedding = SimpleNamespace(weight=weight(embedding_name, slice(rank * 31040, (rank + 1) * 31040)), shard_indices=shard)
    head = SimpleNamespace(weight=weight(head_name, slice(rank * 31040, (rank + 1) * 31040)), shard_indices=shard)
    model = SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding), lm_head=head, logits_processor=SimpleNamespace(use_fp32_lm_head=False, vocab_size=248320, do_tensor_parallel_all_gather=True, logit_scale=None, final_logit_softcapping=None))
    kernel = DecodeIO(model, mixer, peers)
    generator = torch.Generator(device=device).manual_seed(3836)
    checks = []
    for rows in (1, 2):
        token_ids = torch.tensor([31040, 248319][:rows], dtype=torch.int64, device=device)
        hidden = torch.randn((rows, 10240), dtype=torch.bfloat16, device=device, generator=generator)
        for mode in ("eager", "graph"):
            graph = None
            if mode == "graph":
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual_embedding = kernel.entry(token_ids)
                    logits, mixed = kernel.tail(hidden)
            for step in range(3):
                token_ids.sub_(1)
                hidden.add_(0.015625)
                if graph is None:
                    actual_embedding = kernel.entry(token_ids)
                    logits, mixed = kernel.tail(hidden)
                else:
                    graph.replay()
                expected_embedding = torch.stack([weight(embedding_name, int(token)) for token in token_ids.cpu().tolist()])
                native_local = torch.nn.functional.linear(mixed, kernel.head).cpu()
                gathered = [None] * 8
                distributed.all_gather_object(gathered, native_local)
                expected_logits = torch.cat(gathered, dim=-1).to(device)
                result = {"rows": rows, "mode": mode, "step": step, "embedding": error_metrics(actual_embedding, expected_embedding.repeat(1, 4)), "logits": error_metrics(logits, expected_logits), "argmax_equal": bool(torch.equal(logits.argmax(-1), expected_logits.argmax(-1))), "state": peers.check()}
                checks.append(result)
                destination.write_text(json.dumps({"passed": False, "checks": checks}, indent=2) + "\n")
                assert result["embedding"]["max_absolute_error"] == 0 and result["logits"]["finite"] and result["logits"]["nrmse"] < 0.004, result
    destination.write_text(json.dumps({"passed": True, "scope": "Real TP8 BF16 embedding/head shards; peer embedding reduce and global logits gather, tail HC mix, eager/Graph", "checks": checks, "program": kernel.program.manifest}, indent=2) + "\n")
    print(json.dumps({"rank": rank, "passed": True, "checks": len(checks)}), flush=True)
    peers.close(distributed.group.WORLD)
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
