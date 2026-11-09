import argparse
import ctypes
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import torch

from sgl_amd_mono.qwen38_27b.gfx936.attention_check import round_native_bf16


def native_probabilities(query, key, table, length):
    from sglang.srt.layers.attention.flashattention_interface import vllm_flash_attn_with_kvcache

    probabilities = torch.empty((24, length), device="cuda", dtype=torch.float32)
    for start in range(0, length, 512):
        count = min(length - start, 512)
        pages = (count + 63) // 64
        selected = table[0, start // 64:start // 64 + pages].long()
        native_key = torch.cat((torch.zeros_like(key[:1]), key[selected])).permute(0, 2, 1, 3).contiguous()
        native_table = torch.arange(1, pages + 1, device="cuda", dtype=torch.int32)[None, :]
        lengths = torch.tensor([count], device="cuda", dtype=torch.int32)
        for basis_start in range(0, count, 256):
            basis_count = min(count - basis_start, 256)
            tokens = torch.arange(basis_start, basis_start + basis_count, device="cuda")
            dimensions = torch.arange(basis_count, device="cuda")
            values = torch.zeros((pages + 1, 64, 4, 256), device="cuda", dtype=torch.float32)
            values[tokens[:, None] // 64 + 1, tokens[:, None] % 64, torch.arange(4, device="cuda")[None, :], dimensions[:, None]] = 1
            native_value = values.to(torch.float8_e5m2).permute(0, 2, 3, 1).contiguous()
            actual = vllm_flash_attn_with_kvcache(q=query.reshape(1, 1, 24, 256), k_cache=native_key, v_cache=native_value, page_table=native_table, cache_seqlens=lengths, softmax_scale=0.0625, causal=True, ver=3).reshape(24, 256)
            probabilities[:, start + basis_start:start + basis_start + basis_count] = actual[:, :basis_count].float()
    return probabilities


def probabilities_from_scores(scores):
    padded = torch.nn.functional.pad(scores, (0, 512 - scores.shape[-1]), value=-float("inf"))
    probabilities = torch.exp2(padded - padded.amax(-1, keepdim=True))
    groups = probabilities.reshape(*probabilities.shape[:-1], 64, 8)
    denominator = groups[..., 0].clone()
    for element in range(1, 8):
        denominator += groups[..., element]
    lanes = torch.arange(64, device=denominator.device)
    for offset in (32, 16, 8, 4, 2, 1):
        denominator = denominator + denominator[..., lanes ^ offset]
    return round_native_bf16(probabilities / (denominator[..., :1] + 1.0e-6))[..., :scores.shape[-1]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-scores", type=Path)
    arguments = parser.parse_args()
    captured = torch.load(arguments.capture, map_location="cuda", weights_only=True)
    if captured["lengths"].numel() != 1:
        raise ValueError("MMAC order diagnosis requires a batch-one serving capture")
    length = int(captured["lengths"][0])
    permutations = list(itertools.permutations(range(6)))
    orders = [[sum(((index >> bit) & 1) << permutation[bit] for bit in range(6)) for index in range(64)] for permutation in permutations]
    order_tensor = torch.tensor(orders, device="cuda", dtype=torch.int32)
    query = captured["stages"]["query"].contiguous()
    key = captured["key_cache"].contiguous()
    table = captured["page_table"].contiguous()
    source = Path(__file__).with_suffix(".cpp")
    directory = arguments.output.parent / ("order_codegen_" + hashlib.sha256(source.read_bytes()).hexdigest()[:16])
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "attention_order.so"
    if not binary.exists():
        subprocess.run(["/opt/dtk/bin/hipcc", "-O3", "-std=c++17", "-fPIC", "-shared", "--offload-arch=gfx936", "-ffp-contract=off", str(source), "-o", str(binary)], check=True, timeout=90)
    library = ctypes.CDLL(str(binary))
    library.probe_attention_scores.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    library.probe_attention_probabilities.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    library.probe_attention_mmac_layout.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    layout = torch.empty((6, 64, 4), device="cuda")
    status = library.probe_attention_mmac_layout(layout.data_ptr(), torch.cuda.current_stream().cuda_stream)
    if status:
        raise RuntimeError(f"MMAC layout probe HIP status {status}")
    mapping = {column: sorted({(lane // 16, element) for lane, element in (layout[column] != 0).nonzero().tolist()}) for column in range(6)}
    print(json.dumps({"mmac_column_mapping": mapping}), flush=True)
    scores = torch.empty((len(orders), 24, length), device="cuda", dtype=torch.float32)
    status = library.probe_attention_scores(query.data_ptr(), key.data_ptr(), table.data_ptr(), order_tensor.data_ptr(), scores.data_ptr(), length, len(orders), torch.cuda.current_stream().cuda_stream)
    if status:
        raise RuntimeError(f"MMAC order probe HIP status {status}")
    torch.cuda.synchronize()
    reference = native_probabilities(query, key, table, length)
    probabilities = scores.clone()
    status = library.probe_attention_probabilities(probabilities.data_ptr(), length, len(orders), torch.cuda.current_stream().cuda_stream)
    if status:
        raise RuntimeError(f"Probability probe HIP status {status}")
    mismatches = (probabilities != reference[None]).sum((-2, -1))
    values, indices = mismatches.sort()
    best = [{"index": index, "bit_permutation": permutations[index], "mismatches": int(mismatches[index])} for index in indices[:12].tolist()]
    derived_index = permutations.index((1, 2, 0, 3, 4, 5))
    report = {"scope": "BF16 MMAC quartet column permutations against native softmax probabilities recovered with basis V", "capture": str(arguments.capture), "layer": captured["layer"], "length": length, "candidates": len(orders), "natural_order_mismatches": int(mismatches[0]), "isa_derived_order": {"index": derived_index, "bit_permutation": permutations[derived_index], "mismatches": int(mismatches[derived_index])}, "best": best}
    if arguments.native_scores is not None:
        native_scores = torch.load(arguments.native_scores, map_location="cuda", weights_only=True)
        score_mismatches = (scores != native_scores).sum((-2, -1))
        score_indices = score_mismatches.argsort()[:24].tolist()
        report["score_best"] = [{"index": index, "bit_permutation": permutations[index], "mismatches": int(score_mismatches[index])} for index in score_indices]
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"scores": scores[indices[:12]].cpu(), "native_probabilities": reference.cpu(), "candidates": best, "natural_scores": scores[0].cpu()}, arguments.output.with_suffix(".pt"))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
