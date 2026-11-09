import argparse
import copy
import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from sgl_amd_mono.qwen38_flash_next.gfx936.program import HCMix
from sgl_amd_mono.qwen38_flash_next.gfx936.qsa import QSAArguments, QSAAttention, cache_views
from sgl_amd_mono.qwen38_flash_next.gfx936.runner import error_metrics


def make_kernel(weights):
    qkv, query_norm, key_norm, output, index, index_query_norm, index_key_norm = weights["attention"]
    namespace = lambda tensor: SimpleNamespace(weight=tensor)
    layer = SimpleNamespace(qkv_proj=namespace(qkv), q_norm=namespace(query_norm), k_norm=namespace(key_norm), o_proj=namespace(output), indexer=SimpleNamespace(index_qk_proj=namespace(index), q_layernorm=namespace(index_query_norm), k_layernorm=namespace(index_key_norm)), rotary_emb=SimpleNamespace(cos_sin_cache=weights["rotary_cache"]))
    return QSAAttention(layer, HCMix(*weights["mixer"]))


def ordered_attention(capture, weights, selected):
    from sglang.srt.layers.attention.qsa.sparse_attn import sparse_gqa_packed_decode_triton
    from sglang.srt.models.qwen4_exp import fused_sigmoid_mul

    query, _, _, gate = capture["native_qkv_gate"]
    rows = capture["hidden"].shape[0]
    query = query.reshape(rows, 3, 256).contiguous()
    key, value = cache_views(*capture["matched_input_cache"][:2])
    packed_keys, packed_values, counts = [], [], []
    for row in range(rows):
        logical = selected[row][selected[row] >= 0].long()
        physical = capture["request_tokens"][capture["requests"][row], logical].long()
        packed_keys.append(key[physical // 64, physical % 64])
        packed_values.append(value[physical // 64, physical % 64])
        counts.append(logical.numel())
    device = query.device
    counts = torch.tensor(counts, dtype=torch.int32, device=device)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32, device=device), counts.cumsum(0).int()))
    relative = torch.arange(2051, dtype=torch.int32, device=device).expand(rows, -1).clone()
    relative.masked_fill_(relative >= counts[:, None], -1)
    attention = sparse_gqa_packed_decode_triton(query, torch.cat(packed_keys), torch.cat(packed_values), relative, torch.arange(rows + 1, dtype=torch.int32, device=device), offsets, counts, 0.0625)
    core = fused_sigmoid_mul(attention.reshape(rows, -1), gate, inplace=False)
    output = torch.nn.functional.linear(core, weights["attention"][3])
    return core, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, default=Path("monokernel_reports/flash_next_tp8/precision_20261106/qsa-before-manifest.json"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.cuda.set_device(arguments.device)
    device = torch.device("cuda", arguments.device)
    capture = torch.load(arguments.capture, map_location=device, weights_only=True)
    weights = torch.load(arguments.capture.parent / capture["weights_file"], map_location=device, weights_only=True)
    kernel = make_kernel(weights)
    new_program = kernel.program
    old_manifest = json.loads(arguments.old_manifest.read_text())
    binary = Path(old_manifest["command"][-1])
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == old_manifest["binary_sha256"]
    old_program = copy.copy(new_program)
    old_program.library = ctypes.CDLL(str(binary))
    old_program.library.flash_qsa.argtypes = [ctypes.POINTER(QSAArguments), ctypes.c_void_p]
    report = {"capture": str(arguments.capture), "capture_sha256": hashlib.sha256(arguments.capture.read_bytes()).hexdigest(), "layer": capture["layer"], "tp_rank": capture["tp_rank"], "step": capture["step"], "native_hc_mixed": error_metrics(capture["mixed"], capture["native_mixed"]), "captured_native_hc_output": capture["checks"]["output"], "captured_matched_input_output": capture["matched_input_check"], "checks": [], "old_program": old_manifest, "new_program": new_program.manifest}
    selected_versions = {}
    rows = capture["hidden"].shape[0]
    for version, program in (("old_sorted", old_program), ("fixed_native_order", new_program)):
        kernel.program = program
        repeated = []
        for repeat in range(3):
            caches = [tensor.clone() for tensor in capture["cache_before"]]
            actual, _ = kernel(capture["hidden"], capture["positions"], capture["lengths"], capture["requests"], capture["locations"], capture["request_tokens"], *caches)
            torch.cuda.synchronize(device)
            assert torch.equal(kernel.mixed[:rows], capture["mixed"])
            check = {"version": version, "repeat": repeat, "output": error_metrics(actual, capture["matched_input_output"]), "projected": error_metrics(kernel.projected[:rows], capture["native_projected"][0]), "queries": error_metrics(kernel.queries[:rows], capture["native_qkv_gate"][0].reshape(rows, 3, 256)), "core": error_metrics(kernel.core[:rows], capture["native_core"].reshape(rows, 768))}
            selected = kernel.selected[:rows].clone()
            native_selected = capture["native_selected"]
            check["selected_set_matches"] = all(torch.equal(selected[row][selected[row] >= 0].sort().values, native_selected[row][native_selected[row] >= 0].sort().values) for row in range(rows))
            check["selected_order_matches"] = all(torch.equal(selected[row][selected[row] >= 0], native_selected[row][native_selected[row] >= 0]) for row in range(rows))
            if repeat == 0:
                selected_versions[version] = selected
            else:
                check["repeat_output"] = error_metrics(actual, repeated[0])
            repeated.append(actual.clone())
            report["checks"].append(check)
    native_core, native_output = ordered_attention(capture, weights, capture["native_selected"])
    old_order_core, old_order_output = ordered_attention(capture, weights, selected_versions["old_sorted"])
    report["same_query_cache_order_only"] = {"native_core_reconstruction": error_metrics(native_core, capture["native_core"]), "native_output_reconstruction": error_metrics(native_output, capture["matched_input_output"]), "old_order_core": error_metrics(old_order_core, native_core), "old_order_output": error_metrics(old_order_output, native_output)}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
