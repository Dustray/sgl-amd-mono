import math

import torch
from torch.nn import functional

from .check import native_norm
from .ffn import numerical_error
from ..errors import ContractError


ATTENTION_LIMITS = {"native_fa3_nrmse_max": 0.003, "precision_reference_nrmse_max": 0.003, "fp32_additional_nrmse_max": 0.003, "kv_write_nrmse_max": 0.003}


def fp32_reference_accepted(actual, native):
    return all(error["finite"] and math.isfinite(error["normalized_rms_error"]) for error in (actual, native)) and actual["normalized_rms_error"] <= native["normalized_rms_error"] + ATTENTION_LIMITS["fp32_additional_nrmse_max"]


def check_fp32_reference(actual, native, reference):
    errors = numerical_error(actual, reference), numerical_error(native, reference)
    if not fp32_reference_accepted(*errors):
        raise ContractError(f"Full attention adds excessive FP32 reference error over native FA3: {errors}")
    return errors


SGLANG_READS = {
    "srt/models/utils.py": ("fused_qk_gemma_rmsnorm",),
    "srt/layers/rotary_embedding/utils.py": ("apply_rotary_emb",),
    "kernels/ops/attention/rotary_triton.py": ("triton_mrope_fused",),
    "kernels/ops/elementwise/elementwise.py": ("fused_sigmoid_mul",),
    "srt/layers/attention/flashattention_interface.py": ("vllm_flash_attn_with_kvcache",),
}


def native_attention_prepare(kernel, hidden, residual, positions, rotary_cache):
    from sglang.srt.models.utils import fused_qk_gemma_rmsnorm
    from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
    from sglang.kernels.ops.attention.rotary_triton import triton_mrope_fused

    norm_weight, qkv_weight, query_weight, key_weight = kernel.weights
    normalized, residual_output = native_norm(hidden, residual, norm_weight)
    projected = functional.linear(normalized, qkv_weight)
    query_gate = projected[:, :12288].reshape(-1, 24, 512)
    query = query_gate[:, :, :256].reshape(-1, 6144)
    gate = query_gate[:, :, 256:].contiguous().reshape(-1, 6144)
    key, value = projected[:, 12288:13312], projected[:, 13312:]
    query, key = fused_qk_gemma_rmsnorm(query, key, query_weight, key_weight, 1.0e-6, 256)
    query = query.view(hidden.shape[0], 6144)
    key = key.view(hidden.shape[0], 1024)
    if positions.ndim == 1:
        cosine, sine = rotary_cache[positions].chunk(2, dim=-1)
        query_heads = query.view(-1, 24, 256)
        key_heads = key.view(-1, 4, 256)
        query = torch.cat((apply_rotary_emb(query_heads[..., :kernel.rotary_dim], cosine, sine, True), query_heads[..., kernel.rotary_dim:]), dim=-1).flatten(1)
        key = torch.cat((apply_rotary_emb(key_heads[..., :kernel.rotary_dim], cosine, sine, True), key_heads[..., kernel.rotary_dim:]), dim=-1).flatten(1)
    else:
        triton_mrope_fused(query, key, rotary_cache, positions, kernel.sections, 256, kernel.rotary_dim, kernel.interleaved, False, True, None)
    return {"normalized": normalized, "projected": projected, "query": query, "key": key, "value": value.contiguous(), "gate": gate}, residual_output


def native_attention(kernel, hidden, residual, positions, rotary_cache, key_cache, value_cache, locations, page_table, lengths):
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
    from sglang.srt.layers.attention.flashattention_interface import vllm_flash_attn_with_kvcache

    stages, residual_output = native_attention_prepare(kernel, hidden, residual, positions, rotary_cache)
    active = lengths > 0
    pages = locations[active].long() // 64
    offsets = locations[active].long() % 64
    key_cache[pages, offsets] = stages["key"].view(-1, 4, 256)[active].to(key_cache.dtype)
    value_cache[pages, offsets] = stages["value"].view(-1, 4, 256)[active].to(value_cache.dtype)
    native_key = key_cache.permute(0, 2, 1, 3).contiguous()
    native_value = value_cache.permute(0, 2, 3, 1).contiguous()
    output = vllm_flash_attn_with_kvcache(q=stages["query"].view(-1, 1, 24, 256), k_cache=native_key, v_cache=native_value, page_table=page_table, cache_seqlens=lengths, softmax_scale=0.0625, causal=True, ver=3).reshape(hidden.shape[0], 6144)
    output = fused_sigmoid_mul(output, stages["gate"], inplace=True)
    output = torch.where(active[:, None], output, torch.zeros_like(output))
    return output, residual_output, stages


def round_native_bf16(value):
    return ((value.contiguous().view(torch.int32) + 32768) >> 16).to(torch.int16).view(torch.bfloat16).float()


def fp32_attention(stages, key_cache, value_cache, page_table, lengths):
    return reference_attention(stages, key_cache, value_cache, page_table, lengths, False)


def precision_attention(stages, key_cache, value_cache, page_table, lengths):
    return reference_attention(stages, key_cache, value_cache, page_table, lengths, True)


def reference_attention(stages, key_cache, value_cache, page_table, lengths, native_precision):
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    outputs = []
    for row, length in enumerate(lengths.tolist()):
        if length == 0:
            outputs.append(torch.zeros((1, 6144), device=key_cache.device, dtype=torch.bfloat16))
            continue
        pages = page_table[row, :(length + 63) // 64].long()
        key = key_cache[pages].flatten(0, 1)[:length].float()
        value = value_cache[pages].flatten(0, 1)[:length].float()
        query = stages["query"][row].reshape(4, 6, 256).float()
        scores = torch.einsum("hgd,shd->hgs", query, key)
        if native_precision:
            scores = (scores * (math.log2(math.e) / 128)).half().float() * 8
            maximum = scores.amax(-1, keepdim=True)
            partials, weights = [], []
            for start in range(0, length, 512):
                piece = scores[..., start:start + 512]
                local_maximum = piece.amax(-1, keepdim=True)
                probabilities = torch.exp2(piece - local_maximum)
                padded = functional.pad(probabilities, (0, 512 - probabilities.shape[-1])).reshape(4, 6, 64, 8)
                denominator = padded[..., 0].clone()
                for element in range(1, 8):
                    denominator += padded[..., element]
                lanes = torch.arange(64, device=denominator.device)
                for offset in (32, 16, 8, 4, 2, 1):
                    denominator = denominator + denominator[..., lanes ^ offset]
                denominator = denominator[..., :1]
                probabilities = round_native_bf16(probabilities / (denominator + 1.0e-6))
                partials.append(round_native_bf16(torch.einsum("hgs,shd->hgd", probabilities, value[start:start + 512])))
                weights.append(denominator * torch.exp2(local_maximum - maximum))
            if len(partials) == 1:
                core = partials[0]
            else:
                denominator = torch.stack(weights).sum(0) + 1.0e-6
                core = sum(partial * weight / denominator for partial, weight in zip(partials, weights))
            core = round_native_bf16(core).bfloat16().reshape(1, 6144)
        else:
            probabilities = (scores * 0.0625).softmax(-1)
            core = torch.einsum("hgs,shd->hgd", probabilities, value).bfloat16().reshape(1, 6144)
        outputs.append(fused_sigmoid_mul(core, stages["gate"][row:row + 1], inplace=True))
    return torch.cat(outputs)
