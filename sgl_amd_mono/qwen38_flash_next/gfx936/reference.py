import hashlib
from importlib import import_module
import os
from pathlib import Path

import torch
import triton
import triton.language as tl


_installed = False
_counts = {"hc": 0, "topk": 0}
POLICY = "fixed_hc_split256_ascending_sum_stable_qsa_topk_v1"


@triton.jit
def _down_partials(inputs, weights, partials, rows: tl.constexpr, width: tl.constexpr, rank: tl.constexpr):
    chunk = tl.program_id(0)
    tile = tl.program_id(1)
    row = tl.arange(0, 16)
    column = chunk * 256 + tl.arange(0, 256)
    output = tile * 32 + tl.arange(0, 32)
    hidden = tl.load(inputs + row[:, None] * width + column[None, :], mask=row[:, None] < rows, other=0)
    weight = tl.load(weights + output[:, None] * width + column[None, :], mask=output[:, None] < rank, other=0)
    values = tl.dot(hidden, tl.trans(weight))
    tl.store(partials + (chunk * 16 + row[:, None]) * rank + output[None, :], values, mask=output[None, :] < rank)


@triton.jit
def _reduce_partials(partials, reduced, rank: tl.constexpr, chunks: tl.constexpr):
    column = tl.program_id(0) * 32 + tl.arange(0, 32)
    row = tl.arange(0, 16)
    total = tl.full((16, 32), 0.0, tl.float32)
    for chunk in range(chunks):
        total += tl.load(partials + (chunk * 16 + row[:, None]) * rank + column[None, :], mask=column[None, :] < rank, other=0)
    tl.store(reduced + row[:, None] * rank + column[None, :], total, mask=column[None, :] < rank)


@triton.jit
def _up_mix(inputs, weights, reduced, output, rows: tl.constexpr, rank: tl.constexpr, hidden: tl.constexpr, branches: tl.constexpr):
    column = tl.program_id(0) * 32 + tl.arange(0, 32)
    row = tl.arange(0, 16)
    branch = tl.arange(0, branches)
    coordinates = tl.reshape(branch[:, None] * hidden + column[None, :], (branches * 32,))
    rank_offset = tl.arange(0, 64)
    accumulator = tl.full((16, branches * 32), 0.0, tl.float32)
    for start in range(0, rank, 64):
        inner = start + rank_offset
        values = tl.load(reduced + row[:, None] * rank + inner[None, :], mask=inner[None, :] < rank, other=0) / branches
        activated = (values * tl.sigmoid(values)).to(inputs.dtype.element_ty)
        weight = tl.load(weights + coordinates[:, None] * rank + inner[None, :], mask=(coordinates[:, None] < branches * hidden) & (inner[None, :] < rank), other=0)
        accumulator = tl.dot(activated, tl.trans(weight), accumulator)
    gates = tl.sigmoid(tl.reshape(accumulator, (16, branches, 32)))
    residual = tl.load(inputs + row[:, None, None] * (branches * hidden) + branch[None, :, None] * hidden + column[None, None, :], mask=(row[:, None, None] < rows) & (column[None, None, :] < hidden), other=0).to(tl.float32)
    values = tl.sum(gates * residual, axis=1) / branches
    tl.store(output + row[:, None] * hidden + column[None, :], values, mask=(row[:, None] < rows) & (column[None, :] < hidden))


def deterministic_hc_mix(normalized, down, up, hc, hs):
    if normalized.shape[1] % 256 or not 0 < normalized.shape[0] <= 16 or hc != 4 or hs != 2560:
        raise ValueError("Accuracy HC reference requires Flash-Next decode geometry")
    _counts["hc"] += 1
    rows, width = normalized.shape
    rank = down.shape[0]
    partials = torch.empty((width // 256, 16, rank), dtype=torch.float32, device=normalized.device)
    reduced = torch.empty((16, rank), dtype=torch.float32, device=normalized.device)
    output = torch.empty((rows, hs), dtype=normalized.dtype, device=normalized.device)
    _down_partials[(width // 256, triton.cdiv(rank, 32))](normalized, down, partials, rows, width, rank, num_warps=8)
    _reduce_partials[(triton.cdiv(rank, 32),)](partials, reduced, rank, width // 256, num_warps=8)
    _up_mix[(triton.cdiv(hs, 32),)](normalized, up, reduced, output, rows, rank, hs, hc, num_warps=8)
    return output


def stable_topk(score, lengths, topk, row_starts=None):
    _counts["topk"] += 1
    if score.shape[1] == 0:
        return torch.full((score.shape[0], topk), -1, dtype=torch.int32, device=score.device)
    width = max(score.shape[1], topk)
    columns = torch.arange(width, device=score.device)[None, :]
    starts = torch.zeros_like(lengths) if row_starts is None else row_starts
    absolute = columns + starts[:, None]
    valid = (columns < lengths[:, None]) & (absolute < score.shape[1])
    values = score.gather(1, absolute.clamp(max=max(score.shape[1] - 1, 0)).expand(score.shape[0], -1)).masked_fill(~valid, -torch.inf)
    ranked = values.argsort(dim=-1, descending=True, stable=True)[:, :topk]
    chronological = columns[:, :topk].expand(score.shape[0], -1)
    chosen = torch.where(lengths[:, None] <= topk, chronological, ranked)
    return chosen.masked_fill(chronological >= lengths[:, None], -1).int()


def snapshot():
    return {"enabled": _installed, "policy": POLICY, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "calls": dict(_counts), "scope": "explicit deterministic accuracy reference; not unmodified native performance or equivalence"}


def install():
    global _installed
    if _installed:
        return snapshot()
    if os.environ.get("SGLANG_AMD_FLASH_ACCURACY_REFERENCE", "0") != "1":
        raise ValueError("Accuracy reference requires explicit SGLANG_AMD_FLASH_ACCURACY_REFERENCE=1")
    import_module("sglang.srt.layers.hc_mix_triton").fused_hc_mix = deterministic_hc_mix
    import_module("sglang.srt.layers.hyperconnection").fused_hc_mix = deterministic_hc_mix
    import_module("sglang.kernels.ops.elementwise.fast_topk").fast_topk = stable_topk
    _installed = True
    return snapshot()
