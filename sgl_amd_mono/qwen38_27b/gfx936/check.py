import math

import torch
from torch.nn import functional

from ..errors import ContractError
from .ffn import numerical_error


SGLANG_READS = {
    "srt/layers/layernorm.py": ("op",),
    "kernels/ops/attention/fla/fused_recurrent.py": (
        "fused_recurrent_gated_delta_rule_packed_decode",
    ),
    "kernels/ops/attention/fla/layernorm_gated.py": ("layernorm_fn",),
}


def check_error(actual, expected, limit, label):
    error = numerical_error(actual, expected)
    if not error["finite"] or not math.isfinite(error["normalized_rms_error"]) or error["normalized_rms_error"] > limit:
        raise ContractError(f"Dense Monokernel {label} failed NRMSE limit {limit}: {error}")
    return error


def native_norm(hidden, residual, weight):
    from sglang.srt.layers.layernorm import op
    from lightop import gemma_fused_add_rmsnorm

    if residual is None:
        output = torch.empty_like(hidden)
        op.rms_norm_opt(output, hidden, weight + 1.0, 1.0e-6)
        return output, hidden.clone()
    output, residual_output = hidden.clone(), residual.clone()
    return gemma_fused_add_rmsnorm(output, residual_output, weight, 1.0e-6)


def native_k1(kernel, hidden, residual, conv_state, ssm_state, indices):
    from causal_conv1d.causal_conv1d_interface import causal_conv1d_update
    from sglang.kernels.ops.attention.fla.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
    from sglang.kernels.ops.attention.fla.layernorm_gated import layernorm_fn

    norm_weight, qkvz_weight, ba_weight, conv_weight, gated_weight, a_log, dt_bias = kernel.weights
    normalized, residual_output = native_norm(hidden, residual, norm_weight)
    projected = functional.linear(normalized, qkvz_weight)
    projected_ba = functional.linear(normalized, ba_weight)
    convolved = causal_conv1d_update(projected[:, :10240].contiguous(), conv_state, conv_weight.view(10240, 4), activation="silu", conv_state_indices=indices)
    core = torch.empty((hidden.shape[0], 1, 48, 128), dtype=torch.bfloat16, device=hidden.device)
    fused_recurrent_gated_delta_rule_packed_decode(convolved, projected_ba[:, 48:].contiguous(), projected_ba[:, :48].contiguous(), a_log, dt_bias, 128 ** -0.5, ssm_state, core, indices, use_qk_l2norm_in_kernel=True)
    output = layernorm_fn(core.view(-1, 128), gated_weight, None, z=projected[:, 10240:].contiguous().view(-1, 128), eps=1.0e-6, is_rms_norm=True).view(hidden.shape[0], 6144)
    return output, residual_output, {"normalized": normalized, "projected": projected, "projected_ba": projected_ba, "convolved": convolved}


def native_k2(kernel, core, residual):
    from lightop import fuse_silu_and_mul

    output_weight, norm_weight, gate_up, down = kernel.weights
    attention_output = functional.linear(core, output_weight)
    normalized, residual_output = native_norm(attention_output, residual, norm_weight)
    projected = functional.linear(normalized, gate_up)
    activated = torch.empty((core.shape[0], 17408), device=core.device, dtype=core.dtype)
    fuse_silu_and_mul(projected, activated)
    output = functional.linear(activated, down)
    return output, residual_output, {"attention_output": attention_output, "normalized": normalized, "activated": activated}
