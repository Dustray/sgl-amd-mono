import math

import torch
from torch.nn import functional

from ..errors import ContractError
from .config import FULL_ATTENTION_INTERVAL, LAYERS
from .ffn import numerical_error


SGLANG_READS = {
    "srt/layers/layernorm.py": ("op",),
    "kernels/ops/attention/fla/fused_recurrent.py": (
        "fused_recurrent_gated_delta_rule_packed_decode",
    ),
    "kernels/ops/attention/fla/layernorm_gated.py": ("layernorm_fn",),
}


SHADOW_LIMITS = {
    "logits_cosine_min": 0.999,
    "kl_nats_per_row_max": 0.01,
    "same_input_kernel_nrmse_max": 0.003,
    "model_active_state_nrmse_max": 0.03,
    "logits_nrmse_max": 0.02,
}


def within_error(error, limit):
    return error["finite"] and math.isfinite(error["normalized_rms_error"]) and error["normalized_rms_error"] <= limit


def shadow_accepted(check):
    from .attention_check import ATTENTION_LIMITS, fp32_reference_accepted

    logits = check["logits_per_row"]
    expected_kernels = {(layer, "K2") for layer in range(LAYERS)} | {(layer, "K1") for layer in range(LAYERS) if layer % FULL_ATTENTION_INTERVAL != FULL_ATTENTION_INTERVAL - 1}
    if check.get("full_attention_fused"):
        expected_kernels |= {(layer, "AttentionK1") for layer in range(FULL_ATTENTION_INTERVAL - 1, LAYERS, FULL_ATTENTION_INTERVAL)}
    if check.get("complete_decode_model"):
        if not check.get("full_attention_fused"):
            return False
        expected_kernels.add((LAYERS, "Tail"))
    observed_kernels = {(item["layer"], item["kernel"]) for item in check["local_checks"]}
    return (
        len(logits) == check["rows"] > 0
        and all(within_error(error, SHADOW_LIMITS["logits_nrmse_max"]) and error["cosine_similarity"] >= SHADOW_LIMITS["logits_cosine_min"] for error in logits)
        and len(check["kl_divergence_nats_per_row"]) == len(logits)
        and all(-1e-6 <= value <= SHADOW_LIMITS["kl_nats_per_row_max"] for value in check["kl_divergence_nats_per_row"])
        and len(check["state_checks"]) == LAYERS - LAYERS // FULL_ATTENTION_INTERVAL
        and all(within_error(error, SHADOW_LIMITS["model_active_state_nrmse_max"]) for state in check["state_checks"] for error in state.values())
        and observed_kernels == expected_kernels
        and len(check["local_checks"]) == len(expected_kernels)
        and all(within_error(item["output"], ATTENTION_LIMITS["native_fa3_nrmse_max"] if item["kernel"] == "AttentionK1" else SHADOW_LIMITS["same_input_kernel_nrmse_max"]) and ("ssm_state" not in item or within_error(item["ssm_state"], SHADOW_LIMITS["same_input_kernel_nrmse_max"])) for item in check["local_checks"])
        and all(within_error(item[name], SHADOW_LIMITS["same_input_kernel_nrmse_max"]) for item in check["local_checks"] if item["kernel"] == "AttentionK1" for name in ("key_cache", "value_cache"))
        and all("fp32_output" in item and "fa3_fp32_output" in item and fp32_reference_accepted(item["fp32_output"], item["fa3_fp32_output"]) for item in check["local_checks"] if item["kernel"] == "AttentionK1")
        and all("precision_output" in item and within_error(item["precision_output"], ATTENTION_LIMITS["precision_reference_nrmse_max"]) for item in check["local_checks"] if item["kernel"] == "AttentionK1")
        and (not check.get("complete_decode_model") or all("embedding" in item and within_error(item["embedding"], 0.0) for item in check["local_checks"] if item["layer"] == 0 and item["kernel"] == "K1"))
        and (not check.get("complete_decode_model") or all("normalized" in item and within_error(item["normalized"], 0.0) for item in check["local_checks"] if item["kernel"] == "Tail"))
    )


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
    convolved = causal_conv1d_update(projected[:, :10240].contiguous(), conv_state, conv_weight.view(10240, 4), activation="silu", conv_state_indices=indices.to(torch.int32))
    core = torch.empty((hidden.shape[0], 1, 48, 128), dtype=torch.bfloat16, device=hidden.device)
    fused_recurrent_gated_delta_rule_packed_decode(convolved, projected_ba[:, 48:].contiguous(), projected_ba[:, :48].contiguous(), a_log, dt_bias, 128 ** -0.5, ssm_state, core, indices, use_qk_l2norm_in_kernel=True)
    output = layernorm_fn(core.view(-1, 128), gated_weight, None, z=projected[:, 10240:].contiguous().view(-1, 128), eps=1.0e-6, is_rms_norm=True).view(hidden.shape[0], 6144)
    return output, residual_output, {"normalized": normalized, "projected": projected, "projected_ba": projected_ba, "convolved": convolved, "recurrent": core.view(hidden.shape[0], 6144)}


def native_k2(kernel, core, residual):
    from lightop import fuse_silu_and_mul

    output_weight, norm_weight, gate_up, down = kernel.weights
    attention_output = functional.linear(core, output_weight)
    normalized, residual_output = native_norm(attention_output, residual, norm_weight)
    projected = functional.linear(normalized, gate_up)
    activated = torch.empty((core.shape[0], 17408), device=core.device, dtype=core.dtype)
    fuse_silu_and_mul(projected, activated)
    output = functional.linear(activated, down)
    return output, residual_output, {"attention_output": attention_output, "normalized": normalized, "projected": projected, "activated": activated}
