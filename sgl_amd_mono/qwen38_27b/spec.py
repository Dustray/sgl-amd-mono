"""Model admission and runner construction for SGLang 0.5.18's hook adapter.

This module does not export a native MonoSpec: that interface is absent from
the target SGLang, and a newer SGLang forward has not been validated yet.
"""

from .errors import UnsupportedConfiguration
from .gfx936 import config as geometry


def ffn_disable_reasons(configuration, tp_size, device, expected_flydsl):
    text = configuration.get("text_config", configuration)
    reasons = []
    if device.get("gpu_arch") != geometry.GPU_ARCH:
        reasons.append("BF16 FFN fusion has only been validated on gfx936")
    if not device.get("flydsl_installed") or device.get("flydsl_version") != expected_flydsl:
        reasons.append(f"BF16 FFN fusion requires validated FlyDSL {expected_flydsl}")
    if (
        configuration.get("architectures") != [geometry.ARCHITECTURE]
        or configuration.get("model_type") != geometry.MODEL_TYPE
        or text.get("hidden_size") != geometry.HIDDEN
        or text.get("intermediate_size") != geometry.INTERMEDIATE
        or text.get("num_hidden_layers") != geometry.LAYERS
        or text.get("hidden_act") != "silu"
        or text.get("dtype", text.get("torch_dtype")) != "bfloat16"
        or configuration.get("quantization_config")
        or text.get("quantization_config")
    ):
        reasons.append("BF16 FFN fusion requires the validated dense Qwen3_5 BF16 27B configuration")
    if tp_size != geometry.TP_SIZE:
        reasons.append("BF16 FFN fusion currently requires TP1")
    return reasons


def dense_disable_reasons(configuration, tp_size, device, expected_flydsl):
    reasons = [reason for reason in ffn_disable_reasons(configuration, tp_size, device, expected_flydsl) if "FlyDSL" not in reason]
    if device.get("gpu_compute_units") != geometry.COMPUTE_UNITS:
        reasons.append("Dense cooperative Monokernel requires the validated 80-CU grid")
    text = configuration.get("text_config", configuration)
    for name, expected in geometry.STATE_DIMENSIONS.items():
        if text.get(name) != expected:
            reasons.append(f"Dense Monokernel requires {name}={expected}, got {text.get(name)}")
    if text.get("full_attention_interval", geometry.FULL_ATTENTION_INTERVAL) != geometry.FULL_ATTENTION_INTERVAL:
        reasons.append("Dense Monokernel requires full_attention_interval=4")
    expected_layers = list(geometry.LAYER_TYPES * (geometry.LAYERS // geometry.FULL_ATTENTION_INTERVAL))
    if text.get("layer_types", expected_layers) != expected_layers:
        reasons.append("Dense Monokernel requires three GDN layers followed by one full-attention layer")
    if not geometry.experimental_dense_enabled():
        reasons.append("The dense BF16 prototype has not passed acceptance; experimental runs require SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE=1")
    return reasons


def build(runner, mode):
    if mode == "fused_ffn":
        from .gfx936.ffn_runner import FFNRuntime

        return FFNRuntime(runner)
    if mode == "required":
        raise UnsupportedConfiguration("Strict Monokernel mode refuses the unaccepted dense BF16 prototype")
    if mode == "monokernel":
        if not geometry.experimental_dense_enabled():
            raise UnsupportedConfiguration("Dense BF16 prototype requires explicit experimental opt-in")
        from .gfx936.runner import MonoRuntime

        return MonoRuntime(runner)
    raise UnsupportedConfiguration(f"No Qwen3.8-27B fused runner for mode {mode!r}")
