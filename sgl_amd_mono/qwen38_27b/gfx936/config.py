from pathlib import Path
import os


ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
MODEL_TYPE = "qwen3_5"
GPU_ARCH = "gfx936"
COMPUTE_UNITS = 80
TP_SIZE = 1
DECODE_ROWS = (1, 2)
HIDDEN = 5120
INTERMEDIATE = 17408
LAYERS = 64
FULL_ATTENTION_INTERVAL = 4
LAYER_TYPES = ("linear_attention",) * 3 + ("full_attention",)
STATE_DIMENSIONS = {
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "rms_norm_eps": 1e-6,
}


def experimental_dense_enabled():
    return os.environ.get("SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE") == "1"


def report_directory():
    return Path(os.environ.get("SGLANG_AMD_MONO_0518_REPORT_DIR", "monokernel_reports")).resolve()
