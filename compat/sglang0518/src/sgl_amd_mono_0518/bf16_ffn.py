"""Compatibility imports; the kernel lives in the Qwen model directory."""

from sgl_amd_mono.qwen38_27b.gfx936.ffn_kernel import (
    ffn_partition_kernel,
    ffn_reduce_kernel,
    launch_ffn,
)

__all__ = ["ffn_partition_kernel", "ffn_reduce_kernel", "launch_ffn"]
