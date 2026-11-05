from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Config:
    mode: str
    report_dir: Path

    @classmethod
    def from_environment(cls):
        enabled = os.environ.get("SGLANG_AMD_MONO_0518", "0")
        if enabled not in {"0", "1"}:
            raise ValueError("SGLANG_AMD_MONO_0518 must be 0 or 1")
        mode = os.environ.get("SGLANG_AMD_MONO_0518_MODE", "required")
        if mode not in {"required", "audit", "fused_ffn", "monokernel"}:
            raise ValueError("SGLANG_AMD_MONO_0518_MODE must be required, audit, fused_ffn or monokernel")
        report_dir = Path(
            os.environ.get("SGLANG_AMD_MONO_0518_REPORT_DIR", "monokernel_reports")
        ).resolve()
        return cls(mode=mode if enabled == "1" else "off", report_dir=report_dir)
