#!/usr/bin/env bash
set -euo pipefail

export SGLANG_AMD_FLASH_NEXT_0518=1
export SGLANG_AMD_MONO_0518=0
export SGLANG_AMD_FLASH_REPORT_DIR="${SGLANG_AMD_FLASH_REPORT_DIR:-monokernel_reports/flash_next_tp8/serving_complete}"
export HIP_GRAPH_ACCUMULATE_DISPATCH=0

exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_qwen_flash_next.sh" "$@"
