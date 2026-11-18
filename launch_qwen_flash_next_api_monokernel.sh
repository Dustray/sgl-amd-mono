#!/usr/bin/env bash
set -euo pipefail

# API-server variant of the Flash-Next monokernel launcher.
# This wraps launch_qwen_flash_next_api.sh (OpenAI-compatible API with the qwen3
# reasoning parser and qwen3_coder tool-call parser) and enables the mono decode path.
# Speculative decoding stays off: contract.py rejects speculative_algorithm under mono.

export SGLANG_AMD_FLASH_NEXT_0518=1
export SGLANG_AMD_MONO_0518=0
export SGLANG_AMD_FLASH_REPORT_DIR="${SGLANG_AMD_FLASH_REPORT_DIR:-monokernel_reports/flash_next_tp8/serving_complete}"
export HIP_GRAPH_ACCUMULATE_DISPATCH=0  # must be set before HIP init for decode graph

exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_qwen_flash_next_api.sh" "$@"
