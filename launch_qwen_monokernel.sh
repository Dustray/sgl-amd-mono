#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_AMD_MONO_0518=1
export SGLANG_AMD_MONO_0518_MODE="${SGLANG_AMD_MONO_0518_MODE:-required}"
export HIP_GRAPH_ACCUMULATE_DISPATCH="${HIP_GRAPH_ACCUMULATE_DISPATCH:-0}"
export SGLANG_AMD_MONO_0518_REPORT_DIR="${SGLANG_AMD_MONO_0518_REPORT_DIR:-$script_dir/monokernel_reports}"
export SGLANG_LAUNCH_MODULE=sgl_amd_mono_0518.launcher

exec bash "$script_dir/launch_qwen.sh" --disable-overlap-schedule --cuda-graph-backend-decode disabled --context-length 4096 --mem-fraction-static 0.90 "$@"
