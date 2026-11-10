#!/usr/bin/env bash
set -euo pipefail

export SGLANG_USE_FUSED_TOPK_SOFTMAX=0

exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_qwen_flash_next_monokernel.sh" \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 2 \
  --cuda-graph-bs-decode 1 2 \
  --disable-cuda-graph-padding \
  --max-running-requests 2 \
  --max-mamba-cache-size 8 \
  --max-total-tokens 8192 \
  --numa-node 0 1 2 3 4 5 6 7 \
  "$@"
