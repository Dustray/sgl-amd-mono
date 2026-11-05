#!/usr/bin/env bash
set -euo pipefail

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1

exec python3 -u -m "${SGLANG_LAUNCH_MODULE:-sglang.launch_server}" \
  --model-path /public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master \
  --weight-loader-disable-mmap \
  --served-model-name Qwen3.8-27B \
  --host 0.0.0.0 \
  --port 30000 \
  --numa-node 0 \
  --tp-size 1 \
  --pp-size 1 \
  --mamba-radix-cache-strategy extra_buffer \
  --max-running-requests 2 \
  --cuda-graph-max-bs-decode 2 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --cuda-graph-backend-prefill disabled \
  --context-length 33792 \
  --page-size 64 \
  --mem-fraction-static 0.96 \
  --mm-feature-transport cpu \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  "$@"
