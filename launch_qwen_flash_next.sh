#!/usr/bin/env bash
set -euo pipefail

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

exec python3 -u -m sglang.launch_server \
  --model-path "${QWEN_FLASH_NEXT_MODEL_PATH:-/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master}" \
  --served-model-name Qwen3.8-Flash-Next \
  --host 127.0.0.1 \
  --port 30160 \
  --tp-size 8 \
  --dtype bfloat16 \
  --weight-loader-disable-mmap \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":8}' \
  --language-model-only \
  --no-ple-offload-embedding \
  --attention-backend triton \
  --moe-runner-backend triton \
  --kv-cache-dtype bfloat16 \
  --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer \
  --context-length 4096 \
  --chunked-prefill-size 256 \
  --max-running-requests 1 \
  --max-mamba-cache-size 4 \
  --max-total-tokens 4096 \
  --mem-fraction-static 0.90 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --disable-overlap-schedule \
  --disable-radix-cache \
  --reasoning-parser qwen3 \
  "$@"
