#!/usr/bin/env bash
set -euo pipefail

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HIP_GRAPH_ACCUMULATE_DISPATCH=0  # must be set before HIP init for decode graph
export SGLANG_USE_FUSED_TOPK_SOFTMAX=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# The PATH's python3 may point at a venv without sglang; pin the interpreter that has the
# server stack installed (override with QWEN_FLASH_PYTHON when needed).
QWEN_FLASH_PYTHON="${QWEN_FLASH_PYTHON:-/usr/bin/python3}"

exec "$QWEN_FLASH_PYTHON" -u -m sglang.launch_server \
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
  --context-length 200000 \
  --chunked-prefill-size 4096 \
  --max-running-requests 1 \
  --max-mamba-cache-size 6 \
  --max-total-tokens 200000 \
  --mem-fraction-static 0.90 \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-bs-decode 1 \
  --disable-cuda-graph-padding \
  --cuda-graph-backend-prefill disabled \
  --disable-overlap-schedule \
  --disable-radix-cache \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-moe-runner-backend aiter \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  "$@"
