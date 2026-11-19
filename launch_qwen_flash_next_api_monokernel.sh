#!/usr/bin/env bash
set -euo pipefail

# API-server variant of the Flash-Next monokernel launcher.
# This wraps launch_qwen_flash_next_api.sh (OpenAI-compatible API with the qwen3
# reasoning parser and qwen3_coder tool-call parser) and enables the mono decode path.
# Speculative decoding: contract.py accepts only the pinned MTP geometry (EAGLE/NEXTN, topk 1,
# steps 3, draft tokens 4); any other spec shape fails at startup. The sibling API script turns
# MTP on, and SGLANG_AMD_FLASH_SPEC_VERIFY=1 is what makes mono take part in it: the target
# runner installs the mono runtime and its TARGET_VERIFY batches (4 fused rows per request) run
# the mono kernels, while the draft worker stays native. With it 0 the whole server -- draft and
# target -- is plain native SGLang, i.e. "monokernel + MTP" would measure nothing of monokernel.
# Verify batches cannot be graph-captured yet, so they run eager; the draft keeps its graphs
# (see contract.spec_verify_graph_policy). Set to 0 to compare against the native MTP baseline.
export SGLANG_AMD_FLASH_SPEC_VERIFY="${SGLANG_AMD_FLASH_SPEC_VERIFY:-1}"
# Opt-in: capture the fused TARGET_VERIFY batch into the target decode graph instead of
# running it eager (default 0 keeps the validated eager verify). With 1, SGLang captures
# the verify batch (bs x DRAFT_TOKENS rows) and replays it; the mono kernels run under
# capture and GraphExecution validates the first replay bit-for-bit against eager mono,
# failing the launch loudly if SGLang's replay does not refresh every mono input. Expect a
# ~5-6% decode gain (verify eager->graph) at the cost of the startup check. Experimental.
export SGLANG_AMD_FLASH_SPEC_VERIFY_GRAPH="${SGLANG_AMD_FLASH_SPEC_VERIFY_GRAPH:-0}"
# Shadow one step per batch shape for the life of the server instead of re-arming on every
# prefill: the fused verify path is new, so the first verify batch is still compared module by
# module against native, but an interactive session does not pay a native re-run per request.
# Per-module mismatches stay fatal. The full-model logits comparison downgrades itself to a
# recorded diagnostic whenever native's own re-runs miss the same gate (KL <= 0.01, TV <= 0.05),
# which they do on this model -- native HC mix sums split-K with FP32 atomics, so it is not
# reproducible. Set SGLANG_AMD_FLASH_SHADOW_RECORD_FAILURES=1 to downgrade every check.
export SGLANG_AMD_FLASH_SHADOW_REARM_ON_PREFILL="${SGLANG_AMD_FLASH_SHADOW_REARM_ON_PREFILL:-0}"

export SGLANG_AMD_FLASH_NEXT_0518=1
export SGLANG_AMD_MONO_0518=0
export SGLANG_AMD_FLASH_REPORT_DIR="${SGLANG_AMD_FLASH_REPORT_DIR:-monokernel_reports/flash_next_tp8/serving_complete}"
export HIP_GRAPH_ACCUMULATE_DISPATCH=0  # must be set before HIP init for decode graph

exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_qwen_flash_next_api.sh" "$@"
