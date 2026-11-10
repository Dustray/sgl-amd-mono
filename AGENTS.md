# AGENTS.md — working notes for AI agents

This file is for coding agents. The human-facing performance summary is `Flash-Next 性能对比.md`. Keep exactly these two docs; put new findings here, not in new md files.

## Current focus

Only **Qwen3.8-Flash-Next, BF16 weights, TP8 on 8 × gfx936**, and the performance of its decode path
in two modes: native SGLang vs. the local Flash-Next Monokernel (`mono`). Qwen3.8-27B work and the
upstream gfx950 models are out of scope unless the user asks.

## Machine and environment

- 8 × gfx936 (80 CU, 64 GiB each). Check with `hy-smi` (compute % is column 7, VRAM % column 6; `hy-smi --showuse`
  for compute alone). **The GPUs are shared**: other users' jobs appear as KFD PIDs under
  `/sys/class/kfd/kfd/proc/` that are not visible in `/proc`. Never kill processes you did not start; if a card
  is occupied, report it and wait. **One busy card stalls everything**: every layer does two TP8 all-reduces, so
  a job saturating card 0 alone drops the whole server to ~0.26 tok/s while the other seven GPUs sit idle in
  their barriers (looks like a hang with no load). `bench_flash_next.py` samples hy-smi first and refuses to
  start unless every card is below 5% VRAM / 10% compute (`--allow-busy-gpus` to override; it records
  `gpu_usage_before`/`_after` in `results.json`). Same gate in `tools/profile_flash_next_serving.py`; the shared
  helper is `sgl_amd_mono/common/gpus.py`.
- SGLang `0.5.18+dtk2604…` (installed under `/usr/local/lib/python3.10/dist-packages/sglang`, do not
  edit it), Torch `2.11.0+dtk2604`, DTK HIP compiler at `/opt/dtk/bin/hipcc`.
- Model: `/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master`
  (≈335 GiB BF16, 48 layers = 36 GDN + 12 QSA, 512 experts top-10, HC, PLE). Loading takes ~4–5 min.
- Git branch `local/sglang-0.5.18-gfx936`; upstream base pinned in `UPSTREAM.lock.json`.
  Do not commit unless the user asks.
- Install (editable): `python -m pip install --no-deps --no-build-isolation -e . -e compat/sglang0518`

## Layout

| Path | What |
|---|---|
| `bench_flash_next.py` | **The** benchmark entry: launches/attaches a server, measures TTFT/TPOT/E2E/prefill/throughput, optional torch profile |
| `launch_qwen_flash_next.sh` | Native BF16 server, base flags for every Flash-Next run (BF16 KV, Triton attn/MoE, page 64, ctx 4096, chunked prefill 256, overlap/radix off) |
| `launch_qwen_flash_next_monokernel.sh` | Same + `SGLANG_AMD_FLASH_NEXT_0518=1` (mono decode) |
| `launch_qwen_flash_next_monokernel_graph.sh` | Mono + decode graph for batch 1/2 |
| `sgl_amd_mono/qwen38_flash_next/gfx936/` | Mono implementation: HIP C++ kernels (`kernel.cpp`, `gdn.cpp`, `qsa.cpp`, `ple.cpp`, `decode_io.cpp`), `runner.py` (runtime, shadow checks), `graph.py` (decode-graph capture/replay checks), `program.py` (hipcc build, cached by source hash under `$SGLANG_AMD_FLASH_REPORT_DIR/codegen`), `contract.py` (startup config checks), `native_sources.json` (pinned SGLang interface hashes) |
| `sgl_amd_mono/qwen38_flash_next/gfx936/{reference,accuracy_trace}.py` | Deterministic reference policy (opt-in); per-forward logits capture for native-vs-mono services (`SGLANG_AMD_FLASH_ACCURACY_TRACE_DIR`). `SGLANG_AMD_FLASH_CAPTURE_QSA_STEP` / `_LAYERS` dump QSA state on rank 2 for diagnosis |
| `compat/sglang0518/src/sgl_amd_mono_0518/flash_next.py` | SGLang 0.5.18 HookRegistry plugin: installs the runtime after `ModelRunner.initialize`, wraps decode-graph capture/execute, exposes state in `/server_info` |
| `sgl_amd_mono/common/profiling.py`, `profile_trace.py` | Profile range markers (`SGLANG_AMD_PROFILE_RANGES=1`) and the 8-rank Perfetto merge |
| `tools/profile_flash_next_serving.py` | Standalone profile capture (`--mode native|monokernel`, `--decode-graph`): torch profiler → merged Perfetto, or `--hipprof` for a HIP-level `.pftrace` |
| `tools/summarize_serving_profile.py`, `tools/validate_serving_profile.py`, `tools/profile_queries.sql` | Re-merge raw rank traces; validate a merged trace with Perfetto trace processor (`/tmp/monokernel-perfetto/linux-amd64/trace_processor_shell`, may need re-download) |
| `tools/verify_flash_next_{gdn,qsa,ple,io,hc,moe,tp8}.py` | Per-module real-weight checks |
| `tools/verify_flash_next_{serving,correctness,batch_accuracy,default_accuracy}.py`, `run_flash_next_accuracy_matrix.py`, `diagnose_flash_next_*` | Accuracy acceptance and diagnosis (still failing, see below) |
| `tools/source_status.py --verify` | Checks pinned SGLang interface files |
| `profiles/` | Curated Perfetto traces (git-ignored), b1 512→64: native eager, native decode-graph, mono eager |
| `bench_results/` | Output of `bench_flash_next.py` (git-ignored) |
| `monokernel_reports/` | Historical raw run data (git-ignored, 19 GiB) |
| `archive/` | 27B launch scripts and tools, superseded Flash-Next benchmark orchestration scripts, 27B benchmark data. Reference only |

## Running

```bash
# Native or mono, eager or decode graph, any lengths; prints and saves a table
python3 bench_flash_next.py --mode native --batch-size 1 --input-len 512 --output-len 64
python3 bench_flash_next.py --mode mono --graph --batch-size 1 2 --input-len 64 512 2048 --output-len 16 64 128
python3 bench_flash_next.py --mode mono --graph --profile --input-len 512 --output-len 64   # + Perfetto trace per case
python3 bench_flash_next.py --url http://127.0.0.1:30000 --mode mono --graph ...           # attach to a running server
python3 bench_flash_next.py --mode native ... -- --some-sglang-flag value                   # extra server args after --
```

Output dir (default `bench_results/<stamp>_<mode>_<graph|eager>/`): `summary.md`, `summary.csv`,
`results.json` (every request's token arrival times, server config, input SHA256), `server.log`,
`server_info.json`, `runtime/rank-*.json` (mono state), `profile/<case>/inference.perfetto.json.gz`.

Metric definitions used by the script (keep them when comparing to old data):
- TTFT: request send → first output token arrives (client side, streaming, `stream_interval=1`).
- TPOT: (last token − first token) / (output_len − 1), per request.
- E2E: request send → stream end. Prefill: server `prefill_finished_time − forward_entry_time` (needs `--enable-metrics`).
- decode tok/s: batch × (output_len − 1) / (last token of the wave − earliest first token). output tok/s: batch × output_len / wave wall time.
- Inputs are fixed random token ids (seed 42, ids in [1000, vocab−1000)); `temperature=0`, `ignore_eos`, cache flushed before every wave, cached tokens and retractions must be 0.

## Mono constraints (enforced by `contract.py` / `graph.py`)

- BF16 weights and KV, Triton attention + MoE, page 64, context ≤ 4096, decode batch 1 or 2,
  overlap schedule and radix cache off, PLE embedding on GPU. Prefill always runs native SGLang.
- Decode graph: `--cuda-graph-backend-decode full`, capture only bs ⊆ {1,2}, `--disable-cuda-graph-padding`,
  and `HIP_GRAPH_ACCUMULATE_DISPATCH=0` **before HIP init** (also required by the plugin for native+graph).
- Each decode step = 99 fused launches per rank (1 entry, 36 GDN, 12 QSA, 48 post-attention, 1 PLE, 1 tail).
  `/server_info` → `internal_states[*].amd_flash_next_monokernel` has `completed_decode_steps`, `executions`, `acceptance_status`.
- Always set `SGLANG_USE_FUSED_TOPK_SOFTMAX=0`: the LightOp fused top-k selects wrong experts on real routing inputs.
- Host memory: each rank stages ~45 GiB on the host while loading. Other users' jobs sometimes hold >100 GiB on
  NUMA node 0 (`/sys/devices/system/node/node0/meminfo`), so binding rank 0 to node 0 gets it OOM-killed (exit -9
  at "Multi-thread loading shards 0/131"). `bench_flash_next.py` does not bind by default (`--numa off|auto|on`).

Important environment variables:

| Variable | Meaning |
|---|---|
| `SGLANG_AMD_FLASH_NEXT_0518=1` | Attach the mono runtime (0 = native) |
| `SGLANG_AMD_FLASH_REPORT_DIR` | Per-run dir for `rank-*.json` and the kernel codegen cache |
| `SGLANG_AMD_FLASH_SHADOW_STEPS` / `_REARM_ON_PREFILL=0` | Shadow (native reference) checks per batch size; `REARM=0` keeps them out of timed requests |
| `SGLANG_AMD_FLASH_SHADOW_RECORD_FAILURES=1` | Record shadow accuracy failures and keep serving (needed for any perf run today) |
| `SGLANG_AMD_FLASH_ACCURACY_REFERENCE=1` | Deterministic reference policy for accuracy work only; never for perf |
| `SGLANG_AMD_PROFILE_RANGES=1`, `SGLANG_AMD_PROFILE_EVENT_DIR` | Install profile markers / frontend event capture |

## hipprof traces

`hipprof` (DTK, `/opt/dtk/bin`) traces HIP API, kernels and copies below torch. The server runs under
`hipprof --hip-trace --follow-fork --trace-off --session <name>`, so loading and warmup are not traced, and
the session client turns tracing on around the one measured request:

```bash
python3 tools/profile_flash_next_serving.py --mode native --hipprof \
  --input-tokens 512 --output-tokens 64 --batch-size 1 --output-dir <dir>      # add --decode-graph for native+graph
python3 tools/profile_flash_next_serving.py --mode monokernel --hipprof ...     # mono eager
```

Behaviour worth knowing:
- `--session-client <name> --start|--stop|--flush` work; **never call `--exit`** — it drops the control pipe and
  tracing keeps running (observed db growth after it).
- The `.pftrace` is written only when the traced process exits, so stop the **sglang python process**, not the
  hipprof wrapper (a signal to hipprof kills the exporter). `stop_under_hipprof()` finds it under `/proc` by
  `sglang.launch_server` + `--port`. Export of a large trace takes minutes.
- `-o <path>` gets `.pftrace` appended, and `<path>.db` / `.hipkernel.csv` / `.hiptrace.csv` land next to it.
- Prompts come from `prompts()` (tokenizer round-trip, deterministic): 512 tokens hash to `3ea5909efa03…`,
  the same ids as the existing traces in `profiles/`.
- `SGLANG_AMD_PROFILE_RANGES` is forced to 0 under `--hipprof`: the torch range markers are not in a HIP trace
  and would only slow the eager path.

## Status (2026-11-10)

- Mono decode runs end to end, with and without decode graph. Graph vs. eager mono is bit-identical
  on the first real request (logits and all 48 layers' caches) on all ranks.
- **Full accuracy acceptance vs. native still fails.** Gate per decode row: logits finite, `KL(P_native‖P_mono) ≤ 0.01`,
  `TV ≤ 0.05`, greedy argmax equal (raw NRMSE is reported only). 2026-11-06 matrix (b1: 64/512/2048/3072→64,
  b2: 64/512/1536→64, 472 fused forwards): all 7 combos fail; 13/13 short exact-answer prompts match native.
  Known causes: native HC mix uses FP32 atomic split-K, so the same input can flip the 10th/11th MoE expert
  (tie at the router boundary; reproduced 15×110 / 1×267 on layer 33); native QSA top-k is non-deterministic on
  ties above 512 blocks. Already fixed in mono: HC combine reduction order, short-context QSA block order, TP
  all-reduce rounds to BF16 with RTZ (matches AITER), SiLU rounded to BF16 before × up, HC down-proj split-K by 256.
  `SGLANG_AMD_FLASH_ACCURACY_REFERENCE=1` installs a deterministic reference (fixed HC sum order, stable QSA ties);
  even then b1 2048/3072→64 have 6/63 non-reproducible reference steps. Do not relax thresholds; do not call
  perf numbers "accepted". Raw evidence: `monokernel_reports/flash_next_tp8/` (`precision_*`, `accuracy_*`, `serving_complete_v5`).
- Mono decode ≈ 90 ms/token (b1, ctx 512); graph saves only ~5% because launch overhead is small at 99 launches.
  The kernels themselves are the bottleneck: native BF16 + decode graph (same Triton/BF16-KV config) runs at
  TPOT ≈ 45–49 ms across the whole matrix, so mono + graph is 1.1–1.9× slower end to end at b1 and 1.2–2.1× at b2.
  Full 20-case native+graph data: `bench_results/20261110_native_graph_{grid,16-1024,b2_2048_rerun}/`.
- Latest numbers and how they were measured: `Flash-Next 性能对比.md`.

## Pitfalls already hit

- **Comparing against the wrong native baseline.** `monokernel_reports/flash_next_tp8/native_graph_no_spec_20261109_151454/native_graph_detailed`
  is native with **FP8 KV + FA3 + AITER MoE, overlap/radix on** — not the mono configuration. Only compare
  mono against native runs with BF16 KV + Triton (check `server_info.json`).
- Decode-graph capture feeds dummy rows on SGLang's reserved request slot 0 and KV slot 0 (both rows of a bs=2
  capture share them). Kernel input validation must accept that reserved slot (fixed in `ple.cpp` stage 500
  and `qsa.cpp` stage 400); a failure shows as `Flash-Next device synchronization failed at stage N`.
- Under graph replay `model.forward` is not called: profile step markers come from the wrapped
  `DecodeCudaGraphRunner.execute`, and per-layer `module/...` CPU ranges do not exist for decode.
- PyTorch profiler copies `record_function` ranges onto the GPU timeline (`gpu_user_annotation`) for graph
  replays. Count phases only from `cat == "user_annotation"`.
- `--max-total-tokens` must be ~`batch * context`, not `batch * (input + output)`: with an exact budget the
  scheduler serializes a batch-2 wave (second request starts only after the first finishes), which silently
  inflates TTFT/TPOT. `bench_flash_next.py` now requests `batch * 4096` and warns when a wave was serialized.
- `/proc` paths like `/root/code/playground-cwdN/...` in tracebacks are the same files as in this repo.
- `pytest test/unit` has ~20 collection errors from other models' gfx950-only imports; run the Flash-Next
  files directly: `test/unit/test_qwen38_flash_next_{gfx936,graph,layout}.py test/unit/test_profile_trace.py test/unit/test_sglang0518_compat.py`
  (91 pass, 5 skip on 2026-11-10). `test_qwen38_flash_next_spec.py` is the upstream gfx950 spec and needs a newer SGLang.
- Kernel binaries are cached by source hash; editing a `.cpp` triggers a rebuild on next server start.
