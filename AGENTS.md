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
  **This install is no longer pristine (noticed 2026-11-19).** `srt/layers/attention/qwen_sparse_attn_backend.py`
  was edited in place on 2026-11-19 08:23 (89546 → 89764 bytes; every other file in that directory still has the
  2026-09-21 install mtime), and `native_sources.json` was repinned from `e889f26a…` to `da3a76ac…` to match.
  `e889f26a…` is the hash pip recorded at install time, so the old pin was the shipped file and the new pin only
  exists on this machine: `tools/source_status.py --verify` passes here but will fail on a clean install, and
  anything that depends on the edit will not reproduce there. The original is not recoverable locally (no wheel
  in the pip cache), so the edit is undocumented. Prefer working around SGLang bugs in the plugin, as commit
  `a9ecb7e` did, and if the install must be patched, keep the patch in the repo.
- Model: `/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master`
  (≈335 GiB BF16, 48 layers = 36 GDN + 12 QSA, 512 experts top-10, HC, PLE). Loading takes ~4–5 min.
- Git branch `local/sglang-0.5.18-gfx936`; upstream base pinned in `UPSTREAM.lock.json`.
  Do not commit unless the user asks.
- Install (editable): `python -m pip install --no-deps --no-build-isolation -e . -e compat/sglang0518`

## Layout

| Path | What |
|---|---|
| `bench_flash_next.py` | **The** benchmark entry: launches/attaches a server, measures TTFT/TPOT/E2E/prefill/throughput, optional torch profile |
| `launch_qwen_flash_next.sh` | Native BF16 server, base flags for every Flash-Next run (BF16 KV, Triton attn/MoE, page 64, ctx 4096, chunked prefill 256, overlap/radix off); `bench_flash_next.py --context-length` appends an override |
| `launch_qwen_flash_next_monokernel.sh` | Same + `SGLANG_AMD_FLASH_NEXT_0518=1` (mono decode) |
| `launch_qwen_flash_next_monokernel_graph.sh` | Mono + decode graph for batch 1/2 |
| `sgl_amd_mono/qwen38_flash_next/gfx936/` | Mono implementation: HIP C++ kernels (`kernel.cpp`, `gdn.cpp`, `qsa.cpp`, `ple.cpp`, `decode_io.cpp`), `runner.py` (runtime, shadow checks), `graph.py` (decode-graph capture/replay checks), `program.py` (hipcc build, cached by source hash under `$SGLANG_AMD_FLASH_REPORT_DIR/codegen`), `contract.py` (startup config checks), `native_sources.json` (pinned SGLang interface hashes) |
| `sgl_amd_mono/qwen38_flash_next/gfx936/{reference,accuracy_trace}.py` | Deterministic reference policy (opt-in); per-forward logits capture for native-vs-mono services (`SGLANG_AMD_FLASH_ACCURACY_TRACE_DIR`). `SGLANG_AMD_FLASH_CAPTURE_QSA_STEP` / `_LAYERS` dump QSA state on rank 2 for diagnosis |
| `compat/sglang0518/src/sgl_amd_mono_0518/flash_next.py` | SGLang 0.5.18 HookRegistry plugin: installs the runtime after `ModelRunner.initialize`, wraps decode-graph capture/execute, exposes state in `/server_info` |
| `sgl_amd_mono/qwen38_flash_next/gfx936/{stage_timing,cu_timing}.py` | Device-side timing exporters: per-barrier stages (block 0) and per-compute-unit occupancy (every block, one Perfetto row per CU) |
| `sgl_amd_mono/common/profiling.py`, `profile_trace.py` | Profile range markers (`SGLANG_AMD_PROFILE_RANGES=1`) and the 8-rank Perfetto merge. `merge_profiles(..., mono_only=True)` (CLI: `--mono-only` / `--profile-mono-only`) validates on the full traces but exports only fused kernels, device stages, `inference/` step markers and `mono/` ranges → `inference.mono_only.json.gz`, ~1% of the events (b1 512→64: 313 MB → 2.3 MB); host/frontend/client events dropped |
| `tools/profile_flash_next_serving.py` | Standalone profile capture (`--mode native|monokernel`, `--decode-graph`): torch profiler → merged Perfetto, or `--hipprof` for a HIP-level `.pftrace` |
| `tools/summarize_serving_profile.py`, `tools/validate_serving_profile.py`, `tools/profile_queries.sql` | Re-merge raw rank traces; validate a merged trace with Perfetto trace processor (`/tmp/monokernel-perfetto/linux-amd64/trace_processor_shell`, may need re-download) |
| `tools/verify_flash_next_{gdn,qsa,ple,io,hc,moe,tp8}.py` | Per-module real-weight checks; `verify_flash_next_gdn.py --verify` also covers the fused MTP verify branch |
| `tools/verify_flash_next_cu_timing.py` | Device check for the per-CU marks: runs `hc_mix` on one GPU (no model) and verifies CU placement, mark order and drops |
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

- BF16 weights and KV, Triton attention + MoE, page 64, context ≤ 262144, decode batch 1 or 2 (a fused MTP
  verify batch is `DRAFT_TOKENS` rows per request, so up to `MAX_ROWS` = 8 rows),
  overlap schedule and radix cache off, PLE embedding on GPU. Prefill always runs native SGLang.
- **Long context (2026-11-17).** The QSA index top-k was one 1024-wide bitonic sort, which capped the
  context at 4096. It now sorts the 4-token groups in chunks of 1024 (one block per chunk) and merges
  the per-chunk best-512 lists pairwise, `log2(chunks)` rounds, barriers 416/417. The comparator is a
  total order (score descending, index ascending), so the merged top 512 equal one full sort of every
  group; `tools/verify_flash_next_qsa.py` checks exactly that (`selected_exact_topk`) rather than
  native's tie order. Cap is `QSAChunkLists = 64` chunks → `contract.QSA_CONTEXT_CAP = 262144`.
  The chunk count is chosen **on the device from the longest row of the launch**, not from the server
  context, so a short request on a long-context server costs no extra sorts or merge barriers and, at
  one chunk, runs the former single-sort code bit for bit. That also makes it correct under decode-graph
  replay, where the sequence grows between replays of one captured graph (`grid_sync` is epoch-based, so
  a per-launch barrier count is fine as long as every block agrees — they all read the same lengths).
  Buffers are still sized for the server context (`qsa_lists(context_length)` in `qsa.py`, ~1.5 MiB per
  QSA layer per rank at the cap). `index_scores` entries past the chunks a launch uses are left stale.
- **Above 512 groups the kept set may differ from native, and that is not a kernel bug.** Both sides build
  the index scores in BF16, so `index_queries` differ by ~5e-4 nrmse and scores by up to 6e-2, while at a few
  thousand groups the 512th and 513th scores can sit 1e-3 apart. Measured on `[4099, 16383]`: mono kept group
  1864, native kept 3983, whose native scores differ by 1.6e-3 — 40x below the score noise. So
  `verify_flash_next_qsa.py` now (a) always checks the kernel's selection *is* the exact top-512 of its own
  scores (`selected_exact_topk`, a total order, independent of native), (b) compares the selected set with
  native only when native's own 511/512 gap exceeds twice the score noise (`selection_unambiguous`), and
  (c) when the sets do differ at a tied cutoff, re-runs the reference on the kernel's own token list
  (`native(..., selection=...)`) so the arithmetic is still checked strictly — that moved this layer's output
  from nrmse 0.062 to 0.0015, i.e. the 6% was entirely the tie choice, not the math. Only 1 of 24 long-context
  checks needed (c); at context 4096 all 192 checks still match native's set exactly.
- Module verification runs (2026-11-17, layer 3 rank 0 unless noted, eager + graph replay, real weights):
  `--context 4096` (all layers/ranks, 192 checks), `--context 16384 --lengths 4097 16381`,
  `--context 262144 --lengths 65533 262141` — all pass. Lengths now go per row, so batch 2 covers two
  *different* lengths (`[63, 16381]`, `[16381, 63]`): the short row only reaches the multi-chunk branch when
  a longer row in the same wave pushes the chunk count above 1, which nothing tested before.
- Decode graph: `--cuda-graph-backend-decode full`, capture only bs ⊆ {1,2}, `--disable-cuda-graph-padding`,
  and `HIP_GRAPH_ACCUMULATE_DISPATCH=0` **before HIP init** (also required by the plugin for native+graph).
- Each decode step = 99 fused launches per rank (1 entry, 36 GDN, 12 QSA, 48 post-attention, 1 PLE, 1 tail).
  `/server_info` → `internal_states[*].amd_flash_next_monokernel` has `completed_decode_steps`, `executions`, `acceptance_status`.
- Always set `SGLANG_USE_FUSED_TOPK_SOFTMAX=0`: the LightOp fused top-k selects wrong experts on real routing inputs.
- **Startup weight validation flakes (2026-11-17).** `_validate_weights` can fail at a random layer with
  `reduced_moe` nrmse ~0.2 while `max_absolute_error` is ~0.003, killing every TP scheduler before the server
  is ready (exit -9). Seen at layer 18 rows=2; the same build passed on relaunch. It is the known router tie:
  the dump showed fused vs. native `mixed` differing by 3e-7 relative, `moe_same_input` at nrmse 5e-5, but
  `reduced_moe` (native MoE on *native's* mixed) 23% off — only a flipped expert does that, and native HC mix
  uses FP32 atomic split-K, so it is not reproducible. The layer output itself was fine (nrmse 0.0015).
  Just relaunch; do not relax the 0.02 threshold. Dumps land in `$SGLANG_AMD_FLASH_REPORT_DIR/failed-layer-*.pt`
  (only the ranks that finish writing before SIGQUIT are non-empty).
- Host memory: each rank stages ~45 GiB on the host while loading. Other users' jobs sometimes hold >100 GiB on
  NUMA node 0 (`/sys/devices/system/node/node0/meminfo`), so binding rank 0 to node 0 gets it OOM-killed (exit -9
  at "Multi-thread loading shards 0/131"). `bench_flash_next.py` does not bind by default (`--numa off|auto|on`).

Important environment variables:

| Variable | Meaning |
|---|---|
| `SGLANG_AMD_FLASH_NEXT_0518=1` | Attach the mono runtime (0 = native) |
| `SGLANG_AMD_FLASH_REPORT_DIR` | Per-run dir for `rank-*.json` and the kernel codegen cache |
| `SGLANG_AMD_FLASH_STAGE_TIMING=1`, `_STEP`, `_STEPS`, `_RECORDS` | Block-0 device stage timing: enable, first decode step to export, step count, buffer records |
| `SGLANG_AMD_FLASH_CU_TIMING=1`, `_RECORDS` | Per-CU device timing (one row per compute unit); uses the same `_STEP`/`_STEPS` |
| `SGLANG_AMD_FLASH_SPEC_VERIFY=1` | Route MTP TARGET_VERIFY batches through the mono kernels. **Without it a speculative server installs no runtime at all** — draft and target are both plain native |
| `SGLANG_AMD_FLASH_SPEC_VERIFY_GRAPH=1` | Opt-in, on top of `_SPEC_VERIFY=1`: capture the fused TARGET_VERIFY batch into the target decode graph (SGLang-native TARGET_VERIFY capture, bs x DRAFT_TOKENS rows) instead of running it eager. The first replay is validated bit-for-bit against eager mono and fails the launch if SGLang's replay does not refresh every mono input. Default off = validated eager verify |
| `SGLANG_AMD_FLASH_SHADOW_STEPS` / `_REARM_ON_PREFILL=0` | Shadow (native reference) checks per batch size (verify batches count separately from decode); `REARM=0` keeps them out of timed requests |
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

## Device-side stage timing

Each kernel records a device-side start timestamp and each grid barrier records
`(stage, wall_clock64())`, so every kernel-internal phase appears in the profile even though torch
cannot see inside a fused launch.

```bash
SGLANG_AMD_FLASH_STAGE_TIMING=1 SGLANG_AMD_FLASH_STAGE_TIMING_STEP=5 SGLANG_AMD_FLASH_REPORT_DIR=<run>/runtime \
  python3 bench_flash_next.py --mode mono --graph --input-len 512 --output-len 64
python3 tools/merge_flash_next_stage_timing.py --run-dir <run>/runtime   # 8 ranks -> one Perfetto file + summary
```

- `stage_mark()` lives in `kernel.cpp` next to `grid_sync`; kernel-entry markers are 700-706 and
  barrier stage numbers are disjoint per kernel kind (hc_mix 1-4, post 10-61, gdn 300-314 (314 = MTP verify conv snapshots),
  qsa 400-413, ple 500-524, entry 600-620, tail 630-644).
  Names live in `stage_timing.py:SEGMENTS`; add a name there when you add a barrier.
- Buffer layout: `[0]` cursor, `[1]` dropped flag, then `(stage, ticks)` pairs. Attach happens per
  compiled library — each `.so` includes `kernel.cpp` and has its own copy of the device symbol.
- Only block 0 / thread 0 marks: each barrier gives block 0's arrival (`stage + 1000`) and the release. Since
  2026-11-16 a stage span runs **previous release → this release**, i.e. until the slowest block arrived, so it is
  the whole grid's time for that stage; block 0's own share is in `args.block0_compute_us` / `block0_wait_us`.
  Before that the span ended at block 0's arrival and read like an all-CU time while understating it (layer 8 gdn
  `mix.down`: block 0 235 us vs. 276 us release to release; only 20 tiles = blocks 0-4 work in that gemv, and block 0
  is not the slowest). Old `stage-timing-*.json` files are widened on read (`profile_trace.widen_segments`), so
  re-merging old runs gives the new spans. Ticks convert with `clock_khz` from `flash_info`.
- Per-CU rows are named by hardware CU, **not** by block: block 0 lands on a different CU every launch (one 2-step
  capture: 198 launches over all 80 CUs), so the block-0 stage row matches whichever CU row has `args.block == 0`.
- Ranks do not share a clock in the raw `stage-timing-*` / `cu-timing-*` files. `tools/merge_flash_next_stage_timing.py`
  aligns them (default `--align fit`): it matches the releases of the cross-rank `peer wait` barriers (stages 14, 54,
  514, 614, 634; ~99 per decode step) and fits `t_rank0 = a·t_rank + b` per rank, applied to stage and per-CU rows
  alike; report in `alignment` of the summary. 2026-11-14 on `profiles/20261113_180938_…cu_512-64_b1`: offsets
  −142…+1035 us, drift ≤ 2.4 ppm, residual median ~0.3 us, max 1.7 us — cross-rank differences below ~2 us are noise.
  `--align none` gives the old unaligned layout (then compare durations only). Tests: `test/unit/test_flash_next_stage_alignment.py`.
- Graph replay works unchanged (the device pointer is a global, not a kernel parameter). Set
  `SGLANG_AMD_FLASH_STAGE_TIMING_STEP` to a positive decode-step number to export exactly that step;
  0 disables automatic export (the default). Eager and graph replay use the same step counter.
- Previous device validation (2026-11-10, before kernel-entry markers): 4 records per `hc_mix` launch,
  monotonic ticks; `hc_mix` 342 us, `mix.down` 316 us for one decode row. Unit tests:
  `test/unit/test_flash_next_stage_timing.py`.

## Per-CU timing (one Perfetto row per compute unit)

Stage timing is block 0 only. With `SGLANG_AMD_FLASH_CU_TIMING=1` **every** block marks the same
stages and each record carries its wave's `HW_ID`, so the trace gets one row per compute unit and
shows which operator a given CU was running at an instant, and whether it was computing or parked
in a grid barrier (`sgl_amd_mono/qwen38_flash_next/gfx936/cu_timing.py`).

```bash
SGLANG_AMD_FLASH_STAGE_TIMING=1 SGLANG_AMD_FLASH_CU_TIMING=1 SGLANG_AMD_FLASH_STAGE_TIMING_STEP=5 \
  SGLANG_AMD_FLASH_REPORT_DIR=<run>/runtime python3 bench_flash_next.py --mode mono --graph --input-len 512 --output-len 64
python3 tools/merge_flash_next_stage_timing.py --run-dir <run>/runtime   # stage rows + per-CU rows, one file
python3 tools/profile_flash_next_serving.py --mode monokernel --cu-stages --output-dir <dir>   # inside a profile run
```

- Rows are named `SE<se>.SH<sh>.CU<cu> (block <b>)`; spans are `layer 12 gdn: conv state` (cat
  `cu_compute`) and `wait conv state` (cat `cu_wait`), with `args` carrying block, step, layer,
  kernel, stage and CU. The merged file keeps per-CU processes at pid `100 + rank`, clear of the
  stage-timing pids, and the summary JSON gains a `compute_units` section (per-CU compute vs. wait
  totals and compute fraction, printed as the most barrier-bound / most compute-bound CUs).
- `cu_mark()` sits inside `stage_mark()` in `kernel.cpp`, so the per-CU and block-0 streams always
  agree and no kernel needs its own instrumentation. Only thread 0 of each block writes, into that
  block's own region, so there are no atomics and marks stay in program order.
- Buffer: `[0]` records per block, `[1]` dropped flag, then per block `[cursor, (stage, ticks, hw_id)...]`.
  `SGLANG_AMD_FLASH_CU_TIMING_RECORDS` (default 8192, ~15.7 MiB per rank) holds about two decode
  steps — one step is ~3150 marks per block — so capture one step at a time and check `dropped`.
  `profile_flash_next_serving.py --cu-steps N` sizes `_RECORDS` for N steps itself and requires
  N ≤ output_tokens − 1 (a 64-token request has 63 decode steps; export fires on the last requested step, so
  N = output_tokens silently exported nothing, 2026-11-14). Each step is ~254k span events / 66 MB JSON per rank.
  The export runs synchronously in the scheduler: `--cu-steps 32` took ~7 min (2.1 GB per rank), SGLang's 300 s
  watchdog killed the server and the client only showed `Connection refused` on `/stop_profile` (2026-11-14). The tool
  now passes `--watchdog-timeout max(300, 60·N)`; when launching by hand with CU timing, pass it yourself.
  With the watchdog raised, the next 32-step run exported fine, but rank 7 was then SIGKILLed (exit -9) during torch
  profiler `stop_trace` (host OOM, cgroup `oom_kill` rising, swap full). Streaming the JSON did not help (same crash on
  rank 6 at 20:45). Real cause: the export runs *inside* the torch-profiler capture and `stage_timing.launches()` indexed
  the record tensor per element (`int(records[i, 0])`), several aten ops per record × ~8M records, all recorded with
  shapes; `stop_trace` then OOMs. `launches()`/`cu_timing.export_trace` now `.tolist()` once (one step: 160 profiler
  ops, 4.6 s). Never iterate a tensor element-wise in code that can run during a capture. Keep N ≤ 4 unless needed.
  After that the 32-step capture succeeded, but the client's automatic CU merge (17 GB JSON, 164 GB RSS, one joined string
  gzipped at level 9) ran >40 min with 0 bytes written. `merge()` now streams with gzip level 1, and
  `merge_flash_next_stage_timing.py --steps 1-4` keeps only those per-CU steps (stage rows are kept in full):
  4 of 32 steps → 253 MB `.json.gz` in ~13 min, almost all of it spent reading the 8 × 2.1 GB inputs.
- The CU trace reuses the stage trace's tick base when both are on, so the two sets of rows line up
  in one timeline. The merge aligns ranks onto rank 0's clock (see stage timing above; ~2 us precision).
- `HW_ID` decoding is gfx9 (`cu` bits 8-11, `sh` bit 12, `se` bits 13-15); wave/SIMD/pipe are
  recorded but deliberately not part of a row's identity.
- **A block keeps its CU within one launch but is placed elsewhere on the next launch**, so spans
  are attributed per launch and one CU row legitimately hosts several blocks over a trace
  (`shared_units` reports that; it is not an error). A block that changed CU *inside* a launch
  would break attribution and is counted in `migrated_launch_count` — measured 0 so far.
- Device check (no model, one GPU, ~1 s): `python3 tools/verify_flash_next_cu_timing.py --device 0
  --launches 4`. Run it after touching `kernel.cpp`. 2026-11-13 on gfx936 card 0: 80 blocks land on
  80 distinct CUs on every launch (1 block per CU, 0 idle CUs, 36 marks per block, no drops), SE
  0-7 × CU ids 0-10. Same run: `hc_mix` for one decode row spends ~23 us computing and ~361 us in
  grid barriers per launch per CU — consistent with the 342 us `hc_mix` from the stage timing.
- Merged Perfetto files leave out idle spans (2026-11-15): `cu_wait`, `cu_gap` and `/ block 0 grid wait`
  spans are left out (the block-0 wait is still inside the release-to-release stage span) of `merge_flash_next_stage_timing.py` output and of the stages `merge_profiles` embeds, so an empty
  stretch on a row means wait or idle. Alignment, `cu_check` and the summary wait/gap totals still come from the raw
  inputs. Add them back with `--keep-idle` / `merge(..., keep_idle=True)` / `merge_profiles(..., keep_device_waits=True)`.
- When per-CU traces are merged, `grid (all CUs)` spans cover each `(step, layer, kernel, stage)` from the earliest
  block compute start to the latest block barrier arrival; this is distinct from the block-0 stage row and excludes
  post-arrival barrier release time. `--no-grid` disables these spans. `stage_mark` reads `wall_clock64()` once and
  shares that tick between stage and per-CU records.
- Not embedded in the torch-profiler merged trace (`inference.perfetto.json.gz`) — the per-CU rows
  are a separate Perfetto file. Unit tests: `test/unit/test_flash_next_cu_timing.py`.

## Status (2026-11-17)

- Mono decode runs end to end, with and without decode graph. Graph vs. eager mono is bit-identical
  on the first real request (logits and all 48 layers' caches) on all ranks.
- **Fused MTP verify is in progress (2026-11-19, uncommitted).** `contract.py` accepts only
  the pinned EAGLE/NEXTN geometry (topk 1, steps 3, draft tokens 4); without `SGLANG_AMD_FLASH_SPEC_VERIFY=1`
  a speculative server installs no runtime and decodes fully natively — so a "monokernel + MTP" run with the
  env unset measures plain native SGLang, mono contributes nothing. `launch_qwen_flash_next_api_monokernel.sh`
  therefore defaults it to 1. With it, only the target runner installs the runtime and its TARGET_VERIFY
  batches (`DRAFT_TOKENS` fused rows per request) run on the mono kernels, while the draft stays native.
  `MaxRows` is now 8 (`geometry.py`), so decode buffers grew with it.
  - **Keep `--cuda-graph-backend-decode full` with fused verify.** By default a verify batch is not
    graph-captured (verify runs eager), but `disabled` is the wrong way to express that: SGLang builds the
    *draft* worker's graph runners from the same flag (`_capture_cuda_graphs` returns early on `disabled`), and
    an MTP round is 3 draft forwards plus 1 verify forward. At `full` the plugin by default skips capture on the
    target runner only (`around_graph_capture`); `FullCudaGraphBackend.can_run` is `shape_key in self._graphs`,
    so an empty graph set sends verify down the eager path where the mono kernels run, and the draft keeps its
    graphs. `contract.spec_verify_graph_policy` reports which of the two a launch picked, and the choice shows
    up in `/server_info` and the `AMD_FLASH_TP8 speculative=` startup line. With `--cuda-graph-backend-decode
    full` and the opt-in `SGLANG_AMD_FLASH_SPEC_VERIFY_GRAPH=1` the target instead installs a `GraphExecution`
    and captures the verify batch (see the next bullet).
  - **Verify-graph capture works on device (validated 2026-11-19, behind
    `SGLANG_AMD_FLASH_SPEC_VERIFY_GRAPH=1`, default off).** The key finding: SGLang 0.5.18+dtk2604
    `DecodeCudaGraphRunner` captures TARGET_VERIFY natively — for a spec target it sets
    `capture_forward_mode=TARGET_VERIFY` and a `bs x captured_req_width(=DRAFT_TOKENS=4)` shape, and
    `can_run_graph` only needs `spec_info.num_tokens_per_req == 4` plus the shape in `_graphs`. So the
    capture/replay/static-buffer-refresh machinery was already there; the plugin only had to stop skipping
    it. The change: `flash_next.after_initialize` installs `GraphExecution` on the fused target when the flag
    is on and `full`; `around_graph_capture` now prefers `_runtime.graph.capture` (run SGLang capture, the
    fused `model.forward` supplies the mono kernels under capture) and falls back to the old "skip, verify runs
    eager" only when no graph was installed; `around_graph_execute` routes the replay through
    `GraphExecution.execute`, which for a verify batch calls `_decode_states(batch)` (so the compared state set
    includes the per-draft GDN scratch + PLE side states) and, on the first replay per shape, runs
    `GraphExecution._validate` with `native_shadow=True` — that forces the eager reference through the same
    fatal per-module native-vs-mono shadow (incl. the bf16-ULP gate) AND requires graph-vs-eager-mono logits
    and states to be bit-equal, so a mismatch fails the launch loudly and never serves silently. The eager
    reference pass doubles as the per-module verify shadow that would otherwise never run once verify goes
    through a graph (coverage preserved).
    - The two runtime unknowns both cleared on the first device run (2026-11-19): (1) the target replay's
      `attn_backend.init_forward_metadata_out_graph(fb_view)` in TARGET_VERIFY mode DOES refresh
      `forward_metadata.mamba_cache_indices` in the captured tensor each replay; (2) the PLE verify branch's
      buffers are all in the replay-time static set. `verify_intermediate_state_indices` is a constant `arange`
      built once in `GDNAttnBackend.__init__` (verified); pools/scratch/`req_to_token` are static; QSA verify uses
      the device-chosen chunk count + epoch `grid_sync`, already proven capture-safe by growing-length replay.
    - To try it: `SGLANG_AMD_FLASH_SPEC_VERIFY_GRAPH=1 bash launch_qwen_flash_next_api_monokernel.sh`
      (needs `--cuda-graph-backend-decode full`, which the launcher already sets). Offline unit tests for the
      reordered `around_graph_capture` and the `_validate` verify `native_shadow` path are in
      `test/unit/test_qwen38_flash_next_graph.py` / `test_sglang0518_compat.py` (pure CPU, run masked green).
    - **Device run (2026-11-19, 200k server, b1): first TARGET_VERIFY replay is bit-equal and it is a clear win at
      long context.** All 8 ranks: `route=gfx936_tp8_bf16_verify verify_graph=verify captured into the target
      graph, draft graphs kept`; `AMD_FLASH_GRAPH_CAPTURE batch_sizes=[1]`; `cuda_graph={... target_verify=2.03,
      draft_decode=2.49, draft_extend=2.22}`. On all 8 ranks `AMD_FLASH_GRAPH_EAGER_CHECK` reported
      `{rows:1, logits:{nrmse:0.0, max_absolute_error:0.0}, state_match:...}` on the first verify replay (graph
      replay bit-equal to eager mono — logits AND GDN/PLE/conv states), and the server reached "ready to roll"
      rather than failing `_validate` at startup, so both unknowns above are confirmed refreshed correctly. The
      only logged shadow failure is the known non-fatal full-model logits one (nrmse 0.1089, byte-identical across
      all 8 ranks = reference jitter, self-downgrading, server stays up). Steady decode (ignoring the post-prefill
      first-batch dips of 1.95/9.71/6.93/3.46 tok/s): ~28-41 tok/s at 75-85k context (accept len 2.25-3.38, accept
      rate 0.42-0.79), vs. pre-verify-graph eager-verify ~20 tok/s at ~73k — bigger than the predicted 5-6% because
      verify is where long-context cost concentrates, so graphing it strips fixed per-step overhead that dominates
      there. Perf only: full accuracy acceptance vs. native (below) is still the open gate.
  - GDN verify replays the chain per request: conv advances the working window and snapshots each step, the
    recurrent state stays in registers with per-step snapshots and the working ssm is never written (native
    does the same with `disable_state_update`); the worker's commit scatters the accepted prefix back.
  - Device-validated 2026-11-19: `python3 tools/verify_flash_next_gdn.py --verify` (layers 0/1/46, ranks 0/7,
    1 and 2 requests, 12 checks, real BF16 shards, eager). Reference is `DRAFT_TOKENS` serial decode steps on a
    copy of the state — one verify row is one decode step, so after step j the working conv state *is* that
    step's window snapshot and the working ssm *is* that step's ssm snapshot. Worst nrmse: output 2.1e-3,
    ssm snapshots 6.1e-4, conv window 2.2e-5; the working ssm, the unused scratch rows and the unwritten
    scratch bytes compare bit-identical. Index tables are deliberately non-identity and a negative control
    confirmed the snapshot checks are discriminative (wrong request or wrong step order scores nrmse ~1.4,
    vs. a 0.006 gate).
  - **Second serving run (2026-11-19) cleared every per-module check: 98/98.** Including the four
    GDN layers whose nrmse sat above the old gate (29/30/37 at 0.0061-0.0069, 44 at 0.0111) but
    whose peak-scale ULP was 0.5-1.25 — layer 44 has the worst nrmse in the run *and* the best
    ULP (0.5), which is the clearest demonstration that the two axes are independent.
  - **The full-model logits gate (KL <= 0.01, TV <= 0.05, argmax equal) is not reachable by
    anything on this model, so it now downgrades itself.** On that same verify batch the shadow
    re-ran native three times from the restored state: native against itself scored KL 0.02118 /
    TV 0.085 on row 2, i.e. **the reference misses the gate it is used to judge** (native HC mix
    sums split-K with FP32 atomics — the expert-flip already documented below). mono scored KL
    0.02234 / TV 0.0822 on the same rows, inside native's own spread, and its single argmax flip
    (row 1) is an exact bf16 tie: logit[271] == logit[561] == 14.9375, top-2 margin 0.0, where
    `argmax` returns the lower index. Per-row, mono is 1.7-9.7x native's self-noise, so mono is
    *not* bit-equal to native — but this comparison cannot separate that from reference jitter.
    `_full_model_shadow` therefore scores native's own repeats through the same gate and passes
    `discriminating=all(...)` to `_shadow_failure`: the check records and keeps serving only when
    the reference failed too, and stays fatal when native does reproduce. Per-module checks are
    unaffected and still fatal by default; `SGLANG_AMD_FLASH_SHADOW_RECORD_FAILURES=1` keeps its
    documented meaning of downgrading everything.
  - Not yet done: a perf number, and an accuracy statement that survives a reproducible
    reference. The GDN layer shadow saves, compares and restores the per-step scratch, because
    the native rerun inside the shadow overwrites exactly the buffers mono's verify produces —
    without that the shadow silently graded native against itself.
  - **First real serving run (2026-11-19, rank 6).** `forward_keys` reached
    `[["EXTEND", 1, false], ["TARGET_VERIFY", 1, true]]`, i.e. mono really ran the verify batch.
    Everything passed with margin except one check: QSA verify layers 3/7 bit-identical to native
    and 11/15 at nrmse 0.0012/0.0015 (gate 0.008); post-attention (HC/MoE at 4 rows) worst 0.0007;
    conv-window snapshots and the working conv state bit-identical. GDN output nrmse crept 0.0033
    (layer 0) to 0.0061 (layer 16) and tripped the 0.006 gate there.
  - **That trip was the metric, not the kernel — the GDN verify output is now gated on nrmse OR
    bf16 ULPs at the tensor peak** (`bf16_ulp_error`, tolerance `DRAFT_TOKENS`). Evidence from the
    saved dump: 70% of elements differ, but the elementwise relative error is bimodal — elements
    at |v| >= 0.05 sit at 0.5 ULP (correct rounding) while the 1110 elements below 1e-3 reach
    15843 ULP. Absolute error is capped at 4.9e-4 tensor-wide, exactly half a ULP at the peak of
    0.136, so this is cancellation in near-zero elements, and nrmse reads high only because it
    normalises by an RMS (0.0116) that those elements drag down. Two earlier hypotheses were
    checked and **disproved** by the per-row split: there is no accumulation along the draft chain
    (row 0 0.0061, row 3 0.0050), and the depth trend is not growing absolute error (layer 1 has
    the largest max_abs, 4.9e-3, at one of the lowest nrmse). Under the new gate the real dump
    scores 0.5 ULP (pass) while a single element perturbed by the tensor peak scores 138.8 (fail).
    Decode keeps the plain nrmse gate; only the verify output carries `bf16_ulp`.
  - **ULP tolerance is 16, not `DRAFT_TOKENS` (2026-11-19).** The next serving run died at the very first verify
    batch on GDN layer 32 (rank 3, nrmse 0.0113, 6 ULP at a peak of 0.0615, fatal because the full-model
    reference repeated cleanly that time). Rebuilding the whole verify chain off the checkpoint on CPU
    (bf16 in_proj, conv+silu, exp/softplus gating, l2norm, fp32 delta recurrence, sigmoid(z) norm) produced
    a `recurrent` buffer **bit-identical to mono's** and <= 1.3 ULP in `core`, so the kernel is the exact
    chain and the 6 ULP is the reference side's rounding: native verify runs the aiter Triton kernel with its
    own fp32 accumulation orders, plus native GEMMs and a differently-ordered 8-way bf16 all-reduce. The
    per-element diff is diffuse noise (86% of elements differ, mean ~0), and peak-scaled ULP divides by the
    layer's peak, so the same ~1.5e-3 absolute floor is 0.5 ULP at layer 16's peak 0.136 and 6 ULP at
    layer 32's peak 0.0615. 16 keeps ~2.7x headroom over the worst observed dump while real divergences
    stay orders above (flipped step ~nrmse 1.4, flipped expert ~0.2, mis-indexed row moves whole 128-wide
    head blocks). Updated test: `test_gdn_shadow_gate_combines_nrmse_and_ulp` includes the layer-32 dump.
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
- **Long-context mono output degrades (11-17, open).** Users see garbled tags then an early stop on long prompts, mono
  only. Prime suspect: the decode-time compressed index key write (`qsa.cpp` stage 406-408). It is irrelevant at <= 512
  groups (every group is kept) and was only checked against our own `QSAAttention.native()` plus whole-buffer shadow
  nrmse, which dilutes one wrong row below the gate (one row of ~50k at 200k context is ~0.006). The QSA shadow now also
  reports `written_{key,value,ring,ring_positions,compressed}` on exactly the rows the step wrote,
  `read_slot_matches` (the slot index scores read == the slot the write used) and `unwritten_{key,compressed}_rows_changed`
  (bitwise stray writes); any of them fails the check. Compressed keys are written only when `seq_len % 4 == 0`, so run
  with `SGLANG_AMD_FLASH_SHADOW_STEPS>=4` (plus `_SHADOW_RECORD_FAILURES=1`) on a > 2048-token prompt.
- **Long context works end to end (11-17).** Mono runs on a 128000-token server: b1 eager TPOT 33.03 ms at
  512 input, 43.68 at 8192, 44.16 at 32768; +graph 40.10 at 32768. Native on the same 128k server is
  460–583 ms per token at every input length, because native decode scales with the *allocated* context while
  mono scales with the sequence (mono's 512-token TPOT on a 128k server equals its 4096-server number).
  Prefill is unchanged — it is always native. Accuracy: the one shadow step each run does passed its gate
  (8192: KL 0.00192 / TV 0.0305; 32768 + graph: KL 0.00017 / TV 0.00194, argmax equal, 99 module checks at
  nrmse 0), and 32768 graph replay is bit-identical to eager. That is **not** the full accuracy acceptance
  described above, which still fails and has not been run at long context. Data in
  `bench_results/20261117_mono_longctx_{ctx128k,graph_32768,8192-16_r2}`.
- Mono decode perf (11-16/17, operators #1–#10 in the doc): b1 512→64 eager TPOT 32.92 ms, +graph 29.17 ms
  vs native+graph 45.5 ms — mono is now ~28%/36% faster per token eager/graphed. Was 94.7/90.1 ms before
  #1–#10. Bit-exactness verified only for #1; #2–#10 pending. The full b1/b2 matrices in the doc are still
  pre-optimization (11-09/11-10) and need a rerun; latest runs:
  `bench_results/20261111_monokernel_eager_b1_512-64-12362` (eager) / `…-12348` (graph).
- Full 20-case native+graph data: `bench_results/20261110_native_graph_{grid,16-1024,b2_2048_rerun}/`.
