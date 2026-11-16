# Qwen3.8-Flash-Next BF16：原生 vs Monokernel 性能

更新：2026-11-15（性能数据为 11-10）。

## 环境基本信息

基于光源 Qwen3.8-Flash-Next sglang 0.5.18 的镜像，将 monokernel 插件手动插入 sglang 中，并使用 DAS 的 FlyDSL。

| 项目 | 内容 |
|---|---|
| 平台 | bw1000 × 8（gfx936） |
| 框架 | sglang 0.5.18（来源：sglang-光合开发者社区）。上游带 monokernel 插件的 sglang 版本为 0.5.23 |
| 模型 | Qwen3.8-Flash-Next（来源：Qwen3.8-Flash-Next-光合开发者社区） |
| 数据类型 | 权重 BF16，KV cache BF16 |
| FlyDSL | DAS PyPI |
| 切片 | TP8 |

Monokernel 支持的模型中，唯一一个 Qwen 模型是 Qwen3.8-Flash-Next，也是唯一只需要单张 MI355X 即可跑的（MXFP4）。

原本想从 0 适配 Qwen3.8 27B，但跑出来 monokernel 模式比原生劣化十倍，所以重新适配 Qwen3.8-Flash-Next，将 gfx950 适配为 gfx936。

## 结论

- **原生开 Graph 是最快的，Mono 追不上。** 原生 + Graph 每个 token 约 45–49 ms；Mono 就算也开 Graph 也要约 86–103 ms，端到端比它慢 1.1–1.9 倍（batch 1）、1.2–2.1 倍（batch 2）。瓶颈是 Mono kernel 本身的计算速度。
- **两边都不开 Graph 时，Mono 快 2–3 倍。** 每个 token 从约 246 ms 降到约 91–107 ms。这说明融合本身有效果，但收益比原生开 Graph 小。
- **Mono 开 Graph 只再快约 5 个百分点。** Mono 每步本来只有 99 次 kernel 发射，Graph 能省的发射开销很少。
- **Mono 的完整精度验收还没通过**（和原生逐步对照时，部分步骤的分布差异超过门槛），下面的数字只代表性能。

## 测试配置

四种模式除了下表列出的差异，其余服务配置完全相同：BF16 KV、Triton attention / MoE、page 64、context 4096、chunked prefill 256、关闭 overlap 和 radix cache、关闭 MTP。

| 模式 | decode 路径 | decode Graph |
|---|---|---|
| 原生 | SGLang 原生 | 关 |
| 原生 + Graph | SGLang 原生 | 开（只捕获本次用到的 batch，不 padding） |
| Mono | Monokernel（每步 99 次发射） | 关 |
| Mono + Graph | Monokernel | 开（同上） |

请求：随机 token id 作为输入，`temperature=0`，`ignore_eos`；每轮前清空缓存，确认没有命中缓存。每组预热 1 次、正式测 3 次，取平均。

指标：
- **TTFT**：从发出请求到收到第 1 个 token。
- **TPOT**：(最后一个 token 时间 − 第一个 token 时间) / (输出数 − 1)。
- **E2E**：从发出请求到这一轮全部请求结束。
- **Decode tok/s**：batch × (输出数 − 1) / decode 时间。

## batch 1, 512→64，四种模式

| 模式 | TTFT ms | TPOT ms | E2E s | Decode tok/s |
|---|---:|---:|---:|---:|
| 原生 | 632 | 246.0 | 16.13 | 4.06 |
| 原生 + Graph | 660 | **45.5** | **3.53** | **21.96** |
| Mono | 633 | 94.7 | 6.60 | 10.56 |
| Mono + Graph | 588 | 90.1 | 6.27 | 11.09 |

## batch 1 全部长度

每格四个数的顺序是 **原生 / 原生+Graph / Mono / Mono+Graph**。

| 输入→输出 | TTFT ms | TPOT ms | E2E s | Decode tok/s | Mono+Graph 耗时 ÷ 原生+Graph 耗时（E2E） |
|---|---:|---:|---:|---:|---:|
| 16→1024 | 306 / 324 / 308 / 300 | 244.6 / 46.0 / 94.1 / 89.8 | 250.56 / 47.41 / 96.60 / 92.12 | 4.09 / 21.73 / 10.62 / 11.14 | 1.94× |
| 64→16 | 321 / 345 / 320 / 286 | 248.8 / 44.9 / 90.7 / 86.2 | 4.05 / 1.02 / 1.68 / 1.58 | 4.02 / 22.26 / 11.02 / 11.60 | 1.55× |
| 64→64 | 329 / 346 / 307 / 292 | 248.1 / 44.9 / 91.0 / 86.3 | 15.96 / 3.17 / 6.04 / 5.73 | 4.03 / 22.28 / 10.99 / 11.59 | 1.81× |
| 64→128 | 386 / 350 / 317 / 294 | 283.7 / 45.3 / 91.2 / 86.6 | 36.41 / 6.10 / 11.90 / 11.29 | 3.53 / 22.09 / 10.97 / 11.55 | 1.85× |
| 512→16 | 657 / 687 / 635 / 611 | 246.4 / 45.6 / 94.7 / 90.2 | 4.35 / 1.37 / 2.06 / 1.96 | 4.06 / 21.95 / 10.56 / 11.09 | 1.43× |
| 512→64 | 632 / 660 / 633 / 588 | 246.0 / 45.5 / 94.7 / 90.1 | 16.13 / 3.53 / 6.60 / 6.27 | 4.06 / 21.96 / 10.56 / 11.09 | 1.78× |
| 512→128 | 661 / 686 / 620 / 582 | 250.1 / 46.4 / 94.5 / 90.4 | 32.42 / 6.58 / 12.62 / 12.06 | 4.00 / 21.53 / 10.58 / 11.07 | 1.83× |
| 2048→16 | 2517 / 2767 / 2600 / 2260 | 240.3 / 49.0 / 108.4 / 102.4 | 6.12 / 3.50 / 4.23 / 3.80 | 4.16 / 20.41 / 9.22 / 9.77 | 1.09× |
| 2048→64 | 2520 / 2784 / 2517 / 2369 | 251.6 / 49.0 / 107.2 / 102.7 | 18.37 / 5.87 / 9.27 / 8.84 | 3.98 / 20.43 / 9.33 / 9.73 | 1.51× |
| 2048→128 | 2477 / 2713 / 2531 / 2340 | 247.6 / 48.8 / 107.3 / 102.9 | 33.93 / 8.92 / 16.16 / 15.40 | 4.04 / 20.47 / 9.32 / 9.72 | 1.73× |

## batch 2 全部长度

每格三个数的顺序是 **原生 / 原生+Graph / Mono**。Mono + Graph 还没测 batch 2。

| 输入→输出 | TTFT ms | TPOT ms | E2E s | Decode tok/s（2 路合计） | Mono 耗时 ÷ 原生+Graph 耗时（E2E） |
|---|---:|---:|---:|---:|---:|
| 16→1024 | 330 / 489 / 316 | 244.1 / 45.8 / 96.3 | 250.05 / 47.35 / 98.86 | 8.19 / 43.51 / 20.76 | 2.09× |
| 64→16 | 323 / 535 / 297 | 243.4 / 56.2 / 92.6 | 3.97 / 1.38 / 1.69 | 8.22 / 29.76 / 21.60 | 1.22× |
| 64→64 | 313 / 536 / 312 | 246.5 / 48.3 / 93.1 | 15.84 / 3.58 / 6.18 | 8.11 / 39.21 / 21.47 | 1.73× |
| 64→128 | 316 / 547 / 321 | 243.5 / 47.1 / 93.4 | 31.24 / 6.53 / 12.19 | 8.21 / 41.19 / 21.41 | 1.87× |
| 512→16 | 918 / 1049 / 934 | 265.1 / 67.9 / 117.6 | 4.90 / 2.07 / 2.70 | 7.01 / 22.23 / 14.46 | 1.30× |
| 512→64 | 947 / 1078 / 913 | 250.7 / 52.3 / 101.5 | 16.74 / 4.37 / 7.30 | 7.83 / 34.47 / 18.84 | 1.67× |
| 512→128 | 917 / 1039 / 940 | 249.5 / 49.3 / 99.5 | 32.60 / 7.30 / 13.57 | 7.94 / 38.53 / 19.64 | 1.86× |
| 2048→16 | 3767 / 3756 / 3751 | 333.0 / 131.1 / 193.8 | 8.76 / 5.72 / 6.66 | 4.79 / 9.32 / 7.20 | 1.16× |
| 2048→64 | 3740 / 3779 / 3628 | 269.9 / 67.5 / 128.8 | 20.75 / 8.03 / 11.74 | 6.90 / 22.86 / 13.53 | 1.46× |
| 2048→128 | 3819 / 3943 / 3677 | 258.0 / 57.9 / 119.3 | 36.58 / 11.29 / 18.83 | 7.47 / 29.33 / 15.52 | 1.67× |

batch 2 的两个请求同时发出，但 prefill 是一个接一个做的。前一个请求的 decode 会和后一个请求的 prefill 重叠，所以长输入、短输出时 TPOT 偏高（例如 2048→16）。

## 数据来源和注意事项

- 原生、Mono（都不开 Graph）：11-09 测的，两种模式共用同一个服务进程，输入完全相同。
- Mono + Graph：11-10，用的是同一份输入。
- 原生 + Graph：11-10，用 `bench_flash_next.py` 测的。随机输入和上面几组不同，长度相同；这对 decode 速度基本没影响，但 TTFT 不适合和上面几组逐个比较。
- 之前常提的“原生 Graph 约 24.7 tok/s”用的是 FP8 KV + FA3 + AITER，还开了 overlap 和 radix，和这里的 BF16 配置不同，所以没放进表里。
- 每组只测 3 次，单个数字有波动，不适合用来估 P95 / P99。
- 还缺 Mono + Graph 的 batch 2 数据。

## 怎么测

```bash
# 原生 / Mono，开或不开 Graph，batch 与输入输出长度可以给多个值
python3 bench_flash_next.py --mode native --graph --batch-size 1 --input-len 512 --output-len 64
python3 bench_flash_next.py --mode mono --graph --batch-size 1 2 --input-len 64 512 2048 --output-len 16 64 128
# 额外为每组采一个 Perfetto trace（会让 TTFT/prefill 偏高）
python3 bench_flash_next.py --mode mono --graph --profile --input-len 512 --output-len 64
```

测试开始前会检查 8 张卡是否空闲，有别人的任务在跑就直接报错退出（`--allow-busy-gpus` 可以跳过检查）。这台机器是共用的，**任何一张卡被别人占满，整个服务都会卡住**：每层都要做两次 8 卡通信，一张卡慢，其余 7 张就全在等它，看起来像没有负载地卡死。

`bench_flash_next.py` 全部参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode native\|mono` | 必填 | decode 走 SGLang 原生还是 Monokernel |
| `--graph` | 关 | 只为本次用到的 batch 捕获 decode graph（不 padding） |
| `--batch-size N [N ...]` | 1 | 每轮同时发出的请求数；Mono 只支持 1、2 |
| `--input-len N [N ...]` | 512 | 每个请求的输入 token 数 |
| `--output-len N [N ...]` | 64 | 每个请求的输出 token 数（`ignore_eos`） |
| `--repeats` | 3 | 每组正式测量轮数 |
| `--warmups` | 1 | 每组正式测量前的预热轮数 |
| `--seed` | 42 | 随机输入 token 和服务端的种子 |
| `--profile` | 关 | 正式测量跑完后，每组再采一份 torch-profiler trace（见下面用例 1） |
| `--profile-mono-only` | 关 | 配合 `--profile`，只导出 Monokernel 相关事件的精简 trace |
| `--url` | — | 连接已经在跑的服务，不另外启动（服务须与 `--mode` / `--graph` 一致） |
| `--port` | 30160 | 自己启动服务时用的端口 |
| `--model-path` | Flash-Next 快照 | 模型路径 |
| `--output-dir` | `bench_results/<时间>_<模式>_<graph\|eager>_...` | 结果目录 |
| `--keep-server` | 关 | 测完不关服务，方便再用 `--url` 接着测别的组合 |
| `--numa auto\|on\|off` | off | rank i 绑 NUMA 节点 i；别人的任务常占满 NUMA 0，绑上会让 rank 0 加载时被 OOM 杀掉，所以默认不绑 |
| `--strict-accuracy` | 关 | Mono：shadow 精度检查失败时拒绝请求，而不是只记录 |
| `--allow-busy-gpus` | 关 | 卡被占用时只警告，不退出 |
| `--startup-timeout` | 1800 | 等服务启动的秒数（加载模型约 4–5 分钟） |
| `-- <sglang 参数>` | — | `--` 之后的内容原样传给 `sglang.launch_server` |

想看每个融合 kernel 内部各阶段各用了多久（torch profiler 只能看到 kernel 的起止，看不到里面）：

```bash
SGLANG_AMD_FLASH_STAGE_TIMING=1 SGLANG_AMD_FLASH_STAGE_TIMING_STEP=5 \
  python3 bench_flash_next.py --mode mono --graph --input-len 512 --output-len 64 --output-dir <运行目录>
python3 tools/merge_flash_next_stage_timing.py --run-dir <运行目录>/runtime
```

只采指定的第 5 个 fused decode step（可改），并覆盖 kernel 入口到末尾的所有阶段；eager 和 decode graph 都支持。八张卡分别记录设备端时间，合并后导出 `<运行目录>/runtime/stage-timing.perfetto.json` 和各阶段耗时汇总。粒度到混音、投影、卷积、稀疏注意力选块、路由、专家、两处 all-reduce 等待等。
kernel 入口时间由 kernel 内的设备端标记记录；`layer gdn/qsa` 从 attention kernel 的入口标记开始，在对应的 `post` kernel 最后一个 barrier 结束，不再用下一个 attention kernel 的开始时间代替结束时间。核内阶段只画 block 0 到达下一个 grid barrier 之前的计算区间。block 0 在 barrier 里等其余 block 的时间（`block 0 grid wait`，不等于整卡空闲）从 11-15 起默认不放进合并后的 trace，所以时间轴上两段计算之间的空白就是等待；要看这些等待，合并时加 `--keep-idle`（见下文“设备端阶段计时与每 CU 计时”）。

要把每一步 decode 的核内阶段直接放进完整 torch-profiler trace（与 GPU kernel 同一时间轴），用全量采集工具：

```bash
python3 tools/profile_flash_next_serving.py --mode monokernel --kernel-stages \
  --input-tokens 512 --output-tokens 64 --batch-size 1 --output-dir <运行目录>
```

完整 trace 输出在 `<运行目录>/monokernel/inference.perfetto.json.gz`。工具会跳过预热，只对正式请求的每个 decode step 记录核内阶段；trace 采集会有额外开销，不用于性能验收。

结果默认写到 `bench_results/<时间>_<模式>_<graph|eager>/`。其中 `summary.md` / `summary.csv` 是汇总表，`results.json` 记录每个 token 的到达时间和服务配置，`profile/` 里是 trace。Mono 只支持 batch 1 和 2，输入加输出不能超过 4096。

已有的 Perfetto trace（batch 1，512→64）在 `profiles/` 下：原生、原生 + Graph、Mono 各一份，用 Perfetto UI 的 Open trace file 打开。这三份是 torch profiler 采的，带模型分层标记。

要采 hipprof 的 trace（HIP 层面的 kernel 和 API，比 torch profiler 更底层），用同一个采集工具加 `--hipprof`：

```bash
python3 tools/profile_flash_next_serving.py --mode native --hipprof \
  --input-tokens 512 --output-tokens 64 --batch-size 1 --output-dir <目录>
python3 tools/profile_flash_next_serving.py --mode native --hipprof --decode-graph ... # 原生 + Graph
python3 tools/profile_flash_next_serving.py --mode monokernel --hipprof ...            # Mono
```

加载模型和预热不会被采集，只采被测的那一次请求。输入 token 和 `profiles/` 下现有三份完全一致。结果是 `<目录>/<模式>/hipprof/<模式>.pftrace`，同样用 Perfetto 打开。

## 三个 profile 工具的区别（11-13 更新）

三个工具都走同一个合并核心 `merge_profiles`，但定位完全不同：要性能数字顺带 trace 用基准脚本；要干净可复现、带核内阶段的深度 trace 用专用采集工具；trace 已抓过只想换种方式重导出用离线重合并工具（不碰 GPU，秒级）。

| 工具 | 定位 | 服务器 | trace 数量 |
|---|---|---|---|
| `bench_flash_next.py --profile` | 性能基准跑完后顺手采 trace | 自己拉起或 `--url` 附加，配置即基准配置 | 每个 case 一份 |
| `tools/profile_flash_next_serving.py` | 专用单次全推理采集，支持 hipprof / 核内阶段 | 自己拉起（或 `--url` 附加），一次采集一个请求 | 每 mode 一份 |
| `tools/summarize_serving_profile.py` | 对已保存的原始 rank trace 离线重新合并 | 不启动，必须已有 `ranks/*TP-*.trace.json.gz` | 覆盖 / 重建 |

校验（decode 覆盖、8 rank 完整性、token 行数）三者在**完整** trace 上做，只有导出内容受 `mono_only` 影响。

### 用例 1：性能基准 + 顺带 trace（`bench_flash_next.py`）

先跑完全部 timed waves 拿到干净性能数字，然后每个 case 单独 `start_profile → 一轮 wave → stop_profile`。`with_stack` 固定关闭，`--profile` 会拖慢 eager 步（prefill 和不开 graph 的 decode 的 timed 数字也受影响）。

```bash
# 基准 + 每 case 一份完整 Perfetto trace
python3 bench_flash_next.py --mode mono --graph --profile \
  --batch-size 1 --input-len 512 --output-len 64

# 基准 + 精简 trace：只保留融合 kernel、核内阶段、inference/ 步进标记
python3 bench_flash_next.py --mode mono --graph --profile --profile-mono-only \
  --batch-size 1 2 --input-len 64 512 2048 --output-len 16 64 128
```

| 参数 | 说明 |
|---|---|
| `--profile` | timed 重复跑完后，每个 case 抓一份 torch-profiler trace（Perfetto `.json.gz`） |
| `--profile-mono-only` | 配合 `--profile`：导出精简版 `inference.mono_only.json.gz`，约 1% 事件量（b1 512→64 实测 313 MB → 2.3 MB）；丢弃 host / 前端 / client 事件 |
| `--url` | 附加到已在跑的服务（须已开 `SGLANG_AMD_PROFILE_RANGES=1`） |

输出在 `bench_results/<stamp>/profile/b{batch}_{in}-{out}/` 下，含 `inference.perfetto.json.gz`（或精简版）、`summary.json`、`ranks/`、`host/`。

### 用例 2：专用单次采集（`tools/profile_flash_next_serving.py`）

一次启动只服务一次采集，全程只包裹一个被测请求。可 `--mode both` 依次跑 native + monokernel 做对比。独有能力：`--hipprof`（HIP 层 `.pftrace`）、`--with-stack`、`--kernel-stages`（把核内阶段嵌入 torch trace 时间轴）、`--keep-python-functions`。额外校验：mode 与实际路由匹配、shadow 工作未污染请求、fused 路由 decode 计数与 trace 对齐、tokenize / detokenize host range 必须捕获到。写 `manifest.json` 记录源码 SHA256，保证可复现。

```bash
# Mono + decode graph + 核内阶段，精简导出
python3 tools/profile_flash_next_serving.py --mode monokernel --decode-graph --kernel-stages --mono-only \
  --input-tokens 512 --output-tokens 64 --batch-size 1 --output-dir <目录>

# native + monokernel 各采一份完整 trace 对比
python3 tools/profile_flash_next_serving.py --mode both \
  --input-tokens 512 --output-tokens 64 --output-dir <目录>
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode native\|monokernel\|both` | both | 采哪种模式；`both` 让 native 和 monokernel 各启动一次，输出到同一目录方便对比 |
| `--input-tokens` / `--output-tokens` | 512 / 64 | 被测请求的输入 / 输出长度 |
| `--batch-size 1\|2` | 1 | 被测请求数 |
| `--warmup` | 2 | 正式采集前的预热轮数（不采集） |
| `--decode-graph` | 关 | 只为被测 batch 捕获 decode graph |
| `--mono-only` | 关 | 合并时只导出 monokernel 相关事件 → `inference.mono_only.json.gz`（校验仍用完整 trace）；要求 torch-profiler 采集且 `--mode monokernel` |
| `--kernel-stages` | 关 | 把每个被测 decode step 的设备端阶段 span 嵌入 torch trace；要求自启动的 monokernel torch-profiler 采集 |
| `--cu-stages` | 关 | 同时记录每个 CU 在每个时刻跑的是什么（每 CU 一行），采完自动合并成 `inference.perfetto.cu_stages.json.gz`（加 `--mono-only` 时为 `inference.mono_only.cu_stages.json.gz`），见下文 |
| `--cu-steps N` | 1 | 配合 `--cu-stages`：记录几个 decode step，取值 1..输出数−1；每多一步每卡多约 66 MB JSON，建议 ≤ 4 |
| `--hipprof` | 关 | 改用 hipprof 在 HIP 层采集（`.pftrace`），与 `--mono-only` 互斥 |
| `--with-stack` / `--keep-python-functions` | 关 | 保留 python 调用栈 / python 函数事件 |
| `--url` / `--host-event-dir` | — | 附加到已开 `SGLANG_AMD_PROFILE_RANGES=1` 的服务；后者是该服务配置的前端事件目录 |
| `--port` | 30160 | 自己启动服务时的端口 |
| `--model-path` | Flash-Next 快照 | 模型路径 |
| `--output-dir` | `profiles/<时间>_<模式>_<选项>` | 输出目录 |
| `--resume` | 关 | 往已有目录里补采缺少的模式，不覆盖已有结果 |
| `--allow-busy-gpus` | 关 | 卡被占用时只警告，不退出 |
| `--startup-timeout` | 1200 | 等服务启动的秒数 |

输出在 `<目录>/{native,monokernel}/` 下，含 `inference.perfetto.json.gz`（或精简版）、`operators.csv`、`summary.json`、`manifest.json`、`README.md` 汇总表。

### 用例 3：离线重合并（`tools/summarize_serving_profile.py`）

对用例 2 保存的原始 rank trace 重新合并，不启动服务器、不加载模型。原始 `ranks/*TP-*.trace.json.gz` 一直保留在运行目录里，任何时候都能换一种合并方式重导出。

```bash
# 从已有运行目录重建精简版 trace（无需重新加载模型）
python3 tools/summarize_serving_profile.py --run-dir <目录>/monokernel --mono-only

# 重建带 python 函数事件的完整版
python3 tools/summarize_serving_profile.py --run-dir <目录>/monokernel --keep-python-functions
```

| 参数 | 说明 |
|---|---|
| `--run-dir` | 用例 2 的运行目录（须含 `capture.json` 和 `ranks/*TP-*.trace.json.gz`） |
| `--mono-only` | 重建成 `inference.mono_only.json.gz` 精简版 |
| `--keep-python-functions` | 重建成保留 python 函数事件的完整版 |

每次重导出会在 `<run-dir>/postprocess.json` 记录所用 `profile_trace.py` 的源码 SHA256 和选项，之前 summary 里的 shadow 状态、prompt 长度等字段会合并回去。

## 设备端阶段计时与每 CU 计时（11-15 更新）

torch profiler 只能看到融合 kernel 的起止。要看 kernel 内部，有两种设备端计时，都只对 Mono 生效，eager 和 decode graph 都支持：

- **阶段计时**（`SGLANG_AMD_FLASH_STAGE_TIMING=1`）：只有 block 0 打点，得到“block 0 何时到达每个 grid barrier”，每卡一行。
- **每 CU 计时**（再加 `SGLANG_AMD_FLASH_CU_TIMING=1`）：每个 block 都打点并记下所在的 CU，trace 里每个计算单元一行（`SE<se>.SH<sh>.CU<cu> (block <b>)`），能看出某一时刻每个 CU 在跑哪个算子。

### 采集

```bash
# 方式 A：跑基准时顺带采（第 5 个 decode step；只要阶段计时就去掉 CU_TIMING）
SGLANG_AMD_FLASH_STAGE_TIMING=1 SGLANG_AMD_FLASH_CU_TIMING=1 SGLANG_AMD_FLASH_STAGE_TIMING_STEP=5 \
  SGLANG_AMD_FLASH_REPORT_DIR=<运行目录>/runtime \
  python3 bench_flash_next.py --mode mono --graph --input-len 512 --output-len 64 --output-dir <运行目录>
python3 tools/merge_flash_next_stage_timing.py --run-dir <运行目录>/runtime

# 方式 B：专用采集工具，采完自动合并（环境变量、buffer 大小和 watchdog 都由工具设置）
python3 tools/profile_flash_next_serving.py --mode monokernel --decode-graph --cu-stages --cu-steps 2 \
  --input-tokens 512 --output-tokens 64 --output-dir <目录>
```

环境变量：

| 变量 | 说明 |
|---|---|
| `SGLANG_AMD_FLASH_STAGE_TIMING=1` | 打开 block 0 阶段计时 |
| `SGLANG_AMD_FLASH_CU_TIMING=1` | 打开每 CU 计时（与上一项共用时间基准，两组行在同一时间轴上对齐） |
| `SGLANG_AMD_FLASH_STAGE_TIMING_STEP` | 从第几个 decode step 开始导出；0（默认）不自动导出 |
| `SGLANG_AMD_FLASH_STAGE_TIMING_STEPS` | 导出几个 step |
| `SGLANG_AMD_FLASH_STAGE_TIMING_RECORDS` / `SGLANG_AMD_FLASH_CU_TIMING_RECORDS` | 记录 buffer 大小；CU 默认 8192 条/ block，约两个 step（每步每 block 约 3150 条），超了会在结果里标 `dropped` |
| `SGLANG_AMD_FLASH_REPORT_DIR` | 原始 `stage-timing-*-rank-*.json` / `cu-timing-*-rank-*.json` 的写出目录 |

注意事项：
- 一个 64 token 的请求只有 63 个 decode step，`--cu-steps` 最多填输出数 − 1，填满会什么都导不出来。
- 导出在调度进程里同步进行，步数多时很慢（32 步约 7 分钟、每卡 2.1 GB）。手动启动服务时要自己加大 `--watchdog-timeout`（工具会自动传 `max(300, 60·N)`），否则 SGLang 的 300 秒 watchdog 会把服务杀掉，客户端只看到 `Connection refused`。
- 每 CU 计时每步每卡约 25 万个 span，打开整个 trace 很吃内存，一次采 1–4 步就够。

### 合并（`tools/merge_flash_next_stage_timing.py`）

八张卡各用自己的时钟打点，合并时用跨卡 all-reduce 最后一个 `peer wait` 的释放时刻（每步约 99 对）拟合出每张卡相对 rank 0 的偏移和漂移，把所有行对齐到 rank 0 的时钟上。实测残差中位数约 0.3 us、最大 1.7 us，**跨卡相差 2 us 以内的差异当作噪声**。

**合并后的 trace 默认只保留计算区间**：每 CU 行上的 `wait ...`（barrier 等待）、`gap ...`（两次 launch 之间的空闲），以及阶段行上的 `block 0 grid wait` 都不导出，所以一行上的空白就是等待或空闲。这些等待只是不画出来，对齐、跨卡检查和汇总里的 compute / wait / gap 合计照样用它们计算。`merge_profiles` 嵌入 torch trace 的核内阶段（`--kernel-stages`）同样默认不带 `block 0 grid wait`。

有每 CU 数据时，合并结果还会在每个 rank 下增加 `grid (all CUs)` 轨道：对每个 `(step, layer, kernel, stage)`，用最早 block 计算开始到最晚 block 到达 barrier 的时间范围。它补足只记录 block 0 的 device-stage 算子行看不到其他 block 尾部工作的限制；例如 block 0 先到 barrier 后等待的时间，会落在 grid 范围行而不是 block 0 的计算行上。`args.blocks` 是参与该范围的 block 数，`median_arrival_us` 是从范围起点到 block 中位到达时间的偏移。用 `--no-grid` 可关闭该轨道；其范围只表示 block 到达 barrier，不包含 barrier 释放后的等待。

```bash
python3 tools/merge_flash_next_stage_timing.py --run-dir <运行目录>/runtime                # 默认：只有计算区间
python3 tools/merge_flash_next_stage_timing.py --run-dir <运行目录>/runtime --keep-idle    # 加回 wait / gap span
python3 tools/merge_flash_next_stage_timing.py --run-dir <运行目录>/runtime --steps 1-4    # 多步采集只保留其中几步
python3 tools/merge_flash_next_stage_timing.py --run-dir <运行目录>/runtime --no-grid     # 不生成 grid (all CUs) 轨道
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--run-dir` | 必填 | 含 `stage-timing-*-rank-*.json` 的目录 |
| `--output` | `<run-dir>/stage-timing.perfetto.json`（有 CU 数据时为 `.json.gz`） | 合并后的 Perfetto 文件 |
| `--summary` | `<run-dir>/stage-timing-summary.json` | 汇总：每阶段耗时、对齐结果（`alignment`）、每 CU 的 compute / wait / gap 合计（`compute_units`） |
| `--no-cu` | 关 | 有 `cu-timing-*` 也不合并每 CU 行 |
| `--steps 5` / `3-6` / `1,4-5` | 全部 | 每 CU 数据只保留这几个 decode step（从 1 开始数，阶段行全保留）；32 步只留 4 步约 253 MB、13 分钟 |
| `--gzip` | 关 | 没有 CU 数据时也输出 `.gz` |
| `--align fit\|offset\|none` | fit | 对齐方式：偏移 + 漂移 / 只偏移 / 不对齐（各卡保留原始时钟，只能比较时长） |
| `--keep-idle` | 关 | 把 barrier 等待和 launch 间空闲也导出成 span |
| `--no-grid` | 关 | 不生成基于每 CU 计算区间的 `grid (all CUs)` 轨道 |

终端会打印对齐结果、所有 CU 的计算 / 等待 / 空闲合计，以及最受 barrier 限制和最受计算限制的 5 个 CU。

在 Python 里调用时对应的开关是 `merge(..., keep_idle=True)` 和 `merge_profiles(..., keep_device_waits=True)`。

### 读 trace

- 每 CU 行上的 span 名形如 `layer 12 gdn: conv state`，`args` 里有 block、step、layer、kernel、stage、CU。
- 一个 block 在同一次 launch 内不会换 CU，但下一次 launch 可能被放到别的 CU 上，所以同一行会先后出现多个 block（汇总里的 `shared_units`），这是正常的；launch 内换 CU 的次数记在 `migrated_launch_count`，目前实测为 0。
- 每 CU 行是单独的 Perfetto 文件，不在 torch-profiler 合并 trace（`inference.perfetto.json.gz`）里。

### 设备自检（不加载模型，单卡约 1 秒）

改过 `kernel.cpp` 后跑一次，确认 CU 分布、打点顺序和有无丢失：

```bash
python3 tools/verify_flash_next_cu_timing.py --device 0 --launches 4
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--device` | 0 | 用哪张卡 |
| `--launches` | 4 | 跑几次 `hc_mix` |
| `--rows 1\|2` | 1 | decode 行数 |
| `--records` | 1024 | 每 block 的记录容量 |
| `--output-dir` | — | 写出 Perfetto trace 和报告 |

11-13 在 card 0 上：80 个 block 落在 80 个不同的 CU 上，每 block 36 个打点，没有丢失；`hc_mix` 每次 launch 每个 CU 约 23 us 在算、约 361 us 在 grid barrier 里等。
