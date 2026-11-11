# Qwen3.8-Flash-Next BF16：原生 vs Monokernel 性能

更新：2026-11-10。

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

主要参数：
- 测试开始前会检查 8 张卡是否空闲，有别人的任务在跑就直接报错退出（`--allow-busy-gpus` 可以跳过检查）。这台机器是共用的，**任何一张卡被别人占满，整个服务都会卡住**：每层都要做两次 8 卡通信，一张卡慢，其余 7 张就全在等它，看起来像没有负载地卡死。
- `--repeats` / `--warmups`：正式测量 / 预热次数。
- `--url`：连接已经在跑的服务，不另外启动。
- `--keep-server`：测完不关服务，方便再用 `--url` 接着测别的组合。
- `--` 之后的内容会原样传给 sglang 作为服务参数。

想看每个融合 kernel 内部各阶段各用了多久（torch profiler 只能看到 kernel 的起止，看不到里面）：

```bash
SGLANG_AMD_FLASH_STAGE_TIMING=1 SGLANG_AMD_FLASH_STAGE_TIMING_STEP=5 \
  python3 bench_flash_next.py --mode mono --graph --input-len 512 --output-len 64 --output-dir <运行目录>
python3 tools/merge_flash_next_stage_timing.py --run-dir <运行目录>/runtime
```

只采指定的第 5 个 fused decode step（可改），并覆盖 kernel 入口到末尾的所有阶段；eager 和 decode graph 都支持。八张卡分别记录设备端时间，合并后导出 `<运行目录>/runtime/stage-timing.perfetto.json` 和各阶段耗时汇总。粒度到混音、投影、卷积、稀疏注意力选块、路由、专家、两处 all-reduce 等待等。

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
