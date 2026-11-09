# Qwen3.8-Flash-Next：多输入 / 输出长度性能对比（2026-11-07）

本节为当前工作区源码的重新实测，未开启 profiler。8 张 gfx936、TP8、BF16、batch 1，输入 64 / 512 / 2048 token 与输出 16 / 64 / 128 token 做完整的 3×3 组合。每种模式每组预热 2 次、正式顺序测量 3 次，表中为正式请求的算术均值。两种模式各 27 次正式请求、18 次预热请求。下方旧日期、旧源码的结果保留作历史记录。

当前融合路径的端到端加速为 **1.498–2.490×**，decode 加速为 **2.129–2.532×**。512→64 从 **14.961 s** 降至 **6.477 s**，端到端 **2.310×**、decode **2.436×**。连续生成阶段原生约 **227–232 ms/token**，Monokernel 约 **90–107 ms/token**，与交互时感到输出更快相符。

**精度范围：本实例在首次预热中完成的一步 batch-1 影子检查，8 卡均为 `serving_shadow_passed`；本轮没有重新进行完整逐步精度验收。27 对同序号正式请求中，完整 output_ids 一致为 7/27，其中 512→16 和 512→64 各 3/3 一致。原生本身也有 7/9 组在重复请求间产生不同序列。因此下表表示当前执行路径的性能，不能视为完整精度验收通过后的性能结论。**

## 端到端、Prefill、Decode

单元格内成对的数值顺序均为 **原生 SGLang / Monokernel**。加速比 = 原生耗时 / Monokernel 耗时，大于 1 表示融合路径更快。Decode TPS =（输出 token 数 - 1）/ Decode 耗时，单位 token/s，不包含 prefill 和首 token。

| 输入→输出 token | E2E（s，原生 / Mono） | Prefill（ms，原生 / Mono） | Decode（s，原生 / Mono） | Decode TPS（token/s，原生 / Mono） | E2E 加速 | Decode 加速 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64→16 | 3.695 / 1.640 | 291.16 / 283.55 | 3.399 / 1.351 | 4.41 / 11.10 | 2.253× | 2.516× |
| 64→64 | 14.685 / 5.986 | 280.07 / 293.18 | 14.400 / 5.687 | 4.38 / 11.08 | 2.453× | 2.532× |
| 64→128 | 29.331 / 11.778 | 297.06 / 288.64 | 29.029 / 11.484 | 4.37 / 11.06 | 2.490× | 2.528× |
| 512→16 | 4.044 / 1.994 | 577.39 / 580.99 | 3.461 / 1.407 | 4.33 / 10.66 | 2.029× | 2.460× |
| 512→64 | 14.961 / 6.477 | 589.36 / 573.49 | 14.366 / 5.898 | 4.39 / 10.68 | 2.310× | 2.436× |
| 512→128 | 29.567 / 12.527 | 594.84 / 583.35 | 28.966 / 11.937 | 4.38 / 10.64 | 2.360× | 2.427× |
| 2048→16 | 5.895 / 3.934 | 2401.78 / 2328.09 | 3.487 / 1.600 | 4.30 / 9.38 | 1.498× | 2.180× |
| 2048→64 | 16.758 / 9.038 | 2329.95 / 2324.56 | 14.422 / 6.707 | 4.37 / 9.39 | 1.854× | 2.150× |
| 2048→128 | 31.143 / 15.946 | 2304.20 / 2398.52 | 28.833 / 13.541 | 4.40 / 9.38 | 1.953× | 2.129× |

Prefill 继续共用原生实现，随输入增加约从 0.29 s、0.58 s 升至 2.3–2.4 s；两种模式在本次样本内没有稳定的 prefill 加速。长输入、短输出时这部分占比更高，所以 2048→16 的端到端仅约 1.50×；输出加长后，decode 加速对总时间的贡献增加。2048-token 上下文下融合 decode 约 106.5–106.6 ms/token，也慢于短上下文的约 90 ms/token。

## 首 token、连续输出速度与吞吐

这里的 Decode TPS = 1 / 平均 TPOT；E2E 输出 token/s 使用全部输出 token 数除以正式请求总墙钟时间，包含 prefill，二者含义不同。仍按原生 / Monokernel 顺序。

| 输入→输出 token | TTFT（ms，原生 / Mono） | TPOT（ms/token，原生 / Mono） | Decode TPS（token/s，原生 / Mono） | E2E 输出 token/s（原生 / Mono） |
| --- | ---: | ---: | ---: | ---: |
| 64→16 | 296.08 / 289.12 | 226.60 / 90.07 | 4.41 / 11.10 | 4.33 / 9.75 |
| 64→64 | 285.30 / 298.59 | 228.56 / 90.28 | 4.38 / 11.08 | 4.36 / 10.69 |
| 64→128 | 302.36 / 294.51 | 228.57 / 90.42 | 4.37 / 11.06 | 4.36 / 10.87 |
| 512→16 | 582.80 / 586.70 | 230.75 / 93.79 | 4.33 / 10.66 | 3.96 / 8.02 |
| 512→64 | 594.78 / 579.17 | 228.03 / 93.62 | 4.39 / 10.68 | 4.28 / 9.88 |
| 512→128 | 600.68 / 589.29 | 228.08 / 94.00 | 4.38 / 10.64 | 4.33 / 10.22 |
| 2048→16 | 2408.14 / 2334.41 | 232.45 / 106.64 | 4.30 / 9.38 | 2.71 / 4.07 |
| 2048→64 | 2335.79 / 2330.84 | 228.92 / 106.47 | 4.37 / 9.39 | 3.82 / 7.08 |
| 2048→128 | 2310.35 / 2405.13 | 227.03 / 106.62 | 4.40 / 9.38 | 4.11 / 8.03 |

## 重复测量波动

每组只有 3 个正式样本，以下为样本标准差和观测范围，不用于估计生产 P95 / P99。

| 输入→输出 token | 原生 E2E 均值 ± 标准差（s） | Mono E2E 均值 ± 标准差（s） | 原生 E2E 最小 / 最大（s） | Mono E2E 最小 / 最大（s） |
| --- | ---: | ---: | ---: | ---: |
| 64→16 | 3.695 ± 0.028 | 1.640 ± 0.008 | 3.665 / 3.720 | 1.634 / 1.649 |
| 64→64 | 14.685 ± 0.097 | 5.986 ± 0.027 | 14.622 / 14.796 | 5.968 / 6.017 |
| 64→128 | 29.331 ± 0.067 | 11.778 ± 0.013 | 29.289 / 29.408 | 11.771 / 11.793 |
| 512→16 | 4.044 ± 0.042 | 1.994 ± 0.006 | 3.997 / 4.076 | 1.986 / 1.999 |
| 512→64 | 14.961 ± 0.053 | 6.477 ± 0.006 | 14.920 / 15.021 | 6.471 / 6.483 |
| 512→128 | 29.567 ± 0.139 | 12.527 ± 0.006 | 29.417 / 29.689 | 12.523 / 12.533 |
| 2048→16 | 5.895 ± 0.112 | 3.934 ± 0.037 | 5.798 / 6.017 | 3.910 / 3.977 |
| 2048→64 | 16.758 ± 0.134 | 9.038 ± 0.035 | 16.664 / 16.911 | 9.010 / 9.077 |
| 2048→128 | 31.143 ± 0.130 | 15.946 ± 0.051 | 31.021 / 31.280 | 15.894 / 15.997 |

## 配置与计时口径

- 模型：`/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master`，原始 BF16；TP8 / PP1，最大运行请求数 1。先测完原生矩阵，停止后再启动 Monokernel 测相同矩阵。
- 两边使用完全相同的 input_ids、随机种子 `42`、`temperature=0`、`ignore_eos=true`，严格生成指定数量的输出 token。每次请求前 flush_cache；正式请求均验证 cached_tokens=0、num_retractions=0、finish_reason=length。固定随机种子没有消除原生重复运行的输出差异。
- BF16 KV、page64、context4096、chunked prefill256、max total tokens4096、显存比例 0.90、Triton attention / MoE、PLE embedding 在 GPU。两边均关闭 prefill / decode Graph、overlap 和 radix cache，使用相同 metrics、cache report、逐 token streaming。性能结论限于这组启动配置。
- `SGLANG_AMD_PROFILE_RANGES=0`，未调用 profiler 采集接口；没有精度 trace。`SGLANG_AMD_FLASH_SHADOW_STEPS=1`、`SGLANG_AMD_FLASH_SHADOW_REARM_ON_PREFILL=0`、`SGLANG_AMD_FLASH_SHADOW_RECORD_FAILURES=1`，首步影子检查只在预热中执行。设备错误检查与每步同步保留。
- E2E：本机 HTTP 请求发出到流结束。输入预先分词并以 input_ids 发送，不包含客户端 prompt 构造和输入分词；包含请求传输、服务端调度、模型执行、采样、反分词及流式输出。
- Prefill：服务端 `prefill_finished_time - forward_entry_time`，包含产生首 token 的 forward；TTFT：客户端首 token 到达时间。
- Decode：客户端最后 token 到达时间减去首 token 到达时间，16 / 64 / 128 个输出对应 15 / 63 / 127 次 decode；TPOT = Decode / (输出 token 数 - 1)。各阶段采用不同的时钟边界，Prefill + Decode 不要求精确等于 E2E。
- 软件：SGLang `0.5.18+dtk2604.torch2110.2609201917.g2b8d2b`、Torch `2.11.0+dtk2604.2609201441.gf74cd7`；完整包版本见本轮 environment.json。原生使用当前启动脚本的默认算子配置，没有另做 autotune。

## 执行路径与完整性核对

原生基线的 server_info 中没有 Flash-Next Monokernel runtime。两边服务配置、随机种子、prompt SHA256、源码 SHA256 完全一致；测试结束后再次核对源码文件未发生变化。两边正式请求都没有与服务端已报告的首次编译告警重叠，首次编译和影子参考推理均留在预热阶段。

融合路径的正式阶段每卡累计 **1845 次 decode forward**，99 个主数学 kernel 的计数逐一匹配对应请求的 decode 步数，每卡共 **182655 次主数学融合发射**，8 卡合计 **1461240 次**。9 组 × 8 卡的 72 条执行审核全部通过，正式时间内局部 / 整模型影子检查数量均未增加，没有计入原生影子参考推理。99 为主数学融合核数量，辅助复制和服务端采样等仍按现有实现执行。

首次短输入预热的一步数值检查通过，不代表这 9 组全部 decode 步骤均完成严格精度验收。完整输出对比见 validation.json：Monokernel 每组 3 次输出均稳定，原生有 7 组的 3 次输出不完全相同；两边仅 7/27 对输出逐 token 完全一致。此前完整精度问题仍需独立验收。

原试用服务和本轮两种测试服务均已关闭，30160 端口已释放；结束时 8 张卡的 GPU 使用率和显存占用均为 0。所有源码和文档改动保留在工作区，未进行 Git 提交。

## 原始数据与复现入口

本轮目录：`monokernel_reports/flash_next_tp8/benchmark_matrix_20261107_111124/`。

- [native.json](monokernel_reports/flash_next_tp8/benchmark_matrix_20261107_111124/native.json) / [monokernel.json](monokernel_reports/flash_next_tp8/benchmark_matrix_20261107_111124/monokernel.json)：每次预热、正式请求、token 到达时间、output_ids、文本、服务 metrics、配置与源码指纹。
- [comparison.json](monokernel_reports/flash_next_tp8/benchmark_matrix_20261107_111124/comparison.json) / [summary.csv](monokernel_reports/flash_next_tp8/benchmark_matrix_20261107_111124/summary.csv)：9 组对比与可导入表格的数据。
- [validation.json](monokernel_reports/flash_next_tp8/benchmark_matrix_20261107_111124/validation.json)：长度 / 缓存 / 编译检查、72 条执行审核的汇总、逐组输出一致性观察。
- [matrix.json](monokernel_reports/flash_next_tp8/benchmark_matrix_20261107_111124/matrix.json)、[environment.json](monokernel_reports/flash_next_tp8/benchmark_matrix_20261107_111124/environment.json)、[source-status.json](monokernel_reports/flash_next_tp8/benchmark_matrix_20261107_111124/source-status.json)：工作负载、软件版本和原生接口指纹核对。
- [原生启动记录](monokernel_reports/flash_next_tp8/benchmark_matrix_20261107_111124/native/launch.json) / [融合启动记录](monokernel_reports/flash_next_tp8/benchmark_matrix_20261107_111124/monokernel/launch.json)：准确命令、环境变量和开始 / 停止时间；同目录包含各自 server.log，融合 runtime 目录保留 rank-0 至 rank-7 的报告。
- `tools/benchmark_flash_next_serving.py run` 新增 `--input-lengths 64 512 2048 --output-lengths 16 64 128`，仍默认单组 512→64；本轮使用 `--warmups 2 --repeats 3`。在对应服务启动后分别执行 native / monokernel 模式，compare 子命令按完整矩阵校对并汇总。
- 本轮指定 `--allow-unaccepted-shadow`，仅允许客户端保留未通过精度状态时测量速度，不改变 kernel 或采样算法；默认仍要求影子检查通过。本实例实际首步影子检查通过，性能报告本身不宣称完整精度验收。

---

# Qwen3.8-27B 历史测试数据

以下保留原有数据，表内顺序为原生 / Monokernel。

| 并发 | 输入→输出 token | 端到端（s） | Prefill（ms） | Decode（s） |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 64→16 | 1.205 / 3.692 | 148.72 / 149.51 | 1.051 / 3.538 |
| 1 | 512→64 | 4.597 / 15.437 | 165.47 / 180.66 | 4.427 / 15.251 |
| 1 | 2048→128 | 9.375 / 33.568 | 437.50 / 438.84 | 8.932 / 33.124 |
| 2 | 64→128 | 6.374 / 77.138 | 147.71 / 156.15 | 6.146 / 76.897 |

# Qwen3.8-Flash-Next：TP8 / gfx936，batch 1，512→64

最新 [精度验收矩阵](Qwen3.8-Flash-Next%20完整精度验收.md)已覆盖 batch 1/2 的 7 个输入长度组合、472 个融合 decode forward，7 组均未通过全部门槛。已定位固定输入下 HC 最后混合的舍入差异，并保留长输入参考重复不一致及独立服务对照。下文性能值为历史源码结果。

日期：2026-11-06。本次使用下载的原始 BF16 Flash-Next 模型，8 张 gfx936，固定输入 512 token、输出 64 token，每种模式先预热 2 次，再顺序测量 5 次。下表为正式测量的算术均值，预热不计入。

**后续正确性补测：两条路径的 13/13 个标准答案及完整 output_ids 一致，但 512→64 的严格逐步对照在第 3 个 decode 步骤失败；同位置原生重放也存在 greedy 分歧。继续诊断完成 39 步（37 步通过），第 40 步被第 47 层 QSA 的局部误差保护终止，未完成全部 63 步。以下保留原始性能实测，尚不能作为通过完整精度验收后的加速结论。详见 [正确性验证](Qwen3.8-Flash-Next%20正确性验证.md)。**

后续 [精度定位与修复](Qwen3.8-Flash-Next%20精度问题定位.md)已对齐 HC combine 归约和短上下文 QSA 顺序。真实 combine 失败样本已逐位一致，但新服务仍在第 33 层因 HC / MoE 分歧被保护终止；原生固定输入重放也复现专家切换。修复后的完整数值验收未通过，下表属于修复前源码，**不能用于评估当前修改后的性能**。

| 模式 | 端到端（s） | Prefill（ms） | TTFT（ms） | Decode（s） | TPOT（ms/token） | 输出吞吐（token/s） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原生 SGLang | 15.1831 | 594.95 | 600.76 | 14.5822 | 231.46 | 4.22 |
| Monokernel | 7.5341 | 602.56 | 607.98 | 6.9260 | 109.94 | 8.49 |

在本次配置和输入输出长度下，Monokernel 的**端到端加速比为 2.015×，decode 加速比为 2.105×**，对应耗时分别减少 50.38% 和 52.50%。Prefill 共用原生 SGLang 路径，本次均值相近。加速比定义为原生耗时 / Monokernel 耗时。

| 模式 | 端到端均值 ± 样本标准差（s） | 端到端最小 / 最大（s） | Prefill 标准差（ms） | Decode 标准差（s） |
| --- | ---: | ---: | ---: | ---: |
| 原生 SGLang | 15.1831 ± 0.2256 | 14.9315 / 15.5223 | 21.68 | 0.2134 |
| Monokernel | 7.5341 ± 0.0317 | 7.5014 / 7.5850 | 25.35 | 0.0149 |

## 配置与计时口径

- 两种模式依次使用同一组 8 卡、同一份模型、相同的 512 个 input_ids；单请求、最大运行请求数为 1。脚本核对两边服务配置、输入 SHA256、源码 SHA256 和测量参数一致。
- TP8 / PP1、BF16 权重和 KV、Triton attention / MoE、page64、context4096、chunked prefill256、max total tokens4096、PLE embedding 在 GPU、显存比例 0.90。
- 两边均关闭 decode / prefill Graph、overlap 和 radix cache；开启相同的 metrics、cache report 和逐 token streaming。原生使用当前 Triton MoE 默认配置；日志提示缺少该卡的专用调优文件，本次未另做 autotune。
- `temperature=0`、`ignore_eos=true`，严格生成 64 个 token。每个请求前 flush_cache，并断言 `prompt_tokens=512`、`completion_tokens=64`、`cached_tokens=0`，没有请求回退重排。
- 端到端：客户端本机 HTTP 发起至流结束；Prefill：服务端 `prefill_finished_time - forward_entry_time`，含首 token；TTFT：客户端首 token 到达时间。
- Decode：客户端最后一个 token 到达时间减去首个 token 到达时间，对应 63 次 decode；TPOT = Decode / 63。阶段时间来自不同计时边界，和不严格等于端到端。这些都是请求级墙钟时间，不是纯 GPU event 时间。
- 输出吞吐：全部正式输出 token 数 / 5 轮请求的总墙钟时间；5 次样本用于本次配置比较，不代表生产 P95 / P99。

## 融合路径与影子开销核对

Monokernel 设置 `SGLANG_AMD_FLASH_SHADOW_REARM_ON_PREFILL=0`、`SGLANG_AMD_FLASH_SHADOW_STEPS=1`，首次 batch-1 影子数值验收在预热阶段完成。正式计时前后各卡的局部和整模型影子检查数量均未增长，**正式时间不包含原生参考推理**。每步设备同步和错误保护仍保留。

8 卡正式阶段各累计 **315 次 decode forward**，99 个主数学 kernel 的计数增量逐一等于 315，每卡共 31185 次数学融合发射。原生基线未挂载 Flash-Next Monokernel runtime。此次只测 batch 1、512→64，不推断其他 batch / 长度的加速效果。

输出序列也保留作核对：原生 5 轮出现 4 种 64-token 序列，Monokernel 5 轮序列一致，且该序列出现在原生结果中。原生 HC 的非确定性与数值验收口径见 Flash-Next 接入进度文档；此处比较相同输入和固定输出量的时延与吞吐，完整序列与文本都保存在原始 JSON 中。

## 原始记录

目录：`monokernel_reports/flash_next_tp8/benchmark_b1_512_64_20261106/`。

- `native.json` / `monokernel.json`：每次请求的全部 64 个 token 到达时间、输出 token、服务 metrics、服务配置与源码指纹。
- `comparison.json` / `comparison.log`：汇总、加速比及八卡执行路径审核。
- `launch.json`、`native/server.log`、`monokernel/server.log`：启动命令、环境和服务日志。
- `monokernel/rank-*.json`：八卡数值验收和执行计数；`source-status.json`：36 个原生接口指纹检查结果。
- `restored-verification.json` / `restored-status.json`：切回 Monokernel 后的额外 1 次预热、2 次请求及健康验证，不计入上面的 5 次性能均值。
- `tools/benchmark_flash_next_serving.py`：本次脚本，复用 `tools/benchmark_qwen38_serving.py` 的输入构造、逐 token 流式计时与阶段统计。

源码改动保留在工作区，未暂存、未提交。性能测试结束时曾切回相同 batch-1 配置的 Monokernel 服务，恢复记录见该目录的 `restored-status.json`；该实例已在后续正确性补测前停止。正确性补测的严格 / 诊断实例也均因验收失败退出，当前服务状态见正确性报告。
