# Qwen3.8-Flash-Next TP8 / gfx936 精度验收

日期：2026-11-06。模型使用本机下载的原始 BF16 权重，TP8，8 张 gfx936。源码改动未暂存、未提交。

**本轮计划矩阵已全部执行完，验收结论：不通过。** 默认原生、确定性参考基线和融合路径的 13 项标准答案均通过，7 个固定长度 / batch 组合均未满足完整数值门槛。所有原有门槛保持不变；诊断模式只让实验继续收集数据，超限仍记为失败，八卡服务状态均锁存为 `serving_shadow_failed`，`whole_decoder_monokernel=false`。三条测试服务均已关闭，30160 端口已停止监听。

## 显式参考策略

此前原生 HC 的 atomic split-K 会因累加顺序改变 BF16 舍入，进而改变专家路由；长上下文 QSA 的原生 TopK 也会在并列分数下改变选择。因此本轮增加可选策略 `fixed_hc_split256_ascending_sum_stable_qsa_topk_v1`，同时用于基线服务和融合服务中的原生参考：

- HC 保持原生 256 元素分块和 Triton BF16 dot，独立保存 40 个 FP32 部分和，再按分块索引升序相加。后续激活、up projection、sigmoid 和分支归约仍由独立 Triton 实现完成。
- QSA 分块数不超过 512 时按时间顺序返回；超过 512 时按分数降序选择，并列分数按块索引升序处理。
- 通过 `SGLANG_AMD_FLASH_ACCURACY_REFERENCE=1` 显式启用，默认关闭。安装目录中的 SGLang 文件未改写，策略由插件在进程内安装。

这项策略保留数学语义并规定浮点累加与并列选择顺序，**不等于未经修改的原生 SGLang**。策略、源码指纹和服务配置写入每份报告。它不是默认生产配置，也不用于报告历史性能数值。

固定的只是 HC 与 TopK 两处规则，不能预先保证整个参考 forward 确定。本轮逐步重复检查发现：batch 1 的 2048→64 和 3072→64 各有 6/63 步参考重放不一致，最大 logits 绝对误差分别为 `11.703125`、`8.4375`；其他 5 个长度 / batch 组合的参考重放均为零误差。这些位置仍判失败，尚未区分其他原生算子的波动与影子协议未纳入快照的状态，不能把不稳定位置的全部 KL / TV 偏差直接归因于融合数学核。

原先第 33 层会切换专家的固定输入，启用该策略后连续 16 次原生重放逐位一致，最后一个专家固定为 110。长上下文 QSA 检查覆盖真实第 3/47 层、TP rank 2/7、batch 1/2、输入 2052/4093、eager 与改变输入的 Graph 重放，共 64 项通过，最大输出 NRMSE 为 `0.00217408198`。这是算子级检查，不是整模型通过证明。

## 验收方法与门槛

环境版本：SGLang `0.5.18+dtk2604.torch2110.2609201917.g2b8d2b`，Torch `2.11.0+dtk2604.2609201441.gf74cd7`，Triton `3.6.0+dtk2604.torch2110.2609202033.gee450e`，HIP 编译器 `dcc 25.10.0-0`。工作区基于 Git HEAD `f5c842b1db8a3c7ae332c8daee3c25ebdc4bdb73`，实际改动源码逐文件 SHA256 保存在报告中。

服务配置一致：BF16、context 4096、page 64、max-total-tokens 4096、max-running-requests 2、chunked-prefill 256；关闭 Graph、overlap 和 radix cache。每个请求前清空缓存，检查缓存命中与请求回退均为零。所有请求 `temperature=0`；固定长度生成使用 `ignore_eos=true`。

每个融合 decode 步骤进行以下检查：

1. 在相同输入与缓存下运行完整原生参考，额外重复 3 次，再恢复融合路径的 123 个持久状态张量和 logits。
2. 检查 logits 有限、`KL(P_reference || P_fused) ≤ 0.01`、`TV ≤ 0.05`，并要求每行 greedy argmax 一致。原始 logits NRMSE 仅报告，不替代这些门槛。
3. 保留 GDN、QSA、PLE、HC、MoE、embedding 和 tail 的局部检查及原有门槛；每次更新检查相关缓存与循环状态。
4. 在全部 8 个 rank 上，将服务返回的 token、每步输入 token、上下文长度、参考检查以及 99 个数学核的执行计数绑定核对。

融合运行累计 472 个 decode forward，合计 661 个 decode token 行；八卡共 3776 份逐步整模型检查，实际数学核计数为 `472 × 8 × 99 = 373824`。这里的 forward 数把 batch 2 的一次双行执行计为一步。全部请求都完成了计划生成长度，执行覆盖成立；数值通过与执行覆盖分别判定。

启动时每卡的 96 项真实权重 HC / post 检查、72 项 GDN 检查和 12 项通信检查均通过，共 1440 项。局部算子门槛通过不能代替完整模型检查。新增 CPU 回归共 53 项通过，覆盖严格失败锁存、诊断不能改判通过、trace 输入和历史前缀绑定等行为。

另外启动独立基线服务，让两边自行生成并累计缓存，保存实际外层 forward 的全词表 logits，排除影子重放。两边输入 token、长度、模式和行数一致时才做分布比较；自由生成发生分歧后的不同输入不作数值等价比较，仍判该用例失败。该对照通过 logits 检查独立缓存累计的影响，没有直接比较两台独立服务的缓存张量。

再单独运行**默认原生 HC / TopK 策略**的服务，关闭融合与参考策略，复测同一批自由生成请求、标准答案和完整 output_ids。这项补测不抓取逐步 logits，和前述确定性基线的分布对照分别报告。

## 严格运行的首次失败

13 项问答均给出标准答案。接着执行 batch 1、64→64，在第 2 个 decode 步骤被严格门槛拒绝，对应上下文长度 66、全局零起始 step 32：

| 指标 | 实测 | 门槛 | 结果 |
| --- | ---: | --- | --- |
| logits 有限 | 是 | 必须有限 | 通过 |
| KL | 0.01852586493 | ≤ 0.01 | 失败 |
| TV | 0.06819564104 | ≤ 0.05 | 失败 |
| greedy token | 两边均为 271 | 必须相同 | 通过 |
| 3 次原生重复的最大绝对误差 | 0 | 必须逐位一致 | 通过 |

该步局部检查全部处于原有容差内，但跨层误差会放大。rank 0 的层输出 NRMSE 在第 0/1/2 层分别为 `0.000113736`、`0.001311477`、`0.003134373`，第 3 层增至 `0.064168274`。稳定参考的 3 次重放为零误差，因此这一处融合偏差不能再归因于参考随机波动。当前证据定位了误差放大的位置，尚未证明全部差异来自某一个算子。

证据：`monokernel_reports/flash_next_tp8/accuracy_20261106/fused-acceptance.json` 与 `fused_tp8_b2/rank-*.json`。

## 固定输入下的新增定位

完整诊断矩阵捕获到全局 step 96 的第 14 层局部失败。固定相同 post-attention 输入，第 14 层的 HC 原生参考连续 16 次逐位一致，选择的第十个专家为 **333**；融合 HC 连续 3 次也稳定，但选择第十个专家 **191**。该位置融合路由第十与第十一候选的 BF16 router logits 相同，不能用仅有“整体误差很小”解释为验收通过。

进一步使用独立 Triton 分段重建，逐项得到：

| 检查 | NRMSE | 最大绝对误差 |
| --- | ---: | ---: |
| HC normalized | 0 | 0 |
| HC down / SiLU 后 lowrank | 0 | 0 |
| HC up 后 sigmoid 前 raw gates | 0 | 0 |
| 用融合 raw gates 做 Triton sigmoid / 分支归约，对原生参考 | 0 | 0 |
| 融合最终 HC mixed，对上述相同 gates 的 Triton 结果 | 0.00014169145 | 0.00390625 |
| 由专家选择变化放大的 MoE 输出 | 0.1014478505 | 0.0087890625 |
| 最终层输出 | 0.01164825913 | 0.0009765625 |

这份样本将 HC 的差异明确缩小到**最后的 sigmoid / 分支归约**；normalized、down、up 在该样本上均已逐位匹配。原有层输出门槛为 `0.006`，故该层必须判失败。这里没有修改生产数学核或调整门槛，当前矩阵仍保持同一份计算源码。

原始输入已单独冻结为 `accuracy_20261106/hc14-captured-step96-rank0.pt`，SHA256 为 `77fde797a73544bf6db158ae0b7aed758359b381575cdedba273006ffa8770fa`；分段对照见 `hc14-stages.json`，路由与 16 次重放见 `hc14-reference.json`。

另做 9 种独立 HIP 组合检查：当前 `expf + 顺序求和` 能复现差异；保持 `expf`、将四分支改为 `(p0+p1)+(p2+p3)` 后，该样本逐位匹配；改用 exp2 的 6 种组合在此样本也匹配。证据为 `hc14-stages-variants.json`。该单样本无法证明候选改法对全部输入成立，尚未据此改写生产 kernel 或宣称精度问题已修复。

## 完整矩阵

运行目录：`monokernel_reports/flash_next_tp8/accuracy_20261106/matrix_v2/`。`summary.json` 为机器可读汇总：`completed_all_cases=true`、`passed=false`。三个服务各自完成 20 组请求，其中 batch 2 的每组含两条序列。

下表每组均完整生成 64 token，对后续 63 个 decode forward 全部检查。全部门槛包含八卡局部检查、整模型分布、greedy 和三次参考重复一致；batch 2 的 greedy 一致要求两行同时一致。步数不是任务准确率。2048 / 3072 输入的参考重复不稳定，不能把这两行的全部偏差视为融合独有误差。

| Batch | 输入→输出 / 每请求 | 同状态全部门槛通过步数 | 最大 KL | 最大 TV | greedy 一致步数 | 参考重复稳定步数 | 最终结果 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 64→64 | 56/63 | 0.01883387 | 0.06819564 | 62/63 | 63/63 | 失败 |
| 1 | 512→64 | 58/63 | 0.08394501 | 0.09155983 | 63/63 | 63/63 | 失败 |
| 1 | 2048→64 | 55/63 | 7.03116083 | 0.98020035 | 60/63 | 57/63 | 失败 |
| 1 | 3072→64 | 56/63 | 0.23968104 | 0.22727503 | 63/63 | 57/63 | 失败 |
| 2 | 64→64 | 58/63 | 0.01834393 | 0.08112230 | 63/63 | 63/63 | 失败 |
| 2 | 512→64 | 52/63 | 2.89814472 | 0.93063784 | 61/63 | 63/63 | 失败 |
| 2 | 1536→64 | 60/63 | 0.01585038 | 0.07358050 | 62/63 | 63/63 | 失败 |

加上 13 项有标准答案的短生成，本轮共 426/472 个 forward 同时满足全部门槛；以逐用例全部通过为要求，七组长生成均失败，不能用平均值改判。

## 独立生成与缓存累计对照

默认原生服务关闭参考策略与融合；确定性基线和融合服务共同启用前述 HC / TopK 策略。三者的 13 个标准答案和完整 output_ids 均一致，包含结束 token。

| Batch | 输入→输出 / 每请求 | 默认原生 output_ids 完全一致 | 确定性基线 output_ids 完全一致 | 确定性基线 prefill 最大绝对误差 | 同前缀分布通过 / 可比帧 / 全部帧 |
| ---: | --- | --- | --- | ---: | --- |
| 1 | 64→64 | 否 | 否 | 0.0 | 17/21/64 |
| 1 | 512→64 | 是 | 是 | 0.0 | 58/65/65 |
| 1 | 2048→64 | 否 | 是 | 0.0 | 69/71/71 |
| 1 | 3072→64 | 否 | 否 | 0.0 | 37/38/75 |
| 2 | 64→64 | 否 | 否 | 0.0 | 3/4/64 |
| 2 | 512→64 | 否 | 否 | 0.0 | 6/7/67 |
| 2 | 1536→64 | 否 | 是 | 0.0 | 66/75/75 |

独立确定性基线的 13 个标准答案与融合输出逐 token 一致，全部实际 forward 的分布对照也通过。七组固定长度生成中，prefill 的实际 logits 均逐位一致；decode 均未满足完整门槛。其中 batch 1 的 512/2048→64、batch 2 的 1536→64 虽然完整 output_ids 一致，同前缀 logits 的 KL / TV 仍会越界；其他四组在自由生成中出现 token 分歧。各组实际 forward 帧数均匹配，分歧之后不再把不同历史前缀的 logits 当成同一输入来比较。

纳入独立服务用例对照的实际 forward 共 531 帧，两边 trace 各有 533 帧，排除用例前的 2 个探测帧。其中 59 个 prefill 帧全部逐位一致；共有 331 帧完整输入前缀一致，306 帧通过分布门槛，其余 200 帧在自由生成已发生历史分歧后不作同输入比较。分块 prefill 的中间帧也保存 logits，因此某些长度会超过 64 帧。默认原生对照只核对输出，没有替换为确定性基线的 logits 数值结论。

## 局部检查汇总

下表汇总全部 8 个 rank 的 serving shadow。均为有限数。缓存 NRMSE 为整张量指标，可能稀释少量活动位置的误差，不能视为逐位一致或独立服务缓存等价的证明。

| 项目 | 检查次数 | 最大 NRMSE | Serving 门槛 / 解释 |
| --- | ---: | ---: | --- |
| embedding | 3776 | 0 | 要求逐位一致，通过 |
| GDN 输出 | 135936 | 0.0059625912 | ≤ 0.006，通过 |
| GDN conv 状态 | 135936 | 0.0009029463 | ≤ 0.006，通过 |
| GDN SSM 状态 | 135936 | 0.0026438753 | ≤ 0.006，通过 |
| QSA 输出 | 45312 | 0.0098070344 | 超过 0.008，失败 |
| QSA key / value / ring / compressed | 各 45312 | 0.0002450408 / 0.0001441693 / 0.0007398823 / 0.0015059419 | ≤ 0.008；rope positions 误差为 0 |
| post-attention TP reduction / HC combine | 各 181248 | 0 / 0 | 当前对照逐位一致 |
| HC / MoE 层输出 | 181248 | 0.0168789271 | 超过 0.006，失败 |
| reduced MoE 中间值 | 181248 | 0.2497346699 | 诊断值；局部层输出与整模型门槛另行判定 |
| PLE 输出 / conv | 各 3776 | 0.0008667580 / 0.0008950519 | ≤ 0.008，通过 |
| tail mixed / logits | 各 3776 | 0.0003499877 / 0.0008356138 | ≤ 0.004，通过 |

QSA 最大误差位于全局 step 223、第 47 层、rank 3；HC / MoE 层输出最大误差位于 step 321、第 19 层、rank 0。原始失败输入、参考中间值及对应二进制指纹保留在报告目录中。故剩余工作既包括 HC 最后混合的数值对齐，也包括 QSA 输出和长输入参考协议的进一步定位；不能将全部失败归因于已发现的单个 HC 样本。

QSA 的 step 223 对应上下文 3076：融合与原生的 mixed 输入逐位相同，用融合 mixed 再独立调用原生 attention，输出 NRMSE 仍为 `0.0098070344`，超过 `0.008`。因此这一处 QSA 差异不能只归因于前置 HC。对应冻结输入为 `matrix_v2/monokernel/qsa-step-223-layer-47-rank-3.pt`，元数据与输入 SHA256 见 `accuracy_20261106/qsa223-capture-summary.json`。

该次双方均选择 2048 个 token，但集合不同：融合多选了块 549（token 2196–2199），原生多选了块 579（token 2316–2319）。因此还须定位压缩 / index-score 数值差异如何改变选集，不能把这份样本全部解释为 sparse softmax 归约误差。固定并列选择规则并没有让不同的输入分数自动相同。

## 复现与原始证据

```bash
python3 tools/run_flash_next_accuracy_matrix.py \
  --output-dir monokernel_reports/flash_next_tp8/accuracy_new_run

python3 tools/verify_flash_next_default_accuracy.py \
  --report-dir monokernel_reports/flash_next_tp8/accuracy_new_run

python3 tools/summarize_flash_next_accuracy.py \
  --report-dir monokernel_reports/flash_next_tp8/accuracy_new_run \
  --json monokernel_reports/flash_next_tp8/accuracy_new_run/summary.json \
  --markdown monokernel_reports/flash_next_tp8/accuracy_new_run/tables.md

python3 -m pytest -q \
  test/unit/test_qwen38_flash_next_gfx936.py \
  test/unit/test_flash_next_accuracy_trace.py
```

第一条命令的输出目录必须尚不存在，以防不同源码或不同实验的 trace 混合；后续补测与汇总读取该目录。矩阵执行器启动自己的两个服务，收集结束后关闭服务，保存启动环境、PID、源码和输入指纹。诊断矩阵结束不表示验收通过，应检查报告中的 `passed` 与各项门槛。

默认原生补测工具随后再启动和关闭一个服务。当前执行器在数值验收失败时返回非零退出码；即便前一条命令失败，也应继续执行其余采集与汇总命令，以得到完整失败范围。

| 文件 | 用途 |
| --- | --- |
| `accuracy_20261106/hc33-reference.json` | 固定 HC 输入的 16 次确定性重放 |
| `accuracy_20261106/qsa-long-reference.json` | 64 项长上下文算子验证 |
| `matrix_v2/launch.json` | 本轮服务命令、环境和启动/停止时间 |
| `matrix_v2/monokernel-batch1.json`、`monokernel-batch2.json` | 逐步同状态对照、局部失败、原生重复与八卡执行审计 |
| `matrix_v2/native-batch1.json`、`native-batch2.json` | 独立基线的实际生成 |
| `matrix_v2/comparison-batch1.json`、`comparison-batch2.json` | 独立累计缓存后的输出与实际 logits 对照 |
| `matrix_v2/native-default-batch1.json`、`native-default-batch2.json`、`comparison-default.json` | 默认原生策略的独立输出对照，无参考策略替换 |
| `matrix_v2/{native,monokernel}/trace/` | 实际 forward 的 logits 与输入元数据 |
| `matrix_v2/summary.json`、`tables.md` | 完整性、验收布尔值、逐组表格与局部指标汇总 |

上述相对路径以 `monokernel_reports/flash_next_tp8/` 为根。旧版 [性能结果](testresult.md)未重新测量，不能当作当前源码通过精度验收后的性能结论。本文的有答案样本与长度矩阵也不替代 PPL、通用任务质量、长于 4096 的上下文、其他 batch 或采样参数验收。
