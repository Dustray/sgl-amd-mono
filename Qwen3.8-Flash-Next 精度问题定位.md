# Flash-Next TP8 / gfx936 精度问题定位

日期：2026-11-06。模型为下载的原始 BF16 Flash-Next，TP8、8 张 gfx936。本文记录对上一轮正确性失败的定位及修复；原始失败记录仍见 [正确性验证](Qwen3.8-Flash-Next%20正确性验证.md)。修改未暂存、未提交，未修改安装的 SGLang 源码。

**结论：已确认并修复 HC combine 和 QSA 读取顺序两处实现差异，但完整精度验收仍未通过。** HC combine 的真实失败样本已修到逐位一致；修复后的服务仍在另一位置触发原生 HC 不确定性与 MoE 路由分歧，不能用单个样本的修复代替完整验收。长上下文另有原生 TopK 并列选择的不确定性。

## HC combine：微小差异触发 MoE 专家切换

只修复 QSA 顺序后，直接发送 512→64 请求，在第一个 decode 的第 15 层被局部保护终止。8 卡均保存了失败张量，以下为 rank 2：

| 阶段 | 修复前 NRMSE | 最大绝对误差 |
| --- | ---: | ---: |
| HC combine 后的状态 | 0.0000388268 | 0.0000610352 |
| MLP HC mixed 输入 | 0.0011798029 | 0.0078125 |
| TP 规约后的 MoE 输出 | 0.2145180106 | 0.0146026611 |
| 整个 post-attention 输出 | **0.0088131633** | 0.0010347366 |

最后一项超过原有 `0.006` 门槛。NRMSE 是张量误差指标，不是任务准确率。

对保存的真实输入、真实权重进行独立回放发现：

- 固定同一份原生 post-attention 状态，原生 HC mix 连续重放 16 次均逐位一致；融合 HC mix 重放 3 次也与该原生结果逐位一致。该样本的差异源头在前面的 **HC combine**，不能全部归因于原生 HC 的 atomic split-K 波动。
- 捕获的融合 mixed 选择专家 `422,489,357,78,381,272,208,238,286,367`；原生 mixed 选择 `422,489,357,78,381,272,208,238,488,286`。两边第 10 / 11 名 router logits 的间距均为零，边界十分敏感。
- 在相同融合 mixed 输入上，融合 / 原生 router 与专家选择一致。MoE 输出的大幅差异由上游 mixed 输入变化导致。

原生 `hc_combine.cuh` 按 256 个参考线程分配数据，每线程累加 5 个 8 元素向量，采用 32-lane XOR 归约，再按固定顺序相加 8 份 partial。原实现使用 MMAC GEMV 计算 injection gate，FP32 累加顺序不同。

`kernel.cpp::inject` 已按上述原生顺序计算 gate，并显式使用 FP32 FMA 完成残差注入。单独改 FMA 未解决该样本；对齐归约顺序后结果如下：

| 真实失败样本回放 | 修复后结果 |
| --- | --- |
| HC combine 输出对原生 | NRMSE = 0，最大绝对误差 = 0 |
| HC mix 输出对原生 | NRMSE = 0，最大绝对误差 = 0 |
| 后续原生 router 专家选择 | 与捕获的原生路径一致 |

原始证据：`precision_20261106/qsa_order_service/failed-shadow-layer-15-rank-*.pt`，以及 `hc-post15-combine-before.json`、`hc-post15-combine-fma.json`、`hc-post15-combine-native-reduction.json`。回放直接调用当前生产 `inject`，没有在 Python 中重新实现一份待测 combine。

## 修复后服务：剩余问题已确认有原生 HC 波动

包含两项修复的新服务通过了 48 层、batch 1/2 的启动数值验证，随后完成前 8 个标准答案用例、13 个 decode 同状态对照。在第 9 个 JSON 提取用例的第一个 decode，第 33 层 post-attention 超限，8 卡均严格终止，没有继续进入 512→64 测试。

| 第 33 层指标 | 结果 |
| --- | --- |
| HC combine 输出对原生 | 8 卡均逐位一致 |
| HC mixed 输入对原生 | rank 0/1/3/4/5 逐位一致；rank 2/6/7 NRMSE = `0.0003395177` |
| TP 规约后的 MoE 输出 NRMSE | `0.1302511543` |
| 整个 post-attention 输出 NRMSE | **`0.0444515832`**，超过原有 `0.006` 门槛 |
| 最终实例状态 | 8 卡均 `serving_shadow_failed`，`whole_decoder_monokernel=false` |

独立加载 rank 2 的真实输入、权重，在**同一份** post-attention / normed 输入上重复运行原生 persistent HC mix 16 次：15 次选择末位专家 **110**，1 次选择 **267**。第 10 / 11 名 router logits 的间距为零。捕获的原生路径选择 267；融合 HC 连续 3 次稳定选择 110，而在同一融合 mixed 上，原生 router 也选择 110。

因此，本次失败已经有直接证据表明：原生 HC mix 的 atomic split-K 累加顺序会跨 BF16 舍入边界，改变 MoE 专家。即使部分 rank 的 mixed 完全一致，其他 rank 的路由变化也会经 TP all-reduce 反映到所有 rank 的输出。这解释了为何不能仅查看 rank 0 就把剩余差异归为 MoE 算法错误。

这证明了该位置原生参考的不确定性，没有证明全部剩余融合误差都已消除。当前门槛保持不变，未将失败改判为通过。后续需要建立明确的确定性 HC / TopK 参考策略，再开展完整逐步、独立累计缓存与任务质量验证。

证据：`hc_combine_fixed_service/rank-*.json`、`failed-shadow-layer-33-rank-*.pt`、`hc-combine-fixed-serving.json` 和 `hc-post33-replay.json`。三个诊断服务均已停止，没有保留失败实例对外服务；本轮没有更新性能数据。

## QSA：短上下文需要保留原始块顺序

检查安装的 SGLang 源码发现：压缩块数不超过 512 时，原生 `fast_topk` 直接返回 `0,1,...,groups-1`，没有按分数排序。此前融合 QSA 对全部块排序，而独立参考也使用自己的 `argsort`，未能发现这处偏差。

已修改：

- `qsa.cpp` 在不超过 512 个压缩块时按原始 token 顺序读取；超过该数量时仍执行 Top512 选择。
- `qsa.py::native` 调用真正的 SGLang `fast_topk`，替换自行编写的排序参考。
- 独立验证除比较数值外，还检查集合一致；在原生要求顺序明确的短上下文中，检查顺序完全一致。失败时保存分数和索引张量。

由于 sparse attention 会分块更新 softmax，并将概率舍入为 BF16，读取顺序可能影响数值结果；但顺序不同并不保证每个样本都会出现误差。短输入的真实第 47 层回放中，旧 / 新顺序的最终输出均与原生一致，因此不能声称该样本已证明上一轮第 40 步 QSA 超限的全部来源。

真实权重回归覆盖第 3 / 47 层、rank 2 / 7、batch 1 / 2、初始长度 1 / 552 / 2044、eager 和 Graph 重放，**96 项通过**，最大 QSA 输出 NRMSE `0.0030841203`，没有放宽 `0.008` 门槛。结果为 `qsa-short-regression.json`。

只包含 QSA 顺序修复、尚未包含 HC combine 修复的独立服务，已完成 13 个答案用例与 512→64 的全部 63 步局部检查，未再在原先的 QSA 位置终止；整模型仅 56/63 步满足既定分布门槛，因此该实例仍为失败。此结果不是在原失败输入上直接复现，不能替代失败样本回放。

## 长上下文：原生 TopK 的零分并列选择

扩展验证到长度 2053（513 个压缩块）后，发现融合和原生选中的块集合不同，QSA 输出 NRMSE 为 **0.0581421554**。

保存分数后确认：8 个候选块的 index score 为零，第 512 名阈值也是零。两边交换的是同为零分的块 `331` / `428`，融合结果本身是合法的 Top512 集合。

使用**完全相同的分数张量**独立重放原生 `fast_topk` 64 次，原生排除的块在 **272、331、428** 之间变化。原生实现通过 atomic 收集候选，未约定并列分数下的选择与返回顺序；因此不能要求当前任意一次原生输出与另一种确定性选择逐位一致，也不能因此忽略实际产生的数值差异。

这部分尚未解决严格等价问题。当前不宣称 `context ≤ 4096` 的全部输入已经通过精度验收，也未修改基线的 TopK 语义或降低门槛。证据为 `qsa-topk-boundary.failure.pt` 和 `qsa-topk-ties.json`。

## 验证与追溯

本轮报告位于 `monokernel_reports/flash_next_tp8/precision_20261106/`。不同源码版本使用独立服务目录，GPU 二进制保留源文件 SHA256、编译命令和二进制 SHA256。

新增 `tools/diagnose_flash_next_hc.py` / `diagnose_flash_next_inject.cpp` 用于 HC combine 真实失败输入回放，`tools/diagnose_flash_next_qsa.py` 用于固定 mixed 输入、缓存和原生中间张量的 QSA 分段对照。QSA 抓取在显式指定步骤或局部失败时执行，并在结束后恢复 hook 与融合缓存。

同时修正了状态报告：局部 `FlashContractError` 也会锁存 `serving_shadow_failed` 并记录错误原因，避免早期局部失败仍显示为 pending，或沿用先前成功状态。CPU 单元测试 **42 项通过**。

已补充 `fast_topk`、HC combine wrapper 与 warp reduction 的 4 项原生源码指纹，合计 40 项，后续依赖变化需要重新核对。旧权重抓取中的自定义 Parameter 序列化已转换为普通 Tensor 并保留原文件，张量值逐位不变，迁移记录位于 `qsa_capture_service/weights-serialization-migration.json`。

复现真实 HC 失败样本：

```bash
python3 tools/diagnose_flash_next_hc.py \
  --capture monokernel_reports/flash_next_tp8/precision_20261106/qsa_order_service/failed-shadow-layer-15-rank-2.pt \
  --layer 15 \
  --output monokernel_reports/flash_next_tp8/precision_20261106/hc-post15-recheck.json
```

复现短上下文 QSA 回归：

```bash
python3 tools/verify_flash_next_qsa.py \
  --layers 3 47 --ranks 2 7 --lengths 1 552 2044 --device 0 \
  --output monokernel_reports/flash_next_tp8/precision_20261106/qsa-short-recheck.json
```
