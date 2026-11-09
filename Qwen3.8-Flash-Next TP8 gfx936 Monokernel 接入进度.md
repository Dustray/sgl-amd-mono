# Qwen3.8-Flash-Next：TP8 / gfx936 / BF16 Monokernel

日期：2026-11-06。开发留在当前 Git 工作区，未暂存、未提交；既有 27B 改动保留。

本轮 [完整精度验收](Qwen3.8-Flash-Next%20完整精度验收.md)扩展到 batch 1 的 64/512/2048/3072→64、batch 2 的 64/512/1536→64，共 472 个融合 decode forward；7 组均未满足全部门槛，不能标记为完整精度通过。固定输入进一步将第 14 层 HC 差异缩小到最后 sigmoid / 分支归约；长输入还有参考重复差异，独立服务对照与逐项表格见报告。

上一轮 [精度定位与修复](Qwen3.8-Flash-Next%20精度问题定位.md)：已对齐 HC combine 的原生 FP32 归约与短上下文 QSA 块顺序。真实 combine 失败样本修复后逐位一致，QSA 短上下文 96 项回归、42 项 CPU 测试通过；新服务仍在第 33 层 HC / MoE 对照超限，原生固定输入重放已复现专家切换。完整精度验收仍未通过，以下旧服务与性能结果不能代表修复后源码已验收。

## 当前实现

计算实现新增在 `sgl_amd_mono/qwen38_flash_next/gfx936/`，接口桥接放在 `compat/sglang0518/src/sgl_amd_mono_0518/flash_next.py`。既有 `gfx950/` 的 Quark / TP1 / NEXTN 实现保持原样。

本地代码已覆盖普通 decode 从 token embedding 到全词表 logits 的数学计算。此前 TP8 服务功能验收中，8 卡报告 `whole_decoder_monokernel=true`、`serving_shadow_passed`，batch 1/2 完成有限步骤的同状态验证。**后续正确性补测中，原生和融合的 13/13 个标准答案与完整 output_ids 一致，但 512→64 严格逐步对照在第 3 个 decode 步骤出现 greedy 分歧和 TV 越界；继续诊断到第 40 步又触发第 47 层 QSA 输出误差保护，完整数值等价验收未通过。** 原生重放自身也存在分歧，详细证据和解释见 [正确性验证](Qwen3.8-Flash-Next%20正确性验证.md)。本轮测试服务均已退出。Prefill、采样、调度、缓存分配以及 PLE token history / 状态检查点元数据仍由 SGLang 执行。

| 每个 decode forward 的融合发射 | 数量 | 内容 |
| --- | ---: | --- |
| Entry | 1 | 分片 embedding、核内 all-reduce、扩展 4 个 HC 分支 |
| GDN K1 | 36 | HC mix、QKVZ / BA、卷积状态、recurrent GDN、gated RMSNorm、输出投影 |
| QSA K1 | 12 | HC mix、QKV / index 投影、norm / MRoPE、KV / ring、4-token 压缩、Top512 block、稀疏注意力、gate / 输出投影 |
| Post-attention K2 | 48 | 注意力 all-reduce、HC combine / mix、router / Top10、BF16 routed / shared experts、MoE all-reduce、HC combine |
| PLE | 1 | N-gram hash、分片查表、all-reduce、投影 / norm / gate、dilation=3 短卷积、残差 |
| Tail | 1 | 最终 HC mix、分片 LM head、核内收集完整 248320 词表 logits |
| 合计 | **99** | 每卡每步产生实际输出的数学融合发射数；原生元数据与影子验收的参考执行另计 |

这与上游 gfx950 的 13-launch verify 路径不同。本实现使用 DTK HIP C++ 和 gfx936 BF16 MMAC，覆盖普通 autoregressive decode；不覆盖 NEXTN / MTP verify。

## 本次完整服务验收

最终报告：`monokernel_reports/flash_next_tp8/serving_complete_v5/verification.json`；`serving_complete/` 是指向该目录的符号链接。最早的失败目录已保留为 `serving_complete_initial_failed/`，其余各轮失败记录仍保留。

| 用例 | 实际输入 token | 实际输出 token | 融合 decode forward | 结果 |
| --- | ---: | ---: | ---: | --- |
| 纯 prefill 路由 | 300 | 1 | 0 | 8 卡的 99 个数学核计数均不增加，保持原生 prefill |
| 算术 | 24 | 3 | 2 | 严格输出 `56` |
| 中文解释张量并行 | 21 | 54 | 53 | 中文回答完整、含权重切分和通信聚合 |
| 英文解释蓝天 | 23 | 38 | 37 | 正常解释 Rayleigh scattering |
| 长输入检索 | 1633 | 5 | 4 | 严格输出 `FLASH936` |
| 两请求并发 | 每请求 32 | 每请求 32 | 31，batch 2 | 两请求均生成 32 token，实际进入 batch-2 融合路径 |

输出 token 计数采用服务 usage，包含结束 token；首个输出 token 来自 prefill，因此不计作 decode forward。8 卡各累计 **127 个融合 decode forward**，每一个主数学 kernel 的计数均为 127，即每卡 12573 次数学融合发射；没有将原生数学 fallback 计作融合路径。

每卡完成 **10 次整模型同状态检查**，包含 8 次 batch 1 和 2 次 batch 2，共比较 12 行全词表 logits；每次另外重复原生模型 3 次作为原生波动记录。融合 / 原生的最大 KL 为 `0.0029144283`，最大 TV 为 `0.0067515760`，greedy argmax 全部相同；均低于固定分布门槛。raw logits NRMSE 最大 `0.1262694`，原生重复执行的最大值为 `0.1998244`，因此不能把 raw NRMSE 单独解释为模型正确率损失。

此次功能验收的服务实例 PID 为 `1938022`，地址 `127.0.0.1:30160`。该实例已在后续性能测试前停止，完整记录仍保留在 `serving_complete_v5/`。以上结果是功能、数值和执行路径验收，不代表通用质量评测。

## 环境和边界

- 模型：`/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master`，原始 BF16，无量化转换。
- 8 张 gfx936，每卡 80 CU、65520 MiB 显存；TP8、PP1、EP1、DP1、CP1。
- SGLang `0.5.18+dtk2604.torch2110.2609201917.g2b8d2b`；Torch `2.11.0+dtk2604.2609201441.gf74cd7`。
- 普通 decode batch 1/2、context ≤ 4096、BF16 KV、page64、PLE embedding 在 GPU。
- 服务 Graph、overlap、radix cache 默认关闭。模块 Graph 测试不等于服务 padding / Graph 已验收。
- `gfx936/native_sources.json` 固定 SGLang 接口指纹，版本或依赖源码变化会拒绝接入。
- HIP 扩展以固定源码副本编译，记录源码 / 二进制 SHA256、参数 ABI、cooperative 驻留检查。
- 状态更新后发生数值或同步错误时拒绝请求，不回退后再更新一次状态。

## TP8 通信

使用 `hipExtMallocWithFlags(..., hipDeviceMallocUncached)` 分配并按 2 MiB 对齐 IPC mailbox。每个 BF16 拆成两个 32 位 packet，各含 24 位 collective epoch 和 8 位数据。消费者检查两个 packet 的轮次，所有消费者确认完成后生产者才能覆盖数据。epoch 达到 24 位上限时拒绝下一次 collective，避免轮次回绕。

普通 `hipMalloc` mailbox 曾在整模型压力下读到跨卡旧值，整数测试曾通过，不能用作完整验收。失败日志保留；当前 uncached 版本补测随机正负数、不同指数、连续两次规约以及 Graph 重放。

规约按 rank 0→7 做 FP32 累加，再截断为 BF16（RTZ），与本机 AITER custom all-reduce 的实际行为一致。最初使用 RNE，真实层 3 的注意力规约有 1058/2560 个元素不同，NRMSE `0.0040774117`，并通过 router 放大为 MoE 差异；改用 RTZ 后，该失败样本逐元素完全一致。该精度规则只作用于 TP 规约，投影等运算保留各自的原生舍入规则。

## 已完成的独立验证

| 验证 | 已通过的范围 | 报告，相对 `monokernel_reports/flash_next_tp8/` |
| --- | --- | --- |
| CPU 契约 / KV layout / 影子状态事务 | 41 项，含共用 logits buffer、失败时恢复状态、原生重放的层输出接口、分布验收、两种 prefill 策略、默认拒绝与诊断记录策略 | `test/unit/test_qwen38_flash_next_gfx936.py` |
| TP8 uncached mailbox / BF16 RTZ | 每卡 516 项、512 次 collective；随机与整数数据零误差，含连续规约 / Graph | `primitive_uncached_rtz/rank-*.json` |
| HC 原生对照 | 776 项；48 层 attention / MLP 与 tail 的真实权重、batch 1/2、随机 / embedding 输入、eager / Graph；最大 mix NRMSE `0.0007706661`，首个 decode 输入逐位一致 | `hc-all-layers.json` |
| BF16 MoE 原生算子对照 | 每卡 8 组，真实层 0、batch 1/2、4 组输入；实际 Triton experts、SGL Kernel / LightOp SiLU、shared gate，最大 local NRMSE `0.0003480044`，规约后 `0.0001545377` | `moe/rank-*.json` |
| GDN / sigmoid 门 | 最终源码复测 120 项；真实层 0/1/46、TP rank 0/7、batch 1/2、eager / Graph，比较输出及 conv / SSM；最大输出 NRMSE `0.0030223345` | `gdn-final.json` |
| QSA | 160 项；真实层 3/47、TP rank 0/7、batch 1/2、长短上下文、eager / Graph；TopK 集合 160/160 一致 | `qsa.json` |
| QSA 4096 边界 | 最终源码复测 32 项，4093→4096、batch 1/2、TP rank 0/7、真实层 3/47、eager / Graph；最大输出 NRMSE `0.0022341488` | `qsa-4096-final.json` |
| PLE | 每卡 24 项；真实投影 / norm / conv / multiplier，小型合成 embedding 表，EOS、eager / Graph | `ple/rank-*.json` |
| Entry / Tail | 最终源码复测每卡 12 项；真实 embedding / head 分片、跨分片 token、eager / Graph、全词表收集；最大 logits NRMSE `0.0000597541`，argmax 全部一致 | `io-final/rank-*.json` |

GDN / QSA 的独立验证不含 TP 通信；PLE 的小表测试不替代完整大词表验证。完整模型记录另存于 `serving_complete/`。

早期 GDN 核与独立参考都误用了默认 SiLU output gate，局部测试虽通过，整模型生成异常。模型配置实际要求 `output_gate_type=sigmoid`，现已修正核与独立参考，并把服务 GDN 影子对照改为真正的 `layer.linear_attn(...)` 路径。旧 `gdn.json` 不计作当前 sigmoid 实现的验收证据。QSA 边界首次补测的测试驱动在最后一步之后仍取 `req_to_token[..., 4096]`，触发越界，修正驱动后上述 32 项通过；失败日志保留。

整模型同状态对照另抓到 routed experts 的 SiLU 舍入差异：本机 SGL Kernel 和 LightOp 都先把 SiLU 结果舍入为 BF16，再乘 up；10,240 个激活的直接对照中，单次最终舍入有 2,775 个元素不同，双舍入逐位一致。当前 routed / shared experts 都保留这次中间舍入；shared gate 的点积则保留 FP32，采用原生 `fused_gate_sigmoid_mul_add` 的计算顺序。相关诊断见 `activation-rounding-diagnostic.log`。修正前整模型 logits NRMSE `0.0372586` 的失败报告保留在 `serving_complete_v2/`，不计为服务验收通过。

HC 下投影也需要匹配原生 split-K 的累加方式：将 10240 个 K 元素按 256 分段累加，避免长链 FP32 累加将 SiLU 推过 BF16 舍入边界。此前首个 decode 的低秩激活只有 1 个元素不同，HC mix NRMSE 为 `0.0002848505`；分段后该输入逐位一致，全部 97 组 HC 权重的 776 项检查通过。

### 整模型数值验收口径

`serving_complete_v4/` 的同状态检查测得：融合 / 原生 logits NRMSE `0.0456411`，原生 / 原生重复执行 NRMSE 也有 `0.0195957`。原生 HC persistent kernel 使用 FP32 atomic split-K 累加，BF16 舍入差异会经 router 放大，不能假设原生整模型逐位确定。该样本融合 / 原生的概率分布 KL 仅 `0.0000387387`、TV `0.000604484`，greedy token 相同；原生重复执行的 TV 为 `0.0000792120`。旧的 1% raw logits NRMSE 门槛连原生自身也未通过，这个失败报告保留。

因此整模型改为明确的分布验收：每一行 logits 有限、`KL(P_native || P_fused) ≤ 0.01`、`TV ≤ 0.05`、greedy argmax 完全一致。raw NRMSE 和逐层误差继续完整报告，不作为唯一的整模型拒绝条件；局部 attention / cache / MoE 的数值门槛保留。每次整模型影子检查额外从相同状态重放原生模型 3 次，报告其原生波动。阈值固定在代码中，不按某个请求的误差自动上调。这是当前 BF16 功能验收标准，不代表完成通用任务质量评测。

## 启动和验收

```bash
pip install --no-deps --no-build-isolation -e .
pip install --no-deps --no-build-isolation -e compat/sglang0518
bash launch_qwen_flash_next_monokernel.sh --max-running-requests 2
python3 tools/verify_flash_next_serving.py
```

基础配置由 `launch_qwen_flash_next.sh` 管理，Monokernel 脚本启用独立的 SGLang 0.5.18 TP8 插件；不要混用上游 gfx950 的量化 / NEXTN 参数。

启动时先按 batch 1/2、宽度 2560/10240、每种 3 组随机正负输入，对核内 TP 规约与实际原生 AITER 结果做 12 项逐位比较；随后对全部 48 层做 batch 1/2 的真实权重 HC / MoE 验证，36 个 GDN 层另验 conv / SSM。首次 decode 对 embedding、GDN、QSA、PLE、K2、tail 做同输入影子比较，不允许将影子步数设为 0。QSA 原生会写保留 compressed slot 0，比较时排除此槽，实际有效缓存仍参与核对。

此外，batch 1/2 分别在首次 decode 完整保存 GDN conv / SSM、QSA KV / ring / compressed / positions、PLE conv / N-gram history，从完全相同状态运行原生整模型，比较全词表 logits 和 greedy argmax，记录 48 层的累积误差，再恢复融合状态与共用 logits buffer。新一轮 prefill 会重新启用影子检查，覆盖不同提示词和上下文；已验收过的 batch size 另行保留。`SGLANG_AMD_FLASH_SHADOW_STEPS` 控制每轮每种 batch size 的影子步数，默认 1、本次服务验收设为 2，影子执行会增加这些步骤的耗时。GDN 的混合输入误差和同输入 attention 算子误差分开记录。逐层检查与整模型检查分别记录，防止独立参考和融合核犯同一类错误时被误判为完整验收。

性能测量可设置 `SGLANG_AMD_FLASH_SHADOW_REARM_ON_PREFILL=0`：首次遇到每种 batch size 时仍必须通过影子数值验收，随后请求不重复重启检查。先用相同输入输出量完成预热，再计时；`tools/benchmark_flash_next_serving.py` 会检查正式测量期间各卡的影子检查数量未增长、315 个 decode forward 各执行 99 个数学核，避免把原生参考执行混入 Monokernel 的正式时间。默认策略仍是每轮 prefill 重新检查。

`/server_info` 的 `amd_flash_next_monokernel` 和各卡 `rank-*.json` 提供路由、验收状态、设备执行计数和源码指纹。完整服务工具检查全部八卡的 99 个数学 kernel 计数、prefill 原生路由、算术 / 中文 / 英文 / 长输入检索，以及并发 batch 2。

功能、精度和路径测试之外，2026-11-06 已补充 batch 1、512→64 的 2 次预热 / 5 次正式性能测量：原生 / Monokernel 的端到端均值为 `15.1831 / 7.5341 s`，prefill 为 `594.95 / 602.56 ms`，decode 为 `14.5822 / 6.9260 s`，端到端 / decode 加速比分别为 `2.015× / 2.105×`。配置、计时边界、方差和原始记录见 `testresult.md`；正式测量未包含影子对照开销。其余 batch / 长度尚未形成性能结论。

## 源码和升级

```bash
git status --short
git diff -- compat/sglang0518/pyproject.toml pyproject.toml tools/source_status.py
python3 tools/source_status.py --verify
```

新增文件未暂存，`git diff` 不显示它们；需同时查看 `git status --short` 和模型的 `gfx936/` 目录。源码状态工具输出每个实现文件的 SHA256 和所校验的 SGLang 文件数。升级先检查固定接口，再依次运行独立模块和完整服务测试，不自动放宽契约或关闭数值检查。

功能验收阶段，`tools/source_status.py --verify` 校验 36 个原生接口文件通过；root / compat 两个 wheel 均已构建并检查，包含当时全部 14 个后端源文件和 Flash-Next 插件入口，记录于 `monokernel_reports/flash_next_tp8/wheels-final/verification.json`。兼容层单元测试 18 项通过。后续新增的性能测量开关与脚本由性能 JSON 的 `source_sha256` 另行记录。wheel 构建仅用于检查分发内容，服务使用可编辑安装的工作区源码。
