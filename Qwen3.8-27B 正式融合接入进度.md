# Qwen3.8-27B 正式融合接入进度

更新日期：2026-11-05。

## 完整 decode 接入（本机固定配置验收）

本轮补齐 16 层全注意力 K1，并将 embedding 并入第 0 层 GDN K1、最终 Gemma norm 与完整 BF16 LM head 并入 Tail；已覆盖 embedding 到 logits 的完整 decode 数学路径，每步 129 次 cooperative launch。prefill、采样及调度/状态管理仍使用原生 SGLang。这是 decode 融合范围，不是整个模型只发射一次 kernel。

`acceptance.json` 已更新并绑定当前源码和环境，包含 `dense_eager`、`dense_graph`、`full_attention_eager`、`full_attention_graph`、`decode_model_eager`、`decode_model_graph` 六项能力。`required` 模式自动接入完整 decode，不需要实验开关或 `DECODE_MODEL=1`。本轮新增代码、验证工具及文档均未提交，HEAD 仍为 `f5c842b1db8a3c7ae332c8daee3c25ebdc4bdb73`。

`monokernel_reports/qwen38_27b/full_attention_20261105/` 保存本轮结果。当前验收源码为 `9145803ee647264c89a5064fe19036c54e072a2253386626ce8f5749eb480895`，适配层指纹为 `405ebbb165c46ac6eae6ce78d53b496085458aa46e446977e06bc33b852f46c9`。已对齐 FP16 log2 分数、512-token 分段、BF16 概率及分段输出、softmax 归约、Q/K norm 和 Tail 归约顺序。

| 检查 | 结果 | 主要报告 |
| --- | --- | --- |
| 真实权重 attention，三层、三种分页布局、BF16/FP8 KV、batch 1/2、长度 1～4096 | 324 组通过；输出逐位一致，包含 eager/Graph | `final-attention-kernels.json` |
| embedding / 最终 norm / 全词表 LM head | 6 组 eager、18 次改变 token 的 Graph；embedding、最终 norm、Tail 输出逐位一致 | `final-decode-io.json` |
| QK 两阶段舍入边界 | 190458 个输入，eager 与三次 Graph 重放零差异 | `final-rounding.json` |
| attention 元数据和 padding | 22 组通过，非法输入在 KV 写入前拒绝，错误锁存和新状态恢复通过 | `final-guards.json` |
| 完整 129-launch Graph 结构 | 16 次 batch 1/2 交替重放，输出/状态/KV 与 eager 逐位一致 | `final-graph-chain.json` |
| 同步故障注入 | 8 组 eager/Graph 检查通过，有界超时、锁存、恢复 | `final-barrier.json` |
| 整模型同状态对照 | 244 步；逐请求最大 logits NRMSE `0.0020741280`，最小 cosine `0.9999979138`，最大 KL `1.77366e-8`；48 层活动 GDN 状态零误差，全部 argmax 一致 | `final-shadow-validation.json` |
| eager 服务 | 同配置原生对照 12/12 项逐 token 一致；prefill 的融合执行增量为零 | `final-eager-accepted-serving.json` |
| Graph 服务 | 无实验开关的新 `required` 进程 12/12 项逐 token 匹配同配置原生；六项验收能力自动加载 | `complete-required-serving-verification.json` |
| 服务切换与缓存 | Graph 4 轮、eager 2 轮长短/串行/并发切换；逐层计数符合 129 次发射；冷/热缓存分别匹配原生，命中 2688 token，流式 8 token 和 chat 通过 | `final-graph-transitions.json`、`final-eager-transitions.json` |
| CPU 与安装契约 | 76 项单元测试通过；31 个 SGLang 源码 hash 验证通过，未修改安装目录 | `final-unit.log`、`final-source-status.json` |

每步影子对照包含全部 129 个计算阶段和 48 层活动状态。最大同输入 kernel NRMSE 为 `0.0005638249`，低于 `0.003`；逐请求整模型 NRMSE 门槛维持 `0.02`。`0.00094434896` 是整批 logits 的最大统计，不能代替表中的逐请求最大值。结构 Graph 检查复用少数层的真实权重，整模型精度由独立的真实全权重影子对照验收。

当前支持配置为 TP1、gfx936/80 CU、BF16 权重、decode batch 1/2、FA3、FP8 E5M2 KV、page 64、context 4096、关闭 overlap；服务请求覆盖最大输入 2702 token、最大输出 128 token。prefill、采样、调度及缓存状态管理使用原生 SGLang。2026-11-06 已完成同卡原生/Monokernel 性能对比：64/512/2048 输入、16/64/128 输出、并发 1/2，共 18 组，每组重复三轮，每种模式 81 条正式请求，81/81 条输出逐 token 一致。当前 Monokernel 端到端耗时为原生的 2.89–12.10 倍，18 组均未获得加速，主要差距集中在 decode；prefill 均值接近。阶段计时、吞吐、原始记录与复现方法见 [性能对比](Qwen3.8-27B%20原生与Monokernel性能对比.md)，数据位于 `benchmark_results/qwen38_27b_20261106/`。本节上表仍是正确性和可用性验证。

新严格服务位于 GPU 2、`http://127.0.0.1:30152`，主 PID `1171606`，PID 文件为根目录 `qwen3.8-27b-complete-monokernel.pid`，日志在 `complete-required-serving/server.log`；无实验开关，自动加载六项验收能力。新进程的 12 项生成对照、两轮长短/串行/并发切换、冷/热缓存分别匹配原生、prefill 零融合增量、8 个流式事件及 chat 均通过，结果在 `complete-required-serving-verification.json`、`complete-required-transitions.json`。影子、原生及实验测试服务已关闭；GPU 6 的旧 112-launch 进程 `875611` 继续保留，未替换。

`final-verification.json` 汇总固定源码、环境、统计、证据 SHA256 和当前服务；`qualified-source.tar.gz`、`qualified-evidence.tar.gz`、`worktree.patch`、`wheels/` 与 `SHA256SUMS` 用于复现和审阅。归档与运行日志保留在 Git 忽略目录，实际代码和正式验收记录留在工作区供确认后提交。

复现完整 Graph 服务时，先为实例选择空闲设备和端口：

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 \
SGLANG_AMD_MONO_0518_REPORT_DIR=monokernel_reports/qwen38_27b/complete-serving \
bash launch_qwen_monokernel.sh --host 127.0.0.1 --port 30152 \
  --cuda-graph-backend-decode full
```

attention 同时检查原生 FA3、独立的同精度分段参考和高精度 FP32 参考：前两项门槛为 `0.003`；与 FP32 的误差不得超过同一输入下原生 FA3 的 FP32 误差加 `0.003`，同时要求两者数值有限。原生自身包含 FP16/BF16 中间舍入，因此这里采用原生误差基线。整模型 logits、KL、cosine 与状态验收门槛维持不变。性能对比尚未开始。

本轮已定位的数值根因及回归方法：

- QK 分数要求先做 FP32 乘法，再转 FP16。DTK 编译器即使设置 `-ffp-contract=off`，仍可能合为一次舍入的 `v_fma_mixlo_f16`。计算核用显式 `v_mul_f32` 保留这两个舍入步骤；`tools/verify_dense_attention_rounding.py` 在 FP16 相邻数中点附近检查 190458 个输入，覆盖 eager 和改变输入的 Graph 重放。
- PV 的 MMAC token 排列与 QK 不同：每个 64-token 页内，四个 lane group 各处理 16 个 token，每次 MMAC 读取连续四个元素。修正后，前两轮整模型失败捕获的 attention 输出均与原生逐位相同。
- Q/K norm：中间版本在第 53 步拒绝，logits NRMSE 为 `0.05702764`，原生重复运行零误差。第 19 层的首个局部差异来自 Q 归一化：原生 Triton 先在 16 元素内归约，再合并各组；此前通用 block 归约先合并相距 32 的元素。修正后，真实失败输入逐位一致；中间源码 `a18065504c3fbf4b0b241bf93eb4573f67f9b538ba927b54ccce6b2c48c5ad03` 的 244 步整模型同状态检查通过，最大 logits NRMSE `0.000079414656`，剩余局部差异仅在 Tail。
- LM head：根据实际 rocBLAS ISA 复现 1024 步长、两组各 512 元素的 BF16 MMAC 累加和 FP32 归约。组内先对四个对角元素两两相加，再按 `3→0→1→2` 顺序累加四个部分和，最后合并两组。该顺序由实际 LDS 写入及六组全词表探针共同验证；仅检查前 512 个输出通道会漏掉少量差异，因此最终回归检查全部 248320 个通道，并将 Tail 的数值门槛收紧为零。

`native-order-*` 报告对应修正 QK/PV 后、尚未修正 Q/K norm 归约的中间源码 `595e159b001c13e633bcc8111e33791ac3711a6ff7de77cfd738fca6d85f3c13`：324 组 attention、6 组 embedding/Tail、16 次完整 129-launch Graph 结构重放、8 组同步故障注入通过。结构重放复用少数层的真实权重，不能替代整模型数值验收；324 组中有 12 组仍存在很小的 Q 向量差异。后续验收必须重新绑定修正后的源码指纹。

长输入缓存单独保存了原生基线 `native-prefix-reference.json`：同一 2702-token 提示词命中 2688 token 前缀后，原生 SGLang 的冷/热缓存生成也不逐 token 相同。因此完整服务检查分别对齐原生冷缓存及原生热缓存结果，并检查实际缓存命中；不以冷/热缓存相互相等代替原生对照。`norm-order-graph-transitions.json` 中原先这一断言失败的记录保留，最终工具用 `--native-prefix-reference` 明确提供基线。

旧原生生成基线的 context 为 33792、静态显存比例为 0.96，且开启 overlap；它与当前配置不匹配，导致中间 9/12 token 统计不能作为最终对照结论。已在未启用插件的原生服务上重建 `final-native-reference-matrix.json`，固定 context 4096、0.90、关闭 overlap，并保留并发请求提交顺序。原默认基线备份为 `legacy-native-reference-matrix-33792-overlap.json`，默认基线已替换为新记录。服务验证工具新增配置一致性检查，失配直接拒绝；`final-graph-profile-match.json` 明确记录对已保存实际 Graph 输出的重新比较，原中间报告保持原样。

## 源码与版本管理现状

当前目录已改为真实 `sgl-project/sgl-amd-mono` Git 工作区，保留上游完整历史；上游 remote 为 `upstream`，基线为 `e52803f9d2b7dabff4bfd2f53ec25c03d56af947`，开发分支为 `local/sglang-0.5.18-gfx936`。迁移前代码有 `standalone-snapshot-20261105` tag 及完整 tar 备份。

模型实现现统一维护在 `sgl_amd_mono/qwen38_27b/`，包含 spec、权重检查、gfx936 kernel、runner 和验证入口；`compat/sglang0518/` 保留旧版接口桥接及旧导入转发。原有上游模型与 common 的计算实现保持原样，新目录中的计算核心由此前本地实验迁入，尚未完成上游 BF16 stage 的移植。逐项修改、安装和升级命令见根目录 `LOCAL_CHANGES.md`、`LOCAL_DEVELOPMENT.md`；通过 `git diff upstream-base-20261105 --` 审阅包含未提交修改的已跟踪文件差异，新增文件另见 `git status --short`。

**dense K1/K2 是本地 DTK HIP 实现，不是上游 kernel 直接移植的成功结果。** 下文为本轮改动前的 112-launch 固定配置验收基线：两组共 251 步同状态影子对照全部零误差；eager 与 Graph 各 12 例逐 token 匹配同配置原生；缓存命中、长短请求交错及无实验开关的 `required` 模式通过。报告在 `monokernel_reports/qwen38_27b/production_20261105/`。这些结果不替代当前 129-launch 新路径验收。

下文保留此前 FFN 阶段记录；其中旧路径 `src/sgl_amd_mono_0518/` 的计算模块已迁至上述模型目录，旧 wheel 和报告不代表最新 Git 分支的完整融合验收。当前只通过旧版 HookRegistry 接入，未导出新版原生 `MonoSpec`。

## 前一阶段 112-launch 可用性基线（工作区未提交）

支持边界保持明确：TP1、gfx936/80 CU、BF16、decode batch 1/2；48 个 GDN K1 与 64 个 K2，每步 112 次 launch。embedding、全注意力 QKV/RoPE/FA3/gate、最终 norm、LM head、prefill 和采样仍由原生 SGLang 执行。报告改为 `dense_decode_monokernel=true`、`whole_model_monokernel=false`，不能把它理解成整个模型只发射一次 kernel。

本轮实质修改：

- 用设备原子 epoch 栅栏替换硬件 `grid.sync()`；所有 block 发布写入后按 epoch 同步。超时依据设备查询的恒定时钟计算，默认一秒；失败锁存，后续请求不得使用部分更新的状态。仍使用 cooperative launch 与 occupancy 检查保证 80-block 的驻留条件。
- 捕获期间通过 `hipGraphAddKernelNode` 显式创建节点，设置并回读 cooperative 属性，然后更新 stream 的捕获依赖。`HIP_GRAPH_ACCUMULATE_DISPATCH=0` 在 HIP 初始化前生效。
- 兼容层新增第五个 hook：检查 ModelRunner 的 Graph 返回。普通 forward、Graph warmup 和 replay 返回后均检查错误状态；同步超时不会静默回退到原生路径。
- 正式模式改为绑定 `acceptance.json`：包含模型源码、适配源码、Torch/SGLang 版本及 HIP 编译器指纹。只有对应测试完成后才写入记录；源码/环境失配时 `required` 拒绝运行，实验 opt-in 不能绕过它。
- 原生数值依赖要求 `SGLANG_USE_LIGHTOP=1`、`SGLANG_USE_CAUSAL_CONV1D=1`、`SGLANG_USE_AITER_LINEAR_ATTN=1`，启动脚本设置这些变量，兼容层检查；检查失败不会错误缓存为“已通过”。
- 服务验证逐个保存已完成请求；长影子验证固定先提交 128-token 请求，再交错中文/英文各 64 token。报告分别记录功能与数值验收，功能通过不会自动变成正式验收。

已完成的检查：

| 检查 | 结果 | 本轮报告 |
| --- | --- | --- |
| 缺少一个 block 的故障注入，eager/Graph | 8 项通过；约一秒退出、错误锁存、独立新状态恢复 | `final-barrier.json` |
| 真实第 0/3/61 层，batch 1/2，eager/Graph、槽切换与负 padding | 64 项通过；计算公式未改变 | `epoch-kernels.json` |
| 64 层独立状态/工作空间、三组权重，两种 Graph 共存交替 | 64 次重放，输出/状态逐位匹配 eager，逐层计数正确 | `final-distinct-graphs.json` |
| 同配置原生 Graph 参考 | 12 例及真实前缀命中参考已采集 | `native-graph-reference.json`、`native-graph-prefix-reference.json` |
| 融合 Graph 的六类输入，串行/并发 | 12/12 逐 token 一致；prefill 的融合计数增量为 0 | `epoch-graph-serving.json` |
| 实际长前缀缓存命中 | 第二轮缓存 2688 token；两个轮次均逐 token 匹配原生 Graph | `epoch-graph-prefix.json` |
| 两轮 128-token 与 16-token 请求交错，同时启动/运行另一张卡的影子验证 | 通过，Graph 服务保持可用 | `epoch-graph-transition-stress.json` |
| 短请求同状态影子验证，全部局部算子和活动状态 | 117 步通过；logits 最大 NRMSE 为 0 | `epoch-shadow-suite.json` |
| 独立长输出影子验证，128/64/64 token 交错 | 134 步通过；两组共 251 步、221541 项 NRMSE 指标均为 0 | `epoch-long-shadow-suite.json`、`accepted-short-shadow.json`、`accepted-long-shadow.json` |
| eager 六类输入、串行/并发 | 12/12 逐 token 一致，数值验收通过 | `epoch-eager-serving.json` |
| Graph 功能与独立数值证据合并 | 通过；串行 362、并发 239 次 decode，均为每步 112 次设备执行 | `final-graph-acceptance.json` |
| 单元、拒绝路径和失败后重试 | 51 项通过 | `final-unit.log` |
| 安装目录 SGLang | 21 个固定源码 hash 全部一致，未修改安装文件 | `final-source-status.log` |
| 无实验开关的 `required` 新进程 | prefill 融合增量 0、算术逐 token 一致、聊天 323、流式 8 个 chunk、两条并发相同前缀均命中 2688 token 且输出一致 | `required-serving-smoke.json` |
| 两个 wheel 的构建与内容 | 构建成功；模型 wheel 包含当前验收记录及 C++ 源码 | `wheels/`、`wheel-build.log` |

数值验收快照固定保留前 117/134 步，避免后续功能请求追加诊断记录而改变验收取样；它们与实际测试请求及源码指纹关联。64 项原始 kernel 检查的全部 404 项误差指标同样为 0。

前一阶段模型实现指纹：`c0c613185dca788f2d08dc0eea6df2bbceb085ff662afc35c2c219171eb260fe`。适配层指纹：`96ba0e3f46fe9c3bc7f8fea5d8d7b708794d57071d23db91400d2dafbfcbd801`。Git HEAD 保持 `f5c842b1db8a3c7ae332c8daee3c25ebdc4bdb73`，本轮全部改动未提交。

该阶段保留 GPU 6 的 112-launch Graph 服务 `http://127.0.0.1:30148`，主 PID `875611`，PID 文件为根目录 `qwen3.8-27b-monokernel.pid`。当时的参数和日志在 `final-serving/server-info.json`、`final-serving/server.log`；其环境中两个 experimental opt-in 均不存在，模式为 `required`。当前完整 129-launch 服务、PID 文件和源码记录见首节。

启动脚本的默认值调整为已验证的 context 4096、静态显存比例 0.90、关闭 overlap/Graph；其他参数可显式覆盖。启用已验收 Graph 的复现命令如下，运行前先停止已占用同一 GPU/端口的实例：

```bash
HIP_VISIBLE_DEVICES=6 CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 \
SGLANG_AMD_MONO_0518_REPORT_DIR=monokernel_reports/qwen38_27b/production-serving \
bash launch_qwen_monokernel.sh --host 127.0.0.1 --port 30148 \
  --cuda-graph-backend-decode full
```

当前验收配置为 TP1、batch 1/2、BF16 权重、FA3、FP8 E5M2 KV cache、context 4096、最大输入 2702 token、最大输出 128 token、关闭 overlap。它不等于任意驱动、上下文长度或执行特性已验收；尤其未支持 TP2、LoRA、推测解码及 overlap 调度。Eager 与 Graph 的正确性通过不代表已证明加速，后续性能对比须使用相同的配置和请求。

本轮目录中的 `final-verification.json` 汇总最终结果；`qualified-source.tar.gz`、`worktree.patch` 保存未提交源码，`qualified-evidence.tar.gz` 保存数值/服务证据和原生参考，`SHA256SUMS` 记录这些归档及 wheel 的校验值。大报告和二进制仍放在 Git 忽略目录，正式记录和实现代码保留在工作区供审阅。

旧捕获方式在本轮又发生两次进程退出及设备接口异常，证据分别保存在 `graph-capture-interruption.json`、`explicit-graph-interruption.json`；仅显式构造节点不能消除该问题。换用软件 epoch 栅栏后，上述场景通过，但旧 SIGKILL 的驱动根因仍未证实。没有复位驱动，也没有修改安装目录的 SGLang。

合并长短影子诊断的请求提交顺序曾错误沿用历史结果的完成顺序，导致长请求排到最后并触及 HTTP 超时；这不算模型验收通过。记录保留在 `combined-shadow-request-timeout.json`，之后改用独立进程及 `tools/verify_dense_long_shadow.py`，明确记录提交顺序和 600 秒请求超时。

## 前一轮数值修正（历史记录）

计算实现仍在 `sgl_amd_mono/qwen38_27b/gfx936/`，本次没有修改安装目录中的 SGLang，也没有 Git 提交。仍是 48 个 K1 与 64 个 K2，每次 decode 共 112 次协作 launch。

已定位并修正以下实际数值路径差异，验收阈值保持不变：

- **rocBLAS 投影归约**：FP64 精确点积不能复现原生 BF16 舍入，已移除该修正。batch 1 的大投影复现 stagger 起点，输出/down 和 batch 2 的全部投影复现 4096 分段及两级 FP32 MMAC 归约；batch 1 的 b/a 使用 GSU8。b/a 在 batch 2 实际使用另一个 split kernel，已单独分派，不能沿用 GSU8。
- **GDN 与卷积**：复现 AITER 的 Q/K 前缀归约、状态点积、状态更新 FMA、硬件 exp2、补偿 ln2 的 softplus，以及卷积使用近似倒数的 SiLU。
- **decode gated norm**：原生 batch 1/2 对应 `ROWS_PER_BLOCK=1`，使用 64 线程各 2 元素的前缀归约。此前误用 prefill 的 16 线程各 8 元素布局，导致单个 BF16 舍入差异在后续层中放大。失败长输入的首层重放已确认这个根因。
- **residual/norm**：纯输入 RMS norm 和带 residual 两条路径均按当前 LightOp 的 1024 个逻辑线程、每线程 8 元素复现归约，用原有 256 线程协作 block 执行。纯输入路径保留权重加一的 BF16 舍入，residual 路径保留方差累加的 FMA，均使用硬件 RSQ。此前长生成暴露了首层纯 norm 的偶发 BF16 差异，因此没有沿用较早的 `decode-ba-*` 验收结果。
- **协作 Graph 捕获**：K1/K2 捕获后检查唯一尾节点，显式设置并回读 `hipKernelNodeAttributeCooperative=1`；配合 `HIP_GRAPH_ACCUMULATE_DISPATCH=0` 保留独立 kernel 节点。此修正已通过小型链和串行整模型请求，但并发稳定性仍未通过。
- **可审阅的诊断**：增加原生层输入及活动状态捕获、`tools/diagnose_dense_layers.py` 重放；K1 保存 recurrent 中间值，区分递推和 gated norm 误差。48 层新增工作空间共 1.125 MiB，C++/ctypes ABI 同步更新且继续校验。

最新验证记录在 `monokernel_reports/qwen38_27b/numerics_20261105/`。该目录中的中间失败报告保留用于追溯；只有源码指纹匹配的最终报告可作为当前实现的证据。当前投影实现优先排查精度，尚无性能优化结论。启动脚本默认关闭整模型 Graph/overlap；Graph 仅保留显式实验开关，不作为稳定支持，`required` 继续拒绝未完成全部验收的实现。

| 验证项 | 结果 | 证据（相对于上述报告目录） |
| --- | --- | --- |
| 真实第 0/3/61 层，batch 1/2，eager/单 kernel Graph、槽切换、负 padding | 64 项通过；全部 404 项误差指标 NRMSE 均为 `0` | `graph-repair-kernels.json` |
| 64 层真实权重启动检查 | 224 项通过；全部 640 项误差指标 NRMSE 均为 `0` | `graph-repair-shadow/dense_real_weight_validation.json` |
| 串行与并发各六类输入，每项生成 12 token，另做短输入重复 | 117 步通过；每步全部 112 项局部检查与 48 层状态检查；全部误差指标及 logits NRMSE 为 `0`；短输入重复未实际命中前缀缓存 | `graph-repair-suite.json`、`graph-repair-shadow/dense_model_shadow_validation.json` |
| 并发长程影子对照：128 token 长输出，与中文/英文各 64 token 交错 | 134 步通过；batch 1/2、每步 112 项局部及 48 层状态检查；全部误差指标及 logits NRMSE 为 `0` | `final-long-shadow-suite.json`、`final-long-shadow/dense_model_shadow_validation.json` |
| eager 六类输入，串行/并发，16–128 输出 token，另测 prefill-only、短输入重复和 chat | 功能通过；与新采集的同配置原生共 12/12 例逐 token 一致；串行执行 362 次 decode、40544 次 K1/K2，并发执行 239 次 decode、26768 次 K1/K2；prefill-only 计数为 0 | `final-eager-matched-verification.json`、`native-matched-reference.json` |
| 长输入真实前缀缓存命中及状态恢复 | 原生与融合第二次均命中 2688 token；首次与命中后两轮分别逐 token 匹配原生；原生自身首次与命中后输出有差异，融合也复现该行为 | `prefix-cache-comparison.json` |
| 64 层协作链 Graph，batch 1/2，改变输入各重放三次 | 通过；各形状 K1 执行 144 次、K2 192 次，输出有限；使用重复第 0 层权重，不能代替完整模型数值验收 | `graph-repair-chain.json` |
| 整模型 Graph 串行生成，六类输入、16–128 输出 token | 六例全部逐 token 匹配原生；GPU 共执行 362 次 decode、40544 次 K1/K2；prefill-only 无 K1/K2 | `graph-repair-serving-verification.json` |
| 整模型 Graph 并发生成 | **未完成**；并发长输出阶段调度进程退出 `-9`，请求断开；不能据串行结果判定 Graph 通过 | `graph-repair-interruption.json`、`graph-repair-serving/server.log` |
| CPU 测试与源码契约 | 35 项测试通过，21 个 SGLang 文件 hash 通过 | `unit-tests-graph.log`、`source-status-graph.log` |

以上最新验收对应源码指纹 `12b2e2d4ade71ff2042969cb10ea844fa56fd512b3b5ca8f365ca0a75e5d187c`，kernel 源码 SHA256 为 `b37dfbfe76311d884762060daaae281436dee8dfe305dd8538b04c45115bd285`，二进制 SHA256 为 `178e377a4037bf15c356772a2110a7a0cb0f3c0a20973c3b9044678230050502`。固定 TP1、gfx936、batch 1/2、context 4096、BF16 权重和 FP8 E5M2 KV，关闭 overlap；同状态影子数值验证关闭 Graph，Graph 功能验证单独运行。

机器可读汇总为同目录 `final-verification.json`，包含各项通过/未通过状态、运行版本、源码/二进制指纹、证据文件 SHA256、服务信息及未提交 Git 的状态；Graph 稳定性与正式模式明确记录为未通过/未启用。

旧原生基线使用 context 33792、启用 overlap 与整模型 Graph，与最新 eager 配置不同。最新 eager 相对该旧记录为 9/12 例逐 token 一致；重新按相同 context/Graph/overlap 配置启动原生 SGLang 后，原生自身也在相同三例发生变化，最新融合与新原生为 12/12 一致。旧对照保留在 `final-eager-serving-verification.json`，新原生及两份输出的离线逐 token 重比较分别保存在 `native-matched-reference.json`、`native-matched-token-comparison.json`、`final-eager-matched-verification.json`，没有覆盖旧基线。

短提示词的重复请求不足一个 cache page，因此原测试的 `cached_tokens=0` 只能说明重复生成一致。新增长输入验证记录了真实命中量，分别比较原生与融合首次、命中后输出；没有把原生自身缓存命中带来的生成差异误判为融合偏差。初次诊断错误要求原生首次与命中后 token 完全相同，失败记录保留为 `prefix-cache-repeat-initial-failure.json`；它不参与融合数值验收。

Graph 排查新增一个可复现的问题：默认 DTK 将连续协作发射聚合为内部 `type=200` 节点；设置 `HIP_GRAPH_ACCUMULATE_DISPATCH=0` 后保留独立 kernel 节点，但其 cooperative 属性仍为 0。捕获后显式设置并回读该属性后，1/2/112 个协作 kernel 的小型 Graph 均完成改变输入后的三次重放。证据为 `graph-capture-metadata.json`、`graph-capture-no-accumulation.json`、`graph-capture-repaired.json`、`graph-cooperative-chain.json` 与对应源码/日志。**这些是最小复现，尚不能替代整模型 Graph 验收，也尚未证明它是此前停滞的唯一原因。**

本次整模型 Graph 服务于日志时间 `2026-11-05 13:42:45` 报告 scheduler PID 763888 退出 `-9`，没有 Python 数值异常；随后 `hy-smi` 对八张设备均报告 `Open mkfd failed`，只读设备打开探针曾处于 `D / drm_open`。cgroup 的 `oom` 和 `oom_kill` 计数均为 0，当前权限不能读取内核日志，**尚不能确定 SIGKILL 和设备异常的根因**。同机 eager 影子服务此前于 13:42:26 收到本次主动清理的 SIGTERM，13:42:32 清理其子进程；时间关系保留在中断报告中，不能据此断言因果。设备接口随后恢复；未复位驱动、未重试整模型 Graph，改用默认 eager 路径继续功能验收。

该轮曾保留 GPU 6 的 eager 服务 `http://127.0.0.1:30144`，主 PID 773257，现已退出；旧 PID、日志及参数见 `final-eager-serving/`，不能作为当前存活服务。该历史配置为：

```bash
HIP_VISIBLE_DEVICES=6 CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 \
SGLANG_AMD_MONO_0518_MODE=monokernel \
SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE=1 \
SGLANG_AMD_MONO_0518_SHADOW_STEPS=0 \
SGLANG_AMD_MONO_0518_REPORT_DIR=monokernel_reports/qwen38_27b/eager \
bash launch_qwen_monokernel.sh --host 127.0.0.1 --port 30144 \
  --context-length 4096 --mem-fraction-static 0.90
```

后续重点是整模型 Graph 在并发长生成下的稳定性，以及保持上述数值基线的计算效率优化。当前尚未证明性能收益；正式模式和端到端、prefill、decode、输入/输出长度矩阵的性能比较继续等待完整验收。全注意力核心、final norm、LM head 和 prefill 仍按原生路径执行。

## 上一轮完整层开发与验证（历史记录）

按用户要求，当前修改均保留在工作区，**未执行 Git 提交**，HEAD 仍为 `f5c842b1db8a3c7ae332c8daee3c25ebdc4bdb73`。提交须等用户确认。

执行路径现覆盖 48 个 GDN K1 和 64 个 K2：K1 包含输入 norm、qkv/z/b/a 投影、卷积、SSM 更新和 gated norm；K2 包含输出投影、residual/postnorm、gate/up、SiLU 和 down。每次 decode 共 112 次协作 launch；全注意力核心、最终 norm、LM head 和 prefill 保留原生路径。这一覆盖范围不等于整模型精度已经合格。

本轮修改：

- 投影使用 `__builtin_hcu_mmac_f32_16x16x16_bf16`，每个 wave 处理 16 个输出通道和 batch 1/2。实际汇编确认包含 `v_mmac_f32_16x16x16_bf16`。
- 对接近 BF16 舍入中点的结果做 FP64 点积重算，处理再次转 FP32 时的双重舍入。未放宽原有数值阈值。
- K2 保存 gate/up 中间结果，诊断工具分开比较 output projection、norm、gate/up、SiLU、down，可重放影子验证保存的真实层输入。
- 启动验证全部随机输入固定种子；原生卷积参考路径按接口要求使用 int32 索引；Graph 测试同时覆盖 int64 槽索引、负 padding 槽和槽切换。
- 影子验收逐请求检查 logits，要求全部 112 个局部检查及 48 层状态检查齐全；拒绝缺项、重复项和非有限结果。失败时重复原生 forward，排除原生自身不稳定。
- 报告记录模型源码 SHA256；服务功能成功和数值验收分开记录。旧报告不会因修改源码而自动成为当前实现的验收依据。
- 实验服务入口关闭 overlap 和整模型 decode Graph，兼容层也拒绝这两个未经验证的组合，防止默认启动进入已观察到停滞的路径。单 kernel 的 Graph 重放验证仍保留。

| 验证项 | 本轮结果 | 证据 |
| --- | --- | --- |
| 修正后 K1/K2，真实第 0/3/61 层，batch 1/2，eager/Graph | 64 项通过；K1 最大 NRMSE `0.0006152053`，K2 `0.0012853003` | `mmac-refined-kernel-verification.json` |
| 修正后 64 层真实权重启动检查 | 224 项通过；K1 最大 NRMSE `0.0013403624`，K2 `0.0023530120` | `mmac-refined-shadow/dense_real_weight_validation.json` |
| MMAC 第 40 层失败输入重放 | K2 NRMSE 从 `0.0035131955` 降至 `0.0003708658` | `mmac-layer40.json`、`mmac-refined-layer40.json` |
| 修正后整模型同状态对照 | 未通过；第 13 步 logits NRMSE `0.0438190211`、KL `0.02375693`，分别超过 `0.02` 和 `0.01` | `mmac-refined-shadow/dense_model_shadow_validation.json` |
| 失败步原生重复执行 | logits NRMSE `0`，与第一次原生输出逐位相同 | 同上 `native_repeat_logits` |
| 关闭 overlap 后整模型复核 | 同一算术请求仍在第 12 步出现 NRMSE `0.0438190211`；原生重复误差仍为 `0`，说明该精度偏差不由 overlap 引起 | `serial-shadow/dense_model_shadow_validation.json` |
| 整模型 Graph | batch 1/2 捕获成功；实际 warmup 重放后未继续推进，已停止测试服务 | `mmac-refined-graph/server.log` |
| 本地 CPU 单测 | 31 项通过，包含逐请求验收、漏项/重复项/非有限数拒绝及调度准入 | `test/unit/test_qwen38_27b.py`、`test/unit/test_sglang0518_compat.py` |

表中报告均位于 `monokernel_reports/qwen38_27b/full_mono_20261105/`。早先标量版本的 512 组 K2 诊断和 32 步短对照通过，不能替代这里较长对照的失败结果。

下一项精度工作是继续对齐原生 GEMM、norm 和 GDN 的累加与舍入行为；Graph 需要单独排查协作 launch 的重放和并发语义。此次模型测试服务均已停止，启动脚本保留串行、eager 的实验入口。当前不能宣称完整 Monokernel 正式支持，也不据此开展性能结论。

## 模型目录迁移验证（2026-11-05）

- 本地模型与兼容层的 22 项 CPU 测试、4 项上游风格 SGLang 接口契约测试通过。
- 原 dense C++ kernel 与 `source-managed-20261105` 逐字节一致；FFN kernel 的计算 AST 保持一致。
- 新入口 `python -m sgl_amd_mono.qwen38_27b.check` 在 256/512 和真实 5120/17408 尺寸、batch 1/2 下完成 eager 与三次 Graph 重放，共 16 组检查，最大 NRMSE `0.0005089479 < 0.002`。
- 服务通过根包 `sgl_amd_mono` 的插件入口接入旧版 hook；64 层真实权重共 128 组检查通过，最大 NRMSE `0.0013608205 < 0.002`。
- 并发 1/2 的三项短生成请求各返回 8 token，均观察到 GPU FFN 执行计数增加；prefill-only 请求的融合执行增量为 0。
- 根包 wheel 包含模型 Python 文件及 `gfx936/kernel.cpp`，隔离编译入口能从新包路径运行。

结果保存在 `monokernel_reports/qwen38_27b/ffn-check.json`、`serving-check.json`、`refactor-verification.json`。这些验证说明模型目录与旧版接口接通，不作为完整 Monokernel 精度验收或性能对比。

## 此前 FFN 阶段记录

已实现 gfx936 上真实的 BF16 dense FFN 融合，并接入 `fused_ffn` 模式；decode batch 1/2 进入自定义 GPU kernel，prefill 保留原始路径。**目前完成的是 FFN 部分融合；GDN、全注意力、residual/norm 尚未融合，完整 Monokernel 仍未完成。性能对比尚未开始。**

## 本次 FFN 实现

新增 `src/sgl_amd_mono_0518/bf16_ffn.py`、`compile_ffn.py`、`hip_ffn.py` 和 `ffn_runtime.py`，通过现有 HookRegistry 接入，不修改安装的 SGLang。

| 项目 | 实现 |
| --- | --- |
| 计算范围 | gate/up GEMV → SiLU-mul → down GEMV |
| 数据格式 | 原始 BF16 权重，FP32 累加，BF16 中间舍入和输出 |
| 发射次数 | 每层 2 次 GPU launch |
| 第一次 launch | 68 个独立工作组，每组 256 线程；处理 256 个 intermediate 元素，并计算该分块的 down 部分和 |
| 第二次 launch | 归并 68 份 FP32 部分和，输出 BF16，更新设备执行计数 |
| 权重副本 | 0，直接引用原始 gate/up、down 参数 |
| 工作空间 | 每层 2.65625 MiB，64 层合计 170 MiB |
| 支持路由 | TP1，decode batch 1/2，gfx936，hidden 5120 / intermediate 17408 |
| 回退路由 | prefill 和其他 batch 在执行模型前选择原始 FFN |
| Graph | 当前 HIP stream 发射，支持 batch 1/2 捕获及重放 |
| 计数 | GPU kernel 写入设备计数，不能将 Python forward 次数当成 replay 次数 |

这里采用分块和第二次 launch 完成全局归并，未使用跨工作组自旋 barrier。GEMV 当前使用 BF16 向量加载与 FP32 FMA，尚未改成 MMAC；前一阶段 MMAC 基础测试通过不能视为当前 FFN 在使用 MMAC。

集成时解决了两个实际问题：FlyDSL 和现有 CuTe 的 MLIR/nanobind 重复注册导致进程 abort，因此编译放在独立子进程、推理只加载 HSACO；LightOp 的 BF16 SiLU 会先舍入再乘 up，融合实现也保留这次舍入，避免额外模型输出偏差。HSACO 缓存按源码、FlyDSL 和 Torch 版本分组，并验证二进制 SHA256 和参数 ABI。

启动时对 64 层真实权重各做 batch 1/2 数值验证；NRMSE 超过 `0.002` 或出现非有限数即拒绝接入。实例验证与服务对照结果分别保存在 `monokernel_reports/bf16_ffn_hip_verification.json` 和 `monokernel_reports/ffn_serving_verification.json`。服务对照仅用于正确性和执行路径验证，不是性能测试。

```bash
bash launch_qwen_fused_ffn.sh
HIP_VISIBLE_DEVICES=4 python3 tools/verify_bf16_ffn.py --full-shape \
  --output monokernel_reports/bf16_ffn_hip_verification.json
python3 tools/verify_fused_ffn_serving.py
```

当前服务使用 `qwen3.8-27b-fused-ffn.pid` 管理，日志为 `qwen3.8-27b-fused-ffn.log`。独立算子测试使用空闲 GPU4，模型服务使用 GPU0。

## 2026-11-05 验证结果与限制

| 验证 | 结果 |
| --- | --- |
| 独立 FFN | 小尺寸及真实尺寸，batch 1/2，eager 和三次改变输入后的 graph 重放全部通过；每组设备执行计数增量为 4 |
| 真实尺寸独立 FFN 误差 | NRMSE 最大 `0.000508948`，最大绝对误差 `0.0078125` |
| 64 层真实 checkpoint 权重 | batch 1/2 共 128 组全部通过；最大 NRMSE `0.0013608205`，低于 `0.002` 门槛 |
| 模型 decode graph | batch 1/2 均捕获成功，服务日志确认实际生成阶段使用 graph |
| 纯 prefill | `max_new_tokens=0` 请求的 FFN 设备执行计数增量为 0 |
| 串行生成 | 每层 368 次真实 FFN 执行 |
| 双并发生成 | 每层 243 次真实 FFN 执行 |
| 两轮生成合计 | 39,104 次 FFN 执行，78,208 次融合 kernel launch；覆盖全部 64 层 |
| 支持检查 | 正常配置通过；TP2、未验证 GPU、未验证 FlyDSL 以及缺失完整 Monokernel spec 的严格模式均拒绝 |

原生基线使用 GPU4、端口 30001，融合服务使用 GPU0、端口 30000；两者分别以并发 1 和 2 生成同样的六种输入，每轮前清空 radix cache。测试固定温度 0，强制输出 16–128 tokens，用于比较执行路径和生成结果，未计算性能对比。原生服务的同条件串行重复测试为 6/6 完全一致。原生临时服务测试结束后已停止并释放 GPU4。

| 用例 | 输出 tokens | 串行：共同前缀 / 总长 | 双并发：共同前缀 / 总长 |
| --- | --- | --- | --- |
| 算术 | 16 | 16 / 16，一致 | 16 / 16，一致 |
| 中文 | 64 | 19 / 64，有差异 | 64 / 64，一致 |
| 代码 | 64 | 64 / 64，一致 | 64 / 64，一致 |
| 英文 | 64 | 49 / 64，有差异 | 64 / 64，一致 |
| 长输入 | 32 | 4 / 32，有差异 | 32 / 32，一致 |
| 长输出 | 128 | 128 / 128，一致 | 128 / 128，一致 |

**12 项生成对照有 9 项逐 token 一致，3 项串行生成存在措辞或推理文本差异。模型运行与局部 FFN 数值验证已通过，完整模型的生成一致性尚未完全通过。** JSON 保留全部 output IDs、文本及共同前缀上的 logprob 差异；这不是完整 logits 或 KL 验证。

进一步对第 0、61 层各抽查 batch 1/2，以 FP64 点积、BF16 中间舍入作为独立参考，记录在 `monokernel_reports/ffn_reference_precision.json`。四项的融合 NRMSE 范围为 `1.9431e-5`–`3.1163e-4`。第 61 层 batch 1 的融合误差为 `6.2809e-5`，原生 BF16 路径为 `0.0013612052`；说明原生 BF16 点积本身也存在累加误差。该抽查不能代替完整模型逐步 logits 验证，也不能据此忽略上述生成差异。

主要复现产物：

- `monokernel_reports/bf16_ffn_hip_verification.json`：独立 kernel 数值和图重放。
- `monokernel_reports/process-472063.json`：本次模型进程的真实权重验证、路由及设备计数；重启后使用新的 `process-<pid>.json`。
- `monokernel_reports/ffn_native_reference_matrix.json`：按相同并发分别采集的原生生成基线。
- `monokernel_reports/ffn_native_repeatability.json`：原生串行生成重复性控制。
- `monokernel_reports/ffn_serving_verification.json`：12 项融合服务对照和计数快照。
- `monokernel_reports/ffn_configuration_verification.json`：不支持配置的拒绝验证。

```bash
python3 tools/verify_fused_ffn_serving.py \
  --reference monokernel_reports/ffn_native_reference_matrix.json
HIP_VISIBLE_DEVICES=4 python3 tools/verify_ffn_reference_precision.py \
  --model-path /public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master \
  --output monokernel_reports/ffn_reference_precision.json
```

原生基线的重采集方式是在独立原生服务上运行：

```bash
python3 tools/verify_fused_ffn_serving.py --url http://127.0.0.1:30001 \
  --capture-native-reference --reference monokernel_reports/ffn_native_reference.json \
  --output monokernel_reports/ffn_native_reference_matrix.json
```

串行计数包含 overlap scheduler 在消费最后一次采样结果前可能多启动的 decode，因此设备执行次数允许每个请求多一步。状态查询会等待 GPU 完成后读取计数。生成对照用例只用于正确性验证，性能请求测量数仍为 0。

## FlyDSL 安装与验证

用户给出的安装版本：

```bash
pip install flydsl==0.2.0+dtk2610.torch2110.2610101746.gad9161 \
  -i https://pypi.sourcefind.cn/nightly/dtk/
```

索引确实包含这个版本，但仅提供 `cp312-cp312` wheel。当前 Python 为 3.10.12，pip 对该精确版本返回 `No matching distribution found`。同一索引还提供相同源码提交 `ad9161`、匹配当前 DTK 2604 和 Torch 2.11 的 cp310 构建，已安装：

```bash
python3 -m pip install --no-deps \
  'flydsl==0.2.0+dtk2604.torch2110.2610101749.gad9161' \
  -i https://pypi.sourcefind.cn/nightly/dtk/
```

wheel SHA256：`4c1ed83b8fd05285273d769c0e4000bf7ba68107048e0373a89edc2acb52ad74`。wheel 元数据要求 `torch==2.11.0`，当前已安装 `2.11.0+dtk2604.2609201441.gf74cd7`。安装使用 `--no-deps`，没有更换当前 Torch 或 SGLang。

该 DTK 构建明确包含 gfx936 的 HCU 架构识别、HCU clang codegen、HCU MMAC 布局及 lowering。实际测试在空闲的物理 GPU1 完成，未占用原先用于模型服务的 GPU0。

```bash
HIP_VISIBLE_DEVICES=1 \
FLYDSL_DUMP_IR=1 \
FLYDSL_DUMP_DIR=/tmp/qwen-flydsl-gfx936-ir \
python3 tools/verify_flydsl_gfx936.py \
  --output monokernel_reports/flydsl_gfx936_verification.json
```

验证脚本执行真实 FlyDSL kernel，再与 PyTorch 结果比较；每个形状还捕获 HIP Graph，连续三次改变输入并重放，确认 graph 确实执行新输入。

| 项目 | 验证范围 | 结果 |
| --- | --- | --- |
| 目标架构 | gfx936，wave64，80 CU | 检测正确 |
| HCU clang | DTK 外部 codegen | 启用 |
| BF16 SiLU-mul 融合算子 | 127、5120、17408 元素 | eager 与 graph 最大绝对误差均为 0 |
| BF16 MMA | `[64,16] @ [64,16].T`，FP32 累加和输出 | eager 与 graph 最大绝对误差均为 `9.5367431640625e-7` |

具体误差、各次 graph 重放结果和运行版本保存在 `monokernel_reports/flydsl_gfx936_verification.json`；完整编译日志为同目录的 `flydsl_gfx936_verification.log`。生成的 LLVM IR 已复制到 `monokernel_reports/flydsl_gfx936_codegen/`，其中可以直接确认 `llvm.hcu.mmac.f32.16x16x16.bf16` 调用。上述测试只验证编译器和基本算子，不能替代 GDN 状态、完整层或模型输出验证。

该阶段安装后重新检查了 SGLang 源码哈希及 hook 契约，均通过。严格模式已识别到 FlyDSL 版本，继续因缺少完整模型 kernel 而拒绝，日志为 `monokernel_reports/preflight_after_flydsl.log`。当时恢复了原审计服务，`/health` 返回 200，`17 × 19` 返回 `323`，记录为 `monokernel_reports/serving_after_flydsl.json`。2026-11-05 的 FFN 接入改用独立的 `fused_ffn` 启动脚本及 PID 文件。

## 27B dense 上游实现的核对

除 `sgl-project/sgl-amd-mono` 的 Flash-Next 实现，还找到了 SGLang PR #43558，标题为 `[ROCm][Perf] Qwen3.8-27B dense mono decode: persistent FlyDSL GDN and FFN launches (TP1 and TP2, MI355X/MI350X)`。

核对的 PR head 为 `b25e92fd224a196bf610ac05405073208ff7d513`，查询时状态为 closed、未合并。核对对象及版本固定在 `src/sgl_amd_mono_0518/manifest.json`，不能将未合并 PR 描述为当前 SGLang 已具备的功能。

| 项目 | 本地 checkpoint / 硬件 | PR #43558 |
| --- | --- | --- |
| 模型族 | Qwen3_5，27B dense | Qwen3.8-27B dense |
| hidden / intermediate | 5120 / 17408 | 5120 / 17408 |
| 权重格式 | BF16，未量化 | QuarkW4A4MXFP4，uint8 packed + E8M0 scales |
| GPU | Hygon gfx936 | AMD gfx950 |
| CU 数 | 80 | guard 要求至少 256 |
| 工作组计划 | FFN 68 个独立工作组 + 第二次归并 launch；非 persistent | `BLOCKS = 256`，persistent |
| 计算指令 | 当前 FFN 使用 BF16 加载 / FP32 FMA；MMAC 已单独验证 | gfx950 scaled FP4 MFMA |
| TP | 1 | 1 或 2 |

关键源码位置（相对该 PR head）：

- `python/sglang/srt/layers/qwen3_8_dense_mono.py`：架构、量化和常驻资源 guard；K1/K2 接入与数值检查。
- `python/sglang/kernels/ops/moe/qwen3_5_mono_flydsl/dense_gdn.py`：输入投影、卷积、GDN 状态更新与 gated norm。
- `python/sglang/kernels/ops/moe/qwen3_5_mono_flydsl/dense_ffn.py`：输出投影、residual/norm、gate/up、SiLU-mul 与 down。
- `python/sglang/kernels/ops/moe/k3_mono_flydsl/common/plan.py`：固定 256 个常驻工作组及其工作划分。
- `python/sglang/kernels/ops/moe/k3_mono_flydsl/common/mx.py`：`mfma_scale_f32_16x16x128_f8f6f4`，按 FP4 packed 权重和缩放读取。

这些差异需要修改计算实现、权重读取与工作划分。安装 FlyDSL、修改架构名称或将 `BLOCKS` 单独改成 80 都不能使该 kernel 正确运行。没有把 BF16 checkpoint 重新量化，也没有跳过这些 guard。

## 原 FFN 阶段的待办（历史记录）

以下为仅有 FFN 融合时的原计划，当前 K1/K2 实现与验收以文首为准。当时定位的 GDN 接入点为本机 `GDNAttnBackend.forward_decode`：卷积状态来自 `mamba2_layer_cache(layer_id).conv[0]`，SSM 状态为同一缓存的 `temporal`，请求槽映射来自 `forward_metadata.mamba_cache_indices`。融合必须保留 `_track_mamba_state_decode` 的状态跟踪语义，不能在写入部分状态后静默回退；当时尚未替换该路径。

1. dense FFN 已通过 BF16 分块融合实现；继续覆盖输出投影和 residual/norm，并评估使用 HCU MMAC 及常驻工作划分的适用性。
2. 接入本地 SGLang 0.5.18 的 GDN 卷积状态、SSM 状态、请求槽映射、全注意力 KV 布局，以及 residual/norm 数据流。
3. 为跨工作组同步实现 HCU 内存可见性、epoch、deadline 与失败终止，避免部分状态更新后静默 fallback。
4. 在加载的真实 BF16 权重上验证 GDN、完整层和完整模型的 hidden states / logits / 生成 token；覆盖 graph 捕获和重放、batch 1/2 及回退往返。
5. 通过模型级数值与状态检查后，才启用正式融合模式，再进行端到端、prefill、decode 和长度矩阵的性能对比。

当时 `bundled_monokernel_specs` 为空，严格模式在权重加载前拒绝运行；`experimental_fusions` 仅注册 FFN 局部融合，`monokernel_executions` 为 0，真实 FFN 执行单独记录为 `fused_ffn_device_executions`。这段历史状态不代表文首最新 dense 服务的执行计数。
