# Qwen3.8-Flash-Next Monokernel 接入评估

评估日期：2026-11-06。范围为当前下载的 BF16 checkpoint、8 × gfx936 / 80 CU / 约 64 GiB 显存，以及当前安装的 SGLang 0.5.18 DTK 分支。源码依据 [UPSTREAM.lock.json](UPSTREAM.lock.json) 锁定的上游基线 `e52803f9d2b7dabff4bfd2f53ec25c03d56af947`；本地当前 HEAD 为 `f5c842b1db8a3c7ae332c8daee3c25ebdc4bdb73`。本次检查时，`sgl_amd_mono/qwen38_flash_next/` 相对 HEAD 无改动。

**结论：模型结构已有对应的上游实现，可以复用算法和测试设计；在当前机器上接入完整 Flash-Next Monokernel，难度高，属于增加 gfx936、多卡和 BF16 执行后端的工作。更换模型名称或放宽设备检查无法完成接入。**

## 服务状态与已有验证

已按要求向原生 Flash-Next 服务进程组发送 SIGTERM，PID `1600911` 已退出，`127.0.0.1:30160` 不再监听。停止后 GPU 0–7 利用率均为 0%，每卡显存占用 2 MiB，已清理失效的 PID 文件。

此前真实 BF16 权重已完成原生 TP8 启动和四组文本冒烟验证，运行时显存约 46.92–47.02 GiB/卡，见 [启动验证](Qwen3.8-Flash-Next%20启动验证.md)。这说明当前安装版具备该模型的原生文本推理路径。本次没有重新启动模型，没有执行 Flash-Next Monokernel，也没有验证原生 NEXTN/MTP 推理。

停止记录、停止后的显存快照和本次静态评估数据分别保存在：

- `monokernel_reports/qwen38_flash_next/20261106/stop-service.json`
- `monokernel_reports/qwen38_flash_next/20261106/gpu-stopped.json`
- `monokernel_reports/qwen38_flash_next/20261106/assessment.json`

评估数据包含源码 SHA256、权重 header 统计和安装版 SGLang 的接口路径检查。路径存在不代表接口兼容；本次没有做新内核的编译、完整 ABI 或数值验证。

## 现成实现覆盖什么

依据 [上游模型说明](sgl_amd_mono/qwen38_flash_next/README.md)、[target spec](sgl_amd_mono/qwen38_flash_next/spec.py) 和 [draft spec](sgl_amd_mono/qwen38_flash_next/draft.py)：

| 项目 | 现成 Flash-Next Monokernel | 当前环境 |
| --- | --- | --- |
| 模型结构 | hidden 2560、48 层，3 GDN + 1 QSA 循环，HC、PLE、512 experts / top-10 | 主要结构和维度一致，有直接参考实现 |
| GPU | gfx950 / MI355X | gfx936 / 80 CU，需要重做设备执行部分 |
| 权重 | Quark：MXFP4 experts、FP8 attention/GDN、FP8 PLE | 全 BF16，无 `quantization_config` |
| 并行 | TP/PP/CP/EP 均为 1，模型权重在 GPU | 已验证原生 TP8，单卡容纳不了当前权重 |
| target 路由 | NEXTN target verify，最多 8 请求，每请求 4 个 verify token | 已验证常规 prefill/decode；未验证 NEXTN |
| draft 路由 | MTP draft extend 和 draft decode | 安装版有对应 MTP 模型类，尚未启用验证 |
| 发射边界 | target 48 层共 13 次发射；draft 一层 1 次发射 | 尚无 Flash-Next Monokernel 发射 |
| 其余阶段 | prefill 和其他未覆盖阶段走原生；target logits processor / LM head 仍走 SGLang | 原生文本路径已跑通 |

因此，即使成功迁移现成实现，也应称为“覆盖上游规定的 NEXTN target verify 与 MTP draft 路径”。它不等于把 prefill、普通 target decode、采样和整个请求都变成单次 GPU 发射。若还要求普通 target decode 也走融合路径，需要单独设计对应入口及状态提交逻辑。

## 主要工作与难点

| 工作项 | 可以复用 | 当前必须补齐的内容 | 难度 |
| --- | --- | --- | --- |
| 模型算法 | 48 层分组、HC、GDN、QSA、PLE、MoE 路由与 tagged mailbox 的设计 | 多卡下的局部维度、数据归属和状态布局 | 中到高 |
| gfx936 计算 | 本地已验证 FlyDSL 基础 BF16 MMA、SiLU 与 Graph；27B 有 HIP/MMAC 实现经验 | 上游特定 MFMA、FP8 量化/投影及寄存器布局的替代实现，独立数值和吞吐验证 | 高 |
| 权重读取 | 权重检查、pointer table 和 packing 的组织方式 | BF16 projections / experts 的实际读取与计算路径；原有 MXFP4/FP8 参数和 scale 布局不能直接读取 BF16 | 高 |
| 常驻执行 | 上游分阶段任务与 guard；27B 的设备验证工具 | 根据 80 CU 重新设计任务、CTA、寄存器/LDS 和 mailbox；测量实际驻留率与同步进展 | 高 |
| 多卡执行 | 仓库其他模型的跨卡同步/规约模式，原生 TP8 模型的分片规则 | Flash-Next 专用 TP 设计、QSA/KV head 分配、MoE/PLE 分片、规约和错误传播 | 很高 |
| SGLang 0.5.18 | 原生模型、QSA、PLE、GDN、MTP 及本地插件 hook 经验 | Flash-Next 专用接入、cache view、NEXTN 生命周期、Graph 路由和源码版本约束 | 中到高 |
| 完整正确性 | 上游 check、kernel/e2e 测试思路 | 真实 BF16 逐阶段比对，accept/reject 后所有状态一致，多请求/Graph/长期运行验证 | 高 |

### 1. gfx936 不是改一个设备名称

[gfx950 kernel](sgl_amd_mono/qwen38_flash_next/gfx950/kernel.py) 的 FP8 projections 使用 `mfma_scale_f32_16x16x128_f8f6f4`，BF16 skinny GEMM 使用 `mfma_f32_16x16x32_bf16`，还依赖 FP8 转换、AITER shuffle 和特定 lane/register 布局。MXFP4 experts 会按其压缩格式解码，再参与 BF16 MFMA。

本地 27B 的 [gfx936 HIP kernel](sgl_amd_mono/qwen38_27b/gfx936/kernel.cpp) 使用 DTK 的 `__builtin_hcu_mmac_f32_16x16x16_bf16` 等指令接口。现有 FlyDSL 测试已经证明 gfx936 可以执行小型 BF16 MMA 和 Graph，所以不能笼统地说“FlyDSL 不支持 gfx936”；但该测试不等于验证了 Flash-Next 所用的完整 gfx950 指令链、tile 布局与持久化执行。

上游 [执行计划](sgl_amd_mono/common/plan.py) 和 [layout](sgl_amd_mono/qwen38_flash_next/gfx950/layout.py) 固定为 `256 CTAs × 512 threads`，设计上每 CU 一个 CTA，并要求所有 CTA 同时驻留以保证依赖等待能够推进。当前只有 80 CU，需要实测资源占用并重新调度。CU 数量本身不能单独证明一定死锁，但不能在未验证驻留和依赖进展的情况下照搬该执行配置，也不应直接修改所有模型共用的 `BLOCKS` 常量。

### 2. 量化 checkpoint 仍然不能直接解决单卡容量问题

对本地 131 个 safetensors 分片只读取 header，进一步区分 target/MTP experts 与纯 PLE embedding 后，得到以下容量估计：

| 权重部分 | 本地 BF16 有效载荷 | 按上游精度换算的理想有效载荷 |
| --- | ---: | ---: |
| target 48 层的 routed experts | 225.000 GiB | MXFP4：56.250 GiB |
| MTP 的 routed experts | 4.688 GiB | MXFP4：1.172 GiB |
| PLE embedding 表 | 95.368 GiB | FP8：47.684 GiB |
| 合计 | 325.055 GiB | **105.106 GiB** |

换算为 experts 字节数除以 4、PLE embedding 字节数除以 2。它不是实际量化 checkpoint 的测量结果；未计入量化 scales、expert padding、shared expert、attention/GDN、HC、embedding/head、KV/state 和 scratch。即便先不运行 MTP，仅 target routed experts 加 FP8 PLE embedding 也约为 **103.934 GiB**，已经超过本机单卡约 64 GiB。

因此，更换 Quark 权重后仍然需要多卡或新的 offload 设计。现成 [spec](sgl_amd_mono/qwen38_flash_next/spec.py) 明确限定 TP1，并且不允许 PLE embedding CPU offload；Flash-Next 的 [device wait](sgl_amd_mono/qwen38_flash_next/gfx950/device.py) 也按无 peer 的单卡路径实现。这是当前环境最重要的结构性障碍。

此前四卡 BF16 + PLE CPU offload 尝试在权重加载完成前被主动停止，没有证明四卡一定成功或一定 OOM；当前已经验证可用的配置是原生 TP8。CPU offload 若用于融合路径，还需要设计 host/device gather、数据预取和 Graph 交互，不能直接放开现有拒绝条件。

### 3. 0.5.18 的状态接口与投机生命周期需要专门桥接

静态检查发现，Flash-Next 声明依赖的安装源码中，以下路径在当前 SGLang 分支不存在：

- monokernel spec API：`kernels/plugin_kernels/amd/spec.py` 或 `srt/monokernels/spec.py`。
- binding/runtime API：`srt/kernel_plugins/amd/binding.py` 或 `srt/monokernels/runtime.py`。
- 参数字段目录 `srt/arg_groups/fields/`，以及 KV router `srt/mem_cache/kv/router.py`。

[runner](sgl_amd_mono/qwen38_flash_next/gfx950/runner.py) 通过 `token_to_kv_pool.layer_router.read(layer_id, "state" / "short_conv")` 获取状态；当前安装版在 `HybridReqToTokenPool` 提供 `mamba2_layer_cache()`、`short_conv_layer_cache()` 和 `short_conv_layer_intermediate_cache()` 等入口。整个安装版 mem_cache 源码搜索未发现 `layer_router`，需要适配真实对象、层映射、stride 和 scratch 归属。

当前 [0.5.18 bootstrap](compat/sglang0518/src/sgl_amd_mono_0518/bootstrap.py) 构建的是 Qwen 27B 路径，并对 speculative 等配置设有限制，不能自动接管 Flash-Next target/draft。模型原生 MTP 类 `Qwen4ExpForCausalLMMTP` 已存在于安装版，但原生 NEXTN 的运行行为尚未验证。

上游 verify 会产生每个候选 token 对应的 GDN、PLE、n-gram 中间状态，native accept/reject 再提交被接受位置的状态；MTP 还会读取 target residual streams。只检查生成文本正常，无法证明这些状态在拒绝、请求复用和 Graph replay 后仍然正确。

## 可选路径

| 路径 | 与现成实现的距离 | 判断 |
| --- | --- | --- |
| 保留当前 BF16 + gfx936 + TP8 | 设备、精度、并行和 API 均需适配 | 难度最高；能以当前已跑通的原生 BF16 结果作为直接对照 |
| 使用匹配的 Quark 权重，继续 gfx936 多卡 | 部分权重格式可复用；仍需设备、多卡和 API 移植 | 仍然是高难度，不是下载量化权重就结束 |
| gfx950、足够单卡显存、匹配 Quark 权重与 SGLang API | 与上游设计条件最接近 | 最接近现成路径，仍需在实际环境完成正确性和性能验收 |
| 先在 gfx936 做单阶段或单层融合，保留原生 MoE/通信等边界 | 可以分块验证并尽早检查性能 | 适合降低研发风险；中间结果必须标注为部分融合 |

若限定当前机器和当前 BF16 权重，建议以最后一条作为开发顺序，逐步完成第一条；保留现有 gfx950 实现，另建 `sgl_amd_mono/qwen38_flash_next/gfx936/`，将模型计算放在模型目录内，`compat/sglang0518/` 只处理旧版接口和生命周期。这样源码差异和将来的上游升级都更容易追踪。

## 建议的推进阶段与验收条件

| 阶段 | 要完成的内容 | 通过条件 |
| --- | --- | --- |
| 0：原生 NEXTN 基线 | 在现有原生 TP8 上启用 MTP/NEXTN，检查真实权重加载、page/cache 参数和 accept/reject | 多请求、长输入、连续生成正常；保存原生普通 decode 与 NEXTN 的独立结果和接受率 |
| 1：设备与容量原型 | 小尺寸 BF16 projection/MoE、HC/GDN 原型；明确 TP8 权重与状态布局，测量 CTA 驻留与等待 | 真实权重小样本数值通过；无资源超限或同步停滞；GPU 时间与原生算子有可比较结果 |
| 2：单层执行与跨卡 | GDN/QSA/PLE/MoE 分别完成，加入对应分片与规约；中间可保留原生计算边界 | 逐阶段输出、KV、GDN/PLE/n-gram 状态与原生 BF16 一致；多 rank 稳定，错误传播可验证 |
| 3：完整 target verify 与 MTP draft | 串联全部层，正确交付 logits/residual 与候选状态，完成 0.5.18 桥接 | 验证实际融合路径计数；accept/reject、并发切换、缓存复用、Graph replay 均通过，拒绝不支持配置 |
| 4：端到端验收与性能 | 独立生成/精度验证后，比较相同硬件和精度下的原生与融合 NEXTN；另列无投机基线 | 保存 E2E、TTFT/prefill、decode、TPOT、吞吐、接受率及不同输入/输出/并发的完整矩阵 |

上述阶段是工程拆分，不是已完成状态，也不是工期承诺。gfx936 指令/资源利用和多卡同步需要先通过小原型，才能可靠估算剩余时间。不建议在这些原型完成前承诺“改几个检查即可跑通”或确定的性能提升。

27B 已有的源码绑定验收、HIP/Graph 调用、guard、诊断和服务对照方法可以复用；其 dense FFN 和完整注意力实现不能直接代替 Flash-Next 的 MoE、HC 与 QSA。此前 [27B 实测](Qwen3.8-27B%20原生与Monokernel性能对比.md) 中，现有融合实现端到端仍比原生慢 2.89–12.10 倍，说明减少发射不保证整体加速；这不是 Flash-Next 的性能预测，但支持在阶段 1/2 设置性能检查点。

本次只完成服务停止、源码/容量/接口评估及文档更新；没有量化权重、下载新 checkpoint、修改 Flash-Next 计算实现或提交 Git。
