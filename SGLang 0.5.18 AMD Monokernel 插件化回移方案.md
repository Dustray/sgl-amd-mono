# SGLang 0.5.18 AMD Monokernel 插件化回移方案

更新时间：2026-11-05  
目标平台：ROCm、AMD Instinct MI350X/MI355X（gfx950）  
目标框架：SGLang 0.5.18（提交 `71de97b`）  
目标模型：本文默认 DeepSeek-V4.1-Flash；如果实际模型不同，kernel、权重格式和支持矩阵必须重新评估。

## 0. 当前工作目录的实施状态

本环境与本文固定的 kernel 支持范围存在实质差异：

| 项目 | 当前环境 | 本文固定版本 | 已核对的上游 Qwen kernel |
| --- | --- | --- | --- |
| GPU ISA | 海光 gfx936 | AMD gfx950 | AMD gfx950 |
| 模型架构 | Qwen3_5ForConditionalGeneration | DeepSeek-V4.1-Flash | Qwen4ExpForConditionalGeneration |
| 层数 / 权重 | 64 层，BF16 dense | DeepSeek 专用布局 | 48 层，Quark MXFP4/FP8 |
| TP / 推理阶段 | TP1，普通 decode | TP4，普通 decode | TP1，NEXTN verify |
| SGLang | 0.5.18 DTK 下游 wheel | 上游 71de97b | 依赖更新的 Monokernel API |

上游核对 revision 为 `ca929b7dcc4a955a7f4204d209fe303422a00351`。上游 Qwen kernel 的存在，不代表支持本地同属 Qwen3.8 命名系列的 27B checkpoint。

本目录已实现并安装 `sgl-amd-mono-sglang0518` **审计与实验性 FFN 融合插件**，入口为 `sgl_amd_mono_0518.bootstrap:register`。它使用原生 HookRegistry 接入配置检查、流式权重观察、模型完整 forward 和状态查询，校验当前 DTK wheel 版本及十一个关键源码 SHA256，未修改已安装的 SGLang 文件。

严格模式在加载权重前拒绝当前组合。显式设置 `SGLANG_AMD_MONO_0518_MODE=audit` 可验证插件加载与 Qwen 原始路径；报告保留 `execution_path=sglang_original`、`monokernel_eligible=false`、`monokernel_executions=0`。

**已完成 FFN 局部融合，完整 Monokernel 覆盖尚未完成。** `fused_ffn` 模式使用 gfx936 BF16 kernel，以每层两次 GPU launch 覆盖 64 层 dense FFN 的 decode batch 1/2；prefill、GDN、全注意力和 residual/norm 保留原始实现。当前包没有完整模型的可执行 spec，也未回移上游 runtime、peer memory、mailbox 或 fail-stop kernel。

运行与报告说明见 `README.md`，环境及上游 revision 清单见 `src/sgl_amd_mono_0518/manifest.json`。

已验证：严格模式在加载权重前失败；审计模式观察 1199 个 tensor，decode graph 捕获成功，HTTP 健康检查、模型列表及两条并发文本对话通过，`17 × 19` 返回 `323`。prefill/decode 报告均标记 `sglang_original`，Monokernel 执行次数为 0。该结果只完成接入验证，不构成阶段 B 的融合执行验收。

后续已安装 DTK FlyDSL `0.2.0+dtk2604.torch2110.2610101749.gad9161`（cp310），在 gfx936 实跑 BF16 MMA 和 SiLU-mul，并通过改变输入后的 HIP Graph 重放校验。原语测试生成代码调用 `llvm.hcu.mmac.f32.16x16x16.bf16`；新增 FFN 目前使用 FP32 向量累加。FlyDSL 编译与现有 CuTe 的 MLIR 冲突通过独立编译子进程及 HIP HSACO 加载解决。64 层真实权重数值验证全部通过；相同条件下 12 项生成对照有 9 项逐 token 一致，剩余差异保留记录，完整模型精度尚未验收。新核对的 27B dense PR #43558（head `b25e92fd224a196bf610ac05405073208ff7d513`）只支持 gfx950 / Quark MXFP4，常驻计划固定 256 个工作组，不能直接用于本机。详细依据和验证命令见 `Qwen3.8-27B 正式融合接入进度.md`。性能测试按用户要求暂停。

## 1. 结论

基于 SGLang 0.5.18 做“纯插件、零修改 SGLang 源码”的 Monokernel 回移，**技术上可行**，但不能直接安装当前 `sgl-amd-mono` 主分支。

推荐做法是构建一个专门面向 0.5.18 的兼容发行包，例如：

```text
sgl-amd-mono-sglang0518
```

这个 wheel 同时包含：

1. 固定版本的 AMD Monokernel kernel/spec，建议从上游框架 PR 验证过的 `sgl-amd-mono` 提交 `8f1bff3c06f8467cda1512122844d58d389ad7ca` 起步。
2. 从 SGLang Monokernel 框架 PR #44353 回移并改造的兼容层，包括 spec、guard、runtime 和 fail-stop 契约。
3. 面向 SGLang 0.5.18 的 HookRegistry 适配代码。
4. CPU 契约测试、ROCm 数值测试、CUDA/HIP Graph 测试、跨卡故障注入和端到端 serving 测试。

这里的“纯插件”指：不修改已安装的 SGLang 文件，不在容器启动时覆盖 `site-packages/sglang`。它不等于只写一个注册函数；插件内部仍需携带缺失的 Monokernel 框架兼容代码。

**不建议**把当前 `sgl-amd-mono` 主分支直接 `pip install` 到 0.5.18。当前插件依赖较新的 SGLang Monokernel API 和 kernel 模块布局，而这些接口在 0.5.18 中尚不存在。

## 2. 对 issue #60 的判断

issue #60 对 Monokernel 的原理、vLLM/ROCm 实现和 mailbox 同步机制总结得比较准确。对本项目最有价值的不是某个单独 kernel，而是以下四个工程约束：

1. 每个 TP rank 必须以相同顺序进入 Monokernel，并使用一致的 step/epoch。
2. 对称 peer memory 中的 mailbox 按 source rank 分区，每个 rank 向所有目标 rank 的对应分区写入 partial data 和 tag。
3. 双 slot 交替只能在所有参与者步调一致时避免下一 epoch 覆盖上一 epoch；它不是通用的异步消息队列。
4. 所有跨 CTA、跨 GPU 的自旋等待必须有 deadline、共享错误状态和 fail-stop，不能无限等待，也不能在部分 rank 失败后继续生成 token。

因此，将 Monokernel 接入 SGLang 的主要风险不是 Python 插件加载，而是：

- 权重和 KV cache 是否完全符合 kernel 假设；
- 每个 TP rank 是否作出相同的启用、路由和 fallback 决定；
- graph capture 与 replay 时是否只读取稳定的静态路由键；
- 任意 rank 或 workgroup 出错时，是否能在 watchdog 之前让整个 step 一致失败。

## 3. 固定版本基线

第一版不要追踪任何主分支，应固定下列基线：

| 组件 | 建议基线 | 说明 |
| --- | --- | --- |
| SGLang | `v0.5.18` / `71de97b` | wheel 和源码提交都要校验 |
| Monokernel 框架参考 | SGLang PR #44353 | 回移 spec、guard、runtime 和测试契约 |
| AMD 插件参考 | `8f1bff3c06f8467cda1512122844d58d389ad7ca` | 上游框架 PR 的准确性测试使用过该版本 |
| GPU | gfx950 | 初版只接受 MI350X/MI355X |
| 模型 | DeepSeek-V4.1-Flash | checkpoint revision 也要固定 |
| 并行配置 | 单机 TP4 | 先完成 TP4，再扩 TP2 |
| 推理模式 | 普通 decode | DSpark/speculative 放到第二阶段 |

除了版本号，还应保存以下信息：

- SGLang wheel 的 SHA256；
- 容器镜像 digest；
- PyTorch/ROCm 版本；
- FlyDSL、AITER 和相关 kernel 源码提交；
- 模型 checkpoint revision 和配置文件 hash；
- 每个生成的 HSACO/code object 的构建参数和 hash。

SGLang 0.5.18 的 ROCm 镜像早于上游 Monokernel 集成，不应假设其中的 FlyDSL/AITER 刚好满足新 kernel。必须把 SGLang API 兼容、PyTorch/ROCm ABI 兼容和 FlyDSL/AITER kernel 兼容分成三组独立测试。

## 4. 为什么 0.5.18 可以走纯插件

SGLang 0.5.18 已经包含通用插件系统：

```toml
[project.entry-points."sglang.srt.plugins"]
sgl_amd_mono_0518 = "sgl_amd_mono_0518.bootstrap:register"
```

它还包含 `HookRegistry`，支持 `BEFORE`、`AFTER`、`AROUND` 和 `REPLACE` 四种 hook，可作用于函数、静态方法、实例方法和类。插件在 engine 启动前注册并应用 hook，因此可以完成所需接入，而不修改 SGLang 包内文件。

0.5.18 缺少的是后来 PR #44353 引入的 Monokernel 专用框架：

- backend/spec 注册和 API 版本契约；
- 启动配置校验；
- checkpoint tensor 观察；
- 权重加载后、KV pool 定容前的构建时机；
- decode/verify graph key 路由；
- TP rank 一致性检查；
- 在线权重更新限制；
- fail-stop 和故障传播。

这些能力需要放进兼容插件，而不是假设 0.5.18 已经具备。

## 5. 推荐包结构

建议将框架兼容层和 AMD kernel 放在同一个发行包中，避免两个 entry point 的加载顺序不确定：

```text
sgl-amd-mono-sglang0518/
├── pyproject.toml
├── src/sgl_amd_mono_0518/
│   ├── __init__.py
│   ├── bootstrap.py
│   ├── config.py
│   ├── contract.py
│   ├── framework/
│   │   ├── spec.py
│   │   ├── guards.py
│   │   ├── runtime.py
│   │   ├── routing.py
│   │   └── fail_stop.py
│   ├── deepseek_v41/
│   │   ├── spec.py
│   │   ├── forward.py
│   │   ├── weights.py
│   │   └── gfx950/
│   └── common/
│       ├── peer_memory.py
│       ├── mailbox.py
│       └── sync.py
└── tests/
    ├── unit/
    ├── gpu/
    └── serving/
```

不要把兼容模块安装到 `sglang.srt.monokernels` 命名空间。这样做容易覆盖、混用或污染原始 SGLang 包。应修改回移插件的 import，让它只引用：

```python
sgl_amd_mono_0518.framework
```

## 6. 必需的 Hook 接入点

### 6.1 插件注册与配置

初版以环境变量作为主开关：

```bash
SGLANG_AMD_MONO_0518=1
```

可选地通过 hook 给 `ServerArgs` 增加兼容 CLI 参数，但环境变量更容易跨 launcher、scheduler 和 TP worker 进程保持一致。

建议 hook：

| 目标 | 类型 | 作用 |
| --- | --- | --- |
| `ServerArgs.add_cli_args` | `AFTER` | 可选增加 `--enable-amd-monokernel-0518` |
| `ServerArgs.from_cli_args` | `AFTER` | 将开关写入环境变量或插件全局配置 |
| `ServerArgs.check_server_args` | `AROUND` | 执行 fail-closed 的兼容性校验 |

无论通过 CLI 还是环境变量启用，最终配置都必须在子进程创建之前确定。禁止不同 rank 通过本地探测独立决定是否启用。

### 6.2 权重观察

上游框架要求在模型专用转换之前观察 checkpoint tensor，以便保留或重排 Monokernel 所需权重。0.5.18 可对以下方法注册 `AROUND` hook：

```text
sglang.srt.model_loader.loader.DefaultModelLoader.load_weights_and_postprocess
```

概念代码：

```python
def around_load_weights(original, model, weights, target_device):
    observed_weights = observe_checkpoint_tensors(weights)
    result = original(model, observed_weights, target_device)
    mark_weight_observer_completed(model)
    return result
```

`weights` 是迭代器，观察器必须保持惰性，不能无界地转成字典。只复制 spec 明确声明需要保留的 tensor。

必须拒绝任何绕过这个观察点的加载路径。第一版只允许已验证的默认/safetensors 路径；远端权重传输、量化在线转换、定制 loader 和热更新都先拒绝。插件需要在构建阶段检查 `observer_completed`，如果 spec 声明了派生权重但观察器没有执行，启动必须失败。

### 6.3 构建和安装 forward dispatcher

对下列方法注册 `AFTER` hook：

```text
sglang.srt.model_executor.model_runner.ModelRunner.initialize
```

0.5.18 的 `initialize()` 在返回前已经完成模型加载、MoE top-k 准备、layer 范围解析、post-load transform、LoRA 初始化和 KV dtype 配置；而 KV pool 初始化、attention backend 初始化和 decode graph capture 在后续阶段发生。这个位置适合完成：

1. 再次验证 model config、checkpoint、量化布局和 TP 配置。
2. 汇总每个 rank 的 spec lookup 结果和拒绝原因。
3. 从已加载模型和观察到的 checkpoint tensor 构建派生权重。
4. 分配对称 peer memory、mailbox、epoch 和 sticky error buffer。
5. 构建支持宽度集合，例如 decode rows `1..48` 对应的离散 graph keys。
6. 用保持原始函数签名的 dispatcher 包装模型 forward。
7. 向 `model_runner.capture_tail_hooks` 添加 capture 收尾检查或资源初始化钩子。

不要仅替换某个 decoder layer 的 Python `forward`。上游框架把路由放在模型完整 forward 边界，目的是让 eager、capture 和 replay 使用相同契约，并确保未覆盖形状完整回退到 SGLang 原路径。

### 6.4 Graph 路由

路由只能依赖启动时固定且可在 graph replay 中安全读取的 step key，例如：

```text
(mode, batch_rows, verify_tokens, tp_size, is_draft)
```

禁止根据 GPU tensor 的实时内容、某个 rank 的局部状态或 Python 侧临时分支决定是否进入 Monokernel。

每个 TP rank 必须在启动时交换并比较：

- spec 是否注册；
- spec 名称和 API 版本；
- 启用或拒绝结果；
- 支持的 graph key 集合；
- kernel/code object hash；
- 派生权重布局版本。

任一项不一致都应让所有 rank 在 serving 前失败。

## 7. 启动拒绝矩阵

插件必须 fail closed。第一版建议只接受以下配置：

| 维度 | MVP 接受值 | 其他值 |
| --- | --- | --- |
| SGLang | 精确 `0.5.18` 且 wheel hash 匹配 | 拒绝 |
| GPU | gfx950 | 拒绝 |
| 节点 | 单机 | 拒绝 |
| 模型架构 | 精确匹配的 DeepSeek-V4.1-Flash | 拒绝 |
| checkpoint | 固定 revision、config hash 和权重格式 | 拒绝 |
| dtype/量化 | 与 kernel spec 完全一致 | 拒绝 |
| TP | TP4 | 其他先拒绝；TP2 第二阶段加入 |
| PP/EP | PP1、EP1 | 拒绝 |
| DP/CP/DCP | 关闭 | 拒绝 |
| speculative | 关闭 | MVP 拒绝 |
| LoRA | 关闭 | 拒绝 |
| online weight update | 关闭 | 拒绝 |
| offload | 关闭 | 拒绝 |
| `torch.compile` | 关闭 | 拒绝 |
| two-batch overlap | 关闭 | 未验证前拒绝 |
| graph | 使用已验证的 capture 配置 | 配置不一致则拒绝 |

支持范围应由代码中的显式 `Requirements` 和 `config_disable_reasons` 生成，而不是散落在启动脚本的条件判断里。

## 8. SGLang 内部 API 契约

回移插件会读取大量 SGLang 内部字段，这些字段不是稳定公共 API。每个模块应显式声明读取集合，例如：

```python
SGLANG_READS = {
    "ModelRunner": {
        "model",
        "model_config",
        "server_args",
        "capture_tail_hooks",
        "is_draft_worker",
    },
    "ForwardBatch": {
        "forward_mode",
        "batch_size",
        "positions",
    },
}
```

CPU 契约测试至少检查：

- 模块和符号存在；
- 字段存在且类型/语义符合预期；
- hook 目标仍是预期的 staticmethod/实例方法；
- forward 签名没有漂移；
- loader 确实经过权重观察器；
- capture hook 的调用时机正确；
- DeepSeek-V4 模型层级和参数名符合 spec；
- 所有 fallback key 都能调用原始 forward；
- 插件关闭时没有行为变化。

版本判断不能只使用 `sglang.__version__ == "0.5.18"`。应同时校验关键 Python 文件 hash 或 wheel hash，防止使用同版本号的下游修改包。

## 9. 跨卡同步与 fail-stop

Monokernel 把 TP all-reduce 放入持久化 kernel 后，原有 collective 库不再替你处理所有失败场景。兼容插件必须实现：

1. 对称 peer memory 的分配、IPC handle 交换、导入和地址表校验。
2. 每个 source rank 独立的 mailbox 分区。
3. 双 slot 或更多 slot 的 epoch/tag 协议。
4. 所有 rank 完全一致的 epoch 推进规则。
5. 所有 spin-wait 的 deadline。
6. peer 可见的 sticky error word。
7. 超时或错误后阻止任何该 step 的 token 返回客户端。
8. 在 ROCm watchdog 之前触发一致的进程失败，而不是让某些 rank fallback、另一些 rank 继续执行 Monokernel。

需要特别测试以下故障：

- 一个 rank 不写 mailbox；
- 一个 CTA 不更新完成计数；
- 一个 rank 使用错误 epoch；
- 一个 rank 的 spec/hash 不一致；
- graph replay 使用了未注册宽度；
- peer mapping 缺失或地址不一致；
- kernel 在写 KV cache 前后分别失败。

## 10. 分阶段实施计划

### 阶段 A：兼容框架和拒绝逻辑

目标：不运行 Monokernel，也能证明插件加载、hook、contract 和 fail-closed 行为正确。

交付物：

- wheel 和 entry point；
- 0.5.18 精确版本/hash 校验；
- ServerArgs/loader/ModelRunner hook；
- spec registry、requirements、routing skeleton；
- CPU contract tests；
- 插件开关关闭时的回归测试。

### 阶段 B：TP4 普通 decode

目标：单机 gfx950、TP4、无 speculative，只覆盖普通 decode 的小 batch 宽度。

交付物：

- 权重观察和派生权重；
- peer memory 和 mailbox；
- K1/K2 kernel 构建；
- eager 模式逐层数值对比；
- graph capture/replay；
- 超出支持宽度时的原路径 fallback；
- 端到端生成正确性测试。

### 阶段 C：故障安全

目标：证明任何单 rank 故障都不会导致错误 token、静默挂起或 rank 分叉。

交付物：

- deadline 和 sticky error；
- rank-uniform startup failure；
- kernel fault injection；
- watchdog 前终止测试；
- serving 连接的错误传播测试。

### 阶段 D：TP2 和 DSpark

目标：在普通 decode 稳定后加入 TP2，以及 target verify 和 draft stages。

DSpark 需要单独处理 draft `ModelRunner`、draft attention backend、verify token 数和 EAGLE worker 生命周期。不能假设 target runner 的 hook 自动覆盖 draft runner。

### 阶段 E：性能和生产化

目标：证明收益在真实请求分布下成立，并完成发布、回滚和观测。

交付物：

- TPOT、inter-token latency、吞吐和显存对比；
- 并发 1/4/8、不同输入/输出长度矩阵；
- kernel fallback 计数、拒绝原因、deadline 和 fault 指标；
- 镜像 digest、wheel hash 和 checkpoint revision 清单；
- 一键关闭插件的回滚路径。

## 11. 验证矩阵

### 11.1 CPU 测试

- entry point 自动发现和 `SGLANG_PLUGINS` 白名单；
- 所有 hook 目标和签名；
- requirements 的每一种接受/拒绝路径；
- spec API 版本；
- rank lookup 结果序列化；
- graph key 集合一致性；
- 权重观察执行与绕过检测；
- forward 签名保持；
- fallback 覆盖；
- 插件禁用时零行为变化。

### 11.2 GPU 单元测试

- K1/K2 与原始 SGLang layer 的输出比较；
- KV cache 写入比较；
- attention/index cache 格式比较；
- 不同 batch rows、sequence length、page table stride；
- eager、capture、replay 三种路径；
- 支持宽度边界及超过边界后的 fallback；
- TP4 所有 rank 的输出和 epoch 一致；
- 连续多轮 replay 后没有 mailbox 污染。

### 11.3 Serving 正确性

可先沿用上游 PR 的验收思路：

- GSM8K 200 题 8-shot 准确率不低于约定下限，上游使用 `> 0.85`；
- 覆盖 step 的 token 分布相对正常路径 mean KL `< 0.03`；
- 开关前后的确定性样例逐 token 对比；
- batch 从小于最大宽度增长到超过宽度，再下降回来，确保 Monokernel/fallback 往返正确；
- 注入单 rank 故障时，所有 rank 停止且失败 step 没有 token 到达客户端。

### 11.4 性能验收

至少报告：

- TTFT；
- TPOT / inter-token latency 的 P50、P90、P99；
- 每用户 decode tok/s；
- 总输出 tok/s；
- GPU 显存和 KV pool 容量变化；
- Monokernel 命中率和 fallback 原因；
- 并发 1、4、8；
- 典型 8k/1k 和实际业务长度分布。

只比较 microbenchmark 不足以决定上线。Monokernel 的主要收益在小 batch decode，必须用真实 scheduler、KV cache、sampling 和请求涨落做 A/B。

## 12. 部署方式

### 12.1 镜像构建

示意 Dockerfile：

```dockerfile
FROM <sglang-0.5.18-rocm-base>@sha256:<image-digest>

COPY dist/sgl_amd_mono_sglang0518-*.whl /tmp/
RUN python -m pip install --no-deps /tmp/sgl_amd_mono_sglang0518-*.whl

ENV SGLANG_PLUGINS=sgl_amd_mono_0518
ENV SGLANG_AMD_MONO_0518=1
```

FlyDSL/AITER 如需更新，应使用独立、固定提交构建的内部 wheel 或系统包，并在镜像构建阶段跑 ABI smoke test。不要在生产镜像中安装 `latest`。

### 12.2 启动

MVP 启动示意：

```bash
export SGLANG_PLUGINS=sgl_amd_mono_0518
export SGLANG_AMD_MONO_0518=1

python -m sglang.launch_server \
  --model-path <deepseek-v4.1-flash-checkpoint> \
  --tp-size 4 \
  --host 0.0.0.0 \
  --port 30000
```

具体量化、attention backend、page size 和 graph batch size 参数必须来自已通过测试的 manifest，不应由运维人员自由组合。

### 12.3 回滚

回滚只需要移除或关闭插件：

```bash
unset SGLANG_AMD_MONO_0518
export SGLANG_PLUGINS=
```

但同一个进程内不要动态切换；应重启 server，确保 graph、peer memory 和 forward dispatcher 全部重新初始化。

## 13. 不建议的方案

### 13.1 直接安装当前 `sgl-amd-mono` 主分支

当前主分支面向较新的 SGLang Monokernel 接口，支持模型和依赖持续变化。直接安装到 0.5.18 很可能在 import、spec API、runtime 字段或 kernel ABI 上失败，即使可以 import，也不能证明路由和错误处理正确。

### 13.2 在容器启动时复制文件到 `site-packages/sglang`

这实际上是不可审计的源码打补丁，会造成 SGLang wheel hash 与运行内容不一致，也让升级、回滚和问题定位困难。

### 13.3 只 monkeypatch `model.forward`

这种方案没有权重加载观察、TP rank 一致性、graph key 管理、fallback 契约和 fail-stop，最危险的结果不是崩溃，而是静默生成错误 token。

### 13.4 第一版同时支持所有并行和 speculative 模式

Monokernel 与 TP、graph、KV cache、量化、attention backend 和 speculative runner 强耦合。扩大支持矩阵会使故障定位失控。应先固定单机 TP4 普通 decode。

## 14. 主要风险

| 风险 | 后果 | 控制措施 |
| --- | --- | --- |
| SGLang 内部 API 漂移 | import 或运行时字段错误 | 精确 wheel hash + CPU contract tests |
| FlyDSL/AITER/ROCm ABI 不匹配 | 编译失败、非法指令或错误结果 | 固定镜像和提交，构建时 ABI smoke test |
| checkpoint/量化布局不匹配 | 数值错误 | 固定 revision/hash，逐 tensor layout 检查 |
| 权重观察器被绕过 | 派生权重缺失或过期 | observer sentinel，启动时强制检查 |
| TP rank 路由分叉 | 死锁或错误 token | 启动 collective 比对 spec/key/hash |
| graph 捕获错误分支 | replay 时调用错误 kernel | 只使用静态 graph key，覆盖 capture/replay 测试 |
| mailbox epoch 不一致 | 无限自旋或读到旧数据 | deadline、双 slot/tag、sticky error、故障注入 |
| 显存额外占用 | KV pool 缩小、OOM | 构建前后测量，记录派生权重和 peer buffer |
| fallback 未充分验证 | batch 涨落时错误 | 覆盖边界往返和混合命中 serving 测试 |
| 在线更新改变权重 | Monokernel 使用过期副本 | 初版禁止在线权重更新 |

## 15. 最终建议

1. 可以坚持 SGLang 0.5.18，但要做一个专用兼容 wheel，不要直接使用 `sgl-amd-mono` 当前主分支。
2. 从上游验证过的 `sgl-amd-mono` 提交 `8f1bff3...` 和 SGLang PR #44353 开始回移，避免从当前多模型版本反向删功能。
3. 初版只做 gfx950、单机 TP4、DeepSeek-V4.1-Flash、普通 decode，并严格拒绝所有未验证组合。
4. 用 0.5.18 已有的通用插件和 HookRegistry 接入 loader、ModelRunner 和 ServerArgs，不改 SGLang 安装目录。
5. 把权重观察、rank 一致性、静态 graph 路由和 fail-stop 视为上线必需项，不是后续增强项。
6. 普通 decode 的准确性、故障安全和真实 serving A/B 全部通过后，再扩 TP2 和 DSpark。

这条路径不依赖安装 vLLM。vLLM 的实现适合作为 peer-memory、mailbox、fault handling 和测试设计的参考，但运行时仍是 SGLang。

## 16. 参考资料

- [pyhip issue #60：学习 Monokernel](https://github.com/tingqli/pyhip/issues/60)
- [SGLang v0.5.18 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.18)
- [SGLang 0.5.18 通用插件加载器](https://github.com/sgl-project/sglang/blob/v0.5.18/python/sglang/srt/plugins/__init__.py)
- [SGLang 0.5.18 HookRegistry](https://github.com/sgl-project/sglang/blob/v0.5.18/python/sglang/srt/plugins/hook_registry.py)
- [SGLang RFC #43949：Monokernel framework](https://github.com/sgl-project/sglang/issues/43949)
- [SGLang PR #44353：Monokernel framework](https://github.com/sgl-project/sglang/pull/44353)
- [早期 in-tree 实现 PR #43497](https://github.com/sgl-project/sglang/pull/43497)
- [sgl-project/sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono)

