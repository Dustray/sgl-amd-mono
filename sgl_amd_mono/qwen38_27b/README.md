# Qwen3.8-27B：gfx936 开发适配

该目录是本机 Qwen3.8-27B BF16 模型实现的唯一维护位置，按上游的模型/spec/权重/ISA/runner/check 结构组织。对应 checkpoint 的 HF architecture 是 `Qwen3_5ForConditionalGeneration`，model type 为 `qwen3_5`。

## 文件职责与来源

| 路径 | 内容与来源 |
| --- | --- |
| `spec.py` | 模型、权重格式、TP、设备及状态布局准入；构造本机 runner |
| `weights.py` | 直接引用 SGLang 已加载的 BF16 gate/up 和 down 权重，检查格式与尺寸，不复制权重 |
| `gfx936/config.py` | 5120 hidden、17408 intermediate、64 层、GDN/全注意力布局及设备限制 |
| `gfx936/ffn_kernel.py` | 原本地 FlyDSL 向量 FMA FFN；两次发射，尚未换成上游 BF16 MFMA stage |
| `gfx936/compile_ffn.py`、`gfx936/ffn.py` | FFN 的隔离编译、HSACO 校验、HIP 加载与调用 |
| `gfx936/ffn_runner.py` | 在 64 层真实权重上逐一验证后接入 decode FFN |
| `gfx936/kernel.cpp`、`gfx936/program.py` | 本地 DTK HIP K1/K2 协作 kernel；gfx936 BF16 MMAC，复现当前 DTK 的原生归约和舍入路径 |
| `gfx936/attention.cpp`、`gfx936/attention.py` | 本地完整 full-attention K1：norm/QKV/QK norm/MRoPE/paged KV/softmax/gate 一次发射；复现原生 512-token 分段及中间舍入 |
| `gfx936/attention_check.py` | 安装版本 FA3、同精度分段参考与 FP32 参考；仅用于独立验证及 shadow |
| `gfx936/decode_io.cpp`、`gfx936/decode_io.py` | embedding 加首层 GDN K1，以及最终 norm 加整个 LM head 的 Tail，使用已加载的原权重 |
| `gfx936/runner.py`、`gfx936/check.py` | 旧版 SGLang 的层路由、状态对接及数值对照 |
| `check.py` | 可独立运行的 FFN 数值、执行计数和 Graph 重放验证 |

计算代码最初由原 `compat/sglang0518/` 迁入，没有把它标成上游原版。后续 kernel 修改直接维护在本目录。原路径保留导入转发，便于旧工具继续工作；新的实现不得反向导入兼容包。迁移前源码保存在 `standalone-snapshot-20261105`，目录整理前版本为 `source-managed-20261105`。

上游已有可供后续评估的 BF16 实现：`kimi_k3/gfx950/stages/gemv.py`、`kimi_k3/gfx950/stages/moe.py` 和 `kimi_k31/gfx950/stages/tile_mma.py`。这些实现使用 gfx950 的 MFMA、LDS 和线程布局。本机此前验证过的是 gfx936 的 `llvm.hcu.mmac.f32.16x16x16.bf16`；将上游 gfx950 stage 移植到这里需要重新验证指令 lowering、真实尺寸、舍入语义及 Graph。

当前 dense kernel 使用 DTK 的 `__builtin_hcu_mmac_f32_16x16x16_bf16`，编译汇编已确认发出 `v_mmac_f32_16x16x16_bf16`。每个 wave 同时计算 16 个输出通道及 batch 1/2；没有增加投影 launch，也不复制模型权重。

2026-11-05 后续数值排查发现，FP64 精确点积不能复现当前 rocBLAS 的 BF16 输出。最新代码已替换该修正：batch 1 的 qkv/z 和 gate/up 按原生输出 tile 复现 K 方向循环起点；输出/down 投影及 batch 2 的全部投影完整复现原生 4096 分段、BF16 MMAC 及两级 FP32 MMAC 归约。batch 1 的 b/a 投影采用原生 GSU8 的交错分段、循环起点及顺序合并；batch 2 的 b/a 同样走 split 路径。这些实现优先排查精度，尚未作为性能优化结果；整模型阈值须独立通过。这里依赖当前 DTK/rocBLAS 的实际调度，升级后必须重跑真实权重和整模型检查。

GDN 对齐当前 AITER/Triton 的 Q/K 成对归约、状态点积布局、状态更新 FMA、exp2 及补偿 ln2 乘法的 softplus 路径。decode batch 1/2 的 gated norm 对应原生 `ROWS_PER_BLOCK=1`，采用 64 线程各 2 元素的前缀归约；不能复用 prefill 的 16 线程各 8 元素布局。卷积使用原生近似倒数的 SiLU。Gemma norm 的纯输入和 residual 两条路径均按当前 LightOp 的 1024 个逻辑线程、每线程 8 元素复现归约，由 256 个物理线程执行；两者分别保留权重加一的 BF16 舍入和 residual 方差累加的 FMA 语义，并使用硬件 RSQ。这些均为本地实现，验证范围见根目录进度文档。

K1 额外保存 BF16 recurrent 输出供逐阶段检查，48 层合计增加 1.125 MiB 工作空间。C++ 参数结构与 Python ctypes 同步增加该指针，加载二进制时继续检查 ABI 大小及校验和；这不会增加 GPU launch 或复制权重。

## 接口与当前状态

当前安装的 SGLang 0.5.18 DTK 没有上游新版 `MonoSpec` API，因此 `spec.py` 提供本机 HookRegistry 接入所需的准入和构建函数，**不导出尚未验证的原生 `SPEC`**，也不将该模型加入新版 `SPECS` 注册表。根包的插件入口会在旧版环境下接入已安装的兼容桥，在有原生接口的环境继续使用原有注册流程。

`compat/sglang0518/` 负责旧版 hook、启动参数、SGLang 文件 hash 和进程报告。模型计算及其验证由本目录维护。

- `fused_ffn`：仅融合 FFN；prefill、attention、norm 等仍走 SGLang。
- `monokernel`：接受源码匹配的正式记录；开发中的实现须显式设置 `SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE=1`。
- `required`：只接受源码及环境匹配的 `acceptance.json`；实验开关不能绕过它。启动脚本默认使用该模式。

启动脚本默认采用已验证的 context 4096、静态显存比例 0.90，并关闭 overlap 与整模型 decode Graph；overlap 仍拒绝开启。选择 Graph 时传入 `--cuda-graph-backend-decode full`，要求验收记录包含 `dense_graph`，且 shadow steps 为 0。脚本在 HIP 初始化前设置 `HIP_GRAPH_ACCUMULATE_DISPATCH=0`。未验收 Graph 只能通过 `monokernel` 模式及单独的实验 Graph 开关运行。

2026-11-05 的当前记录已覆盖 129-launch 完整 decode：`dense_eager` / `dense_graph`、`full_attention_eager` / `full_attention_graph`、`decode_model_eager` / `decode_model_graph` 六项特征。`required` 自动启用 embedding、所有 decoder 层、最终 norm 与 LM head，无须实验开关。244 步同状态影子对照通过，逐请求最坏 logits NRMSE 为 `0.0020741280`，活动 GDN 状态零误差；eager 与 Graph 各 12 项实际输出均匹配同配置原生。具体证据和本机配置边界见根目录进度文档。112-launch 与 FFN 的旧记录保留为历史。

Graph 捕获通过 `hipGraphAddKernelNode` 显式添加节点、设置并回读 cooperative 属性，再更新捕获依赖。核内使用设备原子 epoch 栅栏，超时由查询到的设备恒定时钟计算，默认一秒；故障会锁存，模型及 Graph 返回后检查并拒绝部分更新的状态。80-block 全驻留条件仍通过 cooperative launch 和 occupancy 检查约束。旧硬件 grid sync 下的 SIGKILL 根因尚未确认，旧失败报告保留；新路径的实测支持范围见根目录进度文档。

`dense_decode_monokernel=true` 表示 dense decode 接入；应同时检查 `full_attention_fused`、`complete_decode_model` 和 `accepted_features`。完整配置为 48 个 GDN K1、16 个 attention K1、64 个 K2 和 1 个 Tail，每步 129 次 kernel launch；`whole_model_monokernel=true` 仅描述 decode 模型数学覆盖，正式可用性仍以源码匹配的验收特征为准。prefill、采样及调度/状态管理由原生执行，报告列出 `model_forward_modes` 与 `native_operations`；没有将整个模型称为一次发射。2026-11-06 同卡 18 组性能实测已完成，当前端到端耗时为原生的 2.89–12.10 倍，未获得加速；结果和计时口径见根目录 `Qwen3.8-27B 原生与Monokernel性能对比.md`。

此前 K2 阈值失败及后续长输入偏差的排查记录保留在进度文档。当前数值实现已变更，验收必须匹配最新源码指纹；旧 FFN 或失败中间版本的报告不能代替当前 dense 验收。

影子验证逐请求检查 logits NRMSE `≤0.02`、cosine `≥0.999`、KL `≤0.01`，完整配置还检查全部 129 个计算阶段的同输入误差及 48 层活动状态。attention 对照原生 FA3 和独立同精度参考，另将 FP32 误差限制在原生基线加 `0.003`；全词表 Tail 单独执行零误差回归。失败时补做原生重复执行，保存各层 K2 输入供重放。可选设置 `SGLANG_AMD_MONO_0518_CAPTURE_SHADOW_INPUTS=1`，还会保存原生层输入和 attention 分页 KV 历史供重放。报告记录实现 SHA256；更改源码后须重新验收。

## 安装与验证

从仓库根目录安装两个项目，不替换 Torch、FlyDSL 或 SGLang：

```bash
python -m pip install --no-deps --no-build-isolation -e . -e compat/sglang0518
python tools/source_status.py --verify
python -m pytest -q test/unit/test_qwen38_27b.py test/unit/test_sglang0518_compat.py
CUDA_VISIBLE_DEVICES=0 python -m sgl_amd_mono.qwen38_27b.check --full-shape --output monokernel_reports/qwen38_27b/ffn-check.json
```

GPU 命令应选择空闲设备。检查包含 256/512 和 5120/17408 两组尺寸、batch 1/2、eager 与改变输入后的三次 Graph 重放，阈值维持 NRMSE `0.002`。它验证 FFN，不代表完整模型或性能测试通过。

dense kernel 与整模型验证命令：

```bash
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python tools/verify_dense_monokernel.py \
  --model-path /public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master \
  --output monokernel_reports/qwen38_27b/dense-kernels.json

HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
SGLANG_AMD_MONO_0518_MODE=required \
SGLANG_AMD_MONO_0518_SHADOW_STEPS=512 \
SGLANG_AMD_MONO_0518_REPORT_DIR=monokernel_reports/qwen38_27b/shadow \
bash launch_qwen_monokernel.sh --host 127.0.0.1 --port 30128 \
  --cuda-graph-backend-decode disabled --context-length 4096

python tools/verify_dense_shadow.py --url http://127.0.0.1:30128 \
  --report-dir monokernel_reports/qwen38_27b/shadow \
  --output monokernel_reports/qwen38_27b/shadow-suite.json

HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python tools/verify_complete_decode_graph.py \
  --model-path /public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master \
  --output monokernel_reports/qwen38_27b/graph-chain.json

HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
HIP_GRAPH_ACCUMULATE_DISPATCH=0 \
SGLANG_AMD_MONO_0518_MODE=required \
SGLANG_AMD_MONO_0518_SHADOW_STEPS=0 \
SGLANG_AMD_MONO_0518_REPORT_DIR=monokernel_reports/qwen38_27b/graph-serving \
bash launch_qwen_monokernel.sh --host 127.0.0.1 --port 30129 \
  --cuda-graph-backend-decode full --context-length 4096

python tools/verify_monokernel_serving.py --url http://127.0.0.1:30129 \
  --shadow-report monokernel_reports/qwen38_27b/shadow/dense_model_shadow_validation.json \
  --output monokernel_reports/qwen38_27b/graph-serving-verification.json
```

`verify_complete_decode_graph.py` 使用独立的 64 层工作空间/状态、真实 embedding/head，复用少数层的真实 GDN、attention、K2 权重，验证全部 129 次发射；batch 1/2 Graph 交替重放，核对 eager 输出、状态、KV 与逐层执行计数。它验证执行结构；整模型数学另由真实全部权重的同状态影子对照确认。旧 `verify_dense_graph_chain.py` 保留为 112-launch 的诊断。`verify_dense_barrier.py` 检查同步超时及错误锁存。

服务工具要求原生参考的模型、TP、context、overlap、显存比例、KV 和分页等配置一致，失配时拒绝。本轮匹配基线位于 `monokernel_reports/qwen38_27b/full_attention_20261105/final-native-reference-matrix.json`，并同步到默认参考文件。`tools/verify_complete_serving.py --native-prefix-reference <native-prefix-reference.json>` 检查长短/串行/并发切换、完整逐层计数、prefill 路由、冷/热缓存分别匹配原生、流式输出和 chat；加 `--required` 会检查六项正式能力。

数值诊断工具包括 `verify_dense_attention_rounding.py` 的 190458 个两阶段舍入边界输入，以及 `diagnose_decode_head.py` 的全词表 MMAC 归约顺序检查。`replay_dense_attention.py` 可重放真实失败分页 attention 输入；原生二进制捕获工具只处理校验过的诊断副本，不改安装库。升级 DTK、Torch 或相关 native kernel 后须重跑这些检查。

`verify_dense_long_shadow.py` 在独立影子验证进程中先提交 128-token 请求，再交错中文/英文各 64 token，默认请求超时 600 秒。使用新的报告目录，设置 shadow steps 至少 140；验证记录不能刚好碰到采样上限。`verify_monokernel_serving.py --functional-only` 可单独检查服务，报告仍保留 `numerical_passed=false`，不会把功能成功冒充数值验收。

正式记录包含当前源码及适配层指纹、Torch/SGLang 版本和 HIP 编译器指纹。更新任一相关实现或环境后，原记录失配，`required` 会拒绝；需重新跑真实权重、同状态精度、服务和 Graph 验证后再生成新记录，不能仅更新指纹。

`tools/diagnose_dense_k2.py --captured-inputs <dense_failed_shadow_logits.pt>` 可在指定 `--layers` 上重放失败的真实 K2 输入。实验开关不会放行 `required` 模式。

`tools/diagnose_dense_layers.py --model-path <model> --captured-inputs <dense_failed_shadow_layers.pt> --layers 0 1 2 5 --output <report.json>` 从原生层输入及状态独立重放 K1/K2，并核对独立参考与 SGLang 实际输出；`--profile-native` 额外保存原生 kernel trace。诊断允许使用旧实现捕获的输入，因此同时记录捕获和当前实现的 SHA256，不能替代当前实现的整模型验收。

服务通过原有脚本默认选择正式 `required` 模式，也可显式选择 `audit`、`fused_ffn` 或开发实验模式；其 hook 会调用本目录的 `spec.build()`。后续修改保留在本目录的工作区，按用户确认流程提交；升级上游时按 `LOCAL_DEVELOPMENT.md` 审阅补丁并重新验收。
