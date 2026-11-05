# Qwen3.8-27B 正式融合接入进度

更新日期：2026-11-05。

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

## 正式融合仍需完成的实现与验收

GDN 接入点已定位为本机 `GDNAttnBackend.forward_decode`：卷积状态来自 `mamba2_layer_cache(layer_id).conv[0]`，SSM 状态为同一缓存的 `temporal`，请求槽映射来自 `forward_metadata.mamba_cache_indices`。后续融合必须保留 `_track_mamba_state_decode` 的状态跟踪语义，不能在写入部分状态后静默回退。目前没有替换该路径。

1. dense FFN 已通过 BF16 分块融合实现；继续覆盖输出投影和 residual/norm，并评估使用 HCU MMAC 及常驻工作划分的适用性。
2. 接入本地 SGLang 0.5.18 的 GDN 卷积状态、SSM 状态、请求槽映射、全注意力 KV 布局，以及 residual/norm 数据流。
3. 为跨工作组同步实现 HCU 内存可见性、epoch、deadline 与失败终止，避免部分状态更新后静默 fallback。
4. 在加载的真实 BF16 权重上验证 GDN、完整层和完整模型的 hidden states / logits / 生成 token；覆盖 graph 捕获和重放、batch 1/2 及回退往返。
5. 通过模型级数值与状态检查后，才启用正式融合模式，再进行端到端、prefill、decode 和长度矩阵的性能对比。

当前 `bundled_monokernel_specs` 仍为空；严格模式继续在权重加载前拒绝运行。新增的 `experimental_fusions` 注册 FFN 局部融合，不能用它宣称完整 Monokernel 已实现。`monokernel_executions` 保持为 0，真实 FFN 执行单独记录为 `fused_ffn_device_executions`。
