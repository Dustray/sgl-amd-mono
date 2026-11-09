# 本地改动归属

基线：`upstream-base-20261105` / `e52803f9d2b7dabff4bfd2f53ec25c03d56af947`。

本清单描述实际源码，不把先前自行实现的 BF16 kernel 称作上游原版。每项均可通过 Git 历史和相对基线的 diff 查看。

| 文件或模块 | 做了什么 | 来源及状态 |
| --- | --- | --- |
| `sgl_amd_mono/` 原有模型与 common | 保留上游模型/spec/kernel 源码 | 原有计算文件保持原样 |
| `sgl_amd_mono/qwen38_27b/` | 新增 Qwen 27B 模型目录，模型、权重、gfx936 计算及验证归位 | 本地实现，计算核心由先前实验迁入；尚未移植为上游 BF16 MFMA 实现 |
| `sgl_amd_mono/__init__.py` | 旧版缺少 MonoSpec API 时接入兼容桥 | 本地新增分支；原生 spec 注册行为不变 |
| 根 `README.md` | 加入本地开发说明链接 | 对上游文件的一处说明性修改 |
| `compat/.../bootstrap.py` | 接入配置、权重加载、ModelRunner 初始化/返回及状态查询五个 hook；Graph 返回后检查同步错误；拒绝 overlap | 自写兼容层；不覆盖安装目录的 SGLang 文件 |
| `compat/.../contract.py`、`manifest.json` | 固定 SGLang 版本、31 个文件 hash、hook 签名和模型/设备条件；新增 attention/MRoPE、embedding 和 logits 依赖 | 自写；旧上游引用和当前源码基线分别保存 |
| `compat/.../config.py`、`launcher.py`、`cli.py` | 模式选择、启动和预检 | 自写 |
| `compat/.../observer.py`、`report.py` | 观察权重并输出运行报告 | 自写，无权重副本 |
| `compat/.../bf16_ffn.py`、`compile_ffn.py`、`hip_ffn.py` 等旧计算模块 | 转发到模型目录，保留旧工具导入兼容 | 不再维护第二份计算实现 |
| `qwen38_27b/gfx936/ffn_kernel.py`、`compile_ffn.py`、`ffn.py` | 两次发射的 BF16 FFN、隔离编译和 HIP 调用 | 原自写 FlyDSL 向量 FMA，数值公式未因迁移改变 |
| `qwen38_27b/weights.py`、`gfx936/ffn_runner.py` | 权重检查、64 层真实权重验证和 decode 路由 | 无权重副本；旧生成仍存在差异 |
| `qwen38_27b/gfx936/kernel.cpp` | GDN K1 与输出投影到 FFN 的 K2；复现原生数值路径；显式构建 cooperative Graph 节点；使用有超时和错误锁存的软件 epoch 栅栏 | 自写 DTK HIP；工作区未提交；同卡性能对照已完成，当前未获得加速 |
| `qwen38_27b/gfx936/attention.cpp`、`attention.py` | 全注意力 K1：norm、QKV、QK norm、MRoPE、paged KV 写入、512-token 分段 attention 与 gate，合为一次发射；核对 batch 1 GSU4 和 BF16/FP8 舍入 | 本轮新增本地 DTK HIP；沿用 80-block cooperative 启动和有超时的 epoch 栅栏 |
| `qwen38_27b/gfx936/attention_check.py` | 安装版本 FA3 对照、独立同精度分段参考和 FP32 高精度参考 | 只在验证/影子模式调用原生算子，融合 kernel 内不回调原生 attention |
| `qwen38_27b/gfx936/decode_io.cpp`、`decode_io.py` | embedding 并入首层 GDN K1，最终 Gemma norm 和整个 BF16 LM head 合为 Tail；共 129 次 decode 发射 | 本轮新增本地 DTK HIP；直接引用现有权重，未新增权重副本 |
| `tools/verify_dense_attention.py`、`verify_dense_attention_guards.py`、`verify_dense_decode_io.py`、`diagnose_dense_attention.py` | 真实权重、分页布局、512 分段边界、状态拒绝/锁存、改变 token 的连续 Graph 重放与数值路径诊断 | 本轮新增验证；同状态整模型验收独立执行，尚非性能测试 |
| `qwen38_27b/gfx936/program.py`、`runner.py`、`check.py` | 协作启动、状态对接、逐层和影子验证 | 原自写实验代码；保留实验 opt-in 和数值拒绝逻辑 |
| `tools/verify_*.py`、`probe_cooperative.cpp` | 数值、状态、Graph、生成对照 | 自写验证，不代表性能测量 |
| `tools/source_status.py` | 显示 Git 基线、本地 diff、实际导入路径和 SGLang 契约 | 本次源码管理新增 |
| `tools/diagnose_dense_k2.py`、`tools/diagnose_dense_layers.py`、`tools/verify_dense_shadow.py` | 逐阶段数值诊断、原生层输入及活动状态捕获/重放、串行/并发整模型对照 | 本轮新增，报告包含源码指纹，未提交 |
| `tools/verify_dense_graph_chain.py` | 64 层独立工作空间/状态、三组真实权重；共存的 batch 1/2 Graph 交替重放，每次逐位核对 eager 输出、状态及逐层计数 | 连续协作发射验证，不代替真实 attention 的整模型验收 |
| `tools/verify_dense_barrier.py` | 故意缺少一个 block，验证约一秒退出、错误锁存和独立新状态恢复；覆盖 eager/Graph | 实际 GPU 故障注入，不复位设备 |
| `tools/verify_dense_attention_rounding.py` | 在 FP16 舍入中点附近检查生产 QK 分数 helper，覆盖 eager 与改变输入的 Graph | 190458 个输入，独立 CPU 两阶段舍入参考 |
| `tools/diagnose_decode_head.cpp`、`diagnose_decode_head.py` | 检查原生 LM head 的 1024 步长、BF16 MMAC 对角元素及四组部分和归约次序 | 六组全词表输入、四种循环归约次序；生产采用唯一零误差的 `3→0→1→2` 顺序 |
| `tools/replay_dense_attention.py`、`diagnose_attention_order.cpp`、`diagnose_attention_order.py` | 重放真实失败 attention 输入；诊断 MMAC 的通道、K 方向排列与原生分数 | 不进入生产计算路径；支持独立移除 gate 的诊断 |
| `tools/capture_native_attention_launch.cpp`、`capture_native_attention_scores.py`、`capture_native_attention_scores.s` | 只在独立诊断进程捕获 HIP 参数和分数；按原始 SHA256 校验后修改 HSACO 的诊断副本 | 不修改安装库；具体地址仅适用于已校验的 gfx936 二进制 |
| `tools/verify_complete_decode_graph.py` | 独立 64 层工作空间和状态、真实 embedding/head、复用部分真实层权重；129 次发射的 batch 1/2 Graph 交替重放 | 核对 eager/Graph 输出、状态、KV 和每个 kernel 的执行计数；独立整模型精度另测 |
| `tools/benchmark_qwen38_serving.py` | 同卡原生/完整 Monokernel 性能矩阵；固定 token、清空缓存、三轮采样、流式端到端及阶段计时 | 原始 JSON/CSV、18 组 Markdown 表格；逐层计数、输出长度和配置一致性校验 |
| `launch_qwen_flash_next.sh`、`tools/smoke_qwen38_flash_next.py`、`Qwen3.8-Flash-Next 启动验证.md` | 下载的 BF16 Flash-Next 的原生 TP8 启动、文本冒烟测试和运行记录 | 本地新增运行工具；原有 gfx950 Flash-Next monokernel 与本次 BF16/gfx936 启动条件分开说明 |
| `Qwen3.8-Flash-Next Monokernel 接入评估.md` | 实现之前的 gfx936 / BF16 / TP8 容量和旧版接口评估 | 历史评估；当前实现和验收进度见 `Qwen3.8-Flash-Next TP8 gfx936 Monokernel 接入进度.md` |
| `sgl_amd_mono/qwen38_flash_next/gfx936/` | 新增 BF16 / TP8 / gfx936 普通 decode：embedding、36 GDN、12 QSA、48 HC/MoE、PLE、最终 HC/LM head 与核内通信 | 每卡每步 99 次数学融合发射；保留原生 prefill、采样、调度与状态元数据。既有 gfx950 实现未改 |
| `compat/.../flash_next.py`、`launch_qwen_flash_next_monokernel.sh` | 独立插件及服务入口；固定模型、TP8、依赖源码和当前验收范围 | 不修改安装的 SGLang 源文件；拒绝量化、MTP、服务 Graph 和超出范围的参数 |
| `tools/verify_flash_next_{tp8,hc,moe,gdn,qsa,ple,io,serving}.py`、`test/unit/test_qwen38_flash_next_gfx936.py` | 独立数值/状态/Graph 测试与完整模型同状态 logits / 执行计数验证 | 使用真实权重；PLE 独立测试使用小型合成 embedding 表，完整大表由服务影子测试验证。不是性能测试 |
| `tools/benchmark_flash_next_serving.py` | Flash-Next TP8、batch 1、512→64 的原生 / Monokernel 请求级实测 | 相同输入与服务配置，2 次预热、5 次正式测量；预热保留首次影子验收，正式计时检查无影子执行并核对八卡 99 个数学核计数 |
| `tools/verify_flash_next_correctness.py` | 两个独立服务的 13 个标准答案用例及计划 63 步的 512→64 数值对照 | 固定相同 input_ids、温度和配置；逐步绑定 token / 长度并核对核计数；保留严格失败与诊断轨迹，不放宽门槛。本次严格运行第 3 步失败，继续诊断在第 40 步被局部保护终止 |
| `qwen38_flash_next/gfx936/kernel.cpp`、`qsa.cpp`、`qsa.py` | 对齐原生 HC combine 归约与短上下文 QSA 块顺序；独立参考使用真实 `fast_topk` | HC combine 真实失败样本修复后逐位一致；QSA 短上下文 96 项通过。新服务仍被另一 HC / MoE 超限终止，未完成整模型精度验收 |
| `tools/diagnose_flash_next_{hc,qsa}.py`、`diagnose_flash_next_inject.cpp`、`runner.py` | 真实失败输入的 HC / QSA 分段回放、定点抓取与局部失败状态锁存 | 直接调用生产 inject；固定输入证实原生 HC 会切换专家，长上下文原生 TopK 的零分并列集合也会变化。详见 `Qwen3.8-Flash-Next 精度问题定位.md` |
| `gfx936/reference.py`、`accuracy_trace.py`、`tools/*flash_next*accuracy*.py` | 可选的确定性 HC / QSA 参考、独立服务实际 logits 抓取、batch 1/2 完整长度矩阵与逐步核计数审计 | 两边显式启用同一参考策略；默认策略不变，SGLang 安装文件不改写。诊断模式也保留失败判定，详见 `Qwen3.8-Flash-Next 完整精度验收.md` |
| `tools/diagnose_flash_next_hc_stages.{py,cpp}`、`test/unit/test_flash_next_accuracy_trace.py` | 固定 HC 输入的 normalized / lowrank / raw gates / 最终混合分段回放，9 种 sigmoid / 归约组合；验收失败与历史前缀绑定回归 | 第 14 层样本的前三段逐位一致，差异定位到最后混合。候选只作单样本实验，生产数学核未据此修改；当前相关 CPU 回归共 53 项通过 |
| `tools/verify_complete_serving.py` | 完整 decode 服务的串行/并发切换、prefill 路由、长输入缓存复用、流式和 chat | 每阶段核对 129 次发射与全部逐层计数；支持无实验开关的严格入口验收 |
| `tools/verify_dense_long_shadow.py` | 固定先提交 128-token 请求，与中文/英文各 64 token 交错；验证端点属于指定进程并保存部分结果 | 独立长生成数值验证，避免按历史完成顺序重放请求 |
| `qwen38_27b/acceptance.py`、`acceptance.json` | 正式支持记录绑定模型源码、适配源码、Torch/SGLang 与编译器；源码失配后严格模式拒绝 | JSON 只在对应验证完成后生成；开发实验入口保留 |

表中的 `compat/.../` 均指 `compat/sglang0518/src/sgl_amd_mono_0518/`。
表中的 `qwen38_27b/` 均指 `sgl_amd_mono/qwen38_27b/`。

打包校验发现根目录历史 `build/lib/sgl_amd_mono_0518` 会把迁移前的适配层混入模型 wheel。根项目改为只发现 `sgl_amd_mono` 及其子包，历史生成目录移到报告目录备份，重新构建后分别核对模型与兼容 wheel 的源码指纹，并拒绝两个 wheel 包含同一路径。未安装这些旧 wheel，也未修改当前 SGLang。

前一轮的 251 步 eager 验收与 Graph 中断保留在 `numerics_20261105/`，112-launch 基线在 `production_20261105/`。完整 129-launch decode 的代码、精度及服务结果在 `full_attention_20261105/`：244 步同状态对照、逐请求最大 logits NRMSE `0.0020741280`、活动 GDN 状态零误差，eager/Graph 输出分别与同配置原生 12/12 项匹配。当前 `whole_model_monokernel=true` 描述 embedding 到 logits 的 decode 数学覆盖；prefill、采样及调度/缓存管理保持原生。默认关闭 overlap/Graph，可显式选择已验收的 Graph；2026-11-06 已完成同卡 18 组[性能实测](Qwen3.8-27B%20原生与Monokernel性能对比.md)，当前 Monokernel 端到端耗时为原生的 2.89–12.10 倍。历史原生基线的 context/overlap 配置与当前不同，已备份并重建匹配基线；服务工具现会拒绝配置失配。

## 提交分层

1. `standalone-snapshot-20261105`：原样导入迁移前适配、实验代码、启动脚本和文档，方便追溯。
2. 源码管理提交：固定真实上游基线，补充本地说明、来源检查、忽略规则及源码安装路径。
3. 实验隔离提交：给未验收 dense 原型增加显式 opt-in，防止默认严格模式被误认为已正式支持。
4. 模型目录提交：把计算实现迁到 `sgl_amd_mono/qwen38_27b/`；适配层保留旧入口，固定模型维度与层布局检查，并提供独立验证入口。该提交不表示完整 Monokernel 已验收。

后续升级直接在此 Git 仓库提交修改。计算 kernel、兼容 API 和验证变更应分别提交；提交说明需记录变更原因、验证命令及结果。运行日志和编译缓存留在忽略目录，不随补丁混入。

2026-11-05 后续开发按用户要求保留在工作区，**每次提交必须先由用户确认**。使用 `git diff` 查看本轮已跟踪文件修改，`git status --short` 同时显示新增文件；`git diff upstream-base-20261105 --` 查看包含未提交修改的整体差异。当前 HEAD 未因本轮开发改变。

## 全推理 Profile

新增 `tools/profile_flash_next_serving.py`、`sgl_amd_mono/common/profiling.py` 和 `profile_trace.py`；通过 SGLang 原生 profiler 接口及进程内插件采集文本分词、prefill、逐步 decode、GPU kernel 与反分词，导出八卡合并的 Perfetto 兼容 trace 和算子 CSV。采集标记仅在 `SGLANG_AMD_PROFILE_RANGES=1` 时安装，不修改安装目录或模型数学核。使用与计时边界见 `PROFILING.md`。
