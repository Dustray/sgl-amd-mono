# Flash-Next BF16 / TP8 / gfx936

本地 SGLang 0.5.18 DTK 后端，目标实现范围为 8 张 gfx936 / 80 CU、普通 decode batch 1/2、context ≤ 4096；完整精度验收尚未通过。实际验收进度见仓库根目录 `Qwen3.8-Flash-Next TP8 gfx936 Monokernel 接入进度.md`。

最新完整矩阵见根目录 `Qwen3.8-Flash-Next 完整精度验收.md`：7 个长度 / batch 组合的 472 个融合 decode forward 已测完，未满足全部验收门槛。HC 最后混合存在可复现的舍入差异；固定 HC / TopK 规则后，长输入仍有参考重放不一致，不能将全部偏差归因于融合 kernel。

2026-11-06 正确性补测：原生 / 融合的 13 个标准答案通过，512→64 严格逐步对照未通过；原生同状态重放也有 greedy 分歧，完整等价性待确认。证据见根目录 `Qwen3.8-Flash-Next 正确性验证.md`。`SGLANG_AMD_FLASH_SHADOW_RECORD_FAILURES=1` 仅用于收集失败后的诊断轨迹，失败会锁存为 `serving_shadow_failed`，不构成验收通过；默认仍拒绝失败请求。

后续修复对齐了 HC combine 归约和 QSA 短上下文顺序，局部失败也会锁存失败状态。原生 HC 固定输入重放会切换专家；超过 512 个压缩块时，原生 TopK 的零分并列选择也不确定。证据、已完成的修复和剩余限制见根目录 `Qwen3.8-Flash-Next 精度问题定位.md`。`SGLANG_AMD_FLASH_CAPTURE_QSA_STEP` 与 `SGLANG_AMD_FLASH_CAPTURE_QSA_LAYERS` 仅用于 rank 2 的定点抓取；默认关闭定点抓取，局部 QSA 失败自动保存现场。

| 文件 | 职责 |
| --- | --- |
| `kernel.cpp` / `program.py` | BF16 MMAC、HC、MoE、uncached TP8 IPC、cooperative 启动、错误锁存 |
| `gdn.cpp` / `gdn.py` | GDN K1、状态及原生参考 |
| `qsa.cpp` / `qsa.py` | QSA K1、KV / indexer cache、独立参考 |
| `ple.cpp` / `ple.py` | N-gram hash、分片 embedding、规约、PLE gate / short conv |
| `decode_io.cpp` / `decode_io.py` | Token embedding、最终 HC mix / LM head、全词表收集 |
| `runner.py` | 逐层真实权重验证、decode 路由、影子比较、设备执行计数 |
| `reference.py` | 显式启用的确定性 HC 累加 / QSA 并列选择参考；不是默认原生路径 |
| `accuracy_trace.py` | 保存独立服务实际外层 forward 的 logits 与输入元数据，排除影子重放 |
| `contract.py` / `native_sources.json` | 模型 / 并行 / cache / 服务契约、SGLang 接口指纹 |

数学路径每卡每步 99 次融合发射。Prefill、调度、采样、token history 元数据及检查点复制继续使用 SGLang。服务 Graph、MTP、量化及范围以外的配置会被拒绝。

从根目录运行 `bash launch_qwen_flash_next_monokernel.sh --max-running-requests 2` 和 `python3 tools/verify_flash_next_serving.py`。`whole_decoder_monokernel` / `acceptance_status` 及逐卡报告描述当前实例，成功编译不代表完成服务验收。
