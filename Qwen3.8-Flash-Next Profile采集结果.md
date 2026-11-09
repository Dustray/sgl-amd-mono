# Qwen3.8-Flash-Next TP8 / gfx936 完整推理 Profile

2026-11-06，原始 BF16 Flash-Next，8 张 gfx936，batch 1，512 输入 token → 64 输出 token。原生与融合分别启动服务、预热两次，再各采集一次完整文本请求。输入 shapes 开启，源码栈关闭，Graph / overlap / radix cache 关闭。

两份文件均已通过真实 **Perfetto Trace Processor v59.1** 导入和事件覆盖核验。采集与分析工具使用说明见 [PROFILING.md](PROFILING.md)。

## 文件

| 模式 | 合并文件 | 压缩大小 MiB | 全部事件数 | GPU kernel 事件数 | CPU 算子事件数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 原生 SGLang | [原生 trace](monokernel_reports/flash_next_tp8/profile_20261106_b1_512_64_complete/native/inference.perfetto.json.gz) | 573.93 | 18,329,535 | 1,528,232 | 7,375,232 |
| Monokernel | [融合 trace](monokernel_reports/flash_next_tp8/profile_20261106_b1_512_64_complete/monokernel/inference.perfetto.json.gz) | 57.27 | 1,907,805 | 162,392 | 793,054 |

格式是 gzip 压缩的 Chrome Trace Event JSON，可由 Perfetto 导入。原生八卡文件较大，建议使用文档中的外部 Trace Processor 方式打开；也可以先打开 `ranks/` 内单卡原始文件。

数据根目录：`monokernel_reports/flash_next_tp8/profile_20261106_b1_512_64_complete/`。

- `README.md`：两种模式的采集概况及 rank 0 GPU kernel 耗时前十。
- `native/`、`monokernel/` 下的 `operators.csv`：八卡各算子的调用数、累计、平均、最小、最大时间，单位 μs。
- 各模式下的 `summary.json`：每卡事件数、阶段计数、GPU 累计时间与影子执行状态。
- `ranks/`：完整八卡原始 PyTorch profiler 文件；`host/`：分词、反分词及客户端时间线原始事件。
- `capture.json`、`requests.json`、`launch.json`、`manifest.json`：采集配置、请求、服务生命周期与源码 SHA256。
- 各模式下的 `perfetto-validation.json` / CSV / log：真实 Perfetto 的核验结果和健康统计。

## 覆盖核验

| 项目 | 原生 | Monokernel |
| --- | ---: | ---: |
| GPU 进程 / 卡数 | 8 | 8 |
| 每卡 Prefill forward | 2 | 2 |
| 每卡 Decode forward | 63 | 63 |
| 实际文本分词 | 1 | 1 |
| 反分词区间 | 64 | 64 |
| 客户端 token 到达事件 | 64 | 64 |
| CPU → GPU flow 关联 | 1,604,456 | 121,184 |
| 输入 shape 参数条目 | 16,549,663 | 1,579,917 |
| 落在客户端请求区间之外的模型 forward | 0 | 0 |
| 落在客户端请求区间之外的前后端区间 | 0 | 0 |

Perfetto 导入后的 GPU / CPU / memcpy 事件数量与原始 trace 统计逐一匹配。八卡的 Prefill 合计 16 次、Decode 合计 504 次；最后一次 Prefill 生成首 token，因此每卡后续有 63 次 Decode。

## 融合执行粒度

时间线保留层级标记、ATen、HIP API、GPU kernel、stream、memcpy 和 launch flow。融合 Decode 的主要数学 kernel 在每卡实测为：

| Kernel | 63 步累计调用数 / 每卡 | 每步调用数 |
| --- | ---: | ---: |
| `gdn_attention(GDNArguments)` | 2,268 | 36 |
| `qsa_attention(QSAArguments)` | 756 | 12 |
| `post_attention(PostArguments)` | 3,024 | 48 |
| `ple_decode(PLEArguments)` | 63 | 1 |
| `decode_entry(EntryArguments)` | 63 | 1 |
| `decode_tail(TailArguments)` | 63 | 1 |
| 合计 | 6,237 | 99 |

总时间线也包括原生 Prefill、采样和缓存 / 调度辅助操作，因此全部 GPU kernel 数量高于上述数学核总数。融合核内部的 HC、router、专家 GEMV 和 TP peer 通信等待仍包含在融合 kernel 内，当前未单独拆分这些内部阶段的耗时。

## 数据边界

原始 DTK profiler 文件存在部分完整区间相互交叠。Perfetto 在原生 / 融合合并文件中分别报告 `149574` / `5581` 次 `slice_spill_overlapping_complete_event`，将这些区间放到 overflow 轨道显示；事件被保留。原始未合并单卡文件也复现此项。验证器完整记录该统计，并拒绝其他未解释的 error / data_loss。

融合预热后的局部 / 整模型影子检查数量分别保持在 `99` / `1`，正式采集期间均未增长。精度状态仍为 `serving_shadow_failed`，完整精度结论见 [精度验收](Qwen3.8-Flash-Next%20完整精度验收.md)。这些 profile 用于执行路径与耗时定位；采集开销包含在时间中，性能结论应使用关闭 profiler 的 benchmark。

84 项相关 CPU 回归通过；源码指纹已和运行 manifest 核对一致。两条服务均已退出，30160 端口关闭。Git HEAD 保持 `f5c842b1db8a3c7ae332c8daee3c25ebdc4bdb73`，本轮没有提交或暂存修改。
