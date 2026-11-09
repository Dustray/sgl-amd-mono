# 全推理时间线采集

当前入口针对本地 Qwen3.8-Flash-Next BF16、TP8、gfx936 服务。每次启动一个独立服务，预热后从 HTTP 请求开始前采集，直到全部输出 token 返回后停止。默认依次采集原生与 Monokernel，结束或异常时关闭自己启动的服务。

本机已完成并通过真实 Perfetto 导入核验的 512→64 样例见 [采集结果](Qwen3.8-Flash-Next%20Profile采集结果.md)。

```bash
python3 tools/profile_flash_next_serving.py \
  --mode both \
  --input-tokens 512 \
  --output-tokens 64 \
  --batch-size 1 \
  --output-dir monokernel_reports/flash_next_tp8/profile_b1_512_64
```

输出目录必须不存在。默认使用端口 30160，端口被占用时拒绝启动；可以通过 `--port` 指定其他端口。支持 `--mode native`、`--mode monokernel` 和 batch 1/2；输入与输出长度之和不超过 4096。batch 2 为同时发起两个独立请求，每个请求使用指定输入输出长度。

也支持连接已经启用采集标记的服务。该服务启动时须设置 `SGLANG_AMD_PROFILE_RANGES=1` 和 `SGLANG_AMD_PROFILE_EVENT_DIR=<绝对路径>`；工具验证实际执行模式，并只复制本次时间窗口内的前后端事件。此模式保留服务运行：

```bash
python3 tools/profile_flash_next_serving.py \
  --mode native --url http://127.0.0.1:30160 \
  --host-event-dir /path/to/server-profile-events \
  --input-tokens 512 --output-tokens 64 \
  --output-dir monokernel_reports/flash_next_tp8/attached_profile
```

## 文件与打开方式

```text
profile_b1_512_64/
  manifest.json
  README.md
  native/
    inference.perfetto.json.gz
    operators.csv
    summary.json
    capture.json
    requests.json
    server_info_after.json
    launch.json
    server.log
    ranks/*TP-*.trace.json.gz
    host/host-*.jsonl
  monokernel/
    ...
```

用 Perfetto UI 的 **Open trace file** 打开 `inference.perfetto.json.gz`。这是 gzip 压缩的 Chrome Trace Event JSON，供 Perfetto 直接导入。也可先执行 `gzip -dk inference.perfetto.json.gz` 后导入 JSON。

八卡详细 trace 可能达到数百万到上千万事件。浏览器内存不足时，可以先打开 `ranks/` 中的单卡原始文件，或在有足够内存的机器上运行原生 Trace Processor，再使用 Perfetto UI 的外部处理器连接：

```bash
trace_processor_shell server http --ip-address 127.0.0.1 \
  monokernel_reports/flash_next_tp8/profile_b1_512_64/native/inference.perfetto.json.gz
```

每份合并文件包含八卡时间线及前后端 CPU 范围；`ranks/` 保留每卡原始 PyTorch profiler 文件。合并视图默认省略 Python interpreter 的逐函数 `python_function` 区间，保留 ATen、模块、HIP、GPU、输入 shapes 等明细；完整逐函数数据仍在原始文件中。可以用 `--keep-python-functions` 保留全部区间，文件会显著增大。合并时保留相同主机的时钟关系，使用 `baseTimeNanoseconds` 校准各 trace 的基准，重新分配进程、线程及 flow ID，避免不同卡的关联线互相串接。

已保存的采集可离线重新合并并生成统计，无需再次加载模型：

```bash
python3 tools/summarize_serving_profile.py \
  --run-dir monokernel_reports/flash_next_tp8/profile_b1_512_64
```

先采集了单个模式时，可用 `--resume --mode monokernel` 或 `--resume --mode native` 向同一输出目录追加另一种模式。工具拒绝覆盖已有模式，并保留之前的启动与源码记录。

`operators.csv` 按 rank、类别、原始算子名给出调用次数、累计、平均、最小与最大耗时，单位 μs。`summary.json` 包括每卡 prefill / decode forward 数量、各类别事件数、分阶段 GPU kernel 累计时间、未归属 forward 的 GPU kernel 数量、前后端标记数量、精度影子执行数量和融合路径状态。

仓库提供 `tools/profile_queries.sql`，可在 Perfetto 的 SQL 查询窗口逐条执行，查看全局算子、各卡 kernel 累计耗时及导入错误。安装 Perfetto Trace Processor 后，也可离线查询：

```bash
trace_processor_shell query -f tools/profile_queries.sql \
  monokernel_reports/flash_next_tp8/profile_b1_512_64/native/inference.perfetto.json.gz
```

也可用真实 Perfetto 解析器自动核对 GPU / CPU / memcpy 事件数、八卡数量、完整 forward、分词与反分词、客户端 token、输入 shapes 以及 flow 关联。验证结果与导入健康统计保存在各模式的 `perfetto-validation.json` / CSV / log 中：

```bash
python3 tools/validate_serving_profile.py \
  --run-dir monokernel_reports/flash_next_tp8/profile_b1_512_64 \
  --trace-processor /path/to/trace_processor_shell
```

## 可见细节

- 请求：客户端 HTTP generate 区间、每个返回 token 的到达时刻。
- 前后端：真实文本 tokenization、请求发送到 scheduler、逐步 detokenization。前端标记是 Python 方法的墙钟区间，包括异步方法中的等待。
- 推理阶段：`inference/EXTEND/step=.../batch=...`、`inference/DECODE/step=.../batch=...`。EXTEND 对应原生 chunked prefill，512 输入在当前 256 chunk 配置下分成两次 forward。
- 模型层：`module/model.layers...`，包含模块类型；HC 的 `mix` / `combine` / `inject` 额外标记。
- 原生算子：ATen CPU 操作、输入 shapes、HIP runtime API、Triton / HIP GPU kernel 名、stream、memcpy、CPU 到 GPU 关联线。
- 融合算子：`mono/layer.XX/gdn`、`qsa`、`posts`、`ple`、最终 HC、entry / tail 与错误检查。对应 GPU 轨道显示真实融合 kernel。
- `--with-stack` 开启 PyTorch profiler 源码栈采集，会增加采集开销与文件体积；输入 shapes 始终采集。

融合核内的 HC、router、专家 GEMV、归约等数学阶段在同一次 GPU 发射里执行。当前文件记录该融合核的整体执行时间，TP peer 通信与等待也包含在融合核内部，尚未进一步拆分内部阶段的耗时。

## 采集与验证边界

采集使用 SGLang 自带 `/start_profile`、`/stop_profile`，启用 CPU 与 GPU activities；模型和前后端标记通过本仓库插件在进程内安装。安装目录中的 SGLang 源码及模型数学核保持原样；环境变量 `SGLANG_AMD_PROFILE_RANGES=1` 才会安装这些标记。

预热使用真实文本请求，固定 token 数并验证 tokenizer 往返一致。每次采集前清空缓存，实际请求检查输入输出数量、缓存命中与重排。合并阶段要求八卡齐全、每卡均有 GPU kernel、完整 prefill 与指定数量的 decode forward。缺卡、只有 CPU、缺少时钟基准或 decode 覆盖不足会判为采集失败。

融合服务保留启动检查和首个 decode 的精度影子检查，预热后不重新触发影子检查。工具使用已有诊断模式收集已知精度失败，状态仍记录为失败；采集前后要求影子检查次数不增长。完整精度验收结论仍见 `Qwen3.8-Flash-Next 完整精度验收.md`，profile 成功不代表精度通过。

GPU 累计时间按每次 kernel 的 duration 相加，多 stream 并行时可重叠。CPU 算子、模块和 forward 区间采用 inclusive 时间，不能把父子区间直接相加。profile 请求受 shapes、stack 和标记采集影响，性能比较仍使用关闭 profiler 的独立 benchmark。

`unattributed_gpu_kernels` 指不能通过 launch correlation 归属到模型 forward 区间的 kernel，包括 forward 外的采样或调度辅助计算；这些事件仍完整保存在总时间线中。EXTEND / DECODE 的 GPU 小计因此不等于整个请求的全部 GPU kernel 总计。

当前 DTK 原始 PyTorch trace 在 Perfetto 中可能报告 `slice_spill_overlapping_complete_event`：同一轨道上的部分完整区间交叠，Perfetto 将其显示在 overflow 轨道，事件仍被保留。该现象在未合并的原始单卡文件中也存在；验证报告需要如实保留此项健康统计。

源码入口：

- `tools/profile_flash_next_serving.py`：启动、预热、完整请求采集、关闭服务、输出汇总。
- `sgl_amd_mono/common/profiling.py`：模型与融合范围、前后端墙钟范围。
- `sgl_amd_mono/common/profile_trace.py`：校时、合并、完整性检查及算子统计，可用于后续其他模型的同格式 trace。
- `tools/summarize_serving_profile.py`：使用原始文件离线重新生成合并 trace 和算子统计。
