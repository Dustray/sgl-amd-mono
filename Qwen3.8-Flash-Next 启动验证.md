# Qwen3.8-Flash-Next 启动验证

验证日期：2026-11-06。真实 BF16 checkpoint 已通过原生 SGLang TP8 文本推理验证，四组请求全部正常停止。随后按用户要求停止服务：原 PID `1600911` 已退出，`127.0.0.1:30160` 不再监听，GPU 0–7 每卡显存占用降至 2 MiB。本页保留此前的启动验证结果；Monokernel 的后续工作见 [接入评估](Qwen3.8-Flash-Next%20Monokernel%20接入评估.md)。

## 模型与环境

模型目录：`/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master`。

| 项目 | 本地检查结果 |
| --- | --- |
| 架构 | `Qwen4ExpForConditionalGeneration` / `qwen4_exp` |
| 权重 | BF16，无 `quantization_config`；131 个分片，1658 个 tensor，分片齐全 |
| 分片文件合计 | 335.276 GiB |
| 权重分组 | MoE 约 229.688 GiB；PLE 约 95.429 GiB；视觉约 0.836 GiB；其余约 9.323 GiB |
| 文本模型 | 48 层，hidden 2560；每组 3 层 GDN + 1 层 QSA；512 experts、top-10 |
| PLE | 约 2000 万 n-gram 词表、2560 维，checkpoint 拆为 128 份 |
| GPU | 8 × gfx936，单卡约 64 GiB、80 CU |
| SGLang | `0.5.18+dtk2604.torch2110.2609201917.g2b8d2b` |
| Torch | `2.11.0+dtk2604.2609201441.gf74cd7` |
| Transformers | `5.12.1` |
| FlyDSL | `0.2.0+dtk2604.torch2110.2610101749.gad9161` |

权重分组依据 safetensors header 中 tensor 的字节范围统计，不包含逐张量数值校验；文件总大小包含 header。实际启动使用 `--language-model-only`，跳过视觉编码器，并跳过未启用的 MTP 权重。

## 当前启动配置

使用 [launch_qwen_flash_next.sh](launch_qwen_flash_next.sh) 启动原生 SGLang：

```bash
bash launch_qwen_flash_next.sh
```

该脚本使用 GPU 0–7、TP8、BF16 权重及 BF16 KV、page size 64、Mamba extra-buffer、4096 上下文、并发 1、256-token chunked prefill，PLE 保留在显存。首次验证关闭 decode/prefill Graph、overlap 和 radix cache。默认只监听 `127.0.0.1:30160`。

使用 `QWEN_FLASH_NEXT_MODEL_PATH` 可以覆盖模型路径；`HIP_VISIBLE_DEVICES` 可以调整 GPU 顺序，但仍需 8 张空闲卡。脚本末尾可追加服务参数。

初次四卡方案尝试 PLE CPU offload：分布式初始化后每卡可用约 59.9 GiB，而 MoE 权重平均已需约 57.4 GiB/卡，另有复制的 HC 权重、注意力及 embedding 等，占用余量很小，因此主动终止，切到随后释放的八卡。该次未完成权重加载，不作为四卡成功或 OOM 的实测结论。

八卡 mmap 加载前几片速度慢，采样时主线程位于 `copy_ple_rows_to_tp_embedding`。重启采用安装版已有的 `--weight-loader-disable-mmap` 与 `SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1`；首次完整加载使用 2 个 loader 线程及有界缓冲，约 578 秒完成，模型内存增量约 42.36–42.63 GiB/卡。最终启动脚本提高为 8 个 loader 线程，仍使用安装版的有界缓冲；没有修改安装目录的模型计算实现。

首次完整加载后遇到 QSA KV pool 参数错误：默认 `page_size=1`，而此模型压缩比为 4，要求 page size 大于 1 且是 4 的倍数。脚本已显式指定 `--page-size 64 --mamba-radix-cache-strategy extra_buffer`。正式重载前使用相同模型形状的 dummy 权重检查初始化与前向；dummy 结果单独记录，不作为真实 checkpoint 的正确性证据。

该 dummy 检查已通过：512-token 输入拆分为多个 prefill chunk，随后生成 8 token，8 个输出 logprob 均为有限值，结果见 `dummy-shape-check.json`。首轮编译出现 TileLang 对 QSA indexer `Logits(bx, position)` 写入的 data-race 警告，未导致编译/前向终止；本次未关闭该检查，也未完成该内核的独立数值审计。此外，安装版缺少 `E=512,N=80,device_name=BW` 的 MoE 调优配置，使用默认配置；因此本次启动参数和计时均不代表优化后的性能。

## Monokernel 适用范围

仓库已有的 [Flash-Next 实现](sgl_amd_mono/qwen38_flash_next/README.md) 与这份 checkpoint 的模型结构一致，但其现成执行条件不同：

| 条件 | 仓库原有 Flash-Next monokernel | 本次启动 |
| --- | --- | --- |
| GPU | gfx950 / MI355X | gfx936 |
| 权重 | Quark：MXFP4 experts + FP8 attention/GDN + FP8 PLE | 全 BF16 |
| 并行 | TP1，权重保留在 GPU | TP8 |
| 路由 | NEXTN target verify 与 MTP draft 的指定形状 | 常规 prefill / decode，无 speculative |
| 接口 | 新版 SGLang monokernel API | 本地 0.5.18 兼容桥仅为 Qwen 27B 接入 |

原有 `spec.py` 的设备表只列 `gfx950`，量化检查要求 `quark`，并行检查要求 TP/PP/CP/EP 均为 1。不能通过启动 BF16 原生服务就宣称 Flash-Next monokernel 已经接入。本次也没有将 27B 的 BF16 dense kernel 用于这个 MoE 模型。

## 服务验证

真实权重最终加载耗时为 187.18–187.54 秒/TP rank。服务预热、健康检查和以下四组 OpenAI 流式请求均已完成，`validation.json` 的八项检查全部通过。服务自报 `load_format=auto`、`quantization=null`、TP8、BF16，与真实实例的进程命令一致；dummy 实例已停止。

| 用例 | 输入 token | 输出 token | 首个可见文本 / 秒 | 端到端 / 秒 | 输出检查 |
| --- | ---: | ---: | ---: | ---: | --- |
| 算术 | 23 | 3 | 18.381 | 18.886 | 精确返回 `56`，正常停止 |
| 中文解释张量并行 | 21 | 102 | 13.515 | 37.944 | 返回三句话，正常停止 |
| 英文解释天空颜色 | 23 | 52 | 0.308 | 12.735 | 返回两句话，正常停止 |
| 长输入找回开头的验证码 | 871 | 5 | 4.603 | 5.562 | 精确返回 `FLASH936`，正常停止 |

token 数来自服务返回的 usage，包含模板和终止 token 等影响。该表是按顺序各执行一次的功能验证，前两项包含新形状的首次编译开销；首个可见文本时间也包含推理结果解析与网络传输，不等同于纯 prefill 时间。它不能用于正式性能对比。最后一项输入跨越多个 256-token prefill chunk，验证码放在背景资料之前，问题位于输入结尾。

验证后 GPU 显存占用约 46.92–47.02 GiB/卡。当前配置仅验证并发 1、4096 上下文内的文本请求，没有做图像请求、长上下文极限、精度基准或 Monokernel 性能测试。

服务目前已停止。重新执行启动脚本并等待健康检查通过后，可使用以下请求：

```bash
curl -s http://127.0.0.1:30160/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"计算7×8，只输出结果。"}],"temperature":0,"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

复现文本冒烟测试：

```bash
python3 tools/smoke_qwen38_flash_next.py \
  --output monokernel_reports/qwen38_flash_next/20261106/smoke.json
```

测试覆盖算术、中英文回答和超过单个 prefill chunk 的输入，关闭 thinking、temperature=0，每次最多 128 个输出 token。算术和验证码检查精确答案，中英文检查非空且正常停止；这些检查不等同于模型精度评测。脚本记录内容到达时间与 HTTP 端到端时间，包含首次编译/缓存影响，不作为正式性能对比。

预检、启动记录、各次日志、服务身份、GPU 状态及完整输出保存在 `monokernel_reports/qwen38_flash_next/20261106/`，重点文件为 `native-tp8.log`、`smoke.json`、`server-info-final.json`、`validation.json` 和 `SHA256SUMS`。该目录按仓库现有规则忽略；可版本管理的改动是启动脚本、冒烟脚本、本说明以及 README / 改动清单的入口。SGLang 的 31 项安装源码 hash 校验通过；没有执行 Git 提交。
