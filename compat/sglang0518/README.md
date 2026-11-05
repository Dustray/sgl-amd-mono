# SGLang 0.5.18 兼容审计与 gfx936 BF16 FFN 融合插件

当前包包含兼容审计模式和实验性 `fused_ffn` 模式。后者通过真实 FlyDSL kernel 融合 Qwen3.8-27B 的 BF16 dense FFN，每层使用两次 GPU launch，覆盖 decode batch 1/2。**GDN、全注意力、residual 和 norm 继续使用 SGLang；完整模型 Monokernel 尚未完成，尚无加速结论。**

当前硬件是海光 `gfx936`，模型是 `Qwen3_5ForConditionalGeneration` / `qwen3_5`、BF16、64 层、TP1。文档固定的上游版本只覆盖 gfx950 / DeepSeek-V4.1-Flash。核对的上游主分支虽有 Qwen kernel，但对应 `Qwen4ExpForConditionalGeneration`、gfx950、Quark MXFP4/FP8、48 层及 NEXTN verify，不覆盖这里的 Qwen3.8-27B。

已另行核对上游 SGLang PR #43558 的 27B dense kernel：模型尺寸吻合，但仅接收 Quark MXFP4、要求 gfx950 和至少 256 CU；当前 BF16 / gfx936 / 80 CU 不能直接运行。DTK 版 FlyDSL 已安装，BF16 MMAC、SiLU-mul 和 HIP Graph 基础验证通过。详细记录见 `Qwen3.8-27B 正式融合接入进度.md`。按用户要求，真实性能对比等待正式融合及模型数值验证完成后再进行。

## 安装

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -m pip wheel --no-deps --no-build-isolation -w dist .
```

插件以 `sglang.srt.plugins` entry point 自动发现。未设置 `SGLANG_AMD_MONO_0518=1` 时不注册 hook。安装和运行均不修改 `site-packages/sglang` 文件；manifest 固定当前 DTK 下游 wheel 的版本及十一个关键源码文件 SHA256，并核对四个 hook 目标的签名。

### 当前 Python 3.10 环境的 FlyDSL

```bash
python3 -m pip install --no-deps \
  'flydsl==0.2.0+dtk2604.torch2110.2610101749.gad9161' \
  -i https://pypi.sourcefind.cn/nightly/dtk/
HIP_VISIBLE_DEVICES=1 python3 tools/verify_flydsl_gfx936.py \
  --output monokernel_reports/flydsl_gfx936_verification.json
```

用户提供的 `0.2.0+dtk2610.torch2110.2610101746.gad9161` 在该索引只有 cp312 wheel；上面的 cp310 wheel 对应相同源码提交 `ad9161`，匹配当前 DTK 2604 / Torch 2.11。安装未替换 Torch、SGLang 或其依赖。验证工具仅证明 FlyDSL 编译、真实 GPU 运算和 graph 重放可用，不代表模型 Monokernel 已实现。

## 实验性 BF16 FFN 融合

```bash
sgl-amd-mono-0518 preflight --mode fused_ffn \
  --model-path /public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master
bash launch_qwen_fused_ffn.sh
```

后台启动时记录专用 PID：

```bash
nohup bash launch_qwen_fused_ffn.sh > qwen3.8-27b-fused-ffn.log 2>&1 < /dev/null &
echo $! > qwen3.8-27b-fused-ffn.pid
```

融合仅接受已验证的 gfx936、TP1、BF16 dense 权重及模型尺寸。启动时对全部 64 层的真实权重分别验证 batch 1/2，归一化 RMS 误差超过 `0.002` 时拒绝接入。decode batch 1/2 使用融合 FFN，prefill 和其他 batch 使用原始 FFN；每次模型调用结束都会恢复路由开关。

第一次 launch 在 68 个独立工作组内融合 gate/up GEMV、SiLU-mul 和 down 的分块结果，第二次 launch 归并 FP32 部分和并写出 BF16。SiLU 后先舍入为 BF16，再做乘法，与本机 LightOp 一致。权重直接引用原始参数，无额外权重副本；64 层共使用 170 MiB FP32 工作空间。当前 GEMV 使用 BF16 读取和 FP32 向量累加，没有使用上游 FP4 scaled MFMA，也没有引入跨工作组自旋同步。

本机 FlyDSL 和 SGLang 引入的 CuTe 存在 MLIR/nanobind 重复注册冲突，因此编译放在独立子进程中。推理进程只通过 HIP 加载 HSACO，在启动时验证 SHA256 和 kernel 参数 ABI，在当前 stream 上发射 kernel，支持 HIP Graph 捕获和重放。

```bash
HIP_VISIBLE_DEVICES=4 python3 tools/verify_bf16_ffn.py --full-shape \
  --output monokernel_reports/bf16_ffn_hip_verification.json
python3 tools/verify_fused_ffn_serving.py \
  --reference monokernel_reports/ffn_native_reference_matrix.json
curl --fail http://127.0.0.1:30000/server_info
```

`/server_info` 的 `internal_states[0].amd_fused_ffn` 返回设备执行计数、各层计数、工作空间和二进制指纹。计数由 GPU kernel 更新，包含启动预热和 graph 重放；请求验证比较前后快照。读取状态会同步 GPU，正常 decode 路径不做该同步。`monokernel_executions` 仍为 0，FFN 的执行单独记录在 `fused_ffn_device_executions`。

验证工具检查数值、图重放、prefill 回退、并发生成及原生输出对照，不属于性能测试。与原生输出有差异时保留完整 token、共同前缀和该前缀的 logprob 差值，不把有限生成样例视为完整精度验收。

2026-11-05：64 层真实权重的 128 组 FFN 数值检查通过，最大 NRMSE 为 `0.0013608205`。相同并发和清缓存条件下，12 项原生/融合生成对照有 9 项逐 token 一致；其余 3 项串行生成存在差异，完整模型一致性仍待验证。FP64 点积抽查和完整结果见 `Qwen3.8-27B 正式融合接入进度.md`。性能对比继续等待。

## 严格检查

```bash
sgl-amd-mono-0518 contract
sgl-amd-mono-0518 preflight \
  --model-path /public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master
bash launch_qwen_monokernel.sh
```

后两个命令在当前组合上必须失败。拒绝发生在权重加载前；报告列出硬件、模型、TP、框架 API 和缺少可执行 spec 等原因，并记录当前 FlyDSL 版本。通用插件加载器会捕获注册错误，因此专用 launcher 还检查 hook 是否真正安装。

## 审计模式运行 Qwen

```bash
SGLANG_AMD_MONO_0518_MODE=audit bash launch_qwen_monokernel.sh
```

审计模式明确保留 SGLang 原始计算路径，记录 `execution_path=sglang_original`、`monokernel_eligible=false` 和 `monokernel_executions=0`。它用于证明插件注册、权重观察、ModelRunner 接入以及 prefill/decode 路由可用，不用于验证 Monokernel 性能。

```bash
sgl-amd-mono-0518 status
curl --fail http://127.0.0.1:30000/health
```

后台启动审计服务：

```bash
SGLANG_AMD_MONO_0518_MODE=audit nohup bash launch_qwen_monokernel.sh \
  > qwen3.8-27b-monokernel-audit.log 2>&1 < /dev/null &
echo $! > qwen3.8-27b-monokernel.pid
```

停止审计服务：

```bash
kill "$(cat qwen3.8-27b-monokernel.pid)"
```

`monokernel_reports/process-*.json` 记录进程 PID、源码哈希、模型配置哈希、观察的 tensor 数量/字节数、权重观察是否完成以及各 forward key。观察器不保留 tensor 副本、不复制权重到 CPU，也不读取 tensor 的数值。模型 forward 的签名通过 `functools.wraps` 保留；CUDA graph replay 的次数不冒充 Python forward 次数或 Monokernel 执行次数。

普通启动及一键关闭插件：

```bash
bash launch_qwen.sh
SGLANG_AMD_MONO_0518=0 bash launch_qwen_monokernel.sh
```

## 尚缺的工作

要完成上游两段式 GDN/FFN 覆盖，还需要融合 GDN 输入投影、卷积与 SSM 状态、gated norm，以及 FFN 前后的输出投影和 residual/norm，并完成状态及 logits 验证。当前 FFN 融合已覆盖 dense MLP；全注意力 KV 布局继续由 SGLang 管理。当前包未回移上游 runtime、peer memory、mailbox 或 fail-stop kernel；上游 gfx950 kernel 也未被安装或改名注册到当前模型。

上游固定版本及核对的主分支 revision 保存在 `src/sgl_amd_mono_0518/manifest.json`。回移设计文档与当前落地状态应分开评估。

## 审计模式验证记录

- wheel 构建和 entry point 发现成功；插件关闭时不注册 hook。
- 初始六个关键源码文件哈希、三个目标方法的签名及 staticmethod 描述符检查通过；FFN 接入后扩展为十一个文件、四个目标。
- 观察器惰性消费、完成标志、无 tensor 副本检查通过；注入源码契约错误后审计模式也拒绝启动。
- 严格模式在权重加载前退出，拒绝日志为 `qwen3.8-27b-monokernel-required.log`。
- 审计模式成功观察 1199 个 checkpoint tensor，共 55,562,855,904 字节；完整模型 forward 记录到 `DECODE` batch 1/2 与 `EXTEND` batch 1，均为 `sglang_original`。
- decode CUDA graph 捕获成功；健康检查、模型列表、两条并发文本对话均通过，`17 × 19` 返回 `323`。
- 该阶段实际 Monokernel 执行次数为 **0**。后续 FFN 数值与服务验证见 `Qwen3.8-27B 正式融合接入进度.md`；完整 Monokernel、fail-stop、跨卡和性能验收尚未完成。

源码契约报告为 `monokernel_contract.json`，运行报告位于 `monokernel_reports/`，对话结果为 `monokernel_serving_verification.json`。wheel 位于 `dist/`，其 SHA256 保存在 `dist/SHA256SUMS`。
