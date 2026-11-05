## 当前环境

- 海光 DCU（`gfx936`），单卡约 64 GiB 显存；默认使用 0 号卡。
- SGLang：`0.5.18+dtk2604.torch2110.2609201917.g2b8d2b`。
- 本地模型：`/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master`。
- 启动参数保存在同目录的 `launch_qwen.sh`，上下文长度 33792，并发上限 2。

当前另有 Monokernel 审计插件接入，启动入口为 `launch_qwen_monokernel.sh`，详细说明见 `README.md`。审计服务使用原始 SGLang 计算路径，实际 Monokernel 执行次数为 0；其 PID 与日志分别保存在 `qwen3.8-27b-monokernel.pid` 和 `qwen3.8-27b-monokernel-audit.log`。停止该实例应使用 `kill "$(cat qwen3.8-27b-monokernel.pid)"`。

## 启动服务

在当前目录前台启动：

```bash
bash launch_qwen.sh
```

后台启动并保存日志和 PID（同一端口只启动一个实例）：

```bash
nohup bash launch_qwen.sh > qwen3.8-27b.log 2>&1 < /dev/null &
echo $! > qwen3.8-27b.pid
tail -f qwen3.8-27b.log
```

监听地址为 `0.0.0.0:30000`。默认 `HIP_VISIBLE_DEVICES=0`，可在启动前设置该变量选择其他空闲卡；如果更换卡，需相应调整 `--numa-node`。

## 检查和调用

```bash
curl --fail --max-time 10 http://127.0.0.1:30000/health
curl --fail --max-time 10 http://127.0.0.1:30000/v1/models
```

```bash
curl --fail-with-body --max-time 300 \
  http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B",
    "messages": [{"role": "user", "content": "请用三句话介绍SGLang是什么。"}],
    "max_tokens": 256,
    "temperature": 0.2,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

停止后台服务：

```bash
kill "$(cat qwen3.8-27b.pid)"
```

## 与原命令的差异

- 使用本地模型目录，避免下载模型。
- 原来的 `--enable-piecewise-cuda-graph` 已不再支持；替代参数 `--cuda-graph-backend-prefill tc_piecewise` 在此环境又因 LightOp 的 `rms_norm_opt` 无法被 TorchDynamo 追踪而失败。因此使用 `--cuda-graph-backend-prefill disabled`，保留 decode graph；同时更新 Mamba 和 decode graph 参数名称。
- 使用 `--weight-loader-disable-mmap`，先将分片读入主存，再复制到 DCU。
- 当前版本的 `--keep-mm-feature-on-device` 会启用 NVIDIA 专用 CUDA IPC，在此环境报错。改用 `--mm-feature-transport cpu`，并移除 `SGLANG_USE_CUDA_IPC_TRANSPORT=1`。
- 33792 小于模型原生上下文长度 262144，因此保留模型自带 RoPE 配置。

## 验证结果

- 已后台启动，PID 记录在 `qwen3.8-27b.pid`，完整启动日志在 `qwen3.8-27b.log`。
- `/health` 和 `/v1/models` 均返回 HTTP 200，模型列表显示 `Qwen3.8-27B`，最大上下文长度 33792。
- 同时发送两条关闭思考的对话请求，均返回 HTTP 200 且正常结束：三句话介绍 SGLang，以及 `17 × 19`（正确返回 `323`）。这里只验证了文本对话。
- 本次实测：默认 mmap 方式加载权重约 886.64 秒；增加 `--weight-loader-disable-mmap` 后约 20.86 秒。decode graph 捕获成功，prefill graph 关闭。
