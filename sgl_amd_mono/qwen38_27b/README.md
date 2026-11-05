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
| `gfx936/kernel.cpp`、`gfx936/program.py` | 原本地 DTK HIP K1/K2 协作 kernel 及启动器，实验实现 |
| `gfx936/runner.py`、`gfx936/check.py` | 旧版 SGLang 的层路由、状态对接及数值对照 |
| `check.py` | 可独立运行的 FFN 数值、执行计数和 Graph 重放验证 |

计算代码由原 `compat/sglang0518/` 迁入，没有因移动目录改变计算公式，也没有把它标成上游原版。原路径保留导入转发，便于旧工具继续工作；新的实现不得反向导入兼容包。迁移前源码保存在 `standalone-snapshot-20261105`，本次整理前版本为 `source-managed-20261105`。

上游已有可供后续评估的 BF16 实现：`kimi_k3/gfx950/stages/gemv.py`、`kimi_k3/gfx950/stages/moe.py` 和 `kimi_k31/gfx950/stages/tile_mma.py`。这些实现使用 gfx950 的 MFMA、LDS 和线程布局。本机此前验证过的是 gfx936 的 `llvm.hcu.mmac.f32.16x16x16.bf16`；将上游 gfx950 stage 移植到这里需要重新验证指令 lowering、真实尺寸、舍入语义及 Graph。

## 接口与当前状态

当前安装的 SGLang 0.5.18 DTK 没有上游新版 `MonoSpec` API，因此 `spec.py` 提供本机 HookRegistry 接入所需的准入和构建函数，**不导出尚未验证的原生 `SPEC`**，也不将该模型加入新版 `SPECS` 注册表。根包的插件入口会在旧版环境下接入已安装的兼容桥，在有原生接口的环境继续使用原有注册流程。

`compat/sglang0518/` 负责旧版 hook、启动参数、SGLang 文件 hash 和进程报告。模型计算及其验证由本目录维护。

- `fused_ffn`：仅融合 FFN；prefill、attention、norm 等仍走 SGLang。
- `monokernel`：K1/K2 原型尚未通过整模型验收，默认关闭，实验 opt-in 后仍运行真实权重检查。
- `required`：拒绝未验收原型，不受实验 opt-in 放行。

此前 dense 启动曾在 K2 NRMSE `0.0038988393 > 0.003` 时失败。目录整理和 FFN 验证不会改变这一整模型验收状态。

## 安装与验证

从仓库根目录安装两个项目，不替换 Torch、FlyDSL 或 SGLang：

```bash
python -m pip install --no-deps --no-build-isolation -e . -e compat/sglang0518
python tools/source_status.py --verify
python -m pytest -q test/unit/test_qwen38_27b.py test/unit/test_sglang0518_compat.py
CUDA_VISIBLE_DEVICES=0 python -m sgl_amd_mono.qwen38_27b.check --full-shape --output monokernel_reports/qwen38_27b/ffn-check.json
```

GPU 命令应选择空闲设备。检查包含 256/512 和 5120/17408 两组尺寸、batch 1/2、eager 与改变输入后的三次 Graph 重放，阈值维持 NRMSE `0.002`。它验证 FFN，不代表完整模型或性能测试通过。

服务仍通过原有脚本选择 `audit`、`fused_ffn` 或显式实验模式；其 hook 会调用本目录的 `spec.build()`。后续对 kernel、权重布局和模型路径的修改直接提交本目录，升级上游时按 `LOCAL_DEVELOPMENT.md` 审阅补丁。
