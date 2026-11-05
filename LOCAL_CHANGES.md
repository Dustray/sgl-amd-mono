# 本地改动归属

基线：`upstream-base-20261105` / `e52803f9d2b7dabff4bfd2f53ec25c03d56af947`。

本清单描述实际源码，不把先前自行实现的 BF16 kernel 称作上游原版。每项均可通过 Git 历史和相对基线的 diff 查看。

| 文件或模块 | 做了什么 | 来源及状态 |
| --- | --- | --- |
| `sgl_amd_mono/` | 保留上游模型/spec/kernel 源码 | 上游原样，当前无本地计算修改 |
| 根 `README.md` | 加入本地开发说明链接 | 对上游文件的一处说明性修改 |
| `compat/.../bootstrap.py` | 接入配置、权重加载、ModelRunner 和状态查询四个 hook | 自写兼容层；不覆盖安装目录的 SGLang 文件 |
| `compat/.../contract.py`、`manifest.json` | 固定 SGLang 版本、21 个文件 hash、hook 签名和模型/设备条件 | 自写；旧上游引用和当前源码基线分别保存 |
| `compat/.../config.py`、`launcher.py`、`cli.py` | 模式选择、启动和预检 | 自写 |
| `compat/.../observer.py`、`report.py` | 观察权重并输出运行报告 | 自写，无权重副本 |
| `compat/.../bf16_ffn.py` | 两次发射的 BF16 FFN | 自写 FlyDSL，局部融合 |
| `compat/.../compile_ffn.py`、`hip_ffn.py` | 独立编译、HSACO 缓存和 HIP 加载 | 自写；为避开本机 FlyDSL/CuTe MLIR 冲突 |
| `compat/.../ffn_runtime.py` | 64 层真实权重验证和 decode FFN 路由 | 自写；生成存在差异，详见既有报告 |
| `compat/.../dense_monokernel.cpp` | GDN K1 与输出投影到 FFN 的 K2 | 自写 DTK HIP，298 行的实验 kernel，整模型尚未验收 |
| `compat/.../hip_monokernel.py`、`mono_runtime.py`、`mono_validation.py` | 协作启动、状态对接、逐层和影子验证 | 自写实验代码；不是上游 dense kernel 的直接移植 |
| `tools/verify_*.py`、`probe_cooperative.cpp` | 数值、状态、Graph、生成对照 | 自写验证，不代表性能测量 |
| `tools/source_status.py` | 显示 Git 基线、本地 diff、实际导入路径和 SGLang 契约 | 本次源码管理新增 |

表中的 `compat/.../` 均指 `compat/sglang0518/src/sgl_amd_mono_0518/`。

## 提交分层

1. `standalone-snapshot-20261105`：原样导入迁移前适配、实验代码、启动脚本和文档，方便追溯。
2. 源码管理提交：固定真实上游基线，补充本地说明、来源检查、忽略规则及源码安装路径。
3. 实验隔离提交：给未验收 dense 原型增加显式 opt-in，防止默认严格模式被误认为已正式支持。

后续升级直接在此 Git 仓库提交修改。计算 kernel、兼容 API 和验证变更应分别提交；提交说明需记录变更原因、验证命令及结果。运行日志和编译缓存留在忽略目录，不随补丁混入。
