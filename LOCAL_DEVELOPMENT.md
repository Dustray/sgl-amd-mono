# 本地源码、改动与升级

当前目录是从 `sgl-project/sgl-amd-mono` 完整克隆并保留上游历史的 Git 工作区。上游 remote 为 `upstream`，本地开发分支为 `local/sglang-0.5.18-gfx936`。2026-11-05 拉取时，上游 main 为 `e52803f9d2b7dabff4bfd2f53ec25c03d56af947`，已固定为 `upstream-base-20261105`，详情见 `UPSTREAM.lock.json`。

当前适配仍通过 SGLang 0.5.18 的 HookRegistry 工作。把代码纳入上游仓库的开发分支，解决了来源、diff 和升级管理问题，**并不代表上游 gfx950 kernel 已在 gfx936 上运行**。

## 哪些是上游，哪些是本地

| 路径 | 来源与用途 |
| --- | --- |
| `sgl_amd_mono/`、`test/` 等原有目录 | 真实上游源码及其历史；当前没有修改上游 kernel 的计算实现 |
| `compat/sglang0518/src/sgl_amd_mono_0518/` | 之前独立编写的兼容层、FFN 和 dense HIP 实验代码，现已进入 Git |
| `compat/sglang0518/pyproject.toml` | 本机兼容项目的独立安装入口；当前 SGLang 缺少上游新版 Monokernel API |
| `tools/verify_*.py`、`tools/source_status.py` | 本地验证和来源检查工具 |
| `launch_qwen*.sh` | 本机模型启动配置；融合脚本使用本地兼容入口 |
| `LOCAL_CHANGES.md`、本地中文文档 | 改动归属、实际验收状态及升级说明 |
| `monokernel_reports/`、`benchmark_results/`、日志及 PID | 运行产物，已在 Git 中忽略，不混入源码提交 |

`README.md` 主体和根目录 `pyproject.toml` 来自上游。上游 README 仅新增本地说明链接。原来的本地 README 和 pyproject 已移至 `compat/sglang0518/`。

## 现有实现的实际状态

- `audit`：走原生 SGLang，记录兼容契约与权重观察。
- `fused_ffn`：此前验证过的 FFN 局部融合，仍不是完整 Monokernel。旧生成对照为 9/12 逐 token 一致，不能写成完整模型精度已通过。
- `monokernel` / `required`：包含自行编写的 K1/K2 原型，**尚未完成整模型验收，默认拒绝启动**。仅设置 `SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE=1` 才允许继续实验，启动后仍执行真实权重检查。

最新一次 dense 原型启动在 K2 检查中失败，测得 NRMSE `0.0038988393`，高于当时设置的 `0.003`。此前的整模型试验也曾超过 2% logits NRMSE。代码中尝试过基于 KL、cosine 和逐段误差的验收方式，但尚未据此完成通过的整模型验收；这些改动也保存在迁移快照中，不作为性能测试前提已满足的证明。

现有 dense 原型的计算并非从上游 kernel 小修得来，见 `LOCAL_CHANGES.md`。Qwen 27B dense 参考代码属于 SGLang PR #43558，基线为 `b25e92f...`，不在此次 Monokernel main 的模型目录中；它的引用单独记录在锁文件中。

## 安装与检查

在仓库根目录执行：

```bash
python -m pip install --no-deps --no-build-isolation -e compat/sglang0518
python tools/source_status.py --verify
python -m sgl_amd_mono_0518.cli contract
git status --short
```

上述安装使用当前分支里的兼容项目，不替换 Torch 或 SGLang。根目录上游包的安装要求见上游 README，不能把两种安装入口混用。

```bash
bash launch_qwen.sh
bash launch_qwen_fused_ffn.sh
```

实验性 dense 路径必须显式选择：

```bash
SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE=1 bash launch_qwen_monokernel.sh
```

该命令可能因数值检查失败而退出。它不代表已通过正式验收。

## 查看和导出改动

```bash
git log --oneline upstream-base-20261105..HEAD
git diff --stat upstream-base-20261105..HEAD
git diff upstream-base-20261105..HEAD -- sgl_amd_mono compat tools
git show standalone-snapshot-20261105:compat/sglang0518/src/sgl_amd_mono_0518/dense_monokernel.cpp
git format-patch upstream-base-20261105..HEAD -o monokernel_reports/migration/patches
```

`standalone-snapshot-20261105` 保存迁移时的原始本地代码，未把旧实验改写成已验收结果。迁移前整个工作区另有 tar 备份，路径和 SHA256 见锁文件。

## 升级上游

先保证当前工作已提交，再在新分支上升级：

```bash
git status --short
git fetch upstream
git switch -c upgrade/monokernel-YYYYMMDD local/sglang-0.5.18-gfx936
git rebase upstream/main
```

解决冲突后，将 `UPSTREAM.lock.json` 的上游基线更新为此次实际使用的完整 SHA、日期和新的基线 tag；保留旧 tag。先审阅 `git diff <旧上游基线> <新上游基线> -- sgl_amd_mono`，核对 API、模型及 kernel 的变化，再运行：

```bash
python -m pip install --no-deps --no-build-isolation -e compat/sglang0518
python tools/source_status.py --verify
python -m pytest -q test/unit/test_sglang0518_compat.py
python -m sgl_amd_mono_0518.cli contract
```

升级源码不自动解除 gfx936、模型格式或 SGLang 接口限制。若更新 SGLang，须重新审查 `manifest.json` 固定的签名与文件 hash，并重新执行真实权重、状态和模型验证，不能只更新 hash。

升级通过后再提交锁文件和适配改动，并用 `git range-diff <旧基线>..<旧本地提交> <新基线>..HEAD` 审阅本地补丁是否被上游变化改变。无需升级时可切回原开发分支；中途冲突可用 `git rebase --abort` 返回升级前状态。
