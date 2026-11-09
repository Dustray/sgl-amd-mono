# 本地源码、改动与升级

当前目录是从 `sgl-project/sgl-amd-mono` 完整克隆并保留上游历史的 Git 工作区。上游 remote 为 `upstream`，本地开发分支为 `local/sglang-0.5.18-gfx936`。2026-11-05 拉取时，上游 main 为 `e52803f9d2b7dabff4bfd2f53ec25c03d56af947`，已固定为 `upstream-base-20261105`，详情见 `UPSTREAM.lock.json`。

当前适配仍通过 SGLang 0.5.18 的 HookRegistry 工作。把代码纳入上游仓库的开发分支，解决了来源、diff 和升级管理问题，**并不代表上游 gfx950 kernel 已在 gfx936 上运行**。

## 哪些是上游，哪些是本地

| 路径 | 来源与用途 |
| --- | --- |
| `sgl_amd_mono/` 的原有模型与 common | 真实上游源码及其历史；原有计算实现保持原样 |
| `sgl_amd_mono/qwen38_27b/` | 本地新增模型目录，包含 spec、权重检查、gfx936 计算、runner 和 check；迁入的计算代码仍是原本地实现 |
| `sgl_amd_mono/__init__.py` | 插件入口增加旧版接口桥接；新版 spec 注册列表不变 |
| `compat/sglang0518/src/sgl_amd_mono_0518/` | 旧版 hook、契约检查和启动入口；原计算模块现在仅转发到模型目录 |
| `compat/sglang0518/pyproject.toml` | 兼容项目安装入口，依赖同一工作区的根模型包 |
| `tools/verify_*.py`、`tools/source_status.py` | 本地验证和来源检查工具 |
| `launch_qwen*.sh` | 本机模型启动配置；融合脚本使用本地兼容入口 |
| `LOCAL_CHANGES.md`、本地中文文档 | 改动归属、实际验收状态及升级说明 |
| `monokernel_reports/`、`benchmark_results/`、日志及 PID | 运行产物，已在 Git 中忽略，不混入源码提交 |

`README.md` 主体和根目录 `pyproject.toml` 来自上游；本地增加了说明链接和 C++ kernel 打包规则。原来的本地 README 和 pyproject 已移至 `compat/sglang0518/`。模型目录的接口差异、来源及验证入口见 `sgl_amd_mono/qwen38_27b/README.md`。

## 现有实现的实际状态

- `audit`：走原生 SGLang，记录兼容契约与权重观察。
- `fused_ffn`：此前验证过的 FFN 局部融合，仍不是完整 Monokernel。旧生成对照为 9/12 逐 token 一致，不能写成完整模型精度已通过。
- `monokernel`：运行本地 DTK HIP decode 实现；接受源码及环境匹配的正式记录，未匹配时须显式实验 opt-in。启动后仍执行全部真实权重检查。
- `required`：启动脚本的默认模式；只接受匹配的 `sgl_amd_mono/qwen38_27b/acceptance.json`。完整 decode 由六项 eager/Graph 能力自动启用，包括 embedding、48 层 GDN、16 层全注意力、64 层 K2、最终 norm 和 LM head，每步 129 次发射。实验开关不能绕过源码和环境准入。

迁移时的 dense 原型曾在 K2 检查中测得 NRMSE `0.0038988393`，整模型试验也曾超过 2% logits NRMSE。这些是保留在迁移快照中的历史失败，不代表当前实现。后续真实权重、同状态影子对照、Graph 和服务的最新结果统一记录在 `Qwen3.8-27B 正式融合接入进度.md`；验收记录绑定实际源码及环境，不依赖文档中的声明。

现有 dense 原型的计算并非从上游 kernel 小修得来，见 `LOCAL_CHANGES.md`。Qwen 27B dense 参考代码属于 SGLang PR #43558，基线为 `b25e92f...`，不在此次 Monokernel main 的模型目录中；它的引用单独记录在锁文件中。

## 安装与检查

在仓库根目录执行：

```bash
python -m pip install --no-deps --no-build-isolation -e . -e compat/sglang0518
python tools/source_status.py --verify
python -m sgl_amd_mono_0518.cli contract
git status --short
```

上述安装使用当前分支里的模型包和旧版兼容项目，不替换 Torch 或 SGLang。只安装根项目仍不能在旧版 SGLang 中提供 hook 桥接；只安装兼容项目也不再包含计算实现。

```bash
bash launch_qwen.sh
bash launch_qwen_fused_ffn.sh
```

修改源码后，尚未重新验收的开发路径须显式选择；测试完整 decode 时另设 `SGLANG_AMD_MONO_0518_DECODE_MODEL=1`：

```bash
SGLANG_AMD_MONO_0518_MODE=monokernel SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE=1 bash launch_qwen_monokernel.sh
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
python -m pip install --no-deps --no-build-isolation -e . -e compat/sglang0518
python tools/source_status.py --verify
python -m pytest -q test/unit/test_qwen38_27b.py test/unit/test_sglang0518_compat.py
python -m sgl_amd_mono_0518.cli contract
```

升级源码不自动解除 gfx936、模型格式或 SGLang 接口限制。若更新 SGLang，须重新审查 `manifest.json` 固定的签名与文件 hash，并重新执行真实权重、状态和模型验证，不能只更新 hash。模型源码、兼容源码、Torch/SGLang 版本或 HIP 编译器变化会使正式记录失配；必须重新完成验收后生成新记录，不能仅重算 `acceptance.json` 的指纹。

升级通过后再提交锁文件和适配改动，并用 `git range-diff <旧基线>..<旧本地提交> <新基线>..HEAD` 审阅本地补丁是否被上游变化改变。无需升级时可切回原开发分支；中途冲突可用 `git rebase --abort` 返回升级前状态。
