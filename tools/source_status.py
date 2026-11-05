import argparse
from importlib import metadata, util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def git(*arguments, check=True):
    result = subprocess.run(["git", "-C", str(ROOT), *arguments], text=True, capture_output=True, check=check)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Check upstream Git provenance and the installed local compatibility source")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    arguments = parser.parse_args()
    lock = json.loads((ROOT / "UPSTREAM.lock.json").read_text())
    expected_package = ROOT / lock["compatibility_project"] / "src" / lock["compatibility_module"] / "__init__.py"
    spec = util.find_spec(lock["compatibility_module"])
    imported_package = Path(spec.origin).resolve() if spec is not None and spec.origin else None
    ancestor = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", lock["base_commit"], "HEAD"], capture_output=True).returncode == 0
    changed_paths = git("diff", "--name-only", lock["base_commit"], "--").splitlines()
    status = git("status", "--short")
    report = {
        "root": str(ROOT),
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "upstream_remote": git("remote", "get-url", lock["remote"]),
        "upstream_base": lock["base_commit"],
        "upstream_base_is_ancestor": ancestor,
        "upstream_kernel_source_changes": [path for path in changed_paths if path.startswith("sgl_amd_mono/")],
        "compatibility_import": str(imported_package) if imported_package else None,
        "expected_compatibility_import": str(expected_package),
        "working_tree": status,
        "diff_stat": git("diff", "--stat", lock["base_commit"], "--"),
        "errors": [],
    }
    if not ancestor:
        report["errors"].append("Locked upstream base is not an ancestor of HEAD")
    if report["upstream_remote"] != lock["repository"]:
        report["errors"].append("Upstream remote does not match the locked repository")
    if git("rev-parse", lock["baseline_tag"]) != lock["base_commit"]:
        report["errors"].append("Baseline tag does not match the locked commit")
    if imported_package != expected_package:
        report["errors"].append("Install the compatibility project from this checkout: pip install --no-deps --no-build-isolation -e compat/sglang0518")
    else:
        from sgl_amd_mono_0518.contract import environment_contract

        contract = environment_contract()
        report["sglang_version"] = contract["sglang_version"]
        report["sglang_file_hashes_checked"] = len(contract["file_hashes"])
        report["errors"].extend(contract["errors"])
        report["compatibility_distribution_version"] = metadata.version("sgl-amd-mono-sglang0518")
    if arguments.require_clean and status:
        report["errors"].append("Working tree has uncommitted or untracked changes")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(arguments.verify and bool(report["errors"]))


if __name__ == "__main__":
    sys.exit(main())
