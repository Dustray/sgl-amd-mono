import argparse
import json
from pathlib import Path


def summarize(directory):
    reports = {}
    for mode in ("native", "monokernel"):
        for batch in (1, 2):
            reports[mode, batch] = json.loads((directory / f"{mode}-batch{batch}.json").read_text())
    comparisons = {batch: json.loads((directory / f"comparison-batch{batch}.json").read_text()) for batch in (1, 2)}
    default_comparison = json.loads((directory / "comparison-default.json").read_text())
    default_rows = {(row["batch"], row["label"]): row for row in default_comparison["rows"]}
    rows = []
    for batch in (1, 2):
        for case, independent in zip(reports["monokernel", batch]["cases"], comparisons[batch]["rows"], strict=True):
            audits = case["rank_audit"]
            assert len(audits) == 8
            checks = audits[0]["checks"]
            failed_steps = {entry["step"] for rank in audits for entry in rank["local_failures"]}
            stable = [all(repeat["logits"]["finite"] and repeat["logits"]["max_absolute_error"] == 0 for rank in audits for repeat in rank["checks"][offset]["native_repeats"]) for offset in range(len(checks))]
            accepted = sum(all(rank["checks"][offset]["accepted"] for rank in audits) and check["step"] not in failed_steps and stable[offset] for offset, check in enumerate(checks))
            output_ids = case["response"]["output_ids"]
            output_lengths = [len(output) for output in output_ids] if batch == 2 else [len(output_ids)]
            prompt = case["case"]["input_ids"]
            row = {"label": case["case"]["label"], "batch": batch, "input_tokens": len(prompt[0]) if batch == 2 else len(prompt), "output_tokens": output_lengths, "completed": case["completed"], "same_state_passed": case["accuracy_passed"], "decode_steps": len(checks), "accepted_decode_steps_all_ranks": accepted, "full_model_accepted_steps": sum(check["accepted"] for check in checks), "greedy_equal_steps": sum(all(check["distribution"]["argmax_equal"]) for check in checks), "max_kl": max(max(check["distribution"]["kl_reference_actual"]) for check in checks), "max_tv": max(max(check["distribution"]["total_variation"]) for check in checks), "max_logit_nrmse": max(check["logits"]["nrmse"] for check in checks), "native_repeat_max_absolute_error": max(repeat["logits"]["max_absolute_error"] for rank in audits for check in rank["checks"] for repeat in check["native_repeats"]), "all_99_kernels_counted": all(rank["all_99_kernels_counted"] for rank in audits), "independent_passed": independent["passed"], "independent_output_ids_equal": independent["output_ids_equal"], "independent_aligned_frames": sum(frame["same_inputs"] for frame in independent["frames"]), "independent_accepted_frames": sum(frame["accepted"] for frame in independent["frames"]), "independent_total_frames": len(independent["frames"]), "answer_passed": case.get("answer_passed"), "failures": [{"tp_rank": rank["tp_rank"], **failure} for rank in audits for failure in rank["local_failures"]]}
            prefill = [frame for frame in independent["frames"] if frame["mode"] != "DECODE" and frame["same_inputs"]]
            row["independent_prefill_max_absolute_error"] = max((frame["logits"]["max_absolute_error"] for frame in prefill), default=None)
            row["default_native_output_ids_equal"] = default_rows[batch, row["label"]]["output_ids_equal"]
            row["native_repeat_stable_steps"] = sum(stable)
            rows.append(row)
    local = {}
    for rank in range(8):
        state = json.loads((directory / "monokernel" / f"rank-{rank}.json").read_text())
        step = -1
        for check in state["shadow_checks"]:
            if check["kind"] == "embedding":
                step += 1
            metrics = check["checks"]
            if "finite" in metrics:
                metrics = {"output": metrics}
            for name, metric in metrics.items():
                key = f"{check['kind']}.{name}"
                aggregate = local.setdefault(key, {"checks": 0, "max_nrmse": 0, "max_absolute_error": 0, "all_finite": True})
                aggregate["checks"] += 1
                if metric["nrmse"] > aggregate["max_nrmse"]:
                    aggregate.update(max_nrmse=metric["nrmse"], worst_rank=rank, worst_layer=check.get("layer"), worst_step=step)
                aggregate["max_absolute_error"] = max(aggregate["max_absolute_error"], metric["max_absolute_error"])
                aggregate["all_finite"] &= metric["finite"]
    return {"passed": all(report["passed"] for report in reports.values()) and all(report["passed"] for report in comparisons.values()) and default_comparison["passed"], "completed_all_cases": all(case["completed"] for report in reports.values() for case in report["cases"]) and len(default_rows) == len(rows), "total_decode_steps": sum(row["decode_steps"] for row in rows), "total_decode_token_rows": sum(row["decode_steps"] * row["batch"] for row in rows), "reference_policy": reports["native", 1]["accuracy_reference"], "profile": reports["native", 1]["profile"], "source_sha256": reports["native", 1]["source_sha256"], "rows": rows, "local_metrics": local}


def markdown(report):
    lines = ["| Batch | 输入→输出 / 每请求 | 同状态全部门槛通过步数 | 最大 KL | 最大 TV | greedy 一致步数 | 参考重复稳定步数 | 最终结果 |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in report["rows"]:
        if row["answer_passed"] is not None:
            continue
        passed = row["same_state_passed"] and row["independent_passed"] and row["default_native_output_ids_equal"]
        lines.append(f"| {row['batch']} | {row['input_tokens']}→{row['output_tokens'][0]} | {row['accepted_decode_steps_all_ranks']}/{row['decode_steps']} | {row['max_kl']:.8f} | {row['max_tv']:.8f} | {row['greedy_equal_steps']}/{row['decode_steps']} | {row['native_repeat_stable_steps']}/{row['decode_steps']} | {'通过' if passed else '失败'} |")
    lines += ["", "| Batch | 输入→输出 / 每请求 | 默认原生 output_ids 完全一致 | 确定性基线 output_ids 完全一致 | 确定性基线 prefill 最大绝对误差 | 同前缀分布通过 / 可比帧 / 全部帧 |", "| ---: | --- | --- | --- | ---: | --- |"]
    for row in report["rows"]:
        if row["answer_passed"] is not None:
            continue
        lines.append(f"| {row['batch']} | {row['input_tokens']}→{row['output_tokens'][0]} | {'是' if row['default_native_output_ids_equal'] else '否'} | {'是' if row['independent_output_ids_equal'] else '否'} | {row['independent_prefill_max_absolute_error']} | {row['independent_accepted_frames']}/{row['independent_aligned_frames']}/{row['independent_total_frames']} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    arguments = parser.parse_args()
    report = summarize(arguments.report_dir)
    arguments.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    arguments.markdown.write_text(markdown(report))
    print(json.dumps({key: value for key, value in report.items() if key not in ("rows", "local_metrics", "source_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
