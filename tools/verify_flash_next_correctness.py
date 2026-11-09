import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests

from benchmark_flash_next_serving import flash_state, source_hashes
from benchmark_qwen38_serving import PROFILE_KEYS, make_prompts, save_json


def make_cases(model_path, lengths=(512,)):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    questions = [
        ("multiply", "只输出 7×8 的结果，不要解释。", "56"),
        ("add", "计算 13+29，只输出一个整数。", "42"),
        ("subtract", "计算 100-37，只输出一个整数。", "63"),
        ("divide", "计算 144÷12，只输出一个整数。", "12"),
        ("negative", "计算 5-12，只输出一个整数。", "-7"),
        ("capital", "中国的首都是哪座城市？只输出城市名称。", "北京"),
        ("translate_en", "Translate 苹果 into English. Output only the lowercase English word.", "apple"),
        ("translate_zh", "把英文单词 cat 翻译成中文，只输出译文。", "猫"),
        ("extract_json", '读取 JSON {"a":17,"b":5}，只输出字段 b 的整数值。', "5"),
        ("sort", "将整数 3、1、2 从小到大排序。仅以英文逗号分隔，不加空格。", "1,2,3"),
        ("prime", "Is 17 a prime number? Reply with only YES or NO.", "YES"),
        ("retrieve_short", "唯一的验证码是 FLASH936。请仅输出验证码，不要解释。", "FLASH936"),
        ("retrieve_long", "这里是无关背景资料，请继续阅读。" * 100 + "\n唯一的验证码是 GFX936TP8。\n" + "这里是无关背景资料，请继续阅读。" * 100 + "\n请仅输出文中的唯一验证码，不要解释。", "GFX936TP8"),
    ]
    cases = []
    for label, question, expected in questions:
        text = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt = tokenizer.encode(text, add_special_tokens=False)
        cases.append({"label": label, "question": question, "input_ids": prompt, "max_new_tokens": 32, "ignore_eos": False, "expected": expected})
    prompts = make_prompts(model_path, list(lengths))
    for length in lengths:
        cases.append({"label": f"benchmark_{length}_64", "input_ids": prompts[str(length)][0], "max_new_tokens": 64, "ignore_eos": True, "expected": None})
    return tokenizer, cases


def read_ranks(directory):
    paths = sorted(directory.glob("rank-*.json"))
    if len(paths) != 8:
        raise AssertionError(f"Expected eight rank reports in {directory}, got {len(paths)}")
    return [json.loads(path.read_text()) for path in paths]


def audit(before, after, case, result, require_acceptance=True):
    expected_steps = len(result["output_ids"]) - 1
    records = []
    keys = [f"gdn.{layer}" if layer % 4 != 3 else f"qsa.{layer}" for layer in range(48)] + [f"post.{layer}" for layer in range(48)] + ["ple.1", "embedding.0", "tail.48"]
    for previous, current in zip(before, after):
        steps = current["completed_decode_steps"] - previous["completed_decode_steps"]
        checks = current["full_model_shadow_checks"][len(previous["full_model_shadow_checks"]):]
        assert steps == expected_steps == len(checks), (case["label"], current["tp_rank"], steps, expected_steps, len(checks))
        assert all(current["executions"][key] - previous["executions"][key] == steps for key in keys)
        for offset, check in enumerate(checks):
            distribution = check["distribution"]
            if require_acceptance:
                assert distribution["finite"] and all(distribution["argmax_equal"])
                assert max(distribution["kl_reference_actual"]) <= 0.01 and max(distribution["total_variation"]) <= 0.05
            assert check["input_token_ids"] == [result["output_ids"][offset]]
            assert distribution["actual_token_ids"] == [result["output_ids"][offset + 1]]
            assert check["sequence_lengths"] == [len(case["input_ids"]) + offset + 1]
            assert check["restored_state_tensors"] == 123 and len(check["native_repeats"]) == 3
        local_failures = current.get("serving_shadow_failures", [])[len(previous.get("serving_shadow_failures", [])):]
        repeats_equal = all(repeated["logits"]["finite"] and repeated["logits"]["max_absolute_error"] == 0 for check in checks for repeated in check["native_repeats"])
        accepted = all(check["accepted"] for check in checks) and not local_failures and repeats_equal
        if require_acceptance:
            assert accepted
        records.append({"tp_rank": current["tp_rank"], "decode_steps": steps, "shadow_checks": len(checks), "all_99_kernels_counted": True, "checks": checks, "accepted": accepted, "local_failures": local_failures, "native_repeats_equal": repeats_equal})
    assert len(records) == 8
    return records


def run(arguments):
    session = requests.Session()
    session.headers["Connection"] = "close"
    info = session.get(arguments.url + "/server_info", timeout=120)
    info.raise_for_status()
    info = info.json()
    runtime = flash_state(info)
    if arguments.mode == "monokernel":
        assert runtime and runtime["shadow_rearmed_on_prefill"] and runtime["shadow_steps_per_batch_size"] >= 63
        assert arguments.rank_report_dir is not None
    else:
        assert runtime is None
    assert info["tp_size"] == 8 and info["max_running_requests"] in (1, 2)
    tokenizer, cases = make_cases(info["model_path"], arguments.lengths)
    profile_keys = (*PROFILE_KEYS, "moe_runner_backend", "max_total_tokens", "language_model_only", "ple_offload_embedding")
    sources = source_hashes()
    sources[str(Path(__file__).relative_to(Path(__file__).resolve().parents[1]))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"passed": False, "mode": arguments.mode, "started_at": datetime.now(timezone.utc).isoformat(), "profile": {key: info.get(key) for key in profile_keys}, "source_sha256": sources, "cases_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), "cases": [], "numeric_policy": {"kl_reference_actual": 0.01, "total_variation": 0.05, "greedy_argmax_equal": True}, "scope": "Independent native/fused generation and strict answers; every fused decode is shadowed against native from the same fused prefix/cache, with three native repeats. This does not compare independently accumulated native cache states."}
    policy = next((state["amd_flash_accuracy_reference"] for state in info.get("internal_states", []) if "amd_flash_accuracy_reference" in state), None)
    report["accuracy_reference"] = {key: value for key, value in policy.items() if key != "calls"} if policy else None
    report["trace_directory"] = str(arguments.trace_dir.resolve()) if arguments.trace_dir else None
    save_json(arguments.output, report)
    try:
        for case in cases:
            flushed = session.get(arguments.url + "/flush_cache", timeout=120)
            flushed.raise_for_status()
            if runtime:
                before = read_ranks(arguments.rank_report_dir)
            entry = {"case": case, "completed": False}
            if arguments.trace_dir:
                index = arguments.trace_dir / "index.json"
                entry["trace_start"] = len(json.loads(index.read_text())["frames"]) if index.exists() else 0
            report["cases"].append(entry)
            save_json(arguments.output, report)
            started = time.monotonic()
            response = session.post(arguments.url + "/generate", json={"input_ids": case["input_ids"], "sampling_params": {"temperature": 0, "max_new_tokens": case["max_new_tokens"], "ignore_eos": case["ignore_eos"]}}, timeout=1800)
            entry["elapsed_s"] = time.monotonic() - started
            response.raise_for_status()
            result = response.json()
            entry["response"] = result
            entry["decoded_output"] = tokenizer.decode(result["output_ids"], skip_special_tokens=True).strip()
            meta = result["meta_info"]
            assert meta["prompt_tokens"] == len(case["input_ids"]) and meta["completion_tokens"] == len(result["output_ids"])
            assert meta.get("cached_tokens", 0) == 0 and meta.get("num_retractions", 0) == 0
            if case["ignore_eos"]:
                assert len(result["output_ids"]) == case["max_new_tokens"]
            entry["answer_passed"] = entry["decoded_output"] == case["expected"] if case["expected"] is not None else None
            if arguments.trace_dir:
                entry["trace_end"] = len(json.loads((arguments.trace_dir / "index.json").read_text())["frames"])
            if runtime:
                session.get(arguments.url + "/server_info", timeout=120).raise_for_status()
                after = read_ranks(arguments.rank_report_dir)
                entry["rank_audit"] = audit(before, after, case, result, require_acceptance=not arguments.diagnostic)
                if policy and not arguments.diagnostic:
                    assert all(repeated["logits"]["finite"] and repeated["logits"]["max_absolute_error"] == 0 for rank in entry["rank_audit"] for check in rank["checks"] for repeated in check["native_repeats"]), "Deterministic native repeats differ"
            entry["accuracy_passed"] = entry["answer_passed"] is not False and all(rank["accepted"] for rank in entry.get("rank_audit", []))
            entry["completed"] = True
            save_json(arguments.output, report)
            print(json.dumps({"mode": arguments.mode, "case": case["label"], "output_tokens": len(result["output_ids"]), "answer_passed": entry["answer_passed"], "elapsed_s": entry["elapsed_s"]}, ensure_ascii=False), flush=True)
        report["diagnostic"] = arguments.diagnostic
        report["passed"] = all(entry["completed"] and entry["accuracy_passed"] for entry in report["cases"])
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json(arguments.output, report)
    except Exception as error:
        report["error"] = repr(error)
        if runtime:
            report["failure_rank_states"] = read_ranks(arguments.rank_report_dir)
        save_json(arguments.output, report)
        raise


def compare(arguments):
    native = json.loads(arguments.native.read_text())
    mono = json.loads(arguments.monokernel.read_text())
    assert native["profile"] == mono["profile"] and native["cases_sha256"] == mono["cases_sha256"]
    assert native["source_sha256"] == mono["source_sha256"]
    assert native.get("accuracy_reference") == mono.get("accuracy_reference"), "Accuracy reference policies differ"
    assert len(native["cases"]) == len(mono["cases"])
    rows = []
    for baseline, fused in zip(native["cases"], mono["cases"]):
        assert baseline["case"] == fused["case"]
        row = {"label": baseline["case"]["label"], "input_tokens": len(baseline["case"]["input_ids"]), "expected": baseline["case"]["expected"], "native_answer_passed": baseline.get("answer_passed"), "monokernel_answer_passed": fused.get("answer_passed"), "native_text": baseline.get("decoded_output"), "monokernel_text": fused.get("decoded_output"), "monokernel_completed": fused["completed"]}
        if "response" in baseline and "response" in fused:
            baseline_tokens, fused_tokens = baseline["response"]["output_ids"], fused["response"]["output_ids"]
            row["output_tokens_equal"] = baseline_tokens == fused_tokens
            row["first_different_token"] = next((index for index, (expected, actual) in enumerate(zip(baseline_tokens, fused_tokens)) if expected != actual), None)
            if row["first_different_token"] is None and len(baseline_tokens) != len(fused_tokens):
                row["first_different_token"] = min(len(baseline_tokens), len(fused_tokens))
        checks = [check for rank in fused.get("rank_audit", []) if rank["tp_rank"] == 0 for check in rank["checks"]]
        row["checked_decode_steps"] = len(checks)
        row["max_kl"] = max((max(check["distribution"]["kl_reference_actual"]) for check in checks), default=None)
        row["max_tv"] = max((max(check["distribution"]["total_variation"]) for check in checks), default=None)
        row["argmax_all_equal"] = all(all(check["argmax_equal"]) for check in checks) if checks else None
        rows.append(row)
    result = {"passed": native["passed"] and mono["passed"], "numeric_policy": mono["numeric_policy"], "scope": mono["scope"], "native_raw": str(arguments.native), "monokernel_raw": str(arguments.monokernel), "rows": rows}
    save_json(arguments.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def trajectory_audit(arguments):
    session = requests.Session()
    session.headers["Connection"] = "close"
    response = session.get(arguments.url + "/server_info", timeout=120)
    response.raise_for_status()
    info = response.json()
    runtime = flash_state(info)
    assert runtime and runtime["shadow_record_failures"] and runtime["shadow_rearmed_on_prefill"] and runtime["shadow_steps_per_batch_size"] >= 63
    reference = json.loads(arguments.native.read_text())
    case = next(entry["case"] for entry in reference["cases"] if entry["case"]["label"] == "benchmark_512_64")
    assert len(case["input_ids"]) == 512 and case["max_new_tokens"] == 64
    profile_keys = (*PROFILE_KEYS, "moe_runner_backend", "max_total_tokens", "language_model_only", "ple_offload_embedding")
    profile = {key: info.get(key) for key in profile_keys}
    assert profile == reference["profile"]
    report = {"passed": False, "completed": False, "started_at": datetime.now(timezone.utc).isoformat(), "diagnostic_only": True, "case": case, "native_raw": str(arguments.native), "profile": profile, "source_sha256": source_hashes(), "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "numeric_policy": {"kl_reference_actual": 0.01, "total_variation": 0.05, "greedy_argmax_equal": True}, "scope": "All 63 same-state native/fused decode comparisons on the fused trajectory. Full-model numeric failures are recorded and remain failed; local numerical/state errors still abort. Not a timing run or independently accumulated native-cache comparison."}
    save_json(arguments.output, report)
    try:
        session.get(arguments.url + "/flush_cache", timeout=120).raise_for_status()
        before = read_ranks(arguments.rank_report_dir)
        response = session.post(arguments.url + "/generate", json={"input_ids": case["input_ids"], "sampling_params": {"temperature": 0, "max_new_tokens": 64, "ignore_eos": True}}, timeout=1800)
        response.raise_for_status()
        result = response.json()
        report["response"] = result
        assert len(result["output_ids"]) == 64 and result["meta_info"]["prompt_tokens"] == 512
        assert result["meta_info"].get("cached_tokens", 0) == 0 and result["meta_info"].get("num_retractions", 0) == 0
        session.get(arguments.url + "/server_info", timeout=120).raise_for_status()
        after = read_ranks(arguments.rank_report_dir)
        report["rank_audit"] = audit(before, after, case, result, require_acceptance=False)
        for state in after:
            failed = any(not check["accepted"] for check in state["full_model_shadow_checks"])
            assert state["acceptance_status"] == ("serving_shadow_failed" if failed else "serving_shadow_passed")
            assert state["whole_decoder_monokernel"] is not failed
            assert state["validated_decode_batch_sizes"] == ([] if failed else [1])
        checks = report["rank_audit"][0]["checks"]
        report["summary"] = {"checked_decode_steps": len(checks), "accepted_steps": sum(check["accepted"] for check in checks), "argmax_equal_steps": sum(all(check["argmax_equal"]) for check in checks), "max_kl": max(max(check["distribution"]["kl_reference_actual"]) for check in checks), "max_tv": max(max(check["distribution"]["total_variation"]) for check in checks), "native_repeat_argmax_mismatch_steps": sum(any(not all(repeat["distribution"]["argmax_equal"]) for repeat in check["native_repeats"]) for check in checks), "failed_steps": [{"decode_step": index + 1, "output_position": index + 2, "check": check} for index, check in enumerate(checks) if not check["accepted"]]}
        report["rank_acceptance_status"] = [{"tp_rank": state["tp_rank"], "acceptance_status": state["acceptance_status"], "whole_decoder_monokernel": state["whole_decoder_monokernel"], "validated_decode_batch_sizes": state["validated_decode_batch_sizes"]} for state in after]
        report["passed"] = all(all(check["accepted"] for check in rank["checks"]) for rank in report["rank_audit"])
        report["completed"] = True
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json(arguments.output, report)
        print(json.dumps({"passed": report["passed"], "completed": True, "summary": {key: value for key, value in report["summary"].items() if key != "failed_steps"}}, ensure_ascii=False), flush=True)
    except Exception as error:
        report["error"] = repr(error)
        report["failure_rank_states"] = read_ranks(arguments.rank_report_dir)
        save_json(arguments.output, report)
        raise


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--mode", choices=("native", "monokernel"), required=True)
    runner.add_argument("--url", default="http://127.0.0.1:30160")
    runner.add_argument("--rank-report-dir", type=Path)
    runner.add_argument("--trace-dir", type=Path)
    runner.add_argument("--lengths", nargs="+", type=int, default=[512])
    runner.add_argument("--diagnostic", action="store_true")
    runner.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--native", type=Path, required=True)
    comparison.add_argument("--monokernel", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    trajectory = commands.add_parser("trajectory-audit")
    trajectory.add_argument("--url", default="http://127.0.0.1:30160")
    trajectory.add_argument("--native", type=Path, required=True)
    trajectory.add_argument("--rank-report-dir", type=Path, required=True)
    trajectory.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "run":
        run(arguments)
    elif arguments.command == "compare":
        compare(arguments)
    else:
        trajectory_audit(arguments)


if __name__ == "__main__":
    main()
