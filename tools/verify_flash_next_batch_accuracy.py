import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

from benchmark_flash_next_serving import flash_state, source_hashes
from benchmark_qwen38_serving import PROFILE_KEYS, make_prompts, save_json
from verify_flash_next_correctness import read_ranks


def run(arguments):
    session = requests.Session()
    session.headers["Connection"] = "close"
    response = session.get(arguments.url + "/server_info", timeout=120)
    response.raise_for_status()
    info = response.json()
    assert info["max_running_requests"] == 2 and info["tp_size"] == 8
    runtime = flash_state(info)
    assert bool(runtime) == (arguments.mode == "monokernel")
    policy = next(state["amd_flash_accuracy_reference"] for state in info["internal_states"] if "amd_flash_accuracy_reference" in state)
    assert policy["enabled"]
    prompts = make_prompts(info["model_path"], arguments.lengths)
    cases = [{"label": f"batch2_{length}_64", "input_ids": prompts[str(length)], "max_new_tokens": 64, "ignore_eos": True, "expected": None} for length in arguments.lengths]
    profile_keys = (*PROFILE_KEYS, "moe_runner_backend", "max_total_tokens", "language_model_only", "ple_offload_embedding")
    sources = source_hashes()
    sources[str(Path(__file__).relative_to(Path(__file__).resolve().parents[1]))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"passed": False, "mode": arguments.mode, "started_at": datetime.now(timezone.utc).isoformat(), "source_sha256": sources, "profile": {key: info.get(key) for key in profile_keys}, "accuracy_reference": {key: value for key, value in policy.items() if key != "calls"}, "trace_directory": str(arguments.trace_dir.resolve()), "cases_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), "cases": []}
    save_json(arguments.output, report)

    def frames():
        index = arguments.trace_dir / "index.json"
        return len(json.loads(index.read_text())["frames"]) if index.exists() else 0

    try:
        for case in cases:
            session.get(arguments.url + "/flush_cache", timeout=120).raise_for_status()
            before = read_ranks(arguments.rank_report_dir) if runtime else None
            entry = {"case": case, "completed": False, "trace_start": frames()}
            report["cases"].append(entry)
            save_json(arguments.output, report)
            response = session.post(arguments.url + "/generate", json={"input_ids": case["input_ids"], "sampling_params": {"temperature": 0, "max_new_tokens": 64, "ignore_eos": True}}, timeout=1800)
            response.raise_for_status()
            outputs = response.json()
            assert len(outputs) == 2
            for output, prompt in zip(outputs, case["input_ids"]):
                assert len(output["output_ids"]) == 64
                meta = output["meta_info"]
                assert meta["prompt_tokens"] == len(prompt) and meta["completion_tokens"] == 64
                assert meta.get("cached_tokens", 0) == 0 and meta.get("num_retractions", 0) == 0
            entry.update(response={"output_ids": [output["output_ids"] for output in outputs], "raw": outputs}, trace_end=frames())
            if runtime:
                session.get(arguments.url + "/server_info", timeout=120).raise_for_status()
                after = read_ranks(arguments.rank_report_dir)
                entry["rank_audit"] = []
                for previous, current in zip(before, after):
                    steps = current["completed_decode_steps"] - previous["completed_decode_steps"]
                    checks = current["full_model_shadow_checks"][len(previous["full_model_shadow_checks"]):]
                    assert len(checks) == steps == 63
                    for offset, check in enumerate(checks):
                        assert check["rows"] == 2 and check["restored_state_tensors"] == 123
                        assert len(check["native_repeats"]) == 3
                        assert check["input_token_ids"] == [output["output_ids"][offset] for output in outputs]
                        assert check["distribution"]["actual_token_ids"] == [output["output_ids"][offset + 1] for output in outputs]
                        assert check["sequence_lengths"] == [len(prompt) + offset + 1 for prompt in case["input_ids"]]
                    repeats_equal = all(repeat["logits"]["finite"] and repeat["logits"]["max_absolute_error"] == 0 for check in checks for repeat in check["native_repeats"])
                    local_failures = current.get("serving_shadow_failures", [])[len(previous.get("serving_shadow_failures", [])):]
                    accepted = all(check["accepted"] for check in checks) and repeats_equal and not local_failures
                    if not arguments.diagnostic:
                        assert accepted
                    keys = [f"gdn.{layer}" if layer % 4 != 3 else f"qsa.{layer}" for layer in range(48)] + [f"post.{layer}" for layer in range(48)] + ["ple.1", "embedding.0", "tail.48"]
                    assert all(current["executions"][key] - previous["executions"][key] == steps for key in keys)
                    batch2 = sum(check["rows"] == 2 for check in checks)
                    assert batch2 > 0
                    entry["rank_audit"].append({"tp_rank": current["tp_rank"], "decode_steps": steps, "batch2_decode_steps": batch2, "all_99_kernels_counted": True, "checks": checks, "accepted": accepted, "local_failures": local_failures, "native_repeats_equal": repeats_equal})
                assert len(entry["rank_audit"]) == 8
            entry["accuracy_passed"] = all(rank["accepted"] for rank in entry.get("rank_audit", []))
            entry["completed"] = True
            save_json(arguments.output, report)
            print(json.dumps({"mode": arguments.mode, "case": case["label"], "accuracy_passed": entry["accuracy_passed"], "trace_frames": entry["trace_end"] - entry["trace_start"]}), flush=True)
        report["diagnostic"] = arguments.diagnostic
        report["passed"] = all(entry["accuracy_passed"] for entry in report["cases"])
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json(arguments.output, report)
    except Exception as error:
        report["error"] = repr(error)
        if runtime:
            report["failure_rank_states"] = read_ranks(arguments.rank_report_dir)
        save_json(arguments.output, report)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("native", "monokernel"), required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30160")
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--rank-report-dir", type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 512, 1536])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
