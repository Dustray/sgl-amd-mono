import argparse
import concurrent.futures
import json
from pathlib import Path

import requests

from sgl_amd_mono.qwen38_27b.gfx936.check import shadow_accepted
from sgl_amd_mono.qwen38_27b.report import implementation_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--reference", type=Path, default=Path("monokernel_reports/dense_native_reference_matrix.json"))
    parser.add_argument("--shadow-report", type=Path)
    parser.add_argument("--functional-only", action="store_true", help="Exit successfully for functional checks while keeping numerical acceptance separate")
    parser.add_argument("--output", type=Path, default=Path("monokernel_reports/dense_serving_verification.json"))
    arguments = parser.parse_args()

    def server_information():
        response = requests.get(arguments.url + "/server_info", timeout=120)
        response.raise_for_status()
        return response.json()

    def snapshot():
        return server_information()["internal_states"][0]["amd_dense_monokernel"]

    def generate(payload):
        response = requests.post(arguments.url + "/generate", json=payload, timeout=600)
        response.raise_for_status()
        return response.json()

    def flush():
        response = requests.post(arguments.url + "/flush_cache", params={"timeout": 10}, timeout=30)
        response.raise_for_status()

    reference = json.loads(arguments.reference.read_text())
    information = server_information()
    profile_fields = ("model_path", "tp_size", "pp_size", "context_length", "mem_fraction_static", "max_running_requests", "disable_overlap_schedule", "attention_backend", "mm_attention_backend", "kv_cache_dtype", "page_size", "chunked_prefill_size", "max_prefill_tokens", "mamba_radix_cache_strategy")
    profile = {name: information[name] for name in profile_fields}
    reference_profile = {name: reference["server_info"][name] for name in profile_fields}
    if profile != reference_profile:
        differences = {name: {"actual": profile[name], "reference": reference_profile[name]} for name in profile_fields if profile[name] != reference_profile[name]}
        raise RuntimeError(f"Native reference uses a different serving profile: {differences}; capture a matching reference and pass --reference")
    report = {"scope": "Dense K1/K2 functional serving and routing checks; numerical acceptance requires a separate same-state model shadow report", "passed": False, "functional_passed": False, "cases": [], "before": information["internal_states"][0]["amd_dense_monokernel"], "reference": str(arguments.reference), "serving_profile": profile, "reference_profile_matches": True, "decode_graph_backend": information.get("cuda_graph_backend_decode"), "reference_decode_graph_backend": reference["server_info"].get("cuda_graph_backend_decode")}

    def save():
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    launches_per_forward = report["before"]["kernel_launches_per_decode_forward"]
    expected_launches = 112 + (16 if report["before"].get("full_attention_fused") else 0) + int(report["before"].get("complete_decode_model", False))
    assert report["before"]["dense_decode_monokernel"] and launches_per_forward == expected_launches
    flush()
    generate({"input_ids": [271], "sampling_params": {"temperature": 0, "max_new_tokens": 0}})
    after_prefill = snapshot()
    report["prefill_only_kernel_delta"] = after_prefill["monokernel_executions"] - report["before"]["monokernel_executions"]
    assert report["prefill_only_kernel_delta"] == 0
    for concurrency in (1, 2):
        cases = [case for case in reference["cases"] if case.get("concurrency", concurrency) == concurrency]
        flush()
        before = snapshot()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending = {executor.submit(generate, case["payload"]): case for case in cases}
            for future in concurrent.futures.as_completed(pending):
                case = pending[future]
                try:
                    actual = future.result()
                except Exception as error:
                    report["error"] = {"label": case["label"], "concurrency": concurrency, "detail": repr(error)}
                    save()
                    raise
                expected_ids = case["response"]["output_ids"]
                actual_ids = actual["output_ids"]
                prefix = 0
                for expected_id, actual_id in zip(expected_ids, actual_ids):
                    if expected_id != actual_id:
                        break
                    prefix += 1
                comparison = {"label": case["label"], "concurrency": concurrency, "exact_token_match": actual_ids == expected_ids, "common_prefix_tokens": prefix, "expected_tokens": len(expected_ids), "actual_tokens": len(actual_ids), "response": actual}
                report["cases"].append(comparison)
                save()
                print(json.dumps({key: value for key, value in comparison.items() if key != "response"}), flush=True)
                assert len(actual_ids) == case["payload"]["sampling_params"]["max_new_tokens"]
                if case["label"] == "arithmetic":
                    assert "323" in actual["text"]
        after = snapshot()
        kernel1_delta = [current - previous for current, previous in zip(after["k1_device_executions_per_layer"], before["k1_device_executions_per_layer"])]
        kernel2_delta = [current - previous for current, previous in zip(after["k2_device_executions_per_layer"], before["k2_device_executions_per_layer"])]
        assert len(set(kernel2_delta)) == 1 and kernel2_delta[0] > 0
        assert all(value == (0 if layer_index % 4 == 3 else kernel2_delta[0]) for layer_index, value in enumerate(kernel1_delta))
        total_delta = after["monokernel_executions"] - before["monokernel_executions"]
        assert total_delta == kernel2_delta[0] * launches_per_forward
        attention_delta = {layer: current - before.get("attention_k1_device_executions_per_layer", {}).get(layer, 0) for layer, current in after.get("attention_k1_device_executions_per_layer", {}).items()}
        if report["before"].get("full_attention_fused"):
            assert len(attention_delta) == 16 and set(attention_delta.values()) == {kernel2_delta[0]}
        if report["before"].get("complete_decode_model"):
            assert after["tail_device_executions"] - before["tail_device_executions"] == kernel2_delta[0]
        report[f"concurrency_{concurrency}_execution"] = {"decode_forwards": kernel2_delta[0], "kernel_delta": total_delta, "k1_per_layer_delta": kernel1_delta, "k2_per_layer_delta": kernel2_delta}
        if concurrency == 1:
            minimum = sum(case["payload"]["sampling_params"]["max_new_tokens"] - 1 for case in cases)
            assert minimum <= kernel2_delta[0] <= minimum + len(cases)
        save()
    flush()
    payload = {"text": "User: What is 17 * 19? Reply with just the number.\nAssistant:", "sampling_params": {"temperature": 0, "max_new_tokens": 16, "ignore_eos": True}}
    first = generate(payload)
    second = generate(payload)
    report["prefix_cache_repeat"] = {"exact_token_match": first["output_ids"] == second["output_ids"], "first": first, "second": second}
    assert report["prefix_cache_repeat"]["exact_token_match"]
    response = requests.post(arguments.url + "/v1/chat/completions", json={"model": "Qwen3.8-27B", "messages": [{"role": "user", "content": "计算17乘19，只回答数字。"}], "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0, "max_tokens": 16}, timeout=300)
    response.raise_for_status()
    report["chat"] = response.json()
    assert "323" in report["chat"]["choices"][0]["message"]["content"]
    report["exact_token_cases"] = sum(case["exact_token_match"] for case in report["cases"])
    report["all_exact_tokens"] = all(case["exact_token_match"] for case in report["cases"])
    report["after"] = snapshot()
    report["functional_passed"] = True
    report["numerical_passed"] = False
    if arguments.shadow_report is not None:
        report["shadow_report"] = str(arguments.shadow_report)
        if arguments.shadow_report.exists():
            shadow = json.loads(arguments.shadow_report.read_text())
            checks = shadow["checks"]
            report["numerical_passed"] = shadow.get("implementation_sha256") == implementation_digest() and shadow["passed"] and len(checks) >= 64 and {check["rows"] for check in checks} == {1, 2} and all(shadow_accepted(check) for check in checks)
        else:
            report["numerical_unavailable"] = "Shadow report does not exist"
    report["passed"] = report["functional_passed"] and report["numerical_passed"]
    save()
    if not report["passed"] and not arguments.functional_only:
        raise RuntimeError("Functional serving checks passed, but numerical acceptance is missing or failed")


if __name__ == "__main__":
    main()
