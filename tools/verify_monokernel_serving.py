import argparse
import concurrent.futures
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--reference", type=Path, default=Path("monokernel_reports/ffn_native_reference_matrix.json"))
    parser.add_argument("--output", type=Path, default=Path("monokernel_reports/dense_serving_verification.json"))
    arguments = parser.parse_args()

    def snapshot():
        response = requests.get(arguments.url + "/server_info", timeout=120)
        response.raise_for_status()
        return response.json()["internal_states"][0]["amd_dense_monokernel"]

    def generate(payload):
        response = requests.post(arguments.url + "/generate", json=payload, timeout=600)
        response.raise_for_status()
        return response.json()

    def flush():
        response = requests.post(arguments.url + "/flush_cache", params={"timeout": 10}, timeout=30)
        response.raise_for_status()

    report = {"scope": "Complete dense K1/K2 correctness, graph execution and routing verification; performance benchmarking remains separate", "passed": False, "cases": [], "before": snapshot()}

    def save():
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    assert report["before"]["whole_model_monokernel"] and report["before"]["kernel_launches_per_decode_forward"] == 112
    flush()
    generate({"input_ids": [271], "sampling_params": {"temperature": 0, "max_new_tokens": 0}})
    after_prefill = snapshot()
    report["prefill_only_kernel_delta"] = after_prefill["monokernel_executions"] - report["before"]["monokernel_executions"]
    assert report["prefill_only_kernel_delta"] == 0
    reference = json.loads(arguments.reference.read_text())
    for concurrency in (1, 2):
        cases = [case for case in reference["cases"] if case.get("concurrency", concurrency) == concurrency]
        flush()
        before = snapshot()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            responses = list(executor.map(lambda case: generate(case["payload"]), cases))
        after = snapshot()
        for case, actual in zip(cases, responses):
            expected_ids = case["response"]["output_ids"]
            actual_ids = actual["output_ids"]
            prefix = 0
            for expected_id, actual_id in zip(expected_ids, actual_ids):
                if expected_id != actual_id:
                    break
                prefix += 1
            comparison = {"label": case["label"], "concurrency": concurrency, "exact_token_match": actual_ids == expected_ids, "common_prefix_tokens": prefix, "expected_tokens": len(expected_ids), "actual_tokens": len(actual_ids), "response": actual}
            report["cases"].append(comparison)
            print(json.dumps({key: value for key, value in comparison.items() if key != "response"}), flush=True)
            assert len(actual_ids) == case["payload"]["sampling_params"]["max_new_tokens"]
            if case["label"] == "arithmetic":
                assert "323" in actual["text"]
        kernel1_delta = [current - previous for current, previous in zip(after["k1_device_executions_per_layer"], before["k1_device_executions_per_layer"])]
        kernel2_delta = [current - previous for current, previous in zip(after["k2_device_executions_per_layer"], before["k2_device_executions_per_layer"])]
        assert len(set(kernel2_delta)) == 1 and kernel2_delta[0] > 0
        assert all(value == (0 if layer_index % 4 == 3 else kernel2_delta[0]) for layer_index, value in enumerate(kernel1_delta))
        total_delta = after["monokernel_executions"] - before["monokernel_executions"]
        assert total_delta == kernel2_delta[0] * 112
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
    report["passed"] = True
    save()


if __name__ == "__main__":
    main()
