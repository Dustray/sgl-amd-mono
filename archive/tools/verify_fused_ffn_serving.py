import argparse
import concurrent.futures
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--reference", type=Path, default=Path("monokernel_reports/ffn_native_reference.json"))
    parser.add_argument("--output", type=Path, default=Path("monokernel_reports/ffn_serving_verification.json"))
    parser.add_argument("--capture-native-reference", action="store_true")
    arguments = parser.parse_args()

    def snapshot():
        response = requests.get(arguments.url + "/server_info", timeout=60)
        response.raise_for_status()
        return response.json()["internal_states"][0]["amd_fused_ffn"]

    def generate(payload):
        response = requests.post(arguments.url + "/generate", json=payload, timeout=600)
        response.raise_for_status()
        return response.json()

    def flush_cache():
        response = requests.post(arguments.url + "/flush_cache", params={"timeout": 10}, timeout=30)
        response.raise_for_status()

    def save():
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    reference = json.loads(arguments.reference.read_text())
    if arguments.capture_native_reference:
        information = requests.get(arguments.url + "/server_info", timeout=60)
        information.raise_for_status()
        information = information.json()
        if any("amd_fused_ffn" in state for state in information["internal_states"]):
            raise RuntimeError("Native reference endpoint has fusion enabled")
        report = {"scope": "Original SGLang correctness reference with a separate baseline for each concurrency", "cases": [], "server_info": information}
        for concurrency in (1, 2):
            cases = [case for case in reference["cases"] if case.get("concurrency", concurrency) == concurrency]
            flush_cache()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                responses = list(executor.map(lambda case: generate(case["payload"]), cases))
            report["cases"].extend({"label": case["label"], "concurrency": concurrency, "payload": case["payload"], "response": response}
                                   for case, response in zip(cases, responses))
            save()
            print(f"Captured {len(cases)} native cases with concurrency {concurrency}", flush=True)
        return

    report = {
        "scope": "Correctness and actual GPU execution verification; not a performance benchmark",
        "serving_checks_passed": False,
        "cases": [],
        "before": snapshot(),
    }
    save()
    if report["before"]["attached_layers"] != 64 or report["before"]["whole_model_monokernel"]:
        raise RuntimeError("Unexpected fusion scope")
    prefill = generate({"input_ids": [271], "sampling_params": {"temperature": 0, "max_new_tokens": 0}})
    after_prefill = snapshot()
    report["prefill_only"] = {
        "response": prefill,
        "ffn_execution_delta": after_prefill["fused_ffn_device_executions"] - report["before"]["fused_ffn_device_executions"],
    }
    save()
    if report["prefill_only"]["ffn_execution_delta"] != 0:
        raise RuntimeError("Prefill unexpectedly entered decode FFN fusion")
    for concurrency in (1, 2):
        cases = [case for case in reference["cases"] if case.get("concurrency", concurrency) == concurrency]
        flush_cache()
        before = snapshot()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            responses = list(executor.map(lambda case: generate(case["payload"]), cases))
        after = snapshot()
        for case, actual in zip(cases, responses):
            expected = case["response"]
            expected_ids = expected["output_ids"]
            actual_ids = actual["output_ids"]
            common_prefix = 0
            for expected_id, actual_id in zip(expected_ids, actual_ids):
                if expected_id != actual_id:
                    break
                common_prefix += 1
            logprob_difference = [
                abs(actual_probability[0] - expected_probability[0])
                for actual_probability, expected_probability in zip(
                    actual["meta_info"]["output_token_logprobs"][:common_prefix],
                    expected["meta_info"]["output_token_logprobs"][:common_prefix],
                )
            ]
            comparison = {
                "label": case["label"],
                "concurrency": concurrency,
                "exact_token_match": expected_ids == actual_ids,
                "common_prefix_tokens": common_prefix,
                "expected_tokens": len(expected_ids),
                "actual_tokens": len(actual_ids),
                "max_logprob_difference_on_common_prefix": max(logprob_difference, default=None),
                "response": actual,
            }
            report["cases"].append(comparison)
            if len(actual_ids) != case["payload"]["sampling_params"]["max_new_tokens"]:
                raise RuntimeError(f"Unexpected generation length: {case['label']}")
            if case["label"] == "arithmetic" and "323" not in actual["text"]:
                raise RuntimeError("Arithmetic smoke check failed")
            print(json.dumps({key: value for key, value in comparison.items() if key != "response"}), flush=True)
        per_layer_delta = [
            current - previous
            for current, previous in zip(after["fused_ffn_device_executions_per_layer"], before["fused_ffn_device_executions_per_layer"])
        ]
        report[f"concurrency_{concurrency}_execution"] = {
            "before": before,
            "after": after,
            "per_layer_delta": per_layer_delta,
            "kernel_launch_delta": after["fused_ffn_kernel_launches"] - before["fused_ffn_kernel_launches"],
        }
        save()
        if not all(value > 0 for value in per_layer_delta) or len(set(per_layer_delta)) != 1:
            raise RuntimeError(f"Not all FFNs executed consistently: {per_layer_delta}")
        if concurrency == 1:
            expected_steps = sum(case["payload"]["sampling_params"]["max_new_tokens"] - 1 for case in cases)
            expected_maximum = expected_steps + len(cases)
            report["serial_decode_step_bounds"] = {
                "minimum_per_layer": expected_steps,
                "maximum_per_layer": expected_maximum,
                "reason": "Overlap scheduling may launch one extra decode before consuming a request's final sampling result",
            }
            if not expected_steps <= per_layer_delta[0] <= expected_maximum:
                raise RuntimeError(f"Graph replay counter mismatch: {per_layer_delta}, expected {expected_steps}..{expected_maximum}")
    report["all_exact_tokens"] = all(case["exact_token_match"] for case in report["cases"])
    report["serving_checks_passed"] = True
    report["after"] = snapshot()
    save()


if __name__ == "__main__":
    main()
