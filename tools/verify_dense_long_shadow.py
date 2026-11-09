import argparse
import concurrent.futures
import copy
import json
from pathlib import Path

import requests

from sgl_amd_mono.qwen38_27b.gfx936.check import shadow_accepted
from sgl_amd_mono.qwen38_27b.report import implementation_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    arguments = parser.parse_args()
    response = requests.get(arguments.url + "/server_info", timeout=30)
    response.raise_for_status()
    runtime = response.json()["internal_states"][0]["amd_dense_monokernel"]
    if Path(runtime["real_weight_validation"]).parent != arguments.report_dir.resolve():
        raise RuntimeError("Endpoint belongs to a different validation process")
    path = arguments.report_dir / "dense_model_shadow_validation.json"
    initial = len(json.loads(path.read_text())["checks"])
    cases = {case["label"]: case for case in json.loads(arguments.reference.read_text())["cases"] if case["concurrency"] == 1}
    order = ("long_output", "chinese", "english")
    report = {"scope": "same-state long-output validation; submit 128-token request first, overlap Chinese 64 tokens, then English 64 tokens", "passed": False, "submitted_order": list(order), "initial_shadow_steps": initial, "requests": []}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def generate(label):
        payload = copy.deepcopy(cases[label]["payload"])
        payload["sampling_params"]["max_new_tokens"] = 128 if label == "long_output" else 64
        response = requests.post(arguments.url + "/generate", json=payload, timeout=arguments.timeout)
        response.raise_for_status()
        actual = response.json()
        if len(actual["output_ids"]) != payload["sampling_params"]["max_new_tokens"]:
            raise RuntimeError(f"Incomplete generation for {label}")
        return {"label": label, "payload": payload, "response": actual}

    save()
    try:
        response = requests.post(arguments.url + "/flush_cache", params={"timeout": 10}, timeout=30)
        response.raise_for_status()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            pending = [executor.submit(generate, label) for label in order]
            for future in concurrent.futures.as_completed(pending):
                actual = future.result()
                report["requests"].append(actual)
                save()
                print(f"Completed {actual['label']}: {len(actual['response']['output_ids'])} tokens", flush=True)
        shadow = json.loads(path.read_text())
        checks = shadow["checks"]
        report.update(implementation_sha256=shadow["implementation_sha256"], shadow_report=str(path), shadow_steps=len(checks), rows_tested=sorted({check["rows"] for check in checks}), logits_nrmse_max=max(error["normalized_rms_error"] for check in checks for error in check["logits_per_row"]))
        report["new_shadow_steps"] = len(checks) - initial
        report["passed"] = shadow["implementation_sha256"] == implementation_digest() and shadow["passed"] and all(shadow_accepted(check) for check in checks) and {check["rows"] for check in checks[initial:]} == {1, 2} and 127 <= report["new_shadow_steps"] < 140
        save()
        if not report["passed"]:
            raise RuntimeError("Long shadow accuracy or complete-step coverage failed")
    except Exception as error:
        report["passed"] = False
        report["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
