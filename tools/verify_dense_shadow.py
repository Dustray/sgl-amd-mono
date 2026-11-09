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
    parser.add_argument("--reference", type=Path, default=Path("monokernel_reports/dense_native_reference_matrix.json"))
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    cases = [case for case in json.loads(arguments.reference.read_text())["cases"] if case.get("concurrency") == 1]
    report = {"scope": "native/fused same-state model accuracy for serial, concurrent, long-input and prefix-cache requests", "passed": False, "requests": []}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def generate(case):
        payload = copy.deepcopy(case["payload"])
        payload["sampling_params"]["max_new_tokens"] = arguments.tokens
        payload.pop("return_logprob", None)
        payload.pop("top_logprobs_num", None)
        payload.pop("logprob_start_len", None)
        response = requests.post(arguments.url + "/generate", json=payload, timeout=240)
        response.raise_for_status()
        result = response.json()
        if len(result["output_ids"]) != arguments.tokens:
            raise RuntimeError(f"Incomplete generation for {case['label']}")
        return {"label": case["label"], "response": result}

    try:
        for concurrency in (1, 2):
            response = requests.post(arguments.url + "/flush_cache", params={"timeout": 10}, timeout=30)
            response.raise_for_status()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                for result in executor.map(generate, cases):
                    report["requests"].append({"concurrency": concurrency, **result})
                    save()
                    print(json.dumps({"concurrency": concurrency, "label": result["label"], "tokens": len(result["response"]["output_ids"])}), flush=True)
        previous_steps = len(json.loads((arguments.report_dir / "dense_model_shadow_validation.json").read_text())["checks"])
        repeated = generate(cases[0])
        report["prefix_cache_repeat"] = repeated
        shadow = json.loads((arguments.report_dir / "dense_model_shadow_validation.json").read_text())
        checks = shadow["checks"]
        report["shadow_steps"] = len(checks)
        report["prefix_shadow_steps"] = len(checks) - previous_steps
        report["rows_tested"] = sorted({check["rows"] for check in checks})
        report["logits_nrmse_max"] = max(error["normalized_rms_error"] for check in checks for error in check["logits_per_row"])
        report["checks_per_forward"] = sorted({len(check["local_checks"]) for check in checks})
        report["acceptance"] = shadow["acceptance"]
        report["implementation_sha256"] = shadow.get("implementation_sha256")
        report["shadow_report"] = str(arguments.report_dir / "dense_model_shadow_validation.json")
        report["passed"] = shadow.get("implementation_sha256") == implementation_digest() and shadow["passed"] and all(shadow_accepted(check) for check in checks) and report["rows_tested"] == [1, 2] and len(checks) >= 64 and report["prefix_shadow_steps"] >= arguments.tokens - 1
        save()
        if not report["passed"]:
            raise RuntimeError("Full model shadow validation failed or lacks required coverage")
    except Exception as error:
        report["passed"] = False
        report["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
