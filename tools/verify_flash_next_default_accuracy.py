import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import requests

from benchmark_flash_next_serving import flash_state, source_hashes
from benchmark_qwen38_serving import make_prompts, save_json
from verify_flash_next_correctness import make_cases


ROOT = Path(__file__).resolve().parents[1]


def collect(directory):
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("SGLANG_AMD_FLASH_"):
            environment.pop(name)
    environment.update(SGLANG_AMD_MONO_0518="0", SGLANG_AMD_FLASH_NEXT_0518="0")
    log_path = directory / "native-default-server.log"
    command = ["bash", "launch_qwen_flash_next.sh", "--skip-server-warmup", "--enable-metrics", "--enable-cache-report", "--stream-interval", "1", "--max-running-requests", "2"]
    with log_path.open("w") as log:
        server = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch = {"pid": server.pid, "command": command, "reference_policy_enabled": False, "started_at": datetime.now(timezone.utc).isoformat()}
    save_json(directory / "native-default-launch.json", launch)
    try:
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"Default native service exited: {server.returncode}")
            try:
                if requests.get("http://127.0.0.1:30160/health", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(3)
        else:
            raise TimeoutError("Default native service startup timed out")
        session = requests.Session()
        session.headers["Connection"] = "close"
        info = session.get("http://127.0.0.1:30160/server_info", timeout=120).json()
        assert flash_state(info) is None
        assert not any(state.get("amd_flash_accuracy_reference", {}).get("enabled") for state in info["internal_states"])
        tokenizer, cases = make_cases(info["model_path"], [64, 512, 2048, 3072])
        prompts = make_prompts(info["model_path"], [64, 512, 1536])
        batch_cases = [{"label": f"batch2_{length}_64", "input_ids": prompts[str(length)], "max_new_tokens": 64, "ignore_eos": True, "expected": None} for length in (64, 512, 1536)]
        comparisons = []
        for batch, planned in ((1, cases), (2, batch_cases)):
            fused = json.loads((directory / f"monokernel-batch{batch}.json").read_text())
            sources = source_hashes()
            assert all(fused["source_sha256"][name] == digest for name, digest in sources.items())
            assert fused["profile"] == {key: info.get(key) for key in fused["profile"]}
            report = {"passed": False, "completed": False, "mode": "native_default", "accuracy_reference": None, "source_sha256": sources, "profile": fused["profile"], "cases": []}
            path = directory / f"native-default-batch{batch}.json"
            save_json(path, report)
            for case, fused_case in zip(planned, fused["cases"], strict=True):
                assert case == fused_case["case"]
                session.get("http://127.0.0.1:30160/flush_cache", timeout=120).raise_for_status()
                response = session.post("http://127.0.0.1:30160/generate", json={"input_ids": case["input_ids"], "sampling_params": {"temperature": 0, "max_new_tokens": case["max_new_tokens"], "ignore_eos": case["ignore_eos"]}}, timeout=600)
                response.raise_for_status()
                raw = response.json()
                outputs = raw if batch == 2 else [raw]
                input_rows = case["input_ids"] if batch == 2 else [case["input_ids"]]
                for output, prompt in zip(outputs, input_rows, strict=True):
                    meta = output["meta_info"]
                    assert meta["prompt_tokens"] == len(prompt) and meta["completion_tokens"] == len(output["output_ids"])
                    assert meta.get("cached_tokens", 0) == 0 and meta.get("num_retractions", 0) == 0
                    if case["ignore_eos"]:
                        assert len(output["output_ids"]) == 64
                output_ids = [output["output_ids"] for output in outputs] if batch == 2 else raw["output_ids"]
                text = tokenizer.decode(raw["output_ids"], skip_special_tokens=True).strip() if batch == 1 else None
                answer_passed = text == case["expected"] if case["expected"] is not None else None
                equal = output_ids == fused_case["response"]["output_ids"]
                entry = {"case": case, "response": {"output_ids": output_ids, "raw": raw}, "decoded_output": text, "answer_passed": answer_passed, "fused_output_ids_equal": equal, "completed": True}
                report["cases"].append(entry)
                comparisons.append({"label": case["label"], "batch": batch, "input_tokens": len(input_rows[0]), "output_ids_equal": equal, "native_answer_passed": answer_passed, "fused_answer_passed": fused_case.get("answer_passed"), "passed": equal and answer_passed is not False and fused_case.get("answer_passed") is not False})
                save_json(path, report)
                print(json.dumps(comparisons[-1], ensure_ascii=False), flush=True)
            report.update(passed=all(entry["answer_passed"] is not False for entry in report["cases"]), completed=True)
            save_json(path, report)
        save_json(directory / "comparison-default.json", {"passed": all(row["passed"] for row in comparisons), "scope": "Unmodified native HC/TopK policy; independent free generation and exact token/answer comparison only, without per-step logits or independent cache tensor comparison", "rows": comparisons})
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        launch["stopped_at"] = datetime.now(timezone.utc).isoformat()
        save_json(directory / "native-default-launch.json", launch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--wait-for-matrix", action="store_true")
    arguments = parser.parse_args()
    directory = arguments.report_dir.resolve()
    if arguments.wait_for_matrix:
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            launch = json.loads((directory / "launch.json").read_text())
            native = launch.get("native", {})
            if native.get("stopped_at") and (directory / "native-batch2.json").exists():
                break
            time.sleep(3)
        else:
            raise TimeoutError("Matrix completion wait timed out")
    collect(directory)
    if not json.loads((directory / "comparison-default.json").read_text())["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
