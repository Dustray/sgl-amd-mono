import argparse
import concurrent.futures
import datetime
import json
from pathlib import Path
import time

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30160")
    parser.add_argument("--output", type=Path, default=Path("monokernel_reports/flash_next_tp8/serving_complete/verification.json"))
    args = parser.parse_args()
    report = {"started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "passed": False, "checks": [], "responses": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def snapshot():
        response = requests.get(args.url + "/server_info", timeout=120)
        response.raise_for_status()
        states = [state["amd_flash_next_monokernel"] for state in response.json()["internal_states"]]
        if len(states) == 8:
            return sorted(states, key=lambda state: state["tp_rank"])
        assert len(states) == 1 and states[0]["tp_size"] == 8
        leader = states[0]
        for attempt in range(1000):
            paths = sorted(args.output.parent.glob("rank-*.json"))
            if len(paths) == 8:
                reports = [json.loads(path.read_text()) for path in paths]
                if all(state["completed_decode_steps"] == leader["completed_decode_steps"] and state["executions"]["post.47"] == leader["executions"]["post.47"] and state["program"]["source_sha256"] == leader["program"]["source_sha256"] for state in reports):
                    return sorted(reports, key=lambda state: state["tp_rank"])
            time.sleep(0.01)
        raise AssertionError("TP leader and eight local rank reports did not reach the same decode step")

    def audit(before, label, prefill_only=False):
        after = snapshot()
        for previous, current in zip(before, after):
            steps = current["completed_decode_steps"] - previous["completed_decode_steps"]
            assert steps == 0 if prefill_only else steps > 0, (label, steps)
            main_counters = [f"gdn.{layer}" if layer % 4 != 3 else f"qsa.{layer}" for layer in range(48)] + [f"post.{layer}" for layer in range(48)] + ["ple.1", "embedding.0", "tail.48"]
            deltas = {key: current["executions"][key] - previous["executions"][key] for key in main_counters}
            assert len(deltas) == 99 and set(deltas.values()) == {steps}, (label, deltas)
            assert len(current["validation"]) == 96 and len(current["gdn_validation"]) == 72
            assert len(current["communication_validation"]) == 12
        report["checks"].append({"label": label, "ranks": len(after), "decode_forwards": after[0]["completed_decode_steps"] - before[0]["completed_decode_steps"], "all_99_kernels_counted": True})
        report["after"] = after
        save()
        return after

    def chat(label, text, expected=None, max_tokens=128):
        payload = {"model": "Qwen3.8-Flash-Next", "messages": [{"role": "user", "content": text}], "temperature": 0, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
        started = time.monotonic()
        response = requests.post(args.url + "/v1/chat/completions", json=payload, timeout=1200)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        passed = content == expected if expected is not None else bool(content)
        report["responses"].append({"label": label, "request": payload, "response": result, "elapsed_s": time.monotonic() - started, "passed": passed})
        save()
        assert passed, (label, content, expected)
        return result

    def raw(tokens):
        response = requests.post(args.url + "/generate", json={"input_ids": tokens, "sampling_params": {"temperature": 0, "max_new_tokens": 32, "ignore_eos": True}}, timeout=1200)
        response.raise_for_status()
        result = response.json()
        assert len(result["output_ids"]) == 32
        return {"input_ids": tokens, "response": result}

    save()
    try:
        before = snapshot()
        report["before"] = before
        response = requests.post(args.url + "/generate", json={"input_ids": [42] * 300, "sampling_params": {"temperature": 0, "max_new_tokens": 1}}, timeout=1200)
        response.raise_for_status()
        before = audit(before, "prefill_only", prefill_only=True)
        chat("arithmetic", "只输出 7×8 的结果，不要解释。", "56", 16)
        before = audit(before, "arithmetic")
        chat("chinese", "请用一句中文解释什么是张量并行。")
        before = audit(before, "chinese")
        chat("english", "In one sentence, explain why the sky appears blue.")
        before = audit(before, "english")
        context = "这里是无关背景资料，请继续阅读。" * 100
        chat("retrieval", context + "\n唯一的验证码是 FLASH936。\n" + context + "\n请仅输出文中的唯一验证码，不要解释。", "FLASH936", 24)
        before = audit(before, "retrieval")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(raw, [42] * 32), executor.submit(raw, [43] * 32)]
            report["concurrent"] = [future.result() for future in futures]
        after = audit(before, "concurrent_batch2")
        assert all(any(key[0] == "DECODE" and key[1] == 2 and key[2] for key in state["forward_keys"]) for state in after), "batch-2 decode was not exercised"
        assert all(any(item["kind"] == "ple" for item in state["shadow_checks"]) and any(item["kind"] == "qsa" for item in state["shadow_checks"]) and any(item["kind"] == "tail" for item in state["shadow_checks"]) for state in after)
        assert all(state["whole_decoder_monokernel"] and state["acceptance_status"] == "serving_shadow_passed" for state in after)
        assert all(state["full_model_shadow_checks"] and all(all(check["argmax_equal"]) for check in state["full_model_shadow_checks"]) for state in after)
        assert all(all(check["distribution"]["finite"] and max(check["distribution"]["kl_reference_actual"]) <= 0.01 and max(check["distribution"]["total_variation"]) <= 0.05 and len(check["native_repeats"]) == 3 for check in state["full_model_shadow_checks"]) for state in after)
        assert all(state["validated_decode_batch_sizes"] == [1, 2] for state in after)
        report["passed"] = True
        report["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        print(json.dumps({"passed": True, "checks": report["checks"], "output": str(args.output)}, ensure_ascii=False))
    except Exception as error:
        report["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
