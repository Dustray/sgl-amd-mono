import argparse
import concurrent.futures
import copy
import json
from pathlib import Path

import requests

from sgl_amd_mono.qwen38_27b.report import implementation_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--reference", type=Path, default=Path("monokernel_reports/dense_native_reference_matrix.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--required", action="store_true")
    parser.add_argument("--native-prefix-reference", type=Path)
    arguments = parser.parse_args()
    cases = {case["label"]: case for case in json.loads(arguments.reference.read_text())["cases"] if case["concurrency"] == 1}
    report = {"scope": "complete decode service transitions, prefill routing, streaming and prefix-cache reuse; execution counters verified after each phase", "implementation_sha256": implementation_digest(), "passed": False, "checks": []}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def snapshot():
        response = requests.get(arguments.url + "/server_info", timeout=30)
        response.raise_for_status()
        state = response.json()["internal_states"][0]["amd_dense_monokernel"]
        assert state["implementation_sha256"] == report["implementation_sha256"]
        assert state["complete_decode_model"] and state["full_attention_fused"]
        if arguments.required:
            assert set(state["accepted_features"]) == {"dense_eager", "dense_graph", "full_attention_eager", "full_attention_graph", "decode_model_eager", "decode_model_graph"}
        return state

    def flush():
        response = requests.post(arguments.url + "/flush_cache", params={"timeout": 10}, timeout=30)
        response.raise_for_status()

    def generate(label, tokens):
        payload = copy.deepcopy(cases[label]["payload"])
        payload.pop("return_logprob", None)
        payload.pop("top_logprobs_num", None)
        payload.pop("logprob_start_len", None)
        payload["sampling_params"].update(temperature=0, max_new_tokens=tokens, ignore_eos=True)
        response = requests.post(arguments.url + "/generate", json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        assert len(result["output_ids"]) == tokens
        return {"label": label, "tokens": tokens, "payload": payload, "response": result}

    def executions(before, label, prefill_only=False):
        after = snapshot()
        decode = after["k2_device_executions_per_layer"][0] - before["k2_device_executions_per_layer"][0]
        assert all(current - previous == decode for current, previous in zip(after["k2_device_executions_per_layer"], before["k2_device_executions_per_layer"]))
        assert all(current - previous == (0 if layer % 4 == 3 else decode) for layer, (current, previous) in enumerate(zip(after["k1_device_executions_per_layer"], before["k1_device_executions_per_layer"])))
        attention = after["attention_k1_device_executions_per_layer"]
        assert len(attention) == 16 and all(current - before["attention_k1_device_executions_per_layer"][layer] == decode for layer, current in attention.items())
        assert after["tail_device_executions"] - before["tail_device_executions"] == decode
        launches = after["monokernel_executions"] - before["monokernel_executions"]
        assert launches == decode * 129
        if prefill_only:
            assert decode == 0
        else:
            assert decode > 0
        report["checks"].append({"label": label, "decode_forwards": decode, "launches": launches, "all_layers_consistent": True})
        save()

    try:
        report["before"] = snapshot()
        flush()
        before = snapshot()
        generate("long_input", 1)
        executions(before, "prefill_only", prefill_only=True)
        report["transitions"] = []
        for round_index in range(arguments.rounds):
            flush()
            before = snapshot()
            serial = generate("arithmetic" if round_index % 2 == 0 else "long_input", 8)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                pending = [executor.submit(generate, label, tokens) for label, tokens in (("long_output", 48), ("chinese", 16))]
                concurrent_results = [future.result() for future in pending]
            report["transitions"].append({"round": round_index, "serial": serial, "concurrent": concurrent_results})
            executions(before, f"serial_concurrent_round_{round_index}")
        flush()
        before = snapshot()
        first = generate("long_input", 16)
        repeated = generate("long_input", 16)
        report["prefix_cache"] = {"first": first, "repeated": repeated, "exact_tokens": first["response"]["output_ids"] == repeated["response"]["output_ids"]}
        save()
        if arguments.native_prefix_reference is not None:
            reference = json.loads(arguments.native_prefix_reference.read_text())
            assert reference["payload"] == first["payload"]
            native_match = [result["response"]["output_ids"] == expected["output_ids"] for result, expected in zip((first, repeated), reference["responses"])]
            report["prefix_cache"].update(native_reference=str(arguments.native_prefix_reference), native_tokens_match=native_match)
            save()
            assert len(reference["responses"]) == 2 and all(native_match)
        else:
            assert first["response"]["output_ids"] == repeated["response"]["output_ids"]
        assert repeated["response"]["meta_info"]["cached_tokens"] > 0
        executions(before, "prefix_cache")
        before = snapshot()
        payload = {"text": "Count from one to ten.", "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True}, "stream": True}
        events = []
        with requests.post(arguments.url + "/generate", json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith(b"data: ") and line[6:] != b"[DONE]":
                    events.append(json.loads(line[6:]))
        assert events and len(events[-1]["output_ids"]) == 8
        assert len({len(event["output_ids"]) for event in events}) > 1
        report["streaming"] = {"events": len(events), "completion_tokens": len(events[-1]["output_ids"]), "final": events[-1]}
        executions(before, "streaming")
        response = requests.post(arguments.url + "/v1/chat/completions", json={"model": "Qwen3.8-27B", "messages": [{"role": "user", "content": "计算17乘19，只回答数字。"}], "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0, "max_tokens": 16}, timeout=300)
        response.raise_for_status()
        report["chat"] = response.json()
        assert "323" in report["chat"]["choices"][0]["message"]["content"]
        report["after"] = snapshot()
        report["passed"] = True
        save()
        print(json.dumps({"passed": True, "rounds": arguments.rounds, "checks": len(report["checks"]), "streaming_events": len(events)}))
    except Exception as error:
        report["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
