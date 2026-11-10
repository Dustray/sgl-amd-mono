import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import random
import statistics
import time

import requests


WORKLOADS = [(input_len, output_len) for input_len in (64, 512, 2048) for output_len in (16, 64, 128)] + [(16, 1024)]
COUNTERS = [f"gdn.{layer}" if layer % 4 != 3 else f"qsa.{layer}" for layer in range(48)] + [f"post.{layer}" for layer in range(48)] + ["ple.1", "embedding.0", "tail.48"]


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def server_info(session, url):
    response = session.get(url + "/server_info", timeout=120)
    response.raise_for_status()
    return response.json()


def rank_states(info, directory):
    leaders = [state["amd_flash_next_monokernel"] for state in info["internal_states"] if "amd_flash_next_monokernel" in state]
    if not leaders:
        return []
    assert directory is not None
    leader = leaders[0]
    for attempt in range(200):
        reports = [json.loads(path.read_text()) for path in sorted(directory.glob("rank-*.json"))]
        if len(reports) == 8 and all(state["completed_decode_steps"] == leader["completed_decode_steps"] and state["execution_mode"] == leader["execution_mode"] for state in reports):
            return reports
        time.sleep(0.02)
    raise AssertionError("TP8 execution reports did not synchronize")


def audit(before, after, mode):
    if not before:
        assert not after
        return []
    audits = []
    for previous, current in zip(before, after):
        steps = current["completed_decode_steps"] - previous["completed_decode_steps"]
        deltas = {key: current["executions"][key] - previous["executions"][key] for key in COUNTERS}
        assert set(deltas.values()) == {steps}
        assert steps > 0 if mode == "monokernel" else steps == 0
        assert len(previous["shadow_checks"]) == len(current["shadow_checks"])
        assert len(previous["full_model_shadow_checks"]) == len(current["full_model_shadow_checks"])
        audits.append({"rank": current["tp_rank"], "decode_forwards": steps, "mathematical_launches": sum(deltas.values()), "timed_shadow_forwards": 0, "acceptance_status": current["acceptance_status"], "execution_mode": current["execution_mode"]})
    return audits


def stream_case(session, url, inputs, output_len):
    response = session.post(url + "/flush_cache", timeout=120)
    response.raise_for_status()
    payload = {"input_ids": inputs, "stream": True, "sampling_params": {"temperature": 0, "ignore_eos": True, "max_new_tokens": output_len, "stream_interval": 1}}
    first = {}
    last = {}
    final = {}
    arrivals = {index: [] for index in range(len(inputs))}
    started = time.perf_counter()
    with session.post(url + "/generate", json=payload, stream=True, timeout=1200) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            received = time.perf_counter() - started
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            event = json.loads(line[5:])
            if "error" in event:
                raise RuntimeError(event)
            index = event.get("index", 0)
            metadata = event["meta_info"]
            tokens = metadata["completion_tokens"]
            if tokens and (index not in final or tokens > final[index]["meta_info"]["completion_tokens"]):
                first.setdefault(index, received)
                last[index] = received
                arrivals[index].append([tokens, received])
            final[index] = event
    elapsed = time.perf_counter() - started
    assert set(final) == set(range(len(inputs))), final.keys()
    prefill = []
    for index, event in final.items():
        metadata = event["meta_info"]
        assert metadata["prompt_tokens"] == len(inputs[index]), metadata
        assert metadata["completion_tokens"] == output_len and len(event["output_ids"]) == output_len
        assert metadata["cached_tokens"] == 0, metadata
        assert metadata.get("num_retractions", 0) == 0, metadata
        assert metadata["finish_reason"]["type"] == "length", metadata
        if "prefill_finished_time" in metadata and "forward_entry_time" in metadata:
            prefill.append(metadata["prefill_finished_time"] - metadata["forward_entry_time"])
    decode = max(last.values()) - min(first.values())
    assert 0 < decode < elapsed
    return {"e2e_s": elapsed, "ttft_s": max(first.values()), "prefill_s": max(prefill) if len(prefill) == len(inputs) else None, "prefill_mean_s": statistics.mean(prefill) if len(prefill) == len(inputs) else None, "decode_s": decode, "decode_tps": len(inputs) * (output_len - 1) / decode, "decode_tps_per_request": (output_len - 1) / decode, "output_tps_e2e": len(inputs) * output_len / elapsed, "first_token_spread_s": max(first.values()) - min(first.values()), "token_arrivals": arrivals, "responses": final}


def run(arguments):
    directory = arguments.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    result_file = directory / "samples.jsonl"
    if result_file.exists():
        raise FileExistsError(result_file)
    if arguments.control_file:
        target = "monokernel" if arguments.mode == "monokernel" else "native"
        temporary = arguments.control_file.with_suffix(".tmp")
        temporary.write_text(target + "\n")
        temporary.replace(arguments.control_file)
    with requests.Session() as session:
        info = server_info(session, arguments.url)
        save(directory / "server_info_before.json", info)
        assert info["tp_size"] == 8 and info["max_running_requests"] >= 2
        ranks = rank_states(info, arguments.rank_report_dir)
        if arguments.mode in ("native_bf16", "monokernel"):
            expected = "monokernel" if arguments.mode == "monokernel" else "native"
            assert len(ranks) == 8 and all(state["execution_mode"] == expected for state in ranks)
        else:
            assert not ranks
        if arguments.mode == "native_graph":
            assert not info.get("speculative_algorithm")
        if arguments.mode == "mtp":
            assert info["speculative_algorithm"] == "EAGLE"
        if arguments.fixture.exists():
            fixture = json.loads(arguments.fixture.read_text())
        else:
            config = json.loads((Path(info["model_path"]) / "config.json").read_text())
            vocab = config.get("text_config", config)["vocab_size"]
            generator = random.Random(42)
            fixture = {str(length): [[generator.randrange(1000, vocab - 1000) for token in range(length)] for batch in range(2)] for length in (16, 64, 512, 2048)}
            save(arguments.fixture, fixture)
        root = Path(__file__).resolve().parents[1]
        sources = sorted((root / "sgl_amd_mono/qwen38_flash_next/gfx936").glob("*.py")) + sorted((root / "sgl_amd_mono/qwen38_flash_next/gfx936").glob("*.cpp")) + [root / "compat/sglang0518/src/sgl_amd_mono_0518/flash_next.py", Path(__file__)]
        report = {"mode": arguments.mode, "started_at": datetime.now(timezone.utc).isoformat(), "passed": False, "repeats": arguments.repeats, "warmups_per_case": arguments.warmups, "batch_sizes": arguments.batch_sizes, "workloads": WORKLOADS, "fixture": str(arguments.fixture.resolve()), "fixture_sha256": hashlib.sha256(arguments.fixture.read_bytes()).hexdigest(), "source_sha256": {str(source.relative_to(root)): hashlib.sha256(source.read_bytes()).hexdigest() for source in sources}, "versions": {name: metadata.version(name) for name in ("sglang", "torch", "lightop", "aiter")}, "definitions": {"e2e_s": "request send to stream EOF", "ttft_s": "last request first output token arrival minus request send", "prefill_s": "maximum per-request scheduler prefill_finished_time minus forward_entry_time; null when metadata unavailable", "decode_s": "last output token arrival in batch minus last request first output token arrival", "decode_tps": "batch_size * (output_len - 1) / decode_s; aggregate throughput", "sampling": "temperature=0, ignore_eos=true; cache flush before every wave; zero cache hits and retractions required"}, "completed_cases": 0, "accuracy_acceptance": False}
        report["definitions"]["decode_s"] = "last output token arrival in batch minus earliest request first output token arrival; includes overlapping prefill for staggered requests"
        report["warmup_output_cap"] = 128
        save(directory / "matrix.json", report)
        try:
            with result_file.open("a") as samples:
                for batch in arguments.batch_sizes:
                    for input_len, output_len in WORKLOADS:
                        inputs = fixture[str(input_len)][:batch]
                        for warmup in range(arguments.warmups):
                            stream_case(session, arguments.url, inputs, min(output_len, 128))
                        before = rank_states(server_info(session, arguments.url), arguments.rank_report_dir)
                        for repeat in range(1, arguments.repeats + 1):
                            result = stream_case(session, arguments.url, inputs, output_len)
                            samples.write(json.dumps({"mode": arguments.mode, "batch_size": batch, "input_len": input_len, "output_len": output_len, "repeat": repeat, **result}, ensure_ascii=False) + "\n")
                            samples.flush()
                            print(f"{arguments.mode} B{batch} {input_len}->{output_len} #{repeat}: E2E={result['e2e_s']:.4f}s TTFT={result['ttft_s']:.4f}s Decode={result['decode_s']:.4f}s TPS={result['decode_tps']:.2f}", flush=True)
                        after = rank_states(server_info(session, arguments.url), arguments.rank_report_dir)
                        checks = audit(before, after, arguments.mode)
                        with (directory / "execution_audit.jsonl").open("a") as audits:
                            audits.write(json.dumps({"batch_size": batch, "input_len": input_len, "output_len": output_len, "ranks": checks}) + "\n")
                        report["completed_cases"] += 1
                        save(directory / "matrix.json", report)
            save(directory / "server_info_after.json", server_info(session, arguments.url))
            report["passed"] = True
        except BaseException as error:
            report["error"] = repr(error)
            raise
        finally:
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            save(directory / "matrix.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("native_graph", "native_bf16", "monokernel", "mtp"), required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--rank-report-dir", type=Path)
    parser.add_argument("--control-file", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--batch-sizes", type=int, nargs="+", choices=(1, 2), default=[1, 2])
    arguments = parser.parse_args()
    assert arguments.repeats >= 2 and arguments.warmups >= 1
    run(arguments)


if __name__ == "__main__":
    main()
