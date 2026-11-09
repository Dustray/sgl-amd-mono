import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import aiohttp

from benchmark_qwen38_serving import PROFILE_KEYS, get_json, make_prompts, save_json, summarize, wave


def flash_state(info):
    return next((state["amd_flash_next_monokernel"] for state in info.get("internal_states", []) if "amd_flash_next_monokernel" in state), None)


def rank_states(info, directory):
    leader = flash_state(info)
    for attempt in range(1000):
        reports = [json.loads(path.read_text()) for path in sorted(directory.glob("rank-*.json"))]
        if len(reports) == 8 and all(state["completed_decode_steps"] == leader["completed_decode_steps"] and state["executions"]["post.47"] == leader["executions"]["post.47"] and state["program"]["source_sha256"] == leader["program"]["source_sha256"] for state in reports):
            return reports
        time.sleep(0.01)
    raise AssertionError("Eight TP reports did not match the live server")


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "sgl_amd_mono/qwen38_flash_next/gfx936").glob("*"))
    sources += [root / "compat/sglang0518/src/sgl_amd_mono_0518/flash_next.py", root / "launch_qwen_flash_next.sh", root / "launch_qwen_flash_next_monokernel.sh", Path(__file__).resolve(), root / "tools/benchmark_qwen38_serving.py"]
    return {str(source.relative_to(root)): hashlib.sha256(source.read_bytes()).hexdigest() for source in sources if source.is_file() and source.suffix in (".py", ".cpp", ".json", ".sh")}


async def run(arguments):
    timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(force_close=True)) as session:
        info = await get_json(session, arguments.url + "/server_info")
        state = flash_state(info)
        assert info["tp_size"] == 8 and info["max_running_requests"] == 1
        assert info["context_length"] == 4096 and info["disable_overlap_schedule"] and info["disable_radix_cache"]
        assert info["cuda_graph_backend_decode"] == "disabled" and info["cuda_graph_backend_prefill"] == "disabled"
        assert info["enable_metrics"] and info["stream_interval"] == 1
        if arguments.mode == "monokernel":
            assert state and not state["shadow_rearmed_on_prefill"] and state["mathematical_launches_per_decode"] == 99
            assert arguments.rank_report_dir is not None
        else:
            assert state is None, "Native baseline must have no Flash-Next runtime attached"
        prompts = make_prompts(info["model_path"], arguments.input_lengths)
        profile_keys = (*PROFILE_KEYS, "moe_runner_backend", "max_total_tokens", "language_model_only", "ple_offload_embedding", "random_seed")
        report = {"passed": False, "mode": arguments.mode, "started_at": datetime.now(timezone.utc).isoformat(), "url": arguments.url,
                  "source_sha256": source_hashes(), "server_info_before": info, "profile": {key: info.get(key) for key in profile_keys},
                  "prompts": prompts, "prompt_sha256": hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest(),
                  "measurement": {"input_lengths": arguments.input_lengths, "output_lengths": arguments.output_lengths, "batch_size": 1, "repeats": arguments.repeats, "warmups_per_case": arguments.warmups,
                                  "sampling": "temperature=0, ignore_eos=true", "cache": "flush before every request; require cached_tokens=0",
                                  "e2e": "client request send to stream EOF", "prefill": "server prefill_finished_time minus forward_entry_time, includes first token",
                                  "decode": "client last token arrival minus first token arrival; output_tokens - 1 decode steps", "aggregation": "arithmetic mean of measured requests; warmups excluded"},
                  "allow_unaccepted_shadow": arguments.allow_unaccepted_shadow,
                  "accuracy_acceptance": False, "warmups": [], "cases": [], "execution_audit": []}
        save_json(arguments.output, report)
        try:
            for input_tokens in arguments.input_lengths:
                for output_tokens in arguments.output_lengths:
                    assert input_tokens + output_tokens <= info["context_length"]
                    workload = {"mode": arguments.mode, "input_tokens": input_tokens, "output_tokens": output_tokens}
                    for warmup in range(arguments.warmups):
                        result = await wave(session, arguments.url, prompts, input_tokens, output_tokens, 1)
                        report["warmups"].append({**workload, **result})
                        save_json(arguments.output, report)
                        print(json.dumps({**workload, "warmup": warmup + 1, "wall_s": result["wall_s"]}), flush=True)
                    before = await get_json(session, arguments.url + "/server_info")
                    if state:
                        before_ranks = rank_states(before, arguments.rank_report_dir)
                        assert all(rank["full_model_shadow_checks"] for rank in before_ranks)
                        if not arguments.allow_unaccepted_shadow:
                            assert all(rank["whole_decoder_monokernel"] and rank["acceptance_status"] == "serving_shadow_passed" and 1 in rank["validated_decode_batch_sizes"] for rank in before_ranks)
                        report.setdefault("rank_states_before", before_ranks)
                    case = {"input_tokens": input_tokens, "output_tokens": output_tokens, "concurrency": 1, "complete": False, "trials": []}
                    report["cases"].append(case)
                    for repeat in range(arguments.repeats):
                        result = await wave(session, arguments.url, prompts, input_tokens, output_tokens, 1)
                        case["trials"].append(result)
                        save_json(arguments.output, report)
                        print(json.dumps({**workload, "repeat": repeat + 1, "wall_s": result["wall_s"]}), flush=True)
                    after = await get_json(session, arguments.url + "/server_info")
                    if state:
                        after_ranks = rank_states(after, arguments.rank_report_dir)
                        keys = [f"gdn.{layer}" if layer % 4 != 3 else f"qsa.{layer}" for layer in range(48)] + [f"post.{layer}" for layer in range(48)] + ["ple.1", "embedding.0", "tail.48"]
                        for previous, current in zip(before_ranks, after_ranks):
                            steps = current["completed_decode_steps"] - previous["completed_decode_steps"]
                            assert steps == arguments.repeats * (output_tokens - 1)
                            deltas = {key: current["executions"][key] - previous["executions"][key] for key in keys}
                            assert len(deltas) == 99 and set(deltas.values()) == {steps}
                            assert len(previous["shadow_checks"]) == len(current["shadow_checks"])
                            assert len(previous["full_model_shadow_checks"]) == len(current["full_model_shadow_checks"])
                            if not arguments.allow_unaccepted_shadow:
                                assert current["whole_decoder_monokernel"] and current["acceptance_status"] == "serving_shadow_passed" and current["validated_decode_batch_sizes"] == [1]
                            report["execution_audit"].append({**workload, "rank": current["tp_rank"], "decode_forwards": steps, "mathematical_launches": sum(deltas.values()), "all_99_kernels_counted": True, "timed_shadow_forwards": 0, "acceptance_status": current["acceptance_status"]})
                        report["rank_states_after"] = after_ranks
                    else:
                        assert flash_state(after) is None
                    case["complete"] = True
                    report["summary"] = summarize(report["cases"])
                    save_json(arguments.output, report)
            report["server_info_after"] = after
            report["passed"] = True
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            save_json(arguments.output, report)
        except Exception as error:
            report["error"] = repr(error)
            save_json(arguments.output, report)
            raise


def compare(arguments):
    native = json.loads(arguments.native.read_text())
    mono = json.loads(arguments.monokernel.read_text())
    assert native["passed"] and mono["passed"]
    assert native["profile"] == mono["profile"], "Server configurations differ"
    assert native["prompt_sha256"] == mono["prompt_sha256"] and native["measurement"] == mono["measurement"]
    assert native["source_sha256"] == mono["source_sha256"], "Source changed between runs"
    assert len(native["summary"]) == len(mono["summary"])
    comparisons = []
    for native_row, mono_row in zip(native["summary"], mono["summary"]):
        workload = {key: native_row[key] for key in ("input_tokens", "output_tokens", "concurrency")}
        assert all(mono_row[key] == value for key, value in workload.items())
        comparisons.append({**workload, "native": native_row, "monokernel": mono_row,
                            "e2e_speedup": native_row["e2e_s"] / mono_row["e2e_s"],
                            "decode_speedup": native_row["decode_s"] / mono_row["decode_s"]})
    result = {"passed": True, "accuracy_acceptance": False, "model": "Qwen3.8-Flash-Next BF16", "hardware": "8 x gfx936, TP8", "comparisons": comparisons,
              "native_raw": str(arguments.native), "monokernel_raw": str(arguments.monokernel), "execution_audit": mono["execution_audit"]}
    if len(comparisons) == 1:
        result.update(comparisons[0])
    save_json(arguments.output, result)
    print("| 输入→输出 | 模式 | E2E s | Prefill ms | TTFT ms | Decode s | TPOT ms | 输出 token/s |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for comparison in comparisons:
        for name, row in (("原生", comparison["native"]), ("Monokernel", comparison["monokernel"])):
            print(f'| {row["input_tokens"]}→{row["output_tokens"]} | {name} | {row["e2e_s"]:.4f} | {row["prefill_s"] * 1000:.2f} | {row["ttft_s"] * 1000:.2f} | {row["decode_s"]:.4f} | {row["tpot_s"] * 1000:.2f} | {row["output_tokens_per_s"]:.2f} |')


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--mode", choices=("native", "monokernel"), required=True)
    runner.add_argument("--url", default="http://127.0.0.1:30160")
    runner.add_argument("--repeats", type=int, default=5)
    runner.add_argument("--warmups", type=int, default=2)
    runner.add_argument("--input-lengths", type=int, nargs="+", default=[512])
    runner.add_argument("--output-lengths", type=int, nargs="+", default=[64])
    runner.add_argument("--allow-unaccepted-shadow", action="store_true", help="Measure the fused route while retaining a failed accuracy status in the report")
    runner.add_argument("--rank-report-dir", type=Path)
    runner.add_argument("--output", type=Path, required=True)
    renderer = commands.add_parser("compare")
    renderer.add_argument("--native", type=Path, required=True)
    renderer.add_argument("--monokernel", type=Path, required=True)
    renderer.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "run":
        assert arguments.repeats >= 2 and arguments.warmups >= 1
        assert all(length > 0 for length in arguments.input_lengths)
        assert all(length > 1 for length in arguments.output_lengths)
        assert len(set(arguments.input_lengths)) == len(arguments.input_lengths)
        assert len(set(arguments.output_lengths)) == len(arguments.output_lengths)
        asyncio.run(run(arguments))
    else:
        compare(arguments)


if __name__ == "__main__":
    main()
