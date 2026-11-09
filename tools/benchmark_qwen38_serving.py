import argparse
import asyncio
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import aiohttp


PROFILE_KEYS = (
    "model_path", "tp_size", "pp_size", "context_length", "mem_fraction_static",
    "max_running_requests", "disable_overlap_schedule", "attention_backend",
    "mm_attention_backend", "dtype", "kv_cache_dtype", "page_size",
    "chunked_prefill_size", "max_prefill_tokens", "mamba_radix_cache_strategy",
    "cuda_graph_backend_decode", "cuda_graph_backend_prefill", "stream_interval",
    "enable_metrics", "enable_cache_report", "disable_radix_cache",
)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


async def get_json(session, url, method="GET"):
    async with session.request(method, url) as response:
        response.raise_for_status()
        if response.content_type == "application/json":
            return await response.json()
        return await response.text()


def monokernel_state(info):
    return next((state["amd_dense_monokernel"] for state in info.get("internal_states", [])
                 if "amd_dense_monokernel" in state), None)


def counter_delta(before, after):
    previous = monokernel_state(before)
    current = monokernel_state(after)
    if current is None:
        assert previous is None
        return None
    forwards = current["tail_device_executions"] - previous["tail_device_executions"]
    launches = current["monokernel_executions"] - previous["monokernel_executions"]
    assert launches == forwards * 129
    assert all(current_count - old_count == forwards for current_count, old_count in
               zip(current["k2_device_executions_per_layer"], previous["k2_device_executions_per_layer"]))
    assert all(current_count - old_count == (0 if layer % 4 == 3 else forwards)
               for layer, (current_count, old_count) in enumerate(zip(
                   current["k1_device_executions_per_layer"], previous["k1_device_executions_per_layer"])))
    assert all(count - previous["attention_k1_device_executions_per_layer"][layer] == forwards
               for layer, count in current["attention_k1_device_executions_per_layer"].items())
    return {"decode_forwards": forwards, "cooperative_launches": launches, "all_layers_consistent": True}


def make_prompts(model_path, lengths):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    passage = (
        "This is a reproducible inference benchmark. Read the following technical notes "
        "and continue with a clear explanation. A distributed inference system handles "
        "requests, maintains a key value cache, and generates one token at a time. "
        "Matrix operations and memory transfers affect latency and throughput. "
        "本测试比较相同模型在不同执行路径下的推理时间，输入长度和输出长度保持一致。 "
    )
    prompts = {}
    for length in lengths:
        sequences = []
        for prefix in ("Alpha: ", "Beta: "):
            token_ids = tokenizer.encode(prefix + passage * (max(lengths) // 40 + 10), add_special_tokens=False)
            assert len(token_ids) >= length
            sequences.append(token_ids[:length])
        assert sequences[0][0] != sequences[1][0]
        prompts[str(length)] = sequences
    return prompts


async def generate(session, url, token_ids, output_tokens):
    payload = {"input_ids": token_ids, "stream": True, "sampling_params": {
        "temperature": 0, "max_new_tokens": output_tokens, "ignore_eos": True,
    }}
    started = time.perf_counter()
    arrivals = []
    final = None
    async with session.post(url + "/generate", json=payload) as response:
        response.raise_for_status()
        async for raw_line in response.content:
            received = time.perf_counter()
            if not raw_line.startswith(b"data: ") or raw_line.strip() == b"data: [DONE]":
                continue
            event = json.loads(raw_line[6:])
            if "error" in event:
                raise RuntimeError(event)
            token_count = len(event.get("output_ids", []))
            if token_count > len(arrivals):
                assert token_count == len(arrivals) + 1, "stream coalesced multiple tokens"
                arrivals.append(received - started)
            final = event
    elapsed = time.perf_counter() - started
    assert final is not None and len(arrivals) == output_tokens
    assert len(final["output_ids"]) == output_tokens
    metadata = final["meta_info"]
    assert metadata["prompt_tokens"] == len(token_ids)
    assert metadata["completion_tokens"] == output_tokens
    assert metadata["cached_tokens"] == 0, "cached prefix contaminates prefill timing"
    assert metadata.get("num_retractions", 0) == 0
    prefill = metadata["prefill_finished_time"] - metadata["forward_entry_time"]
    decode = arrivals[-1] - arrivals[0]
    assert 0 < prefill <= elapsed
    server_decode = (output_tokens - 1) / metadata["decode_throughput"] if output_tokens > 1 else 0
    return {
        "input_tokens": len(token_ids), "output_tokens": output_tokens,
        "e2e_s": elapsed, "prefill_s": prefill, "ttft_s": arrivals[0],
        "decode_s": decode, "tpot_s": decode / (output_tokens - 1) if output_tokens > 1 else 0,
        "server_e2e_s": metadata["e2e_latency"], "server_decode_s": server_decode,
        "token_arrivals_s": arrivals, "meta_info": metadata,
        "output_ids": final["output_ids"], "text": final.get("text"),
    }


async def wave(session, url, prompts, input_tokens, output_tokens, concurrency):
    await get_json(session, url + "/flush_cache?timeout=20", "POST")
    started = time.perf_counter()
    requests = await asyncio.gather(*(generate(session, url, prompts[str(input_tokens)][index], output_tokens)
                                      for index in range(concurrency)))
    elapsed = time.perf_counter() - started
    return {"wall_s": elapsed, "output_tokens_per_s": concurrency * output_tokens / elapsed,
            "requests": requests}


def summarize(cases):
    rows = []
    for case in cases:
        if not case.get("complete"):
            continue
        requests = [request for trial in case["trials"] for request in trial["requests"]]
        row = {key: case[key] for key in ("input_tokens", "output_tokens", "concurrency")}
        row.update(repeats=len(case["trials"]), requests=len(requests))
        for metric in ("e2e_s", "prefill_s", "ttft_s", "decode_s", "tpot_s", "server_e2e_s", "server_decode_s"):
            values = [request[metric] for request in requests]
            row[metric] = statistics.mean(values)
            row[metric + "_min"] = min(values)
            row[metric + "_max"] = max(values)
            row[metric + "_stdev"] = statistics.stdev(values) if len(values) > 1 else 0
        row["output_tokens_per_s"] = sum(request["output_tokens"] for request in requests) / sum(
            trial["wall_s"] for trial in case["trials"])
        row["input_tokens_per_prefill_s"] = case["input_tokens"] / row["prefill_s"]
        rows.append(row)
    return sorted(rows, key=lambda row: (row["concurrency"], row["input_tokens"], row["output_tokens"]))


async def run(arguments):
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(force_close=True)) as session:
        info = await get_json(session, arguments.url + "/server_info")
        state = monokernel_state(info)
        assert info["enable_metrics"] and info["stream_interval"] == 1
        assert info["max_running_requests"] == 2
        assert info["cuda_graph_backend_decode"] == "full"
        assert info["context_length"] == 4096 and info["disable_overlap_schedule"]
        if arguments.mode == "monokernel":
            from sgl_amd_mono.qwen38_27b.acceptance import source_identity

            assert state and state["complete_decode_model"] and state["kernel_launches_per_decode_forward"] == 129
            assert len(state["accepted_features"]) == 6
            assert state["implementation_sha256"] == source_identity()["implementation_sha256"]
            assert state["model_shadow_steps"] == 0
        else:
            assert state is None, "native baseline has monokernel enabled"
        prompts = make_prompts(info["model_path"], arguments.inputs)
        identity = hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
        report = {"schema_version": 1, "mode": arguments.mode, "passed": False,
                  "started_at": datetime.now(timezone.utc).isoformat(), "url": arguments.url,
                  "gpu": arguments.gpu, "server_info_before": info,
                  "profile": {key: info.get(key) for key in PROFILE_KEYS},
                  "prompt_sha256": identity, "prompts": prompts,
                  "measurement": {"repeats": arguments.repeats, "warmup_tokens": arguments.warmup_tokens,
                                  "seed": arguments.seed, "cache": "flush before every wave; assert cached_tokens=0",
                                  "e2e": "client HTTP send to stream EOF, perf_counter",
                                  "prefill": "scheduler prefill_finished_time minus forward_entry_time; wall clock, includes first token",
                                  "decode": "client last token arrival minus first token arrival, N-1 steps",
                                  "tpot": "decode/(output_tokens-1)",
                                  "aggregation": "mean across all measured requests; throughput from total tokens / sum wave durations"},
                  "warmups": [], "cases": []}
        save_json(arguments.output, report)
        try:
            for input_tokens in arguments.inputs:
                for concurrency in arguments.concurrencies:
                    result = await wave(session, arguments.url, prompts, input_tokens, arguments.warmup_tokens, concurrency)
                    report["warmups"].append({"input_tokens": input_tokens, "concurrency": concurrency, "result": result})
                    save_json(arguments.output, report)
                    print(json.dumps({"mode": arguments.mode, "warmup_input": input_tokens, "concurrency": concurrency,
                                      "wall_s": round(result["wall_s"], 3)}), flush=True)
            pairs = [(input_tokens, output_tokens, concurrency) for input_tokens in arguments.inputs
                     for output_tokens in arguments.outputs for concurrency in arguments.concurrencies]
            random.Random(arguments.seed).shuffle(pairs)
            for input_tokens, output_tokens, concurrency in pairs:
                before = await get_json(session, arguments.url + "/server_info")
                case = {"input_tokens": input_tokens, "output_tokens": output_tokens,
                        "concurrency": concurrency, "complete": False, "trials": []}
                report["cases"].append(case)
                for repeat in range(arguments.repeats):
                    result = await wave(session, arguments.url, prompts, input_tokens, output_tokens, concurrency)
                    case["trials"].append(result)
                    save_json(arguments.output, report)
                    print(json.dumps({"mode": arguments.mode, "input_tokens": input_tokens,
                                      "output_tokens": output_tokens, "concurrency": concurrency,
                                      "repeat": repeat + 1, "wall_s": round(result["wall_s"], 3)}), flush=True)
                after = await get_json(session, arguments.url + "/server_info")
                case["counter_delta"] = counter_delta(before, after)
                if state:
                    forwards = case["counter_delta"]["decode_forwards"]
                    minimum = arguments.repeats * (output_tokens - 1)
                    assert minimum <= forwards <= minimum * concurrency
                    if concurrency == 1:
                        assert forwards == minimum
                    else:
                        assert forwards < minimum * concurrency, "no actual batching in concurrency=2 case"
                case["complete"] = True
                report["summary"] = summarize(report["cases"])
                save_json(arguments.output, report)
            report["server_info_after"] = await get_json(session, arguments.url + "/server_info")
            report["passed"] = True
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            save_json(arguments.output, report)
        except Exception as error:
            report["error"] = repr(error)
            save_json(arguments.output, report)
            raise


def render(arguments):
    native = json.loads(arguments.native.read_text())
    mono = json.loads(arguments.monokernel.read_text())
    assert native["passed"] and mono["passed"]
    assert native["profile"] == mono["profile"], "server profiles differ"
    assert native["prompt_sha256"] == mono["prompt_sha256"] and native["measurement"] == mono["measurement"]
    assert native["gpu"] == mono["gpu"]
    native_rows = {(row["concurrency"], row["input_tokens"], row["output_tokens"]): row for row in native["summary"]}
    mono_rows = {(row["concurrency"], row["input_tokens"], row["output_tokens"]): row for row in mono["summary"]}
    assert native_rows.keys() == mono_rows.keys()
    combined = []
    for key, baseline in native_rows.items():
        fused = mono_rows[key]
        combined.append({"concurrency": key[0], "input_tokens": key[1], "output_tokens": key[2],
                         "native": baseline, "monokernel": fused,
                         "e2e_speedup": baseline["e2e_s"] / fused["e2e_s"],
                         "decode_speedup": baseline["decode_s"] / fused["decode_s"]})
    output_json = arguments.native.parent / "summary.json"
    output_csv = arguments.native.parent / "summary.csv"
    save_json(output_json, {"passed": True, "native": str(arguments.native), "monokernel": str(arguments.monokernel),
                            "profile": native["profile"], "rows": combined})
    with output_csv.open("w") as destination:
        fields = ["concurrency", "input_tokens", "output_tokens", "e2e_speedup", "decode_speedup"]
        fields += [mode + "_" + metric for mode in ("native", "monokernel") for metric in
                   ("e2e_s", "prefill_s", "ttft_s", "decode_s", "tpot_s", "output_tokens_per_s")]
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in combined:
            flat = {key: row[key] for key in fields[:5]}
            for mode in ("native", "monokernel"):
                flat.update({mode + "_" + metric: row[mode][metric] for metric in
                             ("e2e_s", "prefill_s", "ttft_s", "decode_s", "tpot_s", "output_tokens_per_s")})
            writer.writerow(flat)
    ratios = [row["e2e_speedup"] for row in combined]
    decode_ratios = [row["decode_speedup"] for row in combined]
    faster = sum(row["e2e_speedup"] > 1 for row in combined)
    conclusion = f"Monokernel 在 {len(combined)} 组中的 {faster} 组端到端更快。"
    if faster == 0:
        conclusion += f" 端到端耗时为原生的 {1/max(ratios):.2f}–{1/min(ratios):.2f} 倍。当前实现尚未获得加速，已完成的功能融合不能代替性能优化。"
    elif faster == len(combined):
        conclusion += " 本轮全部配置均获得端到端加速。"
    baseline_requests = {}
    for case in native["cases"]:
        for repeat, trial in enumerate(case["trials"]):
            for index, request in enumerate(trial["requests"]):
                key = (case["input_tokens"], case["output_tokens"], case["concurrency"], repeat, index)
                baseline_requests[key] = request["output_ids"]
    token_matches = []
    for case in mono["cases"]:
        for repeat, trial in enumerate(case["trials"]):
            for index, request in enumerate(trial["requests"]):
                key = (case["input_tokens"], case["output_tokens"], case["concurrency"], repeat, index)
                token_matches.append(request["output_ids"] == baseline_requests[key])
    summary = json.loads(output_json.read_text())
    summary["exact_output_matches"] = {"matched": sum(token_matches), "total": len(token_matches)}
    save_json(output_json, summary)
    prefill_differences = [(row["monokernel"]["prefill_s"] - row["native"]["prefill_s"]) * 1000 for row in combined]
    mono_throughput = {concurrency: [row["monokernel"]["output_tokens_per_s"] for row in combined
                                   if row["concurrency"] == concurrency] for concurrency in (1, 2)}
    batching_observation = ""
    if all(mono_throughput.values()):
        direction = "当前实现增加到双并发后总吞吐反而下降。" if max(mono_throughput[2]) < min(mono_throughput[1]) else "各配置的吞吐见下表。"
        batching_observation = f"本轮 Monokernel 单请求输出吞吐为 {min(mono_throughput[1]):.2f}–{max(mono_throughput[1]):.2f} token/s，双并发总吞吐为 {min(mono_throughput[2]):.2f}–{max(mono_throughput[2]):.2f} token/s；{direction}上述是阶段计时观察，尚未进行 kernel profiler 根因定位。"
    lines = ["# Qwen3.8-27B 原生 SGLang 与完整 decode Monokernel 性能实测", "",
             "测试日期：2026-11-06。以下均为本机真实请求计时，启动、模型加载、编译和预热不计入。", "",
             f"共 {len(combined)} 组输入/输出/并发配置，每种模式每组重复 {native['measurement']['repeats']} 次；表中时间为逐请求算术均值。",
             f"端到端加速比范围 {min(ratios):.3f}–{max(ratios):.3f}×，decode 加速比范围 {min(decode_ratios):.3f}–{max(decode_ratios):.3f}×。加速比=原生耗时/Monokernel 耗时，大于 1 才表示加速。", "",
             conclusion, "",
             f"Prefill 均值差（Monokernel−原生）为 {min(prefill_differences):.2f}–{max(prefill_differences):.2f} ms，主要耗时差距集中在 decode。",
             batching_observation, "",
             "## 环境与计时口径", "",
             f"- 两种模式依次使用同一张 GPU {native['gpu']}；同一模型、同一输入 token、同一请求顺序和预热策略。",
             "- gfx936/80 CU；SGLang 0.5.18 DTK；TP1、BF16 权重、FP8 E5M2 KV、FA3、page 64、context 4096、显存比例 0.90。",
             "- 两边均开启 decode Graph，关闭 overlap，最大并发 2；原生直接使用 sglang.launch_server，未启用适配入口。",
             "- 两边均开启相同的 metrics/cache report，以取得阶段时间；每次请求建立本机 HTTP 连接，开销同时计入两边端到端时间。",
             "- 每组输入长度/并发先生成 16 token 预热；每次正式测量前 flush_cache，并断言每个请求 cached_tokens=0。双并发的两条输入首 token 不同，防止相互命中前缀。",
             "- 直接提交固定长度 input_ids，避免客户端分词时间干扰；temperature=0、ignore_eos=true，严格生成指定 token 数。HTTP 使用本机 loopback，逐 token 流式返回。",
             "- 端到端：客户端发起 HTTP 至完整流结束；包含服务调度、prefill、decode、反分词与本机通信。",
             "- Prefill：服务端 scheduler 的 prefill_finished_time − forward_entry_time；是阶段 wall-clock，包含首 token 生成及相关处理，不是纯 GPU event 时间。",
             "- Decode：客户端最后一个 token 到达 − 第一个 token 到达；输出 N token 对应 N−1 次 decode。TPOT=Decode/(N−1)。这是请求级延迟，并发时可包含其他请求 prefill、调度和传输等待，不是纯 GPU kernel 时间。",
             "- Prefill 与 decode 来自不同计时边界，其和不严格等于端到端；TTFT 另存 CSV/JSON。并发 2 的时间为每个请求的均值，吞吐按整轮总输出 token/墙钟时间计算。",
             "- 三轮均值适合本次配置比较；原始数据保留范围与标准差，不将少量样本解读为生产环境 P95/P99。", "",
             "## 端到端、prefill 与 decode", ""]
    for concurrency in sorted({row["concurrency"] for row in combined}):
        lines.extend([f"### 并发 {concurrency}", "",
                      "| 输入 token | 输出 token | 原生端到端 s | Mono 端到端 s | 原生 Prefill ms | Mono Prefill ms | 原生 Decode s | Mono Decode s | 端到端加速比 | Decode 加速比 |",
                      "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in combined:
            if row["concurrency"] != concurrency:
                continue
            baseline, fused = row["native"], row["monokernel"]
            lines.append(f"| {row['input_tokens']} | {row['output_tokens']} | {baseline['e2e_s']:.4f} | {fused['e2e_s']:.4f} | {baseline['prefill_s']*1000:.2f} | {fused['prefill_s']*1000:.2f} | {baseline['decode_s']:.4f} | {fused['decode_s']:.4f} | {row['e2e_speedup']:.3f}× | {row['decode_speedup']:.3f}× |")
        lines.append("")
    lines.extend(["## Decode 每 token 延迟与输出吞吐", "",
                  "| 并发 | 输入 token | 输出 token | 原生 TPOT ms | Mono TPOT ms | 原生输出 token/s | Mono 输出 token/s |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in combined:
        baseline, fused = row["native"], row["monokernel"]
        lines.append(f"| {row['concurrency']} | {row['input_tokens']} | {row['output_tokens']} | {baseline['tpot_s']*1000:.2f} | {fused['tpot_s']*1000:.2f} | {baseline['output_tokens_per_s']:.2f} | {fused['output_tokens_per_s']:.2f} |")
    lines.extend(["", "## 执行路径与适用范围", "",
                  "Monokernel 在 required 模式下自动加载六项已验收能力，覆盖 embedding→64 层 decoder→final norm→完整 LM head，每次 decode forward 为 129 次 cooperative launch。逐组验证所有层计数一致；并发 2 要求实际发生 batch 合并。影子对照关闭。",
                  f"运行的实现 SHA256：`{monokernel_state(mono['server_info_before'])['implementation_sha256']}`。模型与兼容层版本、Torch/SGLang 版本和 Git HEAD 见原始目录的 `environment.json`，本次未提交 Git。",
                  f"性能请求均完成指定输出长度、无缓存命中、无请求回退。两种模式按相同请求位置比较，{sum(token_matches)}/{len(token_matches)} 条完整输出逐 token 一致；双并发的实际 prefill 合批时机会影响数值路径，该统计单独记录，不用来替代此前的同状态数值验收。",
                  "Prefill、采样和调度继续使用原生 SGLang，因此 Prefill 的差异主要反映两次运行的调度、预热和测量波动，不能据此声称存在 Prefill 融合加速。上述结果仅适用于本机固定配置，不能外推至其他架构、TP2、overlap 或更大 batch。", "",
                  "## 原始记录与复现", "",
                  f"- 原生逐请求记录、token 到达时间、服务配置和预热：`{arguments.native}`。",
                  f"- Monokernel 对应记录及逐层执行计数：`{arguments.monokernel}`。",
                  f"- 汇总数据：`{output_json}`、`{output_csv}`。",
                  "- 服务启动命令、环境、日志和 GPU 状态保存在原始记录所在目录。",
                  "- `initial-disconnected-native.*` 是测量器校准期间的失败尝试，发生在首轮预热前，正式样本数为零；保留追溯，不计入结果。",
                  "- 测量工具：`tools/benchmark_qwen38_serving.py`。", "",
                  "```bash",
                  "python3 tools/benchmark_qwen38_serving.py run --mode native --url http://127.0.0.1:30160 --gpu 4 --output benchmark_results/qwen38_27b_20261106/native.json",
                  "python3 tools/benchmark_qwen38_serving.py run --mode monokernel --url http://127.0.0.1:30161 --gpu 4 --output benchmark_results/qwen38_27b_20261106/monokernel.json",
                  "python3 tools/benchmark_qwen38_serving.py report --native benchmark_results/qwen38_27b_20261106/native.json --monokernel benchmark_results/qwen38_27b_20261106/monokernel.json --output 'Qwen3.8-27B 原生与Monokernel性能对比.md'",
                  "```", ""])
    arguments.output.write_text("\n".join(lines))
    print(json.dumps({"passed": True, "rows": len(combined), "e2e_speedup_range": [min(ratios), max(ratios)],
                      "decode_speedup_range": [min(decode_ratios), max(decode_ratios)]}))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("run")
    benchmark.add_argument("--url", required=True)
    benchmark.add_argument("--mode", choices=("native", "monokernel"), required=True)
    benchmark.add_argument("--gpu", type=int, required=True)
    benchmark.add_argument("--inputs", type=int, nargs="+", default=[64, 512, 2048])
    benchmark.add_argument("--outputs", type=int, nargs="+", default=[16, 64, 128])
    benchmark.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2])
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--warmup-tokens", type=int, default=16)
    benchmark.add_argument("--seed", type=int, default=20261106)
    benchmark.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("--native", type=Path, required=True)
    report.add_argument("--monokernel", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "run":
        assert arguments.repeats > 0 and min(arguments.inputs) > 0
        assert min(arguments.outputs) > 1 and set(arguments.concurrencies) <= {1, 2}
        asyncio.run(run(arguments))
    else:
        render(arguments)


if __name__ == "__main__":
    main()
