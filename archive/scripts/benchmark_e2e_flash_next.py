#!/usr/bin/env python3
"""
End-to-end / decode performance benchmark for Qwen3.8-Flash-Next on SGLang.

Tests various (input_len, output_len) combinations via the OpenAI-compatible
/v1/chat/completions endpoint and reports:
- TTFT (time to first token)
- TPOT (time per output token)
- end-to-end latency
- throughput (tokens/s)

Usage:
    python3 benchmark_e2e_flash_next.py [--url URL] [--model MODEL]
"""

import argparse
import json
import statistics
import time
from typing import Any

import requests


PROMPT_TEMPLATES = {
    128: "请用三句话介绍SGLang是什么。",
    512: "请详细解释深度学习中的注意力机制，包括它的数学形式、直观理解以及在Transformer模型中的作用。",
    1024: "请撰写一篇关于人工智能发展历史的短文，涵盖图灵测试、专家系统、机器学习、深度学习和大型语言模型五个阶段，每个阶段至少两句话。",
    2048: "请详细说明构建一个高性能大语言模型推理服务需要考虑哪些方面，包括模型并行、量化、KV Cache管理、调度策略、CUDA Graph、连续批处理、投机解码等，并对每个方面给出具体建议。",
    4096: "请从计算机体系结构、编译优化、分布式系统、机器学习算法和软件工程五个维度，系统性地分析如何优化一个大规模Transformer模型的推理延迟和吞吐。每个维度需要包含背景介绍、关键挑战、常用技术和未来方向四个小节。",
}


def build_prompt(target_len: int) -> str:
    """Return a prompt whose token count is approximately target_len."""
    if target_len in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[target_len]
    # Fallback: scale the nearest smaller template by repetition.
    keys = sorted(PROMPT_TEMPLATES.keys())
    base = max(k for k in keys if k <= target_len)
    template = PROMPT_TEMPLATES[base]
    repeat = max(1, target_len // base)
    return "\n".join([f"[{i+1}] {template}" for i in range(repeat)])


def chat_completion(
    url: str,
    model: str,
    prompt: str,
    max_completion_tokens: int,
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    headers = {"Content-Type": "application/json; charset=utf-8"}

    start = time.perf_counter()
    first_token_time: float | None = None
    token_times: list[float] = []
    content_chunks: list[str] = []

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            line = line.decode("utf-8", errors="ignore")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            tok_text = delta.get("content", "") or ""
            if tok_text:
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now)
                content_chunks.append(tok_text)

    end = time.perf_counter()
    e2e_latency = end - start
    ttft = (first_token_time - start) if first_token_time else e2e_latency
    num_tokens = len(token_times)
    tpot = (
        statistics.mean(token_times[i] - token_times[i - 1] for i in range(1, len(token_times)))
        if num_tokens > 1
        else 0.0
    )
    generated_text = "".join(content_chunks)
    return {
        "input_len_approx": target_len,
        "max_completion_tokens": max_completion_tokens,
        "e2e_latency_s": e2e_latency,
        "ttft_s": ttft,
        "num_output_tokens": num_tokens,
        "tpot_s": tpot,
        "decode_throughput_tok_s": num_tokens / e2e_latency if e2e_latency > 0 else 0.0,
        "generated_text_len_chars": len(generated_text),
    }


def run_benchmark(
    url: str,
    model: str,
    cases: list[tuple[int, int]],
    warmup: bool = True,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    if warmup:
        print("Warming up with a short request...")
        chat_completion(url, model, build_prompt(128), 32)
        time.sleep(1.0)

    results: list[dict[str, Any]] = []
    for input_len, output_len in cases:
        print(f"\nBenchmarking input_len≈{input_len}, output_len={output_len}")
        case_results = []
        for r in range(repeats):
            print(f"  repeat {r + 1}/{repeats} ...", end=" ", flush=True)
            res = chat_completion(url, model, build_prompt(input_len), output_len)
            case_results.append(res)
            print(
                f"TTFT={res['ttft_s']:.3f}s, "
                f"TPOT={res['tpot_s']:.3f}s, "
                f"E2E={res['e2e_latency_s']:.3f}s, "
                f"out_tokens={res['num_output_tokens']}"
            )
            time.sleep(0.5)

        # Aggregate
        agg = {
            "input_len_approx": input_len,
            "output_len_target": output_len,
            "repeats": repeats,
            "e2e_latency_s": {
                "mean": statistics.mean(x["e2e_latency_s"] for x in case_results),
                "min": min(x["e2e_latency_s"] for x in case_results),
                "max": max(x["e2e_latency_s"] for x in case_results),
            },
            "ttft_s": {
                "mean": statistics.mean(x["ttft_s"] for x in case_results),
                "min": min(x["ttft_s"] for x in case_results),
                "max": max(x["ttft_s"] for x in case_results),
            },
            "tpot_s": {
                "mean": statistics.mean(x["tpot_s"] for x in case_results),
                "min": min(x["tpot_s"] for x in case_results),
                "max": max(x["tpot_s"] for x in case_results),
            },
            "num_output_tokens": {
                "mean": statistics.mean(x["num_output_tokens"] for x in case_results),
                "min": min(x["num_output_tokens"] for x in case_results),
                "max": max(x["num_output_tokens"] for x in case_results),
            },
            "decode_throughput_tok_s": {
                "mean": statistics.mean(x["decode_throughput_tok_s"] for x in case_results),
                "min": min(x["decode_throughput_tok_s"] for x in case_results),
                "max": max(x["decode_throughput_tok_s"] for x in case_results),
            },
        }
        results.append(agg)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SGLang end-to-end/decode performance")
    parser.add_argument("--url", default="http://127.0.0.1:30000/v1/chat/completions")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output", default="benchmark_results/e2e_flash_next.json")
    args = parser.parse_args()

    # (input_len_approx, max_completion_tokens)
    cases = [
        (128, 128),
        (512, 128),
        (1024, 128),
        (2048, 128),
        (4096, 128),
        (128, 512),
        (512, 512),
        (1024, 512),
        (128, 1024),
        (512, 1024),
    ]

    results = run_benchmark(
        args.url,
        args.model,
        cases,
        warmup=not args.no_warmup,
        repeats=args.repeats,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(
        f"{'input':>8} {'output':>8} {'TTFT(s)':>10} {'TPOT(ms)':>10} "
        f"{'E2E(s)':>10} {'out_toks':>10} {'dec_tok/s':>12}"
    )
    for r in results:
        print(
            f"{r['input_len_approx']:>8} {r['output_len_target']:>8} "
            f"{r['ttft_s']['mean']:>10.3f} "
            f"{r['tpot_s']['mean'] * 1000:>10.2f} "
            f"{r['e2e_latency_s']['mean']:>10.3f} "
            f"{r['num_output_tokens']['mean']:>10.1f} "
            f"{r['decode_throughput_tok_s']['mean']:>12.2f}"
        )
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
