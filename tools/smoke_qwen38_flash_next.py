#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request


def generate(base_url, prompt):
    payload = {
        "model": "Qwen3.8-Flash-Next",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token = None
    last_token = None
    content = []
    reasoning = []
    usage = None
    finish_reason = None
    received_done = False
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                received_done = True
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or ""
                thought = delta.get("reasoning_content") or ""
                if text or thought:
                    arrived = time.perf_counter()
                    if first_token is None:
                        first_token = arrived
                    last_token = arrived
                content.append(text)
                reasoning.append(thought)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    elapsed = time.perf_counter() - started
    if not received_done or finish_reason is None or not usage:
        raise RuntimeError("Incomplete stream or missing finish reason / usage")
    return {
        "prompt": prompt,
        "request": payload,
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "usage": usage,
        "finish_reason": finish_reason,
        "e2e_s": elapsed,
        "first_content_s": None if first_token is None else first_token - started,
        "content_stream_s": (
            None if first_token is None else last_token - first_token
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30160")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        ("arithmetic", "计算 7×8，只输出计算结果。", "56"),
        ("chinese", "用三句话介绍什么是张量并行。", None),
        ("english", "Explain why the sky appears blue in two sentences.", None),
        (
            "long_prefill",
            "资料中的唯一验证码是 FLASH936。以下是背景资料。\n"
            + "本项目在服务器上验证语言模型，记录输入和输出。\n" * 64
            + "请从资料中找出唯一的验证码，只输出验证码。",
            "FLASH936",
        ),
    ]
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "kind": "functional smoke test; timings include warmup and are not a benchmark",
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, prompt, expected in cases:
        result = generate(args.base_url.rstrip("/"), prompt)
        result["case"] = name
        result["expected_exact"] = expected
        result["passed"] = (
            result["content"].strip() == expected
            if expected is not None
            else bool(result["content"].strip())
        ) and result["finish_reason"] == "stop"
        report["results"].append(result)
        report["passed"] = all(item["passed"] for item in report["results"])
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "case",
                        "passed",
                        "content",
                        "usage",
                        "finish_reason",
                        "e2e_s",
                        "first_content_s",
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
