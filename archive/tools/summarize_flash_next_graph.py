import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def aggregate(folder):
    matrix = json.loads((folder / "matrix.json").read_text())
    assert matrix["passed"] and matrix["completed_cases"] == 20
    cases = defaultdict(list)
    for line in (folder / "samples.jsonl").read_text().splitlines():
        row = json.loads(line)
        arrivals = list(row["token_arrivals"].values())
        row["decode_s"] = max(tokens[-1][1] for tokens in arrivals) - min(tokens[0][1] for tokens in arrivals)
        cases[(row["batch_size"], row["input_len"], row["output_len"])].append(row)
    result = {}
    for key, rows in cases.items():
        assert len(rows) == 3
        result[key] = {metric: statistics.mean(row[metric] for row in rows) if all(row[metric] is not None for row in rows) else None for metric in ("e2e_s", "prefill_s", "ttft_s", "decode_s")}
        result[key]["decode_tps"] = key[0] * (key[2] - 1) / result[key]["decode_s"]
        result[key]["e2e_std_s"] = statistics.stdev(row["e2e_s"] for row in rows)
    assert len(result) == 20
    return matrix, result


def run(arguments):
    root = Path(__file__).resolve().parents[1]
    directory = arguments.run_dir.resolve()
    launch = json.loads((directory / "launch.json").read_text())
    baseline = Path(launch["baseline_run_dir"])
    folders = {"mono_eager": baseline / "monokernel", "mono_graph": directory / "monokernel_graph", "native_graph": baseline / "native_graph_detailed"}
    modes = {name: aggregate(folder) for name, folder in folders.items()}
    assert len({report["fixture_sha256"] for report, _ in modes.values()}) == 1
    assert len({json.dumps(report["versions"], sort_keys=True) for report, _ in modes.values()}) == 1
    ranks = [json.loads(path.read_text()) for path in sorted((directory / "runtime").glob("rank-*.json"))]
    assert len(ranks) == 8
    assert all(rank["graph"]["captured_batch_sizes"] == [1, 2] for rank in ranks)
    assert all({check["rows"] for check in rank["graph"]["graph_eager_checks"] if check["accepted"]} == {1, 2} for rank in ranks)
    failed = any(rank["acceptance_status"] == "serving_shadow_failed" for rank in ranks)
    lines = ["# Flash-Next TP8：Mono 开启 Graph 后的性能（2026-11-09）", "", "MTP 关闭；8 张 gfx936、BF16 权重、batch1/2，20组输入输出，每组预热1次、正式3次。固定随机 input_ids、温度0、ignore_eos，每轮清空缓存；与前一轮使用完全相同的输入。", ""]
    if failed:
        lines += ["**Graph 与 eager Mono 的 logits、缓存状态逐位验证通过；Mono 与原生的整模型精度影子检查仍有失败。本表是实验路径性能，Graph 通过不代表模型精度通过。**", ""]
    lines += ["## Graph 执行证据与范围", "", "- Decode Graph backend 为 full，仅捕获 batch1/2，禁用 padding；Prefill Graph 关闭，未使用 MTP。", "- 八个 rank 均通过 batch1/2 首次真实请求的 Graph/eager 同状态验证：完整 logits 零差异，KV、GDN、PLE、ngram 状态逐位相同。验证后恢复真实 Graph 结果和缓存，再继续生成。", "- 正式计时检查八卡99个融合数学核的设备执行计数与 decode 步数一致，并确认无额外影子 forward。Graph 计数只在真实回放时增加。", "- 修正 PLE 对保留 dummy 状态槽0的处理：dummy 输出保持输入且不写短卷积缓存；八卡独立测试覆盖 dummy 预热、捕获后动态切换真实状态槽和连续更新。", "", "## Mono Graph 开关对比", "", "两组都使用 BF16 KV、Triton Attention/MoE、context4096、chunked-prefill256、max-running2、Graph以外的 Overlap/Radix 均关闭，metrics开启。Graph组将 rank0–7 绑定 NUMA0–7，旧的无Graph组未指定NUMA，因此跨轮加速包含CPU绑定差异，不能全部归因于Graph。PLE新增的dummy分支不改变真实非零状态槽的数学计算。", "", "各成对列为 **关闭Graph / 开启Graph**；batch2的TPS为两请求合计。", "", "| batch | 输入→输出 | E2E / s | Prefill / ms | Decode / s | Decode TPS | E2E 加速 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    grouped = {name: values for name, (_, values) in modes.items()}
    for key in sorted(grouped["mono_graph"]):
        eager, graph = (grouped[name][key] for name in ("mono_eager", "mono_graph"))
        values = []
        for metric, scale in (("e2e_s", 1), ("prefill_s", 1000), ("decode_s", 1), ("decode_tps", 1)):
            values.append(" / ".join("未采集" if sample[metric] is None else f"{sample[metric] * scale:.3f}" for sample in (eager, graph)))
        lines.append(f"| {key[0]} | {key[1]}→{key[2]} | " + " | ".join(values) + f" | {eager['e2e_s'] / graph['e2e_s']:.3f}× |")
    lines += ["", "## 原生 Graph 与 Mono Graph", "", "此处原生为 FP8 KV / FA3 / AITER，context32768、chunked-prefill4096、max-running32、Overlap/Radix开启；与Mono的后端配置不同。均关闭问题融合TopK开关，均绑定NUMA0–7。", "", "成对列为 **原生Graph / Mono Graph**。", "", "| batch | 输入→输出 | E2E / s | Decode / s | Decode TPS |", "|---:|---:|---:|---:|---:|"]
    for key in sorted(grouped["mono_graph"]):
        native, graph = (grouped[name][key] for name in ("native_graph", "mono_graph"))
        values = [" / ".join(f"{sample[metric]:.3f}" for sample in (native, graph)) for metric in ("e2e_s", "decode_s", "decode_tps")]
        lines.append(f"| {key[0]} | {key[1]}→{key[2]} | " + " | ".join(values) + " |")
    lines += ["", "## 口径与复现", "", "E2E为请求发出到流结束；Prefill使用服务端阶段时间，包含首token forward；Decode为最晚token到达减最早请求首token到达，可能与另一请求的Prefill重叠。Decode TPS=batch×(output_len−1)/平均Decode秒。Graph核验和影子对照在预热期执行，正式区间不执行。三轮样本不能估计生产P95/P99。", "", "```bash", "bash launch_qwen_flash_next_monokernel_graph.sh --port 30000", "```", "", "性能运行时显式启用影子失败记录以保留既有精度问题，并继续测量实验路径；上述通用启动命令默认仍会拒绝未通过的精度检查。", "", f"本轮运行目录：`{directory}`；启动命令、环境和PID在 `launch.json`。原始样本在 `monokernel_graph/samples.jsonl`，八卡报告在 `runtime/`，独立PLE验证在 `ple_dummy_verification/`。旧数据目录：`{baseline}`。代码未做Git提交。", ""]
    output = root / "Flash-Next Mono Graph 性能对比.md"
    output.write_text("\n".join(lines))
    rows = [{"mode": name, "batch_size": key[0], "input_len": key[1], "output_len": key[2], **values} for name, cases in grouped.items() for key, values in sorted(cases.items())]
    with (directory / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if arguments.update_testresult:
        target = root / "testresult.md"
        target.write_text(output.read_text() + "\n---\n\n" + target.read_text())
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--update-testresult", action="store_true")
    run(parser.parse_args())
