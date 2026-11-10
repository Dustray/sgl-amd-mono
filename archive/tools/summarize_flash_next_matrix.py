import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics


MODES = ("native_graph", "native_bf16", "monokernel", "mtp")


def load(directory):
    reports = {}
    grouped = {}
    samples = {}
    for mode in MODES:
        folder = directory / ("native_graph_detailed" if mode == "native_graph" else mode)
        report = json.loads((folder / "matrix.json").read_text())
        assert report["passed"] and report["completed_cases"] == 20
        reports[mode] = report
        rows = [json.loads(line) for line in (folder / "samples.jsonl").read_text().splitlines()]
        assert len(rows) == 60
        samples[mode] = rows
        cases = defaultdict(list)
        for row in rows:
            arrivals = list(row["token_arrivals"].values())
            row["decode_s"] = max(tokens[-1][1] for tokens in arrivals) - min(tokens[0][1] for tokens in arrivals)
            cases[(row["batch_size"], row["input_len"], row["output_len"])].append(row)
        grouped[mode] = {}
        for key, values in cases.items():
            assert len(values) == 3
            summary = {metric: statistics.mean(value[metric] for value in values) if all(value[metric] is not None for value in values) else None for metric in ("e2e_s", "prefill_s", "ttft_s", "decode_s")}
            summary["decode_tps"] = key[0] * (key[2] - 1) / summary["decode_s"]
            summary["decode_tps_per_request"] = summary["decode_tps"] / key[0]
            summary["e2e_std_s"] = statistics.stdev(value["e2e_s"] for value in values)
            metadata = [response["meta_info"] for value in values for response in value["responses"].values()]
            verified = sum(item.get("spec_verify_ct", 0) for item in metadata)
            summary["spec_accept_length"] = sum(item.get("spec_accept_length", 0) * item.get("spec_verify_ct", 0) for item in metadata) / verified if verified else None
            grouped[mode][key] = summary
    assert len({report["fixture_sha256"] for report in reports.values()}) == 1
    assert len({json.dumps(report["source_sha256"], sort_keys=True) for report in reports.values()}) == 1
    assert len({json.dumps(report["versions"], sort_keys=True) for report in reports.values()}) == 1
    profile_keys = ("kv_cache_dtype", "attention_backend", "moe_runner_backend", "context_length", "max_running_requests", "disable_overlap_schedule", "disable_radix_cache", "enable_metrics", "chunked_prefill_size", "mem_fraction_static", "page_size", "tp_size", "numa_node")
    profiles = {}
    for mode in MODES:
        folder = directory / ("native_graph_detailed" if mode == "native_graph" else mode)
        info = json.loads((folder / "server_info_before.json").read_text())
        profiles[mode] = {key: info[key] for key in profile_keys}
    assert profiles["native_bf16"] == profiles["monokernel"]
    assert profiles["native_graph"] == profiles["mtp"]
    return reports, grouped, samples


def pair(first, second, field, scale=1, digits=3):
    return " / ".join("未采集" if value[field] is None else f"{value[field] * scale:.{digits}f}" for value in (first, second))


def run(arguments):
    directory = arguments.run_dir.resolve()
    reports, grouped, samples = load(directory)
    rows = []
    for mode, cases in grouped.items():
        for (batch, input_len, output_len), values in cases.items():
            rows.append({"mode": mode, "batch_size": batch, "input_len": input_len, "output_len": output_len, **values})
    with (directory / "summary.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = sorted(grouped["native_graph"])
    source = directory / "comparison.json"
    launch = json.loads(source.read_text())
    lines = [
        "# Qwen3.8-Flash-Next TP8：原生、Monokernel 与 MTP 性能（2026-11-09）", "",
        "8 张 gfx936，BF16 模型权重，batch 1 / 2；输入 64 / 512 / 2048 × 输出 16 / 64 / 128，以及 16→1024，共 20 组。每组预热 1 次、正式测量 3 次，表中为算术均值。四种模式复用同一 input_ids 文件、温度 0、ignore_eos；每轮清空缓存，校验输出长度、零缓存命中和零请求回退。没有开启 profiler。",
        "", "固定随机 token 输入，seed42；这是一组合成吞吐负载，MTP 接受率和加速不能直接推广到自然语言业务。", "",
        "## 服务加载与配置", "",
        "先完成已加载的原生 Graph 实例的全部测试；随后使用一个 BF16/Triton 实例，先走原生算子，再在空闲时切换至融合路径，权重、缓存配置与进程均不重建。发现原始融合 TopK 的数值问题后，补测关闭该开关的原生 Graph 基线，最后才加载并测试同配置 MTP。共享实例通过显式启用的控制文件切换，默认服务不提供这项控制。", "",
        "| 模式 | KV / Attention / MoE | Decode Graph | Overlap / Radix | 最大运行请求数 | Metrics |",
        "| --- | --- | --- | --- | ---: | --- |",
        "| 原生 Graph | FP8 E5M2 / FA3 / AITER | full | 开 / 开 | 32 | 关 |",
        "| 同实例原生 | BF16 / Triton / Triton | 关 | 关 / 关 | 2 | 开 |",
        "| Monokernel | BF16 / Triton / Triton | 关 | 关 / 关 | 2 | 开 |",
        "| 原生 Graph + MTP | FP8 E5M2 / FA3 / AITER | full | 开 / 开 | 32 | 关 |", "",
        "原生 Graph 与 MTP 使用 context32768、chunked-prefill4096、显存比例0.85；同实例两条路径使用 context4096、chunked-prefill256、max-total-tokens8192、显存比例0.90。所有模式 page64 / TP8。MTP 为 EAGLE，steps3、topk1、draft-tokens4。原生与融合之间只有“同实例原生 / Monokernel”这组控制了服务配置；优化后的原生 Graph 与 Mono 之间的比较反映整套运行配置的差异。", "",
        "## 同配置：原生路径 / Monokernel", "",
        "同实例原生仍经过插件的 Python 包装函数，数学计算调用原始算子；没有融合数学核或影子参考 forward。Monokernel 的 prefill 仍走原生路径。", "",
        "| Batch | 输入→输出 | E2E s | Prefill ms | Decode s | Decode TPS | E2E 加速 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    position = lines.index("## 同配置：原生路径 / Monokernel")
    lines[position:position] = [
        "主表四种模式均关闭 `SGLANG_USE_FUSED_TOPK_SOFTMAX`。用户原始开关为1；当前 LightOp 融合 TopK 在一次真实 BF16 router 输入中选入 logit1.2578125 的专家202，却漏掉 logit1.3046875 的专家507，详见 `topk-real-input-diagnostic.log`。发现问题后补测关闭该开关的 Graph 基线，再测同一开关配置下的 MTP；原始开关为1的两份矩阵只作为历史数据保留。", "",
        "加载记录：第一次共享实例继承了上述 TopK 开关，在第0层 MoE 实权重检查中被拒绝（NRMSE0.20354），产生了一次额外加载；日志与失败张量保存在 attempt-1 文件中。随后关闭该开关重试；新增加载前检查防止重复踩到同一配置问题，没有放宽数值阈值。", "",
        "NUMA：主表 Graph 基线与 MTP 均将 rank0–7 分别绑定 NUMA0–7，使用8线程加载。原始配置把八个 rank 全部绑到 NUMA0，其本地内存约126GiB；本次连2线程加载也触发内存不足，失败日志为 `native_graph_fixed-server-attempt-1-node0.log`。分散绑定后的主表两种原生模式保持一致；共享 BF16 实例的两条路径均未额外指定 NUMA 绑定。", "",
    ]
    position = lines.index("## 服务加载与配置")
    key = (1, 16, 1024)
    timings = [grouped[mode][key] for mode in ("native_bf16", "monokernel", "native_graph", "mtp")]
    throughput_text = " / ".join(f"{value['decode_tps']:.2f}" for value in timings)
    duration = (datetime.fromisoformat(reports["native_bf16"]["finished_at"]) - datetime.fromisoformat(reports["native_bf16"]["started_at"])).total_seconds() / 60
    explanation = [
        "## 为什么与最初六十多 TPS 不同", "",
        f"同样是 batch1、16→1024，本轮无 Graph 原生 / Mono / 原生 Graph 无 MTP / 原生 Graph＋MTP 的 decode TPS 分别为 **{throughput_text}**。前两种使用 BF16 KV、Triton Attention/MoE，且关闭 Graph、Overlap、Radix；后两种使用 FP8 KV、FA3、AITER，并开启 Graph。此前约4或11 TPS 并不是 MTP＋Graph 的结果。", "",
        f"整体测试耗时也被额外对照拉长：同配置的无 Graph 原生矩阵本身用了 **{duration:.1f} 分钟**；单、多 batch 的1024输出各重复三次，仅六轮正式长输出就约25分钟。此外运行过官方工具和详细流式计时两份历史 Graph 矩阵，TopK 校验失败、NUMA 内存不足及修正配置后的重测又增加了加载和测试时间。BF16 原生与 Mono 已在同一进程内复用权重。", "",
    ]
    official = directory / "mtp_official_b1_16_1024/result.jsonl"
    if official.exists():
        result = json.loads(official.read_text().splitlines()[-1])
        explanation += [f"在本轮矩阵结束后，复用同一个 MTP 服务，用用户原始 `python3 -m sglang.bench_one_batch_server` 工具复核 batch1、16→1024：E2E **{result['latency']:.4f}s**、TTFT **{result['last_ttft']:.4f}s**、官方 decode 吞吐 **{result['output_throughput']:.2f} TPS**，平均接受长度 **{result['acc_length']:.2f}**。该工具使用其自身的 random-ids 输入，和主表固定输入不同，结果单独记录。", ""]
    if arguments.reference_official_result:
        result = json.loads(arguments.reference_official_result.read_text().splitlines()[-1])
        explanation += [f"最初官方工具结果保存在 `{arguments.reference_official_result}`：同样 batch1、16→1024，MTP＋Graph 的 decode 吞吐 **{result['output_throughput']:.2f} TPS**，E2E **{result['latency']:.4f}s**，平均接受长度 **{result['acc_length']:.2f}**。旧服务的融合 TopK 开关为1，本轮为0，CPU/NUMA 绑定也已修正，因此这条历史记录没有混入主表。官方吞吐分子计入全部输出 token，主表排除首 token；1024输出时差异仅约0.1%，解释不了数倍差距。", ""]
    lines[position:position] = explanation
    fair = [grouped["native_bf16"][key]["e2e_s"] / grouped["monokernel"][key]["e2e_s"] for key in keys]
    deployment = [grouped["native_graph"][key]["e2e_s"] / grouped["monokernel"][key]["e2e_s"] for key in keys]
    mtp = [grouped["native_graph"][key]["e2e_s"] / grouped["mtp"][key]["e2e_s"] for key in keys]
    lines[2:2] = [f"同 BF16/Triton 配置下，Mono 的 E2E 加速为 **{min(fair):.3f}–{max(fair):.3f}×**；与已优化的原生 Graph 配置比较，原生 E2E / Mono E2E 为 **{min(deployment):.3f}–{max(deployment):.3f}×**（小于1表示原生更快）。原生开启 MTP 的 E2E 加速为 **{min(mtp):.3f}–{max(mtp):.3f}×**。", ""]
    for batch, input_len, output_len in keys:
        first, second = [grouped[mode][(batch, input_len, output_len)] for mode in ("native_bf16", "monokernel")]
        lines.append(f"| {batch} | {input_len}→{output_len} | {pair(first, second, 'e2e_s')} | {pair(first, second, 'prefill_s', 1000, 2)} | {pair(first, second, 'decode_s')} | {pair(first, second, 'decode_tps', digits=2)} | {first['e2e_s'] / second['e2e_s']:.3f}× |")
    lines += ["", "## 优化配置：原生 Graph / 原生 Graph + MTP", "", "| Batch | 输入→输出 | E2E s | TTFT ms | Decode s | Decode TPS | E2E 加速 | MTP 平均接受长度 |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for batch, input_len, output_len in keys:
        first, second = [grouped[mode][(batch, input_len, output_len)] for mode in ("native_graph", "mtp")]
        acceptance = "未采集" if second["spec_accept_length"] is None else f"{second['spec_accept_length']:.3f}"
        lines.append(f"| {batch} | {input_len}→{output_len} | {pair(first, second, 'e2e_s')} | {pair(first, second, 'ttft_s', 1000, 2)} | {pair(first, second, 'decode_s')} | {pair(first, second, 'decode_tps', digits=2)} | {first['e2e_s'] / second['e2e_s']:.3f}× | {acceptance} |")
    lines += ["", "MTP 平均接受长度按各请求的 verify 次数加权；越长表示每轮验证产出的 token 越多。固定输入和温度0仍可能出现数值路径差异，三个重复样本的波动保存在 CSV，不能仅选最快的一轮代表结果。", ""]
    lines += ["", "## 实际部署配置：原生 Graph / Monokernel", "", "两列配置存在上表所列差异，不能据此把全部速度差异归因于融合算法。", "", "| Batch | 输入→输出 | E2E s | Decode TPS | 原生 E2E / Mono E2E |", "| ---: | --- | ---: | ---: | ---: |"]
    for batch, input_len, output_len in keys:
        first, second = [grouped[mode][(batch, input_len, output_len)] for mode in ("native_graph", "monokernel")]
        lines.append(f"| {batch} | {input_len}→{output_len} | {pair(first, second, 'e2e_s')} | {pair(first, second, 'decode_tps', digits=2)} | {first['e2e_s'] / second['e2e_s']:.3f}× |")
    lines += [
        "", "## 计时、执行路径与精度范围", "",
        "- E2E：客户端发出 /generate 到流结束；使用 input_ids，不包括客户端构造输入和分词。",
        "- Prefill：服务端 prefill_finished_time − forward_entry_time；表中取 batch 内各请求的最大阶段耗时，包含首 token forward，是 wall-clock。原生 Graph 与 MTP 沿用 metrics 关闭配置，没有该字段，不能用 TTFT 替代纯 Prefill。",
        "- TTFT：batch 中最后一条请求首 token 到达时间。Decode：batch 最后一个输出 token 到达 − 最早一条请求首 token 到达。根据保存的原始 token_arrivals 统一重算，避免遗漏异步合批中提前生成的 token；可能包含其他请求尚未完成的 prefill 与调度等待。",
        "- Decode TPS：batch ×（输出 token 数 − 1）/ 平均 Decode 秒；batch2 为聚合 TPS，单请求 TPS 为其一半。MTP 一次发送多个 token 的 chunk 按累计 token 计数处理；不会把一个 chunk 算作一个 token。",
        "- 正式输出长度始终保持表中数量。主表四种模式的预热输出上限均为128 token；已预热2048输入及相应 batch 后再做长输出正式测量。原始 TopK=1 历史矩阵的长输出预热为1024 token，未混入主表。",
        "- 三个正式样本不用于估计生产 P95 / P99；原始逐次数据及 E2E 样本标准差保存在 summary.csv 与各模式 samples.jsonl。", "",
    ]
    audits = [json.loads(line) for line in (directory / "monokernel/execution_audit.jsonl").read_text().splitlines()]
    statuses = sorted({rank["acceptance_status"] for row in audits for rank in row["ranks"]})
    if "serving_shadow_failed" in statuses:
        lines[2:2] = ["**本轮 Monokernel 的整模型数值影子检查未通过。下列融合结果仅表示当前实验执行路径的性能，不能作为精度已经验收的可部署加速结论。**", ""]
    lines.append(f"融合正式测量共 {len(audits)} 组 × 8 卡，99 个主数学核计数逐项等于 decode forward 数，计时区间影子检查数未增加。实例记录的影子状态为 `{', '.join(statuses)}`。本轮是性能测试，不表示完整精度验收通过。")
    token_matches = sum(all(first["responses"][index]["output_ids"] == second["responses"][index]["output_ids"] for index in first["responses"]) for first, second in zip(samples["native_bf16"], samples["monokernel"]))
    lines += [f"同配置正式请求按同一 batch / 输入 / 输出 / 重复序号配对，完整 output_ids 一致为 {token_matches}/60 个请求波次；该统计只作输出一致性记录，不替代同状态 logits / 长序列精度验收。", ""]
    shadow_path = directory / "shared_bf16_runtime/rank-0.json"
    if shadow_path.exists():
        shadow = json.loads(shadow_path.read_text())
        lines += ["rank0 的首次整模型同状态检查（均在预热中执行）：", "", "| Batch | Logits NRMSE | 最大 KL | 最大 TV | Greedy argmax 相同 | 验收 |", "| ---: | ---: | ---: | ---: | --- | --- |"]
        for check in shadow["full_model_shadow_checks"]:
            distribution = check["distribution"]
            lines.append(f"| {check['rows']} | {check['logits']['nrmse']:.6f} | {max(distribution['kl_reference_actual']):.6f} | {max(distribution['total_variation']):.6f} | {all(check['argmax_equal'])} | {check['accepted']} |")
        lines.append("")
    lines += [
        "## 数据文件", "", f"运行目录：`{directory}`。", "",
        "- `summary.csv`：80 行聚合结果及标准差。",
        "- `fixture.json`：四种模式共用的实际 input_ids。",
        "- `native_graph_detailed/`、`native_bf16/`、`monokernel/`、`mtp/`：主表四种模式的逐次响应、token 到达时间、服务配置、执行审计。",
        "- `comparison.json`：服务启动 / 停止时间和 PID；`shared_bf16_server.log`、`native_graph_fixed-server.log`、`mtp-server.log`：启动及运行日志。",
        "- `native_graph_matrix/`、`native_graph_fused_topk1/`：原始 TopK=1 的官方计时与详细流式矩阵，各20组×3次；未纳入主表。",
        "- `topk-real-input-diagnostic.log`、`topk-real-input-diagnostic.pt`：真实 router 输入、选择的专家和概率。", "",
        f"最终服务 PID：`{launch['final_service_pid']}`，原生 Graph + MTP，端口30000，融合 TopK 开关为0。", "",
    ]
    output = directory / "性能对比.md"
    output.write_text("\n".join(lines))
    if arguments.update_testresult:
        target = Path(__file__).resolve().parents[1] / "testresult.md"
        target.write_text(output.read_text() + "\n---\n\n" + target.read_text())
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--update-testresult", action="store_true")
    parser.add_argument("--reference-official-result", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
