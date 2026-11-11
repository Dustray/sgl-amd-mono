"""Merge the per-rank stage-timing traces into one Perfetto file.

Each rank writes `<report-dir>/stage-timing-<label>-rank-<n>.json` when
`SGLANG_AMD_FLASH_STAGE_TIMING=1` and `SGLANG_AMD_FLASH_STAGE_TIMING_STEP=<step>`.
The files share no clock domain, so the merge keeps them side by side (one pid per rank) and
the per-stage duration summary compares ranks instead of adding their timestamps together.

    python3 tools/merge_flash_next_stage_timing.py --run-dir monokernel_reports/.../runtime
"""

import argparse
import json
from pathlib import Path
import statistics


def load(directory, pattern="stage-timing-*-rank-*.json"):
    traces = []
    for path in sorted(Path(directory).glob(pattern)):
        trace = json.loads(path.read_text())
        rank = int(path.stem.rsplit("-", 1)[1])
        traces.append((rank, path, [event for event in trace["traceEvents"] if event.get("ph") == "X"]))
    if not traces:
        raise FileNotFoundError(f"No stage timing traces in {directory}")
    return traces


def merge(traces, output):
    events = []
    for rank, _, spans in traces:
        events.append({"ph": "M", "name": "process_name", "pid": rank, "tid": 0, "args": {"name": f"rank {rank} (device stages)"}})
        events.append({"ph": "M", "name": "thread_name", "pid": rank, "tid": 1, "args": {"name": "decode steps"}})
        events.extend(spans)
    Path(output).write_text('{"displayTimeUnit":"ms","traceEvents":[' + ",".join(json.dumps(event, separators=(",", ":")) for event in events) + "]}")


def summarize(traces):
    """Per (rank, segment) duration statistics over every step the trace contains."""
    rows = []
    for rank, _, spans in traces:
        per_name = {}
        for event in spans:
            per_name.setdefault(event["name"], []).append(event["dur"])
        for name, durations in per_name.items():
            rows.append({"rank": rank, "segment": name, "count": len(durations),
                         "mean_us": statistics.mean(durations), "max_us": max(durations)})
    return sorted(rows, key=lambda row: (row["segment"], row["rank"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="directory with stage-timing-*-rank-*.json")
    parser.add_argument("--output", type=Path, help="merged Perfetto file (default: <run-dir>/stage-timing.perfetto.json)")
    parser.add_argument("--summary", type=Path, help="per-segment summary JSON (default: <run-dir>/stage-timing-summary.json)")
    arguments = parser.parse_args()
    directory = arguments.run_dir.resolve()
    traces = load(directory)
    output = arguments.output or directory / "stage-timing.perfetto.json"
    summary = arguments.summary or directory / "stage-timing-summary.json"
    merge(traces, output)
    rows = summarize(traces)
    summary.write_text(json.dumps({"ranks": [rank for rank, _, _ in traces], "segments": rows}, indent=2) + "\n")
    print(f"{len(traces)} ranks -> {output}")
    per_rank = {}
    for row in rows:
        per_rank.setdefault(row["rank"], 0.0)
        if row["segment"].startswith(("step ", "layer ")):
            continue
        per_rank[row["rank"]] += row["mean_us"]
    for rank in sorted(per_rank):
        print(f"rank {rank}: {per_rank[rank]:.1f} us summed over non-step spans")
    slowest = sorted((row for row in rows if not row["segment"].startswith(("step ", "layer "))), key=lambda row: -row["mean_us"])[:12]
    print("\nslowest segments (mean per occurrence, all ranks):")
    for row in slowest:
        print(f"  {row['segment']:<18} rank {row['rank']}  n={row['count']:<4} mean={row['mean_us']:8.1f} us  max={row['max_us']:8.1f} us")


if __name__ == "__main__":
    main()
