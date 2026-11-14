"""Merge the per-rank stage-timing traces into one Perfetto file.

Each rank writes `<report-dir>/stage-timing-<label>-rank-<n>.json` when
`SGLANG_AMD_FLASH_STAGE_TIMING=1` and `SGLANG_AMD_FLASH_STAGE_TIMING_STEP=<step>`, and
`cu-timing-<label>-rank-<n>.json` when `SGLANG_AMD_FLASH_CU_TIMING=1` adds one row per compute unit.
The files share no clock domain, so the merge keeps them side by side (one pid per rank) and
the per-stage duration summary compares ranks instead of adding their timestamps together.

    python3 tools/merge_flash_next_stage_timing.py --run-dir monokernel_reports/.../runtime
"""

import argparse
import gzip
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


def load_cu(directory, pattern="cu-timing-*-rank-*.json"):
    """Per-CU traces keep their own pids/tids (one thread per compute unit), so pass them through."""
    traces = []
    for path in sorted(Path(directory).glob(pattern)):
        trace = json.loads(path.read_text())
        rank = int(path.stem.rsplit("-", 1)[1])
        traces.append((rank, path, trace["traceEvents"]))
    return traces


def fill_cu_gaps(events):
    """Insert an explicit idle-gap span wherever consecutive spans on one CU row leave a hole.

    Works on already-exported per-CU traces (which record compute and barrier wait only), so
    launch gaps become visible without re-capturing: the hole between two spans is time the
    compute unit was neither computing nor waiting in a barrier."""
    tracks = {}
    passthrough = []
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in ("cu_compute", "cu_wait"):
            tracks.setdefault((event["pid"], event["tid"]), []).append(event)
        else:
            passthrough.append(event)
    filled = []
    for spans in tracks.values():
        spans.sort(key=lambda event: event["ts"])
        for earlier, later in zip(spans, spans[1:]):
            filled.append(earlier)
            hole = later["ts"] - (earlier["ts"] + earlier["dur"])
            if hole > 0.5:
                filled.append({"ph": "X", "cat": "cu_gap", "pid": later["pid"], "tid": later["tid"],
                               "ts": earlier["ts"] + earlier["dur"], "dur": hole,
                               "name": f"gap {earlier['args'].get('kernel', '?')} -> {later['args'].get('kernel', '?')}",
                               "args": {"tp_rank": later["args"].get("tp_rank"), "cu": later["args"].get("cu"),
                                        "step": later["args"].get("step"), "layer": later["args"].get("layer"),
                                        "from_kernel": earlier["args"].get("kernel"),
                                        "kernel": later["args"].get("kernel"), "phase": "gap"}})
        if spans:
            filled.append(spans[-1])
    return passthrough + filled


def merge(traces, output, cu_traces=()):
    events = []
    for rank, _, spans in traces:
        events.append({"ph": "M", "name": "process_name", "pid": rank, "tid": 0, "args": {"name": f"rank {rank} (device stages)"}})
        events.append({"ph": "M", "name": "thread_name", "pid": rank, "tid": 1, "args": {"name": "decode steps"}})
        events.extend(spans)
    for _, _, cu_events in cu_traces:
        events.extend(fill_cu_gaps(cu_events))
    payload = '{"displayTimeUnit":"ms","traceEvents":[' + ",".join(json.dumps(event, separators=(",", ":")) for event in events) + "]}"
    output = Path(output)
    if output.suffix == ".gz":
        with gzip.open(output, "wt") as handle:
            handle.write(payload)
    else:
        output.write_text(payload)


def summarize_cu(cu_traces):
    """Per (rank, CU) compute, barrier-wait and idle-gap totals, from the merged spans themselves."""
    rows = []
    for rank, _, events in cu_traces:
        per_cu = {}
        for event in events:
            if event.get("ph") != "X":
                continue
            cu = event["args"]["cu"]
            totals = per_cu.setdefault(cu, {"compute_us": 0.0, "wait_us": 0.0, "gap_us": 0.0, "spans": 0})
            totals["spans"] += 1
            key = {"cu_wait": "wait_us", "cu_gap": "gap_us"}.get(event["cat"], "compute_us")
            totals[key] += event["dur"]
        for cu, totals in per_cu.items():
            busy = totals["compute_us"] + totals["wait_us"] + totals["gap_us"]
            rows.append({"rank": rank, "cu": cu, **totals,
                         "compute_fraction": totals["compute_us"] / busy if busy else 0.0})
    return sorted(rows, key=lambda row: (row["rank"], row["cu"]))


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
    parser.add_argument("--output", type=Path, help="merged Perfetto file (default: <run-dir>/stage-timing.perfetto.json, or .json.gz when per-CU traces are present)")
    parser.add_argument("--summary", type=Path, help="per-segment summary JSON (default: <run-dir>/stage-timing-summary.json)")
    parser.add_argument("--no-cu", action="store_true", help="skip cu-timing-*-rank-*.json (per-CU rows) even when present")
    parser.add_argument("--gzip", action="store_true", help="force a .gz output even without per-CU traces")
    arguments = parser.parse_args()
    directory = arguments.run_dir.resolve()
    traces = load(directory)
    cu_traces = [] if arguments.no_cu else load_cu(directory)
    default_name = "stage-timing.perfetto.json.gz" if (cu_traces or arguments.gzip) else "stage-timing.perfetto.json"
    output = arguments.output or directory / default_name
    summary = arguments.summary or directory / "stage-timing-summary.json"
    merge(traces, output, cu_traces)
    rows = summarize(traces)
    cu_rows = summarize_cu(cu_traces)
    summary.write_text(json.dumps({"ranks": [rank for rank, _, _ in traces], "segments": rows,
                                   "compute_units": cu_rows}, indent=2) + "\n")
    print(f"{len(traces)} ranks -> {output}")
    if cu_rows:
        print(f"per-CU rows: {len(cu_rows)} over {len({row['rank'] for row in cu_rows})} ranks")
        total = {"compute_us": sum(row["compute_us"] for row in cu_rows),
                 "wait_us": sum(row["wait_us"] for row in cu_rows),
                 "gap_us": sum(row.get("gap_us", 0.0) for row in cu_rows)}
        busy = sum(total.values())
        print(f"  all CUs: compute={total['compute_us'] / 1e3:.1f} ms  wait={total['wait_us'] / 1e3:.1f} ms  "
              f"idle gap={total['gap_us'] / 1e3:.1f} ms" + (f"  (gap {total['gap_us'] / busy * 100:.1f}% of busy time)" if busy else ""))
        idle = sorted(cu_rows, key=lambda row: row["compute_fraction"])[:5]
        busy_cus = sorted(cu_rows, key=lambda row: -row["compute_fraction"])[:5]
        for label, selection in (("most barrier-bound CUs", idle), ("most compute-bound CUs", busy_cus)):
            print(f"  {label}:")
            for row in selection:
                print(f"    rank {row['rank']} {row['cu']}  compute={row['compute_us']:8.1f} us  wait={row['wait_us']:8.1f} us  gap={row.get('gap_us', 0.0):8.1f} us  compute={row['compute_fraction'] * 100:5.1f}%")
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
