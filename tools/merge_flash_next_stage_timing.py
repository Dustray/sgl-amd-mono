"""Merge the per-rank stage-timing traces into one Perfetto file.

Each rank writes `<report-dir>/stage-timing-<label>-rank-<n>.json` when
`SGLANG_AMD_FLASH_STAGE_TIMING=1` and `SGLANG_AMD_FLASH_STAGE_TIMING_STEP=<step>`, and
`cu-timing-<label>-rank-<n>.json` when `SGLANG_AMD_FLASH_CU_TIMING=1` adds one row per compute unit.
Every GPU stamps marks with its own `wall_clock64()`, so the raw files share no clock domain. The
merge maps every rank onto rank 0's clock using the cross-rank barriers the kernels already run:
the release of a TP all-reduce's final `peer wait` (attn 14, moe 54, embed 514, entry tp 614,
tail gather 634) happens only after the slowest rank has signalled, so all ranks leave it at the
same instant up to one xGMI flag latency (a few us). Matching those releases per launch gives
~100 pairs per decode step; a robust least-squares fit `t_rank0 = a * t_rank + b` per rank (offset
plus clock drift) is applied to that rank's stage rows and per-CU rows alike. The residuals are
printed and saved (`alignment` in the summary): they are the precision of the alignment, so do not
read cross-rank differences smaller than them. `--align offset` fits the offset only, `--align none`
keeps the old unaligned layout.

    python3 tools/merge_flash_next_stage_timing.py --run-dir monokernel_reports/.../runtime
"""

import argparse
import gzip
import json
from pathlib import Path
import statistics

from sgl_amd_mono.common.profile_trace import widen_segments


def load(directory, pattern="stage-timing-*-rank-*.json", progress=None):
    traces = []
    paths = sorted(Path(directory).glob(pattern))
    for index, path in enumerate(paths, 1):
        if progress is not None:
            progress(f"stage-timing: reading {path.name} ({index}/{len(paths)})")
        trace = json.loads(path.read_text())
        rank = int(path.stem.rsplit("-", 1)[1])
        traces.append((rank, path, widen_segments([event for event in trace["traceEvents"] if event.get("ph") == "X"])))
    if not traces:
        raise FileNotFoundError(f"No stage timing traces in {directory}")
    return traces


def load_cu(directory, pattern="cu-timing-*-rank-*.json", steps=None, progress=None):
    """Per-CU traces keep their own pids/tids (one thread per compute unit), so pass them through.

    `steps` (a set of 1-based step numbers, `args.step` of each span) keeps only those decode steps;
    a 32-step capture is ~2.1 GB JSON / ~8M events per rank, too much to merge or open whole.
    """
    traces = []
    paths = sorted(Path(directory).glob(pattern))
    for index, path in enumerate(paths, 1):
        if progress is not None:
            size = path.stat().st_size / 2**30
            progress(f"cu-timing: reading {path.name} ({index}/{len(paths)}, {size:.2f} GiB)")
        trace = json.loads(path.read_text())
        rank = int(path.stem.rsplit("-", 1)[1])
        events = trace["traceEvents"]
        del trace
        if steps is not None:
            events = [event for event in events if event.get("ph") == "M" or event.get("args", {}).get("step") in steps]
        traces.append((rank, path, events))
    return traces


def parse_steps(text):
    """'5' or '3-6' or '1,4-5' -> set of ints."""
    steps = set()
    for part in text.split(","):
        first, _, last = part.partition("-")
        steps.update(range(int(first), int(last or first) + 1))
    return steps


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


GRID_TID = 2


def grid_spans(events, pid):
    """One span per (launch, segment) covering every block: earliest start -> latest barrier arrival.

    The block-0 stage row only shows block 0's own compute, which ends when block 0 reaches the
    barrier even if other blocks are still working (e.g. layer 0 gdn mix.down, 2026-11-15: block 0
    arrived at 9681 us, block 3 at 9713 us). This row is what "first CU start to last CU end" means.
    Occurrences are numbered per block in time order, so a segment repeated within one
    (step, layer, kernel) still pairs up across blocks."""
    per_block = {}
    for event in events:
        if event.get("ph") == "X" and event.get("cat") == "cu_compute":
            per_block.setdefault(event["args"].get("block"), []).append(event)
    groups = {}
    for spans in per_block.values():
        spans.sort(key=lambda event: event["ts"])
        seen = {}
        for event in spans:
            args = event["args"]
            key = (args.get("step"), args.get("layer"), args.get("kernel"), args.get("stage"))
            occurrence = seen.get(key, 0)
            seen[key] = occurrence + 1
            groups.setdefault((*key, occurrence), []).append(event)
    out = []
    for (step, layer, kernel, stage, _), spans in groups.items():
        start = min(event["ts"] for event in spans)
        last = max(spans, key=lambda event: event["ts"] + event["dur"])
        end = last["ts"] + last["dur"]
        arrivals = sorted(event["ts"] + event["dur"] for event in spans)
        out.append({"ph": "X", "cat": "grid_stage", "name": spans[0]["name"], "pid": pid, "tid": GRID_TID,
                    "ts": start, "dur": end - start,
                    "args": {"tp_rank": spans[0]["args"].get("tp_rank"), "step": step, "layer": layer, "kernel": kernel,
                             "stage": stage, "blocks": len(spans), "last_block": last["args"].get("block"),
                             "last_cu": last["args"].get("cu"), "median_arrival_us": arrivals[len(arrivals) // 2] - start}})
    out.sort(key=lambda event: event["ts"])
    return out


# Final `peer wait` of each cross-rank all-reduce/gather: released only once every rank has signalled.
SYNC_SEGMENTS = ("tp.peer wait", "attn.peer wait", "moe.peer wait", "embed.peer wait", "gather.peer wait")
SYNC_STAGES = (614, 14, 54, 514, 634)
STAGE_THREAD = "decode steps (stage = all blocks, release to release)"
WAIT_SUFFIX = " / block 0 grid wait"
LAUNCH_DEPTH = 2


def sync_points(spans):
    """{(launch label, occurrence, segment): release_us} from one rank's block-0 stage spans.

    Keys are per launch rather than a running count of the segment, so a zero-length (dropped)
    wait span on one rank cannot shift every later pair. Within a launch the release is the end of
    the block-0 grid wait, or the barrier arrival when that wait span was too short to export."""
    occurrences = {}
    launch = None
    points = {}
    for event in spans:  # export order: launch span, then its segments
        if event.get("args", {}).get("depth") == LAUNCH_DEPTH:
            occurrence = occurrences.get(event["name"], 0)
            occurrences[event["name"]] = occurrence + 1
            launch = (event["name"], occurrence)
            continue
        if launch is None:
            continue
        name = event["name"][:-len(WAIT_SUFFIX)] if event["name"].endswith(WAIT_SUFFIX) else event["name"]
        if name not in SYNC_SEGMENTS:
            continue
        key = (*launch, name)
        points[key] = max(points.get(key, float("-inf")), event["ts"] + event["dur"])
    return points


def fit_clock(pairs, mode="fit"):
    """Robust fit of reference = a * local + b over (local_us, reference_us) pairs.

    Two rounds of rejecting pairs whose residual exceeds max(1 us, 5 x MAD), so a barrier that one
    rank left late (e.g. a preempted wave) does not drag the whole rank."""
    kept = list(pairs)
    a, b = 1.0, 0.0
    for _ in range(3):
        if mode == "fit" and len(kept) >= 8 and max(x for x, _ in kept) - min(x for x, _ in kept) > 0:
            mean_x = statistics.fmean(x for x, _ in kept)
            mean_y = statistics.fmean(y for _, y in kept)
            var = sum((x - mean_x) ** 2 for x, _ in kept)
            a = sum((x - mean_x) * (y - mean_y) for x, y in kept) / var
            b = mean_y - a * mean_x
        else:
            a, b = 1.0, statistics.median(y - x for x, y in kept)
        residuals = [y - (a * x + b) for x, y in pairs]
        mad = statistics.median(abs(r) for r in residuals)
        limit = max(1.0, 5.0 * mad)
        selected = [pair for pair, r in zip(pairs, residuals) if abs(r) <= limit]
        if len(selected) == len(kept) or len(selected) < 2:
            break
        kept = selected
    residuals = [abs(y - (a * x + b)) for x, y in kept]
    return {"scale": a, "offset_us": b, "drift_ppm": (a - 1.0) * 1e6, "pairs": len(pairs), "used": len(kept),
            "residual_median_us": statistics.median(residuals) if residuals else None,
            "residual_max_us": max(residuals) if residuals else None}


def align(traces, mode="fit"):
    """{rank: fit} mapping each rank's timestamps onto the lowest rank's clock."""
    if mode == "none" or not traces:
        return {}
    points = {rank: sync_points(spans) for rank, _, spans in traces}
    reference = min(points)
    fits = {reference: {"scale": 1.0, "offset_us": 0.0, "drift_ppm": 0.0, "pairs": len(points[reference]),
                        "used": len(points[reference]), "residual_median_us": 0.0, "residual_max_us": 0.0, "reference": True}}
    for rank, local in points.items():
        if rank == reference:
            continue
        common = sorted(set(local) & set(points[reference]), key=lambda key: points[reference][key])
        if len(common) < 2:
            raise ValueError(f"rank {rank}: only {len(common)} cross-rank barriers match rank {reference}; "
                             "were all ranks captured at the same decode step? (use --align none to skip)")
        if len(common) < 0.9 * max(len(local), len(points[reference])):
            print(f"warning: rank {rank} matches only {len(common)} of {len(local)}/{len(points[reference])} barriers")
        fits[rank] = fit_clock([(local[key], points[reference][key]) for key in common], mode)
    return fits


def _apply(events, fit, shift):
    scale, offset = fit["scale"], fit["offset_us"] - shift
    out = []
    for event in events:
        if "ts" in event and event.get("ph") != "M":
            event = dict(event, ts=scale * event["ts"] + offset)
            if "dur" in event:
                event["dur"] = scale * event["dur"]
        out.append(event)
    return out


def check_cu_alignment(cu_traces, fits, shift=0.0):
    """After alignment, per rank: median/max |release - reference release| of the per-CU peer waits.

    Independent of the stage-based fit (every block's own marks, not block 0's), so it confirms that
    the per-CU rows really line up across ranks."""
    releases = {}
    for rank, _, events in cu_traces:
        fit = fits.get(rank)
        if fit is None:
            return {}
        per_key = {}
        for event in events:
            args = event.get("args", {})
            if event.get("cat") != "cu_wait" or args.get("stage") not in SYNC_STAGES:
                continue
            end = fit["scale"] * (event["ts"] + event["dur"]) + fit["offset_us"] - shift
            per_key.setdefault((args.get("step"), args.get("layer"), args.get("kernel"), args.get("stage")), []).append(end)
        releases[rank] = {key: statistics.median(values) for key, values in per_key.items()}
    if not releases:
        return {}
    reference = min(releases)
    report = {}
    for rank, values in releases.items():
        deltas = [abs(values[key] - releases[reference][key]) for key in values.keys() & releases[reference].keys()]
        if deltas:
            report[rank] = {"barriers": len(deltas), "median_us": statistics.median(deltas), "max_us": max(deltas)}
    return report


def is_idle(event):
    """Barrier waits and launch gaps: block-0 grid waits and per-CU `cu_wait` / `cu_gap` spans."""
    return event.get("cat") in ("cu_wait", "cu_gap") or event.get("name", "").endswith(WAIT_SUFFIX)


def merge(traces, output, cu_traces=(), align_mode="fit", progress=None, keep_idle=False, grid=True):
    """Write the merged Perfetto file; returns the alignment report ({} with align_mode="none").

    By default the exported trace holds compute spans only: waits and gaps are left out (empty
    time on a row is then wait or idle). Alignment, the per-CU check and the summaries still use
    them from the inputs; `keep_idle=True` exports them too. With per-CU traces, `grid=True` adds a
    `grid (all CUs)` track under each rank's stage process (see `grid_spans`)."""

    def report(message):
        if progress is not None:
            progress(message)

    report(f"aligning {len(traces)} rank stage traces on cross-rank barriers")
    fits = align(traces, align_mode)
    shift = 0.0
    if fits:
        starts = [fits[rank]["scale"] * event["ts"] + fits[rank]["offset_us"]
                  for rank, _, spans in traces for event in spans]
        starts += [fits[rank]["scale"] * event["ts"] + fits[rank]["offset_us"]
                   for rank, _, events in cu_traces if rank in fits
                   for event in events if event.get("ph") == "X"]
        shift = min(starts, default=0.0)
    events = []
    for rank, _, spans in traces:
        events.append({"ph": "M", "name": "process_name", "pid": rank, "tid": 0, "args": {"name": f"rank {rank} (device stages)"}})
        events.append({"ph": "M", "name": "thread_name", "pid": rank, "tid": 1, "args": {"name": STAGE_THREAD}})
        spans = spans if keep_idle else [event for event in spans if not is_idle(event)]
        events.extend(_apply(spans, fits[rank], shift) if rank in fits else spans)
    for rank, _, cu_events in cu_traces:
        report(f"applying clock alignment to rank {rank} per-CU events")
        cu_events = cu_events if keep_idle else [event for event in cu_events if not is_idle(event)]
        cu_events = _apply(cu_events, fits[rank], shift) if rank in fits else cu_events
        if grid:
            events.append({"ph": "M", "name": "thread_name", "pid": rank, "tid": GRID_TID,
                           "args": {"name": "grid (all CUs): first block start -> last block arrival"}})
            events.extend(grid_spans(cu_events, rank))
        events.extend(fill_cu_gaps(cu_events) if keep_idle else cu_events)
    # Stream, and gzip at level 1: joining ~65M events (32 CU steps x 8 ranks) into one string and
    # compressing it at the default level 9 ran for well over 10 minutes with 0 bytes written (2026-11-14).
    output = Path(output)
    handle = gzip.open(output, "wt", compresslevel=1) if output.suffix == ".gz" else output.open("w")
    with handle:
        report(f"writing {len(events)} events to {output.name}")
        handle.write('{"displayTimeUnit":"ms","traceEvents":[')
        for index, event in enumerate(events):
            if index:
                handle.write(",")
            handle.write(json.dumps(event, separators=(",", ":")))
        handle.write("]}")
    if not fits:
        return {}
    return {"mode": align_mode, "reference_rank": min(fits), "shift_us": shift,
            "ranks": {str(rank): fit for rank, fit in sorted(fits.items())},
            "cu_check": {str(rank): row for rank, row in sorted(check_cu_alignment(cu_traces, fits, shift).items())}}


def print_alignment(report):
    if not report:
        print("timestamps NOT aligned across ranks (--align none): compare durations only")
        return
    print(f"aligned to rank {report['reference_rank']}'s clock on cross-rank peer-wait releases ({report['mode']}):")
    for rank, fit in report["ranks"].items():
        if fit.get("reference"):
            continue
        cu = report["cu_check"].get(rank)
        cu_text = f"  per-CU check median={cu['median_us']:.2f} us max={cu['max_us']:.2f} us" if cu else ""
        print(f"  rank {rank}: offset={fit['offset_us']:+.1f} us drift={fit['drift_ppm']:+.1f} ppm  "
              f"pairs={fit['used']}/{fit['pairs']}  residual median={fit['residual_median_us']:.2f} us "
              f"max={fit['residual_max_us']:.2f} us{cu_text}")


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
    parser.add_argument("--steps", type=parse_steps, help="per-CU decode steps to keep, 1-based within the capture (e.g. 5 or 3-6); default all")
    parser.add_argument("--gzip", action="store_true", help="force a .gz output even without per-CU traces")
    parser.add_argument("--align", choices=("fit", "offset", "none"), default="fit",
                        help="map every rank onto rank 0's clock via cross-rank peer-wait releases: offset + drift (fit, default), offset only, or keep raw per-rank clocks (none)")
    parser.add_argument("--no-grid", action="store_true", help="skip the per-rank 'grid (all CUs)' track (earliest block start -> latest arrival per segment, from the per-CU traces)")
    parser.add_argument("--keep-idle", action="store_true", help="also export barrier-wait and idle-gap spans (default: compute spans only; totals are still in the summary)")
    arguments = parser.parse_args()
    directory = arguments.run_dir.resolve()
    traces = load(directory)
    cu_traces = [] if arguments.no_cu else load_cu(directory, steps=arguments.steps)
    default_name = "stage-timing.perfetto.json.gz" if (cu_traces or arguments.gzip) else "stage-timing.perfetto.json"
    output = arguments.output or directory / default_name
    summary = arguments.summary or directory / "stage-timing-summary.json"
    alignment = merge(traces, output, cu_traces, arguments.align, keep_idle=arguments.keep_idle, grid=not arguments.no_grid)
    rows = summarize(traces)
    cu_rows = summarize_cu(cu_traces)
    summary.write_text(json.dumps({"ranks": [rank for rank, _, _ in traces], "alignment": alignment, "segments": rows,
                                   "compute_units": cu_rows}, indent=2) + "\n")
    print(f"{len(traces)} ranks -> {output}")
    print_alignment(alignment)
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
