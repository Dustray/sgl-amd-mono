"""Cross-rank clock alignment in tools/merge_flash_next_stage_timing.py (no torch, no GPU).

The traces are built in the exact layout `stage_timing.export_trace` / `cu_timing.export_trace`
write: per launch one depth-2 span, then per barrier a depth-3 compute span and a depth-4
`/ block 0 grid wait` span; per-CU `cu_wait` spans carry step/layer/kernel/stage args.
"""

import gzip
import json
import random

import pytest

from tools.merge_flash_next_stage_timing import align, fit_clock, merge, sync_points


# (launch label, [(segment, stage, is_cross_rank)]) for one decode step, in launch order.
STEP = [("entry", [("embed lookup", 600, False), ("tp.peer wait", 614, True)])]
STEP += [(f"layer {layer} gdn+post", [("attn.reduce", 13, False), ("attn.peer wait", 14, True),
                                      ("down gemv", 44, False), ("moe.peer wait", 54, True)]) for layer in range(4)]
STEP += [("ple", [("embed.peer wait", 514, True)]), ("tail", [("lm head gemv", 644, False), ("gather.peer wait", 634, True)])]


def true_timeline(steps=2, jitter=0.0, seed=0):
    """[(label, segment, stage, start_us, arrival_us, release_us)] on a global clock.

    Cross-rank barriers release at the same true instant on every rank (plus optional jitter,
    added per rank by the caller); compute before them differs per rank."""
    rng = random.Random(seed)
    clock, timeline = 50.0, []
    for _ in range(steps):
        for label, segments in STEP:
            for segment, stage, cross in segments:
                start = clock
                arrival = start + rng.uniform(5, 40)
                release = arrival + (rng.uniform(2, 20) if cross else 0.5)
                timeline.append((label, segment, stage, cross, start, arrival, release))
                clock = release
            clock += 3.0
    return timeline


def rank_trace(rank, timeline, offset_us, drift_ppm, jitter=0.0, seed=0):
    """Stage + per-CU events as rank `rank` would export them with its own clock."""
    rng = random.Random(seed * 100 + rank)
    scale = 1.0 + drift_ppm * 1e-6
    local = lambda t: t * scale + offset_us  # noqa: E731
    stage, cu = [], []
    current, step, layer = None, 0, -1
    for label, segment, stage_id, cross, start, arrival, release in timeline:
        if cross:
            release += rng.uniform(0, jitter)
        if label != current:
            current = label
            if label == "entry":
                step, layer = step + 1, -1
            elif label.startswith("layer"):
                layer += 1
            stage.append({"ph": "X", "cat": "stage", "name": label, "pid": rank, "tid": 1,
                          "ts": local(start), "dur": 1.0, "args": {"tp_rank": rank, "depth": 2}})
        stage.append({"ph": "X", "cat": "stage", "name": segment, "pid": rank, "tid": 1, "ts": local(start),
                      "dur": local(arrival) - local(start), "args": {"tp_rank": rank, "depth": 3}})
        stage.append({"ph": "X", "cat": "stage", "name": f"{segment} / block 0 grid wait", "pid": rank, "tid": 1,
                      "ts": local(arrival), "dur": local(release) - local(arrival), "args": {"tp_rank": rank, "depth": 4}})
        for block in range(3):
            kind = label.split(" ")[-1].split("+")[-1] if label.startswith("layer") else label
            cu.append({"ph": "X", "cat": "cu_wait", "name": f"wait {segment}", "pid": 100 + rank, "tid": block + 1,
                       "ts": local(arrival), "dur": local(release) - local(arrival),
                       "args": {"tp_rank": rank, "cu": f"SE0.SH0.CU{block}", "step": step, "layer": layer,
                                "kernel": kind, "stage": stage_id, "phase": "grid_wait"}})
    return stage, cu


OFFSETS = {0: (0.0, 0.0), 1: (12_345.0, 30.0), 2: (-800.0, -15.0), 3: (3_000_000.0, 5.0)}


def build(timeline, jitter=0.0):
    traces, cu_traces = [], []
    for rank, (offset, drift) in OFFSETS.items():
        stage, cu = rank_trace(rank, timeline, offset, drift, jitter)
        traces.append((rank, None, stage))
        cu_traces.append((rank, None, cu))
    return traces, cu_traces


def test_sync_points_are_keyed_per_launch():
    stage, _ = rank_trace(0, true_timeline(steps=2), 0.0, 0.0)
    points = sync_points(stage)
    # per step: entry tp, 4 x (attn, moe), ple embed, tail gather
    assert len(points) == 2 * (1 + 8 + 1 + 1)
    assert ("layer 2 gdn+post", 1, "moe.peer wait") in points
    assert all(key[2].endswith("peer wait") for key in points)


def test_sync_point_survives_a_dropped_wait_span():
    stage, _ = rank_trace(0, true_timeline(steps=1), 0.0, 0.0)
    dropped = [event for event in stage if event["name"] != "attn.peer wait / block 0 grid wait"
               or event["ts"] != min(e["ts"] for e in stage if e["name"] == event["name"])]
    points = sync_points(dropped)
    # The first layer's attn barrier falls back to its arrival; nothing after it shifts.
    assert len(points) == len(sync_points(stage))
    assert points[("layer 1 gdn+post", 0, "attn.peer wait")] == sync_points(stage)[("layer 1 gdn+post", 0, "attn.peer wait")]


def test_fit_recovers_offset_and_drift():
    traces, _ = build(true_timeline(steps=3))
    fits = align(traces)
    for rank, (offset, drift) in OFFSETS.items():
        scale = 1.0 + drift * 1e-6
        # reference = (local - offset) / scale
        assert fits[rank]["scale"] == pytest.approx(1.0 / scale, rel=1e-9)
        assert fits[rank]["offset_us"] == pytest.approx(-offset / scale, abs=1e-3)
        assert fits[rank]["residual_max_us"] < 1e-3


def test_offset_mode_ignores_drift():
    traces, _ = build(true_timeline(steps=1))
    fits = align(traces, "offset")
    assert all(fit["scale"] == 1.0 for fit in fits.values())


def test_outlier_barrier_is_rejected():
    pairs = [(float(x), float(x) + 100.0) for x in range(0, 2000, 20)]
    pairs[37] = (pairs[37][0], pairs[37][1] + 500.0)
    fit = fit_clock(pairs)
    assert fit["used"] == len(pairs) - 1
    assert fit["offset_us"] == pytest.approx(100.0, abs=1e-6)


def test_mismatched_captures_are_refused():
    traces, _ = build(true_timeline(steps=1))
    traces[1] = (1, None, [])
    with pytest.raises(ValueError, match="same decode step"):
        align(traces)


def test_merged_trace_lines_up_across_ranks(tmp_path):
    traces, cu_traces = build(true_timeline(steps=2), jitter=2.0)
    output = tmp_path / "merged.json.gz"
    report = merge(traces, output, cu_traces, keep_idle=True)
    events = json.loads(gzip.open(output, "rt").read())["traceEvents"]
    spans = [event for event in events if event.get("ph") == "X"]
    assert min(event["ts"] for event in spans) == pytest.approx(0.0, abs=1e-6)
    # Every rank's per-CU peer-wait releases now coincide within the injected jitter.
    releases = {}
    for event in spans:
        if event.get("cat") == "cu_wait" and event["args"]["stage"] == 54 and event["tid"] == 1:
            releases.setdefault(event["pid"], []).append(event["ts"] + event["dur"])
    reference = releases[100]
    for pid, values in releases.items():
        assert max(abs(a - b) for a, b in zip(values, reference)) < 2.5
    assert set(report["ranks"]) == {"0", "1", "2", "3"}
    assert all(row["max_us"] < 2.5 for row in report["cu_check"].values())


def test_align_none_keeps_raw_clocks(tmp_path):
    traces, cu_traces = build(true_timeline(steps=1))
    report = merge(traces, tmp_path / "raw.json", cu_traces, align_mode="none")
    assert report == {}
    events = json.loads((tmp_path / "raw.json").read_text())["traceEvents"]
    rank3 = [event["ts"] for event in events if event.get("pid") == 3 and event.get("ph") == "X"]
    assert min(rank3) > 3_000_000


def test_merge_leaves_waits_and_gaps_out_by_default(tmp_path):
    traces, cu_traces = build(true_timeline(steps=1))
    report = merge(traces, tmp_path / "compute.json", cu_traces)
    events = json.loads((tmp_path / "compute.json").read_text())["traceEvents"]
    spans = [event for event in events if event.get("ph") == "X"]
    assert spans and not any(event.get("cat") in ("cu_wait", "cu_gap") for event in spans)
    assert not any(event["name"].endswith("block 0 grid wait") for event in spans)
    # Alignment still uses the waits from the inputs.
    assert all(row["max_us"] < 1e-3 for row in report["cu_check"].values())


def test_grid_spans_cover_every_block():
    from tools.merge_flash_next_stage_timing import grid_spans

    def span(block, ts, dur, stage=301, step=1):
        return {"ph": "X", "cat": "cu_compute", "name": "layer 0 gdn: mix.down", "pid": 100, "tid": block + 1,
                "ts": ts, "dur": dur, "args": {"tp_rank": 0, "block": block, "cu": f"CU{block}", "step": step,
                                               "layer": 0, "kernel": "gdn", "stage": stage, "phase": "compute"}}

    events = [span(0, 10.0, 260.0), span(1, 9.0, 2.0), span(3, 11.0, 300.0),
              {"ph": "X", "cat": "cu_wait", "name": "wait mix.down", "pid": 100, "tid": 1, "ts": 270.0, "dur": 45.0,
               "args": {"block": 0, "step": 1, "layer": 0, "kernel": "gdn", "stage": 301}},
              span(0, 500.0, 5.0, step=2), span(1, 501.0, 7.0, step=2)]
    spans = grid_spans(events, pid=0)
    assert [(event["ts"], event["dur"], event["args"]["blocks"]) for event in spans] == [(9.0, 302.0, 3), (500.0, 8.0, 2)]
    assert spans[0]["args"]["last_block"] == 3 and spans[0]["tid"] == 2 and spans[0]["pid"] == 0


def test_merge_adds_grid_track(tmp_path):
    traces, cu_traces = build(true_timeline(steps=1))
    # The fixture only carries peer waits; add a compute span before each one.
    cu_traces = [(rank, path, events + [dict(event, cat="cu_compute", name="compute", ts=event["ts"] - 3.0, dur=3.0)
                                        for event in events if event.get("cat") == "cu_wait"])
                 for rank, path, events in cu_traces]
    merge(traces, tmp_path / "grid.json", cu_traces)
    events = json.loads((tmp_path / "grid.json").read_text())["traceEvents"]
    assert any(event.get("cat") == "grid_stage" and event["pid"] == 0 for event in events)
    merge(traces, tmp_path / "nogrid.json", cu_traces, grid=False)
    assert not any(event.get("cat") == "grid_stage" for event in json.loads((tmp_path / "nogrid.json").read_text())["traceEvents"])
