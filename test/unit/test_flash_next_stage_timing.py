import json

import pytest
import torch

from sgl_amd_mono.qwen38_flash_next.gfx936.stage_timing import (
    SEGMENTS, START_STAGE, export_trace, kind_of, launches, segment_name,
)


def staged(kind, clock):
    """One launch of `kind`, with the barriers that kernel really emits."""
    return [(stage, clock, stage) for stage in SEGMENTS[kind]]


def record_stream(step_count=2):
    """Synthetic decode steps with the real barrier sequence of every kernel."""
    records, clock = [], 0
    for _ in range(step_count):
        for kind in ("entry", "hc_mix", "gdn", "post", "hc_mix", "qsa", "post", "ple", "tail"):
            clock += 100
            records.append((START_STAGE[kind], clock))
            for stage in SEGMENTS[kind]:
                clock += 100
                records.append((stage + 1000, clock))
                clock += 100
                records.append((stage, clock))
    return records


def test_kind_and_segment_names():
    assert kind_of(1) == "hc_mix" and kind_of(4) == "hc_mix"
    assert kind_of(45) == "post"
    assert kind_of(310) == "gdn"
    assert kind_of(413) == "qsa"
    assert kind_of(524) == "ple"
    assert kind_of(610) == "entry"
    assert kind_of(644) == "tail"
    assert kind_of(999) == "unknown"
    assert segment_name("post", 42) == "gate+up gemv"
    assert segment_name("gdn", 311) == "conv state"
    assert segment_name("qsa", 410) == "top512 select"
    # Every barrier a kernel emits must be named.
    emitted = {
        "hc_mix": [1, 2, 3, 4],
        "post": [10, 11, 13, 14, 20, 21, 30, 31, 32, 33, 40, 41, 42, 43, 44, 45, 50, 51, 53, 54, 60, 61],
        "gdn": [300, 301, 302, 303, 310, 311, 312, 313],
        "qsa": [400, 401, 402, 403, 404, 405, 406, 408, 409, 410, 411, 412, 413],
        "ple": [500, 501, 510, 511, 513, 514, 520, 521, 522, 523, 524],
        "entry": [600, 610, 611, 613, 614, 620],
        "tail": [630, 631, 633, 634, 640, 641, 642, 643, 644],
    }
    for kind, stages in emitted.items():
        for stage in stages:
            assert kind_of(stage) == kind, (kind, stage, kind_of(stage))
            assert stage in SEGMENTS[kind], f"missing name for {kind} stage {stage}"


def test_launch_split():
    records = torch.tensor(record_stream(step_count=1), dtype=torch.int64)
    kinds = [launch["kind"] for launch in launches(records)]
    assert kinds == ["entry", "hc_mix", "gdn", "post", "hc_mix", "qsa", "post", "ple", "tail"]
    for launch in launches(records):
        assert launch["marks"][0][0] == START_STAGE[launch["kind"]]


def test_export_layout(tmp_path):
    records = torch.tensor(record_stream(step_count=2), dtype=torch.int64)
    result = export_trace(records, tmp_path / "trace.json", clock_khz=1_000_000, rank=3)
    trace = json.loads((tmp_path / "trace.json").read_text())
    spans = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    names = {event["name"] for event in spans}
    assert {"step 1", "step 2", "layer 0 gdn", "layer 0 gdn+post", "layer 1 qsa",
            "embed lookup", "mix.normalize", "qkvz+ba gemv", "gate+up gemv", "lm head gemv"} <= names
    assert all(event["dur"] > 0 for event in spans)
    assert all(event["args"]["tp_rank"] == 3 for event in spans)
    # Steps must not overlap and must be ordered.
    steps = sorted((event for event in spans if event["name"].startswith("step ")), key=lambda event: event["ts"])
    assert [event["name"] for event in steps] == ["step 1", "step 2"]
    assert steps[0]["ts"] + steps[0]["dur"] <= steps[1]["ts"]
    assert result["steps"] == 2 and not result["dropped"]
    final_layer = next(event for event in spans if event["name"] == "layer 1 qsa")
    final_post = next(event for event in spans if event["name"] == "layer 1 qsa+post")
    tail = next(event for event in spans if event["name"] == "tail")
    assert final_layer["ts"] + final_layer["dur"] == final_post["ts"] + final_post["dur"]
    assert final_layer["ts"] + final_layer["dur"] < tail["ts"]
    # Layer start is the attention kernel's entry mark, not the preceding kernel's end.
    qsa_launch = next(launch for launch in launches(records) if launch["kind"] == "qsa")
    assert final_layer["ts"] == pytest.approx((qsa_launch["marks"][0][1] - int(records[0, 1])) / 1000)
    # Every launch's first mark opens the launch span, so it is not a segment of its own.
    launches_per_step = 9
    assert len(result["segments"]) == len(records) - launches_per_step * 2

    # 100 ticks at 1 GHz = 0.1 us; the check pins the tick->time conversion.
    first_segment = next(event for event in spans if event["name"] == "mix.normalize")
    assert first_segment["dur"] == pytest.approx(0.1, rel=1e-6)
    first_wait = next(event for event in spans if event["name"] == "mix.normalize / block 0 grid wait")
    assert first_wait["dur"] == pytest.approx(0.1, rel=1e-6)
    assert first_segment["ts"] + first_segment["dur"] == pytest.approx(first_wait["ts"])
    assert first_wait["args"]["depth"] == 4


def test_zero_length_spans_are_dropped(tmp_path):
    # Two barriers with the same tick bound a zero-length segment, which must not be emitted.
    records = torch.tensor([(705, 50), (600, 100), (620, 100), (610, 250)], dtype=torch.int64)
    result = export_trace(records, tmp_path / "trace.json", clock_khz=1_000_000, rank=0)
    spans = [event for event in json.loads((tmp_path / "trace.json").read_text())["traceEvents"] if event.get("ph") == "X"]
    assert all(event["dur"] > 0 for event in spans)
    assert result["steps"] == 1
