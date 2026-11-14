import json

import pytest
import torch

from sgl_amd_mono.qwen38_flash_next.gfx936.cu_timing import (
    PID_OFFSET, cu_key, cu_label, decode_hardware_id, export_trace, summarize,
)
from sgl_amd_mono.qwen38_flash_next.gfx936.stage_timing import SEGMENTS, START_STAGE


def hardware_id(se, sh, cu, wave=0, simd=0):
    return wave | (simd << 4) | (cu << 8) | (sh << 12) | (se << 13)


def block_records(hardware, step_count=1, offset=0, kinds=("entry", "hc_mix", "gdn", "post", "ple", "tail")):
    """One block's marks over `step_count` decode steps, with the real barrier sequence."""
    records, clock = [], offset
    for _ in range(step_count):
        for kind in kinds:
            clock += 100
            records.append((START_STAGE[kind], clock, hardware))
            for stage in SEGMENTS[kind]:
                clock += 100
                records.append((stage + 1000, clock, hardware))
                clock += 100
                records.append((stage, clock, hardware))
    return torch.tensor(records, dtype=torch.int64)


def test_hardware_id_decode():
    assert decode_hardware_id(hardware_id(se=3, sh=1, cu=11)) == {
        "wave": 0, "simd": 0, "pipe": 0, "cu": 11, "sh": 1, "se": 3}
    assert cu_key(hardware_id(se=2, sh=0, cu=5)) == (2, 0, 5)
    # Wave/SIMD/pipe differ between marks on one CU and must not split the row.
    assert cu_key(hardware_id(se=2, sh=0, cu=5, wave=7, simd=2)) == (2, 0, 5)
    assert cu_label((2, 0, 5)) == "SE2.SH0.CU5"


def test_one_row_per_compute_unit(tmp_path):
    per_block = [block_records(hardware_id(se=block // 20, sh=0, cu=block % 20 % 16)) for block in range(4)]
    result = export_trace(per_block, tmp_path / "cu.json", clock_khz=1_000_000, rank=5)
    trace = json.loads((tmp_path / "cu.json").read_text())
    threads = [event for event in trace["traceEvents"] if event.get("name") == "thread_name"]
    assert len(threads) == result["compute_units"] == 4
    assert {thread["args"]["name"].split(" ")[0] for thread in threads} == {
        "SE0.SH0.CU0", "SE0.SH0.CU1", "SE0.SH0.CU2", "SE0.SH0.CU3"}
    spans = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    assert spans and all(event["pid"] == PID_OFFSET + 5 for event in spans)
    assert all(event["dur"] > 0 for event in spans)
    # Every span sits on the row of the CU that produced it.
    by_tid = {thread["tid"]: thread["args"]["name"].split(" ")[0] for thread in threads}
    assert all(by_tid[event["tid"]] == event["args"]["cu"] for event in spans)
    assert result["shared_units"] == [] and result["migrated_launch_count"] == 0


def test_compute_and_wait_phases(tmp_path):
    per_block = [block_records(hardware_id(se=0, sh=0, cu=1))]
    export_trace(per_block, tmp_path / "cu.json", clock_khz=1_000_000, rank=0)
    spans = [event for event in json.loads((tmp_path / "cu.json").read_text())["traceEvents"] if event.get("ph") == "X"]
    compute = [event for event in spans if event["cat"] == "cu_compute"]
    waits = [event for event in spans if event["cat"] == "cu_wait"]
    assert len(compute) == len(waits)
    # 100 ticks at 1 GHz = 0.1 us, for both the compute and the barrier wait of every segment.
    assert all(event["dur"] == pytest.approx(0.1, rel=1e-6) for event in spans)
    assert all(event["name"].startswith("wait ") for event in waits)
    assert {event["args"]["phase"] for event in compute} == {"compute"}
    # Operator names carry the layer, so a row reads as "what is this CU running".
    names = {event["name"] for event in compute}
    assert "layer 0 gdn: qkvz+ba gemv" in names
    assert "entry: embed lookup" in names
    assert any(event["args"]["layer"] == 0 and event["args"]["kernel"] == "post" for event in compute)


def test_launch_gaps_get_their_own_spans(tmp_path):
    # block_records leaves 100 idle ticks between launches (clock += 100 before each entry
    # mark); those must show up as cu_gap spans, so the idle time is visible and optimizable.
    per_block = [block_records(hardware_id(se=0, sh=0, cu=3))]
    result = export_trace(per_block, tmp_path / "cu.json", clock_khz=1_000_000, rank=0)
    spans = [event for event in json.loads((tmp_path / "cu.json").read_text())["traceEvents"] if event.get("ph") == "X"]
    gaps = [event for event in spans if event["cat"] == "cu_gap"]
    assert gaps
    assert all(event["args"]["phase"] == "gap" for event in gaps)
    assert all(event["name"].startswith("gap ") for event in gaps)
    # The fixtures idle 100 ticks (0.1 us) between consecutive launches of one step.
    assert all(event["dur"] == pytest.approx(0.1, rel=1e-6) for event in gaps)
    assert any(event["name"] == "gap entry -> hc_mix" for event in gaps)
    # compute + wait + gap now tile the row: no unexplained time left between spans.
    rows = sorted((event for event in spans if event["tid"] == gaps[0]["tid"]), key=lambda event: event["ts"])
    for earlier, later in zip(rows, rows[1:]):
        assert later["ts"] == pytest.approx(earlier["ts"] + earlier["dur"], rel=1e-6, abs=1e-9)
    # The summary reports the idle total alongside compute and wait.
    summary = result["summary"][0]
    assert summary["gap_us"] == pytest.approx(0.1 * len(gaps), rel=1e-6)


def test_blocks_on_the_same_cu_share_a_row(tmp_path):
    # Normal across launches: the scheduler moves a block to another CU on every launch, so one
    # CU row hosts several blocks over time.
    hardware = hardware_id(se=1, sh=0, cu=2)
    per_block = [block_records(hardware), block_records(hardware, offset=50)]
    result = export_trace(per_block, tmp_path / "cu.json", clock_khz=1_000_000, rank=0)
    assert result["compute_units"] == 1
    assert result["shared_units"] == ["SE1.SH0.CU2"]
    threads = [event for event in json.loads((tmp_path / "cu.json").read_text())["traceEvents"]
               if event.get("name") == "thread_name"]
    assert threads[0]["args"]["name"] == "SE1.SH0.CU2 (2 blocks)"


def test_a_launch_that_changes_cu_is_counted_as_migrated(tmp_path):
    records = block_records(hardware_id(se=0, sh=0, cu=1), kinds=("hc_mix",))
    records[-4:, 2] = hardware_id(se=0, sh=0, cu=9)
    result = export_trace([records], tmp_path / "cu.json", clock_khz=1_000_000, rank=0)
    assert result["migrated_launch_count"] == 1
    assert result["migrated_launches"][0]["cus"] == ["SE0.SH0.CU1", "SE0.SH0.CU9"]
    # The launch is still attributed to the CU it started on rather than dropped.
    assert result["compute_units"] == 1


def test_shared_time_base_with_the_stage_trace(tmp_path):
    per_block = [block_records(hardware_id(se=0, sh=0, cu=0), offset=10_000)]
    result = export_trace(per_block, tmp_path / "cu.json", clock_khz=1_000_000, rank=0, base=1_000)
    spans = [event for event in json.loads((tmp_path / "cu.json").read_text())["traceEvents"] if event.get("ph") == "X"]
    assert result["base_ticks"] == 1_000
    # First compute span starts one mark (100 ticks) after the kernel entry at 10_100.
    assert min(event["ts"] for event in spans) == pytest.approx((10_100 - 1_000) / 1000)


def test_idle_and_busy_summary(tmp_path):
    per_block = [block_records(hardware_id(se=0, sh=0, cu=index)) for index in range(3)]
    result = export_trace(per_block, tmp_path / "cu.json", clock_khz=1_000_000, rank=0)
    rows = result["summary"]
    assert [row["cu"] for row in rows] == ["SE0.SH0.CU0", "SE0.SH0.CU1", "SE0.SH0.CU2"]
    assert all(row["compute_us"] > 0 and row["wait_us"] > 0 for row in rows)
    assert all(row["slowest_segment"] for row in rows)
    assert summarize([], 1.0) == []


def test_dropped_records_are_reported(tmp_path):
    per_block = [block_records(hardware_id(se=0, sh=0, cu=0))]
    result = export_trace(per_block, tmp_path / "cu.json", clock_khz=1_000_000, rank=0, dropped=True)
    assert result["dropped"] is True


def test_empty_blocks_are_skipped(tmp_path):
    per_block = [torch.zeros((0, 3), dtype=torch.int64), block_records(hardware_id(se=0, sh=0, cu=7))]
    result = export_trace(per_block, tmp_path / "cu.json", clock_khz=1_000_000, rank=0)
    assert result["blocks"] == 1 and result["compute_units"] == 1
