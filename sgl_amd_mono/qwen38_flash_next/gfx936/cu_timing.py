"""Per-CU device timing for the Flash-Next kernels.

`stage_timing` records block 0 only, so it shows the critical path but not what the other 79
compute units were doing. Here *every* block marks the same stages (`cu_mark` in `kernel.cpp`,
called from `stage_mark`/`grid_sync`), and each record carries the hardware `HW_ID` of the wave
that wrote it, so a trace can put one row per CU and answer "at this instant, which operator was
CU 37 running, and was it computing or stuck in a grid barrier?".

Buffer layout (int64, one per rank): `[0]` records per block, `[1]` dropped flag, then one region
per block of `[cursor, (stage, ticks, hw_id) * records]`. Only thread 0 of a block writes its own
region, so no atomics are needed and each block's marks stay in program order.

Spans come from the same walk as the stage trace (`stage_timing.walk`): between two barriers a
block computes, inside a barrier it waits.

A block keeps its CU for the whole of one launch, but the scheduler places it somewhere else on
the next launch (measured on gfx936: 80 blocks land on 80 distinct CUs per launch, one each, but a
given block visits a different CU each launch). So a span is attributed to the CU of *its own
launch*, and a CU row legitimately hosts different blocks over time — `shared_units` reports that,
it is not an error. A block that changes CU inside a single launch would break attribution, so
those launches are counted in `migrated_launch_count` instead.

    SGLANG_AMD_FLASH_CU_TIMING=1 SGLANG_AMD_FLASH_STAGE_TIMING_STEP=5 \
      SGLANG_AMD_FLASH_REPORT_DIR=<run>/runtime python3 bench_flash_next.py --mode mono ...
    python3 tools/merge_flash_next_stage_timing.py --run-dir <run>/runtime
"""

import ctypes
import json
from pathlib import Path

import torch

from .stage_timing import launches, walk


BLOCKS = 80
PID_OFFSET = 100  # keeps CU processes clear of the per-rank stage-timing pids


def decode_hardware_id(value):
    """Split a gfx9 HW_ID register value into the fields that identify the CU."""
    value = int(value) & 0xFFFFFFFF
    return {"wave": value & 0xF, "simd": (value >> 4) & 0x3, "pipe": (value >> 6) & 0x3,
            "cu": (value >> 8) & 0xF, "sh": (value >> 12) & 0x1, "se": (value >> 13) & 0x7}


def cu_key(value):
    """(se, sh, cu) — the stable physical identity of a compute unit."""
    fields = decode_hardware_id(value)
    return (fields["se"], fields["sh"], fields["cu"])


def cu_label(key):
    se, sh, cu = key
    return f"SE{se}.SH{sh}.CU{cu:<2}".strip()


class CUTiming:
    """One int64 buffer per rank, attached to every Flash-Next library on that rank."""

    def __init__(self, device, rank, records=1 << 13, blocks=BLOCKS):
        self.device = torch.device(device)
        self.rank = rank
        self.blocks = blocks
        self.records = records
        self.stride = 1 + 3 * records
        self.capacity = 2 + blocks * self.stride
        self.buffer = torch.zeros(self.capacity, dtype=torch.int64, device=self.device)
        self.buffer[0] = records
        self.libraries = []

    def attach(self, program):
        """Attach to one compiled library; every program has its own copy of the device symbols."""
        if program.library in self.libraries:
            return
        program.library.flash_cu_attach.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        program.library.flash_cu_attach.restype = ctypes.c_int
        check = getattr(program, "check", None) or program.base.check
        with torch.cuda.device(self.device):
            check(program.library.flash_cu_attach(ctypes.c_void_p(self.buffer.data_ptr()), self.records))
        self.libraries.append(program.library)

    def detach(self):
        for library in self.libraries:
            library.flash_cu_attach.argtypes = [ctypes.c_void_p, ctypes.c_int64]
            with torch.cuda.device(self.device):
                library.flash_cu_attach(None, 0)
        self.libraries = []

    def reset(self):
        self.buffer.zero_()
        self.buffer[0] = self.records

    def read(self):
        """(per_block, dropped): per_block[b] is an (n, 3) int64 CPU tensor of (stage, ticks, hw_id)."""
        host = self.buffer.to("cpu", non_blocking=False)
        dropped = bool(int(host[1].item()))
        per_block = []
        for block in range(self.blocks):
            start = 2 + block * self.stride
            cursor = max(0, min(int(host[start].item()), self.records))
            per_block.append(host[start + 1:start + 1 + 3 * cursor].view(-1, 3))
        return per_block, dropped

    def export(self, path, clock_khz, base=None):
        per_block, dropped = self.read()
        clock_khz = clock_khz or int(torch.cuda.get_device_properties(self.device).clock_rate)
        return export_trace(per_block, path, clock_khz, self.rank, dropped, base)


def export_trace(per_block, path, clock_khz, rank, dropped=False, base=None):
    """Write one rank's per-CU occupancy as a Perfetto trace: one thread per compute unit.

    `per_block` is a list of (n, 3) `(stage, ticks, hw_id)` record tensors, one per block.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ticks_per_us = clock_khz / 1000.0
    pid = PID_OFFSET + rank
    spans = []
    blocks_per_cu = {}
    migrated = []
    for block, records in enumerate(per_block):
        if records is None or len(records) == 0:
            continue
        records = records.tolist() if hasattr(records, "tolist") else records
        step, layer, kinds, offset = 0, -1, {}, 0
        prev_end, prev_kind = None, None
        for launch in launches(records):
            marks = launch["marks"]
            first_index, offset = offset, offset + len(marks)
            if not marks:
                continue
            launch_keys = {cu_key(row[2]) for row in records[first_index:offset]}
            key = cu_key(records[first_index][2])
            if len(launch_keys) > 1:
                migrated.append({"block": block, "step": step, "kind": launch["kind"],
                                 "cus": sorted(cu_label(item) for item in launch_keys)})
            kind = launch["kind"]
            # Idle time on this CU: after the previous launch's last mark the block did nothing
            # until the next kernel's entry mark. That is neither compute nor a barrier wait,
            # and it is exactly the launch gap worth optimizing, so give it its own span.
            if prev_end is not None and marks[0][1] > prev_end:
                spans.append({"key": key, "block": block, "step": step, "layer": layer, "kind": kind,
                              "from_kind": prev_kind, "scope": kind, "phase": "gap", "stage": None,
                              "name": f"{prev_kind} -> {kind}", "start": prev_end, "end": marks[0][1]})
            prev_end, prev_kind = marks[-1][1], kind
            if kind == "entry":
                step += 1
                layer = -1
            if kind in ("gdn", "qsa"):
                layer += 1
                kinds[layer] = kind
            scope = f"layer {layer} {kinds.get(layer, kind)}" if kind in ("gdn", "qsa", "post", "hc_mix") and layer >= 0 else kind
            blocks_per_cu.setdefault(key, set()).add(block)
            for phase, name, stage, start, end in walk(launch):
                if end <= start:
                    continue
                spans.append({"key": key, "block": block, "step": step, "layer": layer, "kind": kind,
                              "scope": scope, "phase": phase, "stage": stage, "name": name,
                              "start": start, "end": end})
    if base is None and spans:
        base = min(span["start"] for span in spans)
    base = base or 0
    # Stream events to disk: 32 steps is ~8M events per rank, and building the full event list plus
    # one joined string held tens of GB per rank while all 8 ranks exported at once (host OOM, 2026-11-14).
    count = 0

    def emit(event):
        nonlocal count
        output.write(("," if count else "") + json.dumps(event, separators=(",", ":")))
        count += 1

    output = path.open("w")
    output.write('{"displayTimeUnit":"ms","traceEvents":[')
    emit({"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
          "args": {"name": f"rank {rank} (per-CU occupancy)"}})
    tids = {}
    for key in sorted(blocks_per_cu):
        tids[key] = len(tids) + 1
        hosted = sorted(blocks_per_cu[key])
        suffix = f"block {hosted[0]}" if len(hosted) == 1 else f"{len(hosted)} blocks"
        emit({"ph": "M", "name": "thread_name", "pid": pid, "tid": tids[key],
              "args": {"name": f"{cu_label(key)} ({suffix})"}})
    for span in spans:
        phase = span["phase"]
        if phase == "gap":
            name = f"gap {span['name']}"
            category = "cu_gap"
        elif phase == "grid_wait":
            name = f"wait {span['name']}"
            category = "cu_wait"
        else:
            name = f"{span['scope']}: {span['name']}"
            category = "cu_compute"
        emit({"ph": "X", "cat": category, "name": name, "pid": pid, "tid": tids[span["key"]],
                       "ts": (span["start"] - base) / ticks_per_us,
                       "dur": (span["end"] - span["start"]) / ticks_per_us,
                       "args": {"tp_rank": rank, "block": span["block"], "cu": cu_label(span["key"]),
                                "se": span["key"][0], "shader_array": span["key"][1], "compute_unit": span["key"][2],
                                "step": span["step"], "layer": span["layer"], "kernel": span["kind"],
                                "stage": span["stage"], "phase": span["phase"]}})
    output.write("]}")
    output.close()
    return {"trace": str(path), "rank": rank, "events": count, "dropped": dropped,
            "base_ticks": base, "compute_units": len(tids), "blocks": sum(1 for records in per_block if records is not None and len(records)),
            "shared_units": sorted(cu_label(key) for key, blocks in blocks_per_cu.items() if len(blocks) > 1),
            "migrated_launches": migrated[:16], "migrated_launch_count": len(migrated),
            "summary": summarize(spans, ticks_per_us)}


def summarize(spans, ticks_per_us):
    """Per-CU compute, barrier-wait and idle-gap totals, plus the slowest segment seen on that CU."""
    rows = {}
    for span in spans:
        row = rows.setdefault(span["key"], {"cu": cu_label(span["key"]), "compute_us": 0.0, "wait_us": 0.0,
                                            "gap_us": 0.0, "spans": 0, "slowest_segment": None, "slowest_us": 0.0})
        microseconds = (span["end"] - span["start"]) / ticks_per_us
        row["spans"] += 1
        row[{"grid_wait": "wait_us", "gap": "gap_us"}.get(span["phase"], "compute_us")] += microseconds
        if span["phase"] not in ("grid_wait", "gap") and microseconds > row["slowest_us"]:
            row["slowest_us"], row["slowest_segment"] = microseconds, f"{span['scope']}: {span['name']}"
    return [rows[key] for key in sorted(rows)]
