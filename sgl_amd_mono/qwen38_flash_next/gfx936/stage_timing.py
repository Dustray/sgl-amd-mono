"""Device-side stage timing for the Flash-Next kernels.

Every grid barrier records block 0's arrival and release timestamps (`kernel.cpp:grid_sync`).
Stage numbers are unique per kernel kind, so a flat record stream is enough to rebuild which
stage of which launch ran when. A segment span runs from the previous barrier release to this
barrier's release, which every block must reach first, so its duration is the whole grid's time
for that stage even though only block 0 records; block 0's own compute and wait are in its args
(`block0_compute_us`, `block0_wait_us`), and the block-0 wait is also a depth-4 child span:

| stage | kernel | stage | kernel |
|---|---|---|---|
| 1-4 | hc_mix | 400-413 | gdn: qsa_attention |
| 10-61 | post_attention | 500-524 | ple_decode |
| 300-313 | gdn_attention | 600-620 | decode_entry |
| 630-644 | decode_tail | | |

The host allocates one buffer per rank and attaches it to every compiled library (each `.so`
has its own copy of the device symbol because it includes `kernel.cpp`). Records are
`(stage, ticks)` pairs after a 2-slot header: `[0]` cursor, `[1]` dropped flag. Barrier-arrival
records encode their stage as `stage + SYNC_ARRIVAL_OFFSET`.
"""

import ctypes
import json
from pathlib import Path

import torch


KINDS = (
    ("hc_mix", 1, 9),
    ("post", 10, 69),
    ("gdn", 300, 319),
    ("qsa", 400, 419),
    ("ple", 500, 529),
    ("entry", 600, 629),
    ("tail", 630, 649),
)
START_STAGE = {"hc_mix": 700, "post": 701, "gdn": 702, "qsa": 703, "ple": 704, "entry": 705, "tail": 706}
SYNC_ARRIVAL_OFFSET = 1000
# Segment spans run barrier release -> release, so they are whole-grid times, not block 0's compute.
STAGE_THREAD = "decode steps (stage = all blocks, release to release)"

# The barrier a segment *ends* at names the work done before it.
SEGMENTS = {
    "hc_mix": {1: "mix.normalize", 2: "mix.down", 3: "mix.up", 4: "mix.branch"},
    "post": {10: "attn.publish", 11: "attn.exchange", 13: "attn.reduce", 14: "attn.peer wait",
             20: "inject.gate", 21: "inject.add",
             30: "mix.normalize", 31: "mix.down", 32: "mix.up", 33: "mix.branch",
             40: "router.top10", 41: "shared gate", 42: "gate+up gemv", 43: "silu act",
             44: "down gemv", 45: "combine",
             50: "moe.publish", 51: "moe.exchange", 53: "moe.reduce", 54: "moe.peer wait",
             60: "inject.gate", 61: "inject.add"},
    "gdn": {300: "mix.normalize", 301: "mix.down", 302: "mix.up", 303: "mix.branch",
            310: "qkvz+ba gemv", 311: "conv state", 312: "recurrent", 313: "output gemv"},
    "qsa": {400: "validate", 401: "mix.normalize", 402: "mix.down", 403: "mix.up", 404: "mix.branch",
            405: "qkv+index gemv", 406: "norm+rope+kv", 408: "index compress", 409: "index scores",
            410: "top512 select", 411: "attn scores", 412: "softmax+attn", 413: "output gemv",
            416: "index chunk sort", 417: "index merge"},
    "ple": {500: "validate", 501: "ngram lookup", 510: "embed.publish", 511: "embed.exchange",
            513: "embed.reduce", 514: "embed.peer wait", 520: "key+value gemv", 521: "normalize",
            522: "gate", 523: "conv norm", 524: "short conv"},
    "entry": {600: "embed lookup", 610: "tp.publish", 611: "tp.exchange", 613: "tp.reduce",
              614: "tp.peer wait", 620: "expand branches"},
    "tail": {640: "mix.normalize", 641: "mix.down", 642: "mix.up", 643: "mix.branch", 644: "lm head gemv",
             630: "gather.publish", 631: "gather.exchange", 633: "gather.reduce", 634: "gather.peer wait"},
}


def kind_of(stage):
    if stage >= SYNC_ARRIVAL_OFFSET:
        stage -= SYNC_ARRIVAL_OFFSET
    for name, start in START_STAGE.items():
        if stage == start:
            return name
    for name, low, high in KINDS:
        if low <= stage <= high:
            return name
    return "unknown"


def segment_name(kind, stage):
    return SEGMENTS.get(kind, {}).get(stage, f"stage{stage}")


class StageTiming:
    """One int64 buffer per rank, attached to every Flash-Next library on that rank."""

    def __init__(self, device, rank, records=1 << 18):
        self.device = torch.device(device)
        self.rank = rank
        self.capacity = 2 + 2 * records
        self.buffer = torch.zeros(self.capacity, dtype=torch.int64, device=self.device)
        self.libraries = []

    def attach(self, program):
        """Attach to one compiled library; call for every program (kernel, gdn, qsa, ple, io).

        Each library includes kernel.cpp, so each has its own copy of the device symbols and
        must be attached separately.
        """
        if program.library in self.libraries:
            return
        program.library.flash_stage_attach.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        program.library.flash_stage_attach.restype = ctypes.c_int
        # Only HIPProgram defines check(); the per-module programs carry it on their base.
        check = getattr(program, "check", None) or program.base.check
        with torch.cuda.device(self.device):
            check(program.library.flash_stage_attach(ctypes.c_void_p(self.buffer.data_ptr()), self.capacity))
        self.libraries.append(program.library)

    def detach(self):
        for library in self.libraries:
            library.flash_stage_attach.argtypes = [ctypes.c_void_p, ctypes.c_int64]
            with torch.cuda.device(self.device):
                library.flash_stage_attach(None, 0)
        self.libraries = []

    def reset(self):
        self.buffer.zero_()

    def read(self):
        """(records, dropped): records is an (n, 2) int64 CPU tensor of (stage, ticks)."""
        cursor = int(self.buffer[0].item())
        dropped = bool(int(self.buffer[1].item()))
        usable = max(0, min(cursor, self.capacity // 2 - 1))
        values = self.buffer[2:2 + 2 * usable].to("cpu", non_blocking=False).view(-1, 2)
        return values, dropped

    def export(self, path, clock_khz, labels=None, base=None):
        records, dropped = self.read()
        clock_khz = clock_khz or int(torch.cuda.get_device_properties(self.device).clock_rate)
        return export_trace(records, path, clock_khz, self.rank, dropped, labels, base)


def launches(records):
    """Split the flat (stage, ticks) stream into launched kernels.

    A launch begins at its kernel's start stage; barriers inside a kernel never repeat that
    stage, so the split is unambiguous.
    """
    # One bulk conversion: per-element tensor indexing is several aten ops per record, which the
    # torch profiler records when export runs inside a capture (8M records for 32 CU steps ->
    # host OOM in stop_trace, 2026-11-14).
    rows = records.tolist() if hasattr(records, "tolist") else records
    current = None
    for row in rows:
        stage, ticks = int(row[0]), int(row[1])
        kind = kind_of(stage)
        if current is None or kind != current["kind"] or stage == START_STAGE.get(kind):
            if current is not None:
                yield current
            current = {"kind": kind, "marks": []}
        current["marks"].append((stage, ticks))
    if current is not None:
        yield current


def walk(launch):
    """Yield (phase, name, stage, start_ticks, end_ticks) for one launch's marks.

    The first mark opens the launch and is not a segment of its own. A barrier contributes two
    spans: the compute that ran before the block arrived, and the block's wait inside the barrier.
    A mark without its paired release falls back to one `legacy` span.
    """
    marks = launch["marks"]
    previous = marks[0][1] if marks else 0
    index = 1
    while index < len(marks):
        stage, ticks = marks[index]
        if stage >= SYNC_ARRIVAL_OFFSET and index + 1 < len(marks) and marks[index + 1][0] == stage - SYNC_ARRIVAL_OFFSET:
            barrier = stage - SYNC_ARRIVAL_OFFSET
            name = segment_name(launch["kind"], barrier)
            release = marks[index + 1][1]
            yield ("compute", name, barrier, previous, ticks)
            yield ("grid_wait", name, barrier, ticks, release)
            previous = release
            index += 2
            continue
        yield ("legacy", segment_name(launch["kind"], stage), stage, previous, ticks)
        previous = ticks
        index += 1


def export_trace(records, path, clock_khz, rank, dropped=False, labels=None, base=None):
    """Write one rank's stages as a Perfetto trace: step -> layer -> kernel -> segment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [{"ph": "M", "name": "process_name", "pid": rank, "tid": 0, "args": {"name": f"rank {rank} (device stages)"}},
              {"ph": "M", "name": "thread_name", "pid": rank, "tid": 1, "args": {"name": STAGE_THREAD}}]
    ticks_per_us = clock_khz / 1000.0
    step, layer, step_open, layer_open = 0, 0, None, None
    summary = []
    for launch in launches(records):
        marks = launch["marks"]
        if not marks:
            continue
        first, last = marks[0][1], marks[-1][1]
        base = first if base is None else base
        if launch["kind"] == "entry":
            step += 1
            layer = -1
            step_open = {"start": first, "step": step}
            layer_open = None
        if launch["kind"] in ("gdn", "qsa"):
            layer += 1
            layer_open = {"kind": launch["kind"], "start": first}
        label = f"{launch['kind']}"
        if launch["kind"] == "post" and layer_open is not None:
            label = f"layer {layer} {layer_open['kind']}+post"
        events.append(_span(rank, 1, label, first, last, base, ticks_per_us, depth_extra=2))
        if launch["kind"] == "post" and layer_open is not None:
            events.append(_span(rank, 1, f"layer {layer} {layer_open['kind']}",
                                layer_open["start"], last, base, ticks_per_us))
            layer_open = None
        segments = list(walk(launch))
        for position, (phase, name, stage, start, end) in enumerate(segments):
            if phase == "grid_wait":
                events.append(_span(rank, 1, f"{name} / block 0 grid wait", start, end, base, ticks_per_us, depth_extra=4))
                summary.append({"kind": launch["kind"], "stage": stage, "segment": name,
                                "phase": "block0_grid_wait", "us": (end - start) / ticks_per_us})
                continue
            following = segments[position + 1] if position + 1 < len(segments) else None
            if phase == "compute" and following is not None and following[0] == "grid_wait":
                # The segment span covers the whole grid: previous release -> this barrier's release,
                # i.e. until the slowest block arrived. Block 0's own compute/wait split goes to args;
                # a span ending at block 0's arrival read as "all CUs" and understated the stage
                # (layer 8 gdn mix.down: block 0 235 us, last block 273 us, 2026-11-16).
                release = following[4]
                span = _span(rank, 1, name, start, release, base, ticks_per_us, depth_extra=3)
                if span is not None:
                    span["args"].update({"span": "barrier release to release (all blocks)",
                                         "block0_compute_us": (end - start) / ticks_per_us,
                                         "block0_wait_us": (release - end) / ticks_per_us})
                events.append(span)
                summary.append({"kind": launch["kind"], "stage": stage, "segment": name,
                                "phase": "compute", "us": (end - start) / ticks_per_us})
                continue
            events.append(_span(rank, 1, name, start, end, base, ticks_per_us, depth_extra=3))
            summary.append({"kind": launch["kind"], "stage": stage, "segment": name,
                            "phase": "compute" if phase == "compute" else "legacy", "us": (end - start) / ticks_per_us})
        if launch["kind"] == "tail":
            if step_open is not None:
                events.append(_span(rank, 1, f"step {step}", step_open["start"], last, base, ticks_per_us))
                step_open = None
    events = [event for event in events if event is not None]
    text = '{"displayTimeUnit":"ms","traceEvents":[' + ",".join(json.dumps(event, separators=(",", ":")) for event in events) + "]}"
    path.write_text(text)
    return {"trace": str(path), "rank": rank, "steps": step, "events": len(events), "dropped": dropped,
            "base_ticks": base, "segments": summary}


def _span(pid, tid, name, start, end, base, ticks_per_us, depth_extra=0):
    if end <= start:
        return None
    return {"ph": "X", "cat": "stage", "name": name, "pid": pid, "tid": tid,
            "ts": (start - base) / ticks_per_us, "dur": (end - start) / ticks_per_us,
            "args": {"tp_rank": pid, "depth": depth_extra}}
