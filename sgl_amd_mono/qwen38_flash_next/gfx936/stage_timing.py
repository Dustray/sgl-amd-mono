"""Device-side stage timing for the Flash-Next kernels.

Every kernel marks `wall_clock64()` at each grid barrier (`kernel.cpp:stage_mark`), tagged with
the barrier's stage number. Stage numbers are unique per kernel kind, so a flat record stream is
enough to rebuild which stage of which launch ran when:

| stage | kernel | stage | kernel |
|---|---|---|---|
| 1-4 | hc_mix | 400-413 | gdn: qsa_attention |
| 10-61 | post_attention | 500-524 | ple_decode |
| 300-313 | gdn_attention | 600-620 | decode_entry |
| 630-644 | decode_tail | | |

The host allocates one buffer per rank and attaches it to every compiled library (each `.so`
has its own copy of the device symbol because it includes `kernel.cpp`). Records are
`(stage, ticks)` pairs after a 2-slot header: `[0]` cursor, `[1]` dropped flag.
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
            410: "top512 select", 411: "attn scores", 412: "softmax+attn", 413: "output gemv"},
    "ple": {500: "validate", 501: "ngram lookup", 510: "embed.publish", 511: "embed.exchange",
            513: "embed.reduce", 514: "embed.peer wait", 520: "key+value gemv", 521: "normalize",
            522: "gate", 523: "conv norm", 524: "short conv"},
    "entry": {600: "embed lookup", 610: "tp.publish", 611: "tp.exchange", 613: "tp.reduce",
              614: "tp.peer wait", 620: "expand branches"},
    "tail": {640: "mix.normalize", 641: "mix.down", 642: "mix.up", 643: "mix.branch", 644: "lm head gemv",
             630: "gather.publish", 631: "gather.exchange", 633: "gather.reduce", 634: "gather.peer wait"},
}


def kind_of(stage):
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

    def export(self, path, clock_khz, labels=None):
        records, dropped = self.read()
        clock_khz = clock_khz or int(torch.cuda.get_device_properties(self.device).clock_rate)
        return export_trace(records, path, clock_khz, self.rank, dropped, labels)


def launches(records):
    """Split the flat (stage, ticks) stream into launched kernels.

    A launch begins at its kernel's start stage; barriers inside a kernel never repeat that
    stage, so the split is unambiguous.
    """
    current = None
    for index in range(records.shape[0]):
        stage, ticks = int(records[index, 0]), int(records[index, 1])
        kind = kind_of(stage)
        if current is None or kind != current["kind"] or stage == START_STAGE.get(kind):
            if current is not None:
                yield current
            current = {"kind": kind, "marks": []}
        current["marks"].append((stage, ticks))
    if current is not None:
        yield current


def export_trace(records, path, clock_khz, rank, dropped=False, labels=None):
    """Write one rank's stages as a Perfetto (Chrome Trace) file: step -> layer kernel -> segment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [{"ph": "M", "name": "process_name", "pid": rank, "tid": 0, "args": {"name": f"rank {rank} (device stages)"}},
              {"ph": "M", "name": "thread_name", "pid": rank, "tid": 1, "args": {"name": "decode steps"}}]
    ticks_per_us = clock_khz / 1000.0
    base = None
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
            if layer_open is not None:
                events.append(_span(rank, 1, f"layer {layer} {layer_open['kind']}", layer_open["start"], first, base, ticks_per_us))
            layer += 1
            layer_open = {"kind": launch["kind"], "start": first}
        label = f"{launch['kind']}"
        if launch["kind"] == "post" and layer_open is not None:
            label = f"layer {layer} {layer_open['kind']}+post"
        events.append(_span(rank, 1, label, first, last, base, ticks_per_us, depth_extra=2))
        previous = first
        for stage, ticks in marks[1:]:
            events.append(_span(rank, 1, segment_name(launch["kind"], stage), previous, ticks, base, ticks_per_us, depth_extra=3))
            summary.append({"kind": launch["kind"], "stage": stage, "segment": segment_name(launch["kind"], stage),
                            "us": (ticks - previous) / ticks_per_us})
            previous = ticks
        if launch["kind"] == "tail":
            if layer_open is not None:
                events.append(_span(rank, 1, f"layer {layer} {layer_open['kind']}", layer_open["start"], first, base, ticks_per_us))
                layer_open = None
            if step_open is not None:
                events.append(_span(rank, 1, f"step {step}", step_open["start"], last, base, ticks_per_us))
                step_open = None
    events = [event for event in events if event is not None]
    text = '{"displayTimeUnit":"ms","traceEvents":[' + ",".join(json.dumps(event, separators=(",", ":")) for event in events) + "]}"
    path.write_text(text)
    return {"trace": str(path), "rank": rank, "steps": step, "events": len(events), "dropped": dropped, "segments": summary}


def _span(pid, tid, name, start, end, base, ticks_per_us, depth_extra=0):
    if end <= start:
        return None
    return {"ph": "X", "cat": "stage", "name": name, "pid": pid, "tid": tid,
            "ts": (start - base) / ticks_per_us, "dur": (end - start) / ticks_per_us,
            "args": {"tp_rank": pid, "depth": depth_extra}}
