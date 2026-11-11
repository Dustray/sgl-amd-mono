from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import re

try:
    import orjson
except ImportError:
    orjson = None


def read_trace(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as source:
        trace = orjson.loads(source.read()) if orjson is not None else json.load(source)
    if not isinstance(trace, dict) or not isinstance(trace.get("traceEvents"), list):
        raise ValueError(f"Invalid Chrome trace: {path}")
    return trace


def _operator_rows(events, rank):
    rows = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset", "cpu_op", "user_annotation", "mono_kernel", "mono_kernel_stage"):
            rows[(event["cat"], event["name"])].append(event.get("dur", 0))
    return [{"rank": rank, "category": category, "name": name, "count": len(values), "total_us": sum(values), "mean_us": sum(values) / len(values), "min_us": min(values), "max_us": max(values)} for (category, name), values in rows.items()]


def _phase_summary(events):
    forwards = sorted((event for event in events if event.get("ph") == "X" and event.get("cat") == "user_annotation" and event.get("name", "").startswith("inference/")), key=lambda event: event["ts"])
    starts = [event["ts"] for event in forwards]
    runtime = {event.get("args", {}).get("correlation"): event for event in events if event.get("ph") == "X" and "runtime" in event.get("cat", "") and "correlation" in event.get("args", {})}
    external = {event.get("args", {}).get("External id"): event for event in events if event.get("cat") == "cpu_op" and "External id" in event.get("args", {})}
    summary = defaultdict(lambda: {"forwards": 0, "batch_rows": 0, "host_forward_inclusive_us": 0, "gpu_kernel_us": 0, "gpu_kernels": 0})
    for event in forwards:
        phase = event["name"].split("/")[1]
        summary[phase]["forwards"] += 1
        match = re.search(r"/batch=(\d+)", event["name"])
        summary[phase]["batch_rows"] += int(match.group(1)) if match else 0
        summary[phase]["host_forward_inclusive_us"] += event["dur"]
    unmatched = 0
    for event in events:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        arguments = event.get("args", {})
        anchor = runtime.get(arguments.get("correlation")) or external.get(arguments.get("External id"))
        index = bisect_right(starts, anchor["ts"]) - 1 if anchor else -1
        if index < 0 or anchor["ts"] > forwards[index]["ts"] + forwards[index]["dur"]:
            unmatched += 1
            continue
        phase = forwards[index]["name"].split("/")[1]
        summary[phase]["gpu_kernel_us"] += event["dur"]
        summary[phase]["gpu_kernels"] += 1
    return dict(summary), unmatched


_MONO_KERNEL_PREFIXES = {
    "entry": "decode_entry(",
    "gdn": "gdn_attention(",
    "qsa": "qsa_attention(",
    "ple": "ple_decode(",
    "post": "post_attention(",
    "tail": "decode_tail(",
}


def _stage_launch_kind(name):
    if name in _MONO_KERNEL_PREFIXES:
        return name
    if name.endswith("+post"):
        return "post"
    return None


def _embed_device_stage_events(events, stage_trace, rank):
    """Align one device-stage decode step with its GPU kernels and return Perfetto spans."""
    stage_spans = [event for event in stage_trace.get("traceEvents", [])
                   if event.get("ph") == "X" and event.get("cat") == "stage"]
    launch_spans = sorted(
        (event for event in stage_spans if event.get("args", {}).get("depth") == 2),
        key=lambda event: event["ts"],
    )
    step_spans = [event for event in stage_spans if event.get("name", "").startswith("step ")]
    if not step_spans or not launch_spans:
        raise ValueError(f"Device stage trace for rank {rank} must contain complete decode steps")
    launch_kinds = [_stage_launch_kind(event["name"]) for event in launch_spans]
    if any(kind is None for kind in launch_kinds):
        raise ValueError(f"Unknown device-stage launch label for rank {rank}")

    fused_kernels = sorted(
        (event for event in events if event.get("ph") == "X" and event.get("cat") == "kernel"
         and any(event.get("name", "").startswith(prefix) for prefix in _MONO_KERNEL_PREFIXES.values())),
        key=lambda event: event["ts"],
    )
    kernel_kinds = []
    for event in fused_kernels:
        kernel_kinds.append(next(kind for kind, prefix in _MONO_KERNEL_PREFIXES.items()
                                 if event.get("name", "").startswith(prefix)))
    launch_count = len(launch_kinds)
    match_start = next(
        (index for index in range(len(kernel_kinds) - launch_count + 1)
         if kernel_kinds[index:index + launch_count] == launch_kinds),
        None,
    )
    if match_start is None:
        expected = Counter(launch_kinds)
        actual = Counter(kernel_kinds)
        raise ValueError(f"Cannot align rank {rank} device stages to torch GPU kernels: "
                         f"stage launches={dict(expected)}, profiled kernels={dict(actual)}")

    pairs = list(zip(launch_spans, fused_kernels[match_start:match_start + launch_count]))
    first_stage, first_kernel = pairs[0]
    global_offset = first_kernel["ts"] - first_stage["ts"]
    embedded = []
    launch_contexts = {}
    layer = -1
    for launch, _ in pairs:
        kind = _stage_launch_kind(launch["name"])
        if kind == "entry":
            layer = -1
            launch_contexts[id(launch)] = (None, launch["name"])
        elif kind in ("gdn", "qsa"):
            layer += 1
            launch_contexts[id(launch)] = (layer, kind)
        elif kind == "post" and layer >= 0:
            launch_contexts[id(launch)] = (layer, "post")
        else:
            launch_contexts[id(launch)] = (None, launch["name"])
    max_launch_start_error = 0.0
    for stage_launch, kernel in pairs:
        max_launch_start_error = max(
            max_launch_start_error,
            abs((kernel["ts"] - first_kernel["ts"]) - (stage_launch["ts"] - first_stage["ts"])),
        )
    for original in stage_spans:
        depth = original.get("args", {}).get("depth", 0)
        parent = None
        if depth == 2:
            parent = next((pair for pair in pairs if pair[0] is original), None)
            if parent is None:
                raise ValueError(f"Unmatched device-stage launch for rank {rank}")
        elif depth == 3:
            parent = next((pair for pair in pairs
                           if pair[0]["ts"] <= original["ts"] <= pair[0]["ts"] + pair[0]["dur"]), None)

        event = dict(original)
        event["cat"] = "mono_kernel_stage" if depth == 3 else "mono_kernel"
        if depth == 3 and parent is not None:
            layer_index, module = launch_contexts[id(parent[0])]
            event["name"] = f"layer {layer_index:02d} {module} / {original['name']}" if layer_index is not None else f"{module} / {original['name']}"
        event["args"] = {**original.get("args", {}), "tp_rank": rank}
        if parent is None:
            event["ts"] += global_offset
            anchor = first_kernel
        elif depth == 2:
            event["ts"] = parent[1]["ts"]
            event["dur"] = parent[1]["dur"]
            anchor = parent[1]
        else:
            event["ts"] = parent[1]["ts"] + original["ts"] - parent[0]["ts"]
            anchor = parent[1]
        event["pid"] = anchor["pid"]
        event["tid"] = anchor["tid"]
        if "correlation" in anchor.get("args", {}):
            event["args"]["kernel_correlation"] = anchor["args"]["correlation"]
        embedded.append(event)

    return embedded, {"steps": len(step_spans), "launches": launch_count, "spans": len(embedded), "pid": first_kernel["pid"],
                      "tid": first_kernel["tid"],
                      "max_launch_start_error_us": max_launch_start_error}


def merge_profiles(trace_paths, output, host_directory=None, client_events=(), expected_ranks=8, expected_decode_steps=None, include_python_functions=False, expected_decode_token_rows=None, device_stage_directory=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    identities = {}
    flow_ids = {}
    rank_summaries = []
    operators = []
    seen_ranks = set()
    common_base = None
    first_event = True
    total_events = 0
    host_names = Counter()
    stage_capture = {"ranks": [], "steps": 0, "launches": 0, "spans": 0, "max_launch_start_error_us": 0.0}

    def identity(kind, rank, value):
        if kind == "pid":
            value = str(value)
        key = (kind, rank, value)
        if key not in identities:
            identities[key] = len(identities) + 100
        return identities[key]

    def flow(rank, category, scope, value):
        key = (rank, category, scope, value)
        if key not in flow_ids:
            flow_ids[key] = len(flow_ids) + 1
        return flow_ids[key]

    with gzip.open(temporary, "wb", compresslevel=1) as destination:
        destination.write(b'{"displayTimeUnit":"ms","traceEvents":[')
        buffer = []

        def flush():
            nonlocal first_event
            if not buffer:
                return
            if not first_event:
                destination.write(b",")
            destination.write(b",".join(buffer))
            first_event = False
            buffer.clear()

        def emit(event):
            nonlocal total_events
            buffer.append(orjson.dumps(event) if orjson is not None else json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode())
            if len(buffer) >= 10000:
                flush()
            total_events += 1

        for path in sorted(map(Path, trace_paths)):
            match = re.search(r"TP-(\d+)", path.name)
            if not match:
                raise ValueError(f"Missing TP rank in filename: {path}")
            rank = int(match.group(1))
            if rank in seen_ranks:
                raise ValueError(f"Duplicate TP rank: {rank}")
            seen_ranks.add(rank)
            trace = read_trace(path)
            base = trace.get("baseTimeNanoseconds")
            if not isinstance(base, int):
                raise ValueError(f"Trace has no wall-clock baseTimeNanoseconds: {path}")
            if common_base is None:
                common_base = base
            events = trace["traceEvents"]
            categories = Counter(event.get("cat", "metadata") for event in events)
            if not categories["kernel"]:
                raise ValueError(f"No GPU kernel events in rank {rank}: {path}")
            phases, unmatched = _phase_summary(events)
            if not any(phase != "DECODE" for phase in phases) or not phases.get("DECODE"):
                raise ValueError(f"Incomplete prefill/decode model ranges for rank {rank}")
            if expected_decode_steps is not None and phases["DECODE"]["forwards"] != expected_decode_steps:
                raise ValueError(f"Rank {rank} decode coverage mismatch: {phases['DECODE']['forwards']} != {expected_decode_steps}")
            if expected_decode_token_rows is not None and phases["DECODE"]["batch_rows"] != expected_decode_token_rows:
                raise ValueError(f"Rank {rank} decode token coverage mismatch: {phases['DECODE']['batch_rows']} != {expected_decode_token_rows}")
            rank_summaries.append({"rank": rank, "source": str(path), "events": len(events), "categories": dict(categories), "phases": phases, "unattributed_gpu_kernels": unmatched, "python_function_events_filtered": 0 if include_python_functions else categories["python_function"]})
            operators.extend(_operator_rows(events, rank))
            process_names = {str(event.get("pid")): event.get("args", {}).get("name") for event in events if event.get("name") == "process_name"}
            for process in dict.fromkeys(str(event.get("pid", 0)) for event in events):
                emit({"ph": "M", "name": "process_name", "pid": identity("pid", rank, process), "tid": 0, "args": {"name": f"TP{rank} / {process_names.get(process) or process}"}})
            for original in events:
                if not include_python_functions and original.get("cat") == "python_function":
                    continue
                event = dict(original)
                if event.get("ph") == "M" and event.get("name") == "process_name":
                    continue
                process = event.get("pid", 0)
                event["pid"] = identity("pid", rank, process)
                event["tid"] = identity("tid", rank, (process, event.get("tid", 0)))
                if "ts" in event:
                    event["ts"] += (base - common_base) / 1000
                for key in ("id", "bind_id"):
                    if key in event:
                        event[key] = flow(rank, event.get("cat"), event.get("scope"), event[key])
                if "id2" in event:
                    event["id2"] = {key: flow(rank, event.get("cat"), event.get("scope"), value) for key, value in event["id2"].items()}
                if event.get("ph") == "X":
                    event["args"] = {**event.get("args", {}), "tp_rank": rank}
                emit(event)
            if device_stage_directory is not None:
                stage_path = Path(device_stage_directory) / f"stage-timing-decode-rank-{rank}.json"
                if not stage_path.is_file():
                    raise ValueError(f"Missing device stage trace for TP rank {rank}: {stage_path}")
                stage_trace = read_trace(stage_path)
                embedded, stage_summary = _embed_device_stage_events(events, stage_trace, rank)
                stage_pid = identity("pid", rank, stage_summary["pid"])
                stage_tid = identity("tid", rank, (stage_summary["pid"], "monokernel-device-stages"))
                emit({"ph": "M", "name": "thread_name", "pid": stage_pid, "tid": stage_tid,
                      "args": {"name": f"Monokernel internal stages / TP{rank} / device clock"}})
                for event in embedded:
                    process = event["pid"]
                    event["pid"] = identity("pid", rank, process)
                    event["tid"] = stage_tid
                    event["ts"] += (base - common_base) / 1000
                    emit(event)
                stage_capture["ranks"].append(rank)
                stage_capture["steps"] = max(stage_capture["steps"], stage_summary["steps"])
                stage_capture["launches"] += stage_summary["launches"]
                stage_capture["spans"] += stage_summary["spans"]
                stage_capture["max_launch_start_error_us"] = max(
                    stage_capture["max_launch_start_error_us"], stage_summary["max_launch_start_error_us"])
                operators.extend(_operator_rows(embedded, rank))
        if seen_ranks != set(range(expected_ranks)):
            raise ValueError(f"Expected TP ranks 0..{expected_ranks - 1}, found {sorted(seen_ranks)}")
        host_events = list(client_events)
        if host_directory is not None:
            for path in sorted(Path(host_directory).glob("host-*.jsonl")):
                with path.open() as source:
                    host_events.extend(json.loads(line) for line in source if line.strip())
        for process in dict.fromkeys(event["pid"] for event in host_events):
            emit({"ph": "M", "name": "process_name", "pid": identity("pid", "host", process), "tid": 0, "args": {"name": f"Request / frontend / PID {process}"}})
        for original in host_events:
            event = dict(original)
            event["pid"] = identity("pid", "host", original["pid"])
            event["tid"] = identity("tid", "host", (original["pid"], original.get("tid", 0)))
            event["ts"] -= common_base / 1000
            host_names[event["name"]] += 1
            emit(event)
        flush()
        destination.write(('],"baseTimeNanoseconds":' + str(common_base) + "}").encode())
    temporary.replace(output)
    operators.sort(key=lambda row: (row["rank"], row["category"], -row["total_us"]))
    csv_path = output.with_name("operators.csv")
    with csv_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["rank", "category", "name", "count", "total_us", "mean_us", "min_us", "max_us"])
        writer.writeheader()
        writer.writerows(operators)
    timing_note = "GPU totals sum kernel durations and can overlap; CPU and module durations are inclusive. Embedded TP peer waits remain inside fused kernel durations."
    if device_stage_directory is not None:
        timing_note += " mono_kernel_stage spans are device wall-clock measurements aligned to the corresponding torch GPU kernel launch."
    summary = {"format": "Chrome Trace Event JSON gzip; Perfetto compatible", "trace": str(output), "operators_csv": str(csv_path), "total_events": total_events, "python_functions_in_merged_trace": include_python_functions, "ranks": rank_summaries, "host_ranges": dict(host_names), "device_kernel_stages": stage_capture if device_stage_directory is not None else None, "timing_note": timing_note, "top_gpu_kernels_rank0": [row for row in operators if row["rank"] == 0 and row["category"] == "kernel"][:30]}
    output.with_name("summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary
