"""Device check for the per-CU instrumentation: run `hc_mix` on one GPU and verify every
compute unit shows up with a sane mark stream.

No model weights and no TP group are needed — `hc_mix` is self-contained — so this is the cheap
way to confirm `cu_mark()`/`HW_ID` work on real hardware after touching `kernel.cpp`.

    python3 tools/verify_flash_next_cu_timing.py --device 0 --launches 4
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sgl_amd_mono.qwen38_flash_next.gfx936.cu_timing import CUTiming, cu_key, cu_label, decode_hardware_id
from sgl_amd_mono.qwen38_flash_next.gfx936.program import HCMix
from sgl_amd_mono.qwen38_flash_next.gfx936.stage_timing import START_STAGE, SYNC_ARRIVAL_OFFSET


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--launches", type=int, default=4)
    parser.add_argument("--rows", type=int, choices=(1, 2), default=1)
    parser.add_argument("--records", type=int, default=1 << 10, help="per-block record capacity")
    parser.add_argument("--output-dir", type=Path, help="write the Perfetto trace and report here")
    arguments = parser.parse_args()

    device = torch.device(f"cuda:{arguments.device}")
    torch.manual_seed(42)
    weights = (torch.randn(10240, dtype=torch.bfloat16, device=device) * 0.02,
               torch.randn(320, 10240, dtype=torch.bfloat16, device=device) * 0.02,
               torch.randn(10240, 320, dtype=torch.bfloat16, device=device) * 0.02)
    mix = HCMix(*weights)
    blocks = int(mix.program.manifest.get("blocks", 80))
    timing = CUTiming(device, rank=arguments.device, records=arguments.records, blocks=blocks)
    timing.attach(mix.program)
    hidden = torch.randn(arguments.rows, 10240, dtype=torch.bfloat16, device=device) * 0.05

    mix(hidden)  # warm up (first launch pays the kernel load)
    torch.cuda.synchronize(device)
    timing.reset()
    for _ in range(arguments.launches):
        mix(hidden)
    torch.cuda.synchronize(device)

    per_block, dropped = timing.read()
    failures = []
    if dropped:
        failures.append(f"records dropped: raise --records above {arguments.records}")
    recorded = [index for index, records in enumerate(per_block) if records.shape[0]]
    if len(recorded) != blocks:
        failures.append(f"{len(recorded)} of {blocks} blocks recorded marks")
    # hc_mix: kernel entry, then (arrival, release) for barriers 1..4, per launch.
    marks_per_launch = 9
    expected = [START_STAGE["hc_mix"]] + [stage for barrier in (1, 2, 3, 4) for stage in (barrier + SYNC_ARRIVAL_OFFSET, barrier)]
    placement = [{} for _ in range(arguments.launches)]
    for block in recorded:
        records = per_block[block]
        stages = [int(value) for value in records[:, 0].tolist()]
        ticks = [int(value) for value in records[:, 1].tolist()]
        if stages != expected * arguments.launches:
            failures.append(f"block {block} stage sequence {stages[:12]}... does not match {arguments.launches} hc_mix launches")
            continue
        if any(later < earlier for earlier, later in zip(ticks, ticks[1:])):
            failures.append(f"block {block} timestamps are not monotonic")
        for launch in range(arguments.launches):
            window = records[launch * marks_per_launch:(launch + 1) * marks_per_launch, 2].tolist()
            keys = {cu_key(value) for value in window}
            if len(keys) != 1:
                # A block must stay on one CU for the duration of a launch, or a span cannot be
                # attributed to a CU at all.
                failures.append(f"block {block} moved between compute units inside launch {launch}: "
                                f"{sorted(cu_label(key) for key in keys)}")
            placement[launch].setdefault(sorted(keys)[0], []).append(block)

    # Across launches a block is rescheduled freely, so only within-launch placement is meaningful.
    occupancy = []
    for launch, units in enumerate(placement):
        if not units:
            continue
        crowded = {cu_label(key): blocks_on_unit for key, blocks_on_unit in units.items() if len(blocks_on_unit) > 1}
        occupancy.append({"launch": launch, "compute_units_used": len(units), "idle_compute_units": blocks - len(units),
                          "max_blocks_per_unit": max(len(blocks_on_unit) for blocks_on_unit in units.values()),
                          "units_with_several_blocks": len(crowded), "example": dict(list(crowded.items())[:3])})
    stable = {cu_label(key) for units in placement for key in units}

    clock_khz = mix.program.information["clock_khz"]
    directory = arguments.output_dir or Path("bench_results") / f"cu_timing_device_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}_dev{arguments.device}_r{arguments.rows}"
    directory.mkdir(parents=True, exist_ok=True)
    exported = timing.export(directory / "cu-timing-hcmix-rank-0.json", clock_khz)
    busiest = sorted(exported["summary"], key=lambda row: -(row["compute_us"] + row["wait_us"]))[:3]
    report = {"device": arguments.device, "blocks": blocks, "launches": arguments.launches, "rows": arguments.rows,
              "mix_occupancy_blocks_per_unit": mix.program.information["mix_occupancy"],
              "compute_units_seen": len(stable), "per_launch_occupancy": occupancy, "dropped": dropped,
              "marks_per_block": int(per_block[recorded[0]].shape[0]) if recorded else 0,
              "hardware_id_sample": decode_hardware_id(per_block[recorded[0]][0, 2]) if recorded else None,
              "compute_unit_labels": sorted(stable),
              "busiest_units": busiest, "trace": exported["trace"], "failures": failures}
    (directory / "cu-timing-device-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "compute_unit_labels"}, indent=2))
    print(f"compute units seen: {len(stable)} -> {', '.join(report['compute_unit_labels'][:8])}{' ...' if len(stable) > 8 else ''}")
    if failures:
        raise SystemExit(f"{len(failures)} per-CU timing check(s) failed")
    print(f"OK: {blocks} blocks, {report['marks_per_block']} marks each, "
          f"{occupancy[0]['compute_units_used']} compute units used per launch")


if __name__ == "__main__":
    main()
