"""Shared GPU occupancy checks for the gfx936 host.

The eight cards are shared with other users' jobs, whose processes are not visible in this
namespace (they show up only as PIDs under /sys/class/kfd/kfd/proc). Every layer of a TP8 step
does two all-reduces, so one card busy with someone else's work stalls all eight ranks: the
server keeps making progress but at a fraction of its speed, with seven GPUs idle in barriers.
Measure nothing until every card is quiet.
"""

import subprocess
import time


def gpu_usage(samples=3, interval=2.0):
    """Per-card (VRAM %, compute %) from hy-smi, each the maximum over a few samples."""
    peak = {}
    for sample in range(samples):
        if sample:
            time.sleep(interval)
        try:
            output = subprocess.run(["hy-smi"], capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        for line in output.splitlines():
            fields = line.split()
            if len(fields) < 7 or not fields[0].isdigit():
                continue
            card = int(fields[0])
            try:
                memory, compute = float(fields[5].rstrip("%")), float(fields[6].rstrip("%"))
            except ValueError:
                continue
            previous = peak.get(card, (0.0, 0.0))
            peak[card] = (max(previous[0], memory), max(previous[1], compute))
    return peak or None


def check_gpus_idle(require):
    """This host is shared. A single card busy with someone else's job stalls every TP8 collective,
    so all eight ranks crawl while seven GPUs look idle."""
    usage = gpu_usage()
    if usage is None:
        print("could not read hy-smi; skipping the GPU idle check", flush=True)
        return None
    busy = {card: values for card, values in usage.items() if values[0] > 5 or values[1] > 10}
    if busy:
        detail = ", ".join(f"card {card}: VRAM {values[0]:.0f}%, compute {values[1]:.0f}%" for card, values in sorted(busy.items()))
        message = f"GPUs are not idle before this run ({detail}); another job on any card slows every rank"
        if require:
            raise RuntimeError(message + ". Wait for it to finish or pass --allow-busy-gpus")
        print(f"WARNING: {message}", flush=True)
    return {str(card): {"vram_percent": values[0], "compute_percent": values[1]} for card, values in sorted(usage.items())}
