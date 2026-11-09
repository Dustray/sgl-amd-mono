import argparse
import csv
import io
import json
from pathlib import Path
import subprocess


QUERY = """
WITH request_bounds AS (
    SELECT MIN(ts) AS started, MAX(ts + dur) AS finished FROM slice
    WHERE category = 'client' AND name GLOB 'request/*/HTTP_generate'
)
SELECT 'kernel' AS metric, COUNT(*) AS value FROM slice WHERE category = 'kernel'
UNION ALL SELECT 'cpu_op', COUNT(*) FROM slice WHERE category = 'cpu_op'
UNION ALL SELECT 'gpu_memcpy', COUNT(*) FROM slice WHERE category = 'gpu_memcpy'
UNION ALL SELECT 'decode', COUNT(*) FROM slice WHERE name GLOB 'inference/DECODE/*'
UNION ALL SELECT 'prefill', COUNT(*) FROM slice WHERE name GLOB 'inference/*' AND name NOT GLOB 'inference/DECODE/*'
UNION ALL SELECT 'tokenize', COUNT(*) FROM slice WHERE category = 'frontend' AND name GLOB '*._tokenize_texts'
UNION ALL SELECT 'detokenize', COUNT(*) FROM slice WHERE category = 'frontend' AND name GLOB '*._decode_batch_token_id_output'
UNION ALL SELECT 'client_tokens', COUNT(*) FROM slice WHERE category = 'client' AND name GLOB 'response/token.*'
UNION ALL SELECT 'gpu_processes', COUNT(DISTINCT thread.upid) FROM slice
JOIN thread_track ON thread_track.id = slice.track_id
JOIN thread ON thread.utid = thread_track.utid WHERE slice.category = 'kernel'
UNION ALL SELECT 'flow', COUNT(*) FROM flow
UNION ALL SELECT 'shape_arguments', COUNT(*) FROM args WHERE key GLOB '*Input Dims*'
UNION ALL SELECT 'forwards_outside_http', COUNT(*) FROM slice, request_bounds
WHERE name GLOB 'inference/*' AND (ts < started OR ts + dur > finished)
UNION ALL SELECT 'frontend_outside_http', COUNT(*) FROM slice, request_bounds
WHERE category = 'frontend' AND (ts < started OR ts + dur > finished)
UNION ALL SELECT 'health/' || name, value FROM stats
WHERE value > 0 AND severity IN ('error', 'data_loss');
"""


def validate(directory, processor):
    summary = json.loads((directory / "summary.json").read_text())
    command = [processor, "query", "--quiet", "--format", "csv", summary["trace"], QUERY]
    result = subprocess.run(command, text=True, capture_output=True, timeout=1800)
    (directory / "perfetto-validation.csv").write_text(result.stdout)
    (directory / "perfetto-validation.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"Perfetto import/query failed: {result.stderr}")
    metrics = {row["metric"]: int(row["value"]) for row in csv.DictReader(io.StringIO(result.stdout))}
    expected = {category: sum(rank["categories"].get(category, 0) for rank in summary["ranks"]) for category in ("kernel", "cpu_op", "gpu_memcpy")}
    expected["decode"] = sum(rank["phases"]["DECODE"]["forwards"] for rank in summary["ranks"])
    expected["prefill"] = sum(value["forwards"] for rank in summary["ranks"] for phase, value in rank["phases"].items() if phase != "DECODE")
    expected["gpu_processes"] = len(summary["ranks"])
    expected["client_tokens"] = summary["output_tokens"] * summary["batch_size"]
    expected["forwards_outside_http"] = 0
    expected["frontend_outside_http"] = 0
    failures = {key: {"actual": metrics.get(key), "expected": value} for key, value in expected.items() if metrics.get(key) != value}
    for key in ("tokenize", "detokenize", "flow", "shape_arguments"):
        if not metrics.get(key, 0):
            failures[key] = "missing"
    health = {key.removeprefix("health/"): value for key, value in metrics.items() if key.startswith("health/")}
    unexpected_health = {key: value for key, value in health.items() if key != "slice_spill_overlapping_complete_event"}
    if unexpected_health:
        failures["unexpected_health"] = unexpected_health
    version = subprocess.check_output([processor, "--version"], text=True).strip()
    report = {"passed": not failures, "processor_version": version, "metrics": metrics, "expected": expected, "failures": failures, "health": health, "health_note": "slice_spill_overlapping_complete_event retains overlapping slices on overflow tracks; also present in unmerged DTK traces."}
    (directory / "perfetto-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{directory.name}: Perfetto verified={report['passed']}; GPU kernels={metrics.get('kernel')}; detokenization ranges={metrics.get('detokenize')}", flush=True)
    if failures:
        raise RuntimeError(f"Perfetto coverage validation failed: {failures}")


def main():
    parser = argparse.ArgumentParser(description="Import saved inference profiles into real Perfetto and verify event coverage")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trace-processor", default="trace_processor_shell")
    arguments = parser.parse_args()
    root = arguments.run_dir.resolve()
    directories = [root] if (root / "summary.json").exists() else [root / mode for mode in ("native", "monokernel") if (root / mode / "summary.json").exists()]
    if not directories:
        parser.error("No completed profile summary found")
    for directory in directories:
        validate(directory, arguments.trace_processor)


if __name__ == "__main__":
    main()
