import argparse
import hashlib
import json
from pathlib import Path

from sgl_amd_mono.common.profile_trace import merge_profiles


def main():
    parser = argparse.ArgumentParser(description="Rebuild full inference traces and operator tables from saved per-rank profiles")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--keep-python-functions", action="store_true")
    arguments = parser.parse_args()
    root = arguments.run_dir.resolve()
    directories = [root] if (root / "capture.json").exists() else [root / mode for mode in ("native", "monokernel") if (root / mode / "capture.json").exists()]
    if not directories:
        parser.error("No captured profile found")
    for directory in directories:
        capture = json.loads((directory / "capture.json").read_text())
        client_path = directory / "host/client.json"
        if not client_path.exists():
            raise FileNotFoundError(f"Missing original client timeline: {client_path}")
        clients = json.loads(client_path.read_text())
        batch_size = len(capture["inputs"])
        summary_path = directory / "summary.json"
        previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        summary = merge_profiles(list((directory / "ranks").glob("*TP-*.trace.json.gz")), directory / "inference.perfetto.json.gz", directory / "host", clients, expected_decode_steps=capture["output_tokens"] - 1 if batch_size == 1 else None, include_python_functions=arguments.keep_python_functions, expected_decode_token_rows=batch_size * (capture["output_tokens"] - 1))
        for key in ("shadow_state_before", "shadow_state_after", "profile_is_accuracy_acceptance", "prompt_tokens", "output_tokens", "batch_size"):
            if key in previous:
                summary[key] = previous[key]
        summary.update(prompt_tokens=len(capture["inputs"][0]["input_ids"]), output_tokens=capture["output_tokens"], batch_size=batch_size, profile_is_accuracy_acceptance=False)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        source = Path(__file__).resolve().parents[1] / "sgl_amd_mono/common/profile_trace.py"
        manifest = root / "manifest.json" if (root / "manifest.json").exists() else root.parent / "manifest.json"
        (directory / "postprocess.json").write_text(json.dumps({"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "include_python_functions": arguments.keep_python_functions, "capture_source_manifest": str(manifest)}, indent=2) + "\n")
        print(f"{directory.name}: {summary['total_events']} events -> {summary['trace']}", flush=True)


if __name__ == "__main__":
    main()
