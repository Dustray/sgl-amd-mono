import argparse
import json
from pathlib import Path

import torch

from sgl_amd_mono.qwen38_flash_next.gfx936.runner import error_metrics, logits_accepted, logits_metrics


def compare(native, fused):
    assert native["source_sha256"] == fused["source_sha256"], "Sources changed between services"
    assert native["profile"] == fused["profile"], "Service configurations differ"
    assert native["accuracy_reference"] == fused["accuracy_reference"]
    assert native["accuracy_reference"] and native["accuracy_reference"]["enabled"]
    assert native["cases_sha256"] == fused["cases_sha256"]
    assert len(native["cases"]) == len(fused["cases"])
    native_root, fused_root = Path(native["trace_directory"]), Path(fused["trace_directory"])
    native_index = json.loads((native_root / "index.json").read_text())
    fused_index = json.loads((fused_root / "index.json").read_text())
    assert native_index["source_sha256"] == fused_index["source_sha256"]
    rows = []
    for baseline, actual in zip(native["cases"], fused["cases"]):
        assert baseline["case"] == actual["case"]
        baseline_frames = native_index["frames"][baseline["trace_start"]:baseline["trace_end"]]
        actual_frames = fused_index["frames"][actual["trace_start"]:actual["trace_end"]]
        inputs = baseline["case"]["input_ids"]
        batched = bool(inputs) and isinstance(inputs[0], list)
        row = {"label": baseline["case"]["label"], "input_tokens": len(inputs[0]) if batched else len(inputs), "batch_size": len(inputs) if batched else 1, "native_completed": baseline["completed"], "fused_completed": actual["completed"], "output_ids_equal": baseline["response"]["output_ids"] == actual["response"]["output_ids"], "frame_count_equal": len(baseline_frames) == len(actual_frames), "frames": []}
        row.update(native_frame_count=len(baseline_frames), fused_frame_count=len(actual_frames))
        for expected_frame, actual_frame in zip(baseline_frames, actual_frames):
            metadata_equal = all(expected_frame[key] == actual_frame[key] for key in ("mode", "input_ids", "seq_lens", "rows"))
            prefix_equal = metadata_equal
            if metadata_equal and actual_frame["mode"] == "DECODE":
                prompts = inputs if batched else [inputs]
                native_outputs = baseline["response"]["output_ids"] if batched else [baseline["response"]["output_ids"]]
                fused_outputs = actual["response"]["output_ids"] if batched else [actual["response"]["output_ids"]]
                prefix_equal = len(prompts) == actual_frame["rows"] and all(native_output[:length - len(prompt)] == fused_output[:length - len(prompt)] for prompt, length, native_output, fused_output in zip(prompts, actual_frame["seq_lens"], native_outputs, fused_outputs))
            same_inputs = metadata_equal and prefix_equal
            check = {"mode": actual_frame["mode"], "seq_lens": actual_frame["seq_lens"], "metadata_equal": metadata_equal, "prefix_equal": prefix_equal, "same_inputs": same_inputs, "accepted": False}
            if same_inputs:
                expected = torch.load(native_root / expected_frame["filename"], map_location="cpu", weights_only=True)["logits"]
                observed = torch.load(fused_root / actual_frame["filename"], map_location="cpu", weights_only=True)["logits"]
                check["logits"] = error_metrics(observed, expected)
                check["distribution"] = logits_metrics(observed, expected)
                check["accepted"] = check["logits"]["finite"] and logits_accepted(check["distribution"])
            row["frames"].append(check)
        row["passed"] = row["native_completed"] and row["fused_completed"] and row["output_ids_equal"] and row["frame_count_equal"] and bool(row["frames"]) and all(check["accepted"] for check in row["frames"]) and baseline.get("answer_passed") is not False and actual.get("answer_passed") is not False
        rows.append(row)
    return {"passed": native["passed"] and fused["passed"] and all(row["passed"] for row in rows), "scope": "Independent services accumulate their own caches; actual outer-forward logits compared at equal input tokens and lengths, including prefill. Cache tensors are not directly compared here; fused same-state shadows check cache updates separately.", "accuracy_reference": native["accuracy_reference"], "numeric_policy": {"kl_reference_actual": 0.01, "total_variation": 0.05, "greedy_argmax_equal": True}, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    report = compare(json.loads(arguments.native.read_text()), json.loads(arguments.fused.read_text()))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "rows": [{key: value for key, value in row.items() if key != "frames"} for row in report["rows"]]}, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
