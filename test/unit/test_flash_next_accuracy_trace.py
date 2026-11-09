import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sgl_amd_mono.qwen38_flash_next.gfx936.accuracy_trace import AccuracyTrace
from tools.compare_flash_next_accuracy_trace import compare
from sgl_amd_mono.qwen38_flash_next.gfx936.program import FlashContractError
from sgl_amd_mono.qwen38_flash_next.gfx936.runner import FlashRuntime


def test_trace_preserves_logits_and_binds_input_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_FLASH_ACCURACY_TRACE_DIR", str(tmp_path))
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    result = SimpleNamespace(next_token_logits=logits)
    model = SimpleNamespace(forward=lambda **kwargs: result)
    runner = SimpleNamespace(ps=SimpleNamespace(tp_rank=0), model=model)
    trace = AccuracyTrace(runner)
    batch = SimpleNamespace(input_ids=torch.tensor([42]), seq_lens=torch.tensor([513]), forward_mode=SimpleNamespace(name="DECODE"))
    assert model.forward(forward_batch=batch) is result
    logits.zero_()
    index = json.loads((tmp_path / "index.json").read_text())
    assert len(index["frames"]) == 1
    frame = index["frames"][0]
    assert frame["input_ids"] == [42] and frame["seq_lens"] == [513]
    saved = torch.load(tmp_path / frame["filename"], weights_only=True)
    assert torch.equal(saved["logits"], torch.tensor([[1.0, 2.0, 3.0]]))
    assert trace.snapshot()["completed_frames"] == 1
    with pytest.raises(RuntimeError, match="already exists"):
        AccuracyTrace(runner)


@pytest.mark.parametrize("different_inputs,bad_logits,different_history,expected", [(False, False, False, True), (True, False, False, False), (False, True, False, False), (False, False, True, False)])
def test_independent_trace_comparison_requires_equal_prefix_and_distribution(tmp_path, different_inputs, bad_logits, different_history, expected):
    reports = []
    for mode in ("native", "fused"):
        directory = tmp_path / mode
        directory.mkdir()
        inputs = [8] if mode == "fused" and different_inputs else [7]
        logits = torch.tensor([[9.0, 2.0, 3.0]]) if mode == "fused" and bad_logits else torch.tensor([[1.0, 2.0, 3.0]])
        torch.save({"logits": logits}, directory / "frame.pt")
        frame = {"mode": "DECODE", "input_ids": inputs, "seq_lens": [12], "rows": 1, "filename": "frame.pt"}
        (directory / "index.json").write_text(json.dumps({"source_sha256": "trace-code", "frames": [frame]}))
        outputs = [0, 2] if mode == "fused" and different_history else [7, 2]
        reports.append({"passed": True, "source_sha256": {"kernel": "same"}, "profile": {"tp_size": 8}, "accuracy_reference": {"enabled": True, "policy": "fixed"}, "trace_directory": str(directory), "cases_sha256": "same", "cases": [{"case": {"label": "case", "input_ids": [1, 2, 3]}, "completed": True, "response": {"output_ids": outputs}, "trace_start": 0, "trace_end": 1}]})
    report = compare(*reports)
    assert report["passed"] is expected
    if different_inputs:
        assert not report["rows"][0]["frames"][0]["same_inputs"]
    if different_history:
        assert report["rows"][0]["frames"][0]["metadata_equal"]
        assert not report["rows"][0]["frames"][0]["prefix_equal"]


@pytest.mark.parametrize("diagnostic", [False, True])
def test_shadow_failure_is_latched_even_in_diagnostic_mode(diagnostic):
    runtime = FlashRuntime.__new__(FlashRuntime)
    runtime.completed_decode_steps = 3
    runtime.shadow_record_failures = diagnostic
    runtime.serving_shadow_failures = []
    runtime.save = lambda: None
    if diagnostic:
        runtime._shadow_failure("accuracy failure")
    else:
        with pytest.raises(FlashContractError, match="accuracy failure"):
            runtime._shadow_failure("accuracy failure")
    assert runtime.serving_shadow_failure == "accuracy failure"
    assert runtime.serving_shadow_failures == [{"step": 3, "reason": "accuracy failure"}]


@pytest.mark.parametrize("local_failure,repeat_error", [(True, 0.0), (False, 0.1)])
def test_diagnostic_audit_never_accepts_local_or_repeat_failures(local_failure, repeat_error, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools"))
    from verify_flash_next_correctness import audit

    keys = [f"gdn.{layer}" if layer % 4 != 3 else f"qsa.{layer}" for layer in range(48)] + [f"post.{layer}" for layer in range(48)] + ["ple.1", "embedding.0", "tail.48"]
    check = {"accepted": True, "input_token_ids": [7], "sequence_lengths": [3], "restored_state_tensors": 123, "distribution": {"finite": True, "argmax_equal": [True], "kl_reference_actual": [0.0], "total_variation": [0.0], "actual_token_ids": [8]}, "native_repeats": [{"logits": {"finite": True, "max_absolute_error": repeat_error}}] * 3}
    before = [{"completed_decode_steps": 0, "full_model_shadow_checks": [], "executions": dict.fromkeys(keys, 0)}] * 8
    after = [{"tp_rank": rank, "completed_decode_steps": 1, "full_model_shadow_checks": [check], "executions": dict.fromkeys(keys, 1), "serving_shadow_failures": [{"reason": "local"}] if local_failure else []} for rank in range(8)]
    case = {"label": "case", "input_ids": [1, 2]}
    result = {"output_ids": [7, 8]}
    records = audit(before, after, case, result, require_acceptance=False)
    assert not any(record["accepted"] for record in records)
    with pytest.raises(AssertionError):
        audit(before, after, case, result)
