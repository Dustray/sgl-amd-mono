from types import SimpleNamespace

import pytest
import torch

from sgl_amd_mono.qwen38_flash_next.gfx936.graph import GraphExecution
from sgl_amd_mono.qwen38_flash_next.gfx936.program import FlashContractError


@pytest.mark.parametrize("padding,batches", [(True, [1, 2]), (False, [1, 4])])
def test_graph_capture_rejects_padding_and_unsupported_shapes(padding, batches):
    graph = GraphExecution(SimpleNamespace())
    runner = SimpleNamespace(disable_padding=not padding, capture_bs=batches)
    with pytest.raises(FlashContractError, match="without padding"):
        graph.capture(lambda runner: None, runner)


def test_failed_capture_restores_shadow_and_capture_state():
    runtime = SimpleNamespace(graph_capture_active=False, shadow_remaining=3, save=lambda: None)
    graph = GraphExecution(runtime)
    runner = SimpleNamespace(disable_padding=True, capture_bs=[1, 2])

    def capture(runner):
        assert runtime.graph_capture_active and runtime.shadow_remaining == 0
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        graph.capture(capture, runner)
    assert not runtime.graph_capture_active and runtime.shadow_remaining == 3
    assert graph.captured_batch_sizes == []


def test_graph_eager_mismatch_restores_live_state_before_raising():
    cache = torch.tensor([7.0])
    hidden = torch.tensor([11.0])
    result = SimpleNamespace(next_token_logits=torch.tensor([[1.0, 2.0]]))
    runtime = SimpleNamespace(enabled=False, shadow_remaining=1, model=SimpleNamespace(last_hc_hidden_states=hidden))

    def eager(*args):
        cache.fill_(99)
        runtime.model.last_hc_hidden_states = torch.tensor([100.0])
        return SimpleNamespace(next_token_logits=torch.tensor([[1.0, 3.0]]))

    runtime.original_forward = eager
    graph = GraphExecution(runtime)
    batch = SimpleNamespace(batch_size=1, input_ids=torch.tensor([1]), positions=torch.tensor([2]))
    with pytest.raises(FlashContractError, match="differs from eager"):
        graph._validate(batch, result, [("cache", cache)], [torch.tensor([6.0])])
    assert torch.equal(cache, torch.tensor([7.0]))
    assert torch.equal(result.next_token_logits, torch.tensor([[1.0, 2.0]]))
    assert runtime.model.last_hc_hidden_states is hidden
    assert not runtime.enabled and runtime.shadow_remaining == 1
    assert not graph.checks[0]["accepted"]


def test_benchmark_route_cannot_change_captured_graph(tmp_path, monkeypatch):
    from sgl_amd_mono_0518.flash_next import apply_benchmark_mode

    control = tmp_path / "mode.txt"
    control.write_text("native\n")
    monkeypatch.setenv("SGLANG_AMD_FLASH_BENCHMARK_CONTROL_FILE", str(control))
    runtime = SimpleNamespace(fusion_enabled=True, graph=object())
    with pytest.raises(RuntimeError, match="Graph capture"):
        apply_benchmark_mode(runtime, None)
    assert runtime.fusion_enabled


def test_graph_match_preserves_separate_native_acceptance_and_unused_nan_bits():
    cache = torch.tensor([float("nan"), 7.0])
    result = SimpleNamespace(next_token_logits=torch.tensor([[1.0, 2.0]]))
    runtime = SimpleNamespace(enabled=False, shadow_remaining=1, model=SimpleNamespace(last_hc_hidden_states=None), full_model_shadow_checks=[], validated_batch_sizes=set(), shadow_remaining_by_batch={})

    def eager(*args):
        cache[1] = 7
        return SimpleNamespace(next_token_logits=result.next_token_logits.clone())

    def native_shadow(*args):
        runtime.full_model_shadow_checks.append({"accepted": False})

    runtime.original_forward = eager
    runtime._full_model_shadow = native_shadow
    graph = GraphExecution(runtime)
    batch = SimpleNamespace(batch_size=1, input_ids=torch.tensor([1]), positions=torch.tensor([2]))
    graph._validate(batch, result, [("cache", cache)], [torch.tensor([float("nan"), 6.0])])
    assert graph.checks[0]["accepted"]
    assert not runtime.full_model_shadow_checks[0]["accepted"] and not runtime.validated_batch_sizes
    assert torch.isnan(cache[0]) and cache[1] == 7
    assert not runtime.enabled and runtime.shadow_remaining == 0
