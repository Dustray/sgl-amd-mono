import copy
import json
from types import SimpleNamespace

import pytest
import torch

from sgl_amd_mono.qwen38_flash_next import HF_CONFIG, INDEXER_CONFIG, LAYER_PATTERN
from sgl_amd_mono.qwen38_flash_next.gfx936.contract import inspect_model, validate_server
from sgl_amd_mono.qwen38_flash_next.gfx936.program import FlashContractError
from sgl_amd_mono.qwen38_flash_next.gfx936.runner import FlashRuntime, logits_accepted, logits_metrics
from sgl_amd_mono.qwen38_flash_next.gfx936.qsa import cache_views
from sgl_amd_mono.qwen38_flash_next.gfx936.reference import stable_topk


@pytest.fixture
def model_config(tmp_path):
    config = {"architectures": ["Qwen4ExpForConditionalGeneration"], "text_config": {**copy.deepcopy(HF_CONFIG), **INDEXER_CONFIG, "layer_types": list(LAYER_PATTERN) * 12}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path, config


def test_tp8_geometry_is_supported(model_config):
    path, _ = model_config
    assert inspect_model(path, 8)["disable_reasons"] == []


def test_accuracy_topk_respects_offsets_padding_and_tie_policy():
    scores = torch.tensor([[99.0, 2.0, 2.0, 5.0, 2.0, -99.0], [3.0, 1.0, 99.0, 99.0, 99.0, 99.0]])
    lengths = torch.tensor([4, 2], dtype=torch.int32)
    starts = torch.tensor([1, 0], dtype=torch.int32)
    actual = stable_topk(scores, lengths, 3, starts)
    assert torch.equal(actual, torch.tensor([[2, 0, 1], [0, 1, -1]], dtype=torch.int32))
    assert torch.equal(stable_topk(torch.empty(2, 0), torch.zeros(2, dtype=torch.int32), 3), torch.full((2, 3), -1, dtype=torch.int32))


def test_accuracy_topk_preserves_short_context_chronological_order():
    actual = stable_topk(torch.tensor([[1.0, 10.0]]), torch.tensor([2]), 3)
    assert torch.equal(actual, torch.tensor([[0, 1, -1]], dtype=torch.int32))


@pytest.mark.parametrize("tp_size", [1, 2, 4, 16])
def test_other_parallel_widths_are_rejected(model_config, tp_size):
    path, _ = model_config
    assert "requires TP8" in inspect_model(path, tp_size)["disable_reasons"]


@pytest.mark.parametrize("key,value", [("hidden_size", 5120), ("num_experts_per_tok", 8), ("hc_count", 2), ("indexer_compress_ratio", 8), ("ple_layer_ids", [1]), ("output_gate_type", "silu")])
def test_model_changes_are_rejected(model_config, key, value):
    path, config = model_config
    config["text_config"][key] = value
    (path / "config.json").write_text(json.dumps(config))
    assert any(key in reason for reason in inspect_model(path, 8)["disable_reasons"])


def test_quantized_checkpoint_is_rejected(model_config):
    path, config = model_config
    config["quantization_config"] = {"quant_method": "quark"}
    (path / "config.json").write_text(json.dumps(config))
    assert "requires unquantized BF16 weights" in inspect_model(path, 8)["disable_reasons"]


def test_unsupported_server_features_are_rejected_before_source_check(model_config):
    path, _ = model_config
    args = SimpleNamespace(model_path=path, tp_size=8, speculative_algorithm="NEXTN", disable_overlap_schedule=True, context_length=4096, max_running_requests=1, dtype="bfloat16", moe_runner_backend="triton")
    with pytest.raises(FlashContractError, match="speculative_algorithm"):
        validate_server(args)


@pytest.mark.parametrize("decode,rows,tokens,expected", [(True, 1, 1, True), (True, 2, 2, True), (False, 1, 1, False), (True, 3, 3, False), (True, 1, 4, False)])
def test_decode_routing_does_not_capture_prefill_or_verify(decode, rows, tokens, expected):
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: decode), batch_size=rows, input_ids=SimpleNamespace(numel=lambda: tokens))
    assert FlashRuntime.supports(None, batch) == expected


@pytest.mark.parametrize("layout", ["nhd", "hnd", "hcu_fa"])
def test_qsa_cache_views_preserve_physical_slots(layout):
    expected_key = torch.arange(128 * 256, dtype=torch.float32).bfloat16().reshape(2, 64, 1, 256)
    expected_value = -expected_key
    if layout == "nhd":
        key, value = expected_key.reshape(128, 1, 256), expected_value.reshape(128, 1, 256)
    else:
        key = expected_key.permute(0, 2, 1, 3).contiguous()
        value = expected_value.permute(0, 2, 3, 1).contiguous() if layout == "hcu_fa" else expected_value.permute(0, 2, 1, 3).contiguous()
    actual_key, actual_value = cache_views(key, value)
    assert torch.equal(actual_key, expected_key)
    assert torch.equal(actual_value, expected_value)
    assert actual_key.data_ptr() == key.data_ptr()
    assert actual_value.data_ptr() == value.data_ptr()


@pytest.mark.parametrize("override,reason", [({"cuda_graph_backend_decode": "full"}, "Graph padding"), ({"cuda_graph_backend_prefill": "piecewise"}, "cuda-graph-backend-prefill disabled"), ({"attention_backend": "aiter"}, "attention-backend triton"), ({"page_size": 32}, "page-size 64"), ({"kv_cache_dtype": "fp8_e4m3"}, "BF16 KV"), ({"ple_offload_embedding": True}, "ple_offload_embedding"), ({"enable_return_indexer_topk": True}, "enable_return_indexer_topk"), ({"load_format": "dummy"}, "real checkpoint"), ({"disable_radix_cache": False}, "checkpoint reuse")])
def test_unvalidated_cache_and_capture_modes_are_rejected(model_config, override, reason):
    path, _ = model_config
    options = dict(model_path=path, tp_size=8, disable_overlap_schedule=True, context_length=4096, max_running_requests=1, dtype="bfloat16", moe_runner_backend="triton", attention_backend="triton", page_size=64, kv_cache_dtype="bfloat16", cuda_graph_backend_decode="disabled", disable_radix_cache=True)
    options.update(override)
    with pytest.raises(FlashContractError, match=reason):
        validate_server(SimpleNamespace(**options))


@pytest.mark.parametrize("native_logits,failed", [([[1.0, 2.0, 4.0001]], False), ([[5.0, 0.0, 1.0]], True)])
@pytest.mark.parametrize("record_failures", [False, True])
def test_full_model_shadow_restores_fused_state_and_shared_logits_buffer(tmp_path, native_logits, failed, record_failures):
    runtime = FlashRuntime.__new__(FlashRuntime)
    runtime.model = SimpleNamespace(last_hc_hidden_states=torch.tensor([7.0]))
    actual_hc = runtime.model.last_hc_hidden_states
    runtime.enabled = True
    runtime.completed_decode_steps = 0
    runtime.full_model_shadow_checks = []
    runtime.shadow_record_failures = record_failures
    runtime.serving_shadow_failures = []
    runtime.failure_captures = set()
    runtime.report_dir = tmp_path
    runtime.rank = 0
    runtime.save = lambda: None
    state = torch.tensor([3.0, 4.0])
    before = [torch.tensor([1.0, 2.0])]
    actual_logits = torch.tensor([[1.0, 2.0, 4.0]])
    result = SimpleNamespace(next_token_logits=actual_logits.clone())

    batch = SimpleNamespace(input_ids=torch.tensor([3]), seq_lens=torch.tensor([10]))

    def native_forward(forward_batch):
        assert not runtime.enabled
        assert forward_batch is batch
        assert torch.equal(state, before[0])
        state.fill_(9)
        runtime.model.last_hc_hidden_states = torch.tensor([8.0])
        result.next_token_logits.copy_(torch.tensor(native_logits))
        return result

    if failed and not record_failures:
        with pytest.raises(FlashContractError, match="full-model same-state shadow"):
            runtime._full_model_shadow(native_forward, (), {"forward_batch": batch}, result, [("cache", state)], before)
        assert (tmp_path / "failed-full-model-step-0-rank-0.pt").exists()
        repeats = runtime.full_model_shadow_checks[0]["native_repeats"]
        assert len(repeats) == 3
        assert all(item["logits"]["nrmse"] == 0 and item["distribution"]["argmax_equal"] == [True] for item in repeats)
    else:
        runtime._full_model_shadow(native_forward, (), {"forward_batch": batch}, result, [("cache", state)], before)
    assert runtime.enabled
    assert runtime.model.last_hc_hidden_states is actual_hc
    assert torch.equal(state, torch.tensor([3.0, 4.0]))
    assert torch.equal(result.next_token_logits, actual_logits)
    assert runtime.full_model_shadow_checks[0]["restored_state_tensors"] == 1
    assert runtime.full_model_shadow_checks[0]["input_token_ids"] == [3]
    assert runtime.full_model_shadow_checks[0]["sequence_lengths"] == [10]
    assert runtime.full_model_shadow_checks[0]["accepted"] is not failed


def test_native_replay_compares_the_layer_return_value():
    runtime = FlashRuntime.__new__(FlashRuntime)
    runtime.enabled = False
    runtime.reference_layer_checks = []
    runtime.layer_outputs = [torch.tensor([[1.0, 2.0]])]
    layer = SimpleNamespace(forward=lambda **kwargs: (torch.tensor([[1.0, 2.0]]), None))
    runtime._wrap_layer(0, layer)
    output, residual = layer.forward()
    assert torch.equal(output, runtime.layer_outputs[0]) and residual is None
    assert runtime.reference_layer_checks == [{"layer": 0, "finite": True, "nrmse": 0.0, "max_absolute_error": 0.0}]


def test_logit_distribution_acceptance_ignores_constant_shift():
    expected = torch.tensor([[1.0, 2.0, 4.0]])
    metrics = logits_metrics(expected + 100, expected)
    assert logits_accepted(metrics)
    assert metrics["actual_token_ids"] == metrics["reference_token_ids"] == [2]
    assert metrics["actual_top2_margin"] == metrics["reference_top2_margin"] == [2.0]


@pytest.mark.parametrize("actual", [[[1.0, 2.0, 40.0]], [[5.0, 2.0, 4.0]], [[1.0, float("nan"), 4.0]]])
def test_logit_distribution_rejects_probability_drift_argmax_change_and_nan(actual):
    assert not logits_accepted(logits_metrics(torch.tensor(actual), torch.tensor([[1.0, 2.0, 4.0]])))


@pytest.mark.parametrize("rearm", [True, False])
def test_prefill_rearms_shadow_checks_without_losing_validated_batch_sizes(monkeypatch, rearm):
    runtime = FlashRuntime.__new__(FlashRuntime)
    runtime.enabled = False
    runtime.shadow_steps = 2
    runtime.shadow_rearm_on_prefill = rearm
    runtime.shadow_remaining = 2
    runtime.shadow_remaining_by_batch = {}
    runtime.validated_batch_sizes = set()
    runtime.forward_keys = set()
    runtime.completed_decode_steps = 0
    runtime.full_model_shadow_checks = []
    runtime.peers = SimpleNamespace(check=lambda: None)
    runtime.save = lambda: None
    runtime._decode_states = lambda: []
    runtime.supports = lambda batch: batch.forward_mode.name == "DECODE"
    model = SimpleNamespace(forward=lambda **kwargs: None)
    runtime._wrap_forward(model)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def batch(rows, name):
        return SimpleNamespace(batch_size=rows, forward_mode=SimpleNamespace(name=name, is_extend=lambda: name == "EXTEND"))

    for rows in (1, 2):
        previous_remaining = runtime.shadow_remaining
        model.forward(forward_batch=batch(rows, "EXTEND"))
        assert runtime.shadow_remaining == (2 if rearm else previous_remaining)
        model.forward(forward_batch=batch(rows, "DECODE"))
        assert runtime.shadow_remaining == 1
        model.forward(forward_batch=batch(rows, "DECODE"))
        assert runtime.shadow_remaining == 0
    assert runtime.validated_batch_sizes == {1, 2}
    model.forward(forward_batch=batch(1, "EXTEND"))
    model.forward(forward_batch=batch(1, "DECODE"))
    assert runtime.shadow_remaining == (1 if rearm else 0)
    assert runtime.validated_batch_sizes == {1, 2}


def test_local_failure_latches_report_after_prior_success(monkeypatch):
    runtime = FlashRuntime.__new__(FlashRuntime)
    runtime.enabled = False
    runtime.serving_shadow_failure = None
    runtime.serving_shadow_failures = []
    runtime.shadow_steps = 1
    runtime.shadow_remaining = 0
    runtime.shadow_remaining_by_batch = {1: 0}
    runtime.shadow_rearm_on_prefill = True
    runtime.shadow_record_failures = False
    runtime.validated_batch_sizes = {1}
    runtime.forward_keys = set()
    runtime.completed_decode_steps = 1
    runtime.full_model_shadow_checks = []
    runtime.shadow_checks = []
    runtime.validation = [None] * 96
    runtime.gdn_validation = [None] * 72
    runtime.communication_validation = [None] * 12
    runtime.rank = 2
    program = SimpleNamespace(program=SimpleNamespace(manifest={}))
    runtime.peers = SimpleNamespace(check=lambda: None, program=program.program)
    runtime.gdn = [program]
    runtime.qsa = [None, None, None, program]
    runtime.ple = [None, program]
    runtime.io = program
    runtime.execution_counts = lambda: {}
    saved = []
    runtime.save = lambda: saved.append(runtime.snapshot())
    runtime.supports = lambda batch: True
    runtime._decode_states = lambda: []
    batch = SimpleNamespace(batch_size=1, forward_mode=SimpleNamespace(name="DECODE", is_extend=lambda: False))

    def failed_forward(**kwargs):
        raise FlashContractError("Flash-Next QSA serving shadow failed at layer 47")

    model = SimpleNamespace(forward=failed_forward)
    runtime._wrap_forward(model)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with pytest.raises(FlashContractError, match="QSA serving shadow failed"):
        model.forward(forward_batch=batch)
    assert not runtime.enabled
    assert len(saved) == 1
    assert saved[0]["acceptance_status"] == "serving_shadow_failed"
    assert not saved[0]["whole_decoder_monokernel"]
    assert saved[0]["serving_shadow_failure"] == "Flash-Next QSA serving shadow failed at layer 47"
