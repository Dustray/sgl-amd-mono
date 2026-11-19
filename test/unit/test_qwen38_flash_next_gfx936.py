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
    args = SimpleNamespace(model_path=path, tp_size=8, disable_overlap_schedule=True, context_length=4096, max_running_requests=1, dtype="bfloat16", moe_runner_backend="triton", speculative_algorithm="EAGLE", speculative_eagle_topk=1, speculative_num_steps=3, speculative_num_draft_tokens=4)
    with pytest.raises(FlashContractError, match="attention-backend triton"):
        validate_server(args)


@pytest.mark.parametrize("override,reason", [
    ({"speculative_algorithm": "STANDALONE"}, "requires EAGLE"),
    ({"speculative_eagle_topk": 2}, r"--speculative-eagle-topk 1, got 2"),
    ({"speculative_num_steps": 2}, r"--speculative-num-steps 3, got 2"),
    ({"speculative_num_draft_tokens": 5}, r"--speculative-num-draft-tokens 4, got 5"),
    ({"speculative_draft_model_path": "/tmp/draft"}, "requires no --speculative-draft-model-path"),
    ({"enable_multi_layer_eagle": True}, "enable-multi-layer-eagle disabled"),
    ({"speculative_use_rejection_sampling": True}, "speculative-use-rejection-sampling disabled"),
])
def test_speculative_geometry_is_pinned(model_config, monkeypatch, override, reason):
    import sgl_amd_mono.qwen38_flash_next.gfx936.contract as contract

    path, _ = model_config
    monkeypatch.setattr(contract, "validate_sources", lambda: {})
    options = dict(model_path=path, tp_size=8, disable_overlap_schedule=True, context_length=4096, max_running_requests=1, dtype="bfloat16", moe_runner_backend="triton", attention_backend="triton", page_size=64, kv_cache_dtype="bfloat16", cuda_graph_backend_decode="disabled", disable_radix_cache=True, speculative_algorithm="EAGLE", speculative_eagle_topk=1, speculative_num_steps=3, speculative_num_draft_tokens=4)
    options.update(override)
    with pytest.raises(FlashContractError, match=reason):
        validate_server(SimpleNamespace(**options))


@pytest.mark.parametrize("algorithm", ["EAGLE", "NEXTN", "eagle", "NextN"])
def test_pinned_mtp_geometry_is_accepted_unfused(model_config, monkeypatch, algorithm):
    """A valid MTP launch passes the contract but leaves fused=False; the runtime stays native."""
    import sgl_amd_mono.qwen38_flash_next.gfx936.contract as contract

    path, _ = model_config
    monkeypatch.setattr(contract, "validate_sources", lambda: {})
    options = dict(model_path=path, tp_size=8, disable_overlap_schedule=True, context_length=4096, max_running_requests=1, dtype="bfloat16", moe_runner_backend="triton", attention_backend="triton", page_size=64, kv_cache_dtype="bfloat16", cuda_graph_backend_decode="disabled", disable_radix_cache=True, speculative_algorithm=algorithm, speculative_eagle_topk=1, speculative_num_steps=3, speculative_num_draft_tokens=4)
    report = validate_server(SimpleNamespace(**options))
    assert report["speculative"] == {"algorithm": algorithm.upper(), "eagle_topk": 1, "num_steps": 3, "num_draft_tokens": 4, "fused": False}
    from sgl_amd_mono.qwen38_flash_next.gfx936.contract import spec_enabled

    assert spec_enabled(SimpleNamespace(**options))
    options.pop("speculative_algorithm")
    assert not spec_enabled(SimpleNamespace(**options))


@pytest.mark.parametrize("graph_backend,policy", [
    ("full", "target capture skipped, draft graphs kept"),
    ("disabled", "no graphs anywhere (draft included)"),
])
def test_fused_verify_keeps_the_draft_graph(model_config, monkeypatch, graph_backend, policy):
    """Fused verify must not force the decode graph off: SGLang builds the draft worker's graph
    runners from the same flag, and 3 of the 4 forwards in an MTP round are draft steps."""
    import sgl_amd_mono.qwen38_flash_next.gfx936.contract as contract

    path, _ = model_config
    monkeypatch.setattr(contract, "validate_sources", lambda: {})
    monkeypatch.setenv("SGLANG_AMD_FLASH_SPEC_VERIFY", "1")
    monkeypatch.setenv("HIP_GRAPH_ACCUMULATE_DISPATCH", "0")
    options = dict(model_path=path, tp_size=8, disable_overlap_schedule=True, context_length=4096, max_running_requests=1, dtype="bfloat16", moe_runner_backend="triton", attention_backend="triton", page_size=64, kv_cache_dtype="bfloat16", cuda_graph_backend_decode=graph_backend, cuda_graph_bs_decode=[1], disable_cuda_graph_padding=True, disable_radix_cache=True, speculative_algorithm="EAGLE", speculative_eagle_topk=1, speculative_num_steps=3, speculative_num_draft_tokens=4)
    report = validate_server(SimpleNamespace(**options))
    assert report["speculative"] == {"algorithm": "EAGLE", "eagle_topk": 1, "num_steps": 3, "num_draft_tokens": 4, "fused": True, "verify_graph": policy}


@pytest.mark.parametrize("env,algorithm,expected", [
    ("1", "EAGLE", True),
    ("0", "EAGLE", False),
    # A plain decode server that happens to have the env exported must not be treated as
    # fused-verify: the plugin reads this to decide whether to skip decode-graph capture.
    ("1", None, False),
])
def test_spec_verify_needs_a_speculative_server_not_just_the_env(monkeypatch, env, algorithm, expected):
    from sgl_amd_mono.qwen38_flash_next.gfx936.contract import spec_verify_enabled

    monkeypatch.setenv("SGLANG_AMD_FLASH_SPEC_VERIFY", env)
    assert spec_verify_enabled(SimpleNamespace(speculative_algorithm=algorithm)) is expected


def _shadow_runtime(record_failures=False):
    runtime = SimpleNamespace(
        serving_shadow_failure=None, serving_shadow_failures=[], completed_decode_steps=0,
        shadow_record_failures=record_failures, save=lambda: None,
    )
    runtime._shadow_failure = FlashRuntime._shadow_failure.__get__(runtime, SimpleNamespace)
    return runtime


@pytest.mark.parametrize("discriminating,record_failures,raises", [
    # A per-module mismatch is fatal by default and only the documented flag downgrades it.
    (True, False, True),
    (True, True, False),
    # A check whose own reference missed the gate can never stop the server.
    (False, False, False),
    (False, True, False),
])
def test_only_discriminating_shadow_failures_stop_the_server(discriminating, record_failures, raises):
    runtime = _shadow_runtime(record_failures)
    if raises:
        with pytest.raises(FlashContractError, match="boom"):
            runtime._shadow_failure("boom", discriminating=discriminating)
    else:
        runtime._shadow_failure("boom", discriminating=discriminating)
    assert runtime.serving_shadow_failures[-1]["discriminating"] is discriminating


def test_full_model_gate_is_discriminating_only_when_native_reproduces():
    """Replays the 2026-11-19 verify batch: native's own re-runs miss the gate it is judged by,
    so that comparison must not be fatal -- but it must stay fatal when native does reproduce."""
    from sgl_amd_mono.qwen38_flash_next.gfx936.runner import logits_accepted

    measured_native_repeats = [
        {"finite": True, "argmax_equal": [True] * 4, "kl_reference_actual": [4e-05, 0.00034, 0.0032, 0.008], "total_variation": [0.0011, 0.0063, 0.0237, 0.0288]},
        {"finite": True, "argmax_equal": [True] * 4, "kl_reference_actual": [1e-05, 0.00042, 0.02118, 0.00366], "total_variation": [0.0002, 0.0102, 0.085, 0.0223]},
        {"finite": True, "argmax_equal": [True] * 4, "kl_reference_actual": [5e-05, 0.00042, 0.00241, 0.00166], "total_variation": [0.0015, 0.0127, 0.0287, 0.0115]},
    ]
    # Repeat 1 blows both limits, so native is not reproducible on this batch.
    assert [logits_accepted(d) for d in measured_native_repeats] == [True, False, True]
    assert not all(logits_accepted(d) for d in measured_native_repeats)
    # mono's own numbers from the same batch sit inside that spread.
    mono = {"finite": True, "argmax_equal": [True, False, True, True], "kl_reference_actual": [0.00046, 0.02134, 0.02234, 0.00269], "total_variation": [0.006, 0.0741, 0.0822, 0.0126]}
    assert not logits_accepted(mono)
    assert max(mono["total_variation"]) < max(measured_native_repeats[1]["total_variation"])
    # A clean reference keeps the check fatal.
    clean = [{"finite": True, "argmax_equal": [True] * 4, "kl_reference_actual": [1e-05] * 4, "total_variation": [1e-04] * 4}] * 3
    assert all(logits_accepted(d) for d in clean)


def test_bf16_ulp_error_is_scale_free_where_nrmse_is_not():
    """The point of the ULP metric: the same one-step-apart tensors score 1 ULP whether the
    values are large or small, while nrmse grows as the magnitude falls."""
    from sgl_amd_mono.qwen38_flash_next.gfx936.runner import bf16_ulp_error, error_metrics

    for exponent in (-4, 0, 6):
        expected = torch.full((64,), 2.0 ** exponent * 1.5, dtype=torch.bfloat16).float()
        one_ulp = 2.0 ** (exponent - 7)
        actual = expected + one_ulp
        assert bf16_ulp_error(actual, expected) == pytest.approx(1.0, rel=1e-6)
        assert bf16_ulp_error(expected, expected) == 0.0
    # Same one-ULP difference, but a small-magnitude tensor inflates nrmse past the 0.006
    # decode gate -- which is the failure mode this metric exists to distinguish.
    small = torch.full((64,), 0.0807, dtype=torch.bfloat16).float()
    assert error_metrics(small + 2.0 ** -11, small)["nrmse"] > 0.006
    assert bf16_ulp_error(small + 2.0 ** -11, small) < 1.01


def test_bf16_ulp_error_flags_a_real_divergence():
    from sgl_amd_mono.qwen38_flash_next.gfx936.runner import bf16_ulp_error

    expected = torch.full((64,), 0.0807, dtype=torch.bfloat16).float()
    assert bf16_ulp_error(expected + 0.05, expected) > 100
    assert bf16_ulp_error(torch.full((64,), float("nan")), expected) == float("inf")
    assert bf16_ulp_error(torch.zeros(64), torch.zeros(64)) == 0.0
    assert bf16_ulp_error(torch.ones(64), torch.zeros(64)) == float("inf")


@pytest.mark.parametrize("checks,expected", [
    # Plain decode output keeps the nrmse gate: no bf16_ulp key, so no escape hatch.
    ({"output": {"finite": True, "nrmse": 0.0061}}, ["output"]),
    ({"output": {"finite": True, "nrmse": 0.0059}}, []),
    # Verify output over the nrmse gate but within the reference-rounding band passes.
    # 6.0 is the layer-32 serving dump (see _gdn_shadow_failures); it clears the gate,
    # while anything a real divergence produces stays well above 16.
    ({"output": {"finite": True, "nrmse": 0.0061, "bf16_ulp": 1.0}}, []),
    ({"output": {"finite": True, "nrmse": 0.0113, "bf16_ulp": 6.0}}, []),
    ({"output": {"finite": True, "nrmse": 0.0061, "bf16_ulp": 17.0}}, ["output"]),
    ({"output": {"finite": False, "nrmse": 0.0, "bf16_ulp": 0.0}}, ["output"]),
    # A non-finite or over-gate sibling still fails even when the output is clean.
    ({"output": {"finite": True, "nrmse": 0.0061, "bf16_ulp": 1.0}, "conv": {"finite": True, "nrmse": 0.9}}, ["conv"]),
])
def test_gdn_shadow_gate_combines_nrmse_and_ulp(checks, expected):
    assert FlashRuntime._gdn_shadow_failures(checks) == expected


def test_fused_verify_needs_the_per_draft_scratch(model_config, monkeypatch):
    """ReplaySSM drops intermediate_ssm, which the fused GDN verify writes its snapshots into."""
    import sgl_amd_mono.qwen38_flash_next.gfx936.contract as contract

    path, _ = model_config
    monkeypatch.setattr(contract, "validate_sources", lambda: {})
    monkeypatch.setenv("SGLANG_AMD_FLASH_SPEC_VERIFY", "1")
    options = dict(model_path=path, tp_size=8, disable_overlap_schedule=True, context_length=4096, max_running_requests=1, dtype="bfloat16", moe_runner_backend="triton", attention_backend="triton", page_size=64, kv_cache_dtype="bfloat16", cuda_graph_backend_decode="disabled", disable_radix_cache=True, speculative_algorithm="EAGLE", speculative_eagle_topk=1, speculative_num_steps=3, speculative_num_draft_tokens=4, enable_linear_replayssm_spec=True)
    with pytest.raises(FlashContractError, match="per-draft GDN scratch"):
        validate_server(SimpleNamespace(**options))


def test_speculative_algorithm_none_value_is_not_spec(model_config, monkeypatch):
    """ServerArgs keeps SpeculativeAlgorithm.NONE when spec is off; that is not spec."""
    import sgl_amd_mono.qwen38_flash_next.gfx936.contract as contract

    path, _ = model_config
    monkeypatch.setattr(contract, "validate_sources", lambda: {})
    options = dict(model_path=path, tp_size=8, disable_overlap_schedule=True, context_length=4096, max_running_requests=1, dtype="bfloat16", moe_runner_backend="triton", attention_backend="triton", page_size=64, kv_cache_dtype="bfloat16", cuda_graph_backend_decode="disabled", disable_radix_cache=True, speculative_algorithm=SimpleNamespace(name="NONE"))
    report = validate_server(SimpleNamespace(**options))
    assert "speculative" not in report


def test_unvalidated_fused_topk_environment_is_rejected_before_loading(model_config, monkeypatch):
    path, _ = model_config
    monkeypatch.setenv("SGLANG_AMD_FLASH_NEXT_0518", "1")
    monkeypatch.setenv("SGLANG_USE_FUSED_TOPK_SOFTMAX", "1")
    args = SimpleNamespace(model_path=path, tp_size=8)
    with pytest.raises(FlashContractError, match="SGLANG_USE_FUSED_TOPK_SOFTMAX"):
        validate_server(args)


@pytest.mark.parametrize("decode,rows,tokens,expected", [(True, 1, 1, True), (True, 2, 2, True), (False, 1, 1, False), (True, 3, 3, False), (True, 1, 4, False)])
def test_decode_routing_does_not_capture_prefill_or_verify(decode, rows, tokens, expected):
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: decode, is_target_verify=lambda: False), batch_size=rows, input_ids=SimpleNamespace(numel=lambda: tokens))
    assert FlashRuntime.supports(None, batch) == expected


@pytest.mark.parametrize("rows,tokens,spec_verify,expected", [(1, 4, True, True), (2, 8, True, True), (1, 2, True, False), (1, 8, True, False), (2, 8, False, False)])
def test_fused_verify_routing(rows, tokens, spec_verify, expected):
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: False, is_target_verify=lambda: True), batch_size=rows, input_ids=SimpleNamespace(numel=lambda: tokens))
    assert FlashRuntime.supports(SimpleNamespace(spec_verify=spec_verify), batch) == expected


def test_gdn_decode_arguments_keep_scalars_out_of_verify_pointers(monkeypatch):
    """The verify pointers sit between the tensor pointers and the strides; decode must not
    shift its strides into them (positional construction did exactly that)."""
    import contextlib

    import sgl_amd_mono.qwen38_flash_next.gfx936.gdn as gdn

    captured = {}

    def launch(pointer, stream):
        captured["args"] = gdn.GDNArguments.from_buffer_copy(pointer._obj)
        return 0

    monkeypatch.setattr(gdn, "require_tensor", lambda *args, **kwargs: None)
    monkeypatch.setattr(gdn.torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(gdn.torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=0))
    attention = object.__new__(gdn.GDNAttention)
    attention.device = torch.device("cpu")
    attention.weights = tuple(torch.zeros(1) for _ in range(7))
    for name in ("projected", "projected_ba", "convolved", "recurrent", "core"):
        setattr(attention, name, torch.zeros(1))
    attention.mixed = torch.zeros((8, 2560))
    attention.executions = torch.zeros(1)
    attention.mixer = SimpleNamespace(weights=(torch.zeros(1),) * 3, normalized=torch.zeros((8, 10240)), lowrank=torch.zeros(1), gates=torch.zeros(1), partials=torch.zeros(1), state=torch.zeros(1), executions=torch.zeros(1), program=SimpleNamespace(timeout_ticks=7))
    attention.program = SimpleNamespace(base=SimpleNamespace(check=lambda status: None), library=SimpleNamespace(flash_gdn=launch))
    conv = torch.zeros((5, 3, 1280), dtype=torch.bfloat16).transpose(1, 2)
    ssm = torch.zeros((5, 6, 128, 128))
    attention(torch.zeros((2, 10240), dtype=torch.bfloat16), conv, ssm, torch.zeros(2, dtype=torch.int64))
    args = captured["args"]
    assert (args.conv_window, args.intermediate_ssm, args.scratch_rows) == (None, None, None)
    assert (args.conv_slot_stride, args.conv_channel_stride, args.conv_history_stride) == conv.stride()
    assert (args.ssm_slot_stride, args.index_bytes, args.slot_count) == (ssm.stride(0), 8, 5)
    assert (args.requests, args.steps, args.scratch_slots) == (0, 0, 0)


@pytest.mark.parametrize("dedup", [True, False])
def test_verify_scratch_views_round_trip_without_overlap(dedup):
    """Shadow saves mono's verify snapshots, lets native overwrite, and restores them; that
    needs a writable non-overlapping view of SGLang's deduplicated conv-window layout."""
    slots, steps, dim, width = 3, 4, 1280, 3
    if dedup:
        physical = torch.zeros((slots, dim, steps + width - 1), dtype=torch.bfloat16)
        window = physical.as_strided((slots, steps, dim, width), (physical.stride(0), physical.stride(2), physical.stride(1), physical.stride(2)))
    else:
        physical = window = torch.zeros((slots, steps, dim, width), dtype=torch.bfloat16)
    cache = SimpleNamespace(intermediate_conv_window=[window], intermediate_ssm=torch.zeros((slots, steps, 6, 2, 2)))
    backend = SimpleNamespace(verify_intermediate_state_indices=torch.tensor([2, 0, 1], dtype=torch.int32))
    views = FlashRuntime._verify_scratch_views(cache, backend, 2)
    assert [name for name, _ in views] == ["conv_window.2", "intermediate_ssm.2", "conv_window.0", "intermediate_ssm.0"]
    for _, view in views:
        view.copy_(torch.randn(view.shape).to(view.dtype))
    saved = [view.clone() for _, view in views]
    expected_window = window[2].clone()
    for _, view in views:
        view.zero_()
    for (_, view), value in zip(views, saved):
        view.copy_(value)
    assert torch.equal(window[2], expected_window)
    assert not window[1].any() and not cache.intermediate_ssm[1].any()


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


@pytest.mark.parametrize("override,reason", [({"cuda_graph_backend_decode": "full"}, "disable-cuda-graph-padding"), ({"cuda_graph_backend_prefill": "piecewise"}, "cuda-graph-backend-prefill disabled"), ({"attention_backend": "aiter"}, "attention-backend triton"), ({"page_size": 32}, "page-size 64"), ({"kv_cache_dtype": "fp8_e4m3"}, "BF16 KV"), ({"ple_offload_embedding": True}, "ple_offload_embedding"), ({"enable_return_indexer_topk": True}, "enable_return_indexer_topk"), ({"load_format": "dummy"}, "real checkpoint"), ({"disable_radix_cache": False}, "checkpoint reuse")])
def test_unvalidated_cache_and_capture_modes_are_rejected(model_config, override, reason):
    path, _ = model_config
    options = dict(model_path=path, tp_size=8, disable_overlap_schedule=True, context_length=4096, max_running_requests=1, dtype="bfloat16", moe_runner_backend="triton", attention_backend="triton", page_size=64, kv_cache_dtype="bfloat16", cuda_graph_backend_decode="disabled", disable_radix_cache=True)
    options.update(override)
    with pytest.raises(FlashContractError, match=reason):
        validate_server(SimpleNamespace(**options))


def test_exact_graph_batch_sizes_are_accepted(model_config, monkeypatch):
    import sgl_amd_mono.qwen38_flash_next.gfx936.contract as contract

    path, _ = model_config
    monkeypatch.setenv("HIP_GRAPH_ACCUMULATE_DISPATCH", "0")
    monkeypatch.setattr(contract, "validate_sources", lambda: {})
    options = dict(model_path=path, tp_size=8, disable_overlap_schedule=True, context_length=4096, max_running_requests=2, dtype="bfloat16", moe_runner_backend="triton", attention_backend="triton", page_size=64, kv_cache_dtype="bfloat16", cuda_graph_backend_decode="full", cuda_graph_bs_decode=[1, 2], disable_cuda_graph_padding=True, disable_radix_cache=True)
    assert not validate_server(SimpleNamespace(**options))["disable_reasons"]
    options["cuda_graph_bs_decode"] = [1, 4]
    with pytest.raises(FlashContractError, match="explicit --cuda-graph-bs-decode"):
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
    runtime.stage_timing = None
    runtime.cu_timing = None
    runtime.peers = SimpleNamespace(check=lambda: None)
    runtime.save = lambda: None
    runtime._decode_states = lambda batch=None: []
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


def test_benchmark_native_route_preserves_fused_shadow_state(monkeypatch):
    runtime = FlashRuntime.__new__(FlashRuntime)
    runtime.enabled = False
    runtime.fusion_enabled = False
    runtime.forward_keys = set()
    runtime.shadow_remaining = 1
    runtime.shadow_rearm_on_prefill = False
    runtime.completed_decode_steps = 0
    runtime.supports = lambda batch: True
    observed = []
    model = SimpleNamespace(forward=lambda **kwargs: observed.append(runtime.enabled))
    runtime._wrap_forward(model)
    batch = SimpleNamespace(batch_size=2, forward_mode=SimpleNamespace(name="DECODE", is_extend=lambda: False))
    model.forward(forward_batch=batch)
    assert observed == [False]
    assert runtime.shadow_remaining == 1
    assert runtime.completed_decode_steps == 0
    assert runtime.forward_keys == {("DECODE", 2, False)}


@pytest.mark.parametrize("busy", ["running_batch", "waiting_queue", "cur_batch", None])
def test_benchmark_mode_switch_requires_idle_scheduler(tmp_path, monkeypatch, busy):
    from sgl_amd_mono_0518.flash_next import apply_benchmark_mode

    control = tmp_path / "mode.txt"
    control.write_text("native\n")
    monkeypatch.setenv("SGLANG_AMD_FLASH_BENCHMARK_CONTROL_FILE", str(control))
    runtime = SimpleNamespace(fusion_enabled=True)
    scheduler = SimpleNamespace(running_batch=SimpleNamespace(reqs=[]), waiting_queue=[], cur_batch=None)
    if busy == "running_batch":
        scheduler.running_batch.reqs.append(object())
    elif busy == "waiting_queue":
        scheduler.waiting_queue.append(object())
    elif busy == "cur_batch":
        scheduler.cur_batch = object()
    if busy:
        with pytest.raises(RuntimeError, match="idle"):
            apply_benchmark_mode(runtime, scheduler)
        assert runtime.fusion_enabled
    else:
        apply_benchmark_mode(runtime, scheduler)
        assert not runtime.fusion_enabled
        control.write_text("monokernel\n")
        apply_benchmark_mode(runtime, scheduler)
        assert runtime.fusion_enabled
        control.write_text("invalid\n")
        with pytest.raises(ValueError, match="execution mode"):
            apply_benchmark_mode(runtime, scheduler)


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
    runtime._decode_states = lambda batch=None: []
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
