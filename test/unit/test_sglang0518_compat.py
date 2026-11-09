import copy
import json
from types import SimpleNamespace

import pytest

from sgl_amd_mono_0518 import contract


@pytest.fixture(autouse=True)
def unaccepted_source_by_default(monkeypatch):
    monkeypatch.setattr(contract.model_spec.geometry, "accepted_features", lambda: frozenset())
    monkeypatch.setenv("SGLANG_USE_LIGHTOP", "1")
    monkeypatch.setenv("SGLANG_USE_CAUSAL_CONV1D", "1")


def test_graph_replay_cannot_bypass_device_failure_check(monkeypatch):
    from sgl_amd_mono_0518 import bootstrap

    def reject():
        raise contract.ContractError("persistent device failure")

    monkeypatch.setattr(bootstrap, "_mono_runtime", SimpleNamespace(enabled=False, check_synchronization=reject))
    with pytest.raises(contract.ContractError, match="persistent device failure"):
        bootstrap.after_forward(SimpleNamespace(can_run_graph=True), None)
    result = SimpleNamespace(can_run_graph=False)
    assert bootstrap.after_forward(result, None) is result


@pytest.mark.parametrize("missing", ["SGLANG_USE_LIGHTOP", "SGLANG_USE_CAUSAL_CONV1D", "SGLANG_USE_AITER_LINEAR_ATTN"])
def test_dense_refuses_unvalidated_native_numerical_path(monkeypatch, missing):
    from sgl_amd_mono_0518 import bootstrap

    monkeypatch.setenv("SGLANG_AMD_MONO_0518", "1")
    monkeypatch.setenv("SGLANG_AMD_MONO_0518_MODE", "required")
    monkeypatch.setenv("SGLANG_AMD_MONO_0518_SHADOW_STEPS", "0")
    for name in ("SGLANG_USE_LIGHTOP", "SGLANG_USE_CAUSAL_CONV1D", "SGLANG_USE_AITER_LINEAR_ATTN"):
        monkeypatch.setenv(name, "1")
    monkeypatch.delenv(missing)
    monkeypatch.setattr(bootstrap, "probe", lambda *args: {"accepted_features": ["dense_eager"]})
    monkeypatch.setattr(bootstrap, "validate", lambda *args: None)
    monkeypatch.setattr(bootstrap, "_save", lambda: None)
    arguments = SimpleNamespace(model_path="unused", tp_size=1, pp_size=1, disable_overlap_schedule=True, cuda_graph_backend_decode="disabled")
    with pytest.raises(contract.ContractError, match=missing):
        bootstrap.before_resolve(arguments)
    assert not hasattr(arguments, "_mono_0518_checked")
    monkeypatch.setenv(missing, "1")
    bootstrap.before_resolve(arguments)


@pytest.mark.parametrize("overlap_disabled,graph_backend", [(False, "disabled"), (True, "full")])
def test_dense_refuses_unvalidated_scheduling(monkeypatch, overlap_disabled, graph_backend):
    from sgl_amd_mono_0518 import bootstrap

    monkeypatch.setenv("SGLANG_AMD_MONO_0518", "1")
    monkeypatch.setenv("SGLANG_AMD_MONO_0518_MODE", "monokernel")
    monkeypatch.setenv("SGLANG_AMD_MONO_0518_SHADOW_STEPS", "0")
    monkeypatch.delenv("SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_GRAPH", raising=False)
    monkeypatch.setattr(bootstrap, "probe", lambda *arguments: {})
    monkeypatch.setattr(bootstrap, "validate", lambda *arguments: None)
    monkeypatch.setattr(bootstrap, "_save", lambda: None)
    arguments = SimpleNamespace(model_path="unused", tp_size=1, pp_size=1, disable_overlap_schedule=overlap_disabled, cuda_graph_backend_decode=graph_backend)
    with pytest.raises(contract.ContractError, match="overlap|graph replay"):
        bootstrap.before_resolve(arguments)


@pytest.mark.parametrize("accumulation", [None, "1", "0"])
def test_cooperative_graph_requires_runtime_configuration(monkeypatch, accumulation):
    from sgl_amd_mono_0518 import bootstrap

    monkeypatch.setenv("SGLANG_AMD_MONO_0518", "1")
    monkeypatch.setenv("SGLANG_AMD_MONO_0518_MODE", "monokernel")
    monkeypatch.setenv("SGLANG_AMD_MONO_0518_SHADOW_STEPS", "0")
    monkeypatch.setenv("SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_GRAPH", "1")
    monkeypatch.setenv("SGLANG_USE_AITER_LINEAR_ATTN", "1")
    if accumulation is None:
        monkeypatch.delenv("HIP_GRAPH_ACCUMULATE_DISPATCH", raising=False)
    else:
        monkeypatch.setenv("HIP_GRAPH_ACCUMULATE_DISPATCH", accumulation)
    monkeypatch.setattr(bootstrap, "probe", lambda *arguments: {})
    monkeypatch.setattr(bootstrap, "validate", lambda *arguments: None)
    monkeypatch.setattr(bootstrap, "_save", lambda: None)
    arguments = SimpleNamespace(model_path="unused", tp_size=1, pp_size=1, disable_overlap_schedule=True, cuda_graph_backend_decode="full")
    if accumulation == "0":
        bootstrap.before_resolve(arguments)
    else:
        with pytest.raises(contract.ContractError, match="HIP_GRAPH_ACCUMULATE_DISPATCH"):
            bootstrap.before_resolve(arguments)


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    configuration = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "hidden_act": "silu",
            "dtype": "bfloat16",
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "rms_norm_eps": 1e-6,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(configuration))
    monkeypatch.setattr(contract, "environment_contract", lambda: {"errors": [], "upstream_framework_present": False})
    monkeypatch.delenv("SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE", raising=False)
    return tmp_path


@pytest.fixture
def device():
    return {
        "gpu_arch": "gfx936",
        "gpu_compute_units": 80,
        "flydsl_installed": True,
        "flydsl_version": contract.manifest()["validated_local_flydsl"]["version"],
    }


def test_default_refuses_unaccepted_dense_but_keeps_ffn(model_path, device):
    report = contract.probe(model_path, device=device)
    assert not report["monokernel_eligible"]
    assert report["fused_ffn_eligible"]
    contract.validate(report, "fused_ffn")
    contract.validate(report, "audit")
    with pytest.raises(contract.UnsupportedConfiguration, match="prototype"):
        contract.validate(report, "monokernel")


@pytest.mark.parametrize("opt_in", [False, True])
def test_strict_mode_refuses_experiment_even_with_opt_in(model_path, device, monkeypatch, opt_in):
    if opt_in:
        monkeypatch.setenv("SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE", "1")
    report = contract.probe(model_path, device=device)
    with pytest.raises(contract.UnsupportedConfiguration, match="Strict Monokernel"):
        contract.validate(report, "required")


def test_explicit_experiment_does_not_claim_acceptance(model_path, device, monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE", "1")
    report = contract.probe(model_path, device=device)
    assert report["monokernel_eligible"]
    assert report["monokernel_acceptance"] == "not_accepted_experimental"
    contract.validate(report, "monokernel")


def test_matching_record_enables_strict_eager_without_experimental_opt_in(model_path, device, monkeypatch):
    monkeypatch.setattr(contract.model_spec.geometry, "accepted_features", lambda: frozenset({"dense_eager"}))
    report = contract.probe(model_path, device=device)
    assert report["monokernel_eligible"]
    assert report["monokernel_acceptance"] == "accepted"
    assert not report["experimental_dense_opt_in"]
    contract.validate(report, "required")


def test_strict_eager_record_cannot_enable_unaccepted_graph(monkeypatch):
    from sgl_amd_mono_0518 import bootstrap

    monkeypatch.setenv("SGLANG_AMD_MONO_0518", "1")
    monkeypatch.setenv("SGLANG_AMD_MONO_0518_MODE", "required")
    monkeypatch.setenv("SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_GRAPH", "1")
    monkeypatch.setenv("SGLANG_AMD_MONO_0518_SHADOW_STEPS", "0")
    monkeypatch.setattr(bootstrap, "probe", lambda *args: {"accepted_features": ["dense_eager"]})
    monkeypatch.setattr(bootstrap, "validate", lambda *args: None)
    monkeypatch.setattr(bootstrap, "_save", lambda: None)
    arguments = SimpleNamespace(model_path="unused", tp_size=1, pp_size=1, disable_overlap_schedule=True, cuda_graph_backend_decode="full")
    with pytest.raises(contract.ContractError, match="dense_graph acceptance record"):
        bootstrap.before_resolve(arguments)


@pytest.mark.parametrize("changes", [{"gpu_arch": "gfx950"}, {"gpu_compute_units": 79}])
def test_experiment_preserves_hardware_guards(model_path, device, monkeypatch, changes):
    monkeypatch.setenv("SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE", "1")
    changed_device = copy.copy(device)
    changed_device.update(changes)
    report = contract.probe(model_path, device=changed_device)
    assert not report["monokernel_eligible"]
    with pytest.raises(contract.UnsupportedConfiguration):
        contract.validate(report, "monokernel")


def test_experiment_preserves_state_layout_guards(model_path, device, monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE", "1")
    path = model_path / "config.json"
    configuration = json.loads(path.read_text())
    configuration["text_config"]["linear_num_value_heads"] = 32
    path.write_text(json.dumps(configuration))
    report = contract.probe(model_path, device=device)
    assert not report["monokernel_eligible"]
    with pytest.raises(contract.UnsupportedConfiguration):
        contract.validate(report, "monokernel")
