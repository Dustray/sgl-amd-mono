import copy
import json

import pytest

from sgl_amd_mono_0518 import contract


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
