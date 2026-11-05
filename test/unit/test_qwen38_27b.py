import ast
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

import sgl_amd_mono
from sgl_amd_mono.qwen38_27b import spec
from sgl_amd_mono.qwen38_27b.errors import UnsupportedConfiguration
from sgl_amd_mono.qwen38_27b.gfx936 import config


@pytest.fixture
def configuration():
    return {
        "architectures": [config.ARCHITECTURE],
        "model_type": config.MODEL_TYPE,
        "text_config": {
            "hidden_size": config.HIDDEN,
            "intermediate_size": config.INTERMEDIATE,
            "num_hidden_layers": config.LAYERS,
            "hidden_act": "silu",
            "dtype": "bfloat16",
            "layer_types": list(config.LAYER_TYPES * 16),
            **config.STATE_DIMENSIONS,
        },
    }


@pytest.fixture
def device():
    return {"gpu_arch": "gfx936", "gpu_compute_units": 80, "flydsl_installed": True, "flydsl_version": "validated"}


@pytest.mark.parametrize("change", ["quantization", "fp16", "moe", "hidden", "tp"])
def test_ffn_rejects_other_weight_contracts(configuration, device, change):
    tp_size = 1
    if change == "quantization":
        configuration["text_config"]["quantization_config"] = {"quant_method": "quark"}
    elif change == "fp16":
        configuration["text_config"]["dtype"] = "float16"
    elif change == "moe":
        configuration["architectures"] = ["Qwen4ExpForConditionalGeneration"]
    elif change == "hidden":
        configuration["text_config"]["hidden_size"] = 2560
    else:
        tp_size = 2
    assert spec.ffn_disable_reasons(configuration, tp_size, device, "validated")


@pytest.mark.parametrize("change", ["order", "interval", "ssm"])
def test_dense_rejects_incompatible_state_or_layer_layout(configuration, device, monkeypatch, change):
    monkeypatch.setenv("SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE", "1")
    assert not spec.dense_disable_reasons(configuration, 1, device, "validated")
    if change == "order":
        configuration["text_config"]["layer_types"][0] = "full_attention"
    elif change == "interval":
        configuration["text_config"]["full_attention_interval"] = 8
    else:
        configuration["text_config"]["linear_num_value_heads"] = 32
    assert spec.dense_disable_reasons(configuration, 1, device, "validated")


def test_hip_dense_does_not_require_flydsl(configuration, device, monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE", "1")
    device.update(flydsl_installed=False, flydsl_version=None)
    assert spec.ffn_disable_reasons(configuration, 1, device, "validated")
    assert not spec.dense_disable_reasons(configuration, 1, device, "validated")


def test_direct_build_cannot_bypass_experimental_guard(monkeypatch):
    monkeypatch.delenv("SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE", raising=False)
    with pytest.raises(UnsupportedConfiguration, match="opt-in"):
        spec.build(None, "monokernel")
    monkeypatch.setenv("SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE", "1")
    with pytest.raises(UnsupportedConfiguration, match="unaccepted"):
        spec.build(None, "required")


def test_legacy_aliases_use_the_model_implementation():
    from sgl_amd_mono_0518 import contract, hip_ffn, hip_monokernel, mono_runtime
    from sgl_amd_mono.qwen38_27b.gfx936 import ffn, program, runner

    assert hip_ffn.FusedBF16FFN is ffn.FusedBF16FFN
    assert hip_monokernel.DenseK1 is program.DenseK1
    assert mono_runtime.MonoRuntime is runner.MonoRuntime
    assert contract.ContractError is program.ContractError
    assert Path(program.__file__).with_name("kernel.cpp").is_file()


def test_model_never_imports_the_legacy_bridge():
    root = Path(spec.__file__).parent
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert all(not module.startswith("sgl_amd_mono_0518") for module in modules), path


@pytest.mark.parametrize("enabled", [False, True])
def test_package_entry_point_handles_legacy_sglang(monkeypatch, enabled):
    from sgl_amd_mono_0518 import bootstrap

    imported = builtins.__import__
    called = []

    def missing_native_interface(name, *arguments, **keywords):
        if name in {"sglang.kernels.plugin_kernels.amd.spec", "sglang.srt.monokernels.spec"}:
            raise ModuleNotFoundError(name=name)
        return imported(name, *arguments, **keywords)

    monkeypatch.setattr(builtins, "__import__", missing_native_interface)
    monkeypatch.setattr(bootstrap, "register", lambda: called.append(True))
    monkeypatch.setenv("SGLANG_AMD_MONO_0518", "1" if enabled else "0")
    sgl_amd_mono.register_specs()
    assert called == ([True] if enabled else [])


def test_package_entry_point_keeps_native_registration(monkeypatch):
    imported = builtins.__import__
    registrations = []

    def native_interface(name, *arguments, **keywords):
        if name == "sglang.kernels.plugin_kernels.amd.spec":
            return SimpleNamespace(register=lambda **entry: registrations.append(entry))
        return imported(name, *arguments, **keywords)

    monkeypatch.setattr(builtins, "__import__", native_interface)
    monkeypatch.delenv("SGLANG_AMD_MONO_ROCM_LLVM", raising=False)
    sgl_amd_mono.register_specs()
    assert len(registrations) == len(sgl_amd_mono.SPECS)
    assert all(entry["backend"] == "amd" for entry in registrations)
