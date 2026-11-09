import ast
import builtins
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

import sgl_amd_mono
from sgl_amd_mono.qwen38_27b import spec
from sgl_amd_mono.qwen38_27b.errors import UnsupportedConfiguration
from sgl_amd_mono.qwen38_27b.gfx936 import config


@pytest.fixture(autouse=True)
def unaccepted_source_by_default(monkeypatch):
    monkeypatch.setattr(config, "accepted_features", lambda: frozenset())


@pytest.mark.parametrize("failure", [None, "implementation_sha256", "compatibility_sha256", "torch_version", "hip_compiler_sha256", "failed", "features", "malformed"])
def test_acceptance_record_requires_exact_source_and_environment(tmp_path, monkeypatch, failure):
    import json
    from sgl_amd_mono.qwen38_27b import acceptance

    identity = {name: "current" for name in ("implementation_sha256", "compatibility_sha256", "torch_version", "hip_compiler_sha256")}
    record = {"schema_version": 1, "passed": True, "source_identity": dict(identity), "features": ["dense_eager", "dense_graph"]}
    if failure in identity:
        record["source_identity"][failure] = "stale"
    elif failure == "failed":
        record["passed"] = False
    elif failure == "features":
        record["features"] = ["dense_graph"]
    elif failure == "malformed":
        record = []
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(record))
    monkeypatch.setattr(acceptance, "RECORD_PATH", path)
    monkeypatch.setattr(acceptance, "source_identity", lambda: identity)
    assert acceptance.accepted_features() == (frozenset({"dense_eager", "dense_graph"}) if failure is None else frozenset())


def test_strict_build_uses_current_accepted_runner(monkeypatch):
    from sgl_amd_mono.qwen38_27b.gfx936 import runner

    result = object()
    monkeypatch.setattr(config, "accepted_features", lambda: frozenset({"dense_eager"}))
    monkeypatch.setattr(runner, "MonoRuntime", lambda model_runner: result)
    assert spec.build(None, "required") is result


@pytest.fixture
def shadow_check():
    error = {"finite": True, "normalized_rms_error": 0.001, "cosine_similarity": 0.999999}
    return {
        "rows": 2,
        "logits_per_row": [dict(error), dict(error)],
        "kl_divergence_nats_per_row": [0.0001, 0.0001],
        "state_checks": [{"conv": dict(error), "ssm": dict(error)} for _ in range(48)],
        "local_checks": [{"layer": layer, "kernel": kind, "output": dict(error), **({"ssm_state": dict(error)} if kind == "K1" else {})} for layer in range(64) for kind in (("K2",) if layer % 4 == 3 else ("K1", "K2"))],
    }


def test_shadow_accepts_complete_checks(shadow_check):
    from sgl_amd_mono.qwen38_27b.gfx936.check import shadow_accepted

    assert shadow_accepted(shadow_check)


@pytest.mark.parametrize("failure", [None, "missing_attention", "missing_tail", "missing_fp32", "bad_fp32", "bad_kv", "missing_precision", "bad_precision", "missing_native_fp32"])
def test_complete_decode_shadow_requires_all_stages_and_independent_reference(shadow_check, failure):
    from sgl_amd_mono.qwen38_27b.gfx936.check import shadow_accepted

    error = dict(shadow_check["logits_per_row"][0])
    shadow_check.update(full_attention_fused=True, complete_decode_model=True)
    attention = [{"layer": layer, "kernel": "AttentionK1", "output": dict(error), "fp32_output": dict(error), "fa3_fp32_output": dict(error), "precision_output": dict(error), "key_cache": dict(error), "value_cache": dict(error)} for layer in range(3, 64, 4)]
    shadow_check["local_checks"].extend(attention)
    exact = {**error, "normalized_rms_error": 0.0}
    shadow_check["local_checks"][0]["embedding"] = dict(exact)
    shadow_check["local_checks"].append({"layer": 64, "kernel": "Tail", "output": dict(error), "normalized": dict(exact)})
    if failure == "missing_attention":
        shadow_check["local_checks"].remove(attention[0])
    elif failure == "missing_tail":
        shadow_check["local_checks"].pop()
    elif failure == "missing_fp32":
        attention[0].pop("fp32_output")
    elif failure == "bad_fp32":
        attention[0]["fp32_output"]["normalized_rms_error"] = 0.007
    elif failure == "bad_kv":
        attention[0]["value_cache"]["normalized_rms_error"] = 0.004
    elif failure == "missing_precision":
        attention[0].pop("precision_output")
    elif failure == "bad_precision":
        attention[0]["precision_output"]["normalized_rms_error"] = 0.004
    elif failure == "missing_native_fp32":
        attention[0].pop("fa3_fp32_output")
    assert shadow_accepted(shadow_check) == (failure is None)


def test_layer_snapshot_preserves_inputs_before_inplace_native_updates():
    import torch
    from sgl_amd_mono.qwen38_27b.gfx936.runner import MonoRuntime

    runtime = object.__new__(MonoRuntime)
    runtime.enabled = False
    runtime.capture_shadow_inputs = True
    runtime.shadow_phase = "native"
    runtime.shadow_native_inputs = {}
    runtime.shadow_native_outputs = {}
    layer = SimpleNamespace(layer_id=3, o_proj=SimpleNamespace(forward=lambda hidden: (hidden * 2, None)))

    def original(positions, hidden_states, forward_batch, residual, captured_last_layer_outputs):
        hidden_states.add_(1)
        projected, _ = layer.o_proj.forward(hidden_states)
        residual.add_(2)
        return projected, residual

    hidden = torch.tensor([[1.0, 2.0]])
    residual = torch.tensor([[3.0, 4.0]])
    forward = runtime._wrap_layer(layer, original, None, None)
    forward(None, hidden, None, residual)
    captured = runtime.shadow_native_inputs[3]
    assert torch.equal(captured["hidden"], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(captured["residual"], torch.tensor([[3.0, 4.0]]))
    assert torch.equal(captured["core"], torch.tensor([[2.0, 3.0]]))
    residual.zero_()
    assert torch.equal(runtime.shadow_native_outputs[3][1], torch.tensor([[5.0, 6.0]]))


def test_synchronization_failure_is_latched_and_rejects_partial_state(tmp_path, monkeypatch):
    import json
    import torch
    from sgl_amd_mono.qwen38_27b.errors import ContractError
    from sgl_amd_mono.qwen38_27b.gfx936 import runner

    runtime = object.__new__(runner.MonoRuntime)
    failure = torch.tensor(0, dtype=torch.int32)
    runtime.synchronization_errors = [(5, "K1", failure), (5, "K2", torch.tensor(0, dtype=torch.int32))]
    monkeypatch.setattr(runner, "report_directory", lambda: tmp_path)
    runtime.check_synchronization()
    failure.fill_(3)
    for attempt in range(2):
        with pytest.raises(ContractError, match="partially updated state"):
            runtime.check_synchronization()
    report = json.loads((tmp_path / "dense_synchronization_failure.json").read_text())
    assert report["failures"] == [{"layer": 5, "kernel": "K1", "barrier_stage": 3}]
    assert report["stateful_fallback_allowed"] is False


@pytest.mark.parametrize("failure", ["one_request", "nonfinite_state", "missing_kernel", "duplicate_kernel", "empty_batch", "invalid_kl"])
def test_shadow_rejects_partial_or_inaccurate_results(shadow_check, failure):
    from sgl_amd_mono.qwen38_27b.gfx936.check import shadow_accepted

    if failure == "one_request":
        shadow_check["logits_per_row"][1]["normalized_rms_error"] = 0.025
    elif failure == "nonfinite_state":
        shadow_check["local_checks"][0]["ssm_state"]["finite"] = False
    elif failure == "missing_kernel":
        shadow_check["local_checks"].pop()
    elif failure == "duplicate_kernel":
        shadow_check["local_checks"][-1] = copy.deepcopy(shadow_check["local_checks"][0])
    elif failure == "empty_batch":
        shadow_check["rows"] = 0
        shadow_check["logits_per_row"] = []
        shadow_check["kl_divergence_nats_per_row"] = []
    else:
        shadow_check["kl_divergence_nats_per_row"][0] = float("nan")
    assert not shadow_accepted(shadow_check)


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
