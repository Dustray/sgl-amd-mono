import ast
import hashlib
from importlib import metadata, resources, util
import json
import os
from pathlib import Path


HOOK_TARGETS = (
    "sglang.srt.server_args.ServerArgs.resolve_once",
    "sglang.srt.model_loader.loader.DefaultModelLoader.load_weights_and_postprocess",
    "sglang.srt.model_executor.model_runner.ModelRunner.initialize",
    "sglang.srt.managers.scheduler.Scheduler.get_internal_state",
)


class ContractError(RuntimeError):
    pass


class UnsupportedConfiguration(RuntimeError):
    pass


def manifest():
    return json.loads(
        resources.files("sgl_amd_mono_0518").joinpath("manifest.json").read_text()
    )


def environment_contract():
    expected = manifest()
    package = util.find_spec("sglang")
    if package is None or package.origin is None:
        return {"errors": ["SGLang is not installed"], "file_hashes": {}}
    root = Path(package.origin).parent
    actual_version = metadata.version("sglang")
    errors = []
    if actual_version != expected["sglang_version"]:
        errors.append(f"SGLang version {actual_version} != {expected['sglang_version']}")
    hashes = {}
    for relative, expected_hash in expected["sglang_files"].items():
        source = root / relative
        if not source.is_file():
            errors.append(f"Missing SGLang source: {relative}")
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        hashes[relative] = digest
        if digest != expected_hash:
            errors.append(f"SGLang source hash mismatch: {relative}")
    contracts = (
        ("srt/server_args.py", "ServerArgs", "resolve_once", ("self",), False),
        (
            "srt/model_loader/loader.py",
            "DefaultModelLoader",
            "load_weights_and_postprocess",
            ("model", "weights", "target_device"),
            True,
        ),
        ("srt/model_executor/model_runner.py", "ModelRunner", "initialize", ("self",), False),
        ("srt/managers/scheduler.py", "Scheduler", "get_internal_state", ("self", "recv_req"), False),
    )
    for relative, class_name, method_name, arguments, is_static in contracts:
        source = root / relative
        if not source.is_file():
            continue
        tree = ast.parse(source.read_text())
        owner = next(
            (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name),
            None,
        )
        method = next(
            (
                node
                for node in owner.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name
            ),
            None,
        ) if owner is not None else None
        if method is None:
            errors.append(f"Missing hook target: {class_name}.{method_name}")
            continue
        actual_arguments = tuple(argument.arg for argument in method.args.args)
        if actual_arguments != arguments:
            errors.append(f"Hook signature drift: {class_name}.{method_name}: {actual_arguments}")
        static = any(
            isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
            for decorator in method.decorator_list
        )
        if static != is_static:
            errors.append(f"Hook descriptor drift: {class_name}.{method_name}")
    return {
        "errors": errors,
        "sglang_version": actual_version,
        "file_hashes": hashes,
        "hook_targets": list(HOOK_TARGETS),
        "upstream_framework_present": (root / "srt/monokernels/spec.py").is_file()
        or (root / "kernels/plugin_kernels/amd/spec.py").is_file(),
    }


def device_environment():
    import torch

    available = torch.cuda.is_available()
    properties = torch.cuda.get_device_properties(0) if available else None
    flydsl_installed = util.find_spec("flydsl") is not None
    return {
        "torch_version": metadata.version("torch"),
        "hip_version": torch.version.hip,
        "device_count": torch.cuda.device_count(),
        "gpu_arch": getattr(properties, "gcnArchName", "unavailable").split(":")[0],
        "gpu_compute_units": getattr(properties, "multi_processor_count", None),
        "flydsl_installed": flydsl_installed,
        "flydsl_version": metadata.version("flydsl") if flydsl_installed else None,
    }


def probe(model_path, tp_size=1, device=None):
    model_path = Path(model_path).resolve()
    config = json.loads((model_path / "config.json").read_text())
    contract = environment_contract()
    device = device_environment() if device is None else device
    upstream = manifest()["upstream"]
    target = upstream["document_target"]
    architectures = config.get("architectures", [])
    text = config.get("text_config", config)
    reasons = []
    if device["gpu_arch"] != target["gpu_arch"]:
        reasons.append(f"GPU {device['gpu_arch']}: the documented kernel requires gfx950")
    if target["architecture"] not in architectures or config.get("model_type") != target["model_type"]:
        reasons.append(
            f"No documented kernel spec for {architectures}, model_type={config.get('model_type')}; "
            "the pinned revision covers DeepSeek-V4.1-Flash"
        )
    if tp_size != target["tp_size"]:
        reasons.append(f"TP{tp_size}: the documented MVP requires TP4")
    if not device["flydsl_installed"]:
        reasons.append("FlyDSL is not installed")
    if not contract["upstream_framework_present"]:
        reasons.append("The installed SGLang lacks the upstream Monokernel framework API")
    reasons.append(
        "The documented gfx950 Monokernel spec is not bundled; local gfx936 dense support is checked separately"
    )
    ffn_reasons = []
    if device["gpu_arch"] != "gfx936":
        ffn_reasons.append("BF16 FFN fusion has only been validated on gfx936")
    expected_flydsl = manifest()["validated_local_flydsl"]["version"]
    if device.get("flydsl_version") != expected_flydsl:
        ffn_reasons.append(f"BF16 FFN fusion requires validated FlyDSL {expected_flydsl}")
    if (
        architectures != ["Qwen3_5ForConditionalGeneration"]
        or config.get("model_type") != "qwen3_5"
        or text.get("hidden_size") != 5120
        or text.get("intermediate_size") != 17408
        or text.get("num_hidden_layers") != 64
        or text.get("hidden_act") != "silu"
        or text.get("dtype", text.get("torch_dtype")) != "bfloat16"
        or config.get("quantization_config")
        or text.get("quantization_config")
    ):
        ffn_reasons.append("BF16 FFN fusion requires the validated dense Qwen3_5 BF16 27B configuration")
    if tp_size != 1:
        ffn_reasons.append("BF16 FFN fusion currently requires TP1")
    mono_reasons = [reason for reason in ffn_reasons if "FlyDSL" not in reason]
    if device.get("gpu_compute_units") != 80:
        mono_reasons.append("Dense cooperative Monokernel requires the validated 80-CU grid")
    expected_dimensions = {"linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256, "rms_norm_eps": 1e-6}
    if any(text.get(name) != value for name, value in expected_dimensions.items()):
        mono_reasons.append("Dense Monokernel attention/state dimensions or RMSNorm epsilon mismatch")
    specs = manifest()["bundled_monokernel_specs"]
    if not any(spec.get("name") == "gfx936_bf16_dense_qwen" for spec in specs):
        mono_reasons.append("Dense BF16 Monokernel spec is missing")
    experimental_dense = os.environ.get("SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE") == "1"
    if not experimental_dense:
        mono_reasons.append("The dense BF16 prototype has not passed acceptance; experimental runs require SGLANG_AMD_MONO_0518_ALLOW_EXPERIMENTAL_DENSE=1")
    return {
        "contract": contract,
        "device": device,
        "model": {
            "path": str(model_path),
            "config_sha256": hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest(),
            "architectures": architectures,
            "model_type": config.get("model_type"),
            "dtype": text.get("dtype", text.get("torch_dtype")),
            "num_hidden_layers": text.get("num_hidden_layers"),
            "hidden_size": text.get("hidden_size"),
            "quantization_config": config.get("quantization_config"),
            "tp_size": tp_size,
        },
        "upstream": upstream,
        "monokernel_eligible": not mono_reasons,
        "monokernel_acceptance": "not_accepted_experimental",
        "experimental_dense_opt_in": experimental_dense,
        "disable_reasons": mono_reasons,
        "documented_upstream_disable_reasons": reasons,
        "execution_path": "sglang_original",
        "monokernel_executions": 0,
        "fused_ffn_eligible": not ffn_reasons,
        "fused_ffn_disable_reasons": ffn_reasons,
    }


def validate(report, mode):
    if report["contract"]["errors"]:
        raise ContractError("; ".join(report["contract"]["errors"]))
    if mode == "required" and report.get("monokernel_acceptance") != "accepted":
        raise UnsupportedConfiguration("Strict Monokernel mode refuses the unaccepted dense BF16 prototype; see LOCAL_DEVELOPMENT.md")
    if mode in {"required", "monokernel"} and not report["monokernel_eligible"]:
        raise UnsupportedConfiguration("Monokernel startup refused:\n- " + "\n- ".join(report["disable_reasons"]))
    if mode == "fused_ffn" and not report["fused_ffn_eligible"]:
        raise UnsupportedConfiguration("BF16 FFN fusion refused:\n- " + "\n- ".join(report["fused_ffn_disable_reasons"]))
