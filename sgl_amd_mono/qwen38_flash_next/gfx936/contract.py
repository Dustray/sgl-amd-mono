import hashlib
import os
from importlib import metadata, util
import json
from pathlib import Path

from .. import HF_CONFIG, INDEXER_CONFIG, LAYER_PATTERN
from .program import FlashContractError


def inspect_model(path, tp_size):
    source = Path(path) / "config.json"
    config = json.loads(source.read_text())
    text = config.get("text_config", config)
    reasons = []
    if config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
        reasons.append("requires Qwen4ExpForConditionalGeneration")
    for key, expected in {**HF_CONFIG, **INDEXER_CONFIG}.items():
        if text.get(key) != expected:
            reasons.append(f"{key}: {text.get(key)!r} != {expected!r}")
    if text.get("layer_types") != list(LAYER_PATTERN) * 12:
        reasons.append("requires 48 layers in groups of three GDN and one QSA")
    if config.get("quantization_config") or text.get("quantization_config"):
        reasons.append("requires unquantized BF16 weights")
    if tp_size != 8:
        reasons.append("requires TP8")
    return {"model_path": str(Path(path).resolve()), "architecture": config.get("architectures"), "tp_size": tp_size, "disable_reasons": reasons}


def validate_sources():
    expected = json.loads(Path(__file__).with_name("native_sources.json").read_text())
    package = util.find_spec("sglang")
    if package is None or package.origin is None:
        raise FlashContractError("SGLang is not installed")
    root = Path(package.origin).parent
    errors = []
    hashes = {}
    for package_name, version in expected["versions"].items():
        if metadata.version(package_name) != version:
            errors.append(f"{package_name} version differs from {version}")
    for name, digest in expected["files"].items():
        source = root / name
        actual = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        hashes[name] = actual
        if actual != digest:
            errors.append(f"SGLang source differs: {name}")
    if errors:
        raise FlashContractError("; ".join(errors))
    return {"versions": expected["versions"], "sglang_source_sha256": hashes}


def validate_server(server_args):
    report = inspect_model(server_args.model_path, server_args.tp_size)
    if os.environ.get("SGLANG_AMD_FLASH_NEXT_0518") == "1" and os.environ.get("SGLANG_USE_FUSED_TOPK_SOFTMAX", "0").lower() in ("1", "true"):
        report["disable_reasons"].append("SGLANG_USE_FUSED_TOPK_SOFTMAX=1 is not numerically validated for Flash-Next fusion; use 0")
    unsupported = [name for name in ("enable_lora", "enable_torch_compile", "enable_dp_attention", "enable_dp_lm_head", "enable_deterministic_inference", "speculative_algorithm", "enable_linear_replayssm", "enable_unified_memory", "enable_return_hidden_states", "enable_return_routed_experts", "enable_return_indexer_topk", "ple_offload_embedding", "enable_two_batch_overlap", "enable_single_batch_overlap") if getattr(server_args, name, None)]
    if unsupported:
        report["disable_reasons"].append(f"unsupported execution features: {unsupported}")
    for name in ("pp_size", "moe_ep_size", "attn_cp_size", "dp_size"):
        if getattr(server_args, name, 1) != 1:
            report["disable_reasons"].append(f"requires {name}=1")
    if not getattr(server_args, "disable_overlap_schedule", False):
        report["disable_reasons"].append("requires --disable-overlap-schedule")
    if getattr(server_args, "context_length", None) is None or not 1 <= server_args.context_length <= 4096:
        report["disable_reasons"].append("requires explicit context length <= 4096")
    if getattr(server_args, "max_running_requests", None) not in (1, 2):
        report["disable_reasons"].append("requires max-running-requests 1 or 2")
    if str(getattr(server_args, "dtype", None)) not in ("bfloat16", "bf16"):
        report["disable_reasons"].append("requires explicit BF16 dtype")
    if getattr(server_args, "moe_runner_backend", None) != "triton":
        report["disable_reasons"].append("requires the BF16 triton MoE weight layout")
    if getattr(server_args, "attention_backend", None) != "triton":
        report["disable_reasons"].append("requires --attention-backend triton")
    if getattr(server_args, "page_size", None) != 64:
        report["disable_reasons"].append("requires --page-size 64")
    if getattr(server_args, "kv_cache_dtype", None) not in ("bfloat16", "bf16"):
        report["disable_reasons"].append("requires explicit BF16 KV caches")
    graph_backend = getattr(server_args, "cuda_graph_backend_decode", None)
    if graph_backend == "full":
        batches = getattr(server_args, "cuda_graph_bs_decode", None)
        if not batches or not set(batches).issubset({1, 2}) or 1 not in batches:
            report["disable_reasons"].append("Graph requires explicit --cuda-graph-bs-decode 1 [2]")
        if not getattr(server_args, "disable_cuda_graph_padding", False):
            report["disable_reasons"].append("Graph requires --disable-cuda-graph-padding")
        if os.environ.get("HIP_GRAPH_ACCUMULATE_DISPATCH") != "0":
            report["disable_reasons"].append("Graph requires HIP_GRAPH_ACCUMULATE_DISPATCH=0 before HIP initialization")
    elif graph_backend != "disabled":
        report["disable_reasons"].append("requires decode Graph backend full or disabled")
    if getattr(server_args, "cuda_graph_backend_prefill", "disabled") != "disabled":
        report["disable_reasons"].append("requires --cuda-graph-backend-prefill disabled")
    if not getattr(server_args, "disable_radix_cache", False):
        report["disable_reasons"].append("requires --disable-radix-cache while checkpoint reuse is unvalidated")
    if getattr(server_args, "load_format", None) in ("dummy", "dummy_cpu"):
        report["disable_reasons"].append("requires real checkpoint weights")
    if report["disable_reasons"]:
        raise FlashContractError("; ".join(report["disable_reasons"]))
    report.update(validate_sources())
    return report
