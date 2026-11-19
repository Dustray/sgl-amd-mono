import hashlib
import os
from importlib import metadata, util
import json
from pathlib import Path
from typing import Optional

from .. import HF_CONFIG, INDEXER_CONFIG, LAYER_PATTERN
from .geometry import DRAFT_TOKENS
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


QSA_CONTEXT_CAP = 262144
# MTP (EAGLE) draft geometry. The fused verify path always evaluates the whole draft
# chain in one target pass, so these pins are what size the verify rows per request; a
# server launched with any other shape would need kernel changes, not config changes.
SPEC_ALGORITHMS = ("EAGLE", "NEXTN")
SPEC_EAGLE_TOPK = 1
SPEC_NUM_STEPS = 3
SPEC_NUM_DRAFT_TOKENS = DRAFT_TOKENS


def spec_algorithm_name(value) -> Optional[str]:
    """Normalise ``server_args.speculative_algorithm`` (enum, string, or None).

    ``before_resolve`` runs before SGLang's own NEXTN -> EAGLE alias hook, so the
    raw flag spelling has to be accepted here too.
    """
    if value is None:
        return None
    name = str(getattr(value, "name", value)).upper()
    return None if name in ("", "NONE") else name


def spec_enabled(server_args) -> bool:
    return spec_algorithm_name(getattr(server_args, "speculative_algorithm", None)) is not None


def spec_fused_requested() -> bool:
    """Whether the fused TARGET_VERIFY path is enabled. Off by default: with the
    env unset a speculative server keeps installing no runtime at all and decodes
    fully natively (P0 behaviour); with it set, the mono runtime installs on the
    target runner and routes verify batches (4 rows per request) through the fused
    kernels. Capturing a verify batch into a decode graph is unsupported (graph.py
    only captures plain decode), so verify always runs eager; see
    ``spec_verify_graph_policy`` for why that does not mean disabling the graph."""
    return os.environ.get("SGLANG_AMD_FLASH_SPEC_VERIFY", "0") == "1"


def spec_verify_enabled(server_args) -> bool:
    """Whether this server routes TARGET_VERIFY batches through the mono kernels.

    Gated on the server really being speculative, not just on the env: the plugin reads
    this to decide whether to skip decode-graph capture, and a plain decode server that
    happens to have the env exported must keep its (validated) decode graph.
    """
    return spec_fused_requested() and spec_enabled(server_args)


def spec_verify_graph_requested() -> bool:
    """Opt-in: capture the fused TARGET_VERIFY batch into a decode graph.

    Off by default, so a normal fused-verify launch keeps verifying eagerly (the
    validated path). On top of ``SGLANG_AMD_FLASH_SPEC_VERIFY=1`` it installs a
    ``GraphExecution`` on the spec *target* runner and lets SGLang's own
    ``DecodeCudaGraphRunner`` capture the verify batch in TARGET_VERIFY mode (bs x
    DRAFT_TOKENS rows) and replay it; the mono kernels run because the target's
    ``model.forward`` (the fused one) executes during capture. Experimental: the first
    replay is validated bit-for-bit against an eager mono verify (see
    ``graph.GraphExecution``), and the plugin fails the launch loudly if SGLang's
    replay-time ``init_forward_metadata_out_graph`` does not refresh every input the mono
    kernels captured (``mamba_cache_indices``, PLE verify statics). It never serves a
    mismatched verify silently.
    """
    return os.environ.get("SGLANG_AMD_FLASH_SPEC_VERIFY_GRAPH", "0") == "1"


def spec_verify_graph_enabled(server_args) -> bool:
    """A graphed fused verify needs a speculative fused-verify target with decode Graph on."""
    return spec_verify_enabled(server_args) and spec_verify_graph_requested() and getattr(server_args, "cuda_graph_backend_decode", None) == "full"


def spec_verify_graph_policy(graph_backend, verify_graph=False) -> str:
    """What the decode-graph flag means for a fused-verify server.

    One MTP round is 3 draft forwards plus 1 target verify forward, and SGLang builds
    the draft's graph runners from this same flag (``_capture_cuda_graphs`` returns
    early on ``disabled``). So ``disabled`` would drop the graph from all four
    forwards, including the three draft steps that run native kernels either way.
    With ``full`` and the verify-graph flag on, the plugin also captures the verify batch
    on the *target* runner and replays it, while the draft keeps its graphs. With ``full``
    but the flag off, the plugin still skips target capture: nothing is captured there,
    ``can_run_graph`` finds no shape and verify falls back to eager, where the mono
    kernels run. Prefer ``full``.
    """
    if graph_backend != "full":
        return "no graphs anywhere (draft included)"
    return "verify captured into the target graph, draft graphs kept" if verify_graph else "target capture skipped, draft graphs kept"


def validate_server(server_args):
    report = inspect_model(server_args.model_path, server_args.tp_size)
    mono = os.environ.get("SGLANG_AMD_FLASH_NEXT_0518") == "1"
    if mono and os.environ.get("SGLANG_USE_FUSED_TOPK_SOFTMAX", "0").lower() in ("1", "true"):
        report["disable_reasons"].append("SGLANG_USE_FUSED_TOPK_SOFTMAX=1 is not numerically validated for Flash-Next fusion; use 0")
    unsupported = [name for name in ("enable_lora", "enable_torch_compile", "enable_dp_attention", "enable_dp_lm_head", "enable_deterministic_inference", "enable_linear_replayssm", "enable_unified_memory", "enable_return_hidden_states", "enable_return_routed_experts", "enable_return_indexer_topk", "ple_offload_embedding", "enable_two_batch_overlap", "enable_single_batch_overlap") if getattr(server_args, name, None)]
    if unsupported:
        report["disable_reasons"].append(f"unsupported execution features: {unsupported}")
    for name in ("pp_size", "moe_ep_size", "attn_cp_size", "dp_size"):
        if getattr(server_args, name, 1) != 1:
            report["disable_reasons"].append(f"requires {name}=1")
    if not getattr(server_args, "disable_overlap_schedule", False):
        report["disable_reasons"].append("requires --disable-overlap-schedule")
    # The QSA index top-k sorts 4096-token chunks and merges up to 64 of them (qsa.cpp QSAChunkLists);
    # native decode has no such limit.
    if mono and (getattr(server_args, "context_length", None) is None or not 1 <= server_args.context_length <= QSA_CONTEXT_CAP):
        report["disable_reasons"].append(f"requires explicit context length <= {QSA_CONTEXT_CAP}")
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
    # MTP: only the pinned single-chain EAGLE geometry is accepted. Without
    # SGLANG_AMD_FLASH_SPEC_VERIFY=1 the runtime stays out of the loop and the server
    # decodes fully natively ("fused": False below); with it, verify batches go
    # through the mono kernels on the target runner (eager only for now).
    spec_name = spec_algorithm_name(getattr(server_args, "speculative_algorithm", None))
    if spec_name is not None:
        speculative = {
            "algorithm": spec_name,
            "eagle_topk": getattr(server_args, "speculative_eagle_topk", None),
            "num_steps": getattr(server_args, "speculative_num_steps", None),
            "num_draft_tokens": getattr(server_args, "speculative_num_draft_tokens", None),
            "fused": spec_fused_requested(),
        }
        if speculative["fused"]:
            speculative["verify_graph"] = spec_verify_graph_policy(graph_backend, spec_verify_graph_requested())
        report["speculative"] = speculative
        if spec_name not in SPEC_ALGORITHMS:
            report["disable_reasons"].append(f"speculative_algorithm {spec_name} requires EAGLE (NEXTN) with the pinned MTP chain")
        if speculative["fused"] and bool(getattr(server_args, "enable_linear_replayssm_spec", False)):
            report["disable_reasons"].append("fused speculative verify needs the per-draft GDN scratch, disabled by --enable-linear-replayssm-spec")
        for attribute, want in (
            ("speculative_eagle_topk", SPEC_EAGLE_TOPK),
            ("speculative_num_steps", SPEC_NUM_STEPS),
            ("speculative_num_draft_tokens", SPEC_NUM_DRAFT_TOKENS),
        ):
            got = getattr(server_args, attribute, None)
            if got != want:
                report["disable_reasons"].append(f"requires --{attribute.replace('_', '-')} {want}, got {got}")
        for attribute in ("speculative_draft_model_path", "speculative_token_map"):
            if getattr(server_args, attribute, None):
                report["disable_reasons"].append(f"requires no --{attribute.replace('_', '-')}")
        for attribute in ("enable_multi_layer_eagle", "speculative_use_rejection_sampling", "speculative_skip_dp_mlp_sync", "speculative_adaptive"):
            if bool(getattr(server_args, attribute, False)):
                report["disable_reasons"].append(f"requires {attribute.replace('_', '-')} disabled")
    if report["disable_reasons"]:
        raise FlashContractError("; ".join(report["disable_reasons"]))
    report.update(validate_sources())
    return report
