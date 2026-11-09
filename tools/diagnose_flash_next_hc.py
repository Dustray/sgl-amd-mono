import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from safetensors import safe_open

from sgl_amd_mono.qwen38_flash_next.gfx936.program import HCMix, PostArguments, SyncArguments
from sgl_amd_mono.qwen38_flash_next.gfx936.runner import error_metrics


def main():
    from sglang.srt.layers.hyperconnection import GroupedGemmaRMSNorm
    from sglang.srt.layers.moe.topk import fused_topk

    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-Flash-Next/snapshots/master"))
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--deterministic-reference", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.deterministic_reference:
        import os
        from sgl_amd_mono.qwen38_flash_next.gfx936.reference import install

        os.environ["SGLANG_AMD_FLASH_ACCURACY_REFERENCE"] = "1"
        install()
    from sglang.srt.layers.hc_mix_triton import fused_hc_mix
    torch.cuda.set_device(arguments.device)
    device = torch.device("cuda", arguments.device)
    capture = torch.load(arguments.capture, map_location=device, weights_only=True)
    mapping = json.loads((arguments.model_path / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(suffix):
        name = f"model.language_model.layers.{arguments.layer}.{suffix}"
        with safe_open(arguments.model_path / mapping[name], framework="pt", device="cpu") as checkpoint:
            return checkpoint.get_tensor(name).contiguous().to(device)

    mixer = HCMix(*(weight(f"mlp_hyper_connection.{name}.weight") for name in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up")))
    norm = GroupedGemmaRMSNorm(10240, group_size=2560).to(device=device, dtype=torch.bfloat16)
    norm.weight = torch.nn.Parameter(mixer.weights[0], requires_grad=False)
    router = weight("mlp.gate.weight")
    after_attention = capture["intermediates"]["post_attention"]
    native_mixed = capture["intermediates"]["mixed"]
    actual_mixed = capture["actual_mixed"]

    def route(mixed):
        logits = torch.nn.functional.linear(mixed, router)
        probabilities, indices, *_ = fused_topk(mixed, logits, 10, True)
        sorted_logits = logits.float().sort(descending=True).values
        return {"top_ids": indices.cpu().tolist(), "top_weights": probabilities.cpu().tolist(), "boundary_margin": (sorted_logits[:, 9] - sorted_logits[:, 10]).cpu().tolist()}, logits

    report = {"capture": str(arguments.capture), "capture_sha256": hashlib.sha256(arguments.capture.read_bytes()).hexdigest(), "layer": arguments.layer, "captured_mixed": error_metrics(actual_mixed, native_mixed), "captured_moe": error_metrics(capture["actual_moe"], capture["intermediates"]["reduced_moe"]), "captured_output": error_metrics(capture["actual"], capture["expected"]), "captured_fused_route": capture["top_ids"].cpu().tolist(), "checks": []}
    for name, mixed in (("captured_fused", actual_mixed), ("captured_native", native_mixed)):
        report[name], logits = route(mixed)
        report[name]["router_logits"] = logits.float().cpu().tolist()
    normalized = norm(after_attention)
    reference_outputs = []
    for repeat in range(arguments.repeats):
        mixed = fused_hc_mix(normalized, mixer.weights[1], mixer.weights[2], 4, 2560)
        check, _ = route(mixed)
        check.update(kind="native_repeat", repeat=repeat, mixed=error_metrics(mixed, native_mixed), first_repeat=error_metrics(mixed, reference_outputs[0] if reference_outputs else mixed))
        report["checks"].append(check)
        reference_outputs.append(mixed.clone())
    actual_outputs = []
    for repeat in range(3):
        mixed, _ = mixer(after_attention)
        check, _ = route(mixed)
        check.update(kind="fused_same_post_attention", repeat=repeat, mixed=error_metrics(mixed, native_mixed), first_repeat=error_metrics(mixed, actual_outputs[0] if actual_outputs else mixed), first_native=error_metrics(mixed, reference_outputs[0]), normalized=error_metrics(mixer.normalized[:mixed.shape[0]], normalized))
        report["checks"].append(check)
        actual_outputs.append(mixed.clone())
    report["native_distinct_routes"] = sorted({tuple(check["top_ids"][0]) for check in report["checks"] if check["kind"] == "native_repeat"})
    source = Path(__file__).with_name("diagnose_flash_next_inject.cpp")
    kernel_path = Path(__file__).resolve().parents[1] / "sgl_amd_mono/qwen38_flash_next/gfx936/kernel.cpp"
    report["combine_source_sha256"] = hashlib.sha256(kernel_path.read_bytes()).hexdigest()
    digest = hashlib.sha256(source.read_bytes() + kernel_path.read_bytes()).hexdigest()
    cache = arguments.output.parent / ("combine-probe-" + digest[:24])
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / "combine.so"
    if not binary.exists():
        (cache / "kernel.cpp").write_bytes(kernel_path.read_bytes())
        command = ["/opt/dtk/bin/hipcc", "-O3", "-std=c++17", "-shared", "-fPIC", "--offload-arch=gfx936", "-ffp-contract=off", "-I", str(cache.resolve()), str(source), "-o", str(binary)]
        with (cache / "compile.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    library = ctypes.CDLL(str(binary.resolve()))
    library.flash_probe_combine.argtypes = [ctypes.POINTER(PostArguments), ctypes.c_void_p]
    inject_weight = weight("attn_hyper_connection.block_inject_weight.weight")
    combined = torch.empty_like(after_attention)
    gates = torch.empty((after_attention.shape[0], 4), dtype=torch.float32, device=device)
    state = torch.zeros(16, dtype=torch.int32, device=device)
    probe = PostArguments()
    for name, tensor in (("attention", capture["intermediates"]["reduced_attention"]), ("residual", capture["residuals"][0]), ("normalized_residual", capture["residuals"][1]), ("attention_inject", inject_weight), ("inject_gates", gates), ("post_attention", combined)):
        setattr(probe, name, tensor.data_ptr())
    probe.rows = after_attention.shape[0]
    probe.sync = SyncArguments((ctypes.c_void_p * 8)(), state.data_ptr(), 0, mixer.program.timeout_ticks)
    mixer.program.check(library.flash_probe_combine(ctypes.byref(probe), torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize(device)
    mixer.program.check(0 if int(state[2]) == 0 else -1)
    combined_mixed, _ = mixer(combined)
    combine_route, _ = route(combined_mixed)
    report["combine_replay"] = {"post_attention": error_metrics(combined, after_attention), "mixed_vs_captured_native": error_metrics(combined_mixed, native_mixed), "mixed_vs_captured_fused": error_metrics(combined_mixed, actual_mixed), "route": combine_route}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: value for name, value in report.items() if name not in {"captured_fused", "captured_native"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
