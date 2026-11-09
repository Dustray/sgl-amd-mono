import argparse
import json
import math
from pathlib import Path

import torch

from sgl_amd_mono.qwen38_27b.gfx936.ffn import numerical_error


def round_bf16(value):
    return ((value.contiguous().view(torch.int32) + 32768) >> 16).to(torch.int16).view(torch.bfloat16).float()


def main():
    from sglang.srt.layers.attention.flashattention_interface import vllm_flash_attn_with_kvcache

    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 4096])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--layer", type=int, default=31)
    arguments = parser.parse_args()
    report = {"scope": "native paged attention intermediate precision diagnosis", "checks": []}
    for length in arguments.lengths:
        generator = torch.Generator(device="cuda").manual_seed(20261105)
        pages = (length + 63) // 64
        query = torch.randn((1, 1, 24, 256), device="cuda", dtype=torch.bfloat16, generator=generator)
        key = torch.randn((pages + 1, 4, 64, 256), device="cuda", dtype=torch.bfloat16, generator=generator).to(torch.float8_e5m2)
        value = torch.randn(key.shape, device="cuda", dtype=torch.bfloat16, generator=generator).to(torch.float8_e5m2)
        table = torch.arange(1, pages + 1, device="cuda", dtype=torch.int32)[None, :]
        lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
        if arguments.capture:
            capture = torch.load(arguments.capture, map_location="cuda", weights_only=True)
            if capture["hidden"].shape[0] != 1:
                raise ValueError("Capture diagnosis requires one row")
            generator.manual_seed(20261105 + arguments.layer + 1 + length)
            torch.randn((1, 5120), device="cuda", dtype=torch.bfloat16, generator=generator)
            torch.randn((1, 5120), device="cuda", dtype=torch.bfloat16, generator=generator)
            canonical_key = torch.randn((132, 64, 4, 256), device="cuda", dtype=torch.bfloat16, generator=generator).to(torch.float8_e5m2)
            canonical_value = torch.randn((132, 64, 4, 256), device="cuda", dtype=torch.bfloat16, generator=generator).to(torch.float8_e5m2)
            table = torch.randperm(128, device="cuda", generator=generator).view(2, 64)[:1].int() + 1
            location = int(table[0, (length - 1) // 64]) * 64 + (length - 1) % 64
            canonical_key[location // 64, location % 64] = capture["stages"]["key"].reshape(4, 256).to(torch.float8_e5m2)
            canonical_value[location // 64, location % 64] = capture["stages"]["value"].reshape(4, 256).to(torch.float8_e5m2)
            key = canonical_key.permute(0, 2, 1, 3).contiguous()
            value = canonical_value.permute(0, 2, 1, 3).contiguous()
            query = capture["stages"]["query"].reshape(1, 1, 24, 256)
        native_value = value.transpose(2, 3).contiguous()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
            expected = vllm_flash_attn_with_kvcache(q=query, k_cache=key, v_cache=native_value, page_table=table, cache_seqlens=lengths, softmax_scale=0.0625, causal=True, ver=3).reshape(4, 6, 256)
            torch.cuda.synchronize()
        kernels = sorted({event.name for event in profiler.events() if event.device_type == torch.autograd.DeviceType.CUDA})
        selected_pages = table[0, :pages].long()
        keys = key[selected_pages].permute(0, 2, 1, 3).flatten(0, 1)[:length].float()
        values = value[selected_pages].permute(0, 2, 1, 3).flatten(0, 1)[:length].float()
        scores = torch.einsum("hgd,shd->hgs", query.reshape(4, 6, 256).float(), keys) * (math.log2(math.e) / 128)
        scores = scores.half().float() * 8
        variants = {}
        for partition in [length, 64, 128, 256, 512, 1024, 2048]:
            if partition > length:
                continue
            for bf16_probabilities in (False, True):
                for bf16_partials in (False, True):
                    outputs, weights = [], []
                    maximum = scores.amax(-1, keepdim=True)
                    for start in range(0, length, partition):
                        piece = scores[..., start:start + partition]
                        local_maximum = piece.amax(-1, keepdim=True)
                        probabilities = torch.exp2(piece - local_maximum)
                        denominator = probabilities.sum(-1, keepdim=True)
                        probabilities /= denominator + 1.0e-6
                        if bf16_probabilities:
                            probabilities = round_bf16(probabilities)
                        partial = torch.einsum("hgs,shd->hgd", probabilities, values[start:start + partition])
                        if bf16_partials:
                            partial = round_bf16(partial)
                        outputs.append(partial)
                        weights.append(denominator * torch.exp2(local_maximum - maximum))
                    denominator = torch.stack(weights).sum(0)
                    actual = sum(output * weight / denominator for output, weight in zip(outputs, weights))
                    error = numerical_error(round_bf16(actual), expected)
                    variants[f"partition_{partition}_pbf16_{bf16_probabilities}_obf16_{bf16_partials}"] = error["normalized_rms_error"]
        best = sorted(variants.items(), key=lambda item: item[1])[:8]
        check = {"length": length, "kernels": kernels, "variants": variants, "best": best}
        report["checks"].append(check)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"length": length, "kernels": kernels, "best": best}), flush=True)


if __name__ == "__main__":
    main()
