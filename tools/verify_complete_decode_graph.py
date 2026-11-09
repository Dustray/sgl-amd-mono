import argparse
import ctypes
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HIP_GRAPH_ACCUMULATE_DISPATCH", "0")

import torch
from safetensors import safe_open

from sgl_amd_mono.qwen38_27b.gfx936.attention import DenseAttentionK1
from sgl_amd_mono.qwen38_27b.gfx936.decode_io import DenseEntryK1, DenseTail
from sgl_amd_mono.qwen38_27b.gfx936.program import DenseK1, DenseK2
from sgl_amd_mono.qwen38_27b.report import implementation_digest, write_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20)
    arguments = parser.parse_args()
    index = json.loads((arguments.model_path / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(name):
        with safe_open(arguments.model_path / index[name], framework="pt", device="cpu") as checkpoint:
            return checkpoint.get_tensor(name).cuda()

    def layer_weight(layer, suffix):
        return weight(f"model.language_model.layers.{layer}.{suffix}")

    embedding = weight("model.language_model.embed_tokens.weight")
    tail = DenseTail(weight("model.language_model.norm.weight"), weight("lm_head.weight"))
    gdn_weights = (layer_weight(0, "input_layernorm.weight"), torch.cat([layer_weight(0, "linear_attn.in_proj_qkv.weight"), layer_weight(0, "linear_attn.in_proj_z.weight")]), torch.cat([layer_weight(0, "linear_attn.in_proj_b.weight"), layer_weight(0, "linear_attn.in_proj_a.weight")]), layer_weight(0, "linear_attn.conv1d.weight"), layer_weight(0, "linear_attn.norm.weight"), layer_weight(0, "linear_attn.A_log").float(), layer_weight(0, "linear_attn.dt_bias"))
    attention_weights = (layer_weight(3, "input_layernorm.weight"), torch.cat([layer_weight(3, f"self_attn.{name}_proj.weight") for name in ("q", "k", "v")]), layer_weight(3, "self_attn.q_norm.weight"), layer_weight(3, "self_attn.k_norm.weight"))
    k2_weights = (layer_weight(0, "linear_attn.out_proj.weight"), layer_weight(0, "post_attention_layernorm.weight"), torch.cat([layer_weight(0, f"mlp.{name}_proj.weight") for name in ("gate", "up")]), layer_weight(0, "mlp.down_proj.weight"))
    entries, kernels = [], []
    for layer in range(64):
        kernel1 = DenseAttentionK1(*attention_weights) if layer % 4 == 3 else DenseEntryK1(*gdn_weights) if layer == 0 else DenseK1(*gdn_weights)
        kernel2 = DenseK2(*k2_weights)
        entries.append((kernel1, kernel2))
        kernels.extend((kernel1, kernel2))
    kernels.append(tail)
    angles = torch.arange(8192, device="cuda", dtype=torch.float32)[:, None] / (10000000.0 ** (torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64))
    rotary = torch.cat((angles.cos(), angles.sin()), dim=1).bfloat16()
    report = {"scope": "complete 129-launch decode Graph structure with distinct layer workspaces, real embedding/head and reused real layer-0/3 weights; exact eager/replay outputs, state and KV; does not substitute for model accuracy", "implementation_sha256": implementation_digest(), "passed": False, "checks": []}

    def save():
        write_report(arguments.output.parent, report, arguments.output.name)

    def wait():
        event = torch.cuda.Event()
        event.record()
        deadline = time.monotonic() + arguments.timeout
        while not event.query():
            if time.monotonic() > deadline:
                raise RuntimeError("Complete decode Graph did not finish before deadline")
            time.sleep(0.005)
        failures = [(index, int(kernel.synchronization[2].item())) for index, kernel in enumerate(kernels) if int(kernel.synchronization[2].item())]
        if failures:
            raise RuntimeError(f"Complete decode Graph device failure latches: {failures}")

    def counters():
        return torch.stack([kernel.executions for kernel in kernels]).cpu().clone()

    def forward(batch):
        rows = batch["ids"].numel()
        hidden = torch.empty((rows, 5120), device="cuda", dtype=torch.bfloat16)
        residual = None
        gdn_index, attention_index = 0, 0
        for layer, (kernel1, kernel2) in enumerate(entries):
            if layer % 4 == 3:
                core, residual = kernel1(hidden, residual, batch["positions"], rotary, batch["key"][attention_index], batch["value"][attention_index], batch["locations"], batch["table"], batch["lengths"], request_indices=batch["requests"])
                attention_index += 1
            else:
                extra = {"embedding_weight": embedding, "input_ids": batch["ids"]} if layer == 0 else {}
                core, residual = kernel1(hidden, residual, batch["convolution"][gdn_index], batch["state"][gdn_index], batch["indices"], **extra)
                gdn_index += 1
            hidden, residual = kernel2(core, residual)
        return tail(hidden, residual)

    def metadata(batch, length, step):
        rows = batch["ids"].numel()
        lengths = [length, max(1, length - 17)][:rows]
        batch["ids"].copy_(torch.tensor([(271 + step * 17 + row * 23) % embedding.shape[0] for row in range(rows)], device="cuda"))
        batch["lengths"].copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        batch["positions"].copy_(torch.tensor([[value - 1 + axis * 3 for value in lengths] for axis in range(3)], device="cuda"))
        batch["locations"].copy_(torch.tensor([(1 + row * 64 + (value - 1) // 64) * 64 + (value - 1) % 64 for row, value in enumerate(lengths)], device="cuda"))

    batches = {}
    save()
    try:
        for rows in (2, 1):
            batch = {"ids": torch.zeros(rows, device="cuda", dtype=torch.int64), "indices": torch.arange(rows, device="cuda", dtype=torch.int32), "requests": torch.arange(1, rows + 1, device="cuda", dtype=torch.int64), "lengths": torch.empty(rows, device="cuda", dtype=torch.int32), "locations": torch.empty(rows, device="cuda", dtype=torch.int64), "positions": torch.empty((3, rows), device="cuda", dtype=torch.int64), "table": torch.arange(1, rows * 64 + 1, device="cuda", dtype=torch.int32).view(rows, 64), "convolution": torch.zeros((48, 2, 10240, 3), device="cuda", dtype=torch.bfloat16), "state": torch.zeros((48, 2, 48, 128, 128), device="cuda"), "key": torch.zeros((16, 129, 64, 4, 256), device="cuda", dtype=torch.float8_e5m2), "value": torch.zeros((16, 129, 64, 4, 256), device="cuda", dtype=torch.float8_e5m2)}
            metadata(batch, 64, 0)
            forward(batch)
            wait()
            before = counters()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                output = forward(batch)
            assert torch.equal(counters(), before)
            graph.instantiate()
            batch.update(graph=graph, output=output)
            batches[rows] = batch
            try:
                runtime = ctypes.CDLL("libamdhip64.so")
                runtime.hipGraphDebugDotPrint.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
                path = arguments.output.with_name(f"{arguments.output.stem}-bs{rows}.dot")
                status = runtime.hipGraphDebugDotPrint(graph.raw_cuda_graph(), os.fsencode(path), 0)
                report[f"graph_{rows}_dot"] = {"path": str(path), "hip_status": status}
            except (AttributeError, OSError, TypeError) as error:
                report[f"graph_{rows}_dot"] = {"unavailable": str(error)}
            save()
        for round_index in range(arguments.rounds):
            for rows in (2, 1, 1, 2):
                batch = batches[rows]
                length = (65, 513, 2704, 4096)[round_index % 4]
                metadata(batch, length, round_index + 1)
                original = {name: batch[name].clone() for name in ("convolution", "state", "key", "value")}
                expected = forward(batch).clone()
                expected_states = {name: batch[name].clone() for name in original}
                wait()
                for name, saved in original.items():
                    batch[name].copy_(saved)
                initial = counters()
                batch["graph"].replay()
                wait()
                equal = torch.equal(batch["output"], expected) and bool(torch.isfinite(batch["output"]).all())
                states_equal = all(torch.equal(batch[name].view(torch.uint8), expected_states[name].view(torch.uint8)) for name in original)
                delta = counters() - initial
                check = {"round": round_index, "rows": rows, "length": length, "output_exact": equal, "states_and_kv_exact": states_equal, "launches": int(delta.sum()), "each_kernel_once": bool((delta == 1).all())}
                report["checks"].append(check)
                save()
                print(json.dumps(check), flush=True)
                if not equal or not states_equal or not check["each_kernel_once"] or check["launches"] != 129:
                    raise RuntimeError(f"Complete decode Graph mismatch: {check}")
        report["passed"] = True
        save()
    except Exception as error:
        report["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
