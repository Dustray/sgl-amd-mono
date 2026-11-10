import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HIP_GRAPH_ACCUMULATE_DISPATCH", "0")

import torch
from safetensors import safe_open

from sgl_amd_mono.qwen38_27b.gfx936.program import DenseK1, DenseK2
from sgl_amd_mono.qwen38_27b.report import implementation_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--weight-sets", type=int, choices=(1, 2, 3), default=3)
    arguments = parser.parse_args()
    if arguments.rounds < 1 or arguments.timeout <= 0:
        parser.error("rounds and timeout must be positive")
    torch.manual_seed(20261105)
    mapping = json.loads((arguments.model_path / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(layer, suffix):
        name = f"model.language_model.layers.{layer}." + suffix
        with safe_open(arguments.model_path / mapping[name], framework="pt", device="cpu") as checkpoint:
            return checkpoint.get_tensor(name).cuda()

    weights = []
    for layer in range(arguments.weight_sets):
        qkvz = torch.cat((weight(layer, "linear_attn.in_proj_qkv.weight"), weight(layer, "linear_attn.in_proj_z.weight")))
        k1_weights = (weight(layer, "input_layernorm.weight"), qkvz, torch.cat((weight(layer, "linear_attn.in_proj_b.weight"), weight(layer, "linear_attn.in_proj_a.weight"))), weight(layer, "linear_attn.conv1d.weight"), weight(layer, "linear_attn.norm.weight"), weight(layer, "linear_attn.A_log").float(), weight(layer, "linear_attn.dt_bias"))
        k2_weights = (weight(layer, "linear_attn.out_proj.weight"), weight(layer, "post_attention_layernorm.weight"), torch.cat((weight(layer, "mlp.gate_proj.weight"), weight(layer, "mlp.up_proj.weight"))), weight(layer, "mlp.down_proj.weight"))
        weights.append((k1_weights, k2_weights))
    entries = []
    for layer in range(64):
        k1_weights, k2_weights = weights[layer % arguments.weight_sets]
        entries.append((DenseK1(*k1_weights) if layer % 4 != 3 else None, DenseK2(*k2_weights), k1_weights[1]))
    report = {"scope": "64-layer chain with distinct workspaces and GDN states per layer, alternating real weight sets and native linear operations; batch-1/2 graphs coexist and alternate with eager references; not full-model numerical acceptance", "weight_sets": arguments.weight_sets, "implementation_sha256": implementation_digest(), "checks": [], "program": entries[0][0].program.manifest, "passed": False}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")

    def wait_for_device():
        event = torch.cuda.Event()
        event.record()
        deadline = time.monotonic() + arguments.timeout
        while not event.query():
            if time.monotonic() > deadline:
                report["error"] = "GPU completion deadline exceeded"
                save()
                print(report["error"], flush=True)
                os._exit(124)
            time.sleep(0.01)

    def forward(source, convolution, state, indices):
        hidden, residual = source, None
        state_index = 0
        for kernel1, kernel2, qkvz in entries:
            if kernel1 is not None:
                core, residual = kernel1(hidden, residual, convolution[state_index], state[state_index], indices)
                state_index += 1
            else:
                residual = hidden if residual is None else (hidden.float() + residual.float()).bfloat16()
                core = torch.nn.functional.linear(hidden, qkvz[:6144])
            hidden, residual = kernel2(core, residual)
        return hidden

    def counts():
        return ([int(kernel1.executions.item()) if kernel1 is not None else 0 for kernel1, _, _ in entries], [int(kernel2.executions.item()) for _, kernel2, _ in entries])

    graphs = {}
    stream = torch.cuda.Stream()
    save()
    for rows in (2, 1):
        source = torch.randn((rows, 5120), device="cuda", dtype=torch.bfloat16)
        convolution = torch.zeros((48, rows, 10240, 3), device="cuda", dtype=torch.bfloat16)
        state = torch.zeros((48, rows, 48, 128, 128), device="cuda")
        indices = torch.arange(rows, device="cuda", dtype=torch.int32)

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward(source, convolution, state, indices)
        torch.cuda.current_stream().wait_stream(stream)
        wait_for_device()
        initial = counts()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            output = forward(source, convolution, state, indices)
        assert counts() == initial
        graph.instantiate()
        graphs[rows] = (graph, source, convolution, state, indices, output)
    for round_index in range(arguments.rounds):
        for rows in (2, 1, 1, 2):
            graph, source, convolution, state, indices, output = graphs[rows]
            source.normal_()
            original_convolution, original_state = convolution.clone(), state.clone()
            expected = forward(source, convolution, state, indices).clone()
            expected_convolution, expected_state = convolution.clone(), state.clone()
            convolution.copy_(original_convolution)
            state.copy_(original_state)
            wait_for_device()
            initial1, initial2 = counts()
            graph.replay()
            wait_for_device()
            final1, final2 = counts()
            delta1 = [current - previous for current, previous in zip(final1, initial1)]
            delta2 = [current - previous for current, previous in zip(final2, initial2)]
            check = {"round": round_index, "rows": rows, "k1_executions": sum(delta1), "k2_executions": sum(delta2), "finite": bool(torch.isfinite(output).all()), "output_exact": torch.equal(output, expected), "convolution_exact": torch.equal(convolution, expected_convolution), "state_exact": torch.equal(state, expected_state), "per_layer_counts_exact": delta1 == [int(layer % 4 != 3) for layer in range(64)] and delta2 == [1] * 64}
            report["checks"].append(check)
            save()
            print(json.dumps(check), flush=True)
            assert check["k1_executions"] == 48 and check["k2_executions"] == 64
            assert all(check[name] for name in ("finite", "output_exact", "convolution_exact", "state_exact", "per_layer_counts_exact"))
    report["passed"] = True
    save()


if __name__ == "__main__":
    main()
