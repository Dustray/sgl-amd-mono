from dataclasses import dataclass

import torch

from .errors import ContractError
from .gfx936.config import HIDDEN, INTERMEDIATE, LAYERS, TP_SIZE


@dataclass(frozen=True)
class FFNWeights:
    gate_up: torch.Tensor
    down: torch.Tensor


def decoder_layers(runner):
    layers = getattr(getattr(runner.model, "model", None), "layers", None)
    if runner.ps.tp_size != TP_SIZE or layers is None or len(layers) != LAYERS:
        raise ContractError("Qwen3.8-27B runner requires TP1 and all 64 decoder layers")
    return layers


def ffn_weights(layer, layer_index):
    mlp = layer.mlp
    if type(mlp).__name__ != "Qwen2MoeMLP":
        raise ContractError(f"Layer {layer_index} is not a dense Qwen MLP")
    for projection in (mlp.gate_up_proj, mlp.down_proj):
        if type(projection.quant_method).__name__ != "UnquantizedLinearMethod" or projection.bias is not None:
            raise ContractError(f"Layer {layer_index} has an unsupported quantized or biased projection")
    weights = FFNWeights(mlp.gate_up_proj.weight, mlp.down_proj.weight)
    for name, tensor, shape in (
        ("gate_up", weights.gate_up, (2 * INTERMEDIATE, HIDDEN)),
        ("down", weights.down, (HIDDEN, INTERMEDIATE)),
    ):
        if tensor.shape != shape or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous() or tensor.device.type != "cuda":
            raise ContractError(f"Layer {layer_index} {name} must be contiguous BF16 {shape} on GPU")
    if weights.gate_up.device != weights.down.device:
        raise ContractError(f"Layer {layer_index} FFN weights are on different devices")
    return weights
