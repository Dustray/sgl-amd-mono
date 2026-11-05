import functools
import logging

import torch

from .hip_ffn import FusedBF16FFN, validate_result
from .contract import ContractError


logger = logging.getLogger(__name__)


class FFNRuntime:
    def __init__(self, runner):
        if runner.ps.tp_size != 1:
            raise ContractError("Fused FFN runtime requires TP1")
        layers = getattr(getattr(runner.model, "model", None), "layers", None)
        if layers is None or len(layers) != 64:
            raise ContractError("Fused FFN could not locate all 64 Qwen decoder layers")
        self.enabled = False
        self.entries = []
        self.validation = []
        for layer_index, layer in enumerate(layers):
            mlp = layer.mlp
            if type(mlp).__name__ != "Qwen2MoeMLP":
                raise ContractError(f"Layer {layer_index} is not a dense Qwen MLP")
            for projection in (mlp.gate_up_proj, mlp.down_proj):
                if type(projection.quant_method).__name__ != "UnquantizedLinearMethod" or projection.bias is not None:
                    raise ContractError(f"Layer {layer_index} has an unsupported quantized or biased projection")
            original = mlp.forward
            fused = FusedBF16FFN(mlp.gate_up_proj.weight, mlp.down_proj.weight)
            generator = torch.Generator(device=fused.down.device)
            generator.manual_seed(20261105 + layer_index)
            for rows in (1, 2):
                hidden = torch.randn(rows, fused.hidden_size, device=fused.down.device, dtype=torch.bfloat16, generator=generator)
                expected = original(hidden)
                actual = fused(hidden)
                error = validate_result(actual, expected)
                self.validation.append({"layer": layer_index, "rows": rows, **error})
            self.entries.append((mlp, original, fused))
        for mlp, original, fused in self.entries:
            mlp.forward = self._wrap(original, fused)
            fused.executions.zero_()
        torch.cuda.synchronize()
        logger.warning("AMD_FUSED_FFN attached %d dense FFNs after real-weight numerical validation; two GPU launches per FFN; attention remains SGLang", len(self.entries))

    def _wrap(self, original, fused):
        @functools.wraps(original)
        def forward(hidden):
            if self.enabled:
                return fused(hidden)
            return original(hidden)

        return forward

    def supports(self, batch):
        return batch.forward_mode.is_decode() and 1 <= int(batch.batch_size) <= 2

    def snapshot(self):
        torch.cuda.synchronize()
        counts = [int(fused.executions.item()) for _, _, fused in self.entries]
        return {
            "scope": "FFN only; attention, residual and normalization use SGLang",
            "execution_path": "fused_bf16_ffn_and_sglang_attention",
            "whole_model_monokernel": False,
            "monokernel_executions": 0,
            "fused_ffn_device_executions": sum(counts),
            "fused_ffn_device_executions_per_layer": counts,
            "fused_ffn_kernel_launches": 2 * sum(counts),
            "counter_scope": "post-attachment warmup, graph capture, and serving; compare snapshots for request deltas",
            "attached_layers": len(self.entries),
            "workspace_bytes": sum(fused.partial.numel() * fused.partial.element_size() for _, _, fused in self.entries),
            "extra_weight_copies": 0,
            "compiler_isolated": True,
            "codegen_manifest": self.entries[0][2].program.codegen_manifest,
            "binary_sha256": self.entries[0][2].program.binary_hashes,
            "activation_contract": "BF16 SiLU rounding followed by BF16 multiplication, matching HCU LightOp",
            "validation_nrmse_limit": 0.002,
        }
