import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch


class HIPFFNProgram:
    _instances = {}

    @classmethod
    def get(cls, hidden_size, intermediate_size, device):
        key = (hidden_size, intermediate_size, str(device))
        if key not in cls._instances:
            cls._instances[key] = cls(hidden_size, intermediate_size, device)
        return cls._instances[key]

    def __init__(self, hidden_size, intermediate_size, device):
        source_dir = Path(__file__).parent
        digest = hashlib.sha256()
        for filename in ("ffn_kernel.py", "compile_ffn.py", "ffn.py"):
            digest.update((source_dir / filename).read_bytes())
        digest.update(importlib.metadata.version("flydsl").encode())
        digest.update(torch.__version__.encode())
        cache_root = Path(os.environ.get("SGLANG_AMD_MONO_0518_REPORT_DIR", "monokernel_reports"))
        cache = cache_root.resolve() / "ffn_codegen" / f"{hidden_size}_{intermediate_size}_{digest.hexdigest()[:20]}"
        cache.mkdir(parents=True, exist_ok=True)
        manifest_path = cache / "manifest.json"
        if not manifest_path.exists():
            environment = os.environ.copy()
            environment["SGLANG_AMD_MONO_0518"] = "0"
            with (cache / "compile.log").open("w") as log:
                result = subprocess.run(
                    [sys.executable, "-m", "sgl_amd_mono.qwen38_27b.gfx936.compile_ffn", "--output-dir", str(cache),
                     "--hidden-size", str(hidden_size), "--intermediate-size", str(intermediate_size)],
                    env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=180,
                )
            if result.returncode:
                raise RuntimeError(f"Isolated FlyDSL compilation failed: {cache / 'compile.log'}")
        manifest = json.loads(manifest_path.read_text())
        self.codegen_manifest = str(manifest_path)
        self.binary_hashes = {rows: specialization["sha256"] for rows, specialization in manifest["specializations"].items()}
        if manifest["architecture"] != "gfx936":
            raise RuntimeError("FFN binary architecture mismatch")
        self.library = ctypes.CDLL("/opt/dtk/lib/libgalaxyhip.so")
        self.library.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.library.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        self.library.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p, *([ctypes.c_uint] * 7), ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ]
        self.modules = []
        self.functions = {}
        with torch.cuda.device(device):
            for rows, specialization in manifest["specializations"].items():
                binary = (cache / specialization["file"]).read_bytes()
                if hashlib.sha256(binary).hexdigest() != specialization["sha256"]:
                    raise RuntimeError("FFN GPU binary checksum mismatch")
                module = ctypes.c_void_p()
                self._check(self.library.hipModuleLoadData(ctypes.byref(module), ctypes.c_char_p(binary)))
                self.modules.append(module)
                functions = []
                for index, metadata in enumerate(specialization["kernels"]):
                    expected_name = ("ffn_partition_kernel_0", "ffn_reduce_kernel_1")[index]
                    expected_sizes = [8, 4] * (4 if index == 0 else 3)
                    if metadata[".name"] != expected_name or [argument[".size"] for argument in metadata[".args"]] != expected_sizes:
                        raise RuntimeError(f"Unsupported FlyDSL tensor ABI: {metadata}")
                    function = ctypes.c_void_p()
                    self._check(self.library.hipModuleGetFunction(ctypes.byref(function), module, expected_name.encode()))
                    functions.append(function)
                self.functions[int(rows)] = functions

    def _check(self, status):
        if status:
            raise RuntimeError(f"HIP FFN module operation failed with status {status}")

    def _launch(self, function, tensors, grid, stream):
        arguments = []
        for tensor in tensors:
            arguments.extend((ctypes.c_void_p(tensor.data_ptr()), ctypes.c_int32(tensor.numel())))
        parameters = (ctypes.c_void_p * len(arguments))(*(ctypes.addressof(argument) for argument in arguments))
        self._check(self.library.hipModuleLaunchKernel(
            function, grid, 1, 1, 256, 1, 1, 0, stream, parameters, None,
        ))

    def __call__(self, hidden, gate_up, down, partial, output, executions):
        rows, hidden_size = hidden.shape
        stream = torch.cuda.current_stream(hidden.device).cuda_stream
        partition, reduce = self.functions[rows]
        self._launch(partition, (hidden, gate_up, down, partial), down.shape[1] // 256, stream)
        self._launch(reduce, (partial, output, executions), (rows * hidden_size + 255) // 256, stream)


class FusedBF16FFN:
    def __init__(self, gate_up, down, max_rows=2):
        if gate_up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16:
            raise ValueError("Fused FFN requires BF16 weights")
        if gate_up.ndim != 2 or down.ndim != 2:
            raise ValueError("Fused FFN requires matrix weights")
        self.hidden_size, self.intermediate_size = down.shape
        if gate_up.shape != (2 * self.intermediate_size, self.hidden_size):
            raise ValueError("Gate/up and down weight shapes disagree")
        if self.hidden_size % 256 or self.intermediate_size % 256:
            raise ValueError("Fused FFN dimensions must be multiples of 256")
        if not gate_up.is_contiguous() or not down.is_contiguous():
            raise ValueError("Fused FFN requires contiguous weights")
        if gate_up.device != down.device or gate_up.device.type != "cuda":
            raise ValueError("Fused FFN weights must share a GPU")
        architecture = torch.cuda.get_device_properties(gate_up.device).gcnArchName.split(":")[0]
        if architecture != "gfx936" or max_rows not in (1, 2):
            raise ValueError(f"Unvalidated FFN configuration: {architecture}, rows={max_rows}")
        self.gate_up = gate_up
        self.down = down
        self.max_rows = max_rows
        self.program = HIPFFNProgram.get(self.hidden_size, self.intermediate_size, down.device)
        self.partial = torch.empty(max_rows * (self.intermediate_size // 256) * self.hidden_size,
                                   dtype=torch.float32, device=down.device)
        self.executions = torch.zeros(1, dtype=torch.int32, device=down.device)

    def __call__(self, hidden):
        if (
            hidden.dtype != torch.bfloat16 or hidden.device != self.down.device or hidden.ndim != 2
            or hidden.shape[1] != self.hidden_size or not 1 <= hidden.shape[0] <= self.max_rows
            or not hidden.is_contiguous()
        ):
            raise ValueError("Fused FFN input must be contiguous BF16 decode rows on the weight device")
        output = torch.empty_like(hidden)
        self.program(hidden, self.gate_up, self.down, self.partial, output, self.executions)
        return output


def numerical_error(actual, expected):
    difference = actual.float() - expected.float()
    expected_rms = expected.float().square().mean().sqrt().item()
    error_rms = difference.square().mean().sqrt().item()
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "max_absolute_error": difference.abs().max().item(),
        "mean_absolute_error": difference.abs().mean().item(),
        "normalized_rms_error": error_rms / max(expected_rms, 1e-12),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            actual.float().reshape(1, -1), expected.float().reshape(1, -1)
        ).item(),
    }


def validate_result(actual, expected):
    error = numerical_error(actual, expected)
    if not error["finite"] or not math.isfinite(error["normalized_rms_error"]) or error["normalized_rms_error"] > 0.002:
        raise RuntimeError(f"Fused BF16 FFN numerical validation failed: {error}")
    return error


def reference_ffn(hidden, gate_up, down):
    projected = torch.nn.functional.linear(hidden, gate_up)
    gate, value = projected.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * value
    return torch.nn.functional.linear(activated, down)
