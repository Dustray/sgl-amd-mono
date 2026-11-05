import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from ..errors import ContractError


class K1Arguments(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "hidden", "residual", "norm_weight", "qkvz_weight", "ba_weight", "conv_weight",
        "gated_weight", "a_log", "dt_bias", "conv_state", "ssm_state", "cache_indices",
        "normalized", "projected", "projected_ba", "convolved", "output", "residual_output", "executions",
    )] + [(name, ctypes.c_int64) for name in (
        "rows", "conv_slot_stride", "conv_channel_stride", "conv_history_stride",
        "ssm_slot_stride", "index_bytes", "slot_count",
    )]


class K2Arguments(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "core", "residual", "output_weight", "norm_weight", "gate_up_weight", "down_weight",
        "attention_output", "normalized", "activated", "output", "residual_output", "executions",
    )] + [("rows", ctypes.c_int64)]


def require_tensor(tensor, shape, dtype=torch.bfloat16, contiguous=True):
    if tensor.shape != shape or tensor.dtype != dtype or tensor.device.type != "cuda" or (contiguous and not tensor.is_contiguous()):
        raise ContractError(f"Monokernel tensor contract: expected {shape} {dtype}, got {tensor.shape} {tensor.dtype} {tensor.device} strides={tensor.stride()}")


class HIPMonoProgram:
    _instances = {}

    @classmethod
    def get(cls, device):
        device = torch.device(device)
        if device not in cls._instances:
            cls._instances[device] = cls(device)
        return cls._instances[device]

    def __init__(self, device):
        properties = torch.cuda.get_device_properties(device)
        if properties.gcnArchName.split(":")[0] != "gfx936" or properties.multi_processor_count != 80:
            raise ContractError("Dense BF16 Monokernel requires the validated 80-CU gfx936 device")
        source = Path(__file__).with_name("kernel.cpp")
        compiler = "/opt/dtk/bin/hipcc"
        compiler_version = subprocess.check_output([compiler, "--version"], text=True)
        digest = hashlib.sha256(source.read_bytes() + compiler_version.encode() + torch.__version__.encode()).hexdigest()
        cache = Path(os.environ.get("SGLANG_AMD_MONO_0518_REPORT_DIR", "monokernel_reports")).resolve() / "dense_codegen" / digest[:20]
        cache.mkdir(parents=True, exist_ok=True)
        binary = cache / "dense_monokernel.so"
        manifest_path = cache / "manifest.json"
        if not manifest_path.exists():
            command = [compiler, "-O3", "-std=c++17", "-fPIC", "-shared", "--offload-arch=gfx936", "-ffp-contract=off", str(source), "-o", str(binary)]
            with (cache / "compile.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            if result.returncode:
                raise ContractError(f"DTK HIP compilation failed: {cache / 'compile.log'}")
            manifest_path.write_text(json.dumps({"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "compiler": compiler_version, "command": command, "architecture": "gfx936", "blocks": 80, "threads": 256, "cooperative": True, "launch_api": "hipLaunchKernelExC"}, indent=2) + "\n")
        self.manifest_path = str(manifest_path)
        self.manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(binary.read_bytes()).hexdigest() != self.manifest["binary_sha256"]:
            raise ContractError("Dense Monokernel binary checksum mismatch")
        self.library = ctypes.CDLL(str(binary))
        self.library.mono_info.argtypes = [ctypes.POINTER(ctypes.c_int64)]
        self.library.mono_k1.argtypes = [ctypes.POINTER(K1Arguments), ctypes.c_void_p]
        self.library.mono_k2.argtypes = [ctypes.POINTER(K2Arguments), ctypes.c_void_p]
        information = (ctypes.c_int64 * 6)()
        with torch.cuda.device(device):
            self.check(self.library.mono_info(information))
        self.information = dict(zip(("k1_abi_bytes", "k2_abi_bytes", "cooperative", "multiprocessors", "k1_blocks_per_cu", "k2_blocks_per_cu"), information))
        if list(information[:2]) != [ctypes.sizeof(K1Arguments), ctypes.sizeof(K2Arguments)] or not information[2] or information[3] != 80 or min(information[4:]) < 1:
            raise ContractError(f"Cooperative occupancy / C ABI guard failed: {self.information}")

    @staticmethod
    def check(status):
        if status:
            raise ContractError(f"Cooperative Monokernel HIP launch failed, status={status}; stateful execution cannot fall back")


class DenseK1:
    def __init__(self, norm_weight, qkvz, ba, convolution, gated_weight, a_log, dt_bias):
        for tensor, shape in ((norm_weight, (5120,)), (qkvz, (16384, 5120)), (ba, (96, 5120)), (convolution, (10240, 1, 4)), (gated_weight, (128,)), (dt_bias, (48,))):
            require_tensor(tensor, shape)
        require_tensor(a_log, (48,), torch.float32)
        self.weights = (norm_weight, qkvz, ba, convolution, gated_weight, a_log, dt_bias)
        self.device = qkvz.device
        self.program = HIPMonoProgram.get(self.device)
        self.normalized = torch.empty((2, 5120), dtype=torch.bfloat16, device=self.device)
        self.projected = torch.empty((2, 16384), dtype=torch.bfloat16, device=self.device)
        self.projected_ba = torch.empty((2, 96), dtype=torch.bfloat16, device=self.device)
        self.convolved = torch.empty((2, 10240), dtype=torch.bfloat16, device=self.device)
        self.executions = torch.zeros(1, dtype=torch.int32, device=self.device)

    def __call__(self, hidden, residual, conv_state, ssm_state, cache_indices):
        rows = hidden.shape[0]
        require_tensor(hidden, (rows, 5120))
        if rows not in (1, 2):
            raise ContractError("Dense K1 supports decode rows 1/2")
        if residual is not None:
            require_tensor(residual, (rows, 5120))
        if conv_state.ndim != 3 or conv_state.shape[1:] != (10240, 3):
            raise ContractError(f"Unsupported convolution cache shape: {conv_state.shape}")
        require_tensor(conv_state, tuple(conv_state.shape), contiguous=False)
        require_tensor(ssm_state, (conv_state.shape[0], 48, 128, 128), torch.float32)
        if cache_indices.dtype not in (torch.int32, torch.int64):
            raise ContractError("Cache indices must be int32 or int64")
        require_tensor(cache_indices, (rows,), cache_indices.dtype)
        if any(tensor.device != self.device for tensor in (hidden, conv_state, ssm_state, cache_indices)):
            raise ContractError("Dense K1 tensors must be on the weight device")
        output = torch.empty((rows, 6144), device=self.device, dtype=torch.bfloat16)
        residual_output = torch.empty_like(hidden)
        norm_weight, qkvz, ba, convolution, gated_weight, a_log, dt_bias = self.weights
        tensors = (hidden, residual, norm_weight, qkvz, ba, convolution, gated_weight, a_log, dt_bias, conv_state, ssm_state, cache_indices, self.normalized, self.projected, self.projected_ba, self.convolved, output, residual_output, self.executions)
        arguments = K1Arguments(*(tensor.data_ptr() if tensor is not None else 0 for tensor in tensors), rows, *conv_state.stride(), ssm_state.stride(0), cache_indices.element_size(), conv_state.shape[0])
        self.program.check(self.program.library.mono_k1(ctypes.byref(arguments), torch.cuda.current_stream(self.device).cuda_stream))
        return output, residual_output


class DenseK2:
    def __init__(self, output_weight, norm_weight, gate_up, down):
        for tensor, shape in ((output_weight, (5120, 6144)), (norm_weight, (5120,)), (gate_up, (34816, 5120)), (down, (5120, 17408))):
            require_tensor(tensor, shape)
        self.weights = (output_weight, norm_weight, gate_up, down)
        self.device = down.device
        self.program = HIPMonoProgram.get(self.device)
        self.attention_output = torch.empty((2, 5120), dtype=torch.bfloat16, device=self.device)
        self.normalized = torch.empty_like(self.attention_output)
        self.activated = torch.empty((2, 17408), dtype=torch.bfloat16, device=self.device)
        self.executions = torch.zeros(1, dtype=torch.int32, device=self.device)

    def __call__(self, core, residual):
        rows = core.shape[0]
        require_tensor(core, (rows, 6144))
        require_tensor(residual, (rows, 5120))
        if rows not in (1, 2) or core.device != self.device or residual.device != self.device:
            raise ContractError("Dense K2 supports rows 1/2 on the weight device")
        output = torch.empty_like(residual)
        residual_output = torch.empty_like(residual)
        output_weight, norm_weight, gate_up, down = self.weights
        tensors = (core, residual, output_weight, norm_weight, gate_up, down, self.attention_output, self.normalized, self.activated, output, residual_output, self.executions)
        arguments = K2Arguments(*(tensor.data_ptr() for tensor in tensors), rows)
        self.program.check(self.program.library.mono_k2(ctypes.byref(arguments), torch.cuda.current_stream(self.device).cuda_stream))
        return output, residual_output
