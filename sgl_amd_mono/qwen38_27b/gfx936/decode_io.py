import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from ..errors import ContractError
from .program import DenseK1, HIPMonoProgram, K1Arguments, require_tensor


class EntryArguments(ctypes.Structure):
    _fields_ = [("k1", K1Arguments), ("embedding_weight", ctypes.c_void_p), ("input_ids", ctypes.c_void_p), ("vocabulary", ctypes.c_int64), ("index_bytes", ctypes.c_int64)]


class TailArguments(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in ("hidden", "residual", "norm_weight", "head_weight", "normalized", "residual_output", "logits", "executions", "synchronization")] + [(name, ctypes.c_int64) for name in ("rows", "vocabulary", "barrier_timeout_ticks")]


class DecodeIOProgram:
    _instances = {}

    @classmethod
    def get(cls, device):
        device = torch.device(device)
        if device not in cls._instances:
            cls._instances[device] = cls(device)
        return cls._instances[device]

    def __init__(self, device):
        self.base = HIPMonoProgram.get(device)
        source = Path(__file__).with_name("decode_io.cpp")
        version = subprocess.check_output(["/opt/dtk/bin/hipcc", "--version"], text=True, timeout=10)
        digest = hashlib.sha256(source.read_bytes() + source.with_name("kernel.cpp").read_bytes() + version.encode() + torch.__version__.encode()).hexdigest()
        directory = Path(os.environ.get("SGLANG_AMD_MONO_0518_REPORT_DIR", "monokernel_reports")).resolve() / "decode_io_codegen" / digest[:20]
        directory.mkdir(parents=True, exist_ok=True)
        binary = directory / "decode_io.so"
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            command = ["/opt/dtk/bin/hipcc", "-O3", "-std=c++17", "-fPIC", "-shared", "--offload-arch=gfx936", "-ffp-contract=off", str(source), "-o", str(binary)]
            with (directory / "compile.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            if result.returncode:
                raise ContractError(f"Decode input/output HIP compilation failed: {directory / 'compile.log'}")
            manifest_path.write_text(json.dumps({"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "base_source_sha256": self.base.manifest["source_sha256"], "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "command": command, "compiler": version}, indent=2) + "\n")
        self.manifest = json.loads(manifest_path.read_text())
        self.manifest_path = str(manifest_path)
        if hashlib.sha256(binary.read_bytes()).hexdigest() != self.manifest["binary_sha256"]:
            raise ContractError("Decode input/output binary checksum mismatch")
        self.library = ctypes.CDLL(str(binary))
        self.library.mono_decode_io_info.argtypes = [ctypes.POINTER(ctypes.c_int64)]
        self.library.mono_entry_k1.argtypes = [ctypes.POINTER(EntryArguments), ctypes.c_void_p]
        self.library.mono_tail.argtypes = [ctypes.POINTER(TailArguments), ctypes.c_void_p]
        information = (ctypes.c_int64 * 4)()
        with torch.cuda.device(device):
            self.base.check(self.library.mono_decode_io_info(information))
        if list(information[:2]) != [ctypes.sizeof(EntryArguments), ctypes.sizeof(TailArguments)] or min(information[2:]) < 1:
            raise ContractError(f"Decode input/output ABI or occupancy mismatch: {list(information)}")
        self.information = dict(zip(("entry_abi_bytes", "tail_abi_bytes", "entry_blocks_per_cu", "tail_blocks_per_cu"), information))


class DenseEntryK1(DenseK1):
    def __init__(self, *weights):
        super().__init__(*weights)
        self.io_program = DecodeIOProgram.get(self.device)

    def __call__(self, hidden, residual, conv_state, ssm_state, cache_indices, embedding_weight=None, input_ids=None):
        if embedding_weight is None:
            return super().__call__(hidden, residual, conv_state, ssm_state, cache_indices)
        if residual is not None or embedding_weight.ndim != 2 or embedding_weight.shape[1] != 5120:
            raise ContractError("Decode embedding fusion requires the first layer and BF16 [vocabulary,5120] weights")
        require_tensor(embedding_weight, tuple(embedding_weight.shape), device=self.device)
        if input_ids is None or input_ids.dtype not in (torch.int32, torch.int64):
            raise ContractError("Decode embedding fusion requires integer input token ids")
        require_tensor(input_ids, (hidden.shape[0],), input_ids.dtype, device=self.device)
        arguments, output, residual_output = self.arguments(hidden, residual, conv_state, ssm_state, cache_indices)
        entry = EntryArguments(arguments, embedding_weight.data_ptr(), input_ids.data_ptr(), embedding_weight.shape[0], input_ids.element_size())
        with torch.cuda.device(self.device):
            self.program.check(self.io_program.library.mono_entry_k1(ctypes.byref(entry), torch.cuda.current_stream(self.device).cuda_stream))
        return output, residual_output


class DenseTail:
    def __init__(self, norm_weight, head_weight):
        self.device = head_weight.device
        if head_weight.ndim != 2 or head_weight.shape[1] != 5120 or head_weight.shape[0] % 16:
            raise ContractError("Decode tail requires a vocabulary divisible by 16 and hidden size 5120")
        require_tensor(norm_weight, (5120,), device=self.device)
        require_tensor(head_weight, tuple(head_weight.shape), device=self.device)
        self.weights = norm_weight, head_weight
        self.program = DecodeIOProgram.get(self.device)
        self.normalized = torch.empty((2, 5120), device=self.device, dtype=torch.bfloat16)
        self.residual_output = torch.empty_like(self.normalized)
        self.executions = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.synchronization = torch.zeros(3, device=self.device, dtype=torch.int32)

    def __call__(self, hidden, residual):
        rows = hidden.shape[0]
        if rows not in (1, 2):
            raise ContractError("Decode tail requires rows 1/2")
        require_tensor(hidden, (rows, 5120), device=self.device)
        if residual is not None:
            require_tensor(residual, (rows, 5120), device=self.device)
        vocabulary = self.weights[1].shape[0]
        output = torch.empty((rows, vocabulary), device=self.device, dtype=torch.bfloat16)
        tensors = hidden, residual, *self.weights, self.normalized, self.residual_output, output, self.executions, self.synchronization
        arguments = TailArguments(*(tensor.data_ptr() if tensor is not None else 0 for tensor in tensors), rows, vocabulary, self.program.base.barrier_timeout_ticks)
        with torch.cuda.device(self.device):
            self.program.base.check(self.program.library.mono_tail(ctypes.byref(arguments), torch.cuda.current_stream(self.device).cuda_stream))
        return output
