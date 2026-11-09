import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from ..errors import ContractError
from .program import HIPMonoProgram, require_tensor


POINTER_FIELDS = (
    "hidden", "residual", "norm_weight", "qkv_weight", "q_norm_weight", "k_norm_weight",
    "rotary_cache", "rotary_axes", "positions", "key_cache", "value_cache", "cache_locations",
    "page_table", "cache_lengths", "request_indices", "normalized", "projected", "query", "key", "value", "gate",
    "scores", "output", "residual_output", "executions", "synchronization",
)
INTEGER_FIELDS = (
    "rows", "rotary_dim", "rotary_cache_rows", "position_axes", "position_axis_stride",
    "position_row_stride", "key_page_stride", "key_token_stride", "key_head_stride", "key_dim_stride",
    "value_page_stride", "value_token_stride", "value_head_stride", "value_dim_stride",
    "page_table_stride", "page_size", "page_count", "cache_dtype", "location_bytes", "position_bytes", "request_index_bytes",
    "context_limit", "barrier_timeout_ticks",
)


class AttentionArguments(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in POINTER_FIELDS] + [(name, ctypes.c_int64) for name in INTEGER_FIELDS]


def rotary_axes(rotary_dim, sections, interleaved):
    if rotary_dim <= 0 or rotary_dim > 256 or rotary_dim % 2 or len(sections) != 3 or sum(sections) != rotary_dim // 2:
        raise ContractError("Full attention requires an even rotary dimension and three matching MRoPE sections")
    if interleaved:
        return [1 if index % 3 == 1 and index <= 3 * sections[1] else 2 if index % 3 == 2 and index <= 3 * sections[2] else 0 for index in range(rotary_dim // 2)]
    return [axis for axis, size in enumerate(sections) for _ in range(size)]


def cache_views(key, value, page_size, layout):
    if layout == "legacy":
        if key.ndim != 4 or key.shape[1:] != (4, page_size, 256) or value.shape != (key.shape[0], 4, 256, page_size):
            raise ContractError("Full attention legacy KV layout drift")
        return key.permute(0, 2, 1, 3), value.permute(0, 3, 1, 2)
    if layout == "hnd":
        if key.ndim != 4 or key.shape[1:] != (4, page_size, 256) or value.shape != key.shape:
            raise ContractError("Full attention HND KV layout drift")
        return key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)
    if layout == "nhd":
        if key.ndim != 3 or key.shape[1:] != (4, 256) or value.shape != key.shape or key.shape[0] % page_size:
            raise ContractError("Full attention NHD KV layout drift")
        return key.view(-1, page_size, 4, 256), value.view(-1, page_size, 4, 256)
    raise ContractError(f"Unsupported full attention cache layout: {layout}")


class AttentionProgram:
    _instances = {}

    @classmethod
    def get(cls, device):
        device = torch.device(device)
        if device not in cls._instances:
            cls._instances[device] = cls(device)
        return cls._instances[device]

    def __init__(self, device):
        self.base = HIPMonoProgram.get(device)
        source = Path(__file__).with_name("attention.cpp")
        compiler = "/opt/dtk/bin/hipcc"
        version = subprocess.check_output([compiler, "--version"], text=True, timeout=10)
        digest = hashlib.sha256(source.read_bytes() + source.with_name("kernel.cpp").read_bytes() + version.encode() + torch.__version__.encode()).hexdigest()
        directory = Path(os.environ.get("SGLANG_AMD_MONO_0518_REPORT_DIR", "monokernel_reports")).resolve() / "attention_codegen" / digest[:20]
        directory.mkdir(parents=True, exist_ok=True)
        binary = directory / "attention_monokernel.so"
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            command = [compiler, "-O3", "-std=c++17", "-fPIC", "-shared", "--offload-arch=gfx936", "-ffp-contract=off", str(source), "-o", str(binary)]
            with (directory / "compile.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            if result.returncode:
                raise ContractError(f"Full attention HIP compilation failed: {directory / 'compile.log'}")
            manifest_path.write_text(json.dumps({"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "base_source_sha256": self.base.manifest["source_sha256"], "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "command": command, "compiler": version, "blocks": 80, "threads": 256, "barriers": 5}, indent=2) + "\n")
        self.manifest = json.loads(manifest_path.read_text())
        self.manifest_path = str(manifest_path)
        if hashlib.sha256(binary.read_bytes()).hexdigest() != self.manifest["binary_sha256"]:
            raise ContractError("Full attention binary checksum mismatch")
        self.library = ctypes.CDLL(str(binary))
        self.library.mono_attention_info.argtypes = [ctypes.POINTER(ctypes.c_int64)]
        self.library.mono_attention_k1.argtypes = [ctypes.POINTER(AttentionArguments), ctypes.c_void_p]
        information = (ctypes.c_int64 * 2)()
        with torch.cuda.device(device):
            self.base.check(self.library.mono_attention_info(information))
        if information[0] != ctypes.sizeof(AttentionArguments) or information[1] < 1:
            raise ContractError(f"Full attention ABI or resident-grid admission failed: {list(information)}")
        self.information = {"abi_bytes": information[0], "blocks_per_cu": information[1]}


class DenseAttentionK1:
    def __init__(self, norm_weight, qkv_weight, q_norm_weight, k_norm_weight, rotary_dim=64, sections=(11, 11, 10), interleaved=True):
        self.device = qkv_weight.device
        for tensor, shape in ((norm_weight, (5120,)), (qkv_weight, (14336, 5120)), (q_norm_weight, (256,)), (k_norm_weight, (256,))):
            require_tensor(tensor, shape, device=self.device)
        axes = rotary_axes(rotary_dim, sections, interleaved)
        self.weights = norm_weight, qkv_weight, q_norm_weight, k_norm_weight
        self.rotary_dim = rotary_dim
        self.sections = tuple(sections)
        self.interleaved = interleaved
        self.axes = torch.tensor(axes, device=self.device, dtype=torch.int32)
        self.program = AttentionProgram.get(self.device)
        self.normalized = torch.empty((2, 5120), device=self.device, dtype=torch.bfloat16)
        self.projected = torch.empty((2, 14336), device=self.device, dtype=torch.bfloat16)
        self.query = torch.empty((2, 6144), device=self.device, dtype=torch.bfloat16)
        self.key = torch.empty((2, 1024), device=self.device, dtype=torch.bfloat16)
        self.value = torch.empty_like(self.key)
        self.gate = torch.empty_like(self.query)
        self.scores = torch.empty((2, 24, 4096), device=self.device, dtype=torch.float32)
        self.executions = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.synchronization = torch.zeros(3, device=self.device, dtype=torch.int32)

    def __call__(self, hidden, residual, positions, rotary_cache, key_cache, value_cache, locations, page_table, lengths, context_limit=4096, request_indices=None):
        rows = hidden.shape[0]
        require_tensor(hidden, (rows, 5120), device=self.device)
        if rows not in (1, 2) or not 1 <= context_limit <= 4096:
            raise ContractError("Full attention supports rows 1/2 and context limits up to 4096")
        if residual is not None:
            require_tensor(residual, tuple(hidden.shape), device=self.device)
        if positions.dtype not in (torch.int32, torch.int64) or locations.dtype not in (torch.int32, torch.int64):
            raise ContractError("Full attention position/cache indices must be int32 or int64")
        if tuple(positions.shape) not in ((rows,), (3, rows)) or positions.stride(-1) != 1:
            raise ContractError("Full attention requires scalar positions or three MRoPE position axes")
        require_tensor(positions, tuple(positions.shape), positions.dtype, contiguous=False, device=self.device)
        require_tensor(locations, (rows,), locations.dtype, device=self.device)
        require_tensor(lengths, (rows,), torch.int32, device=self.device)
        if request_indices is not None:
            if request_indices.dtype not in (torch.int32, torch.int64):
                raise ContractError("Full attention request pool indices must be int32 or int64")
            require_tensor(request_indices, (rows,), request_indices.dtype, device=self.device)
        if page_table.ndim != 2 or page_table.shape[0] != rows or page_table.stride(1) != 1:
            raise ContractError("Full attention requires a row-major int32 page table")
        require_tensor(page_table, tuple(page_table.shape), torch.int32, contiguous=False, device=self.device)
        if key_cache.ndim != 4 or key_cache.shape[2:] != (4, 256) or key_cache.shape != value_cache.shape:
            raise ContractError("Full attention requires logical [page, token, head, dimension] KV views")
        if key_cache.dtype not in (torch.bfloat16, torch.float8_e5m2) or value_cache.dtype != key_cache.dtype:
            raise ContractError("Full attention supports BF16 or unscaled FP8 E5M2 KV")
        for tensor in (key_cache, value_cache):
            require_tensor(tensor, tuple(key_cache.shape), key_cache.dtype, contiguous=False, device=self.device)
            if min(tensor.stride()) <= 0 or torch._debug_has_internal_overlap(tensor) != 0:
                raise ContractError("Full attention refuses overlapping KV cache storage")
        page_size = key_cache.shape[1]
        if page_size != 64 or page_table.shape[1] < 1:
            raise ContractError("Full attention requires page size 64 and a nonempty context page table")
        context_limit = min(context_limit, page_table.shape[1] * page_size)
        if rotary_cache.ndim != 2 or rotary_cache.shape[1] != self.rotary_dim:
            raise ContractError("Full attention rotary cache geometry drift")
        require_tensor(rotary_cache, tuple(rotary_cache.shape), device=self.device)
        output = torch.empty((rows, 6144), device=self.device, dtype=torch.bfloat16)
        residual_output = torch.empty_like(hidden)
        tensors = (hidden, residual, *self.weights, rotary_cache, self.axes, positions, key_cache, value_cache, locations, page_table, lengths, request_indices, self.normalized, self.projected, self.query, self.key, self.value, self.gate, self.scores, output, residual_output, self.executions, self.synchronization)
        arguments = AttentionArguments(*(tensor.data_ptr() if tensor is not None else 0 for tensor in tensors), rows, self.rotary_dim, rotary_cache.shape[0], 1 if positions.ndim == 1 else 3, 0 if positions.ndim == 1 else positions.stride(0), positions.stride(-1), *key_cache.stride(), *value_cache.stride(), page_table.stride(0), page_size, key_cache.shape[0], int(key_cache.dtype == torch.float8_e5m2), locations.element_size(), positions.element_size(), request_indices.element_size() if request_indices is not None else 0, context_limit, self.program.base.barrier_timeout_ticks)
        with torch.cuda.device(self.device):
            self.program.base.check(self.program.library.mono_attention_k1(ctypes.byref(arguments), torch.cuda.current_stream(self.device).cuda_stream))
        return output, residual_output
