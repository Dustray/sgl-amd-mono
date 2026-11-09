import ctypes
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from torch.nn import functional

from .program import FlashContractError, HIPProgram, MixArguments, require_tensor


class GDNArguments(ctypes.Structure):
    _fields_ = [("mix", MixArguments)] + [(name, ctypes.c_void_p) for name in (
        "qkvz_weight", "ba_weight", "conv_weight", "gated_weight", "a_log", "dt_bias", "output_weight", "conv_state", "ssm_state", "indices", "projected", "projected_ba", "convolved", "recurrent", "core", "output", "executions",
    )] + [(name, ctypes.c_int64) for name in ("conv_slot_stride", "conv_channel_stride", "conv_history_stride", "ssm_slot_stride", "index_bytes", "slot_count")]


class GDNProgram:
    _instances = {}

    @classmethod
    def get(cls, device):
        device = torch.device(device)
        if device not in cls._instances:
            cls._instances[device] = cls(device)
        return cls._instances[device]

    def __init__(self, device):
        self.base = HIPProgram.get(device)
        sources = {name: Path(__file__).with_name(name).read_bytes() for name in ("gdn.cpp", "kernel.cpp")}
        compiler = "/opt/dtk/bin/hipcc"
        version = subprocess.check_output([compiler, "--version"], text=True)
        digest = hashlib.sha256(b"".join(sources.values()) + version.encode() + torch.__version__.encode()).hexdigest()
        cache = self.base.cache.parent / ("gdn-" + digest[:24])
        cache.mkdir(parents=True, exist_ok=True)
        binary = cache / "gdn.so"
        manifest_path = cache / "manifest.json"
        with (cache / "compile.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not manifest_path.exists():
                for name, content in sources.items():
                    (cache / name).write_bytes(content)
                command = [compiler, "-O3", "-std=c++17", "-fPIC", "-shared", "--offload-arch=gfx936", "-ffp-contract=off", str(cache / "gdn.cpp"), "-o", str(binary)]
                with (cache / "compile.log").open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240)
                if result.returncode:
                    raise FlashContractError(f"GDN compilation failed: {cache / 'compile.log'}")
                manifest_path.write_text(json.dumps({"source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "command": command}, indent=2) + "\n")
        self.manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(binary.read_bytes()).hexdigest() != self.manifest["binary_sha256"]:
            raise FlashContractError("GDN binary hash mismatch")
        self.library = ctypes.CDLL(str(binary))
        self.library.flash_gdn_info.argtypes = [ctypes.POINTER(ctypes.c_int64)]
        self.library.flash_gdn.argtypes = [ctypes.POINTER(GDNArguments), ctypes.c_void_p]
        information = (ctypes.c_int64 * 2)()
        with torch.cuda.device(device):
            self.base.check(self.library.flash_gdn_info(information))
        if information[0] != ctypes.sizeof(GDNArguments) or information[1] < 1:
            raise FlashContractError(f"GDN ABI or cooperative residency check failed: {list(information)}")
        self.information = {"abi_bytes": information[0], "blocks_per_cu": information[1]}


class GDNAttention:
    def __init__(self, attention, mixer):
        self.mixer = mixer
        self.device = mixer.device
        self.program = GDNProgram.get(self.device)
        self.weights = (attention.in_proj_qkvz.weight, attention.in_proj_ba.weight, attention.conv1d.weight, attention.norm.weight, attention.A_log, attention.dt_bias, attention.out_proj.weight)
        shapes = ((2048, 2560), (12, 2560), (1280, 1, 4), (128,), (6,), (6,), (2560, 768))
        for index, (tensor, shape) in enumerate(zip(self.weights, shapes)):
            require_tensor(tensor, shape, torch.float32 if index == 4 else torch.bfloat16, self.device)
        self.projected = torch.empty((2, 2048), device=self.device, dtype=torch.bfloat16)
        self.projected_ba = torch.empty((2, 12), device=self.device, dtype=torch.bfloat16)
        self.convolved = torch.empty((2, 1280), device=self.device, dtype=torch.bfloat16)
        self.recurrent = torch.empty((2, 768), device=self.device, dtype=torch.bfloat16)
        self.core = torch.empty_like(self.recurrent)
        self.mixed = torch.empty((2, 2560), device=self.device, dtype=torch.bfloat16)
        self.executions = torch.zeros(1, device=self.device, dtype=torch.int32)

    def __call__(self, hidden, conv_state, ssm_state, indices):
        rows = hidden.shape[0]
        require_tensor(hidden, (rows, 10240), device=self.device)
        if rows not in (1, 2) or indices.shape != (rows,) or indices.dtype not in (torch.int32, torch.int64) or indices.device != self.device or not indices.is_contiguous():
            raise FlashContractError("GDN decode rows or cache indices are unsupported")
        if conv_state.ndim != 3 or tuple(conv_state.shape[1:]) != (1280, 3) or conv_state.dtype != torch.bfloat16 or conv_state.device != self.device:
            raise FlashContractError("GDN convolution state layout differs")
        require_tensor(ssm_state, (conv_state.shape[0], 6, 128, 128), torch.float32, self.device)
        mixed = self.mixed[:rows]
        output = torch.empty_like(mixed)
        mix_tensors = (hidden, *self.mixer.weights, self.mixer.normalized, self.mixer.lowrank, self.mixer.gates, mixed, self.mixer.state, self.mixer.executions)
        mix = MixArguments(*(tensor.data_ptr() for tensor in mix_tensors), rows, self.mixer.program.timeout_ticks)
        tensors = (*self.weights, conv_state, ssm_state, indices, self.projected, self.projected_ba, self.convolved, self.recurrent, self.core, output, self.executions)
        arguments = GDNArguments(mix, *(tensor.data_ptr() for tensor in tensors), *conv_state.stride(), ssm_state.stride(0), indices.element_size(), conv_state.shape[0])
        with torch.cuda.device(self.device):
            self.program.base.check(self.program.library.flash_gdn(ctypes.byref(arguments), torch.cuda.current_stream(self.device).cuda_stream))
        return output, (hidden, self.mixer.normalized[:rows])

    def native(self, mixed, conv_state, ssm_state, indices):
        from causal_conv1d.causal_conv1d_interface import causal_conv1d_update
        from sglang.kernels.ops.attention.fla.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
        from sglang.kernels.ops.attention.fla.layernorm_gated import layernorm_fn

        qkvz, ba, conv_weight, norm, a_log, dt_bias, output_weight = self.weights
        projected = functional.linear(mixed, qkvz)
        projected_ba = functional.linear(mixed, ba)
        convolved = causal_conv1d_update(projected[:, :1280].contiguous(), conv_state, conv_weight.view(1280, 4), activation="silu", conv_state_indices=indices.to(torch.int32))
        recurrent = torch.empty((mixed.shape[0], 1, 6, 128), dtype=torch.bfloat16, device=self.device)
        fused_recurrent_gated_delta_rule_packed_decode(convolved, projected_ba[:, 6:].contiguous(), projected_ba[:, :6].contiguous(), a_log, dt_bias, 128 ** -0.5, ssm_state, recurrent, indices, use_qk_l2norm_in_kernel=True)
        core = layernorm_fn(recurrent.reshape(-1, 128), norm, None, z=projected[:, 1280:].contiguous().reshape(-1, 128), eps=1e-6, is_rms_norm=True, activation="sigmoid").reshape(mixed.shape[0], 768)
        return functional.linear(core, output_weight)
