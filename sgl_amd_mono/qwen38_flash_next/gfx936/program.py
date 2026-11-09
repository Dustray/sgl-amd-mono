import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch


class FlashContractError(RuntimeError):
    pass


class SyncArguments(ctypes.Structure):
    _fields_ = [("peers", ctypes.c_void_p * 8), ("state", ctypes.c_void_p), ("rank", ctypes.c_int64), ("timeout_ticks", ctypes.c_int64)]


class MixArguments(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "input", "norm_weight", "down_weight", "up_weight", "normalized", "lowrank", "gates", "output", "synchronization", "executions",
    )] + [("rows", ctypes.c_int64), ("timeout_ticks", ctypes.c_int64)]


class PostArguments(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "attention", "residual", "normalized_residual", "attention_inject", "norm_weight", "mix_down", "mix_up", "mlp_inject", "router_weight", "shared_gate_weight", "expert_gate_up", "expert_down", "shared_gate_up", "shared_down", "reduced_attention", "post_attention", "normalized", "lowrank", "gates", "mixed", "router_logits", "top_ids", "top_weights", "inject_gates", "projected", "activated", "expert_output", "local_moe", "reduced_moe", "output", "executions",
    )] + [("sync", SyncArguments), ("rows", ctypes.c_int64)]


def require_tensor(tensor, shape, dtype=torch.bfloat16, device=None):
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype or tensor.device.type != "cuda" or not tensor.is_contiguous():
        raise FlashContractError(f"Expected contiguous GPU {dtype} tensor {shape}, got {tensor.shape} {tensor.dtype} {tensor.device} strides={tensor.stride()}")
    if device is not None and tensor.device != device:
        raise FlashContractError(f"Tensor device {tensor.device} differs from {device}")


class HIPProgram:
    _instances = {}

    @classmethod
    def get(cls, device):
        device = torch.device(device)
        if device not in cls._instances:
            cls._instances[device] = cls(device)
        return cls._instances[device]

    def __init__(self, device):
        self.device = device
        properties = torch.cuda.get_device_properties(device)
        if properties.gcnArchName.split(":")[0] != "gfx936" or properties.multi_processor_count != 80:
            raise FlashContractError("Flash-Next backend requires gfx936 with 80 CUs")
        source = Path(__file__).with_name("kernel.cpp")
        compiler = "/opt/dtk/bin/hipcc"
        version = subprocess.check_output([compiler, "--version"], text=True)
        source_bytes = source.read_bytes()
        digest = hashlib.sha256(source_bytes + version.encode() + torch.__version__.encode()).hexdigest()
        self.cache = Path(os.environ.get("SGLANG_AMD_FLASH_REPORT_DIR", "monokernel_reports/flash_next_tp8")).resolve() / "codegen" / digest[:24]
        self.cache.mkdir(parents=True, exist_ok=True)
        binary = self.cache / "flash_next.so"
        manifest_path = self.cache / "manifest.json"
        with (self.cache / "compile.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not manifest_path.exists():
                frozen_source = self.cache / "kernel.cpp"
                frozen_source.write_bytes(source_bytes)
                command = [compiler, "-O3", "-std=c++17", "-fPIC", "-shared", "--offload-arch=gfx936", "-ffp-contract=off", str(frozen_source), "-o", str(binary)]
                with (self.cache / "compile.log").open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240)
                if result.returncode:
                    raise FlashContractError(f"HIP compilation failed: {self.cache / 'compile.log'}")
                manifest_path.write_text(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "torch": torch.__version__, "compiler": version, "command": command, "blocks": 80, "threads": 256, "cooperative": True}, indent=2) + "\n")
        self.manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(binary.read_bytes()).hexdigest() != self.manifest["binary_sha256"]:
            raise FlashContractError("Compiled Flash-Next binary hash mismatch")
        self.library = ctypes.CDLL(str(binary))
        self.library.flash_info.argtypes = [ctypes.POINTER(ctypes.c_int64)]
        self.library.flash_allocate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.library.flash_open.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.library.flash_close.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.library.flash_snapshot.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        self.library.flash_mix.argtypes = [ctypes.POINTER(MixArguments), ctypes.c_void_p]
        self.library.flash_post.argtypes = [ctypes.POINTER(PostArguments), ctypes.c_void_p]
        self.library.flash_reduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(SyncArguments), ctypes.c_void_p]
        self.library.flash_reduce_pair.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(SyncArguments), ctypes.c_void_p]
        information = (ctypes.c_int64 * 9)()
        with torch.cuda.device(device):
            self.check(self.library.flash_info(information))
        self.information = dict(zip(("mix_abi_bytes", "post_abi_bytes", "sync_abi_bytes", "peer_bytes", "cooperative", "compute_units", "mix_occupancy", "post_occupancy", "clock_khz"), information))
        if list(information[:3]) != [ctypes.sizeof(MixArguments), ctypes.sizeof(PostArguments), ctypes.sizeof(SyncArguments)]:
            raise FlashContractError(f"Flash-Next C ABI mismatch: {self.information}")
        if information[4] != 1 or information[5] != 80 or min(information[6:]) < 1:
            raise FlashContractError(f"Flash-Next cooperative occupancy unavailable: {self.information}")
        self.timeout_ticks = information[8] * 1000 * 60

    @staticmethod
    def check(status):
        if status:
            raise FlashContractError(f"Flash-Next HIP status={status}; a failed stateful launch cannot fall back")


class PeerGroup:
    def __init__(self, device, rank, cpu_group):
        import torch.distributed as distributed

        if distributed.get_world_size(cpu_group) != 8 or not 0 <= rank < 8:
            raise FlashContractError("Flash-Next peer group requires exactly eight local ranks")
        self.program = HIPProgram.get(device)
        self.device = torch.device(device)
        self.rank = rank
        self.transport = "hip_ipc_uncached_atomic32_tagged_bytes"
        self.state = torch.zeros(16, dtype=torch.int32, device=device)
        self.pointers = []
        self.closed = False
        pointer = ctypes.c_void_p()
        handle = (ctypes.c_byte * 64)()
        with torch.cuda.device(device):
            self.program.check(self.program.library.flash_allocate(ctypes.byref(pointer), handle))
            handles = [None] * 8
            distributed.all_gather_object(handles, bytes(handle), group=cpu_group)
            for peer, peer_handle in enumerate(handles):
                if peer == rank:
                    self.pointers.append(pointer.value)
                else:
                    remote = ctypes.c_void_p()
                    buffer = ctypes.create_string_buffer(peer_handle, 64)
                    self.program.check(self.program.library.flash_open(ctypes.byref(remote), buffer))
                    self.pointers.append(remote.value)
            torch.cuda.synchronize(device)
            distributed.barrier(group=cpu_group)

    def arguments(self):
        if self.closed:
            raise FlashContractError("Flash-Next peer group is closed")
        return SyncArguments((ctypes.c_void_p * 8)(*self.pointers), self.state.data_ptr(), self.rank, self.program.timeout_ticks)

    def reduce(self, tensor, output=None):
        if tensor.numel() > 20480 or tensor.numel() < 1:
            raise FlashContractError("Flash-Next reduction exceeds its validated capacity")
        require_tensor(tensor, tensor.shape, device=self.device)
        output = torch.empty_like(tensor) if output is None else output
        require_tensor(output, tensor.shape, device=self.device)
        sync = self.arguments()
        with torch.cuda.device(self.device):
            self.program.check(self.program.library.flash_reduce(tensor.data_ptr(), output.data_ptr(), tensor.numel(), ctypes.byref(sync), torch.cuda.current_stream(self.device).cuda_stream))
        return output

    def check(self):
        values = self.state.cpu().tolist()
        if values[2]:
            raise FlashContractError(f"Flash-Next device synchronization failed at stage {values[2]} on rank {self.rank}")
        return {"barrier_arrivals": values[0], "barrier_epoch": values[1], "failure_stage": values[2], "collective_epoch": values[3]}

    def reduce_pair(self, first, second, outputs):
        for tensor in (first, second, *outputs):
            require_tensor(tensor, first.shape, device=self.device)
        if first.numel() < 1 or first.numel() > 20480:
            raise FlashContractError("Flash-Next reduction exceeds its validated capacity")
        sync = self.arguments()
        with torch.cuda.device(self.device):
            self.program.check(self.program.library.flash_reduce_pair(first.data_ptr(), second.data_ptr(), outputs[0].data_ptr(), outputs[1].data_ptr(), first.numel(), ctypes.byref(sync), torch.cuda.current_stream(self.device).cuda_stream))
        return outputs

    def snapshot(self):
        peers = []
        for pointer in self.pointers:
            values = (ctypes.c_uint * 18)()
            self.program.check(self.program.library.flash_snapshot(pointer, values))
            peers.append(list(values))
        return {"state": self.state.cpu().tolist(), "peers": peers}

    def close(self, cpu_group):
        import torch.distributed as distributed

        if self.closed:
            return
        torch.cuda.synchronize(self.device)
        distributed.barrier(group=cpu_group)
        with torch.cuda.device(self.device):
            for peer, pointer in enumerate(self.pointers):
                if peer != self.rank:
                    self.program.check(self.program.library.flash_close(pointer, 0))
            distributed.barrier(group=cpu_group)
            self.program.check(self.program.library.flash_close(self.pointers[self.rank], 1))
        self.closed = True


class HCMix:
    def __init__(self, norm_weight, down, up):
        self.device = norm_weight.device
        for tensor, shape in ((norm_weight, (10240,)), (down, (320, 10240)), (up, (10240, 320))):
            require_tensor(tensor, shape, device=self.device)
        self.weights = (norm_weight, down, up)
        self.program = HIPProgram.get(self.device)
        self.normalized = torch.empty((2, 10240), dtype=torch.bfloat16, device=self.device)
        self.lowrank = torch.empty((2, 320), dtype=torch.bfloat16, device=self.device)
        self.gates = torch.empty((2, 10240), dtype=torch.float32, device=self.device)
        self.state = torch.zeros(4, dtype=torch.int32, device=self.device)
        self.executions = torch.zeros(1, dtype=torch.int32, device=self.device)

    def __call__(self, hidden):
        rows = hidden.shape[0]
        if rows not in (1, 2):
            raise FlashContractError("Flash-Next HC mix supports one or two decode rows")
        require_tensor(hidden, (rows, 10240), device=self.device)
        output = torch.empty((rows, 2560), dtype=torch.bfloat16, device=self.device)
        tensors = (hidden, *self.weights, self.normalized, self.lowrank, self.gates, output, self.state, self.executions)
        arguments = MixArguments(*(tensor.data_ptr() for tensor in tensors), rows, self.program.timeout_ticks)
        with torch.cuda.device(self.device):
            self.program.check(self.program.library.flash_mix(ctypes.byref(arguments), torch.cuda.current_stream(self.device).cuda_stream))
        return output, (hidden, self.normalized[:rows])


class PostAttention:
    def __init__(self, layer, peers):
        self.peers = peers
        self.program = peers.program
        self.device = peers.device
        attention_hc = layer.attn_hyper_connection
        mlp_hc = layer.mlp_hyper_connection
        mlp = layer.mlp
        if mlp.shared_expert is None or mlp.num_fused_shared_experts != 0:
            raise FlashContractError("BF16 Flash-Next requires separate shared expert weights")
        self.weights = (
            attention_hc.block_inject_weight.weight,
            mlp_hc.hc_norm.weight,
            mlp_hc.input_mix_weight_down.weight,
            mlp_hc.input_mix_weight_up.weight,
            mlp_hc.block_inject_weight.weight,
            mlp.gate.weight,
            mlp.shared_expert_gate.weight,
            mlp.experts.w13_weight,
            mlp.experts.w2_weight,
            mlp.shared_expert.gate_up_proj.weight,
            mlp.shared_expert.down_proj.weight,
        )
        shapes = ((4, 10240), (10240,), (320, 10240), (10240, 320), (4, 10240), (512, 2560), (1, 2560), (512, 160, 2560), (512, 2560, 80), (160, 2560), (2560, 80))
        for tensor, shape in zip(self.weights, shapes):
            require_tensor(tensor, shape, device=self.device)
        specifications = (
            ("reduced_attention", (2, 2560), torch.bfloat16),
            ("post_attention", (2, 10240), torch.bfloat16),
            ("normalized", (2, 10240), torch.bfloat16),
            ("lowrank", (2, 320), torch.bfloat16),
            ("gates", (2, 10240), torch.float32),
            ("mixed", (2, 2560), torch.bfloat16),
            ("router_logits", (2, 512), torch.bfloat16),
            ("top_ids", (2, 10), torch.int32),
            ("top_weights", (2, 10), torch.float32),
            ("inject_gates", (2, 5), torch.float32),
            ("projected", (2, 11, 160), torch.bfloat16),
            ("activated", (2, 11, 80), torch.bfloat16),
            ("expert_output", (2, 11, 2560), torch.bfloat16),
            ("local_moe", (2, 2560), torch.bfloat16),
            ("reduced_moe", (2, 2560), torch.bfloat16),
        )
        self.buffers = []
        for name, shape, dtype in specifications:
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            setattr(self, name, tensor)
            self.buffers.append(tensor)
        self.executions = torch.zeros(1, dtype=torch.int32, device=self.device)

    def __call__(self, attention, residuals):
        hidden, normalized = residuals
        rows = attention.shape[0]
        if rows not in (1, 2):
            raise FlashContractError("Flash-Next post-attention supports one or two decode rows")
        for tensor, shape in ((attention, (rows, 2560)), (hidden, (rows, 10240)), (normalized, (rows, 10240))):
            require_tensor(tensor, shape, device=self.device)
        output = torch.empty((rows, 10240), dtype=torch.bfloat16, device=self.device)
        tensors = (attention, hidden, normalized, *self.weights, *self.buffers, output, self.executions)
        arguments = PostArguments(*(tensor.data_ptr() for tensor in tensors), self.peers.arguments(), rows)
        with torch.cuda.device(self.device):
            self.program.check(self.program.library.flash_post(ctypes.byref(arguments), torch.cuda.current_stream(self.device).cuda_stream))
        return output
