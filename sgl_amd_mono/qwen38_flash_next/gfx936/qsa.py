import ctypes
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from torch.nn import functional

from .program import FlashContractError, HIPProgram, MixArguments, require_tensor


class QSAArguments(ctypes.Structure):
    _fields_ = [("mix", MixArguments)] + [(name, ctypes.c_void_p) for name in (
        "qkv_weight", "q_norm", "k_norm", "output_weight", "index_weight", "index_q_norm", "index_k_norm", "rotary_cache", "positions", "sequence_lengths", "requests", "locations", "request_tokens", "key_cache", "value_cache", "index_ring", "ring_positions", "compressed_keys", "projected", "index_projected", "queries", "index_queries", "index_scores", "selected", "attention_scores", "core", "output", "executions",
    )] + [(name, ctypes.c_int64) for name in (
        "request_stride", "request_count", "cache_slots", "ring_slots", "rotary_rows", "position_axes", "position_axis_stride", "key_page_stride", "key_token_stride", "key_dim_stride", "value_page_stride", "value_token_stride", "value_dim_stride",
    )]


class QSAProgram:
    _instances = {}

    @classmethod
    def get(cls, device):
        device = torch.device(device)
        if device not in cls._instances:
            cls._instances[device] = cls(device)
        return cls._instances[device]

    def __init__(self, device):
        self.base = HIPProgram.get(device)
        sources = {name: Path(__file__).with_name(name).read_bytes() for name in ("qsa.cpp", "kernel.cpp")}
        compiler = "/opt/dtk/bin/hipcc"
        version = subprocess.check_output([compiler, "--version"], text=True)
        digest = hashlib.sha256(b"".join(sources.values()) + version.encode() + torch.__version__.encode()).hexdigest()
        cache = self.base.cache.parent / ("qsa-" + digest[:24])
        cache.mkdir(parents=True, exist_ok=True)
        binary = cache / "qsa.so"
        manifest_path = cache / "manifest.json"
        with (cache / "compile.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not manifest_path.exists():
                for name, content in sources.items():
                    (cache / name).write_bytes(content)
                command = [compiler, "-O3", "-std=c++17", "-fPIC", "-shared", "--offload-arch=gfx936", "-ffp-contract=off", str(cache / "qsa.cpp"), "-o", str(binary)]
                with (cache / "compile.log").open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240)
                if result.returncode:
                    raise FlashContractError(f"QSA compilation failed: {cache / 'compile.log'}")
                manifest_path.write_text(json.dumps({"source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "command": command}, indent=2) + "\n")
        self.manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(binary.read_bytes()).hexdigest() != self.manifest["binary_sha256"]:
            raise FlashContractError("QSA binary hash mismatch")
        self.library = ctypes.CDLL(str(binary))
        self.library.flash_qsa_info.argtypes = [ctypes.POINTER(ctypes.c_int64)]
        self.library.flash_qsa.argtypes = [ctypes.POINTER(QSAArguments), ctypes.c_void_p]
        information = (ctypes.c_int64 * 2)()
        with torch.cuda.device(device):
            self.base.check(self.library.flash_qsa_info(information))
        if information[0] != ctypes.sizeof(QSAArguments) or information[1] < 1:
            raise FlashContractError(f"QSA ABI or cooperative residency check failed: {list(information)}")
        self.information = {"abi_bytes": information[0], "blocks_per_cu": information[1]}


def cache_views(key, value):
    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise FlashContractError("QSA requires BF16 KV caches")
    if key.ndim == 3 and tuple(key.shape[1:]) == (1, 256) and value.shape == key.shape and key.is_contiguous() and value.is_contiguous():
        return key.reshape(-1, 64, 1, 256), value.reshape(-1, 64, 1, 256)
    if key.ndim == 4 and tuple(key.shape[1:]) == (1, 64, 256):
        key = key.permute(0, 2, 1, 3)
        if tuple(value.shape[1:]) == (1, 256, 64):
            return key, value.permute(0, 3, 1, 2)
        if tuple(value.shape[1:]) == (1, 64, 256):
            return key, value.permute(0, 2, 1, 3)
    raise FlashContractError("Unsupported QSA KV layout")


class QSAAttention:
    def __init__(self, layer, mixer):
        self.mixer = mixer
        self.device = mixer.device
        self.program = QSAProgram.get(self.device)
        self.weights = (layer.qkv_proj.weight, layer.q_norm.weight, layer.k_norm.weight, layer.o_proj.weight, layer.indexer.index_qk_proj.weight, layer.indexer.q_layernorm.weight, layer.indexer.k_layernorm.weight)
        for tensor, shape in zip(self.weights, ((2048, 2560), (256,), (256,), (2560, 768), (640, 2560), (128,), (128,))):
            require_tensor(tensor, shape, device=self.device)
        self.rotary_cache = layer.rotary_emb.cos_sin_cache.float().contiguous()
        require_tensor(self.rotary_cache, self.rotary_cache.shape, torch.float32, self.device)
        if self.rotary_cache.ndim != 2 or self.rotary_cache.shape[1] != 64:
            raise FlashContractError("QSA requires a 64-wide RoPE cache")
        self.projected = torch.empty((2, 2048), dtype=torch.bfloat16, device=self.device)
        self.mixed = torch.empty((2, 2560), dtype=torch.bfloat16, device=self.device)
        self.index_projected = torch.empty((2, 640), dtype=torch.bfloat16, device=self.device)
        self.queries = torch.empty((2, 3, 256), dtype=torch.bfloat16, device=self.device)
        self.index_queries = torch.empty((2, 4, 128), dtype=torch.bfloat16, device=self.device)
        self.index_scores = torch.empty((2, 1024), dtype=torch.float32, device=self.device)
        self.selected = torch.empty((2, 2051), dtype=torch.int32, device=self.device)
        self.attention_scores = torch.empty((2, 3, 2051), dtype=torch.float32, device=self.device)
        self.core = torch.empty((2, 768), dtype=torch.bfloat16, device=self.device)
        self.executions = torch.zeros(1, dtype=torch.int32, device=self.device)

    def __call__(self, hidden, positions, lengths, requests, locations, request_tokens, key, value, ring, ring_positions, compressed):
        rows = hidden.shape[0]
        require_tensor(hidden, (rows, 10240), device=self.device)
        if rows not in (1, 2):
            raise FlashContractError("QSA decode supports one or two rows")
        axes = 1 if positions.ndim == 1 else 3
        require_tensor(positions, (rows,) if axes == 1 else (3, rows), torch.int64, self.device)
        for tensor, dtype in ((lengths, torch.int32), (requests, torch.int64), (locations, torch.int64)):
            require_tensor(tensor, (rows,), dtype, self.device)
        require_tensor(request_tokens, request_tokens.shape, torch.int32, self.device)
        require_tensor(ring, (ring.shape[0], 1, 128), device=self.device)
        require_tensor(ring_positions, (ring.shape[0], 3), torch.int64, self.device)
        require_tensor(compressed, (compressed.shape[0], 1, 128), device=self.device)
        if request_tokens.ndim != 2 or request_tokens.shape[1] < 4096:
            raise FlashContractError("QSA request table must cover context 4096")
        key, value = cache_views(key, value)
        if key.shape != value.shape or key.device != self.device or value.device != self.device or key.shape[0] * 16 > compressed.shape[0]:
            raise FlashContractError("QSA cache capacity or device mismatch")
        output = torch.empty((rows, 2560), dtype=torch.bfloat16, device=self.device)
        mixed = self.mixed[:rows]
        mix_tensors = (hidden, *self.mixer.weights, self.mixer.normalized, self.mixer.lowrank, self.mixer.gates, mixed, self.mixer.state, self.mixer.executions)
        mix = MixArguments(*(tensor.data_ptr() for tensor in mix_tensors), rows, self.mixer.program.timeout_ticks)
        tensors = (*self.weights, self.rotary_cache, positions, lengths, requests, locations, request_tokens, key, value, ring, ring_positions, compressed, self.projected, self.index_projected, self.queries, self.index_queries, self.index_scores, self.selected, self.attention_scores, self.core, output, self.executions)
        arguments = QSAArguments(mix, *(tensor.data_ptr() for tensor in tensors), request_tokens.stride(0), request_tokens.shape[0], key.shape[0] * 64, ring.shape[0], self.rotary_cache.shape[0], axes, 0 if axes == 1 else positions.stride(0), key.stride(0), key.stride(1), key.stride(3), value.stride(0), value.stride(1), value.stride(3))
        with torch.cuda.device(self.device):
            self.program.base.check(self.program.library.flash_qsa(ctypes.byref(arguments), torch.cuda.current_stream(self.device).cuda_stream))
        return output, (hidden, self.mixer.normalized[:rows])

    @staticmethod
    def norm(value, weight):
        value = value.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * (1 + weight.float())).bfloat16()

    def rope(self, value, positions):
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        axis = torch.zeros(32, dtype=torch.int64, device=self.device)
        axis[1:32:3] = 1
        axis[2:30:3] = 2
        lookup = self.rotary_cache[positions[axis, :].transpose(0, 1), torch.arange(32, device=self.device)]
        cosine = lookup.bfloat16().unsqueeze(1)
        lookup = self.rotary_cache[positions[axis, :].transpose(0, 1), torch.arange(32, device=self.device) + 32]
        sine = lookup.bfloat16().unsqueeze(1)
        first, second = value[..., :32], value[..., 32:64]
        return torch.cat((first * cosine - second * sine, second * cosine + first * sine, value[..., 64:]), dim=-1)

    def native(self, mixed, positions, lengths, requests, locations, request_tokens, key, value, ring, ring_positions, compressed):
        from sglang.srt.layers.attention.qsa.sparse_attn import sparse_gqa_packed_decode_triton
        from sglang.kernels.ops.elementwise.fast_topk import fast_topk

        rows = mixed.shape[0]
        key, value = cache_views(key, value)
        qkv_weight, q_norm, k_norm, output_weight, index_weight, index_q_norm, index_k_norm = self.weights
        projected = functional.linear(mixed, qkv_weight)
        query_gate = projected[:, :1536].view(rows, 3, 512)
        query = self.rope(self.norm(query_gate[:, :, :256], q_norm), positions)
        keys = self.rope(self.norm(projected[:, 1536:1792].view(rows, 1, 256), k_norm), positions)
        loc = locations.long()
        key[loc // 64, loc % 64, 0] = keys[:, 0]
        value[loc // 64, loc % 64, 0] = projected[:, 1792:]
        index_projected = functional.linear(mixed, index_weight)
        index_query = self.rope(self.norm(index_projected[:, :512].view(rows, 4, 128), index_q_norm), positions)
        slots = requests * 4 + (lengths.long() - 1) % 4
        ring[slots, 0] = index_projected[:, 512:]
        ring_positions[slots] = positions.unsqueeze(1).expand(rows, 3) if positions.ndim == 1 else positions.transpose(0, 1)
        selected = torch.full((rows, 2051), -1, dtype=torch.int32, device=self.device)
        logits = torch.full((rows, 1024), -torch.inf, dtype=torch.float32, device=self.device)
        packed_keys, packed_values, counts = [], [], []
        for row in range(rows):
            length = int(lengths[row])
            request = int(requests[row])
            groups = length // 4
            if length % 4 == 0:
                average = ring[request * 4:request * 4 + 4, 0].float().mean(0).bfloat16().view(1, 1, 128)
                normed = self.norm(average, index_k_norm)
                compressed[loc[row] // 4, 0] = self.rope(normed, ring_positions[request * 4].view(3, 1))[0, 0]
            if groups:
                group_slots = request_tokens[request, :groups * 4:4].long() // 4
                scores = torch.einsum("hd,nd->nh", index_query[row].float(), compressed[group_slots, 0].float()).relu().sum(-1)
                logits[row, :groups] = scores
                blocks = fast_topk(scores.unsqueeze(0).contiguous(), torch.tensor([groups], dtype=torch.int32, device=self.device), topk=512)[0]
                blocks = blocks[blocks >= 0].long()
                tokens = (blocks[:, None] * 4 + torch.arange(4, device=self.device)).flatten()
            else:
                tokens = torch.empty(0, dtype=torch.int64, device=self.device)
            tokens = torch.cat((tokens, torch.arange(groups * 4, length, device=self.device)))
            selected[row, :tokens.numel()] = tokens.int()
            physical = request_tokens[request, tokens].long()
            packed_keys.append(key[physical // 64, physical % 64])
            packed_values.append(value[physical // 64, physical % 64])
            counts.append(tokens.numel())
        counts = torch.tensor(counts, dtype=torch.int32, device=self.device)
        offsets = torch.cat((torch.zeros(1, dtype=torch.int32, device=self.device), counts.cumsum(0).int()))
        relative = torch.arange(2051, device=self.device, dtype=torch.int32).expand(rows, -1).clone()
        relative.masked_fill_(relative >= counts[:, None], -1)
        attention = sparse_gqa_packed_decode_triton(query.contiguous(), torch.cat(packed_keys), torch.cat(packed_values), relative, torch.arange(rows + 1, dtype=torch.int32, device=self.device), offsets, counts, 0.0625)
        core = (attention * torch.sigmoid(query_gate[:, :, 256:].float())).bfloat16().reshape(rows, 768)
        return functional.linear(core, output_weight), {"queries": query, "index_queries": index_query, "index_scores": logits, "selected": selected, "core": core}
