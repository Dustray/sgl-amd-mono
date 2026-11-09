import ctypes
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from .program import FlashContractError, HIPProgram, SyncArguments, require_tensor


class PLEArguments(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "input", "contexts", "multipliers", "vocab_sizes", "offsets", "embedding_weight", "key_weight", "value_weight", "key_norm_weight", "query_norm_weight", "conv_norm_weight", "conv_weight", "state_indices", "conv_state", "local_embedding", "embedding", "key", "value", "normalized_key", "normalized_query", "gated", "normalized_gated", "output", "executions",
    )] + [("sync", SyncArguments)] + [(name, ctypes.c_int64) for name in ("rows", "vocab_start", "vocab_end", "eos_token", "state_slots")]


class PLEProgram:
    _instances = {}

    @classmethod
    def get(cls, device):
        device = torch.device(device)
        if device not in cls._instances:
            cls._instances[device] = cls(device)
        return cls._instances[device]

    def __init__(self, device):
        self.base = HIPProgram.get(device)
        sources = {name: Path(__file__).with_name(name).read_bytes() for name in ("ple.cpp", "kernel.cpp")}
        compiler = "/opt/dtk/bin/hipcc"
        version = subprocess.check_output([compiler, "--version"], text=True)
        digest = hashlib.sha256(b"".join(sources.values()) + version.encode() + torch.__version__.encode()).hexdigest()
        cache = self.base.cache.parent / ("ple-" + digest[:24])
        cache.mkdir(parents=True, exist_ok=True)
        binary = cache / "ple.so"
        manifest_path = cache / "manifest.json"
        with (cache / "compile.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not manifest_path.exists():
                for name, content in sources.items():
                    (cache / name).write_bytes(content)
                command = [compiler, "-O3", "-std=c++17", "-fPIC", "-shared", "--offload-arch=gfx936", "-ffp-contract=off", str(cache / "ple.cpp"), "-o", str(binary)]
                with (cache / "compile.log").open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240)
                if result.returncode:
                    raise FlashContractError(f"PLE compilation failed: {cache / 'compile.log'}")
                manifest_path.write_text(json.dumps({"source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "command": command}, indent=2) + "\n")
        self.manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(binary.read_bytes()).hexdigest() != self.manifest["binary_sha256"]:
            raise FlashContractError("PLE binary hash mismatch")
        self.library = ctypes.CDLL(str(binary))
        self.library.flash_ple_info.argtypes = [ctypes.POINTER(ctypes.c_int64)]
        self.library.flash_ple.argtypes = [ctypes.POINTER(PLEArguments), ctypes.c_void_p]
        information = (ctypes.c_int64 * 2)()
        with torch.cuda.device(device):
            self.base.check(self.library.flash_ple_info(information))
        if information[0] != ctypes.sizeof(PLEArguments) or information[1] < 1:
            raise FlashContractError(f"PLE ABI or cooperative residency check failed: {list(information)}")
        self.information = {"abi_bytes": information[0], "blocks_per_cu": information[1]}


class PLE:
    def __init__(self, layer, peers):
        self.device = peers.device
        self.peers = peers
        self.program = PLEProgram.get(self.device)
        embedding = layer.ple_embedding
        table = embedding.ngram_embedding
        self.weights = (embedding.layer_multipliers, embedding.ngram_heads_vocab_sizes, embedding.ngram_heads_offsets, table.weight, layer.key_proj.weight, layer.value_proj.weight, layer.norm_key.weight, layer.norm_query.weight, layer.norm_conv.weight, layer.conv1d.weight)
        self.vocab_start = table.shard_indices.org_vocab_start_index
        self.vocab_end = table.shard_indices.org_vocab_end_index
        self.eos_token = embedding.eos_token_id
        shapes = ((3,), (16,), (16,), (table.weight.shape[0], 160), (10240, 2560), (2560, 2560), (10240,), (10240,), (10240,), (10240, 1, 4))
        for index, (tensor, shape) in enumerate(zip(self.weights, shapes)):
            require_tensor(tensor, shape, torch.int64 if index < 3 else torch.bfloat16, self.device)
        if embedding.ngram_size != 3 or embedding.heads_per_ngram != 8 or layer.short_conv_dilation != 3 or layer.short_conv_state_len != 9 or embedding.gather_dp_tokens:
            raise FlashContractError("PLE hash, state, or parallel layout differs")
        self.local_embedding = torch.empty((2, 2560), dtype=torch.bfloat16, device=self.device)
        self.embedding = torch.empty_like(self.local_embedding)
        self.key = torch.empty((2, 10240), dtype=torch.bfloat16, device=self.device)
        self.value = torch.empty_like(self.embedding)
        self.normalized_key = torch.empty_like(self.key)
        self.normalized_query = torch.empty_like(self.key)
        self.gated = torch.empty_like(self.key)
        self.normalized_gated = torch.empty_like(self.key)
        self.executions = torch.zeros(1, dtype=torch.int32, device=self.device)

    def __call__(self, hidden, contexts, state_indices, conv_state):
        rows = hidden.shape[0]
        if rows not in (1, 2):
            raise FlashContractError("PLE decode supports one or two rows")
        require_tensor(hidden, (rows, 10240), device=self.device)
        require_tensor(contexts, (rows, 3), torch.int64, self.device)
        require_tensor(state_indices, (rows,), torch.int64, self.device)
        require_tensor(conv_state, (conv_state.shape[0], 10240, 9), device=self.device)
        output = torch.empty_like(hidden)
        tensors = (hidden, contexts, *self.weights, state_indices, conv_state, self.local_embedding, self.embedding, self.key, self.value, self.normalized_key, self.normalized_query, self.gated, self.normalized_gated, output, self.executions)
        arguments = PLEArguments(*(tensor.data_ptr() for tensor in tensors), self.peers.arguments(), rows, self.vocab_start, self.vocab_end, self.eos_token, conv_state.shape[0])
        with torch.cuda.device(self.device):
            self.program.base.check(self.program.library.flash_ple(ctypes.byref(arguments), torch.cuda.current_stream(self.device).cuda_stream))
        return output
