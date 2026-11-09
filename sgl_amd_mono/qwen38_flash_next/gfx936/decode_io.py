import ctypes
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from .program import FlashContractError, HIPProgram, MixArguments, SyncArguments, require_tensor


class EntryArguments(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in ("input_ids", "weight", "local", "reduced", "output", "executions")] + [("sync", SyncArguments)] + [(name, ctypes.c_int64) for name in ("rows", "index_bytes", "vocab_start")]


class TailArguments(ctypes.Structure):
    _fields_ = [("mix", MixArguments)] + [(name, ctypes.c_void_p) for name in ("weight", "local", "logits", "executions")] + [("sync", SyncArguments)]


class IOProgram:
    _instances = {}

    @classmethod
    def get(cls, device):
        device = torch.device(device)
        if device not in cls._instances:
            cls._instances[device] = cls(device)
        return cls._instances[device]

    def __init__(self, device):
        self.base = HIPProgram.get(device)
        sources = {name: Path(__file__).with_name(name).read_bytes() for name in ("decode_io.cpp", "kernel.cpp")}
        compiler = "/opt/dtk/bin/hipcc"
        version = subprocess.check_output([compiler, "--version"], text=True)
        digest = hashlib.sha256(b"".join(sources.values()) + version.encode() + torch.__version__.encode()).hexdigest()
        cache = self.base.cache.parent / ("io-" + digest[:24])
        cache.mkdir(parents=True, exist_ok=True)
        binary = cache / "io.so"
        manifest_path = cache / "manifest.json"
        with (cache / "compile.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not manifest_path.exists():
                for name, content in sources.items():
                    (cache / name).write_bytes(content)
                command = [compiler, "-O3", "-std=c++17", "-fPIC", "-shared", "--offload-arch=gfx936", "-ffp-contract=off", str(cache / "decode_io.cpp"), "-o", str(binary)]
                with (cache / "compile.log").open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240)
                if result.returncode:
                    raise FlashContractError(f"Decode IO compilation failed: {cache / 'compile.log'}")
                manifest_path.write_text(json.dumps({"source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "command": command}, indent=2) + "\n")
        self.manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(binary.read_bytes()).hexdigest() != self.manifest["binary_sha256"]:
            raise FlashContractError("Decode IO binary hash mismatch")
        self.library = ctypes.CDLL(str(binary))
        self.library.flash_io_info.argtypes = [ctypes.POINTER(ctypes.c_int64)]
        self.library.flash_entry.argtypes = [ctypes.POINTER(EntryArguments), ctypes.c_void_p]
        self.library.flash_tail.argtypes = [ctypes.POINTER(TailArguments), ctypes.c_void_p]
        information = (ctypes.c_int64 * 4)()
        with torch.cuda.device(device):
            self.base.check(self.library.flash_io_info(information))
        if list(information[:2]) != [ctypes.sizeof(EntryArguments), ctypes.sizeof(TailArguments)] or min(information[2:]) < 1:
            raise FlashContractError(f"Decode IO ABI or residency differs: {list(information)}")
        self.information = {"entry_abi_bytes": information[0], "tail_abi_bytes": information[1], "entry_blocks_per_cu": information[2], "tail_blocks_per_cu": information[3]}


class DecodeIO:
    def __init__(self, model, mixer, peers):
        self.device = peers.device
        self.peers = peers
        self.mixer = mixer
        self.program = IOProgram.get(self.device)
        self.embedding = model.model.embed_tokens.weight
        self.head = model.lm_head.weight
        require_tensor(self.embedding, (31040, 2560), device=self.device)
        require_tensor(self.head, (31040, 2560), device=self.device)
        for module in (model.model.embed_tokens, model.lm_head):
            if module.shard_indices.org_vocab_start_index != peers.rank * 31040 or module.shard_indices.org_vocab_end_index != (peers.rank + 1) * 31040:
                raise FlashContractError("Decode IO requires contiguous equal vocabulary shards")
        processor = model.logits_processor
        if processor.use_fp32_lm_head or processor.vocab_size != 248320 or not processor.do_tensor_parallel_all_gather or processor.logit_scale is not None or processor.final_logit_softcapping:
            raise FlashContractError("Decode IO requires the native BF16 TP8 logits contract without scaling")
        self.local_embedding = torch.empty((2, 2560), dtype=torch.bfloat16, device=self.device)
        self.reduced_embedding = torch.empty_like(self.local_embedding)
        self.local_logits = torch.empty((2, 31040), dtype=torch.bfloat16, device=self.device)
        self.entry_executions = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.tail_executions = torch.zeros_like(self.entry_executions)

    def entry(self, input_ids):
        rows = input_ids.numel()
        if rows not in (1, 2) or input_ids.dtype not in (torch.int32, torch.int64):
            raise FlashContractError("Decode embedding requires one or two integer token IDs")
        require_tensor(input_ids, (rows,), input_ids.dtype, self.device)
        output = torch.empty((rows, 10240), dtype=torch.bfloat16, device=self.device)
        tensors = (input_ids, self.embedding, self.local_embedding, self.reduced_embedding, output, self.entry_executions)
        arguments = EntryArguments(*(tensor.data_ptr() for tensor in tensors), self.peers.arguments(), rows, input_ids.element_size(), self.peers.rank * 31040)
        with torch.cuda.device(self.device):
            self.program.base.check(self.program.library.flash_entry(ctypes.byref(arguments), torch.cuda.current_stream(self.device).cuda_stream))
        return output

    def tail(self, hidden):
        rows = hidden.shape[0]
        require_tensor(hidden, (rows, 10240), device=self.device)
        if rows not in (1, 2):
            raise FlashContractError("Decode tail requires one or two rows")
        mixed = torch.empty((rows, 2560), dtype=torch.bfloat16, device=self.device)
        logits = torch.empty((rows, 248320), dtype=torch.bfloat16, device=self.device)
        mix_tensors = (hidden, *self.mixer.weights, self.mixer.normalized, self.mixer.lowrank, self.mixer.gates, mixed, self.mixer.state, self.mixer.executions)
        mix = MixArguments(*(tensor.data_ptr() for tensor in mix_tensors), rows, self.mixer.program.timeout_ticks)
        arguments = TailArguments(mix, self.head.data_ptr(), self.local_logits.data_ptr(), logits.data_ptr(), self.tail_executions.data_ptr(), self.peers.arguments())
        with torch.cuda.device(self.device):
            self.program.base.check(self.program.library.flash_tail(ctypes.byref(arguments), torch.cuda.current_stream(self.device).cuda_stream))
        return logits, mixed
