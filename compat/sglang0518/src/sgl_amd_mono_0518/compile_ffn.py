import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import re
from pathlib import Path

import msgpack
import torch
from elftools.elf.elffile import ELFFile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    arguments = parser.parse_args()
    os.environ["COMPILE_ONLY"] = "1"
    from .bf16_ffn import launch_ffn

    hidden_size = arguments.hidden_size
    intermediate_size = arguments.intermediate_size
    destination = arguments.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "architecture": torch.cuda.get_device_properties(0).gcnArchName.split(":")[0],
        "flydsl": importlib.metadata.version("flydsl"),
        "specializations": {},
    }
    hidden = torch.empty(2 * hidden_size, dtype=torch.bfloat16, device="cuda")
    gate_up = torch.empty(2 * hidden_size * intermediate_size, dtype=torch.bfloat16, device="cuda")
    down = torch.empty(hidden_size * intermediate_size, dtype=torch.bfloat16, device="cuda")
    partial = torch.empty(2 * (intermediate_size // 256) * hidden_size, dtype=torch.float32, device="cuda")
    output = torch.empty_like(hidden)
    executions = torch.empty(1, dtype=torch.int32, device="cuda")
    for rows in (1, 2):
        launch_arguments = (hidden[: rows * hidden_size], gate_up, down, partial, output[: rows * hidden_size], executions,
                            hidden_size, intermediate_size, rows, torch.cuda.current_stream())
        launch_ffn(*launch_arguments)
        bound = launch_ffn._sig.bind(*launch_arguments)
        key = launch_ffn._build_full_cache_key(bound.arguments)
        artifact = launch_ffn._mem_cache[key]
        match = re.search(r'bin = "((?:[^"\\]|\\.)*)"', artifact._ir_text)
        if match is None:
            raise RuntimeError("FlyDSL did not emit a GPU binary")
        encoded = match.group(1)
        binary = bytearray()
        position = 0
        while position < len(encoded):
            if encoded[position] == "\\":
                if encoded[position + 1] in ('\\', '"'):
                    binary.append(ord(encoded[position + 1]))
                    position += 2
                else:
                    binary.append(int(encoded[position + 1:position + 3], 16))
                    position += 3
            else:
                binary.append(ord(encoded[position]))
                position += 1
        binary = bytes(binary)
        kernels = None
        for section in ELFFile(io.BytesIO(binary)).iter_sections():
            if section.header.sh_type != "SHT_NOTE":
                continue
            for note in section.iter_notes():
                if note["n_name"] == "AMDGPU" and note["n_type"] == 32:
                    kernels = msgpack.unpackb(note["n_desc"], raw=False)["amdhsa.kernels"]
        if kernels is None or len(kernels) != 2:
            raise RuntimeError("Expected two kernel definitions in the GPU binary")
        filename = f"ffn_rows_{rows}.hsaco"
        (destination / filename).write_bytes(binary)
        manifest["specializations"][str(rows)] = {
            "file": filename,
            "sha256": hashlib.sha256(binary).hexdigest(),
            "kernels": kernels,
        }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
