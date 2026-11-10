import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--native-code-object", type=Path, required=True)
    parser.add_argument("--expected-native-sha256", required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    binary = bytearray(arguments.native_code_object.read_bytes())
    digest = hashlib.sha256(binary).hexdigest()
    if digest != arguments.expected_native_sha256:
        raise ValueError("Native diagnostic code object SHA256 mismatch")
    position = 0x28919C - 0x1000
    if binary[position:position + 8].hex() != "01008cd2c1000100":
        raise ValueError("Unexpected native QK diagnostic patch site")
    directory = arguments.output.parent
    directory.mkdir(parents=True, exist_ok=True)
    assembly = Path(__file__).with_suffix(".s")
    object_path = arguments.output.with_suffix(".o")
    patch_path = arguments.output.with_suffix(".bin")
    subprocess.run(["/opt/dtk/aillvm/bin/llvm-mc", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx936", "-filetype=obj", str(assembly), "-o", str(object_path)], check=True)
    subprocess.run(["/opt/dtk/aillvm/bin/llvm-objcopy", f"--dump-section=.text={patch_path}", str(object_path)], check=True)
    patch = patch_path.read_bytes()
    binary[position:position + len(patch)] = patch
    patched = arguments.output.with_suffix(".hsaco")
    patched.write_bytes(binary)
    captured = torch.load(arguments.capture, map_location="cuda", weights_only=True)
    length = int(captured["lengths"][0])
    if captured["lengths"].numel() != 1 or not 2560 < length <= 3072:
        raise ValueError("This native ISA diagnostic requires one row and six 512-token partitions")
    query = captured["stages"]["query"].contiguous()
    key = captured["key_cache"].permute(0, 2, 1, 3).contiguous()
    value = captured["value_cache"].permute(0, 2, 3, 1).contiguous()
    output = torch.zeros((1, 24, 256), device="cuda", dtype=torch.bfloat16)
    workspace = torch.zeros(4 * 6 * 6 * 512, device="cuda", dtype=torch.float16)
    runtime = ctypes.CDLL("libamdhip64.so")
    pointer = ctypes.c_void_p
    integer = ctypes.c_int
    runtime.hipModuleLoad.argtypes = [ctypes.POINTER(pointer), ctypes.c_char_p]
    runtime.hipModuleGetFunction.argtypes = [ctypes.POINTER(pointer), pointer, ctypes.c_char_p]
    runtime.hipModuleLaunchKernel.argtypes = [pointer] + [ctypes.c_uint] * 7 + [pointer, ctypes.POINTER(pointer), pointer]
    runtime.hipModuleUnload.argtypes = [pointer]
    module, function = pointer(), pointer()
    if runtime.hipModuleLoad(ctypes.byref(module), str(patched.resolve()).encode()):
        raise RuntimeError("Loading native attention diagnostic code object failed")
    try:
        name = b"_Z22paged_attention_kernelIthLb0ELi256ELi64ELi256ELi8EEvPT_S1_PKS0_PKT0_S6_iiPKiS8_iPKfiiiiS3_ibiii"
        if runtime.hipModuleGetFunction(ctypes.byref(function), module, name):
            raise RuntimeError("Looking up native attention diagnostic kernel failed")
        values = [pointer(output.data_ptr()), pointer(workspace.data_ptr()), pointer(query.data_ptr()), pointer(key.data_ptr()), pointer(value.data_ptr()), integer(24), integer(4), pointer(captured["page_table"].data_ptr()), pointer(captured["lengths"].data_ptr()), integer(captured["page_table"].shape[1]), pointer(0), integer(6144), integer(65536), integer(6), integer(512), pointer(0), integer(1), ctypes.c_bool(False), integer(-1), integer(0), integer(-1)]
        parameters = (pointer * len(values))(*[ctypes.cast(ctypes.byref(value), pointer) for value in values])
        status = runtime.hipModuleLaunchKernel(function, 4, 1, 6, 256, 1, 1, 6304, torch.cuda.current_stream().cuda_stream, parameters, None)
        if status:
            raise RuntimeError(f"Native attention diagnostic launch failed: {status}")
        torch.cuda.synchronize()
        scores = workspace.view(4, 6, 6, 512).permute(0, 2, 1, 3).reshape(24, 3072)[:, :length].float() * 8
        probe = torch.load(arguments.scores, map_location="cuda", weights_only=True)
        report = {"scope": "diagnostic copy of native code object after QK: replace subsequent math with LDS score dump; installed library unchanged", "original_sha256": digest, "assembly_sha256": hashlib.sha256(assembly.read_bytes()).hexdigest(), "patch_offset": position, "patch_bytes": len(patch), "differences": []}
        for order in range(probe["scores"].shape[0]):
            actual = probe["scores"][order]
            for head, token in (scores != actual).nonzero().tolist():
                report["differences"].append({"order": order, "head": head, "token": token, "native": float(scores[head, token]), "actual": float(actual[head, token])})
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        torch.save(scores.cpu(), arguments.output.with_suffix(".pt"))
        print(json.dumps(report), flush=True)
    finally:
        runtime.hipModuleUnload(module)


if __name__ == "__main__":
    main()
