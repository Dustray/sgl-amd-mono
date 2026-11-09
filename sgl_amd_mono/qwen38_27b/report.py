"""Atomic JSON reports shared by the model runtime and its legacy hook bridge."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def implementation_digest():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.suffix in {".py", ".cpp"}:
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def write_report(directory, report, name=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (name or f"process-{os.getpid()}.json")
    descriptor, temporary = tempfile.mkstemp(prefix=".mono-", suffix=".json", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target
