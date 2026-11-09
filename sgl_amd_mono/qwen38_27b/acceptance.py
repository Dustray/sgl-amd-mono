import hashlib
from importlib import metadata, resources
import json
from pathlib import Path
import subprocess

from .report import implementation_digest


RECORD_PATH = Path(__file__).with_name("acceptance.json")


def source_identity():
    compatibility = Path(str(resources.files("sgl_amd_mono_0518")))
    digest = hashlib.sha256()
    for path in sorted(compatibility.iterdir()):
        if path.suffix in {".py", ".json"}:
            digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    compiler = subprocess.check_output(["/opt/dtk/bin/hipcc", "--version"], timeout=10)
    return {
        "implementation_sha256": implementation_digest(),
        "compatibility_sha256": digest.hexdigest(),
        "torch_version": metadata.version("torch"),
        "sglang_version": metadata.version("sglang"),
        "hip_compiler_sha256": hashlib.sha256(compiler).hexdigest(),
    }


def accepted_features():
    try:
        record = json.loads(RECORD_PATH.read_text())
        if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("passed") is not True or record.get("source_identity") != source_identity():
            return frozenset()
        features = frozenset(record.get("features", []))
        if "dense_eager" not in features or not features <= {"dense_eager", "dense_graph", "full_attention_eager", "full_attention_graph", "decode_model_eager", "decode_model_graph"}:
            return frozenset()
        if "full_attention_graph" in features and not {"dense_graph", "full_attention_eager"} <= features:
            return frozenset()
        if "decode_model_eager" in features and "full_attention_eager" not in features:
            return frozenset()
        if "decode_model_graph" in features and not {"decode_model_eager", "full_attention_graph"} <= features:
            return frozenset()
        return features
    except (OSError, ValueError, TypeError, subprocess.SubprocessError, metadata.PackageNotFoundError):
        return frozenset()
