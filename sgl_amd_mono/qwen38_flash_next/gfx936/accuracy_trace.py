import functools
import hashlib
import json
import os
from pathlib import Path

import torch


class AccuracyTrace:
    def __init__(self, runner):
        self.rank = runner.ps.tp_rank
        self.frames = []
        self.directory = Path(os.environ["SGLANG_AMD_FLASH_ACCURACY_TRACE_DIR"]).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index = self.directory / "index.json"
        if self.rank == 0 and self.index.exists():
            raise RuntimeError(f"Accuracy trace already exists: {self.index}")
        original = runner.model.forward

        @functools.wraps(original)
        def forward(*args, **kwargs):
            result = original(*args, **kwargs)
            if self.rank != 0:
                return result
            logits = getattr(result, "next_token_logits", None)
            if not isinstance(logits, torch.Tensor) or not logits.numel():
                return result
            batch = kwargs.get("forward_batch")
            if batch is None and len(args) > 2:
                batch = args[2]
            frame = {"step": len(self.frames), "mode": batch.forward_mode.name, "input_ids": batch.input_ids.cpu().tolist(), "seq_lens": batch.seq_lens.cpu().tolist(), "rows": logits.shape[0], "filename": f"frame-{len(self.frames):06d}.pt"}
            torch.save({"logits": logits.detach().float().cpu(), "input_ids": frame["input_ids"], "seq_lens": frame["seq_lens"], "mode": frame["mode"]}, self.directory / frame["filename"])
            self.frames.append(frame)
            temporary = self.index.with_suffix(".tmp")
            temporary.write_text(json.dumps({"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "frames": self.frames}, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(self.index)
            return result

        runner.model.forward = forward

    def snapshot(self):
        return {"rank": self.rank, "directory": str(self.directory), "completed_frames": len(self.frames), "scope": "actual outer model forwards only; shadow/reference replays excluded"}
