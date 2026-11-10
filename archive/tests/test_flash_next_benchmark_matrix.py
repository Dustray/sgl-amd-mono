import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("flash_benchmark", Path(__file__).resolve().parents[2] / "tools/benchmark_flash_next_matrix.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class Response:
    def __init__(self, events=()):
        self.events = events

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_lines(self):
        for event in self.events:
            yield b"data: " + json.dumps(event).encode()
        yield b"data: [DONE]"


@pytest.mark.parametrize("cached", [0, 1])
def test_stream_metrics_handle_batched_mtp_chunks_and_reject_cache_hits(monkeypatch, cached):
    def event(index, tokens, finished=False):
        return {"index": index, "output_ids": list(range(tokens)), "meta_info": {"completion_tokens": tokens, "prompt_tokens": 2, "cached_tokens": cached, "num_retractions": 0, "forward_entry_time": 100, "prefill_finished_time": 100.25 + index * 0.1, "finish_reason": {"type": "length"} if finished else None}}

    events = [event(0, 1), event(1, 1), event(0, 4, True), event(1, 4, True)]
    times = iter([0.0, 0.4, 0.5, 1.0, 1.2, 1.21, 1.22])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    session = type("Session", (), {"post": lambda self, url, **kwargs: Response(events if url.endswith("/generate") else ())})()
    if cached:
        with pytest.raises(AssertionError):
            benchmark.stream_case(session, "http://test", [[10, 20], [30, 40]], 4)
    else:
        result = benchmark.stream_case(session, "http://test", [[10, 20], [30, 40]], 4)
        assert result["e2e_s"] == pytest.approx(1.22)
        assert result["ttft_s"] == pytest.approx(0.5)
        assert result["decode_s"] == pytest.approx(0.8)
        assert result["decode_tps"] == pytest.approx(6 / 0.8)
        assert result["prefill_s"] == pytest.approx(0.35)
        assert result["token_arrivals"][0] == [[1, 0.4], [4, 1.0]]
