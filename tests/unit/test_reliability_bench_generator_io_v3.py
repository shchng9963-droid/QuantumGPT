from __future__ import annotations

import hashlib

from eval.reliability_bench.generator_io_v3 import write_jsonl_bytes


def test_future_generator_hashes_exact_lf_bytes(tmp_path):
    output = tmp_path / "future.jsonl"
    summary = write_jsonl_bytes([{"中文": "值", "b": 2}, {"a": 1}], output)
    payload = output.read_bytes()
    assert b"\r\n" not in payload
    assert payload.endswith(b"\n")
    assert summary["sha256"] == hashlib.sha256(payload).hexdigest()
    assert len(summary["sha256"]) == 64
