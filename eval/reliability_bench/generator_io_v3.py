"""Byte-stable dataset writer for future ReliabilityBench-Q generators.

Historical v2.1 generation code and sealed artifacts are deliberately untouched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


GENERATOR_IO_VERSION = "reliabilitybench-q/generator-io-3.0"


def canonical_json_line(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def write_jsonl_bytes(
    records: Iterable[dict[str, Any]], output_path: Path
) -> dict[str, Any]:
    """Write exact UTF-8/LF bytes and hash the bytes that reached disk."""
    payload = b"".join(canonical_json_line(item) for item in records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    on_disk = output_path.read_bytes()
    if on_disk != payload:
        raise RuntimeError("dataset bytes changed during write")
    return {
        "generator_io_version": GENERATOR_IO_VERSION,
        "byte_count": len(on_disk),
        "sha256": hashlib.sha256(on_disk).hexdigest(),
        "newline": "LF",
        "encoding": "UTF-8",
    }


__all__ = ["GENERATOR_IO_VERSION", "canonical_json_line", "write_jsonl_bytes"]
