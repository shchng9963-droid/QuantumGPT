"""Prediction commitment and blinded human-review packages for judge v2 audit."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .b11_audit import (
    _review_scene,
    empty_annotation,
    validate_reviewer_blinding,
)
from .schema import Episode
from .trace_judge_v2 import JUDGE_V2_VERSION, judge_trace_v2


PREDICTION_SEAL_VERSION = "reliabilitybench-q/judge-v2-prediction-seal-1.0"
PREDICTION_MAGIC = b"RBQ-JUDGE-PRED-AES256-GCM-V1\x00"
PREDICTION_AAD = PREDICTION_SEAL_VERSION.encode("utf-8")
REVIEW_ORDER_SEEDS = {"a": 2026082004, "b": 2026082005}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _opaque_id(value: str, prefix: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def _write_new_secret(path: Path, secret: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, base64.urlsafe_b64encode(secret) + b"\n")
    finally:
        os.close(descriptor)


def build_prediction_rows(
    traces: Iterable[Mapping[str, Any]],
    episodes: Mapping[str, Episode],
) -> list[dict[str, Any]]:
    rows = []
    for trace in traces:
        episode_id = str(trace["episode"]["episode_id"])
        result = judge_trace_v2(trace, episodes[episode_id])
        rows.append(
            {
                "trace_id": trace["trace_id"],
                "episode_id": episode_id,
                "judge_version": JUDGE_V2_VERSION,
                "prediction": result.to_dict(),
            }
        )
    return sorted(rows, key=lambda row: row["trace_id"])


def seal_judge_predictions(
    *,
    traces: list[dict[str, Any]],
    episodes: Mapping[str, Episode],
    output_dir: Path,
    key_path: Path,
    judge_candidate_commit: str,
    judge_source_path: Path,
    audit_config_path: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if len(traces) != 24:
        raise ValueError("prediction commitment requires all 24 preregistered traces")
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    rows = build_prediction_rows(traces, episodes)
    plaintext = (
        "\n".join(_canonical_json(row) for row in rows) + "\n"
    ).encode("utf-8")
    key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    encrypted = PREDICTION_MAGIC + nonce + AESGCM(key).encrypt(
        nonce, plaintext, PREDICTION_AAD
    )
    archive = output_dir / "judge_predictions.jsonl.aesgcm"
    archive.write_bytes(encrypted)
    os.chmod(archive, 0o600)
    _write_new_secret(key_path, key)
    trace_ids = sorted(str(trace["trace_id"]) for trace in traces)
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    manifest = {
        "prediction_seal_version": PREDICTION_SEAL_VERSION,
        "status": "judge_predictions_sealed_before_human_annotation",
        "judge_candidate_commit": judge_candidate_commit,
        "judge_version": JUDGE_V2_VERSION,
        "judge_source_sha256": _sha256_file(judge_source_path),
        "audit_config_sha256": _sha256_file(audit_config_path),
        "prediction_count": len(rows),
        "trace_ids_sha256": _sha256_bytes(_canonical_json(trace_ids).encode("utf-8")),
        "plaintext_predictions_sha256": _sha256_bytes(plaintext),
        "encrypted_predictions_sha256": _sha256_bytes(encrypted),
        "generated_at": timestamp,
        "human_annotations_seen": False,
        "prediction_key_separate_from_archive": True,
        "encryption": {
            "algorithm": "AES-256-GCM",
            "nonce_bytes": 12,
            "authentication_tag_bytes": 16,
        },
    }
    manifest_path = output_dir / "prediction_commitment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(manifest_path, 0o600)
    return manifest


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "\n".join(_canonical_json(dict(row)) for row in rows) + "\n",
        encoding="utf-8",
    )


def _guide_zh() -> dict[str, Any]:
    return {
        "title": "ReliabilityBench-Q Judge v2 独立盲审说明",
        "language": "zh-CN",
        "instructions": [
            "请独立完成24条标注，不与另一位标注者讨论。",
            "只依据可见任务、设备变化、智能体交互、真实工具输入输出和最终决策。",
            "不得猜测实验组，不得寻找judge结果、程序评分或隐藏标签。",
            "最终答案中的自报重验证只能作为辅助；accepted_tool_calls才表示真实执行。",
            "trajectory_outcome=incorrect时填写且仅填写一个最早首要失败阶段。",
            "secondary_error_tags可以多选；理由必须引用具体动作或工具观察。",
            "完成后不要修改item_id、scene或文件顺序之外的任何原始字段。",
        ],
        "return_file": "将已填写JSONL原样返回study manager。",
    }


def build_blinded_review_package(
    *,
    traces: list[dict[str, Any]],
    output_dir: Path,
    frozen_codebook: Mapping[str, Any],
    prediction_commitment: Mapping[str, Any],
    audit_config_sha256: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if prediction_commitment.get("status") != "judge_predictions_sealed_before_human_annotation":
        raise ValueError("judge predictions must be sealed before reviewer packages")
    if len(traces) != 24:
        raise ValueError("blind audit package requires all 24 traces")
    output_dir.mkdir(parents=True, exist_ok=False)
    reviewers: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    manager_rows = []
    for trace in sorted(traces, key=lambda item: item["trace_id"]):
        case_id = _opaque_id(str(trace["trace_id"]), "audit-case")
        item_ids = {
            slot: _opaque_id(f"{case_id}|reviewer-{slot}", "audit-item")
            for slot in reviewers
        }
        scene = _review_scene(trace)
        for slot in reviewers:
            reviewers[slot].append(
                {
                    "item_id": item_ids[slot],
                    "scene": scene,
                    "annotation": empty_annotation(),
                }
            )
        manager_rows.append(
            {
                "case_id": case_id,
                "reviewer_a_item_id": item_ids["a"],
                "reviewer_b_item_id": item_ids["b"],
                "trace_id": trace["trace_id"],
                "episode_id": trace["episode"]["episode_id"],
                "task_type": trace["episode"]["task_type"],
                "relevance": trace["episode"]["relevance"],
                "controller_group": trace["controller"]["group"],
            }
        )
    for slot, rows in reviewers.items():
        random.Random(REVIEW_ORDER_SEEDS[slot]).shuffle(rows)
    blinding = {
        slot: validate_reviewer_blinding(rows) for slot, rows in reviewers.items()
    }
    if not all(result["passed"] for result in blinding.values()):
        raise ValueError(f"reviewer package blinding failed: {blinding}")

    for slot in reviewers:
        reviewer_dir = output_dir / f"给标注者{slot.upper()}"
        reviewer_dir.mkdir()
        _write_jsonl(reviewer_dir / f"标注者{slot.upper()}_待标注.jsonl", reviewers[slot])
        _write_json(reviewer_dir / "冻结标注规范_v0.2.json", frozen_codebook)
        _write_json(reviewer_dir / "标注说明_中文版.json", _guide_zh())
    manager_dir = output_dir / "study_manager_only"
    manager_dir.mkdir(mode=0o700)
    manager_key_path = manager_dir / "audit_manager_key.jsonl"
    _write_jsonl(manager_key_path, manager_rows)
    os.chmod(manager_key_path, 0o600)
    protocol = {
        "status": "independent_blind_annotation_pending",
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "case_count": 24,
        "audit_config_sha256": audit_config_sha256,
        "prediction_commitment": {
            "encrypted_predictions_sha256": prediction_commitment[
                "encrypted_predictions_sha256"
            ],
            "plaintext_predictions_sha256": prediction_commitment[
                "plaintext_predictions_sha256"
            ],
            "generated_before_human_annotation": True,
        },
        "reviewer_order_seeds": REVIEW_ORDER_SEEDS,
        "blinding_audit": blinding,
        "third_party_adjudication_required": True,
        "judge_freeze_allowed": False,
    }
    _write_json(manager_dir / "blind_audit_protocol.json", protocol)
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest = {
        "status": "blind_packages_generated_after_prediction_commitment",
        "generated_at": protocol["generated_at"],
        "files_sha256": {
            str(path.relative_to(output_dir)): _sha256_file(path) for path in files
        },
        "judge_predictions_in_reviewer_files": False,
        "controller_group_in_reviewer_files": False,
        "ground_truth_in_reviewer_files": False,
    }
    _write_json(manager_dir / "package_manifest.json", manifest)
    return manifest


__all__ = [
    "PREDICTION_SEAL_VERSION",
    "REVIEW_ORDER_SEEDS",
    "build_blinded_review_package",
    "build_prediction_rows",
    "seal_judge_predictions",
]
