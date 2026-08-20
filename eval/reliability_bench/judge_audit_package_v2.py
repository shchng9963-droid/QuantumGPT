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
COORDINATION_VERSION = "reliabilitybench-q/judge-v2-coordination-stage1-1.0"
COORDINATION_ORDER_SEED = 2026082021
ADJUDICATION_VERSION = "reliabilitybench-q/judge-v2-coordination-stage2-1.0"
ADJUDICATION_ORDER_SEED = 2026082022


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


def _coordination_guide_zh() -> dict[str, Any]:
    return {
        "title": "第三标注者阶段1：独立盲标说明",
        "language": "zh-CN",
        "phase": "stage1_independent",
        "instructions": [
            "请只依据场景、轨迹和冻结codebook独立标注，不向研究管理员索取A/B意见。",
            "本阶段看不到A/B标签、judge预测、控制器组名、Oracle字段或程序标签。",
            "填写最终correct/incorrect、唯一首要失败阶段、次级标签、理由和置信度。",
            "若结果为correct但轨迹出现疑似实质错误标签，请独立判断是否存在冲突并说明。",
            "若定义无法区分中间过期引用与最终依赖过期引用，请标记codebook系统性歧义。",
            "提交前不要修改coordination_item_id、phase、scene或文件顺序。",
        ],
        "stage2_release_condition": (
            "研究管理员收到并冻结本阶段返回文件及SHA-256后，才会提供A/B意见用于最终裁决。"
        ),
    }


def _empty_stage1_coordination_annotation() -> dict[str, Any]:
    return {
        "annotator_id": "C",
        "trajectory_outcome": None,
        "primary_failure_stage": None,
        "secondary_error_tags": [],
        "correct_with_substantive_error_tag_conflict": None,
        "conflict_assessment": "",
        "rationale": "",
        "confidence": None,
        "codebook_systematic_ambiguity": None,
        "codebook_ambiguity_description": "",
    }


def _annotation_signature(annotation: Mapping[str, Any]) -> tuple[Any, Any, frozenset[str]]:
    return (
        annotation["trajectory_outcome"],
        annotation["primary_failure_stage"],
        frozenset(annotation["secondary_error_tags"]),
    )


def build_stage1_coordination_package(
    *,
    reviewer_a: list[dict[str, Any]],
    reviewer_b: list[dict[str, Any]],
    original_a: list[dict[str, Any]],
    original_b: list[dict[str, Any]],
    manager_key: list[dict[str, Any]],
    output_dir: Path,
    frozen_codebook: Mapping[str, Any],
    reviewer_a_source_sha256: str,
    reviewer_b_source_sha256: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build only the independent first stage; A/B opinions remain manager-only."""
    by_a = {row["item_id"]: row for row in reviewer_a}
    by_b = {row["item_id"]: row for row in reviewer_b}
    original_a_by_id = {row["item_id"]: row for row in original_a}
    original_b_by_id = {row["item_id"]: row for row in original_b}
    if set(by_a) != set(original_a_by_id) or set(by_b) != set(original_b_by_id):
        raise ValueError("returned reviewer item IDs do not match frozen blind packages")
    for item_id, row in by_a.items():
        if row.get("scene") != original_a_by_id[item_id].get("scene"):
            raise ValueError("reviewer A scene differs from the frozen blind package")
    for item_id, row in by_b.items():
        if row.get("scene") != original_b_by_id[item_id].get("scene"):
            raise ValueError("reviewer B scene differs from the frozen blind package")

    stage1_rows: list[dict[str, Any]] = []
    manager_rows: list[dict[str, Any]] = []
    for manager in manager_key:
        item_a = manager["reviewer_a_item_id"]
        item_b = manager["reviewer_b_item_id"]
        annotation_a = by_a[item_a]["annotation"]
        annotation_b = by_b[item_b]["annotation"]
        if _annotation_signature(annotation_a) == _annotation_signature(annotation_b):
            continue
        coordination_item_id = _opaque_id(
            f"{COORDINATION_ORDER_SEED}|{manager['case_id']}|stage1",
            "coord-item",
        )
        stage1_rows.append(
            {
                "coordination_item_id": coordination_item_id,
                "phase": "stage1_independent",
                "scene": original_a_by_id[item_a]["scene"],
                "independent_annotation": _empty_stage1_coordination_annotation(),
            }
        )
        manager_rows.append(
            {
                "coordination_item_id": coordination_item_id,
                "case_id": manager["case_id"],
                "reviewer_a_item_id": item_a,
                "reviewer_b_item_id": item_b,
                "outcome_disagreement": (
                    annotation_a["trajectory_outcome"]
                    != annotation_b["trajectory_outcome"]
                ),
                "primary_stage_disagreement": (
                    annotation_a["primary_failure_stage"]
                    != annotation_b["primary_failure_stage"]
                ),
                "secondary_tags_disagreement": (
                    set(annotation_a["secondary_error_tags"])
                    != set(annotation_b["secondary_error_tags"])
                ),
            }
        )
    if len(stage1_rows) != 17:
        raise ValueError(f"expected 17 preregistered disagreements, got {len(stage1_rows)}")
    random.Random(COORDINATION_ORDER_SEED).shuffle(stage1_rows)

    serialized = _canonical_json(stage1_rows).lower()
    forbidden = (
        "controller_group",
        "hidden_group_id",
        "program_judge",
        "judge_predictions",
        "ground_truth",
        "reviewer_a_annotation",
        "reviewer_b_annotation",
        "full_selective",
        "oracle_invalidation",
    )
    violations = [token for token in forbidden if token in serialized]
    if violations:
        raise ValueError(f"stage1 coordination blinding failed: {violations}")

    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    stage1_dir = output_dir / "stage1_for_third_annotator"
    manager_dir = output_dir / "study_manager_only"
    stage1_dir.mkdir(mode=0o700)
    manager_dir.mkdir(mode=0o700)
    stage1_path = stage1_dir / "第三标注者_阶段1独立标注_待填写.jsonl"
    guide_path = stage1_dir / "第三标注者_阶段1说明.json"
    codebook_path = stage1_dir / "冻结标注规范_v0.2.json"
    _write_jsonl(stage1_path, stage1_rows)
    _write_json(guide_path, _coordination_guide_zh())
    _write_json(codebook_path, frozen_codebook)
    for path in (stage1_path, guide_path, codebook_path):
        os.chmod(path, 0o600)
    stage1_file_hashes = {
        path.name: _sha256_file(path)
        for path in (stage1_path, guide_path, codebook_path)
    }
    stage1_manifest = {
        "coordination_version": COORDINATION_VERSION,
        "status": "stage1_independent_annotation_pending",
        "item_count": len(stage1_rows),
        "order_seed": COORDINATION_ORDER_SEED,
        "files_sha256": stage1_file_hashes,
        "a_b_opinions_in_stage1": False,
        "judge_predictions_accessed": False,
        "method_identity_fields_present": False,
        "stage2_release_condition": "freeze returned stage1 file and SHA-256 first",
    }
    stage1_manifest_path = stage1_dir / "阶段1文件校验_SHA256.json"
    _write_json(stage1_manifest_path, stage1_manifest)
    os.chmod(stage1_manifest_path, 0o600)

    manager_key_path = manager_dir / "coordination_manager_key.jsonl"
    _write_jsonl(manager_key_path, manager_rows)
    os.chmod(manager_key_path, 0o600)
    manager_manifest = {
        "coordination_version": COORDINATION_VERSION,
        "status": "stage1_generated_stage2_not_generated",
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "disagreement_count": len(stage1_rows),
        "outcome_disagreement_count": sum(
            row["outcome_disagreement"] for row in manager_rows
        ),
        "primary_stage_disagreement_count": sum(
            row["primary_stage_disagreement"] for row in manager_rows
        ),
        "secondary_tags_disagreement_count": sum(
            row["secondary_tags_disagreement"] for row in manager_rows
        ),
        "reviewer_a_source_sha256": reviewer_a_source_sha256,
        "reviewer_b_source_sha256": reviewer_b_source_sha256,
        "stage1_manifest_sha256": _sha256_file(stage1_manifest_path),
        "manager_key_sha256": _sha256_file(manager_key_path),
        "judge_predictions_accessed": False,
        "stage2_generated": False,
    }
    manager_manifest_path = manager_dir / "coordination_protocol_manifest.json"
    _write_json(manager_manifest_path, manager_manifest)
    os.chmod(manager_manifest_path, 0o600)
    return manager_manifest


def validate_stage1_coordination_return(
    *,
    original_rows: list[dict[str, Any]],
    returned_rows: list[dict[str, Any]],
    frozen_codebook: Mapping[str, Any],
) -> dict[str, Any]:
    original_by_id = {row["coordination_item_id"]: row for row in original_rows}
    returned_by_id = {row.get("coordination_item_id"): row for row in returned_rows}
    errors: list[dict[str, Any]] = []
    expected_annotation_fields = {
        "annotator_id",
        "trajectory_outcome",
        "primary_failure_stage",
        "secondary_error_tags",
        "correct_with_substantive_error_tag_conflict",
        "conflict_assessment",
        "rationale",
        "confidence",
        "codebook_systematic_ambiguity",
        "codebook_ambiguity_description",
    }
    allowed_outcomes = set(frozen_codebook["outcomes"])
    allowed_stages = set(frozen_codebook["primary_failure_stages"])
    allowed_tags = set(frozen_codebook["secondary_error_tags"])
    if len(returned_rows) != 17 or len(returned_by_id) != 17:
        errors.append({"error": "stage1 return must contain 17 unique rows"})
    if set(original_by_id) != set(returned_by_id):
        errors.append({"error": "stage1 returned item IDs differ from frozen package"})
    if [row["coordination_item_id"] for row in original_rows] != [
        row.get("coordination_item_id") for row in returned_rows
    ]:
        errors.append({"error": "stage1 returned order differs from frozen package"})
    for index, row in enumerate(returned_rows, start=1):
        item_id = row.get("coordination_item_id")
        original = original_by_id.get(item_id)
        if original is None:
            continue
        if set(row) != {
            "coordination_item_id",
            "phase",
            "scene",
            "independent_annotation",
        }:
            errors.append({"row": index, "error": "unexpected top-level fields"})
        if row.get("phase") != original.get("phase"):
            errors.append({"row": index, "error": "phase changed"})
        if row.get("scene") != original.get("scene"):
            errors.append({"row": index, "error": "scene changed"})
        annotation = row.get("independent_annotation") or {}
        if set(annotation) != expected_annotation_fields:
            errors.append({"row": index, "error": "annotation field set changed"})
            continue
        outcome = annotation["trajectory_outcome"]
        stage = annotation["primary_failure_stage"]
        tags = annotation["secondary_error_tags"]
        confidence = annotation["confidence"]
        conflict = annotation["correct_with_substantive_error_tag_conflict"]
        ambiguity = annotation["codebook_systematic_ambiguity"]
        if annotation["annotator_id"] != "C":
            errors.append({"row": index, "error": "annotator_id must be C"})
        if outcome not in allowed_outcomes:
            errors.append({"row": index, "error": "invalid outcome"})
        if outcome == "correct" and stage is not None:
            errors.append({"row": index, "error": "correct outcome requires null stage"})
        if outcome == "incorrect" and stage not in allowed_stages:
            errors.append({"row": index, "error": "incorrect outcome requires one stage"})
        if (
            not isinstance(tags, list)
            or len(tags) != len(set(tags))
            or not set(tags).issubset(allowed_tags)
        ):
            errors.append({"row": index, "error": "invalid secondary tags"})
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 1
        ):
            errors.append({"row": index, "error": "invalid confidence"})
        if not isinstance(conflict, bool):
            errors.append({"row": index, "error": "conflict flag must be boolean"})
        elif conflict and not str(annotation["conflict_assessment"]).strip():
            errors.append({"row": index, "error": "conflict requires assessment"})
        if not isinstance(ambiguity, bool):
            errors.append({"row": index, "error": "ambiguity flag must be boolean"})
        elif ambiguity and not str(annotation["codebook_ambiguity_description"]).strip():
            errors.append({"row": index, "error": "ambiguity requires description"})
        if not str(annotation["rationale"]).strip():
            errors.append({"row": index, "error": "rationale is required"})
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "row_count": len(returned_rows),
        "error_count": len(errors),
        "errors": errors,
    }


def _empty_final_adjudication() -> dict[str, Any]:
    return {
        "annotator_id": "C",
        "trajectory_outcome": None,
        "primary_failure_stage": None,
        "secondary_error_tags": [],
        "correct_with_substantive_error_tag_conflict": None,
        "conflict_assessment": "",
        "adjudication_rationale": "",
        "confidence": None,
        "codebook_systematic_ambiguity": None,
        "codebook_ambiguity_description": "",
    }


def _stage2_guide_zh() -> dict[str, Any]:
    return {
        "title": "第三标注者阶段2：匿名意见协调与最终裁决",
        "language": "zh-CN",
        "phase": "stage2_final_adjudication",
        "instructions": [
            "阶段1独立意见已经冻结；现在可比较自己的独立意见与两份匿名先前意见。",
            "review_1和review_2每条均重新匿名排序，不代表固定标注者身份。",
            "请形成最终标签、唯一首要失败阶段、次级标签、冲突判断、理由和置信度。",
            "若发现系统性codebook歧义，必须明确标记并说明，不能靠逐条裁决掩盖。",
            "不得修改adjudication_item_id、phase、scene、stage1_independent_annotation或prior_reviews。",
        ],
        "post_submission_condition": (
            "研究管理员冻结最终共识文件及SHA-256后，才可解封预先密封的judge预测。"
        ),
    }


def build_stage2_adjudication_package(
    *,
    stage1_original: list[dict[str, Any]],
    stage1_returned: list[dict[str, Any]],
    coordination_manager_key: list[dict[str, Any]],
    reviewer_a: list[dict[str, Any]],
    reviewer_b: list[dict[str, Any]],
    output_dir: Path,
    frozen_codebook: Mapping[str, Any],
    frozen_stage1_return_sha256: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    validation = validate_stage1_coordination_return(
        original_rows=stage1_original,
        returned_rows=stage1_returned,
        frozen_codebook=frozen_codebook,
    )
    if not validation["passed"]:
        raise ValueError(f"stage1 return failed validation: {validation}")
    returned_by_coord = {
        row["coordination_item_id"]: row for row in stage1_returned
    }
    reviewer_a_by_id = {row["item_id"]: row for row in reviewer_a}
    reviewer_b_by_id = {row["item_id"]: row for row in reviewer_b}
    stage2_rows: list[dict[str, Any]] = []
    manager_rows: list[dict[str, Any]] = []
    for manager in coordination_manager_key:
        coord_id = manager["coordination_item_id"]
        stage1_row = returned_by_coord[coord_id]
        reviews = [
            dict(reviewer_a_by_id[manager["reviewer_a_item_id"]]["annotation"]),
            dict(reviewer_b_by_id[manager["reviewer_b_item_id"]]["annotation"]),
        ]
        for review in reviews:
            review.pop("annotator_id", None)
        random.Random(
            int(hashlib.sha256(f"{ADJUDICATION_ORDER_SEED}|{coord_id}".encode()).hexdigest()[:16], 16)
        ).shuffle(reviews)
        anonymized_reviews = [
            {"review_id": f"review_{index}", "annotation": review}
            for index, review in enumerate(reviews, start=1)
        ]
        adjudication_item_id = _opaque_id(
            f"{ADJUDICATION_ORDER_SEED}|{coord_id}|stage2", "adjud-item"
        )
        stage2_rows.append(
            {
                "adjudication_item_id": adjudication_item_id,
                "phase": "stage2_final_adjudication",
                "scene": stage1_row["scene"],
                "stage1_independent_annotation": stage1_row[
                    "independent_annotation"
                ],
                "prior_reviews": anonymized_reviews,
                "final_adjudication": _empty_final_adjudication(),
            }
        )
        manager_rows.append(
            {
                "adjudication_item_id": adjudication_item_id,
                "coordination_item_id": coord_id,
                "case_id": manager["case_id"],
            }
        )
    random.Random(ADJUDICATION_ORDER_SEED).shuffle(stage2_rows)
    serialized = _canonical_json(stage2_rows).lower()
    forbidden = (
        "controller_group",
        "hidden_group_id",
        "program_judge",
        "judge_predictions",
        "ground_truth",
        "reviewer_a_item_id",
        "reviewer_b_item_id",
        "full_selective",
        "oracle_invalidation",
    )
    violations = [token for token in forbidden if token in serialized]
    if violations:
        raise ValueError(f"stage2 adjudication blinding failed: {violations}")

    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    public_dir = output_dir / "stage2_for_third_annotator"
    manager_dir = output_dir / "study_manager_only"
    public_dir.mkdir(mode=0o700)
    manager_dir.mkdir(mode=0o700)
    rows_path = public_dir / "第三标注者_阶段2最终裁决_待填写.jsonl"
    guide_path = public_dir / "第三标注者_阶段2说明.json"
    codebook_path = public_dir / "冻结标注规范_v0.2.json"
    _write_jsonl(rows_path, stage2_rows)
    _write_json(guide_path, _stage2_guide_zh())
    _write_json(codebook_path, frozen_codebook)
    for path in (rows_path, guide_path, codebook_path):
        os.chmod(path, 0o600)
    public_manifest = {
        "adjudication_version": ADJUDICATION_VERSION,
        "status": "stage2_final_adjudication_pending",
        "item_count": len(stage2_rows),
        "order_seed": ADJUDICATION_ORDER_SEED,
        "frozen_stage1_return_sha256": frozen_stage1_return_sha256,
        "files_sha256": {
            path.name: _sha256_file(path)
            for path in (rows_path, guide_path, codebook_path)
        },
        "judge_predictions_accessed": False,
        "reviewer_identity_exposed": False,
    }
    public_manifest_path = public_dir / "阶段2文件校验_SHA256.json"
    _write_json(public_manifest_path, public_manifest)
    os.chmod(public_manifest_path, 0o600)
    manager_key_path = manager_dir / "adjudication_manager_key.jsonl"
    _write_jsonl(manager_key_path, manager_rows)
    os.chmod(manager_key_path, 0o600)
    manager_manifest = {
        "adjudication_version": ADJUDICATION_VERSION,
        "status": "stage2_generated_final_consensus_pending",
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "item_count": len(stage2_rows),
        "frozen_stage1_return_sha256": frozen_stage1_return_sha256,
        "public_manifest_sha256": _sha256_file(public_manifest_path),
        "manager_key_sha256": _sha256_file(manager_key_path),
        "judge_predictions_accessed": False,
        "judge_unseal_allowed": False,
    }
    manager_manifest_path = manager_dir / "adjudication_protocol_manifest.json"
    _write_json(manager_manifest_path, manager_manifest)
    os.chmod(manager_manifest_path, 0o600)
    return manager_manifest


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
    "ADJUDICATION_ORDER_SEED",
    "ADJUDICATION_VERSION",
    "COORDINATION_ORDER_SEED",
    "COORDINATION_VERSION",
    "PREDICTION_SEAL_VERSION",
    "REVIEW_ORDER_SEEDS",
    "build_blinded_review_package",
    "build_prediction_rows",
    "build_stage1_coordination_package",
    "build_stage2_adjudication_package",
    "seal_judge_predictions",
    "validate_stage1_coordination_return",
]
