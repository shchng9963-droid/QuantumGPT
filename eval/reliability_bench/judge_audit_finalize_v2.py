"""Freeze human consensus and score sealed judge-v2 predictions."""

from __future__ import annotations

import base64
import hashlib
import json
import random
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .judge_audit_package_v2 import PREDICTION_AAD, PREDICTION_MAGIC


FINALIZATION_VERSION = "reliabilitybench-q/judge-v2-finalization-1.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _annotation_labels(annotation: Mapping[str, Any]) -> tuple[str, Any, frozenset[str]]:
    return (
        str(annotation["trajectory_outcome"]),
        annotation["primary_failure_stage"],
        frozenset(annotation["secondary_error_tags"]),
    )


def validate_stage2_return(
    *,
    original_rows: list[dict[str, Any]],
    returned_rows: list[dict[str, Any]],
    frozen_codebook: Mapping[str, Any],
) -> dict[str, Any]:
    original_by_id = {row["adjudication_item_id"]: row for row in original_rows}
    returned_by_id = {row.get("adjudication_item_id"): row for row in returned_rows}
    allowed_outcomes = set(frozen_codebook["outcomes"])
    allowed_stages = set(frozen_codebook["primary_failure_stages"])
    allowed_tags = set(frozen_codebook["secondary_error_tags"])
    expected_fields = {
        "annotator_id",
        "trajectory_outcome",
        "primary_failure_stage",
        "secondary_error_tags",
        "correct_with_substantive_error_tag_conflict",
        "conflict_assessment",
        "adjudication_rationale",
        "confidence",
        "codebook_systematic_ambiguity",
        "codebook_ambiguity_description",
    }
    errors: list[dict[str, Any]] = []
    if len(returned_rows) != 17 or len(returned_by_id) != 17:
        errors.append({"error": "stage2 return must contain 17 unique rows"})
    if set(original_by_id) != set(returned_by_id):
        errors.append({"error": "stage2 returned IDs differ from frozen package"})
    if [row["adjudication_item_id"] for row in original_rows] != [
        row.get("adjudication_item_id") for row in returned_rows
    ]:
        errors.append({"error": "stage2 returned order differs from frozen package"})
    for index, row in enumerate(returned_rows, start=1):
        original = original_by_id.get(row.get("adjudication_item_id"))
        if original is None:
            continue
        for field in ("phase", "scene", "stage1_independent_annotation", "prior_reviews"):
            if row.get(field) != original.get(field):
                errors.append({"row": index, "error": f"{field} changed"})
        annotation = row.get("final_adjudication") or {}
        if set(annotation) != expected_fields:
            errors.append({"row": index, "error": "final field set changed"})
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
        if not str(annotation["adjudication_rationale"]).strip():
            errors.append({"row": index, "error": "adjudication rationale is required"})
    return {
        "passed": not errors,
        "status": "passed" if not errors else "failed",
        "row_count": len(returned_rows),
        "error_count": len(errors),
        "errors": errors,
    }


def build_human_consensus(
    *,
    stage2_returned: list[dict[str, Any]],
    adjudication_manager_key: list[dict[str, Any]],
    coordination_manager_key: list[dict[str, Any]],
    audit_manager_key: list[dict[str, Any]],
    reviewer_a: list[dict[str, Any]],
    reviewer_b: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    adjudication_by_id = {
        row["adjudication_item_id"]: row for row in stage2_returned
    }
    adjudication_key_by_case = {
        row["case_id"]: row for row in adjudication_manager_key
    }
    coordination_by_case = {row["case_id"]: row for row in coordination_manager_key}
    reviewer_a_by_id = {row["item_id"]: row for row in reviewer_a}
    reviewer_b_by_id = {row["item_id"]: row for row in reviewer_b}
    rows: list[dict[str, Any]] = []
    for audit in audit_manager_key:
        case_id = audit["case_id"]
        if case_id in adjudication_key_by_case:
            key = adjudication_key_by_case[case_id]
            annotation = dict(
                adjudication_by_id[key["adjudication_item_id"]]["final_adjudication"]
            )
            source = "stage2_independent_adjudication"
        else:
            coordination = coordination_by_case.get(case_id)
            if coordination is not None:
                raise ValueError("a disagreement case is missing final adjudication")
            annotation_a = reviewer_a_by_id[audit["reviewer_a_item_id"]]["annotation"]
            annotation_b = reviewer_b_by_id[audit["reviewer_b_item_id"]]["annotation"]
            if _annotation_labels(annotation_a) != _annotation_labels(annotation_b):
                raise ValueError("unadjudicated A/B label disagreement")
            annotation = dict(annotation_a)
            source = "independent_a_b_exact_agreement"
        rows.append(
            {
                "consensus_id": "consensus-"
                + hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:20],
                "case_id": case_id,
                "trace_id": audit["trace_id"],
                "episode_id": audit["episode_id"],
                "trajectory_outcome": annotation["trajectory_outcome"],
                "primary_failure_stage": annotation["primary_failure_stage"],
                "secondary_error_tags": sorted(annotation["secondary_error_tags"]),
                "consensus_source": source,
                "codebook_systematic_ambiguity": bool(
                    annotation.get("codebook_systematic_ambiguity", False)
                ),
            }
        )
    rows.sort(key=lambda row: row["trace_id"])
    if len(rows) != 24 or len({row["trace_id"] for row in rows}) != 24:
        raise ValueError("human consensus must contain 24 unique traces")
    return rows


def decrypt_prediction_rows(
    *, archive: bytes, key_encoded: bytes, commitment: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], bytes]:
    if hashlib.sha256(archive).hexdigest() != commitment["encrypted_predictions_sha256"]:
        raise ValueError("encrypted prediction hash mismatch")
    if not archive.startswith(PREDICTION_MAGIC):
        raise ValueError("prediction archive magic mismatch")
    key = base64.urlsafe_b64decode(key_encoded.strip())
    if len(key) != 32:
        raise ValueError("prediction key is not 256 bits")
    offset = len(PREDICTION_MAGIC)
    nonce = archive[offset : offset + 12]
    plaintext = AESGCM(key).decrypt(nonce, archive[offset + 12 :], PREDICTION_AAD)
    if hashlib.sha256(plaintext).hexdigest() != commitment["plaintext_predictions_sha256"]:
        raise ValueError("plaintext prediction hash mismatch")
    rows = [json.loads(line) for line in plaintext.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 24:
        raise ValueError("prediction archive must contain 24 rows")
    return rows, plaintext


def _kappa(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    counts_left = Counter(left)
    counts_right = Counter(right)
    expected = sum(
        counts_left[label] * counts_right[label]
        for label in set(counts_left) | set(counts_right)
    ) / (len(left) ** 2)
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def _binary_metrics(human: Sequence[bool], predicted: Sequence[bool]) -> dict[str, Any]:
    tp = sum(h and p for h, p in zip(human, predicted))
    fp = sum((not h) and p for h, p in zip(human, predicted))
    fn = sum(h and (not p) for h, p in zip(human, predicted))
    tn = sum((not h) and (not p) for h, p in zip(human, predicted))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "human_positive_count": tp + fn,
        "predicted_positive_count": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _macro_f1(human: Sequence[Any], predicted: Sequence[Any]) -> float | None:
    values = []
    for label in set(human) | set(predicted):
        result = _binary_metrics(
            [value == label for value in human],
            [value == label for value in predicted],
        )
        if result["f1"] is not None:
            values.append(result["f1"])
    return sum(values) / len(values) if values else None


def _percentile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def score_judge_predictions(
    *,
    consensus_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    consensus_trace_ids = [str(row["trace_id"]) for row in consensus_rows]
    prediction_trace_ids = [str(row.get("trace_id")) for row in prediction_rows]
    if len(consensus_rows) != 24 or len(set(consensus_trace_ids)) != 24:
        raise ValueError("consensus must contain 24 unique trace IDs")
    if len(prediction_rows) != 24 or len(set(prediction_trace_ids)) != 24:
        raise ValueError("predictions must contain 24 unique trace IDs")
    if set(consensus_trace_ids) != set(prediction_trace_ids):
        raise ValueError("prediction trace IDs differ from frozen consensus")
    predictions = {row["trace_id"]: row["prediction"] for row in prediction_rows}
    human_outcomes = [row["trajectory_outcome"] for row in consensus_rows]
    judge_outcomes = [
        "correct" if predictions[row["trace_id"]]["correct"] is True else "incorrect"
        if predictions[row["trace_id"]]["correct"] is False else "unscorable"
        for row in consensus_rows
    ]
    agreement_count = sum(a == b for a, b in zip(human_outcomes, judge_outcomes))
    outcome_kappa = _kappa(human_outcomes, judge_outcomes)
    human_stages = [row["primary_failure_stage"] for row in consensus_rows]
    judge_stages = [predictions[row["trace_id"]]["primary_failure_stage"] for row in consensus_rows]
    human_tag_sets = [set(row["secondary_error_tags"]) for row in consensus_rows]
    judge_tag_sets = [set(predictions[row["trace_id"]]["secondary_error_tags"]) for row in consensus_rows]
    tag_labels = sorted(set().union(*human_tag_sets, *judge_tag_sets))
    tag_metrics = {
        label: _binary_metrics(
            [label in tags for tags in human_tag_sets],
            [label in tags for tags in judge_tag_sets],
        )
        for label in tag_labels
    }
    estimable_tag_f1 = [value["f1"] for value in tag_metrics.values() if value["f1"] is not None]
    jaccards = [
        len(human & judge) / len(human | judge) if human | judge else 1.0
        for human, judge in zip(human_tag_sets, judge_tag_sets)
    ]

    bootstrap = config["human_review"]["confidence_interval"]
    rng = random.Random(int(bootstrap["seed"]))
    agreement_samples: list[float] = []
    kappa_samples: list[float] = []
    critical_bootstrap: dict[str, dict[str, list[float]]] = {
        label: {"precision": [], "recall": [], "f1": []}
        for label in config["thresholds"]["critical_labels"]["labels"]
    }
    for _ in range(int(bootstrap["replicates"])):
        indices = [rng.randrange(len(consensus_rows)) for _ in consensus_rows]
        left = [human_outcomes[index] for index in indices]
        right = [judge_outcomes[index] for index in indices]
        agreement_samples.append(sum(a == b for a, b in zip(left, right)) / len(left))
        value = _kappa(left, right)
        if value is not None:
            kappa_samples.append(value)
        for label, values in critical_bootstrap.items():
            metrics = _binary_metrics(
                [label in human_tag_sets[index] for index in indices],
                [label in judge_tag_sets[index] for index in indices],
            )
            for metric_name in values:
                metric_value = metrics[metric_name]
                if metric_value is not None:
                    values[metric_name].append(metric_value)
    outcome_threshold = config["thresholds"]["outcome"]
    outcome_passed = (
        agreement_count >= int(outcome_threshold["minimum_agreements_out_of_24"])
        and agreement_count / 24 >= float(outcome_threshold["agreement_min"])
        and outcome_kappa is not None
        and outcome_kappa >= float(outcome_threshold["cohen_kappa_min"])
    )
    critical: dict[str, Any] = {}
    critical_threshold = float(config["thresholds"]["critical_labels"]["f1_min"])
    for label in config["thresholds"]["critical_labels"]["labels"]:
        metrics = tag_metrics.get(
            label,
            _binary_metrics([False] * 24, [False] * 24),
        )
        if metrics["human_positive_count"] == 0:
            status = "inconclusive_no_positive_human_cases"
        elif metrics["f1"] is not None and metrics["f1"] >= critical_threshold:
            status = "passed"
        else:
            status = "failed"
        bootstrap_values = critical_bootstrap[label]
        critical[label] = {
            **metrics,
            "bootstrap_95_ci": {
                metric_name: [
                    _percentile(values, 0.025),
                    _percentile(values, 0.975),
                ]
                for metric_name, values in bootstrap_values.items()
            },
            "status": status,
        }
    diagnostic_thresholds = config["thresholds"]["diagnostics"]
    primary_kappa = _kappa(human_stages, judge_stages)
    primary_macro_f1 = _macro_f1(human_stages, judge_stages)
    secondary_macro_f1 = (
        sum(estimable_tag_f1) / len(estimable_tag_f1) if estimable_tag_f1 else None
    )
    secondary_jaccard = sum(jaccards) / len(jaccards)
    diagnostics = {
        "primary_failure_stage": {
            "exact_agreement": sum(a == b for a, b in zip(human_stages, judge_stages)) / 24,
            "cohen_kappa": primary_kappa,
            "macro_f1": primary_macro_f1,
            "automated_claim_use_allowed": bool(
                primary_kappa is not None
                and primary_kappa >= diagnostic_thresholds["primary_failure_stage_cohen_kappa_min"]
                and primary_macro_f1 is not None
                and primary_macro_f1 >= diagnostic_thresholds["primary_failure_stage_macro_f1_min"]
            ),
        },
        "secondary_error_tags": {
            "exact_agreement": sum(a == b for a, b in zip(human_tag_sets, judge_tag_sets)) / 24,
            "mean_jaccard": secondary_jaccard,
            "macro_f1": secondary_macro_f1,
            "by_label": tag_metrics,
            "automated_claim_use_allowed": bool(
                secondary_jaccard >= diagnostic_thresholds["secondary_multilabel_jaccard_min"]
                and secondary_macro_f1 is not None
                and secondary_macro_f1 >= diagnostic_thresholds["secondary_macro_f1_min"]
            ),
        },
    }
    class_balance = Counter(human_outcomes)
    class_balance_passed = (
        class_balance["correct"] >= config["class_balance"]["minimum_human_correct"]
        and class_balance["incorrect"] >= config["class_balance"]["minimum_human_incorrect"]
    )
    critical_statuses = [value["status"] for value in critical.values()]
    judge_freeze_allowed = bool(
        outcome_passed
        and class_balance_passed
        and all(status == "passed" for status in critical_statuses)
        and not any(row["codebook_systematic_ambiguity"] for row in consensus_rows)
    )
    status = (
        "passed" if judge_freeze_allowed else "inconclusive"
        if outcome_passed and "failed" not in critical_statuses else "failed"
    )
    return {
        "finalization_version": FINALIZATION_VERSION,
        "status": status,
        "judge_freeze_allowed": judge_freeze_allowed,
        "human_consensus_distribution": dict(sorted(class_balance.items())),
        "class_balance_passed": class_balance_passed,
        "outcome": {
            "agreement_count": agreement_count,
            "agreement": agreement_count / 24,
            "cohen_kappa": outcome_kappa,
            "bootstrap_95_ci": {
                "agreement": [
                    _percentile(agreement_samples, 0.025),
                    _percentile(agreement_samples, 0.975),
                ],
                "cohen_kappa": [
                    _percentile(kappa_samples, 0.025),
                    _percentile(kappa_samples, 0.975),
                ],
            },
            "confusion_human_rows_judge_columns": {
                f"human={human},judge={judge}": sum(
                    a == human and b == judge
                    for a, b in zip(human_outcomes, judge_outcomes)
                )
                for human in ("correct", "incorrect")
                for judge in ("correct", "incorrect", "unscorable")
            },
            "passed": outcome_passed,
        },
        "critical_labels": critical,
        "diagnostics": diagnostics,
        "prediction_count": len(prediction_rows),
        "consensus_count": len(consensus_rows),
    }


__all__ = [
    "FINALIZATION_VERSION",
    "build_human_consensus",
    "decrypt_prediction_rows",
    "score_judge_predictions",
    "validate_stage2_return",
]
