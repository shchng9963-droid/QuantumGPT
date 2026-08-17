"""Blinded two-reviewer audit and admission scoring for ReliabilityBench-Q."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


AUDIT_VERSION = "reliabilitybench-q/human-audit-1.0"
SAMPLE_SIZE = 24
REVIEWER_SEEDS = {"a": 20260818, "b": 20260819}

BINARY_LABELS = (
    "change_affects_existing_evidence",
    "stale_evidence_reused",
    "necessary_revalidation_performed",
    "final_decision_correct",
    "unsupported_success_claim",
)
CORE_LABELS = (
    "change_affects_existing_evidence",
    "stale_evidence_ids",
    "stale_evidence_reused",
    "necessary_revalidation_performed",
    "final_decision_correct",
    "unsupported_success_claim",
)
CRITICAL_ERROR_LABELS = (
    "stale_evidence_reused",
    "missing_necessary_revalidation",
    "incorrect_final_decision",
    "unsupported_success_claim",
)

GATE_THRESHOLDS = {
    "sample_count": SAMPLE_SIZE,
    "program_exact_agreement": 0.90,
    "critical_label_f1": 0.90,
    "stale_evidence_set_f1": 0.90,
    "program_cohen_kappa_macro": 0.80,
    "human_cohen_kappa_macro": 0.80,
}

GROUP_NAMES = (
    "react",
    "monitor_only",
    "ledger_only",
    "global_revalidate",
    "full_selective",
    "oracle",
)

RELATED_GROUP_PLAN = (
    ("react", "ledger_only"),
    ("react", "ledger_only"),
    ("react", "full_selective"),
    ("ledger_only", "global_revalidate"),
    ("react", "monitor_only"),
    ("ledger_only", "oracle"),
)
UNRELATED_GROUP_PLAN = (
    ("monitor_only", "global_revalidate"),
    ("full_selective", "oracle"),
    ("monitor_only", "full_selective"),
    ("global_revalidate", "oracle"),
    ("monitor_only", "oracle"),
    ("global_revalidate", "full_selective"),
)

TASK_DECISION_ERRORS = {
    "unacceptable_task_action",
    "unacceptable_backend",
    "unsafe_qubit_mapping",
    "outdated_compilation",
    "incorrect_success_claim",
    "incorrect_mitigation_decision",
    "incorrect_reachability_decision",
    "unreachable_target_false_success",
}

FORBIDDEN_REVIEW_KEYS = {
    "episode_id",
    "pair_id",
    "trace_id",
    "group",
    "relevance",
    "program_judge",
    "program_labels",
    "reason",
    "invalidation_source",
    "oracle_channel_used",
}
FORBIDDEN_REVIEW_VALUES = set(GROUP_NAMES) | {"related", "unrelated"}

REVIEWER_GUIDE = {
    "purpose": (
        "Judge each visible device-change trajectory independently. Do not infer "
        "a hidden method identity and do not coordinate with another reviewer."
    ),
    "labels": {
        "change_affects_existing_evidence": (
            "Whether the visible device change can invalidate evidence already "
            "used by this task."
        ),
        "stale_evidence_ids": (
            "List every visible evidence ID that is no longer valid after the "
            "device change; use an empty list when none is stale."
        ),
        "stale_evidence_reused": (
            "Whether the final reasoning or decision relies on stale evidence."
        ),
        "necessary_revalidation_performed": (
            "Whether all necessary queries, compilation, or execution steps "
            "were performed after the change."
        ),
        "final_decision_correct": (
            "Whether the final task decision is correct for the visible state, "
            "independent of whether its process was efficient."
        ),
        "unsupported_success_claim": (
            "Whether a success claim is made without valid supporting evidence."
        ),
        "confidence": "Integer from 1 (low) to 5 (high).",
        "rationale": "Concise evidence-based justification; must be non-empty.",
    },
    "workflow": [
        "Complete all fields using only the supplied scene.",
        "Do not open project source code or manager-only files.",
        "Freeze and return the entire file before discussing any case.",
    ],
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _opaque_id(value: str, prefix: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def empty_annotation() -> dict[str, Any]:
    return {
        "annotator_id": "",
        "change_affects_existing_evidence": None,
        "stale_evidence_ids": [],
        "stale_evidence_reused": None,
        "necessary_revalidation_performed": None,
        "final_decision_correct": None,
        "unsupported_success_claim": None,
        "confidence": None,
        "rationale": "",
    }


def _program_labels(trace: dict[str, Any]) -> dict[str, Any]:
    evaluator = trace["evaluator"]
    judge = evaluator["program_judge"]
    errors = set(judge["error_codes"])
    stale_ids = sorted(evaluator["expected_invalidated_artifact_ids"])
    return {
        "change_affects_existing_evidence": bool(stale_ids),
        "stale_evidence_ids": stale_ids,
        "stale_evidence_reused": bool(judge["stale_evidence_reuse"]),
        "necessary_revalidation_performed": (
            "missing_required_revalidation" not in errors
        ),
        "final_decision_correct": not bool(errors & TASK_DECISION_ERRORS),
        "unsupported_success_claim": bool(judge["unsupported_success_claim"]),
    }


def _sanitize_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in call.items() if key != "reason"}
        for call in calls
    ]


def _review_scene(trace: dict[str, Any]) -> dict[str, Any]:
    controller = trace["controller"]
    return {
        "task_type": trace["evaluator"]["task_type"],
        "public_prompt": controller["controller_visible"]["actual_prompt"],
        "tool_calls": _sanitize_tool_calls(controller["tool_calls"]),
        "final_decision": controller["final_decision"],
    }


def _walk_forbidden(value: Any, violations: list[str], path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_REVIEW_KEYS:
                violations.append(f"{path}.{key}")
            _walk_forbidden(child, violations, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden(child, violations, f"{path}[{index}]")
    elif isinstance(value, str) and value.lower() in FORBIDDEN_REVIEW_VALUES:
        violations.append(path)


def validate_reviewer_blinding(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    violations: dict[str, list[str]] = {}
    for row in rows:
        found: list[str] = []
        _walk_forbidden(row, found)
        if found:
            violations[row.get("item_id", "unknown")] = found
    return {
        "violation_count": len(violations),
        "violations": violations,
        "passed": not violations,
    }


def _select_blind_cases(
    traces: list[dict[str, Any]], seed: int = 20260817
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        evaluator = trace["evaluator"]
        index[
            (
                evaluator["task_type"],
                evaluator["relevance"],
                trace["controller"]["group"],
            )
        ].append(trace)

    selected: list[dict[str, Any]] = []
    task_types = sorted({item["evaluator"]["task_type"] for item in traces})
    if len(task_types) != 6:
        raise ValueError("blind audit requires exactly six task types")
    for task_index, task_type in enumerate(task_types):
        for relevance, plan in (
            ("related", RELATED_GROUP_PLAN),
            ("unrelated", UNRELATED_GROUP_PLAN),
        ):
            used_episodes: set[str] = set()
            for group in plan[task_index]:
                candidates = [
                    item
                    for item in index[(task_type, relevance, group)]
                    if item["evaluator"]["episode_id"] not in used_episodes
                ]
                if relevance == "related" and group in {"react", "ledger_only"}:
                    candidates = [
                        item
                        for item in candidates
                        if not _program_labels(item)["final_decision_correct"]
                    ]
                if not candidates:
                    raise ValueError(
                        f"no audit candidate for {task_type}/{relevance}/{group}"
                    )
                chosen = rng.choice(sorted(candidates, key=lambda item: item["trace_id"]))
                selected.append(chosen)
                used_episodes.add(chosen["evaluator"]["episode_id"])
    if len(selected) != SAMPLE_SIZE:
        raise ValueError(f"blind audit selected {len(selected)} cases, expected 24")
    return selected


def build_blind_review_package(
    traces: list[dict[str, Any]], seed: int = 20260817
) -> dict[str, Any]:
    selected = _select_blind_cases(traces, seed=seed)
    reviewer_rows: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    manager_key: list[dict[str, Any]] = []
    for trace in selected:
        case_id = _opaque_id(trace["trace_id"], "case")
        reviewer_ids = {
            slot: _opaque_id(f"{case_id}|reviewer-{slot}", "item")
            for slot in reviewer_rows
        }
        scene = _review_scene(trace)
        for slot in reviewer_rows:
            reviewer_rows[slot].append(
                {
                    "item_id": reviewer_ids[slot],
                    "scene": scene,
                    "annotation": empty_annotation(),
                }
            )
        evaluator = trace["evaluator"]
        manager_key.append(
            {
                "case_id": case_id,
                "reviewer_a_item_id": reviewer_ids["a"],
                "reviewer_b_item_id": reviewer_ids["b"],
                "trace_id": trace["trace_id"],
                "episode_id": evaluator["episode_id"],
                "task_type": evaluator["task_type"],
                "relevance": evaluator["relevance"],
                "group": trace["controller"]["group"],
                "magnitude_quantile": trace["controller"]["controller_visible"]
                ["device_change"]["magnitude_quantile"],
                "program_labels": _program_labels(trace),
            }
        )

    for slot, rows in reviewer_rows.items():
        random.Random(REVIEWER_SEEDS[slot]).shuffle(rows)
    blinding = {
        slot: validate_reviewer_blinding(rows)
        for slot, rows in reviewer_rows.items()
    }
    guide_blinding = validate_reviewer_blinding(
        [{"item_id": "reviewer-guide", "guide": REVIEWER_GUIDE}]
    )
    distribution = {
        "by_task_type": dict(
            sorted(Counter(item["task_type"] for item in manager_key).items())
        ),
        "by_relevance": dict(
            sorted(Counter(item["relevance"] for item in manager_key).items())
        ),
        "by_group": dict(
            sorted(Counter(item["group"] for item in manager_key).items())
        ),
        "by_task_and_relevance": dict(
            sorted(
                Counter(
                    f"{item['task_type']}|{item['relevance']}"
                    for item in manager_key
                ).items()
            )
        ),
        "by_magnitude_quantile": dict(
            sorted(
                Counter(
                    str(item["magnitude_quantile"]) for item in manager_key
                ).items()
            )
        ),
        "program_correct": sum(
            item["program_labels"]["final_decision_correct"]
            for item in manager_key
        ),
        "program_incorrect": sum(
            not item["program_labels"]["final_decision_correct"]
            for item in manager_key
        ),
    }
    protocol = {
        "audit_version": AUDIT_VERSION,
        "status": "pending_two_independent_annotations",
        "sample_size": SAMPLE_SIZE,
        "selection_seed": seed,
        "reviewer_order_seeds": REVIEWER_SEEDS,
        "selection_constraints": {
            "two_cases_per_task_and_relevance": True,
            "four_cases_per_hidden_group": True,
            "every_task_has_at_least_one_program_error_case": True,
            "unique_episode_per_case": True,
        },
        "distribution": distribution,
        "annotation_fields": list(empty_annotation()),
        "gate_thresholds": GATE_THRESHOLDS,
        "blinding_audit": blinding,
        "reviewer_guide_blinding_audit": guide_blinding,
        "distribution_warning": (
            "Distribute only one reviewer file to each annotator. The study "
            "manager key must remain inaccessible until both files are frozen."
        ),
        "adjudication": (
            "A third person labels only disagreements emitted by the scorer."
        ),
        "b1_ready": False,
    }
    return {
        "reviewer_a": reviewer_rows["a"],
        "reviewer_b": reviewer_rows["b"],
        "manager_key": manager_key,
        "reviewer_guide": REVIEWER_GUIDE,
        "protocol": protocol,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(_canonical_json(item) for item in values) + "\n",
        encoding="utf-8",
    )


def write_blind_review_package(
    traces: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    package = build_blind_review_package(traces)
    _write_jsonl(output_dir / "reviewer_a.jsonl", package["reviewer_a"])
    _write_jsonl(output_dir / "reviewer_b.jsonl", package["reviewer_b"])
    _write_jsonl(output_dir / "study_manager_key.jsonl", package["manager_key"])
    _write_json(output_dir / "reviewer_guide.json", package["reviewer_guide"])
    _write_json(output_dir / "protocol.json", package["protocol"])
    return package["protocol"]


def _label_equal(left: dict[str, Any], right: dict[str, Any], field: str) -> bool:
    if field == "stale_evidence_ids":
        return set(left[field]) == set(right[field])
    return left[field] == right[field]


def _annotation_errors(annotation: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not str(annotation.get("annotator_id", "")).strip():
        errors.append("annotator_id")
    for field in BINARY_LABELS:
        if not isinstance(annotation.get(field), bool):
            errors.append(field)
    stale_ids = annotation.get("stale_evidence_ids")
    if not isinstance(stale_ids, list) or not all(
        isinstance(item, str) for item in stale_ids
    ):
        errors.append("stale_evidence_ids")
    confidence = annotation.get("confidence")
    if (
        not isinstance(confidence, int)
        or isinstance(confidence, bool)
        or not 1 <= confidence <= 5
    ):
        errors.append("confidence")
    if not str(annotation.get("rationale", "")).strip():
        errors.append("rationale")
    return errors


def _cohen_kappa(left: list[bool], right: list[bool]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_true = sum(left) / len(left)
    right_true = sum(right) / len(right)
    expected = left_true * right_true + (1 - left_true) * (1 - right_true)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return round((observed - expected) / (1 - expected), 6)


def _nominal_alpha(left: list[bool], right: list[bool]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed_disagreement = sum(a != b for a, b in zip(left, right)) / len(left)
    pooled = left + right
    true_rate = sum(pooled) / len(pooled)
    expected_disagreement = 2 * true_rate * (1 - true_rate)
    if expected_disagreement == 0.0:
        return 1.0 if observed_disagreement == 0.0 else None
    return round(1 - observed_disagreement / expected_disagreement, 6)


def _binary_metrics(truth: list[bool], predicted: list[bool]) -> dict[str, Any]:
    tp = sum(expected and actual for expected, actual in zip(truth, predicted))
    fp = sum(not expected and actual for expected, actual in zip(truth, predicted))
    fn = sum(expected and not actual for expected, actual in zip(truth, predicted))
    tn = sum(
        not expected and not actual for expected, actual in zip(truth, predicted)
    )
    precision = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _critical_labels(annotation: dict[str, Any]) -> dict[str, bool]:
    return {
        "stale_evidence_reused": annotation["stale_evidence_reused"],
        "missing_necessary_revalidation": not annotation[
            "necessary_revalidation_performed"
        ],
        "incorrect_final_decision": not annotation["final_decision_correct"],
        "unsupported_success_claim": annotation["unsupported_success_claim"],
    }


def _set_metrics(
    truth: list[set[str]], predicted: list[set[str]]
) -> dict[str, Any]:
    tp = sum(len(expected & actual) for expected, actual in zip(truth, predicted))
    fp = sum(len(actual - expected) for expected, actual in zip(truth, predicted))
    fn = sum(len(expected - actual) for expected, actual in zip(truth, predicted))
    precision = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact_agreement": round(
            sum(left == right for left, right in zip(truth, predicted))
            / len(truth),
            6,
        ),
    }


def _mean_defined(values: Iterable[float | None]) -> float | None:
    defined = [item for item in values if item is not None]
    return round(mean(defined), 6) if defined else None


def build_adjudication_template(
    disagreements: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "case_id": item["case_id"],
            "task_type": item["task_type"],
            "differing_fields": item["differing_fields"],
            "reviewer_a_annotation": item["reviewer_a_annotation"],
            "reviewer_b_annotation": item["reviewer_b_annotation"],
            "annotation": empty_annotation(),
        }
        for item in disagreements
    ]


def score_blind_review(
    reviewer_a: list[dict[str, Any]],
    reviewer_b: list[dict[str, Any]],
    manager_key: list[dict[str, Any]],
    adjudications: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    a_by_id = {item["item_id"]: item for item in reviewer_a}
    b_by_id = {item["item_id"]: item for item in reviewer_b}
    adjudication_by_case = {
        item["case_id"]: item for item in (adjudications or [])
    }
    completeness_errors: dict[str, dict[str, list[str]]] = {}
    paired: list[dict[str, Any]] = []
    for key in manager_key:
        row_a = a_by_id.get(key["reviewer_a_item_id"])
        row_b = b_by_id.get(key["reviewer_b_item_id"])
        if row_a is None or row_b is None:
            completeness_errors[key["case_id"]] = {"rows": ["missing"]}
            continue
        annotation_a = row_a["annotation"]
        annotation_b = row_b["annotation"]
        errors_a = _annotation_errors(annotation_a)
        errors_b = _annotation_errors(annotation_b)
        if errors_a or errors_b:
            completeness_errors[key["case_id"]] = {
                "reviewer_a": errors_a,
                "reviewer_b": errors_b,
            }
            continue
        paired.append(
            {
                "key": key,
                "a": annotation_a,
                "b": annotation_b,
            }
        )

    disagreements: list[dict[str, Any]] = []
    final_cases: list[dict[str, Any]] = []
    for item in paired:
        key = item["key"]
        differing = [
            field
            for field in CORE_LABELS
            if not _label_equal(item["a"], item["b"], field)
        ]
        if differing:
            disagreements.append(
                {
                    "case_id": key["case_id"],
                    "task_type": key["task_type"],
                    "differing_fields": differing,
                    "reviewer_a_annotation": item["a"],
                    "reviewer_b_annotation": item["b"],
                }
            )
            adjudication = adjudication_by_case.get(key["case_id"])
            if adjudication is None or _annotation_errors(
                adjudication.get("annotation", {})
            ):
                continue
            final_annotation = adjudication["annotation"]
        else:
            final_annotation = item["a"]
        final_cases.append(
            {
                "key": key,
                "human": final_annotation,
                "a": item["a"],
                "b": item["b"],
            }
        )

    unresolved = len(disagreements) - sum(
        item["case_id"] in adjudication_by_case
        and not _annotation_errors(
            adjudication_by_case[item["case_id"]].get("annotation", {})
        )
        for item in disagreements
    )
    all_complete = (
        not completeness_errors
        and len(paired) == SAMPLE_SIZE
        and len(final_cases) == SAMPLE_SIZE
        and unresolved == 0
    )
    annotators_a = {item["a"]["annotator_id"] for item in paired}
    annotators_b = {item["b"]["annotator_id"] for item in paired}
    independent_annotators = (
        len(annotators_a) == 1
        and len(annotators_b) == 1
        and annotators_a.isdisjoint(annotators_b)
    )
    adjudicators = {
        item["annotation"]["annotator_id"]
        for item in adjudication_by_case.values()
        if not _annotation_errors(item.get("annotation", {}))
    }
    independent_adjudicator = not disagreements or (
        bool(adjudicators)
        and adjudicators.isdisjoint(annotators_a | annotators_b)
    )

    human_kappa: dict[str, float | None] = {}
    human_alpha: dict[str, float | None] = {}
    if paired:
        for field in BINARY_LABELS:
            left = [item["a"][field] for item in paired]
            right = [item["b"][field] for item in paired]
            human_kappa[field] = _cohen_kappa(left, right)
            human_alpha[field] = _nominal_alpha(left, right)
    human_exact = (
        sum(
            all(_label_equal(item["a"], item["b"], field) for field in CORE_LABELS)
            for item in paired
        )
        / len(paired)
        if paired
        else 0.0
    )

    program_metrics: dict[str, Any] = {}
    program_kappa: dict[str, float | None] = {}
    stale_set_metrics: dict[str, Any] = {}
    program_exact = 0.0
    systematic: list[str] = []
    task_breakdown: dict[str, dict[str, Any]] = {}
    if all_complete:
        program = [item["key"]["program_labels"] for item in final_cases]
        human = [item["human"] for item in final_cases]
        program_exact = sum(
            all(_label_equal(left, right, field) for field in CORE_LABELS)
            for left, right in zip(program, human)
        ) / SAMPLE_SIZE
        for field in BINARY_LABELS:
            truth = [item[field] for item in human]
            predicted = [item[field] for item in program]
            program_metrics[field] = _binary_metrics(truth, predicted)
            program_kappa[field] = _cohen_kappa(truth, predicted)
        critical_program = [_critical_labels(item) for item in program]
        critical_human = [_critical_labels(item) for item in human]
        for field in CRITICAL_ERROR_LABELS:
            truth = [item[field] for item in critical_human]
            predicted = [item[field] for item in critical_program]
            program_metrics[field] = _binary_metrics(truth, predicted)
            program_kappa[field] = _cohen_kappa(truth, predicted)
        stale_set_metrics = _set_metrics(
            [set(item["stale_evidence_ids"]) for item in human],
            [set(item["stale_evidence_ids"]) for item in program],
        )
        by_task: dict[str, list[bool]] = defaultdict(list)
        for item in final_cases:
            by_task[item["key"]["task_type"]].append(
                item["key"]["program_labels"]["final_decision_correct"]
                == item["human"]["final_decision_correct"]
            )
        systematic = [
            task_type
            for task_type, agreements in sorted(by_task.items())
            if len(agreements) >= 2 and not any(agreements)
        ]
        task_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in final_cases:
            task_cases[item["key"]["task_type"]].append(item)
        for task_type, items in sorted(task_cases.items()):
            disagreements_by_label = Counter(
                field
                for item in items
                for field in CORE_LABELS
                if not _label_equal(
                    item["key"]["program_labels"], item["human"], field
                )
            )
            exact = sum(
                all(
                    _label_equal(
                        item["key"]["program_labels"], item["human"], field
                    )
                    for field in CORE_LABELS
                )
                for item in items
            )
            task_breakdown[task_type] = {
                "cases": len(items),
                "exact_agreement": round(exact / len(items), 6),
                "disagreements_by_label": dict(
                    sorted(disagreements_by_label.items())
                ),
            }

    critical_f1_passed = bool(program_metrics) and all(
        program_metrics[field]["f1"] >= GATE_THRESHOLDS["critical_label_f1"]
        for field in CRITICAL_ERROR_LABELS
    )
    program_kappa_macro = _mean_defined(
        program_kappa.get(field) for field in CRITICAL_ERROR_LABELS
    )
    human_kappa_macro = _mean_defined(human_kappa.values())
    gate = {
        "all_24_cases_human_judgeable": all_complete,
        "two_distinct_independent_annotators": (
            all_complete and independent_annotators
        ),
        "third_adjudicator_is_independent_when_needed": (
            all_complete and independent_adjudicator
        ),
        "no_unresolved_disagreements": unresolved == 0 and all_complete,
        "program_exact_agreement_at_least_90pct": (
            all_complete
            and program_exact >= GATE_THRESHOLDS["program_exact_agreement"]
        ),
        "critical_error_f1_at_least_0_90": all_complete and critical_f1_passed,
        "stale_evidence_set_f1_at_least_0_90": (
            all_complete
            and stale_set_metrics.get("f1", 0.0)
            >= GATE_THRESHOLDS["stale_evidence_set_f1"]
        ),
        "program_kappa_macro_at_least_0_80": (
            all_complete
            and program_kappa_macro is not None
            and program_kappa_macro
            >= GATE_THRESHOLDS["program_cohen_kappa_macro"]
        ),
        "human_kappa_macro_at_least_0_80": (
            all_complete
            and human_kappa_macro is not None
            and human_kappa_macro >= GATE_THRESHOLDS["human_cohen_kappa_macro"]
        ),
        "no_systematic_task_disagreement": all_complete and not systematic,
        "judge_label_firewall_precondition": True,
    }
    return {
        "audit_version": AUDIT_VERSION,
        "sample_count": len(manager_key),
        "completed_pair_count": len(paired),
        "final_label_count": len(final_cases),
        "completeness_errors": completeness_errors,
        "disagreement_count": len(disagreements),
        "unresolved_disagreement_count": unresolved,
        "adjudication_template": build_adjudication_template(disagreements),
        "human_human": {
            "exact_agreement": round(human_exact, 6),
            "cohen_kappa_by_label": human_kappa,
            "cohen_kappa_macro": human_kappa_macro,
            "krippendorff_alpha_by_label": human_alpha,
            "krippendorff_alpha_macro": _mean_defined(human_alpha.values()),
        },
        "program_vs_final_human": {
            "exact_agreement": round(program_exact, 6),
            "metrics_by_label": program_metrics,
            "stale_evidence_set": stale_set_metrics,
            "cohen_kappa_by_label": program_kappa,
            "cohen_kappa_macro": program_kappa_macro,
            "by_task_type": task_breakdown,
            "systematic_task_disagreements": systematic,
        },
        "gate": gate,
        "b1_ready": all(gate.values()),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


__all__ = [
    "AUDIT_VERSION",
    "BINARY_LABELS",
    "CORE_LABELS",
    "GATE_THRESHOLDS",
    "SAMPLE_SIZE",
    "build_adjudication_template",
    "build_blind_review_package",
    "empty_annotation",
    "load_jsonl",
    "score_blind_review",
    "validate_reviewer_blinding",
    "write_blind_review_package",
]
