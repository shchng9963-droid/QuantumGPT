"""Blinded failure-taxonomy calibration for ReliabilityBench-Q B1.1."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


AUDIT_VERSION = "reliabilitybench-q/b1.1-failure-audit-0.2-calibration"
CALIBRATION_SIZE = 24
CALIBRATION_SEED = 20260821
REVIEWER_ORDER_SEEDS = {"a": 20260822, "b": 20260823}

GROUPS = (
    "react",
    "monitor_only",
    "ledger_only",
    "global_revalidate",
    "full_selective",
    "oracle",
)

PRIMARY_FAILURE_STAGES = (
    "Evidence",
    "Planning",
    "Action Formation",
    "Entity Binding",
    "Execution",
    "Revalidation",
    "Claim",
    "Evaluator",
)

SECONDARY_ERROR_TAGS = (
    "nonexistent_tool_name",
    "missing_parameter",
    "wrong_parameter_type",
    "invalid_parameter_value",
    "repeated_tool_call",
    "irrelevant_tool_call",
    "stale_evidence_reuse",
    "wrong_entity",
    "wrong_revalidation_target",
    "stale_reference_after_revalidation",
    "unsupported_success_claim",
    "unreachable_target_not_rejected",
    "missing_required_revalidation",
    "tool_execution_failure",
    "final_claim_conflicts_latest_evidence",
)

FORBIDDEN_REVIEW_KEYS = {
    "episode_id",
    "pair_id",
    "trace_id",
    "group",
    "controller_group",
    "hidden_group_id",
    "relevance",
    "program_judge",
    "program_correct",
    "program_error_codes",
    "schedule",
    "schedule_index",
}
FORBIDDEN_REVIEW_VALUES = set(GROUPS) | {"related", "unrelated"}

CODEBOOK_ZH = {
    "audit_version": AUDIT_VERSION,
    "codebook_version": "0.2-draft",
    "status": "draft_for_calibration_not_frozen",
    "language": "zh-CN",
    "purpose": (
        "依据可见任务、设备变化、智能体动作、工具观察和最终决策，定位轨迹的首要失败阶段。"
        "不得猜测实验组，不得查看程序评分或study-manager文件。"
    ),
    "causal_rule": (
        "若轨迹失败，选择错误链中最早且足以解释最终失败的阶段；不要仅选择最后出现或后果最严重的错误。"
    ),
    "outcomes": {
        "correct": "最终决策正确；即使存在低效调用，也可在二级标签中记录。",
        "incorrect": "最终决策错误，或完成任务所需的关键恢复条件未满足。",
        "uncertain": "现有可见信息不足以可靠判定，必须在理由中说明缺失信息。",
    },
    "primary_failure_stages": {
        "Evidence": "未识别已有证据已失效，或仍把失效证据当作当前有效事实。",
        "Planning": "证据状态已知，但选择了错误的恢复方向或任务策略。",
        "Action Formation": "工具名、参数名、必填字段、类型或取值不符合公开Schema。",
        "Entity Binding": "动作结构合法，但绑定了错误的backend、qubit、circuit或证据对象。",
        "Execution": "动作与对象均正确，但工具/API执行失败且不是智能体输入造成。",
        "Revalidation": "已经选择正确恢复方向，但遗漏、重复替代或错误执行了必要重验证。",
        "Claim": "已有最新有效证据，但最终结论与该证据矛盾，或证据不足仍宣称成功。",
        "Evaluator": "可见轨迹实质正确，但任务规范、可接受行动集合或程序化判定条件疑似错误/含糊。",
    },
    "secondary_error_tags": {
        "nonexistent_tool_name": "调用工具目录中不存在的名称。",
        "missing_parameter": "缺少工具Schema要求的参数。",
        "wrong_parameter_type": "参数类型错误。",
        "invalid_parameter_value": "参数取值越界、非法或不在允许集合。",
        "repeated_tool_call": "无新信息或无状态变化时重复同一查询。",
        "irrelevant_tool_call": "调用既不能恢复缺失证据、满足任务目标，也不能验证最终结论。",
        "stale_evidence_reuse": "最终推理或决策继续依赖已失效的旧证据。",
        "wrong_entity": "使用错误backend、qubit、circuit或其他任务实体。",
        "wrong_revalidation_target": "重验证工具或触发证据与需恢复对象不匹配。",
        "stale_reference_after_revalidation": "已完成新观察但最终仍引用旧结果作为依据。",
        "unsupported_success_claim": "缺少有效支持证据却宣称任务成功。",
        "unreachable_target_not_rejected": "目标不可达但没有明确拒绝或声明不可达。",
        "missing_required_revalidation": "在结束任务前缺少至少一个必要重验证步骤。",
        "tool_execution_failure": "合法工具调用发生外部执行错误、超时或API失败。",
        "final_claim_conflicts_latest_evidence": "最终结论与最新工具观察直接矛盾。",
    },
    "annotation_rules": [
        "trajectory_outcome=correct时，primary_failure_stage必须为null。",
        "trajectory_outcome=incorrect时，必须且只能填写一个primary_failure_stage。",
        "trajectory_outcome=uncertain时，primary_failure_stage可为null；理由必须说明缺失信息。",
        "secondary_error_tags可以多选；低效但正确的轨迹也可标记重复或无关调用。",
        "Evaluator仅用于规范或判定条件疑似有误，不用于普通模型错误。",
        "必须引用具体动作、工具观察或最终字段作为理由。",
    ],
    "confidence": "使用0.0到1.0之间的概率表示主观置信度；1.0表示完全确信。",
    "revision_history": [
        {
            "from": "0.1",
            "to": "0.2-draft",
            "change_type": "normative_clarification",
            "scope": "confidence_scale_only",
            "change": "明确confidence采用0.0到1.0概率量表。",
            "annotation_effect": (
                "不得因本次规范澄清自动修改trajectory_outcome、primary_failure_stage、"
                "secondary_error_tags、confidence或rationale中的任何原始标注。"
            ),
        }
    ],
    "calibration_notice": (
        "本版仅用于24条校准样本。校准讨论结束后必须冻结新版本及SHA-256，"
        "才能生成144条正式盲审包和新验证集。"
    ),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _opaque_id(value: str, prefix: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def empty_annotation() -> dict[str, Any]:
    return {
        "annotator_id": "",
        "trajectory_outcome": None,
        "primary_failure_stage": None,
        "secondary_error_tags": [],
        "confidence": None,
        "rationale": "",
    }


def validate_annotation(annotation: dict[str, Any], *, complete: bool) -> list[str]:
    errors: list[str] = []
    expected = set(empty_annotation())
    if set(annotation) != expected:
        errors.append("annotation_fields")
    if not complete:
        return errors

    if not str(annotation.get("annotator_id", "")).strip():
        errors.append("annotator_id")
    outcome = annotation.get("trajectory_outcome")
    if outcome not in {"correct", "incorrect", "uncertain"}:
        errors.append("trajectory_outcome")
    stage = annotation.get("primary_failure_stage")
    if outcome == "incorrect" and stage not in PRIMARY_FAILURE_STAGES:
        errors.append("primary_failure_stage")
    if outcome == "correct" and stage is not None:
        errors.append("primary_failure_stage")
    if outcome == "uncertain" and stage not in {None, *PRIMARY_FAILURE_STAGES}:
        errors.append("primary_failure_stage")
    tags = annotation.get("secondary_error_tags")
    if (
        not isinstance(tags, list)
        or len(tags) != len(set(tags))
        or any(tag not in SECONDARY_ERROR_TAGS for tag in tags)
    ):
        errors.append("secondary_error_tags")
    confidence = annotation.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0.0 <= confidence <= 1.0
    ):
        errors.append("confidence")
    if not str(annotation.get("rationale", "")).strip():
        errors.append("rationale")
    return errors


def _remove_noise(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _remove_noise(child)
            for key, child in value.items()
            if key not in {"timestamp", "system_fingerprint"}
        }
    if isinstance(value, list):
        return [_remove_noise(child) for child in value]
    return value


def _structured_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content", "")
    try:
        payload: Any = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        payload = str(content)
    return {"role": message.get("role"), "payload": _remove_noise(payload)}


def _review_scene(trace: dict[str, Any]) -> dict[str, Any]:
    public_prompt = json.loads(trace["public_prompt"])
    visible_messages = trace["messages"][2:]
    return {
        "task_type": trace["episode"]["task_type"],
        "public_task_and_event": _remove_noise(public_prompt),
        "agent_interaction": [
            _structured_message(message) for message in visible_messages
        ],
        "accepted_tool_calls": _remove_noise(trace["tool_calls"]),
        "final_decision": _remove_noise(trace["model_final_decision"]),
        "budget_usage": trace["budget_usage"],
        "parser_errors": _remove_noise(trace["llm"].get("parser_errors", [])),
        "api_errors": _remove_noise(trace["llm"].get("api_errors", [])),
        "trace_complete": trace["trace_complete"],
        "unscorable": trace["unscorable"],
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
            violations[row.get("item_id", "unknown")] = sorted(set(found))
    return {
        "passed": not violations,
        "violation_count": len(violations),
        "violations": violations,
    }


def _validate_source_traces(traces: list[dict[str, Any]]) -> None:
    if len(traces) != 144:
        raise ValueError(f"B1.1 audit requires 144 traces, got {len(traces)}")
    if len({trace["trace_id"] for trace in traces}) != 144:
        raise ValueError("trace IDs must be unique")
    task_types = {trace["episode"]["task_type"] for trace in traces}
    if len(task_types) != 6:
        raise ValueError("audit requires exactly six task types")
    if {trace["controller"]["group"] for trace in traces} != set(GROUPS):
        raise ValueError("audit requires the frozen six B1 groups")
    by_episode = Counter(trace["episode"]["episode_id"] for trace in traces)
    if len(by_episode) != 24 or set(by_episode.values()) != {6}:
        raise ValueError("audit requires 24 episodes with six traces each")
    if any(not trace["trace_complete"] or trace["unscorable"] for trace in traces):
        raise ValueError("calibration source must contain complete, scoreable traces")


def _select_calibration_cases(
    traces: list[dict[str, Any]], seed: int = CALIBRATION_SEED
) -> list[dict[str, Any]]:
    _validate_source_traces(traces)
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task_type in sorted({trace["episode"]["task_type"] for trace in traces}):
        for relevance in ("related", "unrelated"):
            candidates = [
                trace
                for trace in traces
                if trace["episode"]["task_type"] == task_type
                and trace["episode"]["relevance"] == relevance
            ]
            if len(candidates) != 12:
                raise ValueError(f"expected 12 candidates for {task_type}/{relevance}")
            cells[(task_type, relevance)] = sorted(
                candidates, key=lambda trace: trace["trace_id"]
            )

    rng = random.Random(seed)
    for _ in range(200_000):
        selected = [
            trace
            for cell in sorted(cells)
            for trace in rng.sample(cells[cell], 2)
        ]
        group_counts = Counter(trace["controller"]["group"] for trace in selected)
        correct_count = sum(trace["program_judge"]["correct"] for trace in selected)
        failed_tasks = {
            trace["episode"]["task_type"]
            for trace in selected
            if not trace["program_judge"]["correct"]
        }
        if (
            set(group_counts.values()) == {4}
            and 6 <= correct_count <= 10
            and len(failed_tasks) == 6
        ):
            return selected
    raise RuntimeError("could not construct balanced calibration sample")


def _manager_row(trace: dict[str, Any], reviewer_ids: dict[str, str]) -> dict[str, Any]:
    return {
        "case_id": _opaque_id(trace["trace_id"], "case"),
        "reviewer_a_item_id": reviewer_ids["a"],
        "reviewer_b_item_id": reviewer_ids["b"],
        "trace_id": trace["trace_id"],
        "episode_id": trace["episode"]["episode_id"],
        "task_type": trace["episode"]["task_type"],
        "relevance": trace["episode"]["relevance"],
        "group": trace["controller"]["group"],
        "program_correct": trace["program_judge"]["correct"],
        "program_error_codes": trace["program_judge"]["error_codes"],
    }


def build_calibration_package(
    traces: list[dict[str, Any]], seed: int = CALIBRATION_SEED
) -> dict[str, Any]:
    selected = _select_calibration_cases(traces, seed=seed)
    reviewers: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    manager_key: list[dict[str, Any]] = []
    for trace in selected:
        case_id = _opaque_id(trace["trace_id"], "case")
        reviewer_ids = {
            slot: _opaque_id(f"{case_id}|reviewer-{slot}", "item")
            for slot in reviewers
        }
        scene = _review_scene(trace)
        for slot in reviewers:
            reviewers[slot].append(
                {
                    "item_id": reviewer_ids[slot],
                    "scene": scene,
                    "annotation": empty_annotation(),
                }
            )
        manager_key.append(_manager_row(trace, reviewer_ids))

    for slot, rows in reviewers.items():
        random.Random(REVIEWER_ORDER_SEEDS[slot]).shuffle(rows)
    blinding = {
        slot: validate_reviewer_blinding(rows) for slot, rows in reviewers.items()
    }
    if not all(result["passed"] for result in blinding.values()):
        raise ValueError(f"review package failed blinding audit: {blinding}")

    distribution = {
        "by_task_type": dict(sorted(Counter(row["task_type"] for row in manager_key).items())),
        "by_relevance": dict(sorted(Counter(row["relevance"] for row in manager_key).items())),
        "by_group": dict(sorted(Counter(row["group"] for row in manager_key).items())),
        "program_correct": sum(row["program_correct"] for row in manager_key),
        "program_incorrect": sum(not row["program_correct"] for row in manager_key),
    }
    source_ids = sorted(trace["trace_id"] for trace in traces)
    protocol = {
        "audit_version": AUDIT_VERSION,
        "status": "taxonomy_calibration_pending",
        "calibration_size": CALIBRATION_SIZE,
        "selection_seed": seed,
        "reviewer_order_seeds": REVIEWER_ORDER_SEEDS,
        "source_trace_set_sha256": _sha256_bytes(
            _canonical_json(source_ids).encode("utf-8")
        ),
        "selection_constraints": {
            "two_cases_per_task_and_relevance": True,
            "four_cases_per_hidden_group": True,
            "contains_correct_controls": True,
            "every_task_has_program_incorrect_case": True,
        },
        "distribution": distribution,
        "annotation_fields": list(empty_annotation()),
        "blinding_audit": blinding,
        "validation_set_generation_allowed": False,
        "next_gate": (
            "Two independent calibration annotations, discrepancy discussion, "
            "then freeze codebook version and SHA-256."
        ),
    }
    return {
        "reviewer_a": reviewers["a"],
        "reviewer_b": reviewers["b"],
        "study_manager_key": manager_key,
        "codebook_zh": CODEBOOK_ZH,
        "protocol": protocol,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(_canonical_json(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_calibration_package(
    traces: list[dict[str, Any]],
    output_dir: Path,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    package = build_calibration_package(traces)
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {
        "reviewer_a.jsonl": package["reviewer_a"],
        "reviewer_b.jsonl": package["reviewer_b"],
        "study_manager_key.jsonl": package["study_manager_key"],
        "codebook_zh.json": package["codebook_zh"],
        "protocol.json": package["protocol"],
    }
    for name, value in files.items():
        if name.endswith(".jsonl"):
            _write_jsonl(output_dir / name, value)
        else:
            _write_json(output_dir / name, value)

    hashes = {
        name: _sha256_bytes((output_dir / name).read_bytes()) for name in sorted(files)
    }
    manifest = {
        "audit_version": AUDIT_VERSION,
        "status": "draft_calibration_package",
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat(),
        "files_sha256": hashes,
        "codebook_frozen": False,
        "validation_set_generation_allowed": False,
    }
    _write_json(output_dir / "package_manifest.json", manifest)
    return manifest


def _nominal_kappa(left: list[Any], right: list[Any]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    labels = set(left) | set(right)
    expected = sum(
        (left.count(label) / len(left)) * (right.count(label) / len(right))
        for label in labels
    )
    if expected == 1.0:
        return None
    return round((observed - expected) / (1 - expected), 6)


def _review_file_validation(
    original: list[dict[str, Any]], completed: list[dict[str, Any]]
) -> dict[str, Any]:
    original_by_id = {row["item_id"]: row for row in original}
    completed_ids = [row.get("item_id") for row in completed]
    errors: dict[str, list[str]] = {}
    if len(completed) != len(original):
        errors["$file"] = [f"row_count:{len(completed)}!=expected:{len(original)}"]
    if len(completed_ids) != len(set(completed_ids)):
        errors.setdefault("$file", []).append("duplicate_item_id")
    if set(completed_ids) != set(original_by_id):
        errors.setdefault("$file", []).append("item_id_set_mismatch")
    unchanged_scene_count = 0
    unchanged_trace_count = 0
    unchanged_blinded_public_count = 0
    trace_fields = {
        "accepted_tool_calls",
        "agent_interaction",
        "api_errors",
        "budget_usage",
        "final_decision",
        "parser_errors",
        "trace_complete",
        "unscorable",
    }
    blinded_public_fields = {"public_task_and_event", "task_type"}
    for row in completed:
        item_id = row.get("item_id", "unknown")
        row_errors: list[str] = []
        source = original_by_id.get(item_id)
        if source is not None:
            source_scene = source["scene"]
            completed_scene = row.get("scene")
            if completed_scene == source_scene:
                unchanged_scene_count += 1
            else:
                row_errors.append("scene_modified")
            if isinstance(completed_scene, dict):
                if all(
                    completed_scene.get(field) == source_scene.get(field)
                    for field in trace_fields
                ):
                    unchanged_trace_count += 1
                if all(
                    completed_scene.get(field) == source_scene.get(field)
                    for field in blinded_public_fields
                ):
                    unchanged_blinded_public_count += 1
        row_errors.extend(
            validate_annotation(row.get("annotation", {}), complete=True)
        )
        if row_errors:
            errors[item_id] = sorted(set(row_errors))
    item_ids_unique = len(completed_ids) == len(set(completed_ids))
    item_id_set_equal = set(completed_ids) == set(original_by_id)
    blinding = validate_reviewer_blinding(completed)
    return {
        "passed": not errors and blinding["passed"],
        "error_count": len(errors),
        "errors": errors,
        "integrity": {
            "expected_row_count": len(original),
            "actual_row_count": len(completed),
            "row_count_equal": len(completed) == len(original),
            "item_ids_unique": item_ids_unique,
            "item_id_set_equal": item_id_set_equal,
            "scene_rows_unchanged": unchanged_scene_count,
            "scene_all_unchanged": unchanged_scene_count == len(original),
            "trace_rows_unchanged": unchanged_trace_count,
            "trace_all_unchanged": unchanged_trace_count == len(original),
            "blinded_public_rows_unchanged": unchanged_blinded_public_count,
            "blinded_public_all_unchanged": (
                unchanged_blinded_public_count == len(original)
            ),
            "blinding_scan_passed": blinding["passed"],
            "blinding_violation_count": blinding["violation_count"],
        },
    }


def _annotation_labels(annotation: dict[str, Any]) -> tuple[Any, Any, frozenset[str]]:
    return (
        annotation["trajectory_outcome"],
        annotation["primary_failure_stage"],
        frozenset(annotation["secondary_error_tags"]),
    )


def score_calibration_reviews(
    reviewer_a: list[dict[str, Any]],
    reviewer_b: list[dict[str, Any]],
    original_a: list[dict[str, Any]],
    original_b: list[dict[str, Any]],
    manager_key: list[dict[str, Any]],
) -> dict[str, Any]:
    validation = {
        "reviewer_a": _review_file_validation(original_a, reviewer_a),
        "reviewer_b": _review_file_validation(original_b, reviewer_b),
    }
    if not all(result["passed"] for result in validation.values()):
        return {
            "audit_version": AUDIT_VERSION,
            "status": "invalid_returned_annotations",
            "validation": validation,
            "codebook_freeze_allowed": False,
            "validation_set_generation_allowed": False,
        }

    by_a = {row["item_id"]: row for row in reviewer_a}
    by_b = {row["item_id"]: row for row in reviewer_b}
    original_a_by_id = {row["item_id"]: row for row in original_a}
    pairs: list[dict[str, Any]] = []
    reconciliation: list[dict[str, Any]] = []
    for manager in manager_key:
        row_a = by_a[manager["reviewer_a_item_id"]]
        row_b = by_b[manager["reviewer_b_item_id"]]
        annotation_a = row_a["annotation"]
        annotation_b = row_b["annotation"]
        labels_a = _annotation_labels(annotation_a)
        labels_b = _annotation_labels(annotation_b)
        exact = labels_a == labels_b
        pair = {
            "case_id": manager["case_id"],
            "reviewer_a_item_id": manager["reviewer_a_item_id"],
            "reviewer_b_item_id": manager["reviewer_b_item_id"],
            "outcome_agreement": labels_a[0] == labels_b[0],
            "primary_stage_agreement": labels_a[1] == labels_b[1],
            "secondary_tags_agreement": labels_a[2] == labels_b[2],
            "exact_agreement": exact,
        }
        pairs.append(pair)
        if not exact:
            reconciliation.append(
                {
                    "discussion_id": _opaque_id(manager["case_id"], "discussion"),
                    "scene": original_a_by_id[manager["reviewer_a_item_id"]]["scene"],
                    "reviewer_a_annotation": annotation_a,
                    "reviewer_b_annotation": annotation_b,
                    "resolution": {
                        "resolver_ids": [],
                        "taxonomy_change_needed": None,
                        "agreed_outcome": None,
                        "agreed_primary_failure_stage": None,
                        "agreed_secondary_error_tags": [],
                        "codebook_change_notes": "",
                        "rationale": "",
                    },
                }
            )

    annotations_a = [
        by_a[manager["reviewer_a_item_id"]]["annotation"] for manager in manager_key
    ]
    annotations_b = [
        by_b[manager["reviewer_b_item_id"]]["annotation"] for manager in manager_key
    ]
    outcomes_a = [annotation["trajectory_outcome"] for annotation in annotations_a]
    outcomes_b = [annotation["trajectory_outcome"] for annotation in annotations_b]
    stages_a = [annotation["primary_failure_stage"] for annotation in annotations_a]
    stages_b = [annotation["primary_failure_stage"] for annotation in annotations_b]
    tag_sets_a = [set(annotation["secondary_error_tags"]) for annotation in annotations_a]
    tag_sets_b = [set(annotation["secondary_error_tags"]) for annotation in annotations_b]
    tag_jaccards = [
        len(left & right) / len(left | right) if left | right else 1.0
        for left, right in zip(tag_sets_a, tag_sets_b)
    ]
    tag_kappas = {
        tag: _nominal_kappa(
            [tag in tag_set for tag_set in tag_sets_a],
            [tag in tag_set for tag_set in tag_sets_b],
        )
        for tag in SECONDARY_ERROR_TAGS
    }
    estimable_tag_kappas = [
        value for value in tag_kappas.values() if value is not None
    ]
    program_outcomes = [
        "correct" if manager["program_correct"] else "incorrect"
        for manager in manager_key
    ]
    manager_analysis = {
        "program_vs_reviewer_a_outcome_agreement": round(
            sum(a == b for a, b in zip(program_outcomes, outcomes_a))
            / len(manager_key),
            6,
        ),
        "program_vs_reviewer_b_outcome_agreement": round(
            sum(a == b for a, b in zip(program_outcomes, outcomes_b))
            / len(manager_key),
            6,
        ),
        "program_outcome_distribution": dict(sorted(Counter(program_outcomes).items())),
    }
    agreement = {
        "case_count": len(pairs),
        "full_exact_agreement": round(
            sum(pair["exact_agreement"] for pair in pairs) / len(pairs), 6
        ),
        "outcome_exact_agreement": round(
            sum(pair["outcome_agreement"] for pair in pairs) / len(pairs), 6
        ),
        "primary_stage_exact_agreement": round(
            sum(pair["primary_stage_agreement"] for pair in pairs) / len(pairs), 6
        ),
        "secondary_tags_exact_agreement": round(
            sum(pair["secondary_tags_agreement"] for pair in pairs) / len(pairs), 6
        ),
        "secondary_tags_mean_jaccard": round(sum(tag_jaccards) / len(tag_jaccards), 6),
        "secondary_tags_cohen_kappa_by_label": tag_kappas,
        "secondary_tags_macro_cohen_kappa": (
            round(sum(estimable_tag_kappas) / len(estimable_tag_kappas), 6)
            if estimable_tag_kappas
            else None
        ),
        "secondary_tags_estimable_kappa_label_count": len(estimable_tag_kappas),
        "outcome_cohen_kappa": _nominal_kappa(outcomes_a, outcomes_b),
        "primary_stage_cohen_kappa": _nominal_kappa(stages_a, stages_b),
        "reviewer_a_outcomes": dict(sorted(Counter(outcomes_a).items())),
        "reviewer_b_outcomes": dict(sorted(Counter(outcomes_b).items())),
        "reviewer_a_primary_stages": dict(
            sorted(Counter(str(value) for value in stages_a).items())
        ),
        "reviewer_b_primary_stages": dict(
            sorted(Counter(str(value) for value in stages_b).items())
        ),
    }
    return {
        "audit_version": AUDIT_VERSION,
        "status": "reconciliation_required" if reconciliation else "ready_to_freeze",
        "validation": validation,
        "agreement": agreement,
        "disagreement_count": len(reconciliation),
        "pair_results": pairs,
        "manager_only_analysis": manager_analysis,
        "reconciliation_rows": reconciliation,
        "reconciliation_blinding": validate_reviewer_blinding(reconciliation),
        "codebook_freeze_allowed": not reconciliation,
        "validation_set_generation_allowed": False,
    }


def write_calibration_score(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    manager_report = dict(report)
    reconciliation = manager_report.pop("reconciliation_rows", [])
    _write_json(output_dir / "calibration_manager_report.json", manager_report)
    _write_jsonl(output_dir / "blind_reconciliation.jsonl", reconciliation)
    _write_json(output_dir / "proposed_codebook_v0.2_zh.json", CODEBOOK_ZH)
    hashes = {
        path.name: _sha256_bytes(path.read_bytes())
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    _write_json(
        output_dir / "score_manifest.json",
        {
            "audit_version": AUDIT_VERSION,
            "status": report["status"],
            "files_sha256": hashes,
            "codebook_frozen": False,
            "validation_set_generation_allowed": False,
        },
    )
