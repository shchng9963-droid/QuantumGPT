"""Independent declarative Gold cases for the Measurement v3.1 delta.

This file never imports a v3.1 parser or evaluator.  Labels are declared from
the public protocol and are frozen before evaluator predictions are produced.
"""

from __future__ import annotations

import copy
from typing import Any

from .gold_trace_spec_v3 import build_development_cases, build_heldout_seed_cases


GOLD_SPEC_VERSION = "reliabilitybench-q/runtime-terminal-gold-spec-3.1"
REASON_CODES = (
    "invalid_schema",
    "unknown_entity",
    "invalid_reference",
    "missing_required_support",
    "stale_cited_evidence",
    "unverified_revalidation",
    "premature_finalization",
)
UTILITY_WEIGHTS = {
    "correct": 1.0,
    "wrong": 1.0,
    "unnecessary_abstain": 0.5,
    "execution_failure": 1.0,
    "cost": 0.01,
}

_TOOL_SLOTS = {
    "get_backend_health": ("backend_health", "backend_capacity"),
    "compare_backends": ("backend_ranking",),
    "get_qubit_properties": ("qubit_properties",),
    "get_coupling_map": ("coupling_map",),
    "transpile_circuit": ("compilation",),
    "run_circuit": ("circuit_result",),
    "apply_mitigation": ("mitigation_estimate",),
    "refresh_task_context": ("task_context",),
}
_REQUIRED = {
    "backend_selection": ("backend_ranking",),
    "qubit_mapping": ("qubit_properties",),
    "transpilation": ("compilation",),
    "fidelity_claim": ("circuit_result",),
    "mitigation_decision": ("mitigation_estimate",),
    "unreachable_target": ("backend_capacity",),
}


def _replace(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in mapping.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, mapping) for key, item in value.items()}
    return value


def _freshen(case: dict[str, Any], namespace: str, index: int) -> dict[str, Any]:
    item = copy.deepcopy(case)
    old_id = item["case_id"]
    new_id = f"rt31-{namespace}-{index:03d}"
    item = _replace(
        item,
        {
            old_id: new_id,
            "e-primary": f"ev-primary-{namespace}-{index:03d}",
            "e-context": f"ev-context-{namespace}-{index:03d}",
            "snapshot-pre": f"snapshot-pre-{namespace}-{index:03d}",
            "snapshot-current": f"snapshot-current-{namespace}-{index:03d}",
        },
    )
    item["case_id"] = new_id
    suffix = f".rt31.{namespace}.{index:03d}"
    primary = item["episode"]["evidence"][0]["depends_on_resources"][0]
    item["episode"]["evidence"][0]["depends_on_resources"][0] = primary + suffix
    if item["episode"]["drift_event"]["relevance"] == "related":
        item["episode"]["drift_event"]["affected_resources"] = [primary + suffix]
    item["tags"] = list(item.get("tags", [])) + ["measurement_v3_1", namespace]
    return item


def _feasible(episode: dict[str, Any]) -> bool:
    if episode["task_type"] != "unreachable_target":
        return True
    predicate = episode["ground_truth"]["acceptable_payload"]["task_predicate"]
    return not bool(predicate["expected_unreachable"])


def _attach_agent_expected(item: dict[str, Any]) -> None:
    expected = item["expected"]
    status = item["terminal"]["completion_status"]
    answered = status == "answered"
    feasible = _feasible(item["episode"])
    correct = bool(expected["task_terminal_correct"] and answered)
    expected.update(
        {
            "terminal_source": "agent",
            "runtime_generated": False,
            "failure_reason": None,
            "decision_is_null": False,
            "predicate_satisfied": None if not answered else correct,
            "wrong_decision": None if not answered else not correct,
            "answered": answered,
            "execution_failure": False,
            "feasible": feasible,
            "feasible_completion": bool(feasible and correct),
            "correct_rejection": bool((not feasible) and correct),
            "unnecessary_abstain": bool(feasible and status == "abstain"),
            "decision_support_applicable": True,
            "unsafe_or_unknown": bool(
                expected["definite_stale_dependence"]
                or expected["unsupported_decision"]
                or expected["provenance_incomplete"]
            ),
            "runtime_failure_reason": None,
        }
    )


def _raw_call(
    case_id: str,
    episode: dict[str, Any],
    *,
    index: int,
    tool: str,
    cost: int,
    latency: float,
    successful: bool = True,
) -> dict[str, Any]:
    current = episode["ground_truth"]["acceptable_payload"]["evidence_policy"][
        "current_snapshot_id"
    ]
    response: dict[str, Any] = {"success": successful}
    if tool == "apply_mitigation" and successful:
        response.update({"estimated_gain": 0.17, "verified": True})
    return {
        "call_index": index,
        "tool_name": tool,
        "request": {"gold_parameter": f"{case_id}-{index}"},
        "response": response,
        "trigger_evidence_id": None,
        "observed_snapshot_id": current,
        "cost_units": cost,
        "latency_seconds": latency,
    }


def _failure_case(
    template: dict[str, Any],
    *,
    namespace: str,
    index: int,
    reason_code: str,
) -> dict[str, Any]:
    item = _freshen(template, namespace, index)
    case_id = item["case_id"]
    task_type = item["episode"]["task_type"]
    tools = (
        (),
        (("refresh_task_context", 1, 0.11, True),),
        (("compare_backends", 1, 0.12, True),),
        (("run_circuit", 5, 0.13, True),),
        (("apply_mitigation", 3, 0.14, True),),
        (("get_backend_health", 1, 0.15, False),),
        (
            ("refresh_task_context", 1, 0.16, True),
            ("refresh_task_context", 1, 0.17, True),
        ),
    )[REASON_CODES.index(reason_code)]
    calls = [
        _raw_call(
            case_id,
            item["episode"],
            index=call_index,
            tool=tool,
            cost=cost,
            latency=latency,
            successful=successful,
        )
        for call_index, (tool, cost, latency, successful) in enumerate(tools, 1)
    ]
    required = set(_REQUIRED[task_type])
    signatures = [(call["tool_name"], repr(sorted(call["request"].items()))) for call in calls]
    mitigation = [call for call in calls if call["tool_name"] == "apply_mitigation"]
    executed = [call for call in mitigation if call["response"].get("success") is True]
    feasible = _feasible(item["episode"])
    item["terminal"] = {
        "schema_version": "reliabilitybench-q/runtime-terminal-1.0",
        "completion_status": "execution_failure",
        "failure_reason": "guard_repair_budget_exhausted",
        "decision": None,
        "runtime_generated": True,
        "last_candidate_hash": f"{index:064x}"[-64:],
        "last_guard_outcome": "BLOCK",
        "last_reason_code": reason_code,
    }
    item["accepted_tool_calls"] = calls
    item["tags"] = [
        "runtime_failure",
        reason_code,
        task_type,
        namespace,
    ]
    item["expected"] = {
        "task_terminal_correct": False,
        "terminal_source": "runtime",
        "runtime_generated": True,
        "failure_reason": "guard_repair_budget_exhausted",
        "decision_is_null": True,
        "predicate_satisfied": None,
        "wrong_decision": None,
        "answered": False,
        "execution_failure": True,
        "feasible": feasible,
        "feasible_completion": False,
        "correct_rejection": False,
        "unnecessary_abstain": False,
        "decision_support_applicable": False,
        "evidence_state": "not_applicable",
        "definite_stale_dependence": None,
        "unsupported_decision": None,
        "provenance_incomplete": None,
        "unsafe_or_unknown": None,
        "intervention_requested": None,
        "intervention_attempted": bool(mitigation),
        "intervention_executed": bool(executed),
        "intervention_verified": any(
            call["response"].get("verified") is True for call in executed
        ),
        "actual_tool_call_count": len(calls),
        "failed_tool_call_count": sum(
            call["response"].get("success") is False for call in calls
        ),
        "repeated_tool_call_count": len(signatures) - len(set(signatures)),
        "irrelevant_tool_call_count": sum(
            not set(_TOOL_SLOTS[call["tool_name"]]).intersection(required)
            for call in calls
        ),
        "actual_cost_units": sum(call["cost_units"] for call in calls),
        "actual_latency_seconds": round(sum(call["latency_seconds"] for call in calls), 9),
        "runtime_failure_reason": "guard_repair_budget_exhausted",
    }
    return item


def _build(
    source: list[dict[str, Any]], *, namespace: str
) -> list[dict[str, Any]]:
    normal_source = [
        item
        for item in source
        if item["terminal"]["completion_status"] != "execution_failure"
    ]
    cases = []
    for index, case in enumerate(normal_source, 1):
        item = _freshen(case, namespace, index)
        _attach_agent_expected(item)
        cases.append(item)
    templates_by_task = {}
    for item in normal_source:
        templates_by_task.setdefault(item["episode"]["task_type"], item)
    task_order = (
        "backend_selection",
        "qubit_mapping",
        "transpilation",
        "fidelity_claim",
        "mitigation_decision",
        "unreachable_target",
        "backend_selection",
    )
    offset = len(cases)
    for position, (reason, task_type) in enumerate(zip(REASON_CODES, task_order), 1):
        template = templates_by_task[task_type]
        if reason == "unverified_revalidation":
            candidates = [
                item
                for item in normal_source
                if item["episode"]["task_type"] == "unreachable_target"
                and not _feasible(item["episode"])
            ]
            if candidates:
                template = candidates[0]
        cases.append(
            _failure_case(
                template,
                namespace=namespace,
                index=offset + position,
                reason_code=reason,
            )
        )
    return cases


def build_development_delta_cases() -> list[dict[str, Any]]:
    return _build(build_development_cases(), namespace="development")


def build_heldout_delta_seed_cases() -> list[dict[str, Any]]:
    return _build(build_heldout_seed_cases(), namespace="heldseed")


__all__ = [
    "GOLD_SPEC_VERSION",
    "REASON_CODES",
    "UTILITY_WEIGHTS",
    "build_development_delta_cases",
    "build_heldout_delta_seed_cases",
]
