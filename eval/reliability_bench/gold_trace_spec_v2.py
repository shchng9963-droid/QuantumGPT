"""Independent declarative Gold-Trace specification.

This file intentionally imports no QuantumGPT evaluator or controller module.
It constructs raw public episodes, terminal objects, traces, and literal expected
measurements from a compact declarative case table.  Evaluator code must not be
called while Gold labels are built.
"""

from __future__ import annotations

import copy
from typing import Any


GOLD_SPEC_VERSION = "reliabilitybench-q/gold-trace-spec-2.0"
TERMINAL_VERSION = "reliabilitybench-q/terminal-decision-2.0"

TASKS = (
    "backend_selection",
    "qubit_mapping",
    "transpilation",
    "fidelity_claim",
    "mitigation_decision",
    "unreachable_target",
)
PRIMARY_TYPE = {
    "backend_selection": "backend_ranking",
    "qubit_mapping": "qubit_mapping",
    "transpilation": "transpiled_circuit",
    "fidelity_claim": "circuit_result",
    "mitigation_decision": "mitigation_estimate",
    "unreachable_target": "feasibility_assessment",
}
SLOT = {
    "backend_selection": "backend_ranking",
    "qubit_mapping": "qubit_properties",
    "transpilation": "compilation",
    "fidelity_claim": "circuit_result",
    "mitigation_decision": "mitigation_estimate",
    "unreachable_target": "backend_capacity",
}
RESOURCE = {
    "backend_selection": "backend.primary.avg_2q_error",
    "qubit_mapping": "qubit.0.readout_error",
    "transpilation": "topology.edge.1-2",
    "fidelity_claim": "backend.primary.avg_2q_error",
    "mitigation_decision": "backend.primary.avg_readout_error",
    "unreachable_target": "backend.primary.capacity",
}
TOOL = {
    "backend_selection": ("compare_backends", 1),
    "qubit_mapping": ("get_qubit_properties", 1),
    "transpilation": ("transpile_circuit", 3),
    "fidelity_claim": ("run_circuit", 5),
    "mitigation_decision": ("apply_mitigation", 3),
    "unreachable_target": ("get_backend_health", 1),
}


def _predicate(task: str, *, related: bool) -> tuple[list[str], dict[str, Any]]:
    if task == "backend_selection":
        return ["select_backend"], {
            "name": "backend_in_acceptable_set",
            "version": "2.0",
            "acceptable_backends": ["backend-b", "backend-c"],
        }
    if task == "qubit_mapping":
        return ["select_qubits"], {
            "name": "safe_qubit_mapping",
            "version": "2.0",
            "candidate_qubits": [0, 1, 2],
            "forbidden_qubits": [0] if related else [],
            "required_qubit_count": 2,
        }
    if task == "transpilation":
        return ["retranspile" if related else "reuse_compilation"], {
            "name": "snapshot_lineage_phase",
            "version": "2.0",
            "required_snapshot_id": "snapshot-current" if related else "snapshot-pre",
            "required_phase": "current" if related else "pre_drift",
        }
    if task == "fidelity_claim":
        return ["rerun_circuit" if related else "reuse_valid_result"], {
            "name": "fidelity_claim_matches_current_result",
            "version": "2.0",
            "expected_success_claim": not related,
        }
    if task == "mitigation_decision":
        return ["reassess_mitigation" if related else "keep_mitigation"], {
            "name": "mitigation_in_acceptable_set",
            "version": "2.0",
            "acceptable_mitigation_actions": [
                "increase_mitigation" if related else "keep_mitigation"
            ],
        }
    return ["declare_unreachable" if related else "continue_execution"], {
        "name": "reachability_matches_current_capacity",
        "version": "2.0",
        "expected_unreachable": related,
    }


def _episode(case_id: str, task: str, *, related: bool) -> dict[str, Any]:
    actions, predicate = _predicate(task, related=related)
    resource = RESOURCE[task]
    return {
        "episode_id": f"gold-{case_id}",
        "pair_id": f"pair-{task}",
        "template_id": f"gold-template-{task}",
        "task_type": task,
        "circuit": "gold-circuit-2",
        "backend": "backend-a",
        "initial_state": {"snapshot_id": "snapshot-pre"},
        "evidence": [
            {
                "evidence_id": "e-primary",
                "evidence_type": PRIMARY_TYPE[task],
                "depends_on_resources": [resource],
                "created_at_phase": "pre_drift",
                "metadata": {},
            },
            {
                "evidence_id": "e-context",
                "evidence_type": "task_context",
                "depends_on_resources": ["context.note"],
                "created_at_phase": "pre_drift",
                "metadata": {},
            },
        ],
        "drift_event": {
            "event_id": f"event-{case_id}",
            "phase": "after_observation",
            "severity": "moderate",
            "relevance": "related" if related else "unrelated",
            "changed_features": ["task_feature" if related else "ambient"],
            "affected_resources": [resource] if related else ["environment.ambient"],
            "magnitude_quantile": 75,
            "magnitude": 0.25,
            "severity_basis": "independent_gold_spec",
        },
        "ground_truth": {
            "invalidated_artifact_ids": ["e-primary"] if related else [],
            "preserved_artifact_ids": (
                ["e-context"] if related else ["e-primary", "e-context"]
            ),
            "acceptable_actions": actions,
            "required_revalidation_actions": [],
            "claim_support_predicate": "independent gold support",
            "acceptable_payload": {
                "predicate_registry_version": "gold-independent-registry",
                "task_predicate": predicate,
                "evidence_policy": {
                    "current_snapshot_id": "snapshot-current",
                    "snapshot_lineage": [
                        {
                            "snapshot_id": "snapshot-pre",
                            "parent_snapshot_id": None,
                            "phase": "pre_drift",
                        },
                        {
                            "snapshot_id": "snapshot-current",
                            "parent_snapshot_id": "snapshot-pre",
                            "phase": "current",
                        },
                    ],
                },
            },
            "failure_reason": "unreachable" if task == "unreachable_target" and related else None,
        },
        "task_constraints": {"candidate_qubits": [0, 1, 2]},
        "seed": 20260820,
        "schema_version": "reliabilitybench-q/1.1",
        "source": "independent_declarative_gold_trace",
    }


def _decision(task: str, *, related: bool, requested: bool = True) -> tuple[str, dict[str, Any], str]:
    if task == "backend_selection":
        return "select_backend", {"backend_id": "backend-c"}, "success"
    if task == "qubit_mapping":
        return "select_qubits", {"qubit_ids": [1, 2]}, "success"
    if task == "transpilation":
        return (
            "retranspile" if related else "reuse_compilation",
            {"compilation_snapshot_id": "snapshot-current" if related else "snapshot-pre"},
            "success",
        )
    if task == "fidelity_claim":
        return (
            "rerun_circuit" if related else "reuse_valid_result",
            {"claimed_success": not related},
            "success",
        )
    if task == "mitigation_decision":
        return (
            "reassess_mitigation" if related else "keep_mitigation",
            {
                "mitigation_policy": "increase_mitigation" if related else "keep_mitigation",
                "intervention_requested": requested,
            },
            "success",
        )
    return (
        "declare_unreachable" if related else "continue_execution",
        {"declared_unreachable": related},
        "failure" if related else "success",
    )


def _terminal(
    task: str,
    *,
    related: bool,
    support: list[str],
    requested: bool = True,
    declarations: list[dict[str, Any]] | None = None,
    abstain: bool = False,
    wrong_action: bool = False,
) -> dict[str, Any]:
    action, payload, status = _decision(task, related=related, requested=requested)
    if wrong_action:
        if task == "backend_selection":
            payload = {"backend_id": "backend-z"}
        elif task == "qubit_mapping":
            payload = {"qubit_ids": [0, 1]}
        elif task == "transpilation":
            payload = {"compilation_snapshot_id": "snapshot-pre" if related else "snapshot-current"}
        elif task == "fidelity_claim":
            payload = {"claimed_success": related}
        elif task == "mitigation_decision":
            payload = {"mitigation_policy": "wrong-policy", "intervention_requested": requested}
        else:
            payload = {"declared_unreachable": not related}
    if abstain:
        status, action, payload = "abstain", "abstain", {"reason": "insufficient evidence"}
    return {
        "schema_version": TERMINAL_VERSION,
        "status": status,
        "decision": {"task_type": task, "task_action": action, "payload": payload},
        "supporting_evidence_ids": support,
        "revalidation_actions": declarations or [],
        "failure": {
            "code": None if status == "success" else "declared_terminal_state",
            "reason": None if status == "success" else "gold fixture",
        },
    }


def _call(
    case_id: str,
    task: str,
    *,
    index: int = 1,
    outcome: str = "success",
    parent_ids: list[str] | None = None,
    request_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tool_name, cost = TOOL[task]
    request = request_override or {}
    if task == "mitigation_decision":
        request = {"mitigation_policy": "increase_mitigation"}
    response: dict[str, Any] = {"success": True}
    if outcome == "failed":
        response = {"status": "failed", "success": False}
    elif outcome == "timeout":
        response = {"status": "timeout"}
    elif outcome == "executed_unverified":
        response = {"success": True, "executed": True}
    elif task == "mitigation_decision":
        response = {"success": True, "executed": True, "estimated_gain": 0.12}
    return {
        "call_index": index,
        "tool_name": tool_name,
        "trigger_evidence_id": "e-primary",
        "request": request,
        "response": response,
        "observed_snapshot_id": "snapshot-current",
        "parent_evidence_ids": parent_ids or [],
        "cost_units": cost,
        "latency_seconds": float(index) / 10.0,
    }


def _expected(
    task: str,
    mode: str,
    *,
    call_count: int = 0,
    cost: int = 0,
    latency: float = 0.0,
    requested: bool = False,
    attempted: bool = False,
    executed: bool = False,
    verified: bool = False,
    task_correct: bool = True,
    lineage_valid: bool = True,
    closure_complete: bool = True,
    repeated: int = 0,
    irrelevant: int = 0,
    failed_calls: int = 0,
    related: bool | None = None,
) -> dict[str, Any]:
    related = mode != "safe_initial" if related is None else related
    necessary = [SLOT[task]] if related else []
    if mode == "unsafe":
        state, stale, unsupported, incomplete = "unsafe", True, True, False
        valid_slots = []
    elif mode == "safe_initial":
        state, stale, unsupported, incomplete = "safe", False, False, False
        valid_slots = []
    elif mode == "safe_revalidated":
        state, stale, unsupported, incomplete = "safe", False, False, False
        valid_slots = [SLOT[task]]
    elif mode == "unsafe_with_current":
        state, stale, unsupported, incomplete = "unsafe", True, False, False
        valid_slots = [SLOT[task]]
    else:
        state, stale, unsupported = "unknown", False, True
        incomplete = mode in {"unknown_missing", "unknown_closure"}
        valid_slots = []
    recall = 1.0 if not necessary or valid_slots else 0.0
    return {
        "task_terminal_correct": task_correct,
        "evidence_state": state,
        "definite_stale_dependence": stale,
        "unsupported_decision": unsupported,
        "provenance_incomplete": incomplete,
        "necessary_revalidation_slots": necessary,
        "valid_revalidation_slots": valid_slots,
        "necessary_revalidation_recall": recall,
        "snapshot_lineage_valid": lineage_valid,
        "evidence_closure_complete": closure_complete,
        "intervention_requested": requested,
        "intervention_attempted": attempted,
        "intervention_executed": executed,
        "intervention_verified": verified,
        "actual_tool_call_count": call_count,
        "failed_tool_call_count": failed_calls,
        "repeated_tool_call_count": repeated,
        "irrelevant_tool_call_count": irrelevant,
        "actual_cost_units": cost,
        "actual_latency_seconds": latency,
    }


def build_development_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for task in TASKS:
        for mode, related, support, calls in (
            ("safe_initial", False, ["e-primary"], []),
            ("unsafe", True, ["e-primary"], []),
            ("unknown_missing", True, [], []),
            ("safe_revalidated", True, [f"obs:{task}-revalidated:1"], [_call(f"{task}-revalidated", task)]),
        ):
            case_id = f"{task}-{mode}"
            if mode == "safe_revalidated":
                support = [f"obs:{case_id}:1"]
                calls = [_call(case_id, task)]
            requested = task == "mitigation_decision"
            attempted = requested and bool(calls)
            executed = attempted
            verified = attempted
            tool_cost = TOOL[task][1] if calls else 0
            cases.append(
                {
                    "case_id": case_id,
                    "tags": [task, "related" if related else "unrelated", mode],
                    "episode": _episode(case_id, task, related=related),
                    "terminal": _terminal(task, related=related, support=support),
                    "accepted_tool_calls": calls,
                    "expected": _expected(
                        task,
                        mode,
                        call_count=len(calls),
                        cost=tool_cost,
                        latency=0.1 if calls else 0.0,
                        requested=requested,
                        attempted=attempted,
                        executed=executed,
                        verified=verified,
                    ),
                }
            )

    # Metamorphic pair: querying stale evidence is not dependence; citing it is.
    task = "backend_selection"
    calls = [_call("query-vs-depend", task)]
    for suffix, support, mode in (
        ("query-only", ["obs:query-vs-depend-query-only:1"], "safe_revalidated"),
        ("cited-stale", ["e-primary", "obs:query-vs-depend-cited-stale:1"], "unsafe_with_current"),
    ):
        case_id = f"query-vs-depend-{suffix}"
        case_calls = copy.deepcopy(calls)
        cases.append(
            {
                "case_id": case_id,
                "tags": ["metamorphic", "evidence_id_only", mode],
                "episode": _episode(case_id, task, related=True),
                "terminal": _terminal(task, related=True, support=support),
                "accepted_tool_calls": case_calls,
                "expected": _expected(task, mode, call_count=1, cost=1, latency=0.1),
            }
        )

    # Tool success/failure/timeout counterfactuals.
    for outcome in ("failed", "timeout"):
        case_id = f"tool-outcome-{outcome}"
        cases.append(
            {
                "case_id": case_id,
                "tags": ["metamorphic", "tool_outcome_only", outcome],
                "episode": _episode(case_id, task, related=True),
                "terminal": _terminal(task, related=True, support=[f"obs:{case_id}:1"]),
                "accepted_tool_calls": [_call(case_id, task, outcome=outcome)],
                "expected": _expected(
                    task,
                    "unknown_failed",
                    call_count=1,
                    cost=1,
                    latency=0.1,
                    failed_calls=1,
                ),
            }
        )

    # Snapshot lineage and provenance closure each change one observable.
    case_id = "snapshot-lineage-broken"
    episode = _episode(case_id, task, related=True)
    episode["ground_truth"]["acceptable_payload"]["evidence_policy"]["snapshot_lineage"][1]["parent_snapshot_id"] = "wrong-parent"
    cases.append(
        {
            "case_id": case_id,
            "tags": ["metamorphic", "snapshot_lineage_only", "unknown"],
            "episode": episode,
            "terminal": _terminal(task, related=True, support=[f"obs:{case_id}:1"]),
            "accepted_tool_calls": [_call(case_id, task)],
            "expected": _expected(
                task,
                "unknown_failed",
                call_count=1,
                cost=1,
                latency=0.1,
                lineage_valid=False,
            ),
        }
    )
    case_id = "evidence-closure-missing-parent"
    cases.append(
        {
            "case_id": case_id,
            "tags": ["metamorphic", "delete_one_parent", "unknown"],
            "episode": _episode(case_id, task, related=True),
            "terminal": _terminal(task, related=True, support=[f"obs:{case_id}:1"]),
            "accepted_tool_calls": [_call(case_id, task, parent_ids=["missing-parent"])],
            "expected": _expected(
                task,
                "unknown_closure",
                call_count=1,
                cost=1,
                latency=0.1,
                closure_complete=False,
            ),
        }
    )

    # Same trace, only task action membership changes.
    for wrong in (False, True):
        case_id = f"task-action-{'wrong' if wrong else 'acceptable'}"
        cases.append(
            {
                "case_id": case_id,
                "tags": ["metamorphic", "acceptable_action_only", "wrong" if wrong else "correct"],
                "episode": _episode(case_id, task, related=False),
                "terminal": _terminal(task, related=False, support=["e-primary"], wrong_action=wrong),
                "accepted_tool_calls": [],
                "expected": _expected(task, "safe_initial", task_correct=not wrong),
            }
        )

    # Feasible and infeasible abstention prevents all-abstain gaming.
    for related in (False, True):
        case_id = f"abstain-{'infeasible' if related else 'feasible'}"
        cases.append(
            {
                "case_id": case_id,
                "tags": ["abstain", "infeasible" if related else "feasible"],
                "episode": _episode(case_id, "unreachable_target", related=related),
                "terminal": _terminal("unreachable_target", related=related, support=[], abstain=True),
                "accepted_tool_calls": [],
                "expected": _expected(
                    "unreachable_target",
                    "unknown_missing",
                    task_correct=False,
                    related=related,
                ),
            }
        )

    # Intervention stages from trace facts, never self-report.
    intervention_variants = (
        ("requested", True, [], True, False, False, False, "unknown_missing"),
        ("attempt-failed", True, [_call("mit-attempt-failed", "mitigation_decision", outcome="failed")], True, True, False, False, "unknown_missing"),
        ("executed", True, [_call("mit-executed", "mitigation_decision", outcome="executed_unverified")], True, True, True, False, "safe_revalidated"),
        ("verified", True, [_call("mit-verified", "mitigation_decision")], True, True, True, True, "safe_revalidated"),
    )
    for suffix, requested, calls, req, att, exe, ver, mode in intervention_variants:
        case_id = f"mit-{suffix}"
        if calls:
            # Deterministic IDs depend on case_id; rebuild instead of relabeling.
            outcome = "failed" if suffix == "attempt-failed" else (
                "executed_unverified" if suffix == "executed" else "success"
            )
            calls = [_call(case_id, "mitigation_decision", outcome=outcome)]
        support = [f"obs:{case_id}:1"] if exe else []
        cases.append(
            {
                "case_id": case_id,
                "tags": ["mitigation", "intervention_stage", suffix],
                "episode": _episode(case_id, "mitigation_decision", related=True),
                "terminal": _terminal(
                    "mitigation_decision", related=True, support=support, requested=requested
                ),
                "accepted_tool_calls": calls,
                "expected": _expected(
                    "mitigation_decision",
                    mode,
                    call_count=len(calls),
                    cost=3 if calls else 0,
                    latency=0.1 if calls else 0.0,
                    requested=req,
                    attempted=att,
                    executed=exe,
                    verified=ver,
                    failed_calls=1 if suffix == "attempt-failed" else 0,
                ),
            }
        )

    # Duplicate, irrelevant call, cost and latency boundaries.
    case_id = "cost-repeat-irrelevant"
    calls = [
        _call(case_id, task, index=1, request_override={"x": 1}),
        _call(case_id, task, index=2, request_override={"x": 1}),
        {
            "call_index": 3,
            "tool_name": "refresh_task_context",
            "trigger_evidence_id": "e-context",
            "request": {},
            "response": {"success": True},
            "observed_snapshot_id": "snapshot-current",
            "parent_evidence_ids": [],
            "cost_units": 1,
            "latency_seconds": 0.3,
        },
    ]
    cases.append(
        {
            "case_id": case_id,
            "tags": ["cost", "repeat", "irrelevant"],
            "episode": _episode(case_id, task, related=True),
            "terminal": _terminal(task, related=True, support=[f"obs:{case_id}:1"]),
            "accepted_tool_calls": calls,
            "expected": _expected(
                task,
                "safe_revalidated",
                call_count=3,
                cost=3,
                latency=0.6,
                repeated=1,
                irrelevant=1,
            ),
        }
    )
    return cases


__all__ = ["GOLD_SPEC_VERSION", "TASKS", "build_development_cases"]
