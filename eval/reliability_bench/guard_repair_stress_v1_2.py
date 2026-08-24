"""Controlled Guard repair stress cases, separate from natural LLM preflight."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .action_guard_contract_v1_2 import (
    AcceptedToolCall,
    BeliefValidity,
    CandidateFinalAction,
    CandidateToolAction,
    ControllerBeliefLedger,
    ControllerEvidenceBelief,
    PublicTaskContract,
)
from .action_guard_runtime_v1_2 import GuardRuntimeSessionV1_2
from .terminal_schema_v3 import TERMINAL_SCHEMA_VERSION, parse_terminal_decision


STRESS_PROTOCOL_VERSION = "reliabilitybench-q/action-guard-repair-stress-1.2"


def _task() -> PublicTaskContract:
    return PublicTaskContract(
        task_id="guard-stress-backend",
        task_type="backend_selection",
        backend_ids=("backend-a", "backend-b"),
        qubit_ids=(0, 1),
        circuit_ids=("circuit-a",),
        snapshot_ids=("snapshot-pre", "snapshot-current"),
    )


def _ledger(*, validity: BeliefValidity, include_context: bool = True) -> ControllerBeliefLedger:
    records = [
        ControllerEvidenceBelief(
            "e-primary", ("backend_ranking",), validity, "stress-fixture"
        )
    ]
    if include_context:
        records.append(
            ControllerEvidenceBelief(
                "e-context", ("task_context",), BeliefValidity.BELIEVED_VALID, "stress-fixture"
            )
        )
    return ControllerBeliefLedger(tuple(records))


def _final(
    support: list[str],
    declarations: list[dict[str, Any]] | None = None,
) -> CandidateFinalAction:
    return CandidateFinalAction(
        parse_terminal_decision(
            {
                "schema_version": TERMINAL_SCHEMA_VERSION,
                "completion_status": "answered",
                "decision": {
                    "task_type": "backend_selection",
                    "task_action": "select_backend",
                    "payload": {"backend_id": "backend-b"},
                },
                "supporting_evidence_ids": support,
                "revalidation_actions": declarations or [],
                "failure": {"code": None, "reason": None},
            }
        )
    )


def run_controlled_repair_stress() -> dict[str, Any]:
    """Exercise every public reason code followed by a legal repair."""
    task = _task()
    valid = _ledger(validity=BeliefValidity.BELIEVED_VALID)
    invalid = _ledger(validity=BeliefValidity.BELIEVED_INVALID)
    good_tool = CandidateToolAction(
        "compare_backends", {"backend_ids": ["backend-a", "backend-b"]}
    )
    successful_call = AcceptedToolCall(
        "call-1", "compare_backends", {"backend_ids": ["backend-a", "backend-b"]}, True, ("e-new",)
    )
    revalidated = ControllerBeliefLedger(
        valid.records
        + (
            ControllerEvidenceBelief(
                "e-new",
                ("backend_ranking",),
                BeliefValidity.BELIEVED_VALID,
                "accepted-tool-call",
                "call-1",
            ),
        )
    )
    declaration = [{
        "tool_call_id": "call-1",
        "target_evidence_ids": ["e-primary"],
        "output_evidence_ids": ["e-new"],
    }]

    cases: list[tuple[str, str, Any, Any, tuple[AcceptedToolCall, ...], Any, Any, tuple[AcceptedToolCall, ...]]] = [
        ("invalid_schema", "tool", CandidateToolAction("not_a_tool", {}), valid, (), good_tool, valid, ()),
        ("unknown_entity", "tool", CandidateToolAction("compare_backends", {"backend_ids": ["backend-z"]}), valid, (), good_tool, valid, ()),
        ("invalid_reference", "tool", CandidateToolAction("compare_backends", {"backend_ids": ["backend-a", "backend-b"]}, "missing"), valid, (), good_tool, valid, ()),
        ("missing_required_support", "final", _final(["e-context"]), valid, (), _final(["e-primary"]), valid, ()),
        ("stale_cited_evidence", "final", _final(["e-primary"]), invalid, (), _final(["e-new"], declaration), revalidated, (successful_call,)),
        ("unverified_revalidation", "final", _final(["e-new"], declaration), revalidated, (AcceptedToolCall("call-1", "compare_backends", {}, False, ("e-new",)),), _final(["e-new"], declaration), revalidated, (successful_call,)),
        ("premature_finalization", "final", _final([]), valid, (), _final(["e-primary"]), valid, ()),
    ]

    results: list[dict[str, Any]] = []
    for expected_reason, kind, bad, bad_ledger, bad_calls, repaired, repaired_ledger, repaired_calls in cases:
        session = GuardRuntimeSessionV1_2()
        if kind == "tool":
            first = session.check_tool(bad, task=task, ledger=bad_ledger, llm_turn_count=1, tool_call_count=0)
            second = session.check_tool(repaired, task=task, ledger=repaired_ledger, llm_turn_count=2, tool_call_count=0)
        else:
            first = session.check_final(bad, task=task, ledger=bad_ledger, accepted_tool_calls=bad_calls, llm_turn_count=1, tool_call_count=len(bad_calls))
            second = session.check_final(repaired, task=task, ledger=repaired_ledger, accepted_tool_calls=repaired_calls, llm_turn_count=2, tool_call_count=len(repaired_calls))
        results.append(
            {
                "case": expected_reason,
                "first_outcome": first.outcome.value,
                "first_reason_code": first.reason_code.value if first.reason_code else None,
                "repair_feedback": asdict(session.events[0])["feedback"],
                "repaired_outcome": second.outcome.value,
                "events": [asdict(item) for item in session.events],
            }
        )
    return {
        "protocol_version": STRESS_PROTOCOL_VERSION,
        "population": "controlled synthetic candidates; not natural LLM errors",
        "case_count": len(results),
        "results": results,
        "all_reason_codes_covered": all(item["first_reason_code"] == item["case"] for item in results),
        "all_repairs_allowed": all(item["repaired_outcome"] == "ALLOW" for item in results),
    }


__all__ = ["STRESS_PROTOCOL_VERSION", "run_controlled_repair_stress"]
