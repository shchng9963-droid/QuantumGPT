from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from eval.reliability_bench.action_guard_contract_v1 import (
    AcceptedToolCall,
    CandidateFinalAction,
    CandidateToolAction,
    EvidenceValidity,
    GuardOutcome,
    LedgerEvidence,
    PublicTaskContract,
    ReasonCode,
    VisibleEvidenceLedger,
)
from eval.reliability_bench.action_guard_runtime_v1 import GuardRuntimeSession
from eval.reliability_bench.action_guard_v1 import ActionGuard
from eval.reliability_bench.recovery_planner_v1 import (
    RecoveryCandidate,
    minimum_cost_recovery_plan,
)
from eval.reliability_bench.temporal_evidence_v1 import (
    EvidenceDependency,
    dependency_scoped_validity,
    ledger_only_visible_state,
)
from eval.reliability_bench.terminal_schema_v2 import (
    TERMINAL_SCHEMA_VERSION,
    parse_terminal_decision,
)


ROOT = Path(__file__).resolve().parents[2]


def _task(task_type: str = "backend_selection") -> PublicTaskContract:
    return PublicTaskContract(
        task_id="task-public-1",
        task_type=task_type,
        backend_ids=("backend-a", "backend-b"),
        qubit_ids=(0, 1, 2),
        circuit_ids=("circuit-a",),
        snapshot_ids=("snapshot-pre", "snapshot-current"),
        mitigation_policy_ids=("keep", "increase"),
    )


def _ledger(
    validity: EvidenceValidity = EvidenceValidity.VALID,
    *,
    slot: str = "backend_ranking",
    include_current: bool = False,
) -> VisibleEvidenceLedger:
    records = [
        LedgerEvidence("e-primary", (slot,), validity, "public-ledger")
    ]
    if include_current:
        records.append(
            LedgerEvidence(
                "obs:guard:1",
                (slot,),
                EvidenceValidity.VALID,
                "accepted-tool-call",
                produced_by_tool_call_id="call:1",
            )
        )
    return VisibleEvidenceLedger(tuple(records))


def _terminal(
    *,
    task_type: str = "backend_selection",
    support=("e-primary",),
    backend_id: str = "backend-b",
    status: str = "success",
    declarations=(),
):
    actions = {
        "backend_selection": ("select_backend", {"backend_id": backend_id}),
        "qubit_mapping": ("select_qubits", {"qubit_ids": [1, 2]}),
        "transpilation": (
            "retranspile",
            {"compilation_snapshot_id": "snapshot-current"},
        ),
        "fidelity_claim": ("rerun_circuit", {"claimed_success": True}),
        "mitigation_decision": (
            "reassess_mitigation",
            {"mitigation_policy": "increase", "intervention_requested": True},
        ),
        "unreachable_target": (
            "declare_unreachable",
            {"declared_unreachable": True},
        ),
    }
    action, payload = actions[task_type]
    raw = {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "status": status,
        "decision": {
            "task_type": task_type,
            "task_action": action,
            "payload": payload,
        },
        "supporting_evidence_ids": list(support),
        "revalidation_actions": list(declarations),
        "failure": {
            "code": None if status == "success" else "public_failure",
            "reason": None if status == "success" else "public reason",
        },
    }
    return CandidateFinalAction(parse_terminal_decision(raw))


VALID_TOOL_ACTIONS = (
    CandidateToolAction("get_backend_health", {"backend_id": "backend-a"}),
    CandidateToolAction(
        "compare_backends", {"backend_ids": ["backend-a", "backend-b"]}
    ),
    CandidateToolAction(
        "get_qubit_properties",
        {"backend_id": "backend-a", "qubit_ids": [0, 1]},
    ),
    CandidateToolAction("get_coupling_map", {"backend_id": "backend-a"}),
    CandidateToolAction(
        "transpile_circuit",
        {"circuit_id": "circuit-a", "backend_id": "backend-a"},
    ),
    CandidateToolAction(
        "run_circuit", {"circuit_id": "circuit-a", "backend_id": "backend-a"}
    ),
    CandidateToolAction("apply_mitigation", {"mitigation_policy": "increase"}),
    CandidateToolAction("refresh_task_context", {"task_id": "task-public-1"}),
)


@pytest.mark.parametrize("candidate", VALID_TOOL_ACTIONS)
def test_action_formation_all_public_tool_schemas_allow_valid_actions(candidate):
    result = ActionGuard().check_tool_action(
        candidate, task=_task(), ledger=_ledger()
    )
    assert result.outcome is GuardOutcome.ALLOW


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (CandidateToolAction("not-a-tool", {}), ReasonCode.INVALID_SCHEMA),
        (
            CandidateToolAction("get_backend_health", {}),
            ReasonCode.INVALID_SCHEMA,
        ),
        (
            CandidateToolAction(
                "get_backend_health", {"backend_id": "backend-a", "extra": 1}
            ),
            ReasonCode.INVALID_SCHEMA,
        ),
        (
            CandidateToolAction("get_backend_health", {"backend_id": 3}),
            ReasonCode.INVALID_SCHEMA,
        ),
        (
            CandidateToolAction("get_backend_health", {"backend_id": "ghost"}),
            ReasonCode.UNKNOWN_ENTITY,
        ),
        (
            CandidateToolAction(
                "get_backend_health",
                {"backend_id": "backend-a"},
                trigger_evidence_id="forged",
            ),
            ReasonCode.INVALID_REFERENCE,
        ),
    ],
)
def test_action_formation_and_referential_integrity_block_only_abstractly(
    candidate, reason
):
    result = ActionGuard().check_tool_action(
        candidate, task=_task(), ledger=_ledger()
    )
    assert result.outcome is GuardOutcome.REPAIR_REQUEST
    assert result.reason_code is reason
    assert "backend-b" not in result.violated_public_constraint


def test_safe_finalization_allows_valid_supported_action_without_mutation():
    candidate = _terminal()
    before = candidate
    result = ActionGuard().check_final_action(
        candidate, task=_task(), ledger=_ledger(), accepted_tool_calls=[]
    )
    assert result.outcome is GuardOutcome.ALLOW
    assert result.reason_code is None
    assert candidate == before


@pytest.mark.parametrize(
    ("support", "validity", "slot", "reason"),
    [
        ((), EvidenceValidity.VALID, "backend_ranking", ReasonCode.PREMATURE_FINALIZATION),
        (("forged",), EvidenceValidity.VALID, "backend_ranking", ReasonCode.INVALID_REFERENCE),
        (("e-primary",), EvidenceValidity.INVALID, "backend_ranking", ReasonCode.STALE_CITED_EVIDENCE),
        (("e-primary",), EvidenceValidity.UNKNOWN, "backend_ranking", ReasonCode.MISSING_REQUIRED_SUPPORT),
        (("e-primary",), EvidenceValidity.VALID, "task_context", ReasonCode.MISSING_REQUIRED_SUPPORT),
    ],
)
def test_evidence_preconditions_and_safe_finalization(
    support, validity, slot, reason
):
    result = ActionGuard().check_final_action(
        _terminal(support=support),
        task=_task(),
        ledger=_ledger(validity, slot=slot),
        accepted_tool_calls=[],
    )
    assert result.reason_code is reason
    assert result.outcome is not GuardOutcome.ALLOW


@pytest.mark.parametrize(
    ("task_type", "payload_change"),
    [
        ("backend_selection", {"backend_id": "ghost"}),
        ("qubit_mapping", {"qubit_ids": [1, 99]}),
        ("transpilation", {"compilation_snapshot_id": "ghost"}),
        ("mitigation_decision", {"mitigation_policy": "ghost", "intervention_requested": True}),
    ],
)
def test_final_referential_integrity_rejects_unknown_entities(task_type, payload_change):
    candidate = _terminal(task_type=task_type)
    changed = replace(candidate.terminal, payload=payload_change)
    result = ActionGuard().check_final_action(
        CandidateFinalAction(changed),
        task=_task(task_type),
        ledger=_ledger(slot=_task(task_type).required_slots[0]),
        accepted_tool_calls=[],
    )
    assert result.reason_code is ReasonCode.UNKNOWN_ENTITY


def _revalidation_candidate(
    call_id="call:1", output_id="obs:guard:1", support=("obs:guard:1",)
):
    return _terminal(
        support=support,
        declarations=(
            {
                "tool_call_id": call_id,
                "target_evidence_ids": ["e-primary"],
                "output_evidence_ids": [output_id],
            },
        ),
    )


def _real_call(successful=True, output_id="obs:guard:1"):
    return AcceptedToolCall(
        "call:1", "compare_backends", {"backend_ids": ["backend-a", "backend-b"]},
        successful, (output_id,)
    )


@pytest.mark.parametrize(
    ("candidate", "calls", "ledger", "reason"),
    [
        (_revalidation_candidate(call_id="call:999"), (_real_call(),), _ledger(include_current=True), ReasonCode.INVALID_REFERENCE),
        (_revalidation_candidate(output_id="obs:forged"), (_real_call(),), _ledger(include_current=True), ReasonCode.UNVERIFIED_REVALIDATION),
        (_revalidation_candidate(), (_real_call(False),), _ledger(include_current=True), ReasonCode.UNVERIFIED_REVALIDATION),
        (
            _revalidation_candidate(support=("e-primary",)),
            (_real_call(),),
            _ledger(),
            ReasonCode.UNVERIFIED_REVALIDATION,
        ),
    ],
)
def test_revalidation_integrity_rejects_unbound_or_unverified_claims(
    candidate, calls, ledger, reason
):
    result = ActionGuard().check_final_action(
        candidate, task=_task(), ledger=ledger, accepted_tool_calls=calls
    )
    assert result.reason_code is reason


def test_revalidation_integrity_allows_exact_real_trace_binding():
    result = ActionGuard().check_final_action(
        _revalidation_candidate(),
        task=_task(),
        ledger=_ledger(EvidenceValidity.INVALID, include_current=True),
        accepted_tool_calls=(_real_call(),),
    )
    assert result.outcome is GuardOutcome.ALLOW


def test_abstain_is_visible_but_not_forced_to_fabricate_support():
    raw = {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "status": "abstain",
        "decision": {
            "task_type": "backend_selection",
            "task_action": "abstain",
            "payload": {"reason": "public uncertainty"},
        },
        "supporting_evidence_ids": [],
        "revalidation_actions": [],
        "failure": {"code": "abstained", "reason": "public uncertainty"},
    }
    result = ActionGuard().check_final_action(
        CandidateFinalAction(parse_terminal_decision(raw)),
        task=_task(),
        ledger=_ledger(EvidenceValidity.UNKNOWN),
        accepted_tool_calls=[],
    )
    assert result.outcome is GuardOutcome.ALLOW


def test_component_boundary_same_candidate_uses_only_supplied_ledger_state():
    dependency = (
        EvidenceDependency(
            "e-primary", ("backend_ranking",), ("backend.avg_2q_error",), "pre"
        ),
    )
    candidate = _terminal()
    guard = ActionGuard()
    ledger_state = ledger_only_visible_state(dependency)
    full_unrelated = dependency_scoped_validity(
        dependency, affected_resources=("environment.temperature",)
    )
    full_related = dependency_scoped_validity(
        dependency, affected_resources=("backend.avg_2q_error",)
    )
    assert guard.check_final_action(
        candidate, task=_task(), ledger=ledger_state, accepted_tool_calls=[]
    ).reason_code is ReasonCode.MISSING_REQUIRED_SUPPORT
    assert guard.check_final_action(
        candidate, task=_task(), ledger=full_unrelated, accepted_tool_calls=[]
    ).outcome is GuardOutcome.ALLOW
    assert guard.check_final_action(
        candidate, task=_task(), ledger=full_related, accepted_tool_calls=[]
    ).reason_code is ReasonCode.STALE_CITED_EVIDENCE


def test_unrelated_drift_preserves_non_interference():
    dependency = (
        EvidenceDependency("e-primary", ("backend_ranking",), ("backend.error",), "pre"),
    )
    ledger = dependency_scoped_validity(
        dependency, affected_resources=("room.temperature",)
    )
    candidate = _terminal()
    before = candidate
    result = ActionGuard().check_final_action(
        candidate, task=_task(), ledger=ledger, accepted_tool_calls=[]
    )
    assert result.outcome is GuardOutcome.ALLOW
    assert candidate == before


def test_guard_result_is_invariant_to_hidden_correct_answer_not_in_input():
    candidate = _terminal(backend_id="backend-a")
    first = ActionGuard().check_final_action(
        candidate, task=_task(), ledger=_ledger(), accepted_tool_calls=[]
    )
    second = ActionGuard().check_final_action(
        candidate, task=_task(), ledger=_ledger(), accepted_tool_calls=[]
    )
    assert first == second
    assert first.outcome is GuardOutcome.ALLOW


def test_recovery_planner_is_separate_and_minimizes_abstract_slot_cost():
    plan = minimum_cost_recovery_plan(
        ("slot-a", "slot-b"),
        (
            RecoveryCandidate("expensive-combined", ("slot-a", "slot-b"), 7),
            RecoveryCandidate("recover-a", ("slot-a",), 2),
            RecoveryCandidate("recover-b", ("slot-b",), 3),
        ),
    )
    assert plan.feasible is True
    assert plan.action_ids == ("recover-a", "recover-b")
    assert plan.total_cost_units == 5


def test_guard_dependency_isolation_ast_audit():
    source = (ROOT / "eval/reliability_bench/action_guard_v1.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
    forbidden = (
        "measurement_spec",
        "task_predicate",
        "trace_evidence_evaluator",
        "judge",
        "oracle",
        "ground_truth",
        "recovery_planner",
        "temporal_evidence",
    )
    assert not any(token in name.lower() for name in imports for token in forbidden)
    lowered = source.lower()
    assert "acceptable_actions" not in lowered
    assert "ground_truth" not in lowered
    assert "oracle" not in lowered


def test_repair_loop_has_fixed_budget_and_preserves_failed_attempts():
    session = GuardRuntimeSession()
    invalid = CandidateToolAction("get_backend_health", {"backend_id": "ghost"})
    for turn in (1, 2, 3):
        session.check_tool(
            invalid,
            task=_task(),
            ledger=_ledger(),
            llm_turn_count=turn,
            tool_call_count=0,
        )
    assert [item.repair_allowed for item in session.events] == [True, True, False]
    assert [item.repair_attempt_index for item in session.events] == [1, 2, None]
    assert session.overhead()["guard_check_count"] == 3
    assert session.overhead()["guard_intervention_count"] == 3
    trace = json.loads(session.trace_json())
    assert len(trace) == 3
    assert set(trace[0]["feedback"]) == {
        "guard_outcome",
        "reason_code",
        "violated_public_constraint",
    }
    assert "recommended_action" not in session.trace_json()


def test_measurement_v2_frozen_source_hashes_unchanged():
    manifest = json.loads(
        (ROOT / "eval/reliability_bench/data/measurement_v2_freeze_manifest.json")
        .read_text(encoding="utf-8")
    )
    for relative, expected in manifest["frozen_source_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected
    assert manifest["heldout_gold_v2_1"]["total_exact_comparisons"] == 1197
    assert manifest["method_validation_v2_1_unseal_allowed"] is False
