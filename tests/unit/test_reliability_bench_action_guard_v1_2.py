from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

from eval.reliability_bench.action_guard_contract_v1_2 import (
    AcceptedToolCall,
    BeliefValidity,
    CandidateFinalAction,
    ControllerBeliefLedger,
    ControllerEvidenceBelief,
    GuardOutcome,
    PublicTaskContract,
    ReasonCode,
)
from eval.reliability_bench.action_guard_v1_2 import ActionGuard
from eval.reliability_bench.generator_v2 import (
    GeneratorV2Config,
    generate_method_validation_candidates,
)
from eval.reliability_bench.task_predicate_evaluator_v3 import evaluate_task_decision
from eval.reliability_bench.terminal_schema_v3 import (
    TERMINAL_SCHEMA_VERSION,
    parse_terminal_decision,
)
from eval.reliability_bench.temporal_evidence_v1_2 import (
    EvidenceBeliefDependency,
    dependency_scoped_belief_update,
    ledger_only_belief_state,
    register_revalidated_belief,
)
from eval.reliability_bench.trace_evidence_evaluator_v3 import evaluate_trace_evidence


SLOTS = {
    "backend_ranking": ("backend_ranking",),
    "qubit_mapping": ("qubit_properties",),
    "transpiled_circuit": ("compilation",),
    "circuit_result": ("circuit_result",),
    "mitigation_estimate": ("mitigation_estimate",),
    "feasibility_assessment": ("backend_capacity",),
    "task_context": ("task_context",),
}


def _backend_pair():
    episodes = generate_method_validation_candidates(
        GeneratorV2Config(random_seed=20260911)
    )
    selected = [item for item in episodes if item.task_type.value == "backend_selection"]
    related = next(item for item in selected if item.drift_event.relevance.value == "related")
    unrelated = next(
        item
        for item in selected
        if item.pair_id == related.pair_id
        and item.drift_event.relevance.value == "unrelated"
    )
    return related, unrelated


def _dependencies(episode):
    return tuple(
        EvidenceBeliefDependency(
            item.evidence_id,
            SLOTS[item.evidence_type],
            item.depends_on_resources,
            "episode-initial",
        )
        for item in episode.evidence
    )


def _task(episode) -> PublicTaskContract:
    policy = episode.ground_truth.acceptable_payload["evidence_policy"]
    return PublicTaskContract(
        task_id=episode.episode_id,
        task_type=episode.task_type.value,
        backend_ids=tuple(episode.task_constraints["candidate_backends"]),
        qubit_ids=tuple(episode.task_constraints["candidate_qubits"]),
        circuit_ids=(episode.circuit,),
        snapshot_ids=(
            str(episode.initial_state["snapshot_id"]),
            str(policy["current_snapshot_id"]),
        ),
        mitigation_policy_ids=("keep_mitigation", "increase_mitigation"),
    )


def _terminal(episode, support, declarations=()) -> CandidateFinalAction:
    return CandidateFinalAction(
        parse_terminal_decision(
            {
                "schema_version": TERMINAL_SCHEMA_VERSION,
                "completion_status": "answered",
                "decision": {
                    "task_type": "backend_selection",
                    "task_action": "select_backend",
                    "payload": {
                        "backend_id": episode.ground_truth.acceptable_payload[
                            "task_predicate"
                        ]["acceptable_backends"][0]
                    },
                },
                "supporting_evidence_ids": list(support),
                "revalidation_actions": list(declarations),
                "failure": {"code": None, "reason": None},
            }
        )
    )


def test_ledger_guard_allows_believed_valid_but_objectively_stale_evidence():
    related, _ = _backend_pair()
    primary = related.evidence[0]
    belief = ledger_only_belief_state(_dependencies(related))
    candidate = _terminal(related, (primary.evidence_id,))

    guarded = ActionGuard().check_final_action(
        candidate, task=_task(related), ledger=belief, accepted_tool_calls=()
    )
    measured = evaluate_trace_evidence(
        episode=related,
        decision=candidate.terminal,
        trace_id="ag12-belief-objective-split",
        accepted_tool_calls=(),
    )

    assert guarded.outcome is GuardOutcome.ALLOW
    assert measured.definite_stale_dependence is True
    assert measured.evidence_state == "unsafe"


def test_full_guard_blocks_after_belief_invalidation():
    related, _ = _backend_pair()
    primary = related.evidence[0]
    belief = dependency_scoped_belief_update(
        _dependencies(related),
        changed_features=related.drift_event.changed_features,
    )
    result = ActionGuard().check_final_action(
        _terminal(related, (primary.evidence_id,)),
        task=_task(related),
        ledger=belief,
        accepted_tool_calls=(),
    )
    assert result.outcome is GuardOutcome.BLOCK
    assert result.reason_code is ReasonCode.STALE_CITED_EVIDENCE


def test_unrelated_drift_preserves_belief_and_non_interference():
    _, unrelated = _backend_pair()
    primary = unrelated.evidence[0]
    belief = dependency_scoped_belief_update(
        _dependencies(unrelated),
        changed_features=unrelated.drift_event.changed_features,
    )
    result = ActionGuard().check_final_action(
        _terminal(unrelated, (primary.evidence_id,)),
        task=_task(unrelated),
        ledger=belief,
        accepted_tool_calls=(),
    )
    assert belief.by_id()[primary.evidence_id].belief_validity is BeliefValidity.BELIEVED_VALID
    assert result.outcome is GuardOutcome.ALLOW


def test_new_revalidation_restores_belief_and_objective_validity_for_both_guard_arms():
    related, _ = _backend_pair()
    primary = related.evidence[0]
    trace_id = "ag12-revalidation"
    output_id = f"obs:{trace_id}:1"
    call = AcceptedToolCall(
        "call:1",
        "compare_backends",
        {"backend_ids": list(related.task_constraints["candidate_backends"])},
        True,
        (output_id,),
    )
    declaration = {
        "tool_call_id": "call:1",
        "target_evidence_ids": [primary.evidence_id],
        "output_evidence_ids": [output_id],
    }
    candidate = _terminal(related, (output_id,), (declaration,))
    ledgers = (
        ledger_only_belief_state(_dependencies(related)),
        dependency_scoped_belief_update(
            _dependencies(related),
            changed_features=related.drift_event.changed_features,
        ),
    )
    for initial in ledgers:
        restored = register_revalidated_belief(
            initial,
            evidence_id=output_id,
            slots=("backend_ranking",),
            source="accepted-current-tool-call",
            tool_call_id="call:1",
        )
        result = ActionGuard().check_final_action(
            candidate,
            task=_task(related),
            ledger=restored,
            accepted_tool_calls=(call,),
        )
        assert result.outcome is GuardOutcome.ALLOW

    measured = evaluate_trace_evidence(
        episode=related,
        decision=candidate.terminal,
        trace_id=trace_id,
        accepted_tool_calls=(
            {
                "call_index": 1,
                "tool_name": "compare_backends",
                "request": {
                    "backend_ids": list(related.task_constraints["candidate_backends"])
                },
                "response": {"success": True, "ranking": []},
                "trigger_evidence_id": primary.evidence_id,
                "observed_snapshot_id": related.ground_truth.acceptable_payload[
                    "evidence_policy"
                ]["current_snapshot_id"],
                "cost_units": 1,
                "latency_seconds": 0.1,
            },
        ),
    )
    assert measured.evidence_state == "safe"
    assert measured.definite_stale_dependence is False


def test_guard_is_invariant_to_hidden_actual_validity_and_correct_answer():
    related, unrelated = _backend_pair()
    primary = related.evidence[0]
    belief = ControllerBeliefLedger(
        (
            ControllerEvidenceBelief(
                primary.evidence_id,
                ("backend_ranking",),
                BeliefValidity.BELIEVED_VALID,
                "fixed-visible-belief",
            ),
        )
    )
    candidate = _terminal(related, (primary.evidence_id,))
    guard = ActionGuard()
    first = guard.check_final_action(
        candidate, task=_task(related), ledger=belief, accepted_tool_calls=()
    )
    second = guard.check_final_action(
        candidate, task=_task(related), ledger=belief, accepted_tool_calls=()
    )
    assert first == second

    related_measure = evaluate_trace_evidence(
        episode=related,
        decision=candidate.terminal,
        trace_id="ag12-related",
        accepted_tool_calls=(),
    )
    unrelated_for_same_ids = replace(
        unrelated,
        episode_id=related.episode_id,
        pair_id=related.pair_id,
        evidence=related.evidence,
        initial_state=related.initial_state,
        task_constraints=related.task_constraints,
        circuit=related.circuit,
        ground_truth=replace(
            unrelated.ground_truth,
            acceptable_payload={
                **unrelated.ground_truth.acceptable_payload,
                "evidence_policy": related.ground_truth.acceptable_payload[
                    "evidence_policy"
                ],
            },
        ),
    )
    unrelated_measure = evaluate_trace_evidence(
        episode=unrelated_for_same_ids,
        decision=candidate.terminal,
        trace_id="ag12-unrelated",
        accepted_tool_calls=(),
    )
    assert related_measure.definite_stale_dependence is True
    assert unrelated_measure.definite_stale_dependence is False

    altered_predicate = dict(related.ground_truth.acceptable_payload)
    altered_predicate["task_predicate"] = {
        **altered_predicate["task_predicate"],
        "acceptable_backends": [related.task_constraints["candidate_backends"][-1]],
    }
    changed_answer_episode = replace(
        related,
        ground_truth=replace(related.ground_truth, acceptable_payload=altered_predicate),
    )
    assert evaluate_task_decision(related, candidate.terminal).terminal_correct != (
        evaluate_task_decision(changed_answer_episode, candidate.terminal).terminal_correct
    )
    assert guard.check_final_action(
        candidate, task=_task(related), ledger=belief, accepted_tool_calls=()
    ) == first


def test_v12_runtime_has_no_evaluator_or_objective_truth_imports():
    root = Path(__file__).resolve().parents[2]
    files = (
        root / "eval/reliability_bench/action_guard_contract_v1_2.py",
        root / "eval/reliability_bench/action_guard_v1_2.py",
        root / "eval/reliability_bench/action_guard_runtime_v1_2.py",
        root / "eval/reliability_bench/temporal_evidence_v1_2.py",
    )
    forbidden = {
        "measurement_spec_v3",
        "task_predicate_evaluator_v3",
        "trace_evidence_evaluator_v3",
        "predicates_v2",
        "generator_v2",
    }
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name.rsplit(".", 1)[-1]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        if any(isinstance(node, ast.ImportFrom) and node.module for node in ast.walk(tree)):
            imported.update(
                node.module.rsplit(".", 1)[-1]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            )
        assert not imported.intersection(forbidden)
