"""Measurement v3.1 trace evaluation for decisions and runtime failures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .measurement_spec_v3_1 import TASK_REQUIRED_SLOTS
from .runtime_terminal_v1 import RuntimeExecutionFailure
from .schema import Episode
from .terminal_outcome_v3_1 import TerminalOutcome
from .trace_evidence_evaluator_v3 import (
    evaluate_trace_evidence,
    normalize_accepted_tool_calls,
)


TRACE_EVALUATOR_VERSION = "reliabilitybench-q/trace-evidence-evaluator-3.1-candidate"


@dataclass(frozen=True)
class TraceOutcomeEvaluation:
    evaluator_version: str
    decision_support_applicable: bool
    evidence_state: str
    definite_stale_dependence: bool | None
    unsupported_decision: bool | None
    provenance_incomplete: bool | None
    unsafe_or_unknown: bool | None
    supporting_evidence_ids: tuple[str, ...]
    unresolved_supporting_evidence_ids: tuple[str, ...]
    provenance_closure_ids: tuple[str, ...]
    snapshot_lineage_valid: bool | None
    evidence_closure_complete: bool | None
    required_evidence_slots: tuple[str, ...]
    necessary_revalidation_slots: tuple[str, ...]
    valid_revalidation_slots: tuple[str, ...]
    adopted_revalidation_slots: tuple[str, ...]
    necessary_revalidation_recall: float | None
    valid_revalidation_adoption_rate: float | None
    declaration_mismatches: tuple[str, ...]
    unsupported_success_claim: bool | None
    intervention_requested: bool | None
    intervention_attempted: bool
    intervention_executed: bool
    intervention_verified: bool
    actual_tool_call_count: int
    failed_tool_call_count: int
    repeated_tool_call_count: int
    irrelevant_tool_call_count: int
    actual_cost_units: int
    actual_latency_seconds: float
    runtime_failure_reason: str | None


def _current_snapshot_id(episode: Episode) -> str:
    value = episode.ground_truth.acceptable_payload["evidence_policy"].get(
        "current_snapshot_id"
    )
    if not isinstance(value, str) or not value:
        raise ValueError("episode evidence policy lacks current_snapshot_id")
    return value


def _runtime_failure_trace(
    *,
    episode: Episode,
    outcome: RuntimeExecutionFailure,
    trace_id: str,
    accepted_tool_calls: Sequence[Mapping[str, Any]],
) -> TraceOutcomeEvaluation:
    calls = normalize_accepted_tool_calls(
        trace_id=trace_id,
        raw_tool_calls=accepted_tool_calls,
        current_snapshot_id=_current_snapshot_id(episode),
    )
    signatures = [
        (call.tool_name, repr(sorted(call.request.items()))) for call in calls
    ]
    mitigation = [call for call in calls if call.tool_name == "apply_mitigation"]
    executed = [call for call in mitigation if call.successful]
    verified = any(
        "mitigation_estimate" in call.slots
        and (
            isinstance(call.response.get("estimated_gain"), (int, float))
            or call.response.get("verified") is True
        )
        for call in executed
    )
    required = tuple(TASK_REQUIRED_SLOTS[episode.task_type.value])
    return TraceOutcomeEvaluation(
        evaluator_version=TRACE_EVALUATOR_VERSION,
        decision_support_applicable=False,
        evidence_state="not_applicable",
        definite_stale_dependence=None,
        unsupported_decision=None,
        provenance_incomplete=None,
        unsafe_or_unknown=None,
        supporting_evidence_ids=(),
        unresolved_supporting_evidence_ids=(),
        provenance_closure_ids=(),
        snapshot_lineage_valid=None,
        evidence_closure_complete=None,
        required_evidence_slots=required,
        necessary_revalidation_slots=(),
        valid_revalidation_slots=(),
        adopted_revalidation_slots=(),
        necessary_revalidation_recall=None,
        valid_revalidation_adoption_rate=None,
        declaration_mismatches=(),
        unsupported_success_claim=None,
        intervention_requested=None,
        intervention_attempted=bool(mitigation),
        intervention_executed=bool(executed),
        intervention_verified=verified,
        actual_tool_call_count=len(calls),
        failed_tool_call_count=sum(not call.successful for call in calls),
        repeated_tool_call_count=len(signatures) - len(set(signatures)),
        irrelevant_tool_call_count=sum(
            not set(call.slots).intersection(required) for call in calls
        ),
        actual_cost_units=sum(call.cost_units for call in calls),
        actual_latency_seconds=round(
            sum(call.latency_seconds for call in calls), 9
        ),
        runtime_failure_reason=outcome.failure_reason,
    )


def evaluate_trace_outcome(
    *,
    episode: Episode,
    outcome: TerminalOutcome,
    trace_id: str,
    accepted_tool_calls: Sequence[Mapping[str, Any]],
) -> TraceOutcomeEvaluation:
    if isinstance(outcome, RuntimeExecutionFailure):
        return _runtime_failure_trace(
            episode=episode,
            outcome=outcome,
            trace_id=trace_id,
            accepted_tool_calls=accepted_tool_calls,
        )
    base = asdict(
        evaluate_trace_evidence(
            episode=episode,
            decision=outcome,
            trace_id=trace_id,
            accepted_tool_calls=accepted_tool_calls,
        )
    )
    base.pop("evaluator_version")
    return TraceOutcomeEvaluation(
        evaluator_version=TRACE_EVALUATOR_VERSION,
        decision_support_applicable=True,
        runtime_failure_reason=None,
        **base,
    )


__all__ = [
    "TRACE_EVALUATOR_VERSION",
    "TraceOutcomeEvaluation",
    "evaluate_trace_outcome",
]
