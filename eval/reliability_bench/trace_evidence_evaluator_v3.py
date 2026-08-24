"""Trace-authoritative evidence and intervention measurements.

The evaluator reconstructs evidence from accepted tool calls.  Agent-declared
revalidation fields are audited for consistency but never create execution facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .measurement_spec_v3 import TASK_REQUIRED_SLOTS as REQUIRED_SLOT_REGISTRY
from .schema import Episode, TaskType
from .terminal_schema_v3 import CompletionStatus, TerminalDecision


TRACE_EVALUATOR_VERSION = "reliabilitybench-q/trace-evidence-evaluator-3.0-candidate"

TOOL_EVIDENCE_REGISTRY: dict[str, tuple[str, ...]] = {
    "get_backend_health": ("backend_health", "backend_capacity"),
    "compare_backends": ("backend_ranking",),
    "get_qubit_properties": ("qubit_properties",),
    "get_coupling_map": ("coupling_map",),
    "transpile_circuit": ("compilation",),
    "run_circuit": ("circuit_result",),
    "apply_mitigation": ("mitigation_estimate",),
    "refresh_task_context": ("task_context",),
}

EVIDENCE_TYPE_SLOTS: dict[str, tuple[str, ...]] = {
    "backend_ranking": ("backend_ranking",),
    "qubit_mapping": ("qubit_properties",),
    "transpiled_circuit": ("compilation",),
    "circuit_result": ("circuit_result",),
    "mitigation_estimate": ("mitigation_estimate",),
    "feasibility_assessment": ("backend_capacity",),
    "task_context": ("task_context",),
}

TASK_REQUIRED_SLOTS: dict[TaskType, tuple[str, ...]] = {
    task_type: REQUIRED_SLOT_REGISTRY[task_type.value] for task_type in TaskType
}


@dataclass(frozen=True)
class EvidenceNode:
    evidence_id: str
    source: str
    snapshot_id: str
    slots: tuple[str, ...]
    depends_on_resources: tuple[str, ...]
    valid: bool
    stale: bool = False
    closure_complete: bool = True
    parent_evidence_ids: tuple[str, ...] = ()
    query_reference_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class NormalizedToolCall:
    tool_call_id: str
    call_index: int
    tool_name: str
    request: dict[str, Any]
    response: dict[str, Any]
    output_evidence_id: str
    observed_snapshot_id: str
    slots: tuple[str, ...]
    query_reference_ids: tuple[str, ...]
    parent_evidence_ids: tuple[str, ...]
    successful: bool
    cost_units: int
    latency_seconds: float


@dataclass(frozen=True)
class TraceEvidenceEvaluation:
    evaluator_version: str
    evidence_state: str
    definite_stale_dependence: bool
    unsupported_decision: bool
    provenance_incomplete: bool
    unsafe_or_unknown: bool
    supporting_evidence_ids: tuple[str, ...]
    unresolved_supporting_evidence_ids: tuple[str, ...]
    provenance_closure_ids: tuple[str, ...]
    snapshot_lineage_valid: bool
    evidence_closure_complete: bool
    required_evidence_slots: tuple[str, ...]
    necessary_revalidation_slots: tuple[str, ...]
    valid_revalidation_slots: tuple[str, ...]
    adopted_revalidation_slots: tuple[str, ...]
    necessary_revalidation_recall: float
    valid_revalidation_adoption_rate: float
    declaration_mismatches: tuple[str, ...]
    unsupported_success_claim: bool
    intervention_requested: bool
    intervention_attempted: bool
    intervention_executed: bool
    intervention_verified: bool
    actual_tool_call_count: int
    failed_tool_call_count: int
    repeated_tool_call_count: int
    irrelevant_tool_call_count: int
    actual_cost_units: int
    actual_latency_seconds: float


def _tool_successful(response: Mapping[str, Any]) -> bool:
    if response.get("error") is not None:
        return False
    if response.get("success") is False or response.get("executed") is False:
        return False
    status = response.get("status")
    return status not in {"error", "failed", "rejected", "timeout", "timed_out"}


def _string_ids(value: Any, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must not contain duplicates")
    return tuple(value)


def normalize_accepted_tool_calls(
    *,
    trace_id: str,
    raw_tool_calls: Sequence[Mapping[str, Any]],
    current_snapshot_id: str,
) -> tuple[NormalizedToolCall, ...]:
    """Create deterministic observation IDs from authoritative accepted calls."""
    if not trace_id or any(character.isspace() for character in trace_id):
        raise ValueError("trace_id must be a non-empty whitespace-free string")
    normalized: list[NormalizedToolCall] = []
    seen_indices: set[int] = set()
    for raw in raw_tool_calls:
        index = raw.get("call_index")
        if not isinstance(index, int) or isinstance(index, bool) or index <= 0:
            raise ValueError("accepted tool call requires a positive call_index")
        if index in seen_indices:
            raise ValueError("accepted tool call indices must be unique")
        seen_indices.add(index)
        tool_name = raw.get("tool_name")
        if tool_name not in TOOL_EVIDENCE_REGISTRY:
            raise ValueError(f"tool is absent from frozen evidence registry: {tool_name}")
        request = raw.get("request")
        response = raw.get("response")
        if not isinstance(request, Mapping) or not isinstance(response, Mapping):
            raise ValueError("accepted tool call requires object request and response")
        trigger = raw.get("trigger_evidence_id")
        references = (trigger,) if isinstance(trigger, str) and trigger else ()
        cost = raw.get("cost_units", 0)
        if not isinstance(cost, int) or isinstance(cost, bool) or cost < 0:
            raise ValueError("tool cost_units must be a non-negative integer")
        latency = raw.get("latency_seconds", 0.0)
        if (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or latency < 0
        ):
            raise ValueError("tool latency_seconds must be a non-negative number")
        observed_snapshot = raw.get("observed_snapshot_id", current_snapshot_id)
        if not isinstance(observed_snapshot, str) or not observed_snapshot:
            raise ValueError("observed_snapshot_id must be a non-empty string")
        normalized.append(
            NormalizedToolCall(
                tool_call_id=f"call:{index}",
                call_index=index,
                tool_name=str(tool_name),
                request=dict(request),
                response=dict(response),
                output_evidence_id=f"obs:{trace_id}:{index}",
                observed_snapshot_id=observed_snapshot,
                slots=TOOL_EVIDENCE_REGISTRY[str(tool_name)],
                query_reference_ids=references,
                parent_evidence_ids=_string_ids(
                    raw.get("parent_evidence_ids"), field="parent_evidence_ids"
                ),
                successful=_tool_successful(response),
                cost_units=cost,
                latency_seconds=float(latency),
            )
        )
    return tuple(sorted(normalized, key=lambda item: item.call_index))


def _initial_evidence_nodes(episode: Episode) -> tuple[EvidenceNode, ...]:
    affected = set(episode.drift_event.affected_resources)
    snapshot_id = str(episode.initial_state["snapshot_id"])
    nodes = []
    for evidence in episode.evidence:
        slots = EVIDENCE_TYPE_SLOTS.get(evidence.evidence_type, ())
        valid = not set(evidence.depends_on_resources).intersection(affected)
        nodes.append(
            EvidenceNode(
                evidence_id=evidence.evidence_id,
                source=f"episode:{episode.episode_id}:initial_evidence",
                snapshot_id=snapshot_id,
                slots=slots,
                depends_on_resources=evidence.depends_on_resources,
                valid=valid,
                stale=not valid,
            )
        )
    return tuple(nodes)


def _snapshot_lineage(episode: Episode) -> tuple[set[str], str, bool]:
    policy = episode.ground_truth.acceptable_payload["evidence_policy"]
    lineage = policy.get("snapshot_lineage")
    initial = str(episode.initial_state["snapshot_id"])
    current = _current_snapshot_id(episode)
    if not isinstance(lineage, list):
        return set(), current, False
    by_id = {
        item.get("snapshot_id"): item
        for item in lineage
        if isinstance(item, Mapping) and isinstance(item.get("snapshot_id"), str)
    }
    initial_node = by_id.get(initial)
    current_node = by_id.get(current)
    valid = bool(
        len(by_id) == len(lineage)
        and initial_node
        and initial_node.get("phase") == "pre_drift"
        and initial_node.get("parent_snapshot_id") is None
        and current_node
        and current_node.get("phase") == "current"
        and current_node.get("parent_snapshot_id") == initial
    )
    return set(by_id), current, valid


def _tool_evidence_nodes(
    calls: Sequence[NormalizedToolCall],
    *,
    initial_nodes: Sequence[EvidenceNode],
    registered_snapshots: set[str],
    current_snapshot_id: str,
    snapshot_lineage_valid: bool,
) -> tuple[EvidenceNode, ...]:
    known = {node.evidence_id for node in initial_nodes}
    nodes: list[EvidenceNode] = []
    for call in calls:
        closure_complete = set(call.parent_evidence_ids).issubset(known)
        snapshot_valid = (
            snapshot_lineage_valid
            and
            call.observed_snapshot_id in registered_snapshots
            and call.observed_snapshot_id == current_snapshot_id
        )
        nodes.append(
            EvidenceNode(
                evidence_id=call.output_evidence_id,
                source=f"accepted_tool_call:{call.tool_call_id}:{call.tool_name}",
                snapshot_id=call.observed_snapshot_id,
                slots=call.slots,
                depends_on_resources=(),
                valid=call.successful and snapshot_valid and closure_complete,
                stale=False,
                closure_complete=closure_complete,
                parent_evidence_ids=call.parent_evidence_ids,
                query_reference_ids=call.query_reference_ids,
            )
        )
        known.add(call.output_evidence_id)
    return tuple(nodes)


def _provenance_closure(
    cited_ids: Sequence[str], nodes: Mapping[str, EvidenceNode]
) -> tuple[tuple[str, ...], bool]:
    pending = list(cited_ids)
    visited: set[str] = set()
    complete = True
    while pending:
        evidence_id = pending.pop()
        if evidence_id in visited:
            continue
        visited.add(evidence_id)
        node = nodes.get(evidence_id)
        if node is None:
            complete = False
            continue
        if not node.closure_complete:
            complete = False
        pending.extend(node.parent_evidence_ids)
    return tuple(sorted(visited)), complete


def _current_snapshot_id(episode: Episode) -> str:
    policy = episode.ground_truth.acceptable_payload["evidence_policy"]
    value = policy.get("current_snapshot_id")
    if not isinstance(value, str) or not value:
        raise ValueError("episode evidence policy lacks current_snapshot_id")
    return value


def _declaration_mismatches(
    decision: TerminalDecision,
    calls: Sequence[NormalizedToolCall],
    known_initial_ids: set[str],
) -> tuple[str, ...]:
    by_id = {call.tool_call_id: call for call in calls}
    mismatches: list[str] = []
    for declaration in decision.revalidation_actions:
        call = by_id.get(declaration.tool_call_id)
        if call is None:
            mismatches.append(f"missing_actual_call:{declaration.tool_call_id}")
            continue
        if declaration.output_evidence_ids != (call.output_evidence_id,):
            mismatches.append(f"output_evidence_mismatch:{declaration.tool_call_id}")
        unknown_targets = set(declaration.target_evidence_ids) - known_initial_ids
        if unknown_targets:
            mismatches.append(f"unknown_revalidation_target:{declaration.tool_call_id}")
    return tuple(mismatches)


def _intervention_stages(
    decision: TerminalDecision,
    calls: Sequence[NormalizedToolCall],
) -> tuple[bool, bool, bool, bool]:
    if decision.task_type != TaskType.MITIGATION_DECISION.value:
        return False, False, False, False
    requested = decision.payload.get("intervention_requested") is True
    policy = decision.payload.get("mitigation_policy")
    matching = [
        call
        for call in calls
        if call.tool_name == "apply_mitigation"
        and call.request.get("mitigation_policy") == policy
    ]
    attempted = bool(matching)
    executed_calls = [call for call in matching if call.successful]
    executed = bool(executed_calls)
    verified = any(
        "mitigation_estimate" in call.slots
        and (
            isinstance(call.response.get("estimated_gain"), (int, float))
            or call.response.get("verified") is True
        )
        for call in executed_calls
    )
    return requested, attempted, executed, verified


def evaluate_trace_evidence(
    *,
    episode: Episode,
    decision: TerminalDecision,
    trace_id: str,
    accepted_tool_calls: Sequence[Mapping[str, Any]],
) -> TraceEvidenceEvaluation:
    """Compute provenance, revalidation, intervention, and actual-cost metrics."""
    calls = normalize_accepted_tool_calls(
        trace_id=trace_id,
        raw_tool_calls=accepted_tool_calls,
        current_snapshot_id=_current_snapshot_id(episode),
    )
    registered_snapshots, current_snapshot_id, lineage_valid = _snapshot_lineage(
        episode
    )
    initial_nodes = _initial_evidence_nodes(episode)
    tool_nodes = _tool_evidence_nodes(
        calls,
        initial_nodes=initial_nodes,
        registered_snapshots=registered_snapshots,
        current_snapshot_id=current_snapshot_id,
        snapshot_lineage_valid=lineage_valid,
    )
    nodes = {node.evidence_id: node for node in (*initial_nodes, *tool_nodes)}
    unresolved = tuple(
        evidence_id
        for evidence_id in decision.supporting_evidence_ids
        if evidence_id not in nodes
    )
    supporting_nodes = [
        nodes[evidence_id]
        for evidence_id in decision.supporting_evidence_ids
        if evidence_id in nodes
    ]
    closure_ids, closure_complete = _provenance_closure(
        decision.supporting_evidence_ids, nodes
    )
    definite_stale = any(node.stale for node in supporting_nodes)
    required_slots = TASK_REQUIRED_SLOTS[episode.task_type]
    valid_supported_slots = {
        slot
        for node in supporting_nodes
        if node.valid
        for slot in node.slots
    }
    unsupported = not set(required_slots).issubset(valid_supported_slots)
    provenance_incomplete = (
        bool(unresolved)
        or not decision.supporting_evidence_ids
        or not closure_complete
    )
    if definite_stale:
        evidence_state = "unsafe"
    elif not unsupported and not provenance_incomplete:
        evidence_state = "safe"
    else:
        evidence_state = "unknown"

    valid_initial_slots = {
        slot for node in initial_nodes if node.valid for slot in node.slots
    }
    necessary_slots = tuple(
        slot for slot in required_slots if slot not in valid_initial_slots
    )
    valid_tool_slots = {
        slot
        for node in tool_nodes
        if node.valid
        for slot in node.slots
        if slot in necessary_slots
    }
    cited = set(decision.supporting_evidence_ids)
    adopted_tool_slots = {
        slot
        for node in tool_nodes
        if node.valid and node.evidence_id in cited
        for slot in node.slots
        if slot in necessary_slots
    }
    recall = (
        len(valid_tool_slots) / len(necessary_slots) if necessary_slots else 1.0
    )
    adoption = (
        len(adopted_tool_slots) / len(valid_tool_slots) if valid_tool_slots else 1.0
    )
    mismatches = _declaration_mismatches(
        decision, calls, {node.evidence_id for node in initial_nodes}
    )
    requested, attempted, executed, verified = _intervention_stages(decision, calls)
    call_signatures = [
        (call.tool_name, repr(sorted(call.request.items()))) for call in calls
    ]
    repeated_calls = len(call_signatures) - len(set(call_signatures))
    irrelevant_calls = sum(
        not set(call.slots).intersection(required_slots) for call in calls
    )
    return TraceEvidenceEvaluation(
        evaluator_version=TRACE_EVALUATOR_VERSION,
        evidence_state=evidence_state,
        definite_stale_dependence=definite_stale,
        unsupported_decision=unsupported,
        provenance_incomplete=provenance_incomplete,
        unsafe_or_unknown=definite_stale or unsupported or provenance_incomplete,
        supporting_evidence_ids=decision.supporting_evidence_ids,
        unresolved_supporting_evidence_ids=unresolved,
        provenance_closure_ids=closure_ids,
        snapshot_lineage_valid=lineage_valid,
        evidence_closure_complete=closure_complete,
        required_evidence_slots=required_slots,
        necessary_revalidation_slots=necessary_slots,
        valid_revalidation_slots=tuple(sorted(valid_tool_slots)),
        adopted_revalidation_slots=tuple(sorted(adopted_tool_slots)),
        necessary_revalidation_recall=recall,
        valid_revalidation_adoption_rate=adoption,
        declaration_mismatches=mismatches,
        unsupported_success_claim=bool(
            decision.completion_status is CompletionStatus.ANSWERED and unsupported
        ),
        intervention_requested=requested,
        intervention_attempted=attempted,
        intervention_executed=executed,
        intervention_verified=verified,
        actual_tool_call_count=len(calls),
        failed_tool_call_count=sum(not call.successful for call in calls),
        repeated_tool_call_count=repeated_calls,
        irrelevant_tool_call_count=irrelevant_calls,
        actual_cost_units=sum(call.cost_units for call in calls),
        actual_latency_seconds=round(
            sum(call.latency_seconds for call in calls), 9
        ),
    )


__all__ = [
    "EVIDENCE_TYPE_SLOTS",
    "TASK_REQUIRED_SLOTS",
    "TOOL_EVIDENCE_REGISTRY",
    "TRACE_EVALUATOR_VERSION",
    "EvidenceNode",
    "NormalizedToolCall",
    "TraceEvidenceEvaluation",
    "evaluate_trace_evidence",
    "normalize_accepted_tool_calls",
]
