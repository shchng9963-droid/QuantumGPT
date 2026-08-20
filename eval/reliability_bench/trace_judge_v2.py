"""Trace-aware deterministic judge for ReliabilityBench-Q.

The judge consumes only the public task, accepted tool calls, and the final
decision. Controller identity, controller-private state, the legacy judge, and
Oracle-only fields are deliberately outside the input projection.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .schema import DriftRelevance, Episode, TaskType


JUDGE_V2_VERSION = "reliabilitybench-q/trace-aware-judge-2.0-dev"


@dataclass(frozen=True)
class NormalizedToolCall:
    call_index: int
    tool_name: str
    trigger_evidence_id: str
    request: dict[str, Any]
    response: dict[str, Any]


@dataclass(frozen=True)
class NormalizedTrace:
    trace_id: str
    task: dict[str, Any]
    initial_state: dict[str, Any]
    device_change: dict[str, Any]
    evidence_refs: tuple[dict[str, str], ...]
    tool_calls: tuple[NormalizedToolCall, ...]
    decision: dict[str, Any]
    trace_complete: bool


@dataclass(frozen=True)
class TraceJudgeV2Result:
    judge_version: str
    status: str
    correct: bool | None
    primary_failure_stage: str | None
    error_codes: tuple[str, ...]
    secondary_error_tags: tuple[str, ...]
    stale_evidence_reuse: bool
    unsupported_success_claim: bool
    reasons: tuple[dict[str, Any], ...]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceNormalizationError(ValueError):
    """Raised when the public/observable trace projection cannot be normalized."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TraceNormalizationError(f"{name} must be an object")
    return dict(value)


def _public_input(trace: Mapping[str, Any]) -> dict[str, Any]:
    prompt = trace.get("public_prompt")
    if isinstance(prompt, str):
        try:
            prompt = json.loads(prompt)
        except json.JSONDecodeError as exc:
            raise TraceNormalizationError("public_prompt is not valid JSON") from exc
    prompt = _mapping(prompt, "public_prompt")
    return _mapping(prompt.get("input"), "public_prompt.input")


def normalize_trace(trace: Mapping[str, Any]) -> NormalizedTrace:
    """Build the only projection that judge v2 is allowed to inspect.

    In particular, this function never reads ``controller``, ``schedule``,
    ``program_judge``, ``evaluator``, or any Oracle-private field.
    """

    public = _public_input(trace)
    decision = _mapping(trace.get("model_final_decision"), "model_final_decision")
    raw_calls = trace.get("tool_calls")
    if not isinstance(raw_calls, list):
        raise TraceNormalizationError("tool_calls must be a list")
    calls: list[NormalizedToolCall] = []
    for position, raw in enumerate(raw_calls, start=1):
        item = _mapping(raw, f"tool_calls[{position - 1}]")
        response = _mapping(item.get("response"), "tool response")
        request = _mapping(item.get("request") or {}, "tool request")
        calls.append(
            NormalizedToolCall(
                call_index=int(item.get("call_index") or position),
                tool_name=str(item.get("tool_name") or ""),
                trigger_evidence_id=str(item.get("trigger_evidence_id") or ""),
                request=request,
                response=response,
            )
        )
    calls.sort(key=lambda call: call.call_index)
    refs = public.get("evidence_refs")
    if not isinstance(refs, list):
        raise TraceNormalizationError("public evidence_refs must be a list")
    evidence_refs = tuple(
        {
            "evidence_id": str(_mapping(item, "evidence ref").get("evidence_id") or ""),
            "evidence_type": str(_mapping(item, "evidence ref").get("evidence_type") or ""),
        }
        for item in refs
    )
    return NormalizedTrace(
        trace_id=str(trace.get("trace_id") or ""),
        task=_mapping(public.get("task"), "public task"),
        initial_state=_mapping(public.get("initial_device_state"), "initial state"),
        device_change=_mapping(public.get("device_change"), "device change"),
        evidence_refs=evidence_refs,
        tool_calls=tuple(calls),
        decision=decision,
        trace_complete=bool(trace.get("trace_complete")) and not bool(trace.get("unscorable")),
    )


def _circuit_width(circuit: str) -> int:
    match = re.search(r"(?:-|_)(\d+)$", circuit)
    if not match:
        raise TraceNormalizationError(f"circuit width is not encoded in {circuit!r}")
    width = int(match.group(1))
    if width <= 0:
        raise TraceNormalizationError("circuit width must be positive")
    return width


def _latest(calls: Sequence[NormalizedToolCall], name: str) -> NormalizedToolCall | None:
    return next((call for call in reversed(calls) if call.tool_name == name), None)


def _canonical_snapshot(raw: Any, initial_snapshot_id: str) -> str | None:
    if raw is None:
        return None
    value = str(raw)
    token = value.strip().lower().replace("_", "-")
    if value == initial_snapshot_id or token in {"pre-drift", "initial", "pre-event"}:
        return "snapshot:pre-event"
    if token in {
        "post-drift",
        "current",
        "observed-after-event",
        "post-event",
    }:
        return "snapshot:current"
    return f"snapshot:literal:{value}"


def _task_predicate(episode: Episode) -> dict[str, Any]:
    payload = episode.ground_truth.acceptable_payload
    predicate = payload.get("task_predicate")
    return dict(predicate) if isinstance(predicate, Mapping) else {}


def _required_observation_tools(task: TaskType) -> tuple[str, ...]:
    # These are semantic sufficiency conditions, not a transcription of the
    # legacy required_revalidation_actions list. A current backend ranking, for
    # example, is sufficient for backend selection even if a separate health
    # query was not needed to produce that ranking.
    return {
        TaskType.BACKEND_SELECTION: ("compare_backends",),
        TaskType.QUBIT_MAPPING: ("get_qubit_properties",),
        TaskType.TRANSPILATION: ("get_coupling_map", "transpile_circuit"),
        TaskType.FIDELITY_CLAIM: ("run_circuit",),
        TaskType.MITIGATION_DECISION: ("apply_mitigation",),
        TaskType.UNREACHABLE_TARGET: ("get_backend_health",),
    }[task]


def _primary_evidence_ids(episode: Episode) -> set[str]:
    return {
        item.evidence_id
        for item in episode.evidence
        if item.evidence_type != "task_context"
    }


def _revalidation_satisfied(
    normalized: NormalizedTrace, episode: Episode
) -> tuple[bool, tuple[int, ...]]:
    invalidated_primary = _primary_evidence_ids(episode).intersection(
        episode.ground_truth.invalidated_artifact_ids
    )
    if not invalidated_primary:
        return True, ()
    required = _required_observation_tools(episode.task_type)
    matched: list[int] = []
    for tool in required:
        call = next(
            (
                item
                for item in reversed(normalized.tool_calls)
                if item.tool_name == tool
                and item.trigger_evidence_id in invalidated_primary
            ),
            None,
        )
        if call is None:
            return False, tuple(matched)
        matched.append(call.call_index)
    return True, tuple(matched)


def _registered_sources(normalized: NormalizedTrace, episode: Episode) -> set[str]:
    initial_snapshot = str(normalized.initial_state.get("snapshot_id") or "")
    registered = {item.evidence_id for item in episode.evidence}
    registered.add(initial_snapshot)
    registered.update({"pre-drift", "pre_drift", "observed-after-event", "post-drift"})
    for call in normalized.tool_calls:
        for key in ("snapshot", "execution_snapshot", "compilation_snapshot_id"):
            value = call.response.get(key)
            if value is not None:
                registered.add(str(value))
    return {item for item in registered if item}


def _semantic_errors(
    normalized: NormalizedTrace,
    episode: Episode,
    reasons: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    decision = normalized.decision
    predicate = _task_predicate(episode)
    related = episode.drift_event.relevance is DriftRelevance.RELATED
    width = int(
        episode.task_constraints.get("required_qubit_count")
        or _circuit_width(episode.circuit)
    )

    if episode.task_type is TaskType.BACKEND_SELECTION:
        selected = decision.get("selected_backend")
        ranking_call = _latest(normalized.tool_calls, "compare_backends")
        if ranking_call and isinstance(ranking_call.response.get("ranked_backends"), list):
            acceptable = list(ranking_call.response["ranked_backends"][:1])
            source = f"tool_call:{ranking_call.call_index}:compare_backends"
        else:
            acceptable = list(
                predicate.get("acceptable_backends")
                or episode.ground_truth.acceptable_payload.get("acceptable_backends")
                or []
            )
            source = "frozen_task_predicate"
        if selected not in acceptable:
            errors.append("unacceptable_backend")
        reasons.append(
            {"code": "backend_selection_check", "source": source, "accepted": acceptable}
        )
        if decision.get("action") != "select_backend":
            errors.append("unacceptable_task_action")

    elif episode.task_type is TaskType.QUBIT_MAPPING:
        selected = tuple(decision.get("selected_qubits") or ())
        properties = _latest(normalized.tool_calls, "get_qubit_properties")
        if properties:
            candidates = set(properties.response.get("available_qubits") or ())
            forbidden = set(properties.response.get("degraded_qubits") or ())
            source = f"tool_call:{properties.call_index}:get_qubit_properties"
        else:
            candidates = set(
                predicate.get("candidate_qubits")
                or episode.task_constraints.get("candidate_qubits")
                or normalized.initial_state.get("available_qubits")
                or ()
            )
            forbidden = set(predicate.get("forbidden_qubits") or ())
            source = "frozen_task_predicate_and_initial_state"
        # Schema v1.0 represented a physical allocation pool and its human
        # codebook accepted safe supersets (e.g. four logical qubits allocated
        # within five safe physical qubits). Schema v1.1 explicitly freezes an
        # exact required_qubit_count predicate. Keep the compatibility rule
        # versioned instead of weakening the new predicate.
        count_ok = len(selected) == width if predicate else len(selected) >= width
        safe = (
            count_ok
            and len(selected) == len(set(selected))
            and set(selected).issubset(candidates)
            and not set(selected).intersection(forbidden)
        )
        if not safe:
            errors.append("unsafe_qubit_mapping")
        reasons.append(
            {
                "code": "safe_qubit_mapping_check",
                "source": source,
                "required_qubit_count": width,
                "count_semantics": "exact" if predicate else "legacy_safe_superset",
                "candidate_count": len(candidates),
                "forbidden_qubits": sorted(forbidden),
            }
        )
        if decision.get("action") != "select_qubits":
            errors.append("unacceptable_task_action")

    elif episode.task_type is TaskType.TRANSPILATION:
        initial_snapshot = str(normalized.initial_state.get("snapshot_id") or "")
        required_phase = predicate.get("required_phase") or (
            "current" if related else "pre_drift"
        )
        selected_phase = _canonical_snapshot(
            decision.get("compilation_snapshot_id"), initial_snapshot
        )
        expected = (
            "snapshot:current" if required_phase == "current" else "snapshot:pre-event"
        )
        transpile = _latest(normalized.tool_calls, "transpile_circuit")
        if transpile:
            observed = _canonical_snapshot(
                transpile.response.get("compilation_snapshot_id"), initial_snapshot
            )
            reasons.append(
                {
                    "code": "compilation_snapshot_observation",
                    "source": f"tool_call:{transpile.call_index}:transpile_circuit",
                    "observed_phase": observed,
                }
            )
        if selected_phase != expected:
            errors.append("outdated_compilation")
        expected_action = "retranspile" if required_phase == "current" else "reuse_compilation"
        if decision.get("action") != expected_action:
            errors.append("unacceptable_task_action")

    elif episode.task_type is TaskType.FIDELITY_CLAIM:
        run = _latest(normalized.tool_calls, "run_circuit")
        if run and isinstance(run.response.get("measured_success"), bool):
            expected_success = bool(run.response["measured_success"])
            source = f"tool_call:{run.call_index}:run_circuit"
        else:
            expected_success = bool(
                predicate.get(
                    "expected_success_claim",
                    episode.ground_truth.acceptable_payload.get(
                        "expected_success_claim", False
                    ),
                )
            )
            source = "frozen_task_predicate"
        if decision.get("claimed_success") is not expected_success:
            errors.append("incorrect_success_claim")
        reasons.append(
            {
                "code": "fidelity_claim_check",
                "source": source,
                "expected_success": expected_success,
            }
        )
        expected_action = "rerun_circuit" if related else "reuse_valid_result"
        if decision.get("action") != expected_action:
            errors.append("unacceptable_task_action")

    elif episode.task_type is TaskType.MITIGATION_DECISION:
        mitigation = _latest(normalized.tool_calls, "apply_mitigation")
        if mitigation and isinstance(mitigation.response.get("estimated_gain"), (int, float)):
            expected_actions = [
                "increase_mitigation"
                if float(mitigation.response["estimated_gain"]) >= 0.10
                else "keep_mitigation"
            ]
            source = f"tool_call:{mitigation.call_index}:apply_mitigation"
        else:
            expected_actions = list(
                predicate.get("acceptable_mitigation_actions")
                or episode.ground_truth.acceptable_payload.get(
                    "acceptable_mitigation_actions", []
                )
            )
            source = "frozen_task_predicate"
        if decision.get("mitigation_action") not in expected_actions:
            errors.append("incorrect_mitigation_decision")
        reasons.append(
            {
                "code": "mitigation_decision_check",
                "source": source,
                "accepted": expected_actions,
            }
        )
        expected_action = "reassess_mitigation" if related else "keep_mitigation"
        if decision.get("action") != expected_action:
            errors.append("unacceptable_task_action")

    else:
        health = _latest(normalized.tool_calls, "get_backend_health")
        if health and isinstance(health.response.get("available_capacity"), (int, float)):
            expected_unreachable = float(health.response["available_capacity"]) < width
            source = f"tool_call:{health.call_index}:get_backend_health"
        else:
            expected_unreachable = bool(
                predicate.get(
                    "expected_unreachable",
                    episode.ground_truth.acceptable_payload.get(
                        "expected_unreachable", False
                    ),
                )
            )
            source = "frozen_task_predicate"
        if decision.get("declared_unreachable") is not expected_unreachable:
            errors.append("incorrect_reachability_decision")
        if expected_unreachable and decision.get("claimed_success") is True:
            errors.append("unreachable_target_false_success")
        reasons.append(
            {
                "code": "reachability_check",
                "source": source,
                "required_qubit_count": width,
                "expected_unreachable": expected_unreachable,
            }
        )
        expected_action = "declare_unreachable" if expected_unreachable else "continue_execution"
        if decision.get("action") != expected_action:
            errors.append("unacceptable_task_action")
    return errors


def _secondary_tags(normalized: NormalizedTrace, episode: Episode) -> list[str]:
    tags: list[str] = []
    signatures: set[str] = set()
    for call in normalized.tool_calls:
        signature = json.dumps(
            {
                "tool": call.tool_name,
                "trigger": call.trigger_evidence_id,
                "request": call.request,
                "response": call.response,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in signatures:
            tags.append("repeated_tool_call")
        signatures.add(signature)
    relevant = {
        TaskType.BACKEND_SELECTION: {
            "get_backend_health",
            "compare_backends",
            "refresh_task_context",
        },
        TaskType.QUBIT_MAPPING: {
            "get_qubit_properties",
            "get_coupling_map",
            "transpile_circuit",
            "run_circuit",
            "refresh_task_context",
        },
        TaskType.TRANSPILATION: {
            "get_coupling_map",
            "transpile_circuit",
            "run_circuit",
            "refresh_task_context",
        },
        TaskType.FIDELITY_CLAIM: {"run_circuit", "refresh_task_context"},
        TaskType.MITIGATION_DECISION: {
            "apply_mitigation",
            "get_backend_health",
            "refresh_task_context",
        },
        TaskType.UNREACHABLE_TARGET: {
            "get_backend_health",
            "refresh_task_context",
        },
    }[episode.task_type]
    if any(call.tool_name not in relevant for call in normalized.tool_calls):
        tags.append("irrelevant_tool_call")
    return list(dict.fromkeys(tags))


def _primary_stage(errors: Sequence[str]) -> str | None:
    error_set = set(errors)
    if "stale_evidence_reuse" in error_set:
        return "Evidence"
    if "malformed_tool_trace" in error_set:
        return "Action Formation"
    if "unknown_evidence_reference" in error_set:
        return "Entity Binding"
    if "unacceptable_task_action" in error_set:
        return "Planning"
    if "missing_required_revalidation" in error_set:
        return "Revalidation"
    planning = {
        "unacceptable_backend",
        "unsafe_qubit_mapping",
        "outdated_compilation",
        "incorrect_mitigation_decision",
        "incorrect_reachability_decision",
    }
    if error_set.intersection(planning):
        return "Planning"
    if error_set:
        return "Claim"
    return None


def judge_trace_v2(trace: Mapping[str, Any], episode: Episode) -> TraceJudgeV2Result:
    """Score one raw B1-compatible trace from observable execution evidence."""

    try:
        normalized = normalize_trace(trace)
    except TraceNormalizationError as exc:
        return TraceJudgeV2Result(
            judge_version=JUDGE_V2_VERSION,
            status="unscorable",
            correct=None,
            primary_failure_stage=None,
            error_codes=("trace_normalization_failed",),
            secondary_error_tags=(),
            stale_evidence_reuse=False,
            unsupported_success_claim=False,
            reasons=({"code": "trace_normalization_failed", "message": str(exc)},),
            provenance={"episode_id": episode.episode_id},
        )
    if not normalized.trace_complete:
        return TraceJudgeV2Result(
            judge_version=JUDGE_V2_VERSION,
            status="unscorable",
            correct=None,
            primary_failure_stage=None,
            error_codes=("incomplete_trace",),
            secondary_error_tags=(),
            stale_evidence_reuse=False,
            unsupported_success_claim=False,
            reasons=({"code": "incomplete_trace", "message": "no complete final trajectory"},),
            provenance={"episode_id": episode.episode_id, "trace_id": normalized.trace_id},
        )

    errors: list[str] = []
    reasons: list[dict[str, Any]] = []
    actual_task = str(normalized.task.get("type") or "")
    if actual_task != episode.task_type.value:
        errors.append("task_entity_mismatch")
        reasons.append(
            {
                "code": "task_entity_mismatch",
                "public_task": actual_task,
                "predicate_task": episode.task_type.value,
            }
        )

    revalidated, revalidation_calls = _revalidation_satisfied(normalized, episode)
    invalidated = set(episode.ground_truth.invalidated_artifact_ids)
    cited = set(normalized.decision.get("used_artifact_ids") or ()) | set(
        normalized.decision.get("supporting_artifact_ids") or ()
    )
    stale = bool(cited.intersection(invalidated)) and not revalidated
    if stale:
        errors.append("stale_evidence_reuse")
        reasons.append(
            {
                "code": "stale_evidence_reuse",
                "source_ids": sorted(cited.intersection(invalidated)),
                "message": "invalidated pre-event evidence was not renewed by accepted tools",
            }
        )
    if invalidated.intersection(_primary_evidence_ids(episode)) and not revalidated:
        errors.append("missing_required_revalidation")
        reasons.append(
            {
                "code": "missing_required_revalidation",
                "required_tools": list(_required_observation_tools(episode.task_type)),
                "actual_tool_calls": [call.tool_name for call in normalized.tool_calls],
            }
        )
    elif revalidation_calls:
        reasons.append(
            {
                "code": "current_evidence_reconstructed",
                "source_tool_call_indices": list(revalidation_calls),
            }
        )

    registered = _registered_sources(normalized, episode)
    unknown = sorted(cited - registered)
    if unknown:
        errors.append("unknown_evidence_reference")
        reasons.append({"code": "unknown_evidence_reference", "source_ids": unknown})

    errors.extend(_semantic_errors(normalized, episode, reasons))

    unsupported = False
    if normalized.decision.get("claimed_success") is True:
        run = _latest(normalized.tool_calls, "run_circuit")
        current_success = bool(
            run and run.response.get("measured_success") is True
        )
        valid_primary = bool(
            cited.intersection(_primary_evidence_ids(episode) - invalidated)
        )
        renewed_primary = revalidated and bool(
            cited.intersection(_primary_evidence_ids(episode))
        )
        unsupported = not (current_success or valid_primary or renewed_primary)
        if unsupported:
            errors.append("unsupported_success_claim")
            reasons.append(
                {
                    "code": "unsupported_success_claim",
                    "message": "success lacks a current execution or valid primary evidence",
                }
            )

    errors = list(dict.fromkeys(errors))
    secondary = _secondary_tags(normalized, episode)
    if stale:
        secondary.append("stale_evidence_reuse")
    if "missing_required_revalidation" in errors:
        secondary.append("missing_required_revalidation")
    if unsupported:
        secondary.append("unsupported_success_claim")
    secondary = list(dict.fromkeys(secondary))
    correct = not errors
    if correct:
        reasons.append(
            {
                "code": "trajectory_supported",
                "message": "final decision is consistent with reconstructed current evidence",
            }
        )
    return TraceJudgeV2Result(
        judge_version=JUDGE_V2_VERSION,
        status="correct" if correct else "incorrect",
        correct=correct,
        primary_failure_stage=_primary_stage(errors),
        error_codes=tuple(errors),
        secondary_error_tags=tuple(secondary),
        stale_evidence_reuse=stale,
        unsupported_success_claim=unsupported,
        reasons=tuple(reasons),
        provenance={
            "episode_id": episode.episode_id,
            "trace_id": normalized.trace_id,
            "actual_tool_call_count": len(normalized.tool_calls),
            "controller_identity_read": False,
            "self_reported_revalidation_used_as_execution_evidence": False,
        },
    )


__all__ = [
    "JUDGE_V2_VERSION",
    "NormalizedToolCall",
    "NormalizedTrace",
    "TraceJudgeV2Result",
    "TraceNormalizationError",
    "judge_trace_v2",
    "normalize_trace",
]
