"""Treatment-neutral runtime contract shared by the six planned study arms.

The adapter validates syntax, identifiers, and binding to authoritative tool
calls.  It never repairs task semantics, proposes an action, or fabricates
evidence.  ActionGuard entries are interfaces only; no guard behavior lives here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .terminal_schema_v3 import (
    TASK_ACTIONS,
    TASK_PAYLOAD_KEYS,
    TERMINAL_SCHEMA_VERSION,
    TerminalDecision,
    parse_terminal_decision,
)


PUBLIC_RUNTIME_VERSION = "reliabilitybench-q/public-runtime-3.0"


class StudyArm(str, Enum):
    LEDGER_ONLY = "ledger_only"
    ORIGINAL_FULL = "original_full"
    LEDGER_GUARD = "ledger_only_plus_action_guard"
    FULL_GUARD = "full_plus_action_guard"
    GLOBAL_GUARD = "global_revalidate_plus_action_guard"
    INVALIDATION_CONSISTENCY_CHECK = "invalidation_consistency_check"


@dataclass(frozen=True)
class SharedRuntimeConfig:
    max_tokens: int = 4096
    max_turns: int = 12
    max_tool_calls: int = 10
    max_cost_units: int = 10
    request_timeout_seconds: float = 120.0
    api_max_attempts: int = 2
    api_fixed_backoff_seconds: float = 1.0
    format_max_attempts: int = 2
    trace_save_policy: str = "append_only_keep_failures"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    cost_units: int


SHARED_TOOL_CATALOG = (
    ToolSpec("get_backend_health", 1),
    ToolSpec("compare_backends", 1),
    ToolSpec("get_qubit_properties", 1),
    ToolSpec("get_coupling_map", 1),
    ToolSpec("transpile_circuit", 3),
    ToolSpec("run_circuit", 5),
    ToolSpec("apply_mitigation", 3),
    ToolSpec("refresh_task_context", 1),
)
SHARED_RUNTIME_CONFIG = SharedRuntimeConfig()

# This matrix is the only permitted treatment difference. Guard behavior lives
# in the separately versioned treatment component, never in this common adapter.
COMPONENT_INFORMATION_ACCESS: dict[StudyArm, dict[str, Any]] = {
    StudyArm.LEDGER_ONLY: {
        "evidence_ledger": True,
        "invalidation": "none",
        "action_guard": "absent",
        "objective_consistency_private_channel": False,
    },
    StudyArm.ORIGINAL_FULL: {
        "evidence_ledger": True,
        "invalidation": "dependency_scoped",
        "action_guard": "absent",
        "objective_consistency_private_channel": False,
    },
    StudyArm.LEDGER_GUARD: {
        "evidence_ledger": True,
        "invalidation": "none",
        "action_guard": "external_action_guard_v1_2",
        "objective_consistency_private_channel": False,
    },
    StudyArm.FULL_GUARD: {
        "evidence_ledger": True,
        "invalidation": "dependency_scoped",
        "action_guard": "external_action_guard_v1_2",
        "objective_consistency_private_channel": False,
    },
    StudyArm.GLOBAL_GUARD: {
        "evidence_ledger": True,
        "invalidation": "global",
        "action_guard": "external_action_guard_v1_2",
        "objective_consistency_private_channel": False,
    },
    StudyArm.INVALIDATION_CONSISTENCY_CHECK: {
        "evidence_ledger": True,
        "invalidation": "objective_invalidation_consistency_check_only",
        "action_guard": "external_action_guard_v1_2",
        "objective_consistency_private_channel": True,
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def terminal_schema_public_description() -> dict[str, Any]:
    return {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "required_terminal_fields": [
            "schema_version",
            "completion_status",
            "decision",
            "supporting_evidence_ids",
            "revalidation_actions",
            "failure",
        ],
        "completion_status_values": [
            "answered",
            "abstain",
            "execution_failure",
        ],
        "task_actions": {key: sorted(value) for key, value in TASK_ACTIONS.items()},
        "task_payload_fields": {
            key: sorted(value) for key, value in TASK_PAYLOAD_KEYS.items()
        },
        "facts": [
            (
                "completion_status=answered means a task decision was produced; "
                "it does not mean the physical target is feasible"
            ),
            (
                "a correct declare_unreachable decision uses completion_status=answered; "
                "execution_failure is reserved for failure to produce a task decision"
            ),
            "supporting_evidence_ids must cite IDs actually present in the task or tool observations",
            "revalidation declarations do not create tool calls",
            "mitigation may report intervention_requested but never intervention_executed",
        ],
    }


def build_public_system_prompt() -> str:
    payload = {
        "role": "ReliabilityBench-Q decision agent",
        "instruction": (
            "Use the public task and accepted tool observations. Return exactly one "
            "JSON terminal object matching the shared schema; do not use markdown."
        ),
        "terminal_schema": terminal_schema_public_description(),
        "tool_catalog": [asdict(item) for item in SHARED_TOOL_CATALOG],
        "budget": asdict(SHARED_RUNTIME_CONFIG),
        "format_error_policy": (
            "A schema error is recorded. The same generic schema reminder may be "
            "issued within the shared format-attempt limit; no semantic correction occurs."
        ),
    }
    return canonical_json(payload)


PUBLIC_SYSTEM_PROMPT = build_public_system_prompt()
GENERIC_FORMAT_RETRY = canonical_json(
    {
        "error": "terminal_schema_format_failure",
        "instruction": (
            "Return one JSON object matching terminal-decision-3.0. No task answer, "
            "evidence ID, or semantic correction is supplied by this message."
        ),
    }
)


@dataclass(frozen=True)
class RuntimeParseResult:
    accepted: bool
    decision: TerminalDecision | None
    format_failure: str | None
    identifier_failures: tuple[str, ...]
    retry_message: str | None


def deterministic_evidence_id(trace_id: str, call_index: int) -> str:
    if not trace_id or any(character.isspace() for character in trace_id):
        raise ValueError("trace_id must be a non-empty whitespace-free string")
    if not isinstance(call_index, int) or isinstance(call_index, bool) or call_index <= 0:
        raise ValueError("call_index must be a positive integer")
    return f"obs:{trace_id}:{call_index}"


def bind_authoritative_tool_observation(
    *, trace_id: str, call_index: int, accepted_call: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach an ID to a real accepted call without modifying request/response."""
    if not isinstance(accepted_call.get("request"), Mapping):
        raise ValueError("accepted call request must be an object")
    if not isinstance(accepted_call.get("response"), Mapping):
        raise ValueError("accepted call response must be an object")
    bound = dict(accepted_call)
    bound["call_index"] = call_index
    bound["tool_call_id"] = f"call:{call_index}"
    bound["output_evidence_id"] = deterministic_evidence_id(trace_id, call_index)
    return bound


def _json_object(raw_content: str) -> Mapping[str, Any]:
    parsed = json.loads(raw_content)
    if not isinstance(parsed, Mapping):
        raise ValueError("terminal response must be a JSON object")
    return parsed


def parse_terminal_runtime_output(
    raw_content: str,
    *,
    known_evidence_ids: Sequence[str],
    accepted_tool_calls: Sequence[Mapping[str, Any]],
    format_attempt: int = 1,
) -> RuntimeParseResult:
    """Strict parse and ID audit; never alters the Agent's submitted object."""
    try:
        decision = parse_terminal_decision(_json_object(raw_content))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        retry = (
            GENERIC_FORMAT_RETRY
            if format_attempt < SHARED_RUNTIME_CONFIG.format_max_attempts
            else None
        )
        return RuntimeParseResult(False, None, f"{type(exc).__name__}:{exc}", (), retry)

    known = set(known_evidence_ids)
    calls_by_id: dict[str, Mapping[str, Any]] = {}
    for call in accepted_tool_calls:
        call_id = call.get("tool_call_id")
        output_id = call.get("output_evidence_id")
        if isinstance(call_id, str):
            calls_by_id[call_id] = call
        if isinstance(output_id, str):
            known.add(output_id)
    failures: list[str] = []
    for evidence_id in decision.supporting_evidence_ids:
        if evidence_id not in known:
            failures.append(f"unknown_supporting_evidence_id:{evidence_id}")
    for declaration in decision.revalidation_actions:
        actual = calls_by_id.get(declaration.tool_call_id)
        if actual is None:
            failures.append(f"unbound_tool_call_id:{declaration.tool_call_id}")
            continue
        expected_output = actual.get("output_evidence_id")
        if tuple(declaration.output_evidence_ids) != (expected_output,):
            failures.append(f"unbound_output_evidence:{declaration.tool_call_id}")
        for evidence_id in declaration.target_evidence_ids:
            if evidence_id not in known:
                failures.append(f"unknown_revalidation_target:{evidence_id}")
    # Identifier failures remain evidence about the submitted action.  They are
    # not repaired and do not trigger a semantic retry.
    return RuntimeParseResult(
        accepted=not failures,
        decision=decision,
        format_failure=None,
        identifier_failures=tuple(failures),
        retry_message=None,
    )


def generate_fairness_audit() -> dict[str, Any]:
    prompts = {arm.value: PUBLIC_SYSTEM_PROMPT for arm in StudyArm}
    prompt_hashes = {
        arm: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for arm, value in prompts.items()
    }
    config = asdict(SHARED_RUNTIME_CONFIG)
    tools = [asdict(item) for item in SHARED_TOOL_CATALOG]
    return {
        "runtime_version": PUBLIC_RUNTIME_VERSION,
        "arms": [arm.value for arm in StudyArm],
        "common_public_prompt_sha256": next(iter(prompt_hashes.values())),
        "all_public_prompts_identical": len(set(prompt_hashes.values())) == 1,
        "public_prompt_hash_by_arm": prompt_hashes,
        "shared_tool_catalog": tools,
        "shared_runtime_config": config,
        "component_information_access_matrix": {
            arm.value: COMPONENT_INFORMATION_ACCESS[arm] for arm in StudyArm
        },
        "only_registered_component_differences": True,
        "action_guard_core_implemented": True,
        "action_guard_implementation_location": "external treatment component ActionGuard v1.2",
        "semantic_repair_capability": False,
        "forbidden_adapter_capabilities": [
            "rewrite_action",
            "supply_evidence_id",
            "read_task_predicate",
            "recommend_acceptable_action",
            "fabricate_tool_call",
        ],
    }


def validate_fairness_contract() -> None:
    audit = generate_fairness_audit()
    if len(COMPONENT_INFORMATION_ACCESS) != 6:
        raise AssertionError("exactly six registered study arms are required")
    if not audit["all_public_prompts_identical"]:
        raise AssertionError("public prompt differs across arms")
    if any(
        item["action_guard"] not in {"absent", "external_action_guard_v1_2"}
        for item in COMPONENT_INFORMATION_ACCESS.values()
    ):
        raise AssertionError("ActionGuard behavior is forbidden in this stage")


__all__ = [
    "COMPONENT_INFORMATION_ACCESS",
    "GENERIC_FORMAT_RETRY",
    "PUBLIC_RUNTIME_VERSION",
    "PUBLIC_SYSTEM_PROMPT",
    "SHARED_RUNTIME_CONFIG",
    "SHARED_TOOL_CATALOG",
    "RuntimeParseResult",
    "StudyArm",
    "bind_authoritative_tool_observation",
    "deterministic_evidence_id",
    "generate_fairness_audit",
    "parse_terminal_runtime_output",
    "validate_fairness_contract",
]
