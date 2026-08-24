"""Public runtime v3.1 with runtime-owned execution-failure closure.

The Agent may answer or abstain.  It cannot self-report an execution failure;
that state is produced only by the treatment-neutral runtime state machine.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from .public_runtime_v3 import (
    COMPONENT_INFORMATION_ACCESS,
    SHARED_RUNTIME_CONFIG,
    SHARED_TOOL_CATALOG,
    RuntimeParseResult,
    SharedRuntimeConfig,
    StudyArm,
    ToolSpec,
    bind_authoritative_tool_observation,
    canonical_json,
    deterministic_evidence_id,
    parse_terminal_runtime_output as _parse_v3_agent_output,
)
from .terminal_schema_v3 import (
    TASK_ACTIONS,
    TASK_PAYLOAD_KEYS,
    TERMINAL_SCHEMA_VERSION,
    CompletionStatus,
)


PUBLIC_RUNTIME_VERSION = "reliabilitybench-q/public-runtime-3.1"


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
        "completion_status_values": ["answered", "abstain"],
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
                "a correct declare_unreachable decision uses completion_status=answered"
            ),
            (
                "execution_failure is reserved for the public runtime and cannot "
                "be reported by the Agent"
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
            "Return one answered or abstain JSON object matching terminal-decision-3.0. "
            "Execution failure is runtime-owned. No task answer, evidence ID, or semantic "
            "correction is supplied by this message."
        ),
    }
)


def parse_agent_terminal_runtime_output(
    raw_content: str,
    *,
    known_evidence_ids: Sequence[str],
    accepted_tool_calls: Sequence[Mapping[str, Any]],
    format_attempt: int = 1,
) -> RuntimeParseResult:
    """Parse Agent output and reserve execution_failure for the runtime."""
    parsed = _parse_v3_agent_output(
        raw_content,
        known_evidence_ids=known_evidence_ids,
        accepted_tool_calls=accepted_tool_calls,
        format_attempt=format_attempt,
    )
    if (
        parsed.decision is not None
        and parsed.decision.completion_status is CompletionStatus.EXECUTION_FAILURE
    ):
        retry = (
            GENERIC_FORMAT_RETRY
            if format_attempt < SHARED_RUNTIME_CONFIG.format_max_attempts
            else None
        )
        return RuntimeParseResult(
            accepted=False,
            decision=None,
            format_failure="agent_reported_execution_failure_reserved_for_runtime",
            identifier_failures=(),
            retry_message=retry,
        )
    return parsed


__all__ = [
    "COMPONENT_INFORMATION_ACCESS",
    "GENERIC_FORMAT_RETRY",
    "PUBLIC_RUNTIME_VERSION",
    "PUBLIC_SYSTEM_PROMPT",
    "RuntimeParseResult",
    "SHARED_RUNTIME_CONFIG",
    "SHARED_TOOL_CATALOG",
    "SharedRuntimeConfig",
    "StudyArm",
    "ToolSpec",
    "bind_authoritative_tool_observation",
    "build_public_system_prompt",
    "canonical_json",
    "deterministic_evidence_id",
    "parse_agent_terminal_runtime_output",
    "terminal_schema_public_description",
]
