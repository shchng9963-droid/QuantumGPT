"""ActionGuard v1.2: non-prescriptive shield over controller belief B_t(e)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .action_guard_contract_v1_2 import (
    AcceptedToolCall,
    BeliefValidity,
    CandidateFinalAction,
    CandidateToolAction,
    ControllerBeliefLedger,
    GuardOutcome,
    GuardResult,
    PUBLIC_TOOL_SCHEMAS,
    ParameterRule,
    PublicTaskContract,
    ReasonCode,
    candidate_sha256,
)


ACTION_GUARD_VERSION = "reliabilitybench-q/action-guard-1.2-candidate"


def _result(
    candidate: CandidateToolAction | CandidateFinalAction,
    outcome: GuardOutcome,
    reason: ReasonCode | None = None,
    constraint: str | None = None,
) -> GuardResult:
    return GuardResult(outcome, reason, constraint, candidate_sha256(candidate))


def _valid_parameter_type(value: Any, rule: ParameterRule) -> bool:
    if rule.kind == "string":
        return isinstance(value, str) and (bool(value) or not rule.non_empty)
    if rule.kind == "string_list":
        return (
            isinstance(value, list)
            and all(isinstance(item, str) and item for item in value)
            and (bool(value) or not rule.non_empty)
            and len(value) == len(set(value))
        )
    if rule.kind == "integer_list":
        return (
            isinstance(value, list)
            and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
            and (bool(value) or not rule.non_empty)
            and len(value) == len(set(value))
        )
    raise ValueError("unsupported public parameter kind")


def _parameter_entities(value: Any, rule: ParameterRule) -> tuple[Any, ...]:
    if rule.kind in {"string_list", "integer_list"}:
        return tuple(value)
    return (value,)


class ActionGuard:
    """Pure decision procedure; validity input is exclusively controller B_t."""

    def check_tool_action(
        self,
        candidate: CandidateToolAction,
        *,
        task: PublicTaskContract,
        ledger: ControllerBeliefLedger,
    ) -> GuardResult:
        schema = PUBLIC_TOOL_SCHEMAS.get(candidate.tool_name)
        if schema is None:
            return _result(
                candidate,
                GuardOutcome.REPAIR_REQUEST,
                ReasonCode.INVALID_SCHEMA,
                "tool_name must exist in the public tool catalog",
            )
        if set(candidate.arguments) != set(schema):
            return _result(
                candidate,
                GuardOutcome.REPAIR_REQUEST,
                ReasonCode.INVALID_SCHEMA,
                "tool arguments must exactly match the public tool schema",
            )
        for name, rule in schema.items():
            value = candidate.arguments[name]
            if not _valid_parameter_type(value, rule):
                return _result(
                    candidate,
                    GuardOutcome.REPAIR_REQUEST,
                    ReasonCode.INVALID_SCHEMA,
                    "tool argument type must match the public tool schema",
                )
            if rule.entity_domain is not None:
                domain = task.entity_domain(rule.entity_domain)
                if any(item not in domain for item in _parameter_entities(value, rule)):
                    return _result(
                        candidate,
                        GuardOutcome.REPAIR_REQUEST,
                        ReasonCode.UNKNOWN_ENTITY,
                        "tool arguments must reference public task entities",
                    )
        if (
            candidate.trigger_evidence_id is not None
            and candidate.trigger_evidence_id not in ledger.by_id()
        ):
            return _result(
                candidate,
                GuardOutcome.REPAIR_REQUEST,
                ReasonCode.INVALID_REFERENCE,
                "trigger_evidence_id must exist in the visible ledger",
            )
        return _result(candidate, GuardOutcome.ALLOW)

    def check_final_action(
        self,
        candidate: CandidateFinalAction,
        *,
        task: PublicTaskContract,
        ledger: ControllerBeliefLedger,
        accepted_tool_calls: Sequence[AcceptedToolCall],
    ) -> GuardResult:
        decision = candidate.terminal
        if decision.task_type != task.task_type:
            return _result(
                candidate,
                GuardOutcome.REPAIR_REQUEST,
                ReasonCode.INVALID_SCHEMA,
                "terminal task_type must match the public task contract",
            )

        entity_error = self._terminal_entity_error(decision.payload, task)
        if entity_error:
            return _result(
                candidate,
                GuardOutcome.REPAIR_REQUEST,
                ReasonCode.UNKNOWN_ENTITY,
                entity_error,
            )

        ledger_by_id = ledger.by_id()
        for evidence_id in decision.supporting_evidence_ids:
            if evidence_id not in ledger_by_id:
                return _result(
                    candidate,
                    GuardOutcome.REPAIR_REQUEST,
                    ReasonCode.INVALID_REFERENCE,
                    "supporting evidence IDs must exist in the visible ledger",
                )

        revalidation_error = self._revalidation_error(
            candidate, ledger, accepted_tool_calls
        )
        if revalidation_error is not None:
            return revalidation_error

        # Abstention is not a task answer or success claim.  It remains visible
        # to coverage/selective-risk metrics and is never silently converted.
        if not decision.answered:
            return _result(candidate, GuardOutcome.ALLOW)

        cited_records = [ledger_by_id[item] for item in decision.supporting_evidence_ids]
        if any(
            item.belief_validity is BeliefValidity.BELIEVED_INVALID
            for item in cited_records
        ):
            return _result(
                candidate,
                GuardOutcome.BLOCK,
                ReasonCode.STALE_CITED_EVIDENCE,
                "final decisions must not cite ledger evidence marked invalid",
            )
        if not decision.supporting_evidence_ids:
            return _result(
                candidate,
                GuardOutcome.BLOCK,
                ReasonCode.PREMATURE_FINALIZATION,
                "answered terminal decisions require explicit supporting evidence",
            )
        valid_slots = {
            slot
            for item in cited_records
            if item.belief_validity is BeliefValidity.BELIEVED_VALID
            for slot in item.slots
        }
        if not set(task.required_slots).issubset(valid_slots):
            return _result(
                candidate,
                GuardOutcome.BLOCK,
                ReasonCode.MISSING_REQUIRED_SUPPORT,
                "all public required evidence slots must have cited valid support",
            )
        return _result(candidate, GuardOutcome.ALLOW)

    @staticmethod
    def _terminal_entity_error(
        payload: Mapping[str, Any], task: PublicTaskContract
    ) -> str | None:
        checks: tuple[tuple[str, str], ...] = (
            ("backend_id", "backend"),
            ("qubit_ids", "qubit"),
            ("compilation_snapshot_id", "snapshot"),
            ("mitigation_policy", "mitigation_policy"),
        )
        for field, domain_name in checks:
            if field not in payload:
                continue
            value = payload[field]
            entities = value if isinstance(value, list) else [value]
            domain = task.entity_domain(domain_name)
            if any(item not in domain for item in entities):
                return f"{field} must reference a public task entity"
        return None

    @staticmethod
    def _revalidation_error(
        candidate: CandidateFinalAction,
        ledger: ControllerBeliefLedger,
        accepted_tool_calls: Sequence[AcceptedToolCall],
    ) -> GuardResult | None:
        decision = candidate.terminal
        if not decision.revalidation_actions:
            return None
        ledger_by_id = ledger.by_id()
        calls_by_id = {item.tool_call_id: item for item in accepted_tool_calls}
        for declaration in decision.revalidation_actions:
            call = calls_by_id.get(declaration.tool_call_id)
            if call is None:
                return _result(
                    candidate,
                    GuardOutcome.REPAIR_REQUEST,
                    ReasonCode.INVALID_REFERENCE,
                    "declared tool_call_id must bind to an accepted real call",
                )
            if any(item not in ledger_by_id for item in declaration.target_evidence_ids):
                return _result(
                    candidate,
                    GuardOutcome.REPAIR_REQUEST,
                    ReasonCode.INVALID_REFERENCE,
                    "revalidation targets must exist in the visible ledger",
                )
            if tuple(declaration.output_evidence_ids) != call.output_evidence_ids:
                return _result(
                    candidate,
                    GuardOutcome.BLOCK,
                    ReasonCode.UNVERIFIED_REVALIDATION,
                    "declared outputs must exactly match the accepted real call",
                )
            if not call.successful:
                return _result(
                    candidate,
                    GuardOutcome.BLOCK,
                    ReasonCode.UNVERIFIED_REVALIDATION,
                    "revalidation requires a successful accepted tool response",
                )
            for output_id in call.output_evidence_ids:
                evidence = ledger_by_id.get(output_id)
                if (
                    evidence is None
                    or evidence.belief_validity is not BeliefValidity.BELIEVED_VALID
                    or evidence.produced_by_tool_call_id != call.tool_call_id
                ):
                    return _result(
                        candidate,
                        GuardOutcome.BLOCK,
                        ReasonCode.UNVERIFIED_REVALIDATION,
                        "revalidation output must bind to new valid ledger evidence",
                    )
        return None


__all__ = ["ACTION_GUARD_VERSION", "ActionGuard"]
