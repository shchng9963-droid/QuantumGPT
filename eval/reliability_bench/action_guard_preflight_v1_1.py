"""Development-only real-LLM preflight for the ActionGuard v1.1 six-arm design."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .action_guard_contract_v1_1 import (
    AcceptedToolCall,
    BeliefValidity,
    CandidateFinalAction,
    CandidateToolAction,
    ControllerBeliefLedger,
    ControllerEvidenceBelief,
    PUBLIC_EVIDENCE_TYPE_SLOTS,
    PUBLIC_TOOL_EVIDENCE_SLOTS,
    PUBLIC_TOOL_SCHEMAS,
    PublicTaskContract,
)
from .action_guard_runtime_v1_1 import GuardRuntimeSessionV1_1
from .generator_v2 import GeneratorV2Config, generate_method_validation_candidates
from .public_runtime_v2 import (
    SHARED_RUNTIME_CONFIG,
    StudyArm,
    deterministic_evidence_id,
    parse_terminal_runtime_output,
    terminal_schema_public_description,
)
from .schema import Episode, TaskType, episode_from_dict
from .temporal_evidence_v1_1 import (
    EvidenceBeliefDependency,
    dependency_scoped_belief_update,
    ledger_only_belief_state,
    register_revalidated_belief,
)


PREFLIGHT_VERSION = "reliabilitybench-q/action-guard-real-llm-preflight-1.1"
DEVELOPMENT_NAMESPACE = "AG11DEV-20260824"
GUARD_ARMS = {
    StudyArm.LEDGER_GUARD,
    StudyArm.FULL_GUARD,
    StudyArm.GLOBAL_GUARD,
    StudyArm.ORACLE_INVALIDATION_GUARD,
}
TOOL_COSTS = {
    "get_backend_health": 1,
    "compare_backends": 1,
    "get_qubit_properties": 1,
    "get_coupling_map": 1,
    "transpile_circuit": 3,
    "run_circuit": 5,
    "apply_mitigation": 3,
    "refresh_task_context": 1,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tool_schema_description() -> dict[str, Any]:
    return {
        tool: {
            name: {
                "kind": rule.kind,
                "entity_domain": rule.entity_domain,
                "non_empty": rule.non_empty,
            }
            for name, rule in rules.items()
        }
        for tool, rules in PUBLIC_TOOL_SCHEMAS.items()
    }


SHARED_AGENT_SYSTEM_PROMPT = canonical_json(
    {
        "role": "ReliabilityBench-Q decision agent",
        "instruction": (
            "Return exactly one JSON object per turn, without markdown. Choose either "
            "a tool proposal or a terminal proposal. Use only public observations and "
            "the controller-visible belief ledger. Do not invent entity, evidence, or call IDs."
        ),
        "outer_contract": {
            "tool_proposal_exact_keys": [
                "type",
                "tool_name",
                "arguments",
                "trigger_evidence_id",
            ],
            "final_proposal_exact_keys": ["type", "terminal"],
            "type_values": ["tool", "final"],
            "final_proposal_rule": (
                "Set type to final and put the complete terminal-decision-2.0 object "
                "under the terminal key. Never use a top-level final key."
            ),
        },
        "terminal_contract": {
            "exact_keys": [
                "schema_version",
                "status",
                "decision",
                "supporting_evidence_ids",
                "revalidation_actions",
                "failure",
            ],
            "schema_version_literal": "reliabilitybench-q/terminal-decision-2.0",
            "status_values": ["success", "failure", "abstain"],
            "decision_exact_keys": ["task_type", "task_action", "payload"],
            "decision_rule": "decision must be an object, never a string",
            "revalidation_item_exact_keys": [
                "tool_call_id",
                "target_evidence_ids",
                "output_evidence_ids",
            ],
            "failure_exact_keys": ["code", "reason"],
            "success_failure_value": {"code": None, "reason": None},
            "registered_task_actions": terminal_schema_public_description()["task_actions"],
            "payload_fields_by_task": terminal_schema_public_description()["task_payload_fields"],
        },
        "tool_argument_schemas": _tool_schema_description(),
        "facts": [
            "Tool output evidence IDs are supplied only after accepted real calls.",
            "A revalidation declaration never creates a tool call.",
            "Guard feedback, when present, contains no recommended answer or parameters.",
        ],
        "shared_budget": asdict(SHARED_RUNTIME_CONFIG),
    }
)
SHARED_AGENT_SYSTEM_PROMPT_SHA256 = sha256_text(SHARED_AGENT_SYSTEM_PROMPT)


@dataclass(frozen=True)
class CompletionResult:
    content: str
    response_id: str | None
    returned_model: str | None
    system_fingerprint: str | None
    finish_reason: str | None
    usage: dict[str, Any]
    latency_seconds: float
    request_started_at: str
    request_finished_at: str
    raw_response: dict[str, Any]


class DeepSeekCompletionClient:
    def __init__(self, api_key: str, config: Mapping[str, Any]):
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            api_key=api_key,
            base_url=str(config["base_url"]),
            timeout=float(config["request_timeout_seconds"]),
            max_retries=0,
        )

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> CompletionResult:
        started_at = utc_now()
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=str(self.config["requested_model"]),
            messages=messages,
            temperature=float(self.config["temperature"]),
            top_p=float(self.config["top_p"]),
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": str(self.config["thinking"])}},
        )
        latency = time.perf_counter() - started
        choice = response.choices[0]
        return CompletionResult(
            content=choice.message.content or "",
            response_id=getattr(response, "id", None),
            returned_model=getattr(response, "model", None),
            system_fingerprint=getattr(response, "system_fingerprint", None),
            finish_reason=getattr(choice, "finish_reason", None),
            usage=response.usage.model_dump(mode="json") if response.usage else {},
            latency_seconds=latency,
            request_started_at=started_at,
            request_finished_at=utc_now(),
            raw_response=response.model_dump(mode="json"),
        )


def _recursive_namespace(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("RBQ2-", f"{DEVELOPMENT_NAMESPACE}-")
    if isinstance(value, list):
        return [_recursive_namespace(item) for item in value]
    if isinstance(value, dict):
        return {key: _recursive_namespace(item) for key, item in value.items()}
    return value


def generate_development_preflight_episodes(seed: int) -> list[Episode]:
    """Generate six namespace-isolated related-drift episodes; never reads v2.1."""
    candidates = generate_method_validation_candidates(
        GeneratorV2Config(
            random_seed=seed,
            source="action_guard_v1_1_real_llm_preflight_source",
        )
    )
    selected = []
    for task in TaskType:
        original = next(
            item
            for item in candidates
            if item.task_type is task
            and item.drift_event.relevance.value == "related"
            and item.pair_id.endswith("-00")
        )
        payload = _recursive_namespace(original.to_dict())
        payload["template_id"] = f"{DEVELOPMENT_NAMESPACE}-{task.value}-template"
        payload["source"] = "action_guard_v1_1_real_llm_preflight_source"
        selected.append(episode_from_dict(payload))
    if len({item.episode_id for item in selected}) != 6:
        raise AssertionError("development preflight must contain six unique episode IDs")
    return selected


def build_schedule(episodes: list[Episode], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    schedule = []
    for episode in episodes:
        arms = list(StudyArm)
        rng.shuffle(arms)
        for arm in arms:
            schedule.append(
                {
                    "schedule_index": len(schedule),
                    "episode_id": episode.episode_id,
                    "task_type": episode.task_type.value,
                    "arm": arm.value,
                    "hidden_arm_id": f"arm-{sha256_text(f'{seed}|{episode.episode_id}|{arm.value}')[:16]}",
                }
            )
    if len(schedule) != 36:
        raise AssertionError("real-LLM preflight requires exactly 36 runs")
    return schedule


def _belief_dependencies(episode: Episode) -> tuple[EvidenceBeliefDependency, ...]:
    return tuple(
        EvidenceBeliefDependency(
            evidence_id=item.evidence_id,
            slots=PUBLIC_EVIDENCE_TYPE_SLOTS[item.evidence_type],
            resources=item.depends_on_resources,
            source="episode-initial-last-known-valid",
        )
        for item in episode.evidence
    )


def controller_belief_state(episode: Episode, arm: StudyArm) -> ControllerBeliefLedger:
    dependencies = _belief_dependencies(episode)
    if arm in {StudyArm.LEDGER_ONLY, StudyArm.LEDGER_GUARD}:
        return ledger_only_belief_state(dependencies)
    if arm in {StudyArm.ORIGINAL_FULL, StudyArm.FULL_GUARD}:
        return dependency_scoped_belief_update(
            dependencies,
            changed_features=episode.drift_event.changed_features,
        )
    if arm is StudyArm.GLOBAL_GUARD:
        return ControllerBeliefLedger(
            tuple(
                ControllerEvidenceBelief(
                    item.evidence_id,
                    item.slots,
                    BeliefValidity.BELIEVED_INVALID,
                    "global-invalidation-policy",
                )
                for item in dependencies
            )
        )
    invalidated = set(episode.ground_truth.invalidated_artifact_ids)
    return ControllerBeliefLedger(
        tuple(
            ControllerEvidenceBelief(
                item.evidence_id,
                item.slots,
                (
                    BeliefValidity.BELIEVED_INVALID
                    if item.evidence_id in invalidated
                    else BeliefValidity.BELIEVED_VALID
                ),
                "oracle-invalidation-private-channel",
            )
            for item in dependencies
        )
    )


def _current_snapshot_id(episode: Episode) -> str:
    return str(episode.ground_truth.acceptable_payload["evidence_policy"]["current_snapshot_id"])


def public_task_contract(episode: Episode) -> PublicTaskContract:
    return PublicTaskContract(
        task_id=episode.episode_id,
        task_type=episode.task_type.value,
        backend_ids=tuple(episode.task_constraints["candidate_backends"]),
        qubit_ids=tuple(episode.task_constraints["candidate_qubits"]),
        circuit_ids=(episode.circuit,),
        snapshot_ids=(str(episode.initial_state["snapshot_id"]), _current_snapshot_id(episode)),
        mitigation_policy_ids=("keep_mitigation", "increase_mitigation"),
    )


def public_episode_prompt(episode: Episode) -> str:
    return canonical_json(
        {
            "instruction": "Complete the task for the current device state.",
            "task": {
                "task_id": episode.episode_id,
                "task_type": episode.task_type.value,
                "circuit_id": episode.circuit,
                "backend_id": episode.backend,
                "constraints": episode.task_constraints,
            },
            "snapshots": {
                "pre_drift": episode.initial_state["snapshot_id"],
                "current": _current_snapshot_id(episode),
            },
            "initial_device_state": episode.initial_state,
            "device_change": episode.drift_event.neutral_payload(),
            "initial_evidence": [
                {"evidence_id": item.evidence_id, "evidence_type": item.evidence_type}
                for item in episode.evidence
            ],
        }
    )


def _tool_response(tool_name: str, episode: Episode) -> dict[str, Any]:
    candidates = list(episode.task_constraints["candidate_backends"])
    qubits = list(episode.task_constraints["candidate_qubits"])
    degraded = int(episode.evidence[0].depends_on_resources[0].split(".")[1]) if (
        episode.task_type is TaskType.QUBIT_MAPPING
    ) else qubits[0]
    if tool_name == "get_backend_health":
        return {"success": True, "avg_2q_error": 0.08, "available_capacity": 4}
    if tool_name == "compare_backends":
        return {"success": True, "ranked_backends": [candidates[1], candidates[0], *candidates[2:]]}
    if tool_name == "get_qubit_properties":
        return {"success": True, "available_qubits": qubits, "degraded_qubits": [degraded]}
    if tool_name == "get_coupling_map":
        return {"success": True, "edge_1_2_available": False}
    if tool_name == "transpile_circuit":
        return {"success": True, "compilation_snapshot_id": _current_snapshot_id(episode)}
    if tool_name == "run_circuit":
        return {"success": True, "measured_success": False}
    if tool_name == "apply_mitigation":
        return {"success": True, "estimated_gain": 0.20, "verified": True}
    return {"success": True, "context_refreshed": True}


def _json_object(raw: str) -> dict[str, Any]:
    value = raw.strip()
    if value.startswith("```"):
        lines = value.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("proposal must be a JSON object")
    return parsed


def _public_tool_request_error(
    candidate: CandidateToolAction,
    *,
    task: PublicTaskContract,
    belief: ControllerBeliefLedger,
) -> str | None:
    """Treatment-neutral execution-boundary validation for non-Guard arms."""
    schema = PUBLIC_TOOL_SCHEMAS.get(candidate.tool_name)
    if schema is None or set(candidate.arguments) != set(schema):
        return "invalid_tool_schema"
    for name, rule in schema.items():
        value = candidate.arguments[name]
        if rule.kind == "string":
            valid_type = isinstance(value, str) and (bool(value) or not rule.non_empty)
            values = (value,)
        elif rule.kind == "string_list":
            valid_type = (
                isinstance(value, list)
                and all(isinstance(item, str) and item for item in value)
                and len(value) == len(set(value))
                and (bool(value) or not rule.non_empty)
            )
            values = tuple(value) if isinstance(value, list) else ()
        elif rule.kind == "integer_list":
            valid_type = (
                isinstance(value, list)
                and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
                and len(value) == len(set(value))
                and (bool(value) or not rule.non_empty)
            )
            values = tuple(value) if isinstance(value, list) else ()
        else:
            return "invalid_tool_schema"
        if not valid_type:
            return "invalid_tool_schema"
        if rule.entity_domain and any(item not in task.entity_domain(rule.entity_domain) for item in values):
            return "unknown_public_entity"
    if candidate.trigger_evidence_id is not None and candidate.trigger_evidence_id not in belief.by_id():
        return "unknown_trigger_evidence"
    return None


def _usage_total(results: list[CompletionResult]) -> dict[str, int]:
    totals = {
        key: sum(int(item.usage.get(key) or 0) for item in results)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    cache_hit = 0
    cache_miss = 0
    for item in results:
        details = item.usage.get("prompt_tokens_details") or {}
        hit = int(item.usage.get("prompt_cache_hit_tokens") or details.get("cached_tokens") or 0)
        miss_value = item.usage.get("prompt_cache_miss_tokens")
        miss = int(miss_value) if miss_value is not None else max(
            0, int(item.usage.get("prompt_tokens") or 0) - hit
        )
        cache_hit += hit
        cache_miss += miss
    totals["prompt_cache_hit_tokens"] = cache_hit
    totals["prompt_cache_miss_tokens"] = cache_miss
    return totals


def _estimated_cost(usage: Mapping[str, int], config: Mapping[str, Any]) -> float:
    rates = config["pricing"]["per_million_tokens"]
    return round(
        (
            usage.get("prompt_cache_hit_tokens", 0) * float(rates["input_cache_hit"])
            + usage.get("prompt_cache_miss_tokens", 0) * float(rates["input_cache_miss"])
            + usage.get("completion_tokens", 0) * float(rates["output"])
        )
        / 1_000_000,
        10,
    )


def run_preflight_trace(
    *,
    episode: Episode,
    arm: StudyArm,
    schedule_item: Mapping[str, Any],
    client: DeepSeekCompletionClient,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    trace_id = "ag11-" + sha256_text(
        f"{config['config_sha256']}|{schedule_item['schedule_index']}|{episode.episode_id}|{arm.value}"
    )[:20]
    task = public_task_contract(episode)
    belief = controller_belief_state(episode, arm)
    initial_belief = belief
    guard_session = GuardRuntimeSessionV1_1() if arm in GUARD_ARMS else None
    public_prompt = public_episode_prompt(episode)
    messages = [
        {"role": "system", "content": SHARED_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": public_prompt},
    ]
    accepted_calls: list[AcceptedToolCall] = []
    raw_tool_calls: list[dict[str, Any]] = []
    completions: list[CompletionResult] = []
    api_errors: list[dict[str, Any]] = []
    parser_errors: list[dict[str, Any]] = []
    runtime_rejections: list[dict[str, Any]] = []
    turns = 0
    cost_units = 0
    format_attempts = 0
    terminal_raw: dict[str, Any] | None = None
    fatal_error: str | None = None
    started_at = utc_now()

    while turns < int(config["max_turns"]):
        used_completion_tokens = sum(int(item.usage.get("completion_tokens") or 0) for item in completions)
        remaining_tokens = int(config["max_tokens"]) - used_completion_tokens
        if remaining_tokens <= 0:
            fatal_error = "completion_token_budget_exhausted"
            break
        messages.append(
            {
                "role": "system",
                "content": canonical_json(
                    {
                        "controller_visible_belief_ledger_B_t": [
                            {
                                "evidence_id": item.evidence_id,
                                "slots": list(item.slots),
                                "belief_validity": item.belief_validity.value,
                                "produced_by_tool_call_id": item.produced_by_tool_call_id,
                            }
                            for item in belief.records
                        ],
                        "remaining_budget": {
                            "turns": int(config["max_turns"]) - turns,
                            "tool_calls": int(config["max_tool_calls"]) - len(raw_tool_calls),
                            "cost_units": int(config["max_cost_units"]) - cost_units,
                            "completion_tokens": remaining_tokens,
                        },
                    }
                ),
            }
        )
        result = None
        for attempt in range(1, int(config["retry"]["max_attempts"]) + 1):
            try:
                result = client.complete(messages, remaining_tokens)
                break
            except Exception as exc:
                api_errors.append(
                    {
                        "turn": turns + 1,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "timestamp": utc_now(),
                    }
                )
                if attempt < int(config["retry"]["max_attempts"]):
                    time.sleep(float(config["retry"]["fixed_backoff_seconds"]))
        if result is None:
            fatal_error = "api_attempts_exhausted"
            break
        turns += 1
        completions.append(result)
        messages.append({"role": "assistant", "content": result.content})
        try:
            proposal = _json_object(result.content)
        except Exception as exc:
            format_attempts += 1
            parser_errors.append(
                {"turn": turns, "error_type": type(exc).__name__, "error_message": str(exc)}
            )
            if format_attempts >= int(config["format_max_attempts"]):
                fatal_error = "format_attempts_exhausted"
                break
            messages.append(
                {
                    "role": "user",
                    "content": canonical_json(
                        {
                            "error": "proposal_format_failure",
                            "instruction": "Return one JSON tool or final proposal matching the shared schema.",
                        }
                    ),
                }
            )
            continue

        if proposal.get("type") == "tool":
            raw_arguments = proposal.get("arguments")
            candidate = CandidateToolAction(
                tool_name=str(proposal.get("tool_name") or ""),
                arguments=dict(raw_arguments) if isinstance(raw_arguments, Mapping) else {},
                trigger_evidence_id=proposal.get("trigger_evidence_id"),
            )
            if guard_session is not None:
                guarded = guard_session.check_tool(
                    candidate,
                    task=task,
                    ledger=belief,
                    llm_turn_count=turns,
                    tool_call_count=len(raw_tool_calls),
                )
                if guarded.outcome.value != "ALLOW":
                    event = guard_session.events[-1]
                    if event.repair_allowed:
                        messages.append({"role": "user", "content": canonical_json(event.feedback)})
                        continue
                    fatal_error = "guard_repair_budget_exhausted"
                    break
            else:
                # Non-Guard arms receive only an ordinary failed tool request.
                request_error = _public_tool_request_error(candidate, task=task, belief=belief)
                if request_error:
                    runtime_rejections.append({"turn": turns, "error": request_error})
                    messages.append({"role": "user", "content": canonical_json({"tool_error": request_error})})
                    continue
            if len(raw_tool_calls) >= int(config["max_tool_calls"]):
                fatal_error = "tool_call_budget_exhausted"
                break
            tool_cost = TOOL_COSTS.get(candidate.tool_name)
            if tool_cost is None or cost_units + tool_cost > int(config["max_cost_units"]):
                fatal_error = "tool_cost_budget_exhausted"
                break
            call_index = len(raw_tool_calls) + 1
            output_id = deterministic_evidence_id(trace_id, call_index)
            response = _tool_response(candidate.tool_name, episode)
            raw_call = {
                "call_index": call_index,
                "tool_call_id": f"call:{call_index}",
                "tool_name": candidate.tool_name,
                "request": candidate.arguments,
                "response": response,
                "trigger_evidence_id": candidate.trigger_evidence_id,
                "output_evidence_id": output_id,
                "observed_snapshot_id": _current_snapshot_id(episode),
                "cost_units": tool_cost,
                "latency_seconds": 0.0,
                "turn": turns,
            }
            raw_tool_calls.append(raw_call)
            accepted_calls.append(
                AcceptedToolCall(
                    f"call:{call_index}",
                    candidate.tool_name,
                    candidate.arguments,
                    True,
                    (output_id,),
                )
            )
            belief = register_revalidated_belief(
                belief,
                evidence_id=output_id,
                slots=PUBLIC_TOOL_EVIDENCE_SLOTS[candidate.tool_name],
                source="accepted-current-tool-call",
                tool_call_id=f"call:{call_index}",
            )
            cost_units += tool_cost
            messages.append({"role": "user", "content": canonical_json({"tool_observation": raw_call})})
            continue

        if proposal.get("type") == "final" and isinstance(proposal.get("terminal"), dict):
            parsed = parse_terminal_runtime_output(
                canonical_json(proposal["terminal"]),
                known_evidence_ids=[item.evidence_id for item in belief.records],
                accepted_tool_calls=raw_tool_calls,
                format_attempt=format_attempts + 1,
            )
            if parsed.format_failure:
                format_attempts += 1
                parser_errors.append({"turn": turns, "error": parsed.format_failure})
                if parsed.retry_message and format_attempts < int(config["format_max_attempts"]):
                    messages.append(
                        {
                            "role": "user",
                            "content": canonical_json(
                                {
                                    "error": "terminal_schema_format_failure",
                                    "frozen_runtime_message": parsed.retry_message,
                                    "required_outer_shape": {
                                        "type": "final",
                                        "terminal": "<complete terminal-decision-2.0 object>",
                                    },
                                    "required_terminal_keys": [
                                        "schema_version",
                                        "status",
                                        "decision",
                                        "supporting_evidence_ids",
                                        "revalidation_actions",
                                        "failure",
                                    ],
                                    "required_decision_keys": [
                                        "task_type",
                                        "task_action",
                                        "payload",
                                    ],
                                    "instruction": (
                                        "Repair syntax only. Do not add an answer, entity, evidence ID, "
                                        "or tool call that was not already chosen or observed."
                                    ),
                                }
                            ),
                        }
                    )
                    continue
                fatal_error = "terminal_schema_format_failure"
                break
            if not parsed.accepted or parsed.decision is None:
                parser_errors.append(
                    {"turn": turns, "identifier_failures": list(parsed.identifier_failures)}
                )
                fatal_error = "terminal_identifier_failure"
                break
            if guard_session is not None:
                guarded = guard_session.check_final(
                    CandidateFinalAction(parsed.decision),
                    task=task,
                    ledger=belief,
                    accepted_tool_calls=accepted_calls,
                    llm_turn_count=turns,
                    tool_call_count=len(raw_tool_calls),
                )
                if guarded.outcome.value != "ALLOW":
                    event = guard_session.events[-1]
                    if event.repair_allowed:
                        messages.append({"role": "user", "content": canonical_json(event.feedback)})
                        continue
                    fatal_error = "guard_repair_budget_exhausted"
                    break
            terminal_raw = proposal["terminal"]
            break

        format_attempts += 1
        parser_errors.append({"turn": turns, "error": "unknown_proposal_shape"})
        if format_attempts >= int(config["format_max_attempts"]):
            fatal_error = "format_attempts_exhausted"
            break
        messages.append(
            {
                "role": "user",
                "content": canonical_json(
                    {
                        "error": "unknown_proposal_shape",
                        "required_tool_outer_keys": [
                            "type",
                            "tool_name",
                            "arguments",
                            "trigger_evidence_id",
                        ],
                        "required_final_outer_keys": ["type", "terminal"],
                        "instruction": (
                            "Use type=tool or type=final. For a terminal proposal, set "
                            "type to final and place the terminal-decision object under terminal."
                        ),
                    }
                ),
            }
        )

    if terminal_raw is None and fatal_error is None:
        fatal_error = "max_turns_without_terminal"
    usage = _usage_total(completions)
    trace = {
        "preflight_version": PREFLIGHT_VERSION,
        "development_only": True,
        "trace_id": trace_id,
        "schedule": dict(schedule_item),
        "episode_id": episode.episode_id,
        "arm": arm.value,
        "guard_enabled": arm in GUARD_ARMS,
        "oracle_channel_used": arm is StudyArm.ORACLE_INVALIDATION_GUARD,
        "public_prompt": public_prompt,
        "public_prompt_sha256": sha256_text(public_prompt),
        "shared_system_prompt_sha256": SHARED_AGENT_SYSTEM_PROMPT_SHA256,
        "initial_belief_ledger_B_t": [
            {
                "evidence_id": item.evidence_id,
                "belief_validity": item.belief_validity.value,
                "slots": list(item.slots),
            }
            for item in initial_belief.records
        ],
        "final_belief_ledger_B_t": [
            {
                "evidence_id": item.evidence_id,
                "belief_validity": item.belief_validity.value,
                "slots": list(item.slots),
                "produced_by_tool_call_id": item.produced_by_tool_call_id,
            }
            for item in belief.records
        ],
        "messages": messages,
        "accepted_tool_calls": raw_tool_calls,
        "guard_events": [asdict(item) for item in guard_session.events] if guard_session else [],
        "terminal": terminal_raw,
        "trace_complete": terminal_raw is not None,
        "unscorable_reason": fatal_error,
        "budget_usage": {
            "turns": turns,
            "tool_calls": len(raw_tool_calls),
            "cost_units": cost_units,
            "completion_tokens": usage.get("completion_tokens", 0),
        },
        "llm": {
            "requested_model": config["requested_model"],
            "returned_model_ids": sorted({item.returned_model for item in completions if item.returned_model}),
            "system_fingerprints": sorted({item.system_fingerprint for item in completions if item.system_fingerprint}),
            "usage": usage,
            "latency_seconds": round(sum(item.latency_seconds for item in completions), 6),
            "estimated_cost_usd": _estimated_cost(usage, config),
            "api_errors": api_errors,
            "parser_errors": parser_errors,
            "runtime_rejections": runtime_rejections,
            "responses": [asdict(item) for item in completions],
        },
        "frozen_config": {
            key: config[key]
            for key in (
                "requested_model",
                "thinking",
                "temperature",
                "top_p",
                "max_tokens",
                "max_turns",
                "max_tool_calls",
                "max_cost_units",
                "request_timeout_seconds",
                "retry",
                "format_max_attempts",
                "guard_max_repair_attempts",
            )
        },
        "started_at": started_at,
        "finished_at": utc_now(),
    }
    secret_tokens = {"api_key", "authorization", "deepseek_api_key"}
    lowered = canonical_json(trace).lower()
    if any(f'"{item}"' in lowered for item in secret_tokens):
        raise RuntimeError("secret-like key found in preflight trace")
    return trace


__all__ = [
    "DEVELOPMENT_NAMESPACE",
    "GUARD_ARMS",
    "PREFLIGHT_VERSION",
    "SHARED_AGENT_SYSTEM_PROMPT",
    "SHARED_AGENT_SYSTEM_PROMPT_SHA256",
    "DeepSeekCompletionClient",
    "build_schedule",
    "canonical_json",
    "controller_belief_state",
    "generate_development_preflight_episodes",
    "run_preflight_trace",
]
