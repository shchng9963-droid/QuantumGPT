"""Real-LLM B1 smoke pilot for ReliabilityBench-Q.

B1 is a development workflow check, not a paper-level performance experiment.
The public prompt, model configuration, tools, budgets, retry policy, and
scorer are frozen across arms.  Arms differ only in controller-maintained
state supplied through a private state channel.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

from .b0 import (
    B0_GROUPS,
    SHARED_BUDGET,
    TASK_PRIMARY_TOOLS,
    TOOL_COSTS,
    _canonical_json,
    _event_impacts_dependency,
    _public_input,
    _task_changed,
    _tool_output,
)
from .groups import ExperimentGroup
from .judges import DecisionSubmission, judge_episode
from .schema import Episode


B1_VERSION = "reliabilitybench-q/b1-smoke-1.0"
CORE_TRACE_FORBIDDEN_KEYS = {
    "api_key",
    "authorization",
    "anthropic_api_key",
    "deepseek_api_key",
    "openai_api_key",
}
NON_ORACLE_PRIVATE_FORBIDDEN = (
    "ground_truth",
    "acceptable_actions",
    "acceptable_payload",
    "required_revalidation_actions",
    "affected_resources",
    "oracle_private_channel",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_b1_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_version",
        "provider",
        "base_url",
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
        "schedule_seed",
        "preflight_schedule_seed",
        "groups",
        "episode_ids",
        "preflight_episode_id",
        "pricing",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"B1 config missing keys: {sorted(missing)}")
    if config["protocol_version"] != B1_VERSION:
        raise ValueError("B1 config version does not match runner version")
    if tuple(config["groups"]) != tuple(group.value for group in B0_GROUPS):
        raise ValueError("B1 must use the frozen six-group B0 ordering")
    if len(config["episode_ids"]) != 24:
        raise ValueError("B1 requires exactly 24 frozen episode IDs")
    if len(set(config["episode_ids"])) != 24:
        raise ValueError("B1 episode IDs must be unique")
    if config["preflight_episode_id"] not in config["episode_ids"]:
        raise ValueError("preflight episode must belong to the frozen 24")
    budget = {
        "max_tokens": SHARED_BUDGET.max_tokens,
        "max_turns": SHARED_BUDGET.max_turns,
        "max_tool_calls": SHARED_BUDGET.max_tool_calls,
        "max_cost_units": SHARED_BUDGET.max_cost_units,
    }
    for key, expected in budget.items():
        if int(config[key]) != expected:
            raise ValueError(f"B1 {key} must equal frozen shared budget {expected}")
    return config


def select_frozen_episodes(
    episodes: Iterable[Episode], config: Mapping[str, Any]
) -> list[Episode]:
    by_id = {episode.episode_id: episode for episode in episodes}
    missing = [item for item in config["episode_ids"] if item not in by_id]
    if missing:
        raise ValueError(f"frozen B1 episodes missing from benchmark: {missing}")
    selected = [by_id[item] for item in config["episode_ids"]]
    task_counts = Counter(item.task_type.value for item in selected)
    relevance_counts = Counter(item.drift_event.relevance.value for item in selected)
    if set(task_counts.values()) != {4} or len(task_counts) != 6:
        raise ValueError("B1 requires four episodes from each of six task types")
    if relevance_counts != {"related": 12, "unrelated": 12}:
        raise ValueError("B1 requires 12 related and 12 unrelated episodes")
    return selected


def build_schedule(
    config: Mapping[str, Any], *, preflight: bool
) -> list[dict[str, Any]]:
    seed = int(
        config["preflight_schedule_seed"] if preflight else config["schedule_seed"]
    )
    rng = random.Random(seed)
    episode_ids = (
        [config["preflight_episode_id"]] if preflight else list(config["episode_ids"])
    )
    if not preflight:
        rng.shuffle(episode_ids)
    schedule: list[dict[str, Any]] = []
    for episode_index, episode_id in enumerate(episode_ids):
        groups = list(config["groups"])
        rng.shuffle(groups)
        for within_episode_index, group in enumerate(groups):
            opaque_group_id = sha256_text(f"{seed}|{episode_id}|{group}")[:16]
            schedule.append(
                {
                    "schedule_index": len(schedule),
                    "episode_index": episode_index,
                    "within_episode_index": within_episode_index,
                    "episode_id": episode_id,
                    "controller_group": group,
                    "hidden_group_id": f"arm-{opaque_group_id}",
                }
            )
    expected = 6 if preflight else 144
    if len(schedule) != expected:
        raise AssertionError(f"unexpected B1 schedule length: {len(schedule)}")
    return schedule


def benchmark_sha256(episodes: Iterable[Episode]) -> str:
    payload = [item.to_dict() for item in episodes]
    return sha256_text(_canonical_json(payload))


def git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, encoding="utf-8"
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tracked_worktree_dirty": bool(
            run("status", "--short", "--untracked-files=no")
        ),
    }


def build_manifest(
    root: Path,
    config_path: Path,
    episodes: list[Episode],
    *,
    preflight: bool,
) -> dict[str, Any]:
    config = load_b1_config(config_path)
    frozen_paths = (
        config_path,
        root / "eval/reliability_bench/b1.py",
        root / "eval/reliability_bench/b0.py",
        root / "eval/reliability_bench/judges.py",
        root / "eval/reliability_bench/schema.py",
        root / "eval/reliability_bench/data/stage_a_synthetic_v1.jsonl",
        root / "scripts/run_reliability_bench_b1.py",
    )
    hashes = {str(path.relative_to(root)): sha256_file(path) for path in frozen_paths}
    return {
        "b1_version": B1_VERSION,
        "created_at": utc_now(),
        "mode": "preflight" if preflight else "formal",
        "config": config,
        "config_sha256": sha256_file(config_path),
        "benchmark_sha256": benchmark_sha256(episodes),
        "frozen_file_sha256": hashes,
        "git": git_metadata(root),
        "scorer_frozen": True,
        "scorer_change_policy": (
            "Any scorer bug requires a new version and rerun of every affected arm."
        ),
    }


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


class CompletionClient(Protocol):
    def complete(
        self, messages: list[dict[str, str]], max_tokens: int
    ) -> CompletionResult: ...


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

    def complete(
        self, messages: list[dict[str, str]], max_tokens: int
    ) -> CompletionResult:
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
        raw = response.model_dump(mode="json")
        usage = response.usage.model_dump(mode="json") if response.usage else {}
        choice = response.choices[0]
        return CompletionResult(
            content=choice.message.content or "",
            response_id=getattr(response, "id", None),
            returned_model=getattr(response, "model", None),
            system_fingerprint=getattr(response, "system_fingerprint", None),
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
            latency_seconds=latency,
            request_started_at=started_at,
            request_finished_at=utc_now(),
            raw_response=raw,
        )


COMMON_SYSTEM_PROMPT = """\
You are the decision agent in ReliabilityBench-Q. Return exactly one JSON object
on each turn. Do not use markdown and do not reveal chain-of-thought.

Choose one form:
1) {"type":"tool","tool_name":"<catalog name>","trigger_evidence_id":"<evidence ID>"}
2) {"type":"final","decision":{...}}

The final decision object must contain these keys:
action, used_artifact_ids, revalidation_actions, selected_backend,
selected_qubits, compilation_snapshot_id, claimed_success,
supporting_artifact_ids, mitigation_action, declared_unreachable, metadata.
Use null for inapplicable scalar fields and [] for inapplicable list fields.
Only cite evidence IDs present in the public task. Tool outputs are current
observations; do not claim that a pre-change artifact is current merely because
it was re-queried. The controller records actually executed tools independently,
so never claim a revalidation action that was not executed.

Task action conventions:
- backend_selection: action=select_backend and selected_backend is required.
- qubit_mapping: action=select_qubits and selected_qubits is required.
- transpilation: action is retranspile or reuse_compilation and
  compilation_snapshot_id is required.
- fidelity_claim: action is rerun_circuit or reuse_valid_result and
  claimed_success is required; a success claim needs supporting_artifact_ids.
- mitigation_decision: action is reassess_mitigation or keep_mitigation and
  mitigation_action is required.
- unreachable_target: action is declare_unreachable or continue_execution and
  declared_unreachable is required.

Stay within the shared tool, cost, turn, and token budgets. The controller state
is authoritative state maintained by the assigned mechanism. It never specifies
the correct final action; decide using the public task and observations.
"""


def _private_controller_state(
    episode: Episode,
    group: ExperimentGroup,
    public_input: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...], bool]:
    if group is ExperimentGroup.REACT:
        return {}, (), False
    if group is ExperimentGroup.MONITOR_ONLY:
        return {"drift_monitor": {"event_observed": True}}, (), False

    ledger = [
        {
            "evidence_id": item.evidence_id,
            "evidence_type": item.evidence_type,
            "depends_on_resources": list(item.depends_on_resources),
            "validity": "unchecked_after_event",
        }
        for item in episode.evidence
    ]
    if group is ExperimentGroup.LEDGER_ONLY:
        return {"evidence_ledger": ledger}, (), False
    if group is ExperimentGroup.GLOBAL_REVALIDATE:
        invalidated = tuple(item.evidence_id for item in episode.evidence)
    elif group is ExperimentGroup.FULL_SELECTIVE:
        features = public_input["device_change"]["changed_features"]
        invalidated = tuple(
            item.evidence_id
            for item in episode.evidence
            if any(
                _event_impacts_dependency(feature, dependency)
                for feature in features
                for dependency in item.depends_on_resources
            )
        )
    elif group is ExperimentGroup.ORACLE:
        invalidated = tuple(episode.ground_truth.invalidated_artifact_ids)
    else:
        raise ValueError(f"unsupported B1 group: {group.value}")
    invalidated_set = set(invalidated)
    for item in ledger:
        item["validity"] = (
            "invalid_after_event"
            if item["evidence_id"] in invalidated_set
            else "valid_after_event"
        )
    return (
        {"evidence_ledger": ledger},
        invalidated,
        group is ExperimentGroup.ORACLE,
    )


def _parse_json_object(content: str) -> dict[str, Any]:
    value = content.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def _decision_from_dict(
    value: Mapping[str, Any], actual_revalidation_actions: tuple[str, ...]
) -> tuple[DecisionSubmission, dict[str, Any]]:
    declared = tuple(value.get("revalidation_actions") or ())
    metadata = dict(value.get("metadata") or {})
    metadata["model_declared_revalidation_actions"] = list(declared)
    decision = DecisionSubmission(
        action=str(value.get("action") or ""),
        used_artifact_ids=tuple(value.get("used_artifact_ids") or ()),
        revalidation_actions=actual_revalidation_actions,
        selected_backend=value.get("selected_backend"),
        selected_qubits=tuple(value.get("selected_qubits") or ()),
        compilation_snapshot_id=value.get("compilation_snapshot_id"),
        claimed_success=value.get("claimed_success"),
        supporting_artifact_ids=tuple(value.get("supporting_artifact_ids") or ()),
        mitigation_action=value.get("mitigation_action"),
        declared_unreachable=value.get("declared_unreachable"),
        metadata=metadata,
    )
    return decision, {"model_declared_revalidation_actions": list(declared)}


def _completion_tokens(usage: Mapping[str, Any]) -> int:
    return int(usage.get("completion_tokens") or 0)


def estimate_cost_usd(
    usage: Mapping[str, Any], pricing: Mapping[str, Any]
) -> dict[str, Any]:
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cache_hit = int(
        usage.get("prompt_cache_hit_tokens") or details.get("cached_tokens") or 0
    )
    cache_miss_raw = usage.get("prompt_cache_miss_tokens")
    cache_miss = (
        int(cache_miss_raw)
        if cache_miss_raw is not None
        else max(0, prompt_tokens - cache_hit)
    )
    rates = pricing["per_million_tokens"]
    cost = (
        cache_hit * float(rates["input_cache_hit"])
        + cache_miss * float(rates["input_cache_miss"])
        + completion_tokens * float(rates["output"])
    ) / 1_000_000
    return {
        "currency": pricing["currency"],
        "input_cache_hit_tokens": cache_hit,
        "input_cache_miss_tokens": cache_miss,
        "output_tokens": completion_tokens,
        "estimated_cost": round(cost, 10),
        "pricing_source": pricing["source"],
        "pricing_frozen_on": pricing["frozen_on"],
    }


def _merge_usage(results: Iterable[CompletionResult]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for result in results:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            totals[key] += int(result.usage.get(key) or 0)
        details = result.usage.get("prompt_tokens_details") or {}
        totals["prompt_cache_hit_tokens"] += int(
            result.usage.get("prompt_cache_hit_tokens")
            or details.get("cached_tokens")
            or 0
        )
        miss = result.usage.get("prompt_cache_miss_tokens")
        if miss is None:
            miss = int(result.usage.get("prompt_tokens") or 0) - int(
                result.usage.get("prompt_cache_hit_tokens")
                or details.get("cached_tokens")
                or 0
            )
        totals["prompt_cache_miss_tokens"] += max(0, int(miss))
    return dict(totals)


def _updated_private_state(
    base_state: Mapping[str, Any],
    completed_by_evidence: Mapping[str, set[str]],
) -> dict[str, Any]:
    state = json.loads(_canonical_json(base_state))
    for item in state.get("evidence_ledger", []):
        evidence_id = item["evidence_id"]
        required = set(TASK_PRIMARY_TOOLS[item["evidence_type"]])
        if required and required.issubset(
            completed_by_evidence.get(evidence_id, set())
        ):
            item["revalidation_status"] = "completed_with_current_tool_observation"
        else:
            item["revalidation_status"] = "not_completed"
    return state


def _trace_contains_forbidden_key(value: Any) -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in CORE_TRACE_FORBIDDEN_KEYS:
                hits.append(str(key))
            hits.extend(_trace_contains_forbidden_key(item))
    elif isinstance(value, list):
        for item in value:
            hits.extend(_trace_contains_forbidden_key(item))
    return hits


def run_b1_trace(
    episode: Episode,
    group: ExperimentGroup,
    client: CompletionClient,
    config: Mapping[str, Any],
    schedule_item: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    started_at = utc_now()
    public_input = _public_input(episode)
    public_prompt = public_input["actual_prompt"]
    private_base, controller_invalidated, oracle_used = _private_controller_state(
        episode, group, public_input
    )
    if group is not ExperimentGroup.ORACLE:
        private_text = _canonical_json(private_base).lower()
        if any(token in private_text for token in NON_ORACLE_PRIVATE_FORBIDDEN):
            raise RuntimeError("non-Oracle B1 controller received a forbidden label")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": COMMON_SYSTEM_PROMPT},
        {"role": "user", "content": public_prompt},
    ]
    tool_calls: list[dict[str, Any]] = []
    completed_by_evidence: dict[str, set[str]] = defaultdict(set)
    response_events: list[dict[str, Any]] = []
    completion_results: list[CompletionResult] = []
    parser_errors: list[dict[str, Any]] = []
    api_errors: list[dict[str, Any]] = []
    controller_validation_errors: list[dict[str, Any]] = []
    final_decision: DecisionSubmission | None = None
    normalization: dict[str, Any] = {}
    cost_units = 0
    turns = 0
    output_tokens = 0
    fatal_error: str | None = None
    evidence_ids = {item.evidence_id for item in episode.evidence}

    while turns < int(config["max_turns"]):
        remaining_tokens = int(config["max_tokens"]) - output_tokens
        if remaining_tokens <= 0:
            fatal_error = "completion_token_budget_exhausted"
            break
        state = _updated_private_state(private_base, completed_by_evidence)
        turn_context = {
            "controller_state": state,
            "remaining_budget": {
                "turns": int(config["max_turns"]) - turns,
                "tool_calls": int(config["max_tool_calls"]) - len(tool_calls),
                "cost_units": int(config["max_cost_units"]) - cost_units,
                "completion_tokens": remaining_tokens,
            },
        }
        messages.append({"role": "system", "content": _canonical_json(turn_context)})
        result: CompletionResult | None = None
        attempts = int(config["retry"]["max_attempts"])
        for attempt in range(1, attempts + 1):
            try:
                result = client.complete(messages, remaining_tokens)
                response_events.append(
                    {
                        "turn": turns + 1,
                        "attempt": attempt,
                        "status": "success",
                        "request_started_at": result.request_started_at,
                        "request_finished_at": result.request_finished_at,
                        "response_id": result.response_id,
                    }
                )
                break
            except Exception as exc:  # provider exceptions are retained in trace
                error = {
                    "turn": turns + 1,
                    "attempt": attempt,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "timestamp": utc_now(),
                }
                api_errors.append(error)
                response_events.append(error)
                if attempt < attempts:
                    sleep_fn(float(config["retry"]["fixed_backoff_seconds"]))
        if result is None:
            fatal_error = "api_attempts_exhausted"
            break

        turns += 1
        completion_results.append(result)
        output_tokens += _completion_tokens(result.usage)
        messages.append({"role": "assistant", "content": result.content})
        try:
            action = _parse_json_object(result.content)
        except Exception as exc:
            parser_errors.append(
                {
                    "turn": turns,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "raw_response": result.content,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": _canonical_json(
                        {
                            "error": "invalid_json_action",
                            "instruction": "Return one valid JSON action object.",
                        }
                    ),
                }
            )
            continue

        action_type = action.get("type")
        if action_type == "tool":
            tool_name = str(action.get("tool_name") or "")
            evidence_id = str(action.get("trigger_evidence_id") or "")
            rejection: str | None = None
            if tool_name not in TOOL_COSTS:
                rejection = "unknown_tool"
            elif evidence_id not in evidence_ids:
                rejection = "unknown_evidence_id"
            elif len(tool_calls) >= int(config["max_tool_calls"]):
                rejection = "tool_call_budget_exhausted"
            elif cost_units + TOOL_COSTS[tool_name] > int(config["max_cost_units"]):
                rejection = "tool_cost_budget_exhausted"
            if rejection:
                messages.append(
                    {
                        "role": "user",
                        "content": _canonical_json(
                            {"tool_error": rejection, "request": action}
                        ),
                    }
                )
                continue
            cost = TOOL_COSTS[tool_name]
            response = _tool_output(tool_name, episode, _task_changed(episode))
            call = {
                "call_index": len(tool_calls) + 1,
                "turn": turns,
                "tool_name": tool_name,
                "trigger_evidence_id": evidence_id,
                "request": {"run_id": public_input["run_id"]},
                "response": response,
                "cost_units": cost,
                "timestamp": utc_now(),
            }
            tool_calls.append(call)
            cost_units += cost
            completed_by_evidence[evidence_id].add(tool_name)
            messages.append(
                {
                    "role": "user",
                    "content": _canonical_json({"tool_observation": call}),
                }
            )
            continue
        if action_type == "final" and isinstance(action.get("decision"), dict):
            decision_payload = action["decision"]
            referenced = set(decision_payload.get("used_artifact_ids") or ()) | set(
                decision_payload.get("supporting_artifact_ids") or ()
            )
            invalid_references = sorted(referenced.intersection(controller_invalidated))
            if invalid_references:
                validation_error = {
                    "turn": turns,
                    "error": "invalid_pre_event_evidence_reference",
                    "invalid_evidence_ids": invalid_references,
                    "instruction": (
                        "These IDs identify invalid pre-event artifacts. Current "
                        "tool observations are already recorded by the controller; "
                        "remove the invalid IDs from used_artifact_ids and "
                        "supporting_artifact_ids, and do not invent replacement IDs."
                    ),
                }
                controller_validation_errors.append(validation_error)
                messages.append(
                    {"role": "user", "content": _canonical_json(validation_error)}
                )
                continue
            actual_actions = tuple(
                dict.fromkeys(item["tool_name"] for item in tool_calls)
            )
            final_decision, normalization = _decision_from_dict(
                decision_payload, actual_actions
            )
            break
        messages.append(
            {
                "role": "user",
                "content": _canonical_json(
                    {
                        "error": "unknown_action_shape",
                        "instruction": "Use type=tool or type=final.",
                    }
                ),
            }
        )

    if final_decision is None and fatal_error is None:
        fatal_error = "max_turns_without_final_decision"
    legacy_program_judge_enabled = bool(
        config.get("legacy_program_judge_enabled", True)
    )
    judge = (
        judge_episode(episode, final_decision)
        if final_decision and legacy_program_judge_enabled
        else None
    )
    usage = _merge_usage(completion_results)
    pricing_usage = dict(usage)
    pricing_usage["prompt_cache_hit_tokens"] = usage.get("prompt_cache_hit_tokens", 0)
    pricing_usage["prompt_cache_miss_tokens"] = usage.get("prompt_cache_miss_tokens", 0)
    cost = estimate_cost_usd(pricing_usage, config["pricing"])
    trace_id = (
        "b1-"
        + sha256_text(
            f"{manifest['config_sha256']}|{schedule_item['schedule_index']}|"
            f"{episode.episode_id}|{group.value}"
        )[:20]
    )
    trace = {
        "b1_version": B1_VERSION,
        "trace_id": trace_id,
        "mode": manifest["mode"],
        "schedule": dict(schedule_item),
        "episode": {
            "episode_id": episode.episode_id,
            "pair_id": episode.pair_id,
            "task_type": episode.task_type.value,
            "relevance": episode.drift_event.relevance.value,
            "magnitude_quantile": episode.drift_event.magnitude_quantile,
        },
        "controller": {
            "group": group.value,
            "hidden_group_id": schedule_item["hidden_group_id"],
            "oracle_channel_used": oracle_used,
            "computed_invalidated_artifact_ids": list(controller_invalidated),
            "private_state_initial": private_base,
            "private_state_final": _updated_private_state(
                private_base, completed_by_evidence
            ),
        },
        "frozen_context": {
            "git_commit": manifest["git"]["commit"],
            "config_sha256": manifest["config_sha256"],
            "benchmark_sha256": manifest["benchmark_sha256"],
            "public_prompt_sha256": public_input["actual_prompt_sha256"],
            "requested_model": config["requested_model"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "max_tokens": config["max_tokens"],
            "max_turns": config["max_turns"],
            "max_tool_calls": config["max_tool_calls"],
            "max_cost_units": config["max_cost_units"],
            "retry": config["retry"],
            "legacy_program_judge_enabled": legacy_program_judge_enabled,
        },
        "public_prompt": public_prompt,
        "messages": messages,
        "tool_calls": tool_calls,
        "llm": {
            "response_events": response_events,
            "responses": [asdict(item) for item in completion_results],
            "returned_model_ids": sorted(
                {
                    item.returned_model
                    for item in completion_results
                    if item.returned_model
                }
            ),
            "system_fingerprints": sorted(
                {
                    item.system_fingerprint
                    for item in completion_results
                    if item.system_fingerprint
                }
            ),
            "usage": usage,
            "latency_seconds": round(
                sum(item.latency_seconds for item in completion_results), 6
            ),
            "cost": cost,
            "api_errors": api_errors,
            "parser_errors": parser_errors,
            "controller_validation_errors": controller_validation_errors,
        },
        "budget_usage": {
            "turns": turns,
            "tool_calls": len(tool_calls),
            "cost_units": cost_units,
            "completion_tokens": output_tokens,
        },
        "model_final_decision": (
            json.loads(_canonical_json(asdict(final_decision)))
            if final_decision
            else None
        ),
        "decision_normalization": normalization,
        "program_judge": (
            json.loads(_canonical_json(asdict(judge))) if judge else None
        ),
        "trace_complete": final_decision is not None,
        "unscorable": final_decision is None,
        "unscorable_reason": fatal_error,
        "started_at": started_at,
        "finished_at": utc_now(),
    }
    forbidden = _trace_contains_forbidden_key(trace)
    if forbidden:
        raise RuntimeError(f"secret-like keys found in B1 trace: {forbidden}")
    return trace


def prompt_and_label_audit(traces: list[dict[str, Any]]) -> dict[str, Any]:
    prompts: dict[str, set[str]] = defaultdict(set)
    non_oracle_violations: list[str] = []
    full_oracle_violations: list[str] = []
    secret_key_violations: list[str] = []
    returned_models: set[str] = set()
    for trace in traces:
        episode_id = trace["episode"]["episode_id"]
        prompts[episode_id].add(trace["frozen_context"]["public_prompt_sha256"])
        group = trace["controller"]["group"]
        private_text = _canonical_json(
            trace["controller"]["private_state_initial"]
        ).lower()
        if group != ExperimentGroup.ORACLE.value and any(
            token in private_text for token in NON_ORACLE_PRIVATE_FORBIDDEN
        ):
            non_oracle_violations.append(trace["trace_id"])
        if (
            group == ExperimentGroup.FULL_SELECTIVE.value
            and trace["controller"]["oracle_channel_used"]
        ):
            full_oracle_violations.append(trace["trace_id"])
        if _trace_contains_forbidden_key(trace):
            secret_key_violations.append(trace["trace_id"])
        returned_models.update(trace["llm"]["returned_model_ids"])
    return {
        "all_groups_share_public_prompt_per_episode": all(
            len(values) == 1 for values in prompts.values()
        ),
        "non_oracle_label_violations": non_oracle_violations,
        "full_oracle_channel_violations": full_oracle_violations,
        "secret_key_violations": secret_key_violations,
        "returned_model_ids": sorted(returned_models),
        "passed": (
            all(len(values) == 1 for values in prompts.values())
            and not non_oracle_violations
            and not full_oracle_violations
            and not secret_key_violations
        ),
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def build_b1_report(
    traces: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    *,
    preflight: bool,
) -> dict[str, Any]:
    audit = prompt_and_label_audit(traces)
    by_group: dict[str, dict[str, Any]] = {}
    for group in manifest["config"]["groups"]:
        selected = [item for item in traces if item["controller"]["group"] == group]
        scoreable = [item for item in selected if item["program_judge"] is not None]
        wrong = [item for item in scoreable if not item["program_judge"]["correct"]]
        stale = [
            item for item in scoreable if item["program_judge"]["stale_evidence_reuse"]
        ]
        unsupported = [
            item
            for item in scoreable
            if item["program_judge"]["unsupported_success_claim"]
        ]
        missing_revalidation = [
            item
            for item in scoreable
            if "missing_required_revalidation" in item["program_judge"]["error_codes"]
        ]
        total_cost = sum(
            float(item["llm"]["cost"]["estimated_cost"]) for item in selected
        )
        by_group[group] = {
            "runs": len(selected),
            "completion_rate": _rate(
                sum(bool(item["trace_complete"]) for item in selected), len(selected)
            ),
            "scoreable_rate": _rate(len(scoreable), len(selected)),
            "wrong_decision_rate": _rate(len(wrong), len(scoreable)),
            "stale_evidence_reuse_rate": _rate(len(stale), len(scoreable)),
            "unsupported_success_claim_rate": _rate(len(unsupported), len(scoreable)),
            "necessary_revalidation_success_rate": _rate(
                len(scoreable) - len(missing_revalidation), len(scoreable)
            ),
            "tool_calls": sum(item["budget_usage"]["tool_calls"] for item in selected),
            "tool_cost_units": sum(
                item["budget_usage"]["cost_units"] for item in selected
            ),
            "prompt_tokens": sum(
                item["llm"]["usage"].get("prompt_tokens", 0) for item in selected
            ),
            "completion_tokens": sum(
                item["llm"]["usage"].get("completion_tokens", 0) for item in selected
            ),
            "latency_seconds": round(
                sum(item["llm"]["latency_seconds"] for item in selected), 6
            ),
            "estimated_cost_usd": round(total_cost, 8),
            "api_error_attempts": sum(
                len(item["llm"]["api_errors"]) for item in selected
            ),
            "unscorable_runs": len(selected) - len(scoreable),
        }

    global_metrics = by_group[ExperimentGroup.GLOBAL_REVALIDATE.value]
    full_metrics = by_group[ExperimentGroup.FULL_SELECTIVE.value]
    ledger_metrics = by_group[ExperimentGroup.LEDGER_ONLY.value]
    oracle_metrics = by_group[ExperimentGroup.ORACLE.value]
    full_global_call_saving = (
        1.0 - full_metrics["tool_calls"] / global_metrics["tool_calls"]
        if global_metrics["tool_calls"]
        else None
    )
    api_failures = [item["api_error_attempts"] for item in by_group.values()]
    expected_runs = 6 if preflight else 144
    within_budget = all(
        item["budget_usage"]["turns"] <= int(manifest["config"]["max_turns"])
        and item["budget_usage"]["tool_calls"]
        <= int(manifest["config"]["max_tool_calls"])
        and item["budget_usage"]["cost_units"]
        <= int(manifest["config"]["max_cost_units"])
        and item["budget_usage"]["completion_tokens"]
        <= int(manifest["config"]["max_tokens"])
        for item in traces
    )
    acceptance = {
        "expected_run_count": len(traces) == expected_runs,
        "all_traces_complete": all(item["trace_complete"] for item in traces),
        "all_traces_scoreable": all(not item["unscorable"] for item in traces),
        "all_runs_within_frozen_budget": within_budget,
        "prompt_and_label_audit_passed": audit["passed"],
        "full_did_not_use_oracle_channel": not audit["full_oracle_channel_violations"],
        "no_secret_keys_in_logs": not audit["secret_key_violations"],
        "no_api_error_attempts": not any(api_failures),
        "single_returned_model_id": len(audit["returned_model_ids"]) == 1,
        "returned_model_matches_request": audit["returned_model_ids"]
        == [manifest["config"]["requested_model"]],
    }
    if preflight:
        full_traces = [
            item
            for item in traces
            if item["controller"]["group"] == ExperimentGroup.FULL_SELECTIVE.value
            and item["controller"]["computed_invalidated_artifact_ids"]
        ]
        acceptance["full_responded_to_invalid_state"] = bool(full_traces) and all(
            item["program_judge"] is not None
            and item["program_judge"]["correct"]
            and not (
                set(item["model_final_decision"]["used_artifact_ids"])
                | set(item["model_final_decision"]["supporting_artifact_ids"])
            ).intersection(item["controller"]["computed_invalidated_artifact_ids"])
            and bool(
                {
                    call["trigger_evidence_id"] for call in item["tool_calls"]
                }.intersection(item["controller"]["computed_invalidated_artifact_ids"])
            )
            for item in full_traces
        )
    return {
        "b1_version": B1_VERSION,
        "mode": "preflight" if preflight else "formal",
        "development_only": True,
        "significance_claims_allowed": False,
        "created_at": utc_now(),
        "manifest": dict(manifest),
        "run_count": len(traces),
        "prompt_and_label_audit": audit,
        "group_metrics": by_group,
        "comparisons": {
            "full_vs_global_tool_call_saving": (
                round(full_global_call_saving, 6)
                if full_global_call_saving is not None
                else None
            ),
            "full_minus_ledger_wrong_decision_rate": round(
                full_metrics["wrong_decision_rate"]
                - ledger_metrics["wrong_decision_rate"],
                6,
            ),
            "full_minus_oracle_wrong_decision_rate": round(
                full_metrics["wrong_decision_rate"]
                - oracle_metrics["wrong_decision_rate"],
                6,
            ),
            "full_minus_oracle_tool_calls": (
                full_metrics["tool_calls"] - oracle_metrics["tool_calls"]
            ),
        },
        "acceptance": acceptance,
        "acceptance_passed": all(acceptance.values()),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")


__all__ = [
    "B1_VERSION",
    "CompletionResult",
    "DeepSeekCompletionClient",
    "append_jsonl",
    "benchmark_sha256",
    "build_b1_report",
    "build_manifest",
    "build_schedule",
    "load_b1_config",
    "prompt_and_label_audit",
    "run_b1_trace",
    "select_frozen_episodes",
    "write_json",
]
