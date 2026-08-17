"""Deterministic B0 controller wiring audit for ReliabilityBench-Q.

B0 is a development-only pipeline test.  Its scripted policies are not agent
baselines and its correctness rates must not be reported as paper performance.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .groups import ExperimentGroup
from .judges import DecisionSubmission, judge_episode
from .schema import Episode, TaskType


B0_VERSION = "reliabilitybench-q/b0-1.0"
B0_GROUPS = (
    ExperimentGroup.REACT,
    ExperimentGroup.MONITOR_ONLY,
    ExperimentGroup.LEDGER_ONLY,
    ExperimentGroup.GLOBAL_REVALIDATE,
    ExperimentGroup.FULL_SELECTIVE,
    ExperimentGroup.ORACLE,
)


@dataclass(frozen=True)
class ToolBudget:
    max_tool_calls: int = 10
    max_cost_units: int = 10
    max_turns: int = 12
    max_tokens: int = 4096


@dataclass(frozen=True)
class ToolSpec:
    name: str
    cost_units: int


TOOL_SPECS = (
    ToolSpec("get_backend_health", 1),
    ToolSpec("compare_backends", 1),
    ToolSpec("get_qubit_properties", 1),
    ToolSpec("get_coupling_map", 1),
    ToolSpec("transpile_circuit", 3),
    ToolSpec("run_circuit", 5),
    ToolSpec("apply_mitigation", 3),
    ToolSpec("refresh_task_context", 1),
)
TOOL_COSTS = {item.name: item.cost_units for item in TOOL_SPECS}
SHARED_BUDGET = ToolBudget()


COMPONENT_MATRIX: Mapping[ExperimentGroup, dict[str, Any]] = {
    ExperimentGroup.REACT: {
        "raw_event_available": True,
        "drift_monitor": False,
        "evidence_ledger": False,
        "invalidation": "none",
        "revalidation": "scripted_reuse",
        "oracle_label_channel": False,
    },
    ExperimentGroup.MONITOR_ONLY: {
        "raw_event_available": True,
        "drift_monitor": True,
        "evidence_ledger": False,
        "invalidation": "none",
        "revalidation": "conservative_task_refresh",
        "oracle_label_channel": False,
    },
    ExperimentGroup.LEDGER_ONLY: {
        "raw_event_available": True,
        "drift_monitor": True,
        "evidence_ledger": True,
        "invalidation": "none",
        "revalidation": "scripted_reuse",
        "oracle_label_channel": False,
    },
    ExperimentGroup.GLOBAL_REVALIDATE: {
        "raw_event_available": True,
        "drift_monitor": True,
        "evidence_ledger": True,
        "invalidation": "global",
        "revalidation": "all_invalidated_evidence",
        "oracle_label_channel": False,
    },
    ExperimentGroup.FULL_SELECTIVE: {
        "raw_event_available": True,
        "drift_monitor": True,
        "evidence_ledger": True,
        "invalidation": "dependency_scoped",
        "revalidation": "invalidated_evidence_only",
        "oracle_label_channel": False,
    },
    ExperimentGroup.ORACLE: {
        "raw_event_available": True,
        "drift_monitor": True,
        "evidence_ledger": True,
        "invalidation": "oracle",
        "revalidation": "oracle_invalidated_evidence",
        "oracle_label_channel": True,
    },
}


TASK_PRIMARY_TOOLS: Mapping[str, tuple[str, ...]] = {
    "backend_ranking": ("get_backend_health", "compare_backends"),
    "qubit_mapping": ("get_qubit_properties",),
    "transpiled_circuit": ("get_coupling_map", "transpile_circuit"),
    "circuit_result": ("run_circuit",),
    "mitigation_estimate": ("apply_mitigation",),
    "feasibility_assessment": ("get_backend_health",),
    "task_context": ("refresh_task_context",),
}


TASK_RELEVANT_FEATURE: Mapping[TaskType, str] = {
    TaskType.BACKEND_SELECTION: "avg_2q_error",
    TaskType.QUBIT_MAPPING: "qubit_readout_error",
    TaskType.TRANSPILATION: "coupling_map",
    TaskType.FIDELITY_CLAIM: "avg_2q_error",
    TaskType.MITIGATION_DECISION: "avg_readout_error",
    TaskType.UNREACHABLE_TARGET: "backend_capacity",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _opaque_id(value: str, prefix: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _event_impacts_dependency(feature: str, dependency: str) -> bool:
    if feature in dependency:
        return True
    aliases = {
        "qubit_readout_error": ".readout_error",
        "coupling_map": "topology.",
        "backend_capacity": ".capacity",
    }
    alias = aliases.get(feature)
    return bool(alias and alias in dependency)


def _public_input(episode: Episode) -> dict[str, Any]:
    payload = {
        "run_id": _opaque_id(episode.episode_id, "run"),
        "task": {
            "type": episode.task_type.value,
            "circuit": episode.circuit,
            "backend": episode.backend,
            "constraints": episode.task_constraints,
        },
        "initial_device_state": episode.initial_state,
        "device_change": episode.drift_event.neutral_payload(),
        "evidence_refs": [
            {
                "evidence_id": item.evidence_id,
                "evidence_type": item.evidence_type,
            }
            for item in episode.evidence
        ],
        "tool_catalog": [asdict(item) for item in TOOL_SPECS],
        "budget": asdict(SHARED_BUDGET),
    }
    prompt_payload = {
        "instruction": (
            "Complete the task for the current device state. Use observations "
            "and tools only when the selected policy requires them."
        ),
        "input": payload,
    }
    payload["actual_prompt"] = _canonical_json(prompt_payload)
    payload["actual_prompt_sha256"] = hashlib.sha256(
        payload["actual_prompt"].encode("utf-8")
    ).hexdigest()
    return payload


def _ledger(episode: Episode, group: ExperimentGroup) -> list[dict[str, Any]]:
    if not COMPONENT_MATRIX[group]["evidence_ledger"]:
        return []
    return [
        {
            "evidence_id": item.evidence_id,
            "evidence_type": item.evidence_type,
            "depends_on_resources": list(item.depends_on_resources),
            "validity": "valid_pre_event",
        }
        for item in episode.evidence
    ]


def _select_invalidated(
    group: ExperimentGroup,
    public_input: dict[str, Any],
    ledger: list[dict[str, Any]],
    oracle_invalidated_ids: tuple[str, ...] | None,
) -> tuple[tuple[str, ...], str]:
    if group is not ExperimentGroup.ORACLE and oracle_invalidated_ids is not None:
        raise ValueError("Oracle labels were offered to a non-Oracle controller")
    if group is ExperimentGroup.GLOBAL_REVALIDATE:
        return tuple(item["evidence_id"] for item in ledger), "global_rule"
    if group is ExperimentGroup.FULL_SELECTIVE:
        features = public_input["device_change"]["changed_features"]
        selected = tuple(
            item["evidence_id"]
            for item in ledger
            if any(
                _event_impacts_dependency(feature, dependency)
                for feature in features
                for dependency in item["depends_on_resources"]
            )
        )
        return selected, "dependency_match_from_public_event"
    if group is ExperimentGroup.ORACLE:
        if oracle_invalidated_ids is None:
            raise ValueError("Oracle requires its private evaluator channel")
        return tuple(oracle_invalidated_ids), "oracle_private_channel"
    return (), "none"


def _primary_ref(public_input: dict[str, Any]) -> dict[str, str]:
    return next(
        item
        for item in public_input["evidence_refs"]
        if item["evidence_type"] != "task_context"
    )


def _task_changed(episode: Episode) -> bool:
    feature = TASK_RELEVANT_FEATURE[episode.task_type]
    return feature in episode.drift_event.changed_features


def _degraded_qubit(episode: Episode) -> int:
    primary = next(
        item for item in episode.evidence if item.evidence_type == "qubit_mapping"
    )
    resource = primary.depends_on_resources[0]
    return int(resource.split(".")[1])


def _tool_output(
    tool_name: str,
    episode: Episode,
    task_changed: bool,
) -> dict[str, Any]:
    if tool_name == "get_backend_health":
        return {
            "avg_2q_error": (
                0.08 if task_changed else episode.initial_state["avg_2q_error"]
            ),
            "available_capacity": (
                4
                if task_changed
                and episode.task_type is TaskType.UNREACHABLE_TARGET
                else 10
            ),
            "snapshot": "observed-after-event",
        }
    if tool_name == "compare_backends":
        return {
            "ranked_backends": (
                ["FakeSherbrooke", "FakeBrisbane", "FakeKyiv"]
                if task_changed
                else ["FakeBrisbane", "FakeSherbrooke", "FakeKyiv"]
            )
        }
    if tool_name == "get_qubit_properties":
        degraded = [_degraded_qubit(episode)] if task_changed else []
        return {
            "available_qubits": list(range(5)),
            "degraded_qubits": degraded,
        }
    if tool_name == "get_coupling_map":
        return {"edge_1_2_available": not task_changed}
    if tool_name == "transpile_circuit":
        return {
            "compilation_snapshot_id": (
                "post-drift" if task_changed else "pre-drift"
            )
        }
    if tool_name == "run_circuit":
        return {
            "measured_success": not task_changed,
            "execution_snapshot": "observed-after-event",
        }
    if tool_name == "apply_mitigation":
        return {"estimated_gain": 0.20 if task_changed else 0.02}
    return {"context_refreshed": True}


def _build_tool_plan(
    group: ExperimentGroup,
    public_input: dict[str, Any],
    invalidated_ids: tuple[str, ...],
) -> list[tuple[str, str, str]]:
    refs = {item["evidence_id"]: item for item in public_input["evidence_refs"]}
    targets: list[dict[str, str]] = []
    reason = "no_refresh"
    if group is ExperimentGroup.MONITOR_ONLY:
        targets = [_primary_ref(public_input)]
        reason = "monitor_conservative_refresh"
    elif invalidated_ids:
        targets = [refs[item] for item in invalidated_ids]
        reason = COMPONENT_MATRIX[group]["revalidation"]
    plan: list[tuple[str, str, str]] = []
    for target in targets:
        for tool_name in TASK_PRIMARY_TOOLS[target["evidence_type"]]:
            plan.append((tool_name, target["evidence_id"], reason))
    return plan


def _run_tools(
    plan: Iterable[tuple[str, str, str]],
    episode: Episode,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    task_changed = _task_changed(episode)
    calls: list[dict[str, Any]] = []
    cost = 0
    for index, (tool_name, evidence_id, reason) in enumerate(plan, start=1):
        tool_cost = TOOL_COSTS[tool_name]
        cost += tool_cost
        calls.append(
            {
                "call_index": index,
                "tool_name": tool_name,
                "trigger_evidence_id": evidence_id,
                "reason": reason,
                "cost_units": tool_cost,
                "request": {"run_id": _opaque_id(episode.episode_id, "run")},
                "response": _tool_output(tool_name, episode, task_changed),
            }
        )
    usage = {
        "tool_calls": len(calls),
        "cost_units": cost,
        "turns": 1 + len(calls),
        "tokens": 0,
    }
    if usage["tool_calls"] > SHARED_BUDGET.max_tool_calls:
        raise RuntimeError("B0 controller exceeded the shared tool-call budget")
    if usage["cost_units"] > SHARED_BUDGET.max_cost_units:
        raise RuntimeError("B0 controller exceeded the shared cost budget")
    if usage["turns"] > SHARED_BUDGET.max_turns:
        raise RuntimeError("B0 controller exceeded the shared turn budget")
    return calls, usage


def _observed_changed(
    task: TaskType,
    calls: list[dict[str, Any]],
) -> bool:
    responses = {
        item["tool_name"]: item["response"]
        for item in calls
    }
    if task is TaskType.BACKEND_SELECTION:
        ranking = responses.get("compare_backends", {}).get("ranked_backends", [])
        return bool(ranking and ranking[0] != "FakeBrisbane")
    if task is TaskType.QUBIT_MAPPING:
        return bool(
            responses.get("get_qubit_properties", {}).get("degraded_qubits")
        )
    if task is TaskType.TRANSPILATION:
        return (
            responses.get("transpile_circuit", {}).get(
                "compilation_snapshot_id"
            )
            == "post-drift"
        )
    if task is TaskType.FIDELITY_CLAIM:
        result = responses.get("run_circuit")
        return bool(result) and result["measured_success"] is False
    if task is TaskType.MITIGATION_DECISION:
        return (
            responses.get("apply_mitigation", {}).get("estimated_gain", 0.0)
            >= 0.10
        )
    health = responses.get("get_backend_health")
    return bool(health) and health["available_capacity"] < 5


def _decision(
    episode: Episode,
    public_input: dict[str, Any],
    calls: list[dict[str, Any]],
) -> DecisionSubmission:
    primary_id = _primary_ref(public_input)["evidence_id"]
    context_id = next(
        item["evidence_id"]
        for item in public_input["evidence_refs"]
        if item["evidence_type"] == "task_context"
    )
    changed = _observed_changed(episode.task_type, calls) if calls else False
    used_ids = (context_id,) if changed and calls else (primary_id,)
    actions = tuple(
        item["tool_name"]
        for item in calls
        if item["tool_name"] != "refresh_task_context"
    )
    if episode.task_type is TaskType.BACKEND_SELECTION:
        ranking_call = next(
            (item for item in calls if item["tool_name"] == "compare_backends"),
            None,
        )
        backend = (
            ranking_call["response"]["ranked_backends"][0]
            if ranking_call
            else "FakeBrisbane"
        )
        return DecisionSubmission(
            action="select_backend",
            used_artifact_ids=used_ids,
            revalidation_actions=actions,
            selected_backend=backend,
        )
    if episode.task_type is TaskType.QUBIT_MAPPING:
        property_call = next(
            (
                item
                for item in calls
                if item["tool_name"] == "get_qubit_properties"
            ),
            None,
        )
        degraded = set(
            property_call["response"]["degraded_qubits"] if property_call else []
        )
        qubits = tuple(item for item in range(5) if item not in degraded)[:3]
        return DecisionSubmission(
            action="select_qubits",
            used_artifact_ids=used_ids,
            revalidation_actions=actions,
            selected_qubits=qubits,
        )
    if episode.task_type is TaskType.TRANSPILATION:
        snapshot = "post-drift" if changed else "pre-drift"
        return DecisionSubmission(
            action="retranspile" if changed else "reuse_compilation",
            used_artifact_ids=used_ids,
            revalidation_actions=actions,
            compilation_snapshot_id=snapshot,
        )
    if episode.task_type is TaskType.FIDELITY_CLAIM:
        success = not changed
        support = used_ids if success else ()
        return DecisionSubmission(
            action="rerun_circuit" if changed else "reuse_valid_result",
            used_artifact_ids=used_ids,
            revalidation_actions=actions,
            claimed_success=success,
            supporting_artifact_ids=support,
        )
    if episode.task_type is TaskType.MITIGATION_DECISION:
        mitigation = "increase_mitigation" if changed else "keep_mitigation"
        return DecisionSubmission(
            action="reassess_mitigation" if changed else "keep_mitigation",
            used_artifact_ids=used_ids,
            revalidation_actions=actions,
            mitigation_action=mitigation,
        )
    return DecisionSubmission(
        action="declare_unreachable" if changed else "continue_execution",
        used_artifact_ids=used_ids,
        revalidation_actions=actions,
        declared_unreachable=changed,
        claimed_success=False if changed else None,
    )


def run_b0_controller(episode: Episode, group: ExperimentGroup) -> dict[str, Any]:
    if group not in B0_GROUPS:
        raise ValueError(f"group {group.value} is not part of B0")
    public_input = _public_input(episode)
    ledger = _ledger(episode, group)
    oracle_ids = (
        episode.ground_truth.invalidated_artifact_ids
        if group is ExperimentGroup.ORACLE
        else None
    )
    invalidated, invalidation_source = _select_invalidated(
        group, public_input, ledger, oracle_ids
    )
    plan = _build_tool_plan(group, public_input, invalidated)
    calls, usage = _run_tools(plan, episode)
    decision = _decision(episode, public_input, calls)
    return {
        "b0_version": B0_VERSION,
        "group": group.value,
        "controller_visible": public_input,
        "internal_components": dict(COMPONENT_MATRIX[group]),
        "internal_state": {
            "ledger": ledger,
            "computed_invalidated_artifact_ids": list(invalidated),
            "invalidation_source": invalidation_source,
            "oracle_channel_used": group is ExperimentGroup.ORACLE,
        },
        "tool_calls": calls,
        "budget_usage": usage,
        "final_decision": json.loads(_canonical_json(asdict(decision))),
        "trace_complete": True,
    }


def _score_trace(episode: Episode, controller: dict[str, Any]) -> dict[str, Any]:
    decision = DecisionSubmission(**controller["final_decision"])
    result = judge_episode(episode, decision)
    predicted = set(
        controller["internal_state"]["computed_invalidated_artifact_ids"]
    )
    expected = set(episode.ground_truth.invalidated_artifact_ids)
    false_invalidated = sorted(predicted - expected)
    triggered = {
        item["trigger_evidence_id"] for item in controller["tool_calls"]
    }
    return {
        "episode_id": episode.episode_id,
        "pair_id": episode.pair_id,
        "task_type": episode.task_type.value,
        "relevance": episode.drift_event.relevance.value,
        "program_judge": json.loads(_canonical_json(asdict(result))),
        "expected_invalidated_artifact_ids": sorted(expected),
        "false_invalidated_artifact_ids": false_invalidated,
        "false_invalidation_trigger_coverage": (
            len(set(false_invalidated) & triggered) / len(false_invalidated)
            if false_invalidated
            else 1.0
        ),
    }


def run_b0(episodes: Iterable[Episode]) -> list[dict[str, Any]]:
    traces = []
    for episode in episodes:
        for group in B0_GROUPS:
            controller = run_b0_controller(episode, group)
            traces.append(
                {
                    "trace_id": _opaque_id(
                        f"{episode.episode_id}|{group.value}", "trace"
                    ),
                    "controller": controller,
                    "evaluator": _score_trace(episode, controller),
                }
            )
    return traces


def _prompt_fairness_audit(traces: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_hashes: dict[str, set[str]] = defaultdict(set)
    tool_catalogs: set[str] = set()
    budgets: set[str] = set()
    forbidden = (
        "ground_truth",
        "affected_resources",
        '"relevance"',
        "acceptable_actions",
        "required_revalidation",
        "invalidated_artifact",
        "related-event",
        "unrelated-event",
    )
    leaking_trace_ids: list[str] = []
    for trace in traces:
        visible = trace["controller"]["controller_visible"]
        episode_id = trace["evaluator"]["episode_id"]
        prompt_hashes[episode_id].add(visible["actual_prompt_sha256"])
        tool_catalogs.add(_canonical_json(visible["tool_catalog"]))
        budgets.add(_canonical_json(visible["budget"]))
        prompt = visible["actual_prompt"].lower()
        if any(item in prompt for item in forbidden):
            leaking_trace_ids.append(trace["trace_id"])
    return {
        "episode_count": len(prompt_hashes),
        "all_six_groups_share_exact_prompt": all(
            len(items) == 1 for items in prompt_hashes.values()
        ),
        "shared_tool_catalog": len(tool_catalogs) == 1,
        "shared_budget": len(budgets) == 1,
        "prompt_leakage_trace_ids": leaking_trace_ids,
        "passed": (
            all(len(items) == 1 for items in prompt_hashes.values())
            and len(tool_catalogs) == 1
            and len(budgets) == 1
            and not leaking_trace_ids
        ),
    }


def _label_isolation_audit(traces: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    full_uses_oracle = False
    allowed_visible_keys = {
        "run_id",
        "task",
        "initial_device_state",
        "device_change",
        "evidence_refs",
        "tool_catalog",
        "budget",
        "actual_prompt",
        "actual_prompt_sha256",
    }
    allowed_event_keys = {
        "event_id",
        "phase",
        "changed_features",
        "change_magnitude",
        "magnitude_quantile",
    }
    for trace in traces:
        controller = trace["controller"]
        group = controller["group"]
        visible = controller["controller_visible"]
        visible_text = _canonical_json(visible).lower()
        if set(visible) != allowed_visible_keys:
            violations.append(trace["trace_id"])
        if set(visible["device_change"]) != allowed_event_keys:
            violations.append(trace["trace_id"])
        if any(
            item in visible_text
            for item in (
                "ground_truth",
                "affected_resources",
                '"relevance"',
                "acceptable_payload",
                "expected_invalidated_artifact_ids",
            )
        ):
            violations.append(trace["trace_id"])
        oracle_used = controller["internal_state"]["oracle_channel_used"]
        if group != ExperimentGroup.ORACLE.value and oracle_used:
            violations.append(trace["trace_id"])
        if group == ExperimentGroup.FULL_SELECTIVE.value and oracle_used:
            full_uses_oracle = True
        if group == ExperimentGroup.FULL_SELECTIVE.value:
            if (
                controller["internal_state"]["invalidation_source"]
                != "dependency_match_from_public_event"
            ):
                violations.append(trace["trace_id"])
    return {
        "ordinary_controller_visible_context_violations": sorted(set(violations)),
        "full_oracle_channel_used": full_uses_oracle,
        "evaluator_partitioned_after_controller": True,
        "passed": not violations and not full_uses_oracle,
    }


def _group_metrics(traces: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for group in B0_GROUPS:
        selected = [
            item
            for item in traces
            if item["controller"]["group"] == group.value
        ]
        correct = sum(
            item["evaluator"]["program_judge"]["correct"] for item in selected
        )
        stale = sum(
            item["evaluator"]["program_judge"]["stale_evidence_reuse"]
            for item in selected
        )
        calls = [
            call
            for item in selected
            for call in item["controller"]["tool_calls"]
        ]
        false_invalidated = sum(
            len(item["evaluator"]["false_invalidated_artifact_ids"])
            for item in selected
        )
        result[group.value] = {
            "runs": len(selected),
            "program_judge_correct": correct,
            "program_judge_wrong": len(selected) - correct,
            "stale_evidence_reuse": stale,
            "tool_calls": len(calls),
            "cost_units": sum(item["cost_units"] for item in calls),
            "false_invalidated_artifacts": false_invalidated,
            "tool_calls_by_name": dict(
                sorted(Counter(item["tool_name"] for item in calls).items())
            ),
        }
    return result


def _human_audit_export(
    traces: list[dict[str, Any]],
    seed: int = 20260817,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for trace in traces:
        evaluator = trace["evaluator"]
        by_task[evaluator["task_type"]][evaluator["episode_id"]].append(trace)
    sample: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for task_type in sorted(by_task):
        episode_ids = rng.sample(sorted(by_task[task_type]), 4)
        for episode_id in episode_ids:
            trace = rng.choice(by_task[task_type][episode_id])
            audit_id = _opaque_id(trace["trace_id"], "human-audit")
            controller = trace["controller"]
            sample.append(
                {
                    "audit_id": audit_id,
                    "task_type": task_type,
                    "public_prompt": controller["controller_visible"][
                        "actual_prompt"
                    ],
                    "tool_calls": controller["tool_calls"],
                    "final_decision": controller["final_decision"],
                    "human_correct": None,
                    "human_error_codes": [],
                    "human_notes": "",
                }
            )
            key.append(
                {
                    "audit_id": audit_id,
                    "trace_id": trace["trace_id"],
                    "episode_id": episode_id,
                    "group": controller["group"],
                    "program_judge": trace["evaluator"]["program_judge"],
                }
            )
    return sample, key


def build_b0_report(traces: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = _prompt_fairness_audit(traces)
    labels = _label_isolation_audit(traces)
    metrics = _group_metrics(traces)
    global_metrics = metrics[ExperimentGroup.GLOBAL_REVALIDATE.value]
    full_metrics = metrics[ExperimentGroup.FULL_SELECTIVE.value]
    ledger_metrics = metrics[ExperimentGroup.LEDGER_ONLY.value]
    related_by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for trace in traces:
        if trace["evaluator"]["relevance"] == "related":
            related_by_pair[trace["evaluator"]["pair_id"]][
                trace["controller"]["group"]
            ] = trace
    full_changes = all(
        pair[ExperimentGroup.FULL_SELECTIVE.value]["controller"]["final_decision"]
        != pair[ExperimentGroup.LEDGER_ONLY.value]["controller"]["final_decision"]
        for pair in related_by_pair.values()
    )
    global_coverage = all(
        item["evaluator"]["false_invalidation_trigger_coverage"] == 1.0
        for item in traces
        if item["controller"]["group"]
        == ExperimentGroup.GLOBAL_REVALIDATE.value
    )
    acceptance = {
        "exactly_720_runs": len(traces) == 720,
        "all_traces_complete": all(
            item["controller"]["trace_complete"] for item in traces
        ),
        "all_runs_programmatically_judged": all(
            isinstance(item["evaluator"]["program_judge"]["correct"], bool)
            for item in traces
        ),
        "prompt_fairness_passed": prompts["passed"],
        "label_isolation_passed": labels["passed"],
        "full_changes_decision_after_related_invalidation": full_changes,
        "ledger_can_reuse_stale_evidence": (
            ledger_metrics["stale_evidence_reuse"] > 0
        ),
        "global_false_invalidations_trigger_tools": global_coverage,
        "global_cost_exceeds_full_cost": (
            global_metrics["cost_units"] > full_metrics["cost_units"]
        ),
        "global_tool_calls_exceed_full_tool_calls": (
            global_metrics["tool_calls"] > full_metrics["tool_calls"]
        ),
        "all_runs_within_shared_budget": all(
            item["controller"]["budget_usage"]["tool_calls"]
            <= SHARED_BUDGET.max_tool_calls
            and item["controller"]["budget_usage"]["cost_units"]
            <= SHARED_BUDGET.max_cost_units
            and item["controller"]["budget_usage"]["turns"]
            <= SHARED_BUDGET.max_turns
            for item in traces
        ),
    }
    return {
        "b0_version": B0_VERSION,
        "development_only": True,
        "performance_claim_allowed": False,
        "run_count": len(traces),
        "episode_count": len(
            {item["evaluator"]["episode_id"] for item in traces}
        ),
        "group_count": len(B0_GROUPS),
        "group_metrics": metrics,
        "prompt_fairness_audit": prompts,
        "label_isolation_audit": labels,
        "acceptance": acceptance,
        "acceptance_passed": all(acceptance.values()),
        "independent_human_audit": {
            "sample_size": 24,
            "status": "pending_manual_annotation",
            "required_before_b1": True,
        },
        "b1_readiness": {
            "ready": False,
            "blocker": "complete the independent human audit and compare labels",
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(_canonical_json(item) for item in values) + "\n",
        encoding="utf-8",
    )


def write_b0_outputs(episodes: list[Episode], output_dir: Path) -> dict[str, Any]:
    traces = run_b0(episodes)
    report = build_b0_report(traces)
    sample, key = _human_audit_export(traces)
    component_matrix = {
        "b0_version": B0_VERSION,
        "development_only": True,
        "shared_public_event": True,
        "shared_tools": [asdict(item) for item in TOOL_SPECS],
        "shared_budget": asdict(SHARED_BUDGET),
        "groups": {
            group.value: dict(COMPONENT_MATRIX[group]) for group in B0_GROUPS
        },
    }
    _write_json(output_dir / "b0_component_matrix.json", component_matrix)
    _write_jsonl(output_dir / "b0_traces.jsonl", traces)
    _write_json(output_dir / "b0_summary.json", report)
    _write_json(
        output_dir / "b0_prompt_fairness_audit.json",
        report["prompt_fairness_audit"],
    )
    _write_json(
        output_dir / "b0_label_isolation_audit.json",
        report["label_isolation_audit"],
    )
    _write_jsonl(output_dir / "b0_human_audit_sample.jsonl", sample)
    _write_jsonl(output_dir / "b0_human_audit_key.jsonl", key)
    return report


__all__ = [
    "B0_GROUPS",
    "B0_VERSION",
    "COMPONENT_MATRIX",
    "SHARED_BUDGET",
    "TOOL_SPECS",
    "build_b0_report",
    "run_b0",
    "run_b0_controller",
    "write_b0_outputs",
]
