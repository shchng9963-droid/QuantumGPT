"""Development-only ActionGuard audit over legacy traces and fresh smoke cases."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.reliability_bench.action_guard_contract_v1 import (
    AcceptedToolCall,
    CandidateFinalAction,
    EvidenceValidity,
    GuardOutcome,
    LedgerEvidence,
    PublicTaskContract,
    ReasonCode,
    VisibleEvidenceLedger,
)
from eval.reliability_bench.action_guard_v1 import ActionGuard
from eval.reliability_bench.generator_v2 import (
    GeneratorV2Config,
    generate_method_validation_candidates,
)
from eval.reliability_bench.schema import Episode, TaskType, episode_from_dict
from eval.reliability_bench.terminal_schema_v2 import (
    TERMINAL_SCHEMA_VERSION,
    parse_terminal_decision,
)


AUDIT_VERSION = "reliabilitybench-q/action-guard-development-audit-1.0"

EVIDENCE_SLOTS = {
    "backend_ranking": ("backend_ranking",),
    "qubit_mapping": ("qubit_properties",),
    "transpiled_circuit": ("compilation",),
    "circuit_result": ("circuit_result",),
    "mitigation_estimate": ("mitigation_estimate",),
    "feasibility_assessment": ("backend_capacity",),
    "task_context": ("task_context",),
}
TOOL_SLOTS = {
    "get_backend_health": ("backend_health", "backend_capacity"),
    "compare_backends": ("backend_ranking",),
    "get_qubit_properties": ("qubit_properties",),
    "get_coupling_map": ("coupling_map",),
    "transpile_circuit": ("compilation",),
    "run_circuit": ("circuit_result",),
    "apply_mitigation": ("mitigation_estimate",),
    "refresh_task_context": ("task_context",),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _public_task(episode: Episode, *, observed_snapshots=()) -> PublicTaskContract:
    constraints = episode.task_constraints
    snapshots = tuple(
        dict.fromkeys(
            [
                str(episode.initial_state["snapshot_id"]),
                *[str(item) for item in observed_snapshots if item],
                "observed-after-event",
            ]
        )
    )
    return PublicTaskContract(
        task_id=episode.episode_id,
        task_type=episode.task_type.value,
        backend_ids=tuple(constraints.get("candidate_backends") or [episode.backend]),
        qubit_ids=tuple(constraints.get("candidate_qubits") or episode.initial_state.get("available_qubits") or ()),
        circuit_ids=(episode.circuit,),
        snapshot_ids=snapshots,
        mitigation_policy_ids=("keep_mitigation", "increase_mitigation"),
    )


def _terminal_from_legacy(trace: dict[str, Any]) -> CandidateFinalAction:
    task_type = trace["episode"]["task_type"]
    model = trace["model_final_decision"]
    action = model["action"]
    if task_type == "backend_selection":
        payload = {"backend_id": model["selected_backend"]}
    elif task_type == "qubit_mapping":
        payload = {"qubit_ids": list(model["selected_qubits"] or [])}
    elif task_type == "transpilation":
        payload = {"compilation_snapshot_id": model["compilation_snapshot_id"]}
    elif task_type == "fidelity_claim":
        payload = {"claimed_success": model["claimed_success"]}
    elif task_type == "mitigation_decision":
        payload = {
            "mitigation_policy": model["mitigation_action"],
            "intervention_requested": action == "reassess_mitigation",
        }
    else:
        payload = {"declared_unreachable": model["declared_unreachable"]}
    status = (
        "failure"
        if task_type == "unreachable_target" and model["declared_unreachable"] is True
        else "success"
    )
    support = tuple(
        dict.fromkeys(
            [
                *(model.get("supporting_artifact_ids") or []),
                *(model.get("used_artifact_ids") or []),
            ]
        )
    )
    raw = {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "status": status,
        "decision": {
            "task_type": task_type,
            "task_action": action,
            "payload": payload,
        },
        "supporting_evidence_ids": list(support),
        # Legacy tool-name self-reports are not upgraded to call IDs.
        "revalidation_actions": [],
        "failure": {
            "code": "declared_unreachable" if status == "failure" else None,
            "reason": "legacy development decision" if status == "failure" else None,
        },
    }
    return CandidateFinalAction(parse_terminal_decision(raw))


def _legacy_ledger_and_calls(trace: dict[str, Any]) -> tuple[VisibleEvidenceLedger, tuple[AcceptedToolCall, ...]]:
    mapping = {
        "valid_after_event": EvidenceValidity.VALID,
        "invalid_after_event": EvidenceValidity.INVALID,
        "unchecked_after_event": EvidenceValidity.UNKNOWN,
        "valid_pre_event": EvidenceValidity.UNKNOWN,
    }
    records = []
    for item in trace["controller"].get("private_state_final", {}).get("evidence_ledger", []):
        records.append(
            LedgerEvidence(
                evidence_id=item["evidence_id"],
                slots=EVIDENCE_SLOTS.get(item["evidence_type"], ()),
                validity=mapping.get(item.get("validity"), EvidenceValidity.UNKNOWN),
                source="legacy-visible-ledger",
            )
        )
    calls = []
    for raw in trace.get("tool_calls", []):
        call_id = f"call:{raw['call_index']}"
        output_id = f"obs:{trace['trace_id']}:{raw['call_index']}"
        response = raw.get("response") or {}
        successful = not (
            response.get("error") is not None
            or response.get("success") is False
            or response.get("status") in {"error", "failed", "timeout"}
        )
        calls.append(
            AcceptedToolCall(
                call_id,
                raw["tool_name"],
                dict(raw.get("request") or {}),
                successful,
                (output_id,),
            )
        )
        records.append(
            LedgerEvidence(
                output_id,
                TOOL_SLOTS.get(raw["tool_name"], ()),
                EvidenceValidity.VALID if successful else EvidenceValidity.UNKNOWN,
                "legacy-accepted-tool-call",
                produced_by_tool_call_id=call_id,
            )
        )
    return VisibleEvidenceLedger(tuple(records)), tuple(calls)


def audit_legacy_24(
    *, trace_path: Path, episode_path: Path
) -> dict[str, Any]:
    episodes = {
        item.episode_id: item
        for item in (
            episode_from_dict(json.loads(line))
            for line in episode_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    selected_groups = {"ledger_only", "full_selective"}
    counters: dict[str, Counter] = defaultdict(Counter)
    rows = []
    guard = ActionGuard()
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        trace = json.loads(line)
        group = trace["schedule"]["controller_group"]
        if group not in selected_groups:
            continue
        counters[group]["trace_count"] += 1
        if trace.get("unscorable") or not trace.get("model_final_decision"):
            counters[group]["legacy_unscorable"] += 1
            continue
        episode = episodes[trace["episode"]["episode_id"]]
        observed_snapshots = [
            (item.get("response") or {}).get("snapshot")
            for item in trace.get("tool_calls", [])
        ]
        try:
            candidate = _terminal_from_legacy(trace)
        except (KeyError, TypeError, ValueError) as exc:
            counters[group]["legacy_schema_unparseable"] += 1
            rows.append(
                {
                    "trace_id": trace["trace_id"],
                    "group": group,
                    "outcome": "legacy_schema_unparseable",
                    "error_type": type(exc).__name__,
                }
            )
            continue
        ledger, calls = _legacy_ledger_and_calls(trace)
        task = _public_task(episode, observed_snapshots=observed_snapshots)
        started = time.perf_counter()
        result = guard.check_final_action(
            candidate, task=task, ledger=ledger, accepted_tool_calls=calls
        )
        latency = time.perf_counter() - started
        counters[group]["parseable"] += 1
        counters[group][result.outcome.value] += 1
        counters[group][f"reason:{result.reason_code.value if result.reason_code else 'none'}"] += 1
        counters[group]["answered_before_guard"] += int(candidate.terminal.answered)
        counters[group]["allowed_without_repair"] += int(result.outcome is GuardOutcome.ALLOW)
        rows.append(
            {
                "trace_id": trace["trace_id"],
                "episode_id": episode.episode_id,
                "group": group,
                "outcome": result.outcome.value,
                "reason_code": result.reason_code.value if result.reason_code else None,
                "candidate_sha256": result.candidate_sha256,
                "guard_latency_seconds": latency,
            }
        )
    summaries = {}
    for group, counter in counters.items():
        parseable = counter["parseable"]
        summaries[group] = {
            **dict(counter),
            "legacy_structural_coverage": (
                counter["answered_before_guard"] / parseable if parseable else None
            ),
            "allow_rate_without_repair": (
                counter["allowed_without_repair"] / parseable if parseable else None
            ),
            "abstention_rate": 0.0 if parseable else None,
            "selective_risk": None,
            "selective_risk_note": "not estimable: legacy traces predate frozen Measurement v2 and judge v1 is invalidated",
        }
    return {
        "status": "development-only / offline counterfactual guard replay",
        "episode_count": len({row.get("episode_id") for row in rows if row.get("episode_id")}),
        "trace_count": len(rows),
        "groups": summaries,
        "rows": rows,
    }


def _smoke_terminal(
    episode: Episode,
    *,
    support: tuple[str, ...],
    declarations=(),
) -> CandidateFinalAction:
    task = episode.task_type
    if task is TaskType.BACKEND_SELECTION:
        action, payload, status = "select_backend", {"backend_id": episode.task_constraints["candidate_backends"][0]}, "success"
    elif task is TaskType.QUBIT_MAPPING:
        action, payload, status = "select_qubits", {"qubit_ids": episode.task_constraints["candidate_qubits"][:2]}, "success"
    elif task is TaskType.TRANSPILATION:
        action, payload, status = "retranspile", {"compilation_snapshot_id": episode.ground_truth.acceptable_payload["evidence_policy"]["current_snapshot_id"]}, "success"
    elif task is TaskType.FIDELITY_CLAIM:
        action, payload, status = "rerun_circuit", {"claimed_success": True}, "success"
    elif task is TaskType.MITIGATION_DECISION:
        action, payload, status = "reassess_mitigation", {"mitigation_policy": "increase_mitigation", "intervention_requested": True}, "success"
    else:
        action, payload, status = "declare_unreachable", {"declared_unreachable": True}, "failure"
    return CandidateFinalAction(
        parse_terminal_decision(
            {
                "schema_version": TERMINAL_SCHEMA_VERSION,
                "status": status,
                "decision": {"task_type": task.value, "task_action": action, "payload": payload},
                "supporting_evidence_ids": list(support),
                "revalidation_actions": list(declarations),
                "failure": {
                    "code": "declared_unreachable" if status == "failure" else None,
                    "reason": "development smoke" if status == "failure" else None,
                },
            }
        )
    )


def audit_fresh_smoke() -> dict[str, Any]:
    config = GeneratorV2Config(
        random_seed=20260901,
        pairs_per_task=2,
        source="action_guard_development_smoke_v1",
    )
    episodes = generate_method_validation_candidates(config)
    guard = ActionGuard()
    rows = []
    expected_pass = 0
    for episode in episodes:
        task = _public_task(
            episode,
            observed_snapshots=(
                episode.ground_truth.acceptable_payload["evidence_policy"]["current_snapshot_id"],
            ),
        )
        primary = episode.evidence[0]
        slot = task.required_slots[0]
        initial_validity = (
            EvidenceValidity.INVALID
            if primary.evidence_id in episode.ground_truth.invalidated_artifact_ids
            else EvidenceValidity.VALID
        )
        base_records = (
            LedgerEvidence(primary.evidence_id, (slot,), initial_validity, "smoke-temporal-layer"),
            LedgerEvidence(episode.evidence[1].evidence_id, ("task_context",), EvidenceValidity.VALID, "smoke-context"),
        )
        current_id = f"obs:smoke:{episode.episode_id}:1"
        current = LedgerEvidence(
            current_id, (slot,), EvidenceValidity.VALID, "smoke-tool", "call:1"
        )
        real_call = AcceptedToolCall("call:1", "abstract_recovery", {}, True, (current_id,))
        cases = [
            ("valid_current", _smoke_terminal(episode, support=(current_id,)), VisibleEvidenceLedger((*base_records, current)), (real_call,), GuardOutcome.ALLOW, None),
            ("initial_state_counterfactual", _smoke_terminal(episode, support=(primary.evidence_id,)), VisibleEvidenceLedger(base_records), (), GuardOutcome.ALLOW if initial_validity is EvidenceValidity.VALID else GuardOutcome.BLOCK, None if initial_validity is EvidenceValidity.VALID else ReasonCode.STALE_CITED_EVIDENCE),
            ("missing_support", _smoke_terminal(episode, support=()), VisibleEvidenceLedger(base_records), (), GuardOutcome.BLOCK, ReasonCode.PREMATURE_FINALIZATION),
            ("wrong_slot_only", _smoke_terminal(episode, support=(episode.evidence[1].evidence_id,)), VisibleEvidenceLedger(base_records), (), GuardOutcome.BLOCK, ReasonCode.MISSING_REQUIRED_SUPPORT),
            ("forged_call_id", _smoke_terminal(episode, support=(current_id,), declarations=({"tool_call_id": "call:999", "target_evidence_ids": [primary.evidence_id], "output_evidence_ids": [current_id]},)), VisibleEvidenceLedger((*base_records, current)), (real_call,), GuardOutcome.REPAIR_REQUEST, ReasonCode.INVALID_REFERENCE),
        ]
        for name, candidate, ledger, calls, expected_outcome, expected_reason in cases:
            before = candidate
            result = guard.check_final_action(
                candidate, task=task, ledger=ledger, accepted_tool_calls=calls
            )
            passed = (
                result.outcome is expected_outcome
                and result.reason_code is expected_reason
                and candidate == before
            )
            expected_pass += int(passed)
            rows.append(
                {
                    "case_id": f"{episode.episode_id}:{name}",
                    "task_type": episode.task_type.value,
                    "relevance": episode.drift_event.relevance.value,
                    "outcome": result.outcome.value,
                    "reason_code": result.reason_code.value if result.reason_code else None,
                    "passed": passed,
                }
            )
    return {
        "status": "development-only synthetic smoke",
        "config": asdict(config),
        "episode_count": len(episodes),
        "case_count": len(rows),
        "passed_count": expected_pass,
        "failed_count": len(rows) - expected_pass,
        "invariant_conformance": expected_pass / len(rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-traces", type=Path, required=True)
    parser.add_argument("--legacy-episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "audit_version": AUDIT_VERSION,
        "generated_at": utc_now(),
        "paper_performance_claims_allowed": False,
        "legacy_24": audit_legacy_24(
            trace_path=args.legacy_traces, episode_path=args.legacy_episodes
        ),
        "fresh_smoke": audit_fresh_smoke(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "legacy_episode_count": report["legacy_24"]["episode_count"],
        "legacy_trace_count": report["legacy_24"]["trace_count"],
        "fresh_smoke_case_count": report["fresh_smoke"]["case_count"],
        "fresh_smoke_passed": report["fresh_smoke"]["passed_count"],
        "fresh_smoke_failed": report["fresh_smoke"]["failed_count"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["fresh_smoke"]["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
