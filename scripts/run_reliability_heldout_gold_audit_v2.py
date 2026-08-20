"""One-time encrypted held-out Gold-Trace audit for frozen evaluator sources."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import secrets
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from eval.reliability_bench.gold_trace_spec_v2 import build_development_cases
from eval.reliability_bench.public_runtime_v2 import (
    bind_authoritative_tool_observation,
    parse_terminal_runtime_output,
)
from eval.reliability_bench.schema import episode_from_dict
from eval.reliability_bench.task_predicate_evaluator_v2 import (
    aggregate_selective_metrics,
    evaluate_task_decision,
)
from eval.reliability_bench.trace_evidence_evaluator_v2 import evaluate_trace_evidence


CANDIDATE_COMMIT = "40e1a619b0613453246aa8d9c689d237a942486d"
FROZEN_SOURCE_SHA256 = {
    "eval/reliability_bench/terminal_schema_v2.py": "2fe27dbad45726b35b5be73c6150e590421d756be89573f7c58fd83ce849e544",
    "eval/reliability_bench/public_runtime_v2.py": "ab572138d222fcb99037709cc1864b53209e788166431cacd4b298e4905420a8",
    "eval/reliability_bench/measurement_spec_v2.py": "038ee4075b5f32d9dd74dfe9e69006cb0054941ea881f6e812bbe8e7bf035f3d",
    "eval/reliability_bench/task_predicate_evaluator_v2.py": "7be012fa20ef161258c4ccd604e1ba1b6030662a3001c17add83ffb20e4c0c0e",
    "eval/reliability_bench/trace_evidence_evaluator_v2.py": "6e13316732b7e6d4aed706db1fac59e7ad1c43498a3632f4b9d5c6e38068063e",
    "eval/reliability_bench/gold_trace_spec_v2.py": "2331bc26eb3654d3b1ff18dc778898a7cf9ec04812219feff6cfb7cbf856289d",
}
AUDIT_AAD = b"ReliabilityBench-Q held-out Gold v2"
UTILITY_WEIGHTS = {
    "correct": 1.0,
    "wrong": 1.0,
    "unnecessary_abstain": 0.5,
    "cost": 0.01,
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_candidate(root: Path) -> dict[str, str]:
    actual = {path: sha256_file(root / path) for path in FROZEN_SOURCE_SHA256}
    if actual != FROZEN_SOURCE_SHA256:
        raise RuntimeError("frozen evaluator source hash mismatch")
    if subprocess.check_output(
        ["git", "merge-base", "--is-ancestor", CANDIDATE_COMMIT, "HEAD"], cwd=root
    ).strip() not in {b"", None}:
        raise RuntimeError("candidate commit is not an ancestor of HEAD")
    return actual


def _replace_identifiers(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in mapping.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_identifiers(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _replace_identifiers(item, mapping) for key, item in value.items()}
    return value


def build_heldout_cases(namespace: str) -> list[dict[str, Any]]:
    source = build_development_cases()
    heldout: list[dict[str, Any]] = []
    for index, original in enumerate(source, start=1):
        old_id = original["case_id"]
        new_id = f"heldout-{namespace}-{index:03d}"
        mapping = {
            old_id: new_id,
            "e-primary": f"ev-primary-{namespace}-{index:03d}",
            "e-context": f"ev-context-{namespace}-{index:03d}",
            "snapshot-pre": f"snapshot-pre-{namespace}-{index:03d}",
            "snapshot-current": f"snapshot-current-{namespace}-{index:03d}",
        }
        item = _replace_identifiers(copy.deepcopy(original), mapping)
        item["case_id"] = new_id
        item["tags"] = list(item["tags"]) + ["held_out", "fresh_namespace"]
        resource_suffix = f".heldout.{namespace}.{index:03d}"
        primary_dep = item["episode"]["evidence"][0]["depends_on_resources"][0]
        item["episode"]["evidence"][0]["depends_on_resources"][0] = primary_dep + resource_suffix
        if item["episode"]["drift_event"]["relevance"] == "related":
            item["episode"]["drift_event"]["affected_resources"] = [
                primary_dep + resource_suffix
            ]
        item["expected"]["runtime_accepted"] = True
        item["expected"]["runtime_identifier_failures"] = []
        _attach_task_gold(item)
        heldout.append(item)

    # Counterfactual pair changing only whether a declaration binds to a real call.
    base = next(
        item for item in heldout if "backend_selection" in item["tags"] and "safe_revalidated" in item["tags"]
    )
    for suffix, call_id, accepted in (
        ("legal-tool-call-id", "call:1", True),
        ("forged-tool-call-id", "call:999", False),
    ):
        item = copy.deepcopy(base)
        old_id = item["case_id"]
        new_id = f"heldout-{namespace}-{suffix}"
        item = _replace_identifiers(item, {old_id: new_id})
        item["case_id"] = new_id
        target = item["episode"]["evidence"][0]["evidence_id"]
        output = f"obs:{new_id}:1"
        item["terminal"]["revalidation_actions"] = [
            {
                "tool_call_id": call_id,
                "target_evidence_ids": [target],
                "output_evidence_ids": [output],
            }
        ]
        item["tags"] = ["metamorphic", "tool_call_id_only", suffix, "held_out"]
        item["expected"]["runtime_accepted"] = accepted
        item["expected"]["runtime_identifier_failures"] = (
            [] if accepted else ["unbound_tool_call_id:call:999"]
        )
        _attach_task_gold(item)
        heldout.append(item)
    return heldout


def _attach_task_gold(item: dict[str, Any]) -> None:
    status = item["terminal"]["status"]
    answered = status != "abstain"
    task_type = item["episode"]["task_type"]
    predicate = item["episode"]["ground_truth"]["acceptable_payload"]["task_predicate"]
    feasible = not (
        task_type == "unreachable_target"
        and bool(predicate["expected_unreachable"])
    )
    correct = bool(item["expected"]["task_terminal_correct"])
    item["expected"].update(
        {
            "predicate_satisfied": None if not answered else correct,
            "wrong_decision": None if not answered else not correct,
            "answered": answered,
            "feasible": feasible,
            "feasible_completion": feasible and correct,
            "correct_rejection": (not feasible) and correct,
            "unnecessary_abstain": feasible and not answered,
        }
    )


def _aggregate_gold(cases: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(cases)
    expected = [item["expected"] for item in cases]
    answered = [item for item in expected if item["answered"]]
    feasible = [item for item in expected if item["feasible"]]
    infeasible = [item for item in expected if not item["feasible"]]
    total_utility = 0.0
    for item in expected:
        total_utility += (
            UTILITY_WEIGHTS["correct"] * int(item["task_terminal_correct"])
            - UTILITY_WEIGHTS["wrong"] * int(item["wrong_decision"] is True)
            - UTILITY_WEIGHTS["unnecessary_abstain"] * int(item["unnecessary_abstain"])
            - UTILITY_WEIGHTS["cost"] * float(item["actual_cost_units"])
        )
    return {
        "episode_count": count,
        "answered_count": len(answered),
        "coverage": len(answered) / count,
        "selective_risk": (
            sum(item["wrong_decision"] is True for item in answered) / len(answered)
            if answered else None
        ),
        "feasible_completion_rate": (
            sum(item["feasible_completion"] for item in feasible) / len(feasible)
            if feasible else None
        ),
        "correct_rejection_rate": (
            sum(item["correct_rejection"] for item in infeasible) / len(infeasible)
            if infeasible else None
        ),
        "unnecessary_abstain_rate": (
            sum(item["unnecessary_abstain"] for item in feasible) / len(feasible)
            if feasible else None
        ),
        "overall_utility": total_utility / count,
    }


def generate(root: Path, package_dir: Path, key_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=False)
    key_dir.mkdir(parents=True, exist_ok=False)
    source_hashes = verify_candidate(root)
    namespace = secrets.token_hex(12)
    cases = build_heldout_cases(namespace)
    public_cases = [
        {key: value for key, value in case.items() if key != "expected"}
        for case in cases
    ]
    labels = {
        "audit_version": "reliabilitybench-q/heldout-gold-2.0",
        "labels": {case["case_id"]: case["expected"] for case in cases},
        "aggregate_expected": _aggregate_gold(cases),
        "utility_weights": UTILITY_WEIGHTS,
    }
    public_path = package_dir / "heldout_public_traces.json"
    public_path.write_bytes(canonical_bytes(public_cases))
    key = AESGCM.generate_key(bit_length=256)
    nonce = secrets.token_bytes(12)
    plaintext = canonical_bytes(labels)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, AUDIT_AAD)
    encrypted_path = package_dir / "heldout_gold_labels.aesgcm"
    encrypted_path.write_bytes(nonce + ciphertext)
    key_path = key_dir / "heldout_gold_key.bin"
    key_path.write_bytes(key)
    manifest = {
        "status": "generated_gold_encrypted_predictions_not_run",
        "generated_at": utc_now(),
        "candidate_commit": CANDIDATE_COMMIT,
        "frozen_source_sha256": source_hashes,
        "fresh_namespace_sha256": sha256_bytes(namespace.encode()),
        "case_count": len(cases),
        "public_trace_sha256": sha256_file(public_path),
        "encrypted_gold_sha256": sha256_file(encrypted_path),
        "gold_plaintext_sha256": sha256_bytes(plaintext),
        "aes_gcm": {"key_bits": 256, "nonce_bytes": 12, "tag_bytes": 16},
        "separation": {
            "development_gold": "fresh cryptographic ID/resource/snapshot namespace",
            "judge_v2_audit": "independent declarative traces; no imported audit data",
            "method_validation_v2_1": "sealed archive never opened; collision avoided by fresh namespace",
            "future_action_guard_data": "no ActionGuard behavior or data imported",
        },
        "unseal_condition": "predictions file and prediction manifest must exist and be hashed first",
    }
    (package_dir / "heldout_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _canonical(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    return value


def predict(root: Path, package_dir: Path) -> None:
    source_hashes = verify_candidate(root)
    manifest = json.loads((package_dir / "heldout_manifest.json").read_text(encoding="utf-8"))
    public_path = package_dir / "heldout_public_traces.json"
    if sha256_file(public_path) != manifest["public_trace_sha256"]:
        raise RuntimeError("held-out public trace hash mismatch")
    if (package_dir / "heldout_predictions.json").exists():
        raise RuntimeError("predictions already exist; one-time held-out run refused")
    cases = json.loads(public_path.read_text(encoding="utf-8"))
    predictions = {}
    task_evaluations = []
    trace_costs = []
    for case in cases:
        case_id = case["case_id"]
        bound_calls = [
            bind_authoritative_tool_observation(
                trace_id=case_id, call_index=raw["call_index"], accepted_call=raw
            )
            for raw in case["accepted_tool_calls"]
        ]
        known = [item["evidence_id"] for item in case["episode"]["evidence"]]
        runtime = parse_terminal_runtime_output(
            json.dumps(case["terminal"]),
            known_evidence_ids=known,
            accepted_tool_calls=bound_calls,
        )
        if runtime.decision is None:
            raise RuntimeError(f"held-out terminal unexpectedly unparseable: {case_id}")
        episode = episode_from_dict(case["episode"])
        task = evaluate_task_decision(episode, runtime.decision)
        trace_evaluation = evaluate_trace_evidence(
                episode=episode,
                decision=runtime.decision,
                trace_id=case_id,
                accepted_tool_calls=case["accepted_tool_calls"],
            )
        trace = asdict(trace_evaluation)
        task_evaluations.append(task)
        trace_costs.append(float(trace_evaluation.actual_cost_units))
        predictions[case_id] = {
            "task_terminal_correct": task.terminal_correct,
            "predicate_satisfied": task.predicate_satisfied,
            "wrong_decision": task.wrong_decision,
            "answered": task.answered,
            "feasible": task.feasible,
            "feasible_completion": task.feasible_completion,
            "correct_rejection": task.correct_rejection,
            "unnecessary_abstain": task.unnecessary_abstain,
            "evidence_state": trace["evidence_state"],
            "definite_stale_dependence": trace["definite_stale_dependence"],
            "unsupported_decision": trace["unsupported_decision"],
            "provenance_incomplete": trace["provenance_incomplete"],
            "necessary_revalidation_slots": _canonical(trace["necessary_revalidation_slots"]),
            "valid_revalidation_slots": _canonical(trace["valid_revalidation_slots"]),
            "necessary_revalidation_recall": trace["necessary_revalidation_recall"],
            "snapshot_lineage_valid": trace["snapshot_lineage_valid"],
            "evidence_closure_complete": trace["evidence_closure_complete"],
            "intervention_requested": trace["intervention_requested"],
            "intervention_attempted": trace["intervention_attempted"],
            "intervention_executed": trace["intervention_executed"],
            "intervention_verified": trace["intervention_verified"],
            "actual_tool_call_count": trace["actual_tool_call_count"],
            "failed_tool_call_count": trace["failed_tool_call_count"],
            "repeated_tool_call_count": trace["repeated_tool_call_count"],
            "irrelevant_tool_call_count": trace["irrelevant_tool_call_count"],
            "actual_cost_units": trace["actual_cost_units"],
            "actual_latency_seconds": trace["actual_latency_seconds"],
            "runtime_accepted": runtime.accepted,
            "runtime_identifier_failures": list(runtime.identifier_failures),
        }
    payload = {
        "candidate_commit": CANDIDATE_COMMIT,
        "frozen_source_sha256": source_hashes,
        "predicted_at": utc_now(),
        "predictions": predictions,
        "aggregate_metrics": aggregate_selective_metrics(
            task_evaluations,
            costs=trace_costs,
            utility_weights=UTILITY_WEIGHTS,
        ),
    }
    prediction_path = package_dir / "heldout_predictions.json"
    prediction_path.write_bytes(canonical_bytes(payload))
    prediction_manifest = {
        "status": "predictions_frozen_gold_still_encrypted",
        "frozen_at": utc_now(),
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_count": len(predictions),
        "encrypted_gold_sha256": manifest["encrypted_gold_sha256"],
    }
    (package_dir / "prediction_freeze_manifest.json").write_text(
        json.dumps(prediction_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def adjudicate(package_dir: Path, key_dir: Path) -> None:
    prediction_path = package_dir / "heldout_predictions.json"
    freeze_path = package_dir / "prediction_freeze_manifest.json"
    if not prediction_path.exists() or not freeze_path.exists():
        raise RuntimeError("predictions must be frozen before Gold unseal")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if sha256_file(prediction_path) != freeze["prediction_sha256"]:
        raise RuntimeError("frozen prediction hash mismatch")
    manifest = json.loads((package_dir / "heldout_manifest.json").read_text(encoding="utf-8"))
    encrypted = (package_dir / "heldout_gold_labels.aesgcm").read_bytes()
    if sha256_bytes(encrypted) != manifest["encrypted_gold_sha256"]:
        raise RuntimeError("encrypted Gold hash mismatch")
    key = (key_dir / "heldout_gold_key.bin").read_bytes()
    plaintext = AESGCM(key).decrypt(encrypted[:12], encrypted[12:], AUDIT_AAD)
    if sha256_bytes(plaintext) != manifest["gold_plaintext_sha256"]:
        raise RuntimeError("Gold plaintext hash mismatch")
    gold_document = json.loads(plaintext)
    gold = gold_document["labels"]
    prediction_document = json.loads(prediction_path.read_text(encoding="utf-8"))
    predictions = prediction_document["predictions"]
    mismatches = []
    field_totals: dict[str, dict[str, int]] = {}
    for case_id, expected in gold.items():
        actual = predictions[case_id]
        for field, expected_value in expected.items():
            counts = field_totals.setdefault(field, {"passed": 0, "failed": 0})
            if actual[field] == expected_value:
                counts["passed"] += 1
            else:
                counts["failed"] += 1
                mismatches.append(
                    {
                        "case_id": case_id,
                        "field": field,
                        "expected": expected_value,
                        "actual": actual[field],
                        "classification": "requires_manual_bug_gold_or_identifiability_review",
                    }
                )
    aggregate_mismatches = []
    for field, expected_value in gold_document["aggregate_expected"].items():
        actual_value = prediction_document["aggregate_metrics"][field]
        if actual_value != expected_value:
            aggregate_mismatches.append(
                {"field": field, "expected": expected_value, "actual": actual_value}
            )
            mismatches.append(
                {
                    "case_id": "__aggregate__",
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value,
                    "classification": "requires_manual_bug_gold_or_identifiability_review",
                }
            )
    result = {
        "audit_version": "reliabilitybench-q/heldout-gold-gate-2.0",
        "adjudicated_at": utc_now(),
        "prediction_sha256_before_unseal": freeze["prediction_sha256"],
        "case_count": len(gold),
        "field_gate_results": field_totals,
        "aggregate_gate_result": {
            "expected": gold_document["aggregate_expected"],
            "actual": prediction_document["aggregate_metrics"],
            "mismatches": aggregate_mismatches,
            "passed": not aggregate_mismatches,
        },
        "mismatch_count": len(mismatches),
        "all_unique_truth_gates_passed": not mismatches,
        "mismatches": mismatches,
    }
    (package_dir / "heldout_gate_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "case_count": result["case_count"],
        "mismatch_count": result["mismatch_count"],
        "all_unique_truth_gates_passed": result["all_unique_truth_gates_passed"],
    }, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("generate", "predict", "adjudicate"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "generate":
        generate(args.root, args.package_dir, args.key_dir)
    elif args.mode == "predict":
        predict(args.root, args.package_dir)
    else:
        adjudicate(args.package_dir, args.key_dir)


if __name__ == "__main__":
    main()
