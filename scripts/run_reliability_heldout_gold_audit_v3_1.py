#!/usr/bin/env python3
"""One-time encrypted held-out audit for the Measurement v3.1 delta."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.reliability_bench.gold_runtime_terminal_spec_v3_1 import (  # noqa: E402
    UTILITY_WEIGHTS,
    build_heldout_delta_seed_cases,
)
from scripts.run_reliability_gold_trace_audit_v3_1 import (  # noqa: E402
    FIELDS,
    predict_case,
)


AUDIT_AAD = b"ReliabilityBench-Q Measurement v3.1 held-out Gold"
CANDIDATE_MANIFEST = (
    ROOT / "eval/reliability_bench/data/measurement_v3_1_candidate_manifest.json"
)


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


def verify_candidate() -> dict[str, Any]:
    manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    actual = {
        path: sha256_file(ROOT / path)
        for path in manifest["frozen_source_sha256"]
    }
    if actual != manifest["frozen_source_sha256"]:
        raise RuntimeError("Measurement v3.1 candidate source hash mismatch")
    candidate = manifest["candidate_commit"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", candidate, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    return manifest


def _replace(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, old, new) for key, item in value.items()}
    return value


def build_fresh_cases(namespace: str) -> list[dict[str, Any]]:
    return [
        _replace(copy.deepcopy(item), "heldseed", namespace)
        for item in build_heldout_delta_seed_cases()
    ]


def _aggregate_gold(cases: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [item["expected"] for item in cases]
    count = len(expected)
    answered = [item for item in expected if item["answered"]]
    failures = [item for item in expected if item["execution_failure"]]
    feasible = [item for item in expected if item["feasible"]]
    infeasible = [item for item in expected if not item["feasible"]]
    utility = sum(
        UTILITY_WEIGHTS["correct"] * int(item["task_terminal_correct"])
        - UTILITY_WEIGHTS["wrong"] * int(item["wrong_decision"] is True)
        - UTILITY_WEIGHTS["unnecessary_abstain"] * int(item["unnecessary_abstain"])
        - UTILITY_WEIGHTS["execution_failure"] * int(item["execution_failure"])
        - UTILITY_WEIGHTS["cost"] * float(item["actual_cost_units"])
        for item in expected
    )
    return {
        "episode_count": count,
        "answered_count": len(answered),
        "coverage": len(answered) / count,
        "execution_failure_rate": len(failures) / count,
        "selective_risk": (
            sum(item["wrong_decision"] is True for item in answered) / len(answered)
            if answered
            else None
        ),
        "feasible_completion_rate": (
            sum(item["feasible_completion"] for item in feasible) / len(feasible)
            if feasible
            else None
        ),
        "correct_rejection_rate": (
            sum(item["correct_rejection"] for item in infeasible) / len(infeasible)
            if infeasible
            else None
        ),
        "unnecessary_abstain_rate": (
            sum(item["unnecessary_abstain"] for item in feasible) / len(feasible)
            if feasible
            else None
        ),
        "overall_utility": utility / count,
    }


def generate(package: Path, key_dir: Path) -> None:
    candidate = verify_candidate()
    package.mkdir(parents=True, exist_ok=False)
    key_dir.mkdir(parents=True, exist_ok=False)
    namespace = secrets.token_hex(12)
    cases = build_fresh_cases(namespace)
    public = [{key: value for key, value in item.items() if key != "expected"} for item in cases]
    labels = {
        "labels": {item["case_id"]: item["expected"] for item in cases},
        "aggregate_expected": _aggregate_gold(cases),
        "utility_weights": UTILITY_WEIGHTS,
    }
    public_path = package / "heldout_public_traces.json"
    public_path.write_bytes(canonical_bytes(public))
    plaintext = canonical_bytes(labels)
    key = AESGCM.generate_key(bit_length=256)
    nonce = secrets.token_bytes(12)
    encrypted = nonce + AESGCM(key).encrypt(nonce, plaintext, AUDIT_AAD)
    encrypted_path = package / "heldout_gold_labels.aesgcm"
    encrypted_path.write_bytes(encrypted)
    (key_dir / "heldout_gold_key.bin").write_bytes(key)
    manifest = {
        "status": "gold_frozen_encrypted_predictions_not_run",
        "candidate_commit": candidate["candidate_commit"],
        "frozen_source_sha256": candidate["frozen_source_sha256"],
        "namespace_sha256": sha256_bytes(namespace.encode()),
        "case_count": len(cases),
        "field_count_per_case": len(FIELDS),
        "public_trace_sha256": sha256_file(public_path),
        "encrypted_gold_sha256": sha256_file(encrypted_path),
        "gold_plaintext_sha256": sha256_bytes(plaintext),
        "aes_gcm": {"key_bits": 256, "nonce_bytes": 12, "tag_bytes": 16},
        "unseal_condition": "predictions and prediction hash must exist first",
        "method_validation_v2_1": "sealed and not accessed; custodian declaration",
    }
    (package / "heldout_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def predict(package: Path) -> None:
    candidate = verify_candidate()
    manifest = json.loads((package / "heldout_manifest.json").read_text(encoding="utf-8"))
    public_path = package / "heldout_public_traces.json"
    if sha256_file(public_path) != manifest["public_trace_sha256"]:
        raise RuntimeError("held-out public trace hash mismatch")
    prediction_path = package / "heldout_predictions.json"
    if prediction_path.exists():
        raise RuntimeError("one-time held-out prediction already exists")
    cases = json.loads(public_path.read_text(encoding="utf-8"))
    predictions = {}
    task_evaluations = []
    costs = []
    for case in cases:
        prediction, task, trace = predict_case(case)
        predictions[case["case_id"]] = prediction
        task_evaluations.append(task)
        costs.append(float(trace.actual_cost_units))
    from eval.reliability_bench.task_predicate_evaluator_v3_1 import aggregate_selective_metrics

    payload = {
        "candidate_commit": candidate["candidate_commit"],
        "predictions": predictions,
        "aggregate_metrics": aggregate_selective_metrics(
            task_evaluations, costs=costs, utility_weights=UTILITY_WEIGHTS
        ),
    }
    prediction_path.write_bytes(canonical_bytes(payload))
    prediction_manifest = {
        "prediction_sha256_before_gold_unseal": sha256_file(prediction_path),
        "prediction_count": len(predictions),
        "gold_unsealed": False,
    }
    (package / "prediction_manifest.json").write_text(
        json.dumps(prediction_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(prediction_manifest, ensure_ascii=False, indent=2))


def unseal(package: Path, key_dir: Path) -> int:
    verify_candidate()
    manifest = json.loads((package / "heldout_manifest.json").read_text(encoding="utf-8"))
    prediction_manifest = json.loads(
        (package / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    prediction_path = package / "heldout_predictions.json"
    if sha256_file(prediction_path) != prediction_manifest["prediction_sha256_before_gold_unseal"]:
        raise RuntimeError("prediction hash changed before Gold unseal")
    encrypted = (package / "heldout_gold_labels.aesgcm").read_bytes()
    key = (key_dir / "heldout_gold_key.bin").read_bytes()
    plaintext = AESGCM(key).decrypt(encrypted[:12], encrypted[12:], AUDIT_AAD)
    if sha256_bytes(plaintext) != manifest["gold_plaintext_sha256"]:
        raise RuntimeError("decrypted Gold hash mismatch")
    labels = json.loads(plaintext)
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    mismatches = []
    for case_id, expected in labels["labels"].items():
        actual = predictions["predictions"][case_id]
        for field in FIELDS:
            if actual[field] != expected[field]:
                mismatches.append(
                    {"case_id": case_id, "field": field, "expected": expected[field], "actual": actual[field]}
                )
    aggregate_mismatches = []
    for field, expected in labels["aggregate_expected"].items():
        actual = predictions["aggregate_metrics"][field]
        if actual != expected:
            aggregate_mismatches.append({"field": field, "expected": expected, "actual": actual})
    field_comparisons = len(labels["labels"]) * len(FIELDS)
    aggregate_comparisons = len(labels["aggregate_expected"])
    report = {
        "status": "passed" if not mismatches and not aggregate_mismatches else "failed",
        "case_count": len(labels["labels"]),
        "field_comparisons": field_comparisons,
        "aggregate_comparisons": aggregate_comparisons,
        "total_comparisons": field_comparisons + aggregate_comparisons,
        "exact_agreement": field_comparisons + aggregate_comparisons - len(mismatches) - len(aggregate_mismatches),
        "mismatches": mismatches,
        "aggregate_mismatches": aggregate_mismatches,
        "prediction_sha256_before_gold_unseal": prediction_manifest["prediction_sha256_before_gold_unseal"],
        "gold_plaintext_sha256": manifest["gold_plaintext_sha256"],
    }
    (package / "heldout_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("generate", "predict", "unseal"))
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    if args.mode in {"generate", "unseal"} and args.key_dir is None:
        parser.error("--key-dir is required for generate and unseal")
    if args.mode == "generate":
        generate(package, args.key_dir.resolve())
        return 0
    if args.mode == "predict":
        predict(package)
        return 0
    return unseal(package, args.key_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
