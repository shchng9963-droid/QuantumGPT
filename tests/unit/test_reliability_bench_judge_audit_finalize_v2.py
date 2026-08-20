from __future__ import annotations

import base64
import hashlib
import json

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from eval.reliability_bench.judge_audit_finalize_v2 import (
    build_human_consensus,
    decrypt_prediction_rows,
    score_judge_predictions,
    validate_stage2_return,
)
from eval.reliability_bench.judge_audit_package_v2 import (
    PREDICTION_AAD,
    PREDICTION_MAGIC,
)


def _config() -> dict:
    return {
        "human_review": {
            "confidence_interval": {"replicates": 200, "seed": 2026082003}
        },
        "thresholds": {
            "outcome": {
                "minimum_agreements_out_of_24": 22,
                "agreement_min": 0.9,
                "cohen_kappa_min": 0.8,
            },
            "critical_labels": {
                "labels": [
                    "stale_evidence_reuse",
                    "unsupported_success_claim",
                    "missing_required_revalidation",
                ],
                "f1_min": 0.9,
            },
            "diagnostics": {
                "primary_failure_stage_cohen_kappa_min": 0.6,
                "primary_failure_stage_macro_f1_min": 0.7,
                "secondary_multilabel_jaccard_min": 0.7,
                "secondary_macro_f1_min": 0.7,
            },
        },
        "class_balance": {
            "minimum_human_correct": 3,
            "minimum_human_incorrect": 3,
        },
    }


def _annotation(outcome: str, *, stage: str | None = None, tags=None) -> dict:
    return {
        "annotator_id": "C",
        "trajectory_outcome": outcome,
        "primary_failure_stage": stage,
        "secondary_error_tags": list(tags or []),
        "correct_with_substantive_error_tag_conflict": False,
        "conflict_assessment": "",
        "adjudication_rationale": "trace-grounded test rationale",
        "confidence": 0.95,
        "codebook_systematic_ambiguity": False,
        "codebook_ambiguity_description": "",
    }


def test_stage2_return_validation_preserves_blinded_content_and_requires_complete_fields():
    original = []
    returned = []
    for index in range(17):
        row = {
            "adjudication_item_id": f"adj-{index:02d}",
            "phase": "stage2_final_adjudication",
            "scene": {"task": index, "trace": ["visible"]},
            "stage1_independent_annotation": {"outcome": "correct"},
            "prior_reviews": [{"review": "A"}, {"review": "B"}],
            "final_adjudication": {},
        }
        original.append(row)
        returned_row = json.loads(json.dumps(row))
        returned_row["final_adjudication"] = _annotation("correct")
        returned.append(returned_row)
    codebook = {
        "outcomes": {"correct": "", "incorrect": ""},
        "primary_failure_stages": {"Evidence": ""},
        "secondary_error_tags": {"stale_evidence_reuse": ""},
    }
    result = validate_stage2_return(
        original_rows=original,
        returned_rows=returned,
        frozen_codebook=codebook,
    )
    assert result["passed"] is True
    returned[0]["scene"]["task"] = "changed"
    assert validate_stage2_return(
        original_rows=original,
        returned_rows=returned,
        frozen_codebook=codebook,
    )["passed"] is False


def test_consensus_uses_17_adjudications_and_7_exact_a_b_agreements():
    audit_key = []
    coordination_key = []
    adjudication_key = []
    stage2 = []
    reviewer_a = []
    reviewer_b = []
    for index in range(24):
        case_id = f"case-{index:02d}"
        item_a = f"a-{index:02d}"
        item_b = f"b-{index:02d}"
        audit_key.append(
            {
                "case_id": case_id,
                "trace_id": f"trace-{index:02d}",
                "episode_id": f"episode-{index:02d}",
                "reviewer_a_item_id": item_a,
                "reviewer_b_item_id": item_b,
            }
        )
        common = {
            "trajectory_outcome": "correct",
            "primary_failure_stage": None,
            "secondary_error_tags": [],
        }
        reviewer_a.append({"item_id": item_a, "annotation": dict(common)})
        reviewer_b.append({"item_id": item_b, "annotation": dict(common)})
        if index < 17:
            coordination_key.append({"case_id": case_id})
            adjudication_id = f"adj-{index:02d}"
            adjudication_key.append(
                {"case_id": case_id, "adjudication_item_id": adjudication_id}
            )
            stage2.append(
                {
                    "adjudication_item_id": adjudication_id,
                    "final_adjudication": _annotation(
                        "incorrect" if index < 8 else "correct",
                        stage="Claim" if index < 8 else None,
                    ),
                }
            )
    consensus = build_human_consensus(
        stage2_returned=stage2,
        adjudication_manager_key=adjudication_key,
        coordination_manager_key=coordination_key,
        audit_manager_key=audit_key,
        reviewer_a=reviewer_a,
        reviewer_b=reviewer_b,
    )
    assert len(consensus) == 24
    assert sum(row["trajectory_outcome"] == "correct" for row in consensus) == 16
    assert sum(row["trajectory_outcome"] == "incorrect" for row in consensus) == 8
    assert sum(row["consensus_source"] == "stage2_independent_adjudication" for row in consensus) == 17
    assert sum(row["consensus_source"] == "independent_a_b_exact_agreement" for row in consensus) == 7


def test_perfect_estimable_predictions_remain_inconclusive_when_critical_label_has_no_positive_cases():
    consensus = []
    predictions = []
    for index in range(24):
        outcome = "incorrect" if index < 8 else "correct"
        tags = []
        if index == 0:
            tags = ["stale_evidence_reuse"]
        elif index == 1:
            tags = ["missing_required_revalidation"]
        stage = "Evidence" if outcome == "incorrect" else None
        trace_id = f"trace-{index:02d}"
        consensus.append(
            {
                "trace_id": trace_id,
                "trajectory_outcome": outcome,
                "primary_failure_stage": stage,
                "secondary_error_tags": tags,
                "codebook_systematic_ambiguity": False,
            }
        )
        predictions.append(
            {
                "trace_id": trace_id,
                "prediction": {
                    "correct": outcome == "correct",
                    "primary_failure_stage": stage,
                    "secondary_error_tags": tags,
                },
            }
        )
    report = score_judge_predictions(
        consensus_rows=consensus,
        prediction_rows=predictions,
        config=_config(),
    )
    assert report["outcome"]["passed"] is True
    assert report["critical_labels"]["stale_evidence_reuse"]["status"] == "passed"
    assert report["critical_labels"]["missing_required_revalidation"]["status"] == "passed"
    assert report["critical_labels"]["unsupported_success_claim"]["status"] == "inconclusive_no_positive_human_cases"
    assert report["judge_freeze_allowed"] is False
    assert report["status"] == "inconclusive"
    assert "bootstrap_95_ci" in report["critical_labels"]["stale_evidence_reuse"]


def test_scoring_rejects_prediction_trace_set_mismatch():
    consensus = [
        {
            "trace_id": f"trace-{index:02d}",
            "trajectory_outcome": "correct",
            "primary_failure_stage": None,
            "secondary_error_tags": [],
            "codebook_systematic_ambiguity": False,
        }
        for index in range(24)
    ]
    predictions = [
        {
            "trace_id": f"other-{index:02d}",
            "prediction": {
                "correct": True,
                "primary_failure_stage": None,
                "secondary_error_tags": [],
            },
        }
        for index in range(24)
    ]
    with pytest.raises(ValueError, match="trace IDs differ"):
        score_judge_predictions(
            consensus_rows=consensus,
            prediction_rows=predictions,
            config=_config(),
        )


def test_prediction_decryption_verifies_ciphertext_and_plaintext_hashes():
    rows = [
        {
            "trace_id": f"trace-{index:02d}",
            "prediction": {
                "correct": True,
                "primary_failure_stage": None,
                "secondary_error_tags": [],
            },
        }
        for index in range(24)
    ]
    plaintext = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
    key = AESGCM.generate_key(bit_length=256)
    nonce = bytes(range(12))
    archive = PREDICTION_MAGIC + nonce + AESGCM(key).encrypt(
        nonce, plaintext, PREDICTION_AAD
    )
    commitment = {
        "encrypted_predictions_sha256": hashlib.sha256(archive).hexdigest(),
        "plaintext_predictions_sha256": hashlib.sha256(plaintext).hexdigest(),
    }
    decoded, decoded_plaintext = decrypt_prediction_rows(
        archive=archive,
        key_encoded=base64.urlsafe_b64encode(key) + b"\n",
        commitment=commitment,
    )
    assert decoded == rows
    assert decoded_plaintext == plaintext
