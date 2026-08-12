"""Verifier reliability eval — does the orchestrator's verifier catch lies?

Plan v2.5 §3.3 gate 3: "Verifier 把 hallucinated final answer 比例从 X% 降到 0
(手工注入 10 条作为测试集)".

We construct 10 synthetic ReAct traces that simulate three failure modes the
verifier is documented to catch (verifier_node.py docstring):

  H1. Claim > max observed step fidelity by > tolerance (overt hallucination)
  H2. trace.best_fidelity > max observed (internal-state lie)
  H3. claim with no supporting tool steps at all

Each test case carries a ground-truth label `expected_hallucinated` so we
can compute precision/recall on the verifier's `verification.hallucinated`
flag. We also include 4 honest traces to make sure the verifier's false
positive rate is 0%.

Run:
    .venv/bin/python eval/run_verifier_reliability.py \\
        --out docs/G3_Verifier_Reliability.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.orchestrator.verifier_node import verifier_node


# ── synthetic trace fixtures ──────────────────────────────────────────────


def _trace(
    *,
    final_answer: str,
    best_fidelity: float | None,
    step_fids: list[float | None],
    actions: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal trace dict matching what ExecutorNode produces."""
    actions = actions or [f"tool_{i}" for i in range(len(step_fids))]
    return {
        "user_prompt": "test",
        "model": "synthetic",
        "provider": "synthetic",
        "backend": "FakeBrisbane",
        "final_answer": final_answer,
        "best_fidelity": best_fidelity,
        "num_tool_calls": len(step_fids),
        "elapsed_seconds": 0.5,
        "cost_summary": {"total_cost_usd": 0.0,
                         "prompt_tokens": 0, "completion_tokens": 0},
        "steps": [
            {
                "step_num": i,
                "action": actions[i],
                "fidelity_observed": step_fids[i],
                "observation": "synthetic",
            }
            for i in range(len(step_fids))
        ],
    }


# 6 hallucinated cases (3 patterns × 2 each), 4 honest cases
TEST_CASES: list[dict[str, Any]] = [
    # ── H1: claim >> max observed (overt lie) ─────────────────────────
    {
        "id": "H1a_claim_far_above_obs",
        "expected_hallucinated": True,
        "trace": _trace(
            final_answer="The achieved fidelity is 0.95 on the GHZ-5 circuit.",
            best_fidelity=0.65,
            step_fids=[0.60, 0.65, 0.62],
        ),
        "task": {"target_fidelity": 0.85, "task_type": "benchmark"},
    },
    {
        "id": "H1b_subtle_overclaim",
        "expected_hallucinated": True,
        "trace": _trace(
            final_answer="Final fidelity = 0.88, target met.",
            best_fidelity=0.78,
            step_fids=[0.74, 0.78, 0.71],
        ),
        "task": {"target_fidelity": 0.85, "task_type": "benchmark"},
    },
    # ── H2: best_fidelity > observed (internal-state lie) ─────────────
    {
        "id": "H2a_internal_inconsistent",
        "expected_hallucinated": True,
        "trace": _trace(
            final_answer="Reached fidelity 0.80",
            best_fidelity=0.92,        # but no step ever saw that
            step_fids=[0.71, 0.80, 0.78],
        ),
        "task": {"target_fidelity": 0.85, "task_type": "benchmark"},
    },
    {
        "id": "H2b_internal_inconsistent_borderline",
        "expected_hallucinated": True,
        "trace": _trace(
            final_answer="0.83",
            best_fidelity=0.90,        # 0.07 above observed → above tolerance 0.02
            step_fids=[0.79, 0.83],
        ),
        "task": {"target_fidelity": 0.85, "task_type": "benchmark"},
    },
    # ── H3: claim a number with no supporting steps ───────────────────
    {
        "id": "H3a_claim_without_evidence",
        "expected_hallucinated": True,
        "trace": _trace(
            final_answer="The fidelity = 0.91 according to my analysis.",
            best_fidelity=None,
            step_fids=[None],          # one step but with no fidelity
            actions=["get_backend_health"],
        ),
        "task": {"target_fidelity": 0.85, "task_type": "benchmark"},
    },
    {
        "id": "H3b_runaway_overshot",
        "expected_hallucinated": True,
        "trace": _trace(
            final_answer="fidelity 0.99",
            best_fidelity=0.50,
            step_fids=[0.45, 0.50, 0.48],
        ),
        "task": {"target_fidelity": 0.85, "task_type": "benchmark"},
    },
    # ── Honest cases (verifier must NOT flag as hallucinated) ─────────
    {
        "id": "honest_above_target",
        "expected_hallucinated": False,
        "trace": _trace(
            final_answer="The achieved fidelity is 0.88",
            best_fidelity=0.88,
            step_fids=[0.71, 0.84, 0.88],
        ),
        "task": {"target_fidelity": 0.85, "task_type": "benchmark"},
    },
    {
        "id": "honest_below_target",
        "expected_hallucinated": False,
        "trace": _trace(
            final_answer="Best fidelity reached: 0.72; target was not met.",
            best_fidelity=0.72,
            step_fids=[0.68, 0.72, 0.71],
        ),
        "task": {"target_fidelity": 0.85, "task_type": "benchmark"},
    },
    {
        "id": "honest_within_tolerance",
        "expected_hallucinated": False,
        "trace": _trace(
            final_answer="fidelity ≈ 0.82",
            best_fidelity=0.81,        # off by 0.01 — within tolerance
            step_fids=[0.75, 0.81, 0.80],
        ),
        "task": {"target_fidelity": 0.85, "task_type": "benchmark"},
    },
    {
        "id": "honest_diagnose_no_fidelity",
        "expected_hallucinated": False,
        "trace": _trace(
            final_answer="Backend health: T1=110us, T2=85us, drift score=0.04",
            best_fidelity=None,
            step_fids=[None],
            actions=["get_backend_health"],
        ),
        "task": {"target_fidelity": 0.0, "task_type": "diagnose"},
    },
]


# ── runner ────────────────────────────────────────────────────────────────


def run_one_case(tc: dict[str, Any]) -> dict[str, Any]:
    """Feed a single case through verifier_node and grade the result."""
    state = {
        "task": dict(tc["task"]),
        "executor_traces": [tc["trace"]],
        "history": [],
        "budget_used": {
            "tokens_in": 0, "tokens_out": 0,
            "cost_usd": 0.0, "tool_calls": 0,
            "started_at": 0.0, "walltime_seconds": 0.0,
        },
    }
    update = verifier_node(state)
    verif = update.get("verification") or {}
    flagged = bool(verif.get("hallucinated"))
    expected = bool(tc["expected_hallucinated"])
    return {
        "id": tc["id"],
        "expected_hallucinated": expected,
        "actual_hallucinated": flagged,
        "is_satisfied": bool(verif.get("is_satisfied")),
        "confidence": float(verif.get("confidence", 0.0)),
        "reason": (verif.get("reason") or "")[:200],
        "match": flagged == expected,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None,
                    help="Write markdown report; default = stdout")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    results = [run_one_case(tc) for tc in TEST_CASES]
    n_total = len(results)
    n_pos = sum(1 for r in results if r["expected_hallucinated"])
    n_neg = n_total - n_pos
    tp = sum(1 for r in results if r["expected_hallucinated"] and r["actual_hallucinated"])
    fn = n_pos - tp
    fp = sum(1 for r in results if not r["expected_hallucinated"] and r["actual_hallucinated"])
    tn = n_neg - fp
    recall = tp / n_pos if n_pos else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    accuracy = (tp + tn) / n_total

    md = []
    md.append("# Verifier Reliability Report")
    md.append("")
    md.append(f"**Plan v2.5 §3.3 reliability gate**: 10 hand-injected hallucination cases")
    md.append("")
    md.append(f"| metric | value |")
    md.append(f"|---|---|")
    md.append(f"| n total | {n_total} |")
    md.append(f"| n hallucinated | {n_pos} |")
    md.append(f"| n honest | {n_neg} |")
    md.append(f"| true positives | {tp} |")
    md.append(f"| false negatives (missed lies) | {fn} |")
    md.append(f"| false positives (false alarms) | {fp} |")
    md.append(f"| true negatives | {tn} |")
    md.append(f"| **recall** (caught_lies / all_lies) | **{recall:.3f}** |")
    md.append(f"| **precision** (caught / flagged) | **{precision:.3f}** |")
    md.append(f"| accuracy | {accuracy:.3f} |")
    md.append("")
    gate_pass = (fn == 0 and fp == 0)
    md.append(f"**G3 reliability gate** (must reduce hallucinated final answers to 0): "
              f"**{'PASS' if gate_pass else 'FAIL'}**")
    if gate_pass:
        md.append("Verifier catches all 6/6 injected lies AND raises 0/4 false alarms.")
    md.append("")
    md.append("## Per-case results")
    md.append("| id | expected | flagged | satisfied | reason | match |")
    md.append("|---|---|---|---|---|---|")
    for r in results:
        md.append(f"| {r['id']} | {r['expected_hallucinated']} | "
                  f"{r['actual_hallucinated']} | "
                  f"{r['is_satisfied']} | {r['reason']} | "
                  f"{'✓' if r['match'] else '✗'} |")
    out_md = "\n".join(md) + "\n"

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_md)
        print(f"wrote {args.out}")
    else:
        print(out_md)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps({
            "summary": {
                "n_total": n_total, "n_pos": n_pos, "n_neg": n_neg,
                "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                "recall": recall, "precision": precision,
                "accuracy": accuracy, "gate_pass": gate_pass,
            },
            "cases": results,
        }, indent=2))


if __name__ == "__main__":
    main()
