"""Re-grade B10 traces with the post-fix verifier (offline replayer).

B10 production run (eval/results/v3_orchestrator/) was started before two
verifier_node bugs were fixed:

  1. Regex did not match "fidelity is 0.95" / "fidelity ≈ 0.82" patterns
  2. claim with no fidelity_observed in any tool step was not flagged

The B10 process holds the old verifier bytecode in memory until it exits,
so the per-row verifier_satisfied / verifier_hallucinated flags in
result_summary.jsonl reflect the OLD verifier. This script re-scores
every saved post-drift graph state JSON with the CURRENT verifier and
emits an updated JSONL alongside.

Why this matters:
  - Reliability gate evidence comes from `eval/run_verifier_reliability.py`
    on synthetic traces, not B10 — so G3 is not blocked.
  - But if we report "X% of B10 final answers were hallucinated",
    we want the post-fix detection rates, not the pre-fix ones.

Usage:
    .venv/bin/python eval/regrade_b10_verifier.py \\
        --in-dir eval/results/v3_orchestrator \\
        --out eval/results/v3_orchestrator/verifier_regrade.jsonl
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


def regrade_one_state(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Run the current verifier_node against a saved graph state."""
    # The saved state was written by run_orchestrator_b10._dump_state_compact
    # which keeps executor_traces, verification, history, plan, last_*,
    # budget_used. The verifier only reads task and executor_traces.
    update = verifier_node(state_dict)
    return update.get("verification") or {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", default="eval/results/v3_orchestrator",
                    help="B10 result directory")
    ap.add_argument("--out", default=None,
                    help="JSONL of re-graded verdicts; default = <in-dir>/verifier_regrade.jsonl")
    ap.add_argument("--report", default=None,
                    help="Optional summary markdown")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out) if args.out else (in_dir / "verifier_regrade.jsonl")

    # Find every saved post_graph_state.json under in_dir
    state_files = sorted(in_dir.glob("**/post_graph_state.json"))
    print(f"found {len(state_files)} post_graph_state.json files under {in_dir}")

    n_total = 0
    n_lies_old = 0
    n_lies_new = 0
    n_satisfied_old = 0
    n_satisfied_new = 0
    n_changed = 0
    rows = []

    for fp in state_files:
        try:
            state = json.loads(fp.read_text())
        except Exception as exc:
            print(f"  skip {fp}: {exc}", file=sys.stderr)
            continue

        # Recover the (system, profile, circuit, seed) from path layout:
        # in_dir/<system>/<profile>/<circuit>/<seedN>/post_graph_state.json
        rel = fp.relative_to(in_dir).parts
        if len(rel) >= 4:
            system = rel[0]
            profile = rel[1]
            circuit = rel[2]
            seed_str = rel[3].replace("seed", "")
            try:
                seed = int(seed_str)
            except ValueError:
                seed = -1
        else:
            system = profile = circuit = "?"
            seed = -1

        # The saved state is a TRIMMED dict; we need to reconstruct
        # something verifier_node can run against. It reads:
        #   task → target_fidelity, task_type
        #   executor_traces → final_answer, best_fidelity, steps, etc.
        # We don't have the original `task` (run_orchestrator_b10 only
        # dumped the state compact). Re-derive minimal task fields from
        # what we know — assume task_type=benchmark and lift target from
        # last verification.fidelity_check.target if present.
        old_verif = state.get("verification") or {}
        target = float((old_verif.get("fidelity_check") or {}).get("target", 0.0))
        synth_state = {
            "task": {"target_fidelity": target, "task_type": "benchmark"},
            "executor_traces": state.get("executor_traces") or [],
            "history": [],
            "budget_used": state.get("budget_used") or {
                "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
                "tool_calls": 0, "started_at": 0, "walltime_seconds": 0,
            },
        }

        new_verif = regrade_one_state(synth_state)

        old_h = bool(old_verif.get("hallucinated", False))
        new_h = bool(new_verif.get("hallucinated", False))
        old_s = bool(old_verif.get("is_satisfied", False))
        new_s = bool(new_verif.get("is_satisfied", False))

        n_total += 1
        n_lies_old += int(old_h)
        n_lies_new += int(new_h)
        n_satisfied_old += int(old_s)
        n_satisfied_new += int(new_s)
        if old_h != new_h or old_s != new_s:
            n_changed += 1

        rows.append({
            "system": system, "profile": profile, "circuit": circuit,
            "seed": seed,
            "old_hallucinated": old_h, "new_hallucinated": new_h,
            "old_is_satisfied": old_s, "new_is_satisfied": new_s,
            "old_reason": (old_verif.get("reason") or "")[:200],
            "new_reason": (new_verif.get("reason") or "")[:200],
            "trace_file": str(fp.resolve().relative_to(ROOT)),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print()
    print(f"=== regrade summary ===")
    print(f"  n total                 : {n_total}")
    print(f"  n hallucinated (old)    : {n_lies_old}")
    print(f"  n hallucinated (new)    : {n_lies_new}")
    print(f"  n satisfied (old)       : {n_satisfied_old}")
    print(f"  n satisfied (new)       : {n_satisfied_new}")
    print(f"  n changed verdicts      : {n_changed}  ({n_changed/max(n_total,1)*100:.1f}%)")
    print(f"  output JSONL            : {out_path}")

    if args.report:
        # Per-system breakdown
        from collections import defaultdict, Counter
        by_sys = defaultdict(lambda: {
            "n": 0, "halluc_old": 0, "halluc_new": 0,
            "satisfied_old": 0, "satisfied_new": 0,
        })
        for r in rows:
            s = by_sys[r["system"]]
            s["n"] += 1
            s["halluc_old"] += int(r["old_hallucinated"])
            s["halluc_new"] += int(r["new_hallucinated"])
            s["satisfied_old"] += int(r["old_is_satisfied"])
            s["satisfied_new"] += int(r["new_is_satisfied"])

        md = []
        md.append("# B10 Verifier Regrade Report")
        md.append("")
        md.append(f"Re-scored {n_total} B10 traces with the post-fix verifier.")
        md.append("")
        md.append(f"| metric | old verifier | new verifier |")
        md.append(f"|---|---|---|")
        md.append(f"| hallucinated count | {n_lies_old} ({n_lies_old/max(n_total,1)*100:.1f}%) | {n_lies_new} ({n_lies_new/max(n_total,1)*100:.1f}%) |")
        md.append(f"| satisfied count | {n_satisfied_old} ({n_satisfied_old/max(n_total,1)*100:.1f}%) | {n_satisfied_new} ({n_satisfied_new/max(n_total,1)*100:.1f}%) |")
        md.append(f"| changed verdicts | — | {n_changed} ({n_changed/max(n_total,1)*100:.1f}%) |")
        md.append("")
        md.append("## Per-system breakdown")
        md.append("| system | n | hall (old) | hall (new) | sat (old) | sat (new) |")
        md.append("|---|---|---|---|---|---|")
        for sys_name, s in sorted(by_sys.items()):
            md.append(f"| {sys_name} | {s['n']} | "
                      f"{s['halluc_old']} | {s['halluc_new']} | "
                      f"{s['satisfied_old']} | {s['satisfied_new']} |")

        Path(args.report).write_text("\n".join(md) + "\n")
        print(f"  report written          : {args.report}")


if __name__ == "__main__":
    main()
