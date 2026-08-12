"""Paired statistical tests for v3 / v3_orchestrator paired evaluation.

Plan v2.5 §2.1 A6 + §3.3 G3 gate.

Public surface:
  - load_paired_rows(...)       — pivot v3 / v3_orchestrator JSONL into a frame
  - paired_delta_stats(...)     — Wilcoxon + bootstrap CI + per-circuit Δ
  - mcnemar_target_success(...) — McNemar on target_fidelity hit/miss
  - render_report(...)          — markdown stats report

Each row in the canonical input format (PaperGradeResultRow / write_public_agent_results)
has at least:
    system, circuit, drift_profile, seed, best_score, target_success, total_cost_usd

Pairing key = (circuit, drift_profile, seed). Two systems are paired iff
they share at least one such key with non-null best_score on both sides.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Iterator

# ── data loading ──────────────────────────────────────────────────────────


def load_paired_rows(jsonl_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Read one or more result_summary.jsonl files and return flat row dicts.

    Each row carries: system, circuit, drift_profile, seed, best_score,
    target_success, total_cost_usd, oracle_feasible, oracle_best.
    Diagnostics field is preserved for downstream filtering.
    """
    out: list[dict[str, Any]] = []
    for p in jsonl_paths:
        for ln in Path(p).read_text().splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            diag = r.get("diagnostics") or {}
            out.append({
                "system": r.get("system"),
                "circuit": r.get("circuit"),
                "drift_profile": r.get("drift_profile"),
                "seed": r.get("seed"),
                "best_score": r.get("best_score"),
                "final_score": r.get("final_score"),
                "target_success": bool(r.get("target_success", False)),
                "total_cost_usd": float(r.get("total_cost_usd") or 0.0),
                "elapsed_seconds": float(r.get("elapsed_seconds") or 0.0),
                "oracle_best": float(diag.get("oracle_best", 0.0)
                                     or (r.get("metrics") or {}).get("oracle_best", 0.0)),
                "oracle_feasible": bool(diag.get("oracle_feasible", False)),
                "oracle_best_action": diag.get("oracle_best_action"),
                "tool_calls": int(r.get("tool_calls") or 0),
                "verifier_satisfied": diag.get("verifier_satisfied"),
                "verifier_hallucinated": diag.get("verifier_hallucinated"),
                "stub_verifier": diag.get("stub_verifier"),
                "raw": r,  # preserve raw for downstream
            })
    return out


def pivot_paired(
    rows: list[dict[str, Any]],
    system_a: str,
    system_b: str,
    *,
    metric: str = "best_score",
    require_both: bool = True,
) -> list[dict[str, Any]]:
    """Pivot two systems on (circuit, drift_profile, seed) into paired records.

    Returned record: {circuit, drift_profile, seed, a, b, delta, ...}
    where a / b are the metric values for ``system_a`` / ``system_b``.
    """
    by_key_a: dict[tuple, dict[str, Any]] = {}
    by_key_b: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        if r.get(metric) is None:
            continue
        key = (r["circuit"], r["drift_profile"], r["seed"])
        if r["system"] == system_a:
            by_key_a[key] = r
        elif r["system"] == system_b:
            by_key_b[key] = r

    out = []
    for key in sorted(set(by_key_a) | set(by_key_b)):
        a = by_key_a.get(key)
        b = by_key_b.get(key)
        if require_both and (a is None or b is None):
            continue
        out.append({
            "circuit": key[0],
            "drift_profile": key[1],
            "seed": key[2],
            "a": a.get(metric) if a else None,
            "b": b.get(metric) if b else None,
            "delta": (a[metric] - b[metric]) if (a and b) else None,
            "a_oracle_feasible": a.get("oracle_feasible") if a else None,
            "b_oracle_feasible": b.get("oracle_feasible") if b else None,
            "a_target_success": a.get("target_success") if a else None,
            "b_target_success": b.get("target_success") if b else None,
            "a_cost": a.get("total_cost_usd") if a else None,
            "b_cost": b.get("total_cost_usd") if b else None,
        })
    return out


# ── statistics ────────────────────────────────────────────────────────────


def wilcoxon_signed_rank(deltas: list[float]) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank test using scipy.stats.

    Returns {n, n_nonzero, statistic, p_value, median_delta, mean_delta}.
    """
    from scipy.stats import wilcoxon

    nz = [d for d in deltas if d != 0]
    if len(nz) < 2:
        return {
            "n": len(deltas),
            "n_nonzero": len(nz),
            "statistic": float("nan"),
            "p_value": float("nan"),
            "median_delta": median(deltas) if deltas else 0.0,
            "mean_delta": (sum(deltas) / len(deltas)) if deltas else 0.0,
        }
    stat = wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
    return {
        "n": len(deltas),
        "n_nonzero": len(nz),
        "statistic": float(stat.statistic),
        "p_value": float(stat.pvalue),
        "median_delta": float(median(deltas)),
        "mean_delta": float(sum(deltas) / len(deltas)),
    }


def bootstrap_ci_mean(
    deltas: list[float],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap CI for the mean of a paired Δ vector."""
    import random
    if not deltas:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return float(lo), float(hi)


def mcnemar_target_success(
    paired: list[dict[str, Any]],
) -> dict[str, float]:
    """McNemar on target_success outcomes between system_a (A) and system_b (B).

    Treat A=1 / B=1 as 'both pass', etc. McNemar tests whether the
    discordant cells (A=1,B=0) and (A=0,B=1) differ.
    """
    from scipy.stats import binomtest

    n11 = n10 = n01 = n00 = 0
    for r in paired:
        a = r.get("a_target_success")
        b = r.get("b_target_success")
        if a is None or b is None:
            continue
        if a and b:
            n11 += 1
        elif a and not b:
            n10 += 1
        elif (not a) and b:
            n01 += 1
        else:
            n00 += 1
    n_disc = n10 + n01
    if n_disc == 0:
        p = 1.0
    else:
        p = float(binomtest(n10, n_disc, p=0.5, alternative="two-sided").pvalue)
    return {
        "n11_both_pass": n11,
        "n10_only_a": n10,
        "n01_only_b": n01,
        "n00_both_fail": n00,
        "p_value": p,
        "discordant": n_disc,
    }


def per_circuit_summary(
    paired: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group paired records by circuit and run Wilcoxon per circuit."""
    by_circuit: dict[str, list[float]] = defaultdict(list)
    for r in paired:
        if r.get("delta") is not None:
            by_circuit[r["circuit"]].append(r["delta"])
    out = []
    for circuit, deltas in sorted(by_circuit.items()):
        s = wilcoxon_signed_rank(deltas)
        ci_lo, ci_hi = bootstrap_ci_mean(deltas, n_boot=5000)
        out.append({
            "circuit": circuit,
            "n": s["n"],
            "mean_delta": s["mean_delta"],
            "median_delta": s["median_delta"],
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "wilcoxon_p": s["p_value"],
            "n_pos": sum(1 for d in deltas if d > 0),
            "n_neg": sum(1 for d in deltas if d < 0),
        })
    return out


def per_profile_summary(
    paired: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_profile: dict[str, list[float]] = defaultdict(list)
    for r in paired:
        if r.get("delta") is not None:
            by_profile[r["drift_profile"]].append(r["delta"])
    out = []
    for profile, deltas in sorted(by_profile.items()):
        s = wilcoxon_signed_rank(deltas)
        ci_lo, ci_hi = bootstrap_ci_mean(deltas, n_boot=5000)
        out.append({
            "drift_profile": profile,
            "n": s["n"],
            "mean_delta": s["mean_delta"],
            "median_delta": s["median_delta"],
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "wilcoxon_p": s["p_value"],
        })
    return out


# ── one-shot driver ───────────────────────────────────────────────────────


def paired_delta_stats(
    rows: list[dict[str, Any]],
    system_a: str,
    system_b: str,
    *,
    metric: str = "best_score",
) -> dict[str, Any]:
    """Run the full battery (Wilcoxon, bootstrap, McNemar, per-circuit, per-profile)
    for ``a − b`` on the given metric.

    Convention: ``delta > 0`` means ``a`` wins.
    """
    paired = pivot_paired(rows, system_a, system_b, metric=metric)
    deltas = [r["delta"] for r in paired if r.get("delta") is not None]
    overall = wilcoxon_signed_rank(deltas)
    ci_lo, ci_hi = bootstrap_ci_mean(deltas)
    target = mcnemar_target_success(paired)
    cost_a = sum((r.get("a_cost") or 0.0) for r in paired)
    cost_b = sum((r.get("b_cost") or 0.0) for r in paired)
    return {
        "system_a": system_a,
        "system_b": system_b,
        "metric": metric,
        "n_paired": len(paired),
        "overall": {
            **overall,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "n_pos": sum(1 for d in deltas if d > 0),
            "n_neg": sum(1 for d in deltas if d < 0),
        },
        "per_circuit": per_circuit_summary(paired),
        "per_profile": per_profile_summary(paired),
        "target_success_mcnemar": target,
        "total_cost_a_usd": cost_a,
        "total_cost_b_usd": cost_b,
        "cost_savings_pct": (
            ((cost_b - cost_a) / cost_b * 100.0) if cost_b > 0 else float("nan")
        ),
    }


# ── markdown report ───────────────────────────────────────────────────────


def render_report(stats: dict[str, Any]) -> str:
    """Render a paired_delta_stats dict as a markdown report."""
    a = stats["system_a"]
    b = stats["system_b"]
    ov = stats["overall"]
    lines = []
    lines.append(f"# Paired Δ {a} − {b}  ({stats['metric']})")
    lines.append("")
    lines.append(f"**N paired** = {stats['n_paired']}")
    lines.append(f"**mean Δ** = {ov['mean_delta']:+.4f}")
    lines.append(f"**median Δ** = {ov['median_delta']:+.4f}")
    lines.append(f"**95% CI (mean)** = [{ov['ci95_lo']:+.4f}, {ov['ci95_hi']:+.4f}]")
    p = ov["p_value"]
    lines.append(f"**Wilcoxon p** = {p:.4g}"
                 + ("  ✓ p<0.05" if p < 0.05 else
                    ("  ~ p<0.10" if p < 0.10 else "  (n.s.)")))
    lines.append(f"**signs** = +{ov['n_pos']} / −{ov['n_neg']}")
    lines.append("")
    lines.append("## Target-success McNemar")
    ts = stats["target_success_mcnemar"]
    lines.append(f"  | b=pass | b=fail")
    lines.append(f"a=pass | {ts['n11_both_pass']} | {ts['n10_only_a']}")
    lines.append(f"a=fail | {ts['n01_only_b']} | {ts['n00_both_fail']}")
    lines.append(f"discordant = {ts['discordant']}, p = {ts['p_value']:.4g}")
    lines.append("")
    lines.append("## Cost")
    lines.append(f"total {a} = ${stats['total_cost_a_usd']:.2f}")
    lines.append(f"total {b} = ${stats['total_cost_b_usd']:.2f}")
    cs = stats["cost_savings_pct"]
    if not (cs != cs):  # not NaN
        lines.append(f"savings of {a} vs {b}: {cs:+.1f}%")
    lines.append("")
    lines.append("## Per-circuit Δ")
    lines.append("| circuit | n | mean Δ | median Δ | 95% CI | Wilcoxon p | +/− |")
    lines.append("|---------|---|--------|----------|--------|-----------|-----|")
    for c in stats["per_circuit"]:
        marker = (" ✓" if c["wilcoxon_p"] < 0.05 else
                  (" ~" if c["wilcoxon_p"] < 0.10 else ""))
        lines.append(
            f"| {c['circuit']} | {c['n']} | {c['mean_delta']:+.4f} | "
            f"{c['median_delta']:+.4f} | "
            f"[{c['ci95_lo']:+.4f}, {c['ci95_hi']:+.4f}] | "
            f"{c['wilcoxon_p']:.4g}{marker} | "
            f"+{c['n_pos']}/−{c['n_neg']} |"
        )
    lines.append("")
    lines.append("## Per-profile Δ")
    lines.append("| profile | n | mean Δ | median Δ | 95% CI | Wilcoxon p |")
    lines.append("|---------|---|--------|----------|--------|-----------|")
    for p in stats["per_profile"]:
        marker = (" ✓" if p["wilcoxon_p"] < 0.05 else
                  (" ~" if p["wilcoxon_p"] < 0.10 else ""))
        lines.append(
            f"| {p['drift_profile']} | {p['n']} | {p['mean_delta']:+.4f} | "
            f"{p['median_delta']:+.4f} | "
            f"[{p['ci95_lo']:+.4f}, {p['ci95_hi']:+.4f}] | "
            f"{p['wilcoxon_p']:.4g}{marker} |"
        )
    return "\n".join(lines) + "\n"


# ── CLI ───────────────────────────────────────────────────────────────────


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more result_summary.jsonl files")
    ap.add_argument("--system-a", required=True)
    ap.add_argument("--system-b", required=True)
    ap.add_argument("--metric", default="best_score")
    ap.add_argument("--out-md", default=None,
                    help="Write markdown report; default = stdout")
    ap.add_argument("--out-json", default=None,
                    help="Also dump the raw stats dict as JSON")
    args = ap.parse_args()

    rows = load_paired_rows(args.inputs)
    stats = paired_delta_stats(rows, args.system_a, args.system_b,
                               metric=args.metric)
    md = render_report(stats)
    if args.out_md:
        Path(args.out_md).write_text(md)
        print(f"wrote {args.out_md}")
    else:
        print(md)
    if args.out_json:
        # strip non-json-safe entries (none currently, but defensive)
        Path(args.out_json).write_text(json.dumps(stats, indent=2, default=str))
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    _cli()
