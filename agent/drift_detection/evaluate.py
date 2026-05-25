"""Drift detector evaluation — synthetic benchmark with labeled changepoints.

Generates scenarios with known ground-truth changepoints using SyntheticDriftBackend,
then evaluates PELT / BOCPD / ensemble against them.

Key design: only label ABRUPT changes as changepoints. Gradual/periodic drift
is NOT a changepoint — those test specificity (false positive control).

Usage:
    python -m detection.evaluate          # full benchmark
    python -m detection.evaluate --quick  # sanity check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backends.synthetic_drift import (
    SyntheticDriftBackend,
    DriftProfile,
    STABLE,
    LINEAR_DECAY,
    SUDDEN_DEGRADATION,
)
from backends.properties_stream import PropertiesStream
from agent.drift_detection.drift_detector import DriftDetector, DriftReport


# ──────────────────────────────────────────────
#  Scenario definitions with ground-truth CPs
# ──────────────────────────────────────────────

@dataclass
class Scenario:
    name: str
    profile: DriftProfile
    hours: float
    step_hours: float
    ground_truth_hours: list[float]
    tolerance_hours: float = 1.5

    @property
    def n_steps(self) -> int:
        return int(self.hours / self.step_hours) + 1

    def truth_indices(self) -> list[int]:
        return [round(h / self.step_hours) for h in self.ground_truth_hours]


def _step_profile(shifts: list[tuple[float, float, float, float]]) -> DriftProfile:
    """Create a step-function profile from (time, t1_factor, gate_factor, readout_factor) tuples.

    Each tuple means: "at time >= t, apply these factors."
    """
    shifts = sorted(shifts, key=lambda x: x[0], reverse=True)

    def make_fn(idx):
        def fn(t):
            for s in shifts:
                if t >= s[0]:
                    return s[idx]
            return 1.0
        return fn

    return DriftProfile(
        t1_drift=make_fn(1),
        gate_error_drift=make_fn(2),
        readout_drift=make_fn(3),
    )


def build_scenarios() -> list[Scenario]:
    """Build diverse scenarios totaling ~100+ labeled changepoints."""
    scenarios = []
    rng = np.random.RandomState(42)

    # ── Type 1: Stable (no CPs) — 10 scenarios for false-positive control ──
    for i in range(10):
        # Add tiny noise to make it non-trivial
        noise = 1.0 + rng.uniform(-0.02, 0.02)
        profile = DriftProfile(
            t1_drift=lambda t, n=noise: n,
            gate_error_drift=lambda t, n=noise: 1.0 / n,
        )
        scenarios.append(Scenario(
            name=f"stable_{i}", profile=profile,
            hours=24, step_hours=0.5, ground_truth_hours=[],
        ))

    # ── Type 2: Gradual linear (no abrupt CP) — 5 scenarios ──
    for i in range(5):
        rate = 0.01 + i * 0.005
        profile = DriftProfile(
            t1_drift=lambda t, r=rate: max(0.3, 1.0 - r * t),
            gate_error_drift=lambda t, r=rate: 1.0 + r * t,
        )
        scenarios.append(Scenario(
            name=f"gradual_{i}", profile=profile,
            hours=24, step_hours=0.5, ground_truth_hours=[],
        ))

    # ── Type 3: Single sudden shift — 25 scenarios (1 CP each = 25 CPs) ──
    for i, shift_t in enumerate(np.linspace(3, 21, 25)):
        shift_t = float(round(shift_t * 2) / 2)  # snap to 0.5h grid
        severity = 0.3 + rng.random() * 0.5  # t1 drops to 30–80%
        profile = _step_profile([(shift_t, severity, 1.0 / severity, 1.0 + (1 - severity))])
        scenarios.append(Scenario(
            name=f"sudden_{i}_at_{shift_t}h",
            profile=profile,
            hours=24, step_hours=0.5,
            ground_truth_hours=[shift_t],
        ))

    # ── Type 4: Double shift — 20 scenarios (2 CPs each = 40 CPs) ──
    for i in range(20):
        t1 = float(round(rng.uniform(3, 9) * 2) / 2)
        t2 = float(round(rng.uniform(t1 + 3, 21) * 2) / 2)
        s1 = 0.3 + rng.random() * 0.3  # first shift
        s2 = 0.6 + rng.random() * 0.3  # partial recovery or further degradation
        profile = _step_profile([
            (t1, s1, 1.0 / s1, 1.0 + (1 - s1)),
            (t2, s2, 1.0 / s2, 1.0 + (1 - s2)),
        ])
        scenarios.append(Scenario(
            name=f"double_{i}",
            profile=profile,
            hours=24, step_hours=0.5,
            ground_truth_hours=[t1, t2],
        ))

    # ── Type 5: Triple staircase — 10 scenarios (3 CPs each = 30 CPs) ──
    for i in range(10):
        t1 = float(round(rng.uniform(2, 6) * 2) / 2)
        t2 = float(round(rng.uniform(t1 + 2, 14) * 2) / 2)
        t3 = float(round(rng.uniform(t2 + 2, 21) * 2) / 2)
        profile = _step_profile([
            (t1, 0.8, 1.25, 1.1),
            (t2, 0.6, 1.67, 1.3),
            (t3, 0.4, 2.5, 1.6),
        ])
        scenarios.append(Scenario(
            name=f"staircase_{i}",
            profile=profile,
            hours=24, step_hours=0.5,
            ground_truth_hours=[t1, t2, t3],
        ))

    # ── Type 6: Shift + recovery — 10 scenarios (2 CPs each = 20 CPs) ──
    for i in range(10):
        t_down = float(round(rng.uniform(4, 10) * 2) / 2)
        t_up = float(round(rng.uniform(t_down + 3, 20) * 2) / 2)
        profile = _step_profile([
            (t_down, 0.3, 3.0, 2.0),
            (t_up, 0.95, 1.05, 1.02),  # near-full recovery
        ])
        scenarios.append(Scenario(
            name=f"recovery_{i}",
            profile=profile,
            hours=24, step_hours=0.5,
            ground_truth_hours=[t_down, t_up],
        ))

    total = sum(len(s.ground_truth_hours) for s in scenarios)
    print(f"Built {len(scenarios)} scenarios with {total} total labeled CPs")
    return scenarios


# ──────────────────────────────────────────────
#  Matching and F1 computation
# ──────────────────────────────────────────────

def match_changepoints(
    detected: list[int],
    truth: list[int],
    tolerance: int,
) -> tuple[int, int, int]:
    """Greedy matching. Returns (TP, FP, FN)."""
    truth_matched = set()
    detected_matched = set()

    for d_idx in sorted(detected):
        best_dist = tolerance + 1
        best_truth = -1
        for t_idx in truth:
            if t_idx in truth_matched:
                continue
            dist = abs(d_idx - t_idx)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_truth = t_idx
        if best_truth >= 0:
            truth_matched.add(best_truth)
            detected_matched.add(d_idx)

    tp = len(truth_matched)
    fp = len(detected) - len(detected_matched)
    fn = len(truth) - tp
    return tp, fp, fn


def compute_f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


# ──────────────────────────────────────────────
#  Evaluation runner
# ──────────────────────────────────────────────

def evaluate(
    methods: list[str] = ["pelt", "bocpd", "ensemble"],
    quick: bool = False,
) -> dict:
    scenarios = build_scenarios()

    if quick:
        seen = set()
        filtered = []
        for s in scenarios:
            prefix = s.name.split("_")[0]
            if prefix not in seen:
                seen.add(prefix)
                filtered.append(s)
        scenarios = filtered

    total_labeled = sum(len(s.ground_truth_hours) for s in scenarios)
    print(f"Evaluating {len(scenarios)} scenarios, {total_labeled} labeled CPs")

    # Parameter grids
    param_grids = {
        "pelt": [
            {"pelt_penalty": p, "pelt_model": m}
            for p in ([1.0, 2.0, 3.0, 5.0, 8.0] if not quick else [2.0, 3.0])
            for m in (["rbf", "l2"] if not quick else ["rbf"])
        ],
        "bocpd": [
            {"bocpd_hazard": h, "bocpd_threshold": t}
            for h in ([1/10, 1/20, 1/30] if not quick else [1/10])
            for t in ([0.05, 0.1, 0.2] if not quick else [0.1])
        ],
        "ensemble": [
            {"pelt_penalty": p, "bocpd_hazard": h, "bocpd_threshold": t}
            for p in ([2.0, 3.0, 5.0] if not quick else [3.0])
            for h in ([1/10, 1/20] if not quick else [1/10])
            for t in ([0.05, 0.1] if not quick else [0.1])
        ],
    }

    results = {}

    for method in methods:
        print(f"\n--- {method.upper()} ---")
        best_f1 = -1
        best_params = {}
        best_detail = None

        for params in param_grids[method]:
            total_tp, total_fp, total_fn = 0, 0, 0
            per_scenario = []

            for scenario in scenarios:
                backend = SyntheticDriftBackend(profile=scenario.profile)
                snapshots = PropertiesStream.replay_all(
                    backend, hours=scenario.hours, step_hours=scenario.step_hours
                )

                detector = DriftDetector(
                    method=method,
                    pelt_penalty=params.get("pelt_penalty", 3.0),
                    pelt_model=params.get("pelt_model", "rbf"),
                    bocpd_hazard=params.get("bocpd_hazard", 1/50),
                    bocpd_threshold=params.get("bocpd_threshold", 0.5),
                )
                report = detector.detect(snapshots)

                detected = [cp.index for cp in report.changepoints]
                truth = scenario.truth_indices()
                tol = max(1, round(scenario.tolerance_hours / scenario.step_hours))

                tp, fp, fn = match_changepoints(detected, truth, tol)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                per_scenario.append({
                    "name": scenario.name, "truth": truth,
                    "detected": detected, "tp": tp, "fp": fp, "fn": fn,
                })

            metrics = compute_f1(total_tp, total_fp, total_fn)
            param_str = " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in params.items())
            print(f"  {param_str}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_params = params
                best_detail = {"metrics": metrics, "per_scenario": per_scenario}

        results[method] = {
            "best_params": best_params,
            "metrics": best_detail["metrics"],
            "per_scenario": best_detail["per_scenario"],
        }
        m = best_detail["metrics"]
        print(f"  ★ BEST: {best_params}")
        print(f"    P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
        print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    results = evaluate(quick=args.quick)
    elapsed = time.time() - t0

    print(f"\nTotal time: {elapsed:.1f}s")
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for method, data in results.items():
        m = data["metrics"]
        print(f"  {method:10s}  F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")

    if args.output:
        out = Path(args.output)
        serializable = {m: {"best_params": d["best_params"], "metrics": d["metrics"],
                            "per_scenario": d["per_scenario"]}
                        for m, d in results.items()}
        out.write_text(json.dumps(serializable, indent=2))
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
