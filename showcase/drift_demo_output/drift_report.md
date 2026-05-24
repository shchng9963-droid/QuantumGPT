# QuantumGPT Drift-Aware Demo Report

Goal: show that when synthetic hardware drift is injected mid-experiment,
the drift-aware agent detects the change, invalidates stale results,
and re-runs the affected circuit — while a drift-naive baseline does not.

## Setup

- Backend: SyntheticDriftBackend(FakeBrisbane)
- Drift profile: SUDDEN_DEGRADATION applied at t=6h
- Provider: mock (deterministic rule planner) — no API key required
- Prompt: "Run the ghz_5 circuit and report the fidelity. If hardware drift is detected, recover and re-run before answering."

## Backend state

| Metric | Stable (t=0h) | Drift (t=6h) | Δ |
|---|---|---|---|
| Avg T1 (μs) | 224.5 | 67.3 | -157.1 |
| Avg 1Q error | 0.0004 | 0.0018 | 0.0015 |
| Avg readout error | 0.0294 | 0.0801 | 0.0508 |
| Drift score | 0.0000 | 1.0000 | — |

## Agent runs

| Run | Best fidelity | Tool calls | Drift alerts | Run_circuit count |
|---|---|---|---|---|
| A — Stable | 0.9062 | 2 | 0 | 1 |
| B — Drift + drift-aware | 0.7624 | 5 | 2 | 1 |
| C — Drift + drift-naive | 0.7391 | 4 | 1 | 1 |

## Headline numbers

- Drift hit on the naive baseline: **Δ fidelity = 0.1671** (Phase A → Phase C).
- Residual hit after drift-aware recovery: **Δ fidelity = 0.1438**.
- Drift-aware recovery gain over the naive baseline: **+0.0233**.
- Drift-aware agent issued **2** drift alert(s) and re-ran the circuit **1** time(s).

## Drift-aware trace highlights
- step 1: ⚠️ DRIFT DETECTED (score=1.000). Previous results are invalidated. Re-checking backend health.
- step 4: Checking for hardware drift.

## Files

- `drift_timeline.png` / `drift_timeline.pdf` — three-panel figure
- `drift_trace_stable.json` — Phase A trace
- `drift_trace_aware.json` — Phase B trace (drift-aware)
- `drift_trace_naive.json` — Phase C trace (drift-naive)
- `drift_summary.json` — machine-readable summary
- `talk_track.md` — 30s spoken pitch