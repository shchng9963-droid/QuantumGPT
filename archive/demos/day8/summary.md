# Day 8: CalibrationAdvisor — Autonomous Multi-Step Advisory Chain

## Goal
Build the first autonomous advisory workflow: a 3-phase chain that monitors
device health over time, diagnoses problems using temporal reasoning, and
generates concrete prioritized actions.

## What Was Built

### 1. CalibrationAdvisor (`advisor/advisor.py`)
Multi-step advisory engine with 3 phases:

| Phase | What it does | Techniques |
|-------|-------------|-----------|
| **MONITOR** | Collect health snapshots + run probe circuits at configurable time points | Backend health API, qubit properties, circuit execution |
| **DIAGNOSE** | Temporal analysis over DuckDB history | Linear regression (trend), changepoint detection (jump detector), qubit ranking (composite score) |
| **ACT** | Generate prioritized advisory actions | Rule engine with 7 rules, severity classification |

### 2. Six Advisory Action Types

| Action | When triggered | Priority |
|--------|---------------|----------|
| `RECALIBRATE` | Changepoint detected, drift > 0.5, or T1 degrading fast | P1 |
| `REMAP_LAYOUT` | Fidelity < 0.85, suggests better qubits | P1-P2 |
| `APPLY_MITIGATION` | Fidelity trend degrading (ZNE, readout correction) | P2 |
| `EXCLUDE_QUBITS` | Worst qubits have < 50% of best T1 | P2 |
| `WAIT` | Drift elevated (0.2-0.5) but not critical | P3 |
| `OK` | Device healthy, no action needed | P5 |

### 3. Temporal Analysis
- **Trend detection**: Linear regression on T1, T2, 2Q error, drift over time
- **Changepoint detection**: Normalized jump detector (threshold-based)
- **Qubit ranking**: Composite score (T1×0.4 + T2×0.3 + (1-readout)×0.3)
- **Fidelity trend**: Slope of probe circuit fidelity over time

### 4. Data Structures
```
AdvisoryReport
  ├── severity: NOMINAL | WARNING | CRITICAL
  ├── temporal_analysis: TemporalAnalysis
  │     ├── trend: STABLE | DEGRADING | IMPROVING | SUDDEN_CHANGE
  │     ├── t1_slope, t2_slope, error_slope, drift_slope
  │     ├── changepoints: [time_hours, ...]
  │     ├── worst_qubits, best_qubits
  │     └── fidelity_trend, fidelity_slope
  ├── actions: [AdvisoryAction(type, priority, reason, details)]
  ├── health_timeline: [{time, T1, T2, errors, drift}, ...]
  ├── probe_results: [{time, circuit, fidelity, depth}, ...]
  └── summary: str (human-readable report)
```

## Demo Results (4 Scenarios)

| Scenario | Severity | Trend | Actions | Top Action | Time |
|----------|----------|-------|---------|------------|------|
| STABLE | nominal | stable | 1 | OK | 4.8s |
| LINEAR_DECAY | critical | degrading | 4 | RECALIBRATE | 3.9s |
| SUDDEN_DEGRADATION | critical | sudden_change | 5 | RECALIBRATE | 3.8s |
| DIURNAL_CYCLE | critical | sudden_change | 1 | RECALIBRATE | 5.0s |

### Key Findings
- **STABLE**: Advisor correctly identifies healthy device, returns OK
- **LINEAR_DECAY**: Detects T1 degradation at -7.01 us/h, estimates 9h until unusable, recommends recalibration + remap
- **SUDDEN_DEGRADATION**: Catches changepoint at t=5h exactly, fidelity drops from 0.88→0.67, recommends recalibration + mitigation + remap
- **DIURNAL_CYCLE**: Detects oscillation (changepoints at every sample) — known limitation, needs FFT-based detection for periodic patterns

### GHZ-5 Fidelity Over Time (by scenario)
```
Stable:  0.888 → 0.881  (stable)
Linear:  0.873 → 0.837  (slow decline over 24h)
Sudden:  0.888 → 0.671  (cliff at t=5h, partial recovery)
Diurnal: 0.890 → 0.876  (oscillating)
```

## Unit Tests
25 tests covering all 3 phases + full cycle + temporal queries:
```
TestMonitor        (5 tests) — data collection, DuckDB persistence
TestDiagnose       (6 tests) — trend detection, changepoints, qubit ranking
TestAdvise         (5 tests) — action generation, severity, priority sorting
TestAdvisoryCycle  (6 tests) — end-to-end, summary, persistence
TestTemporalQueries(3 tests) — DuckDB health/fidelity/qubit queries
```

## Visualizations
- `advisory_timeline.png/pdf` — T1 + drift score over time (4 scenarios, dual axis)
- `fidelity_probes.png/pdf` — GHZ-5 fidelity degradation with warning/critical thresholds
- `action_matrix.png/pdf` — Heatmap of action types by scenario
- `changepoint_detection.png/pdf` — Drift score with CP markers (linear, sudden, diurnal)
- `qubit_ranking.png/pdf` — Worst vs best qubit ranking
- `advisor_architecture.png/pdf` — Monitor→Diagnose→Act architecture diagram

## Files Changed
```
advisor/__init__.py           (new)
advisor/advisor.py            (new — 670 lines, full advisory engine)
tests/unit/test_advisor.py    (new — 25 tests)
demos/day8/run_day8_demo.py   (4-scenario demo)
demos/day8/gen_figures.py     (6 visualization figures)
demos/day8/results.json       (serialized results)
demos/day8/advisor.duckdb     (persistent data)
demos/day8/*.png, *.pdf       (6 figure pairs)
```

## Known Limitations
1. **Diurnal cycle false positives**: Simple jump-based changepoint detection
   fires on every oscillation. Needs FFT or autocorrelation for periodic patterns.
2. **Static thresholds**: Warning/critical thresholds are hardcoded.
   Could be learned from historical data.
3. **No real recalibration**: `RECALIBRATE` action is advisory only —
   no actual calibration pulse sequence yet (Phase 3).

## Next: Day 9
- `properties_stream.py` — real-time telemetry stream with time acceleration
- Integration with PropertiesStream for continuous monitoring mode
- Advisory-in-the-loop: stream → advisor → action in real time
