# Day 9: StreamingAdvisor — Real-Time Telemetry + Alert Engine

## Goal
Connect PropertiesStream to a real-time alert engine so the advisor can
continuously monitor a quantum backend and fire threshold-based alerts
as conditions change — closing the loop from passive monitoring to active notification.

## What Was Built

### 1. StreamingAdvisor (`advisor/streaming.py`, ~400 lines)
Real-time advisory engine with 3 components:

| Component | Role |
|-----------|------|
| **HealthTracker** | Sliding window over recent snapshots: T1/drift/error slopes, jump detection |
| **Alert Engine** | 7 alert types, configurable thresholds, cooldown to prevent storms |
| **StreamingAdvisor** | Glues PropertiesStream → HealthTracker → Alert Engine → callbacks |

### 2. Alert System

| Alert Type | Level | Trigger |
|-----------|-------|---------|
| `DRIFT_SPIKE` | critical | Drift jumps by >0.15 between consecutive snapshots |
| `DRIFT_HIGH` | warn/crit | Drift exceeds 0.2 (warning) or 0.5 (critical) |
| `T1_DEGRADING` | warning | T1 slope < -5 μs/h over sliding window |
| `T1_LOW` | warn/crit | Avg T1 below 100μs (warning) or 50μs (critical) |
| `ERROR_HIGH` | warn/crit | 2Q error above 0.03 (warning) or 0.06 (critical) |
| `RECOVERY` | info | Device returns to nominal after a critical episode |

Features:
- **Cooldown**: configurable min interval between same alert type (prevents spam)
- **Callbacks**: register any function to receive `Alert` objects (print, log, Slack, etc.)
- **DuckDB persistence**: all snapshots + alerts logged to shared database
- **Two modes**: batch (synchronous) + streaming (background thread with time acceleration)

### 3. AlertConfig (all thresholds tunable)
```python
AlertConfig(
    drift_warning=0.2, drift_critical=0.5, drift_spike_threshold=0.15,
    t1_warning=100.0, t1_critical=50.0, t1_degradation_rate=-5.0,
    error_2q_warning=0.03, error_2q_critical=0.06,
    alert_cooldown_hours=1.0, window_size=5,
)
```

## Demo Results (4 Scenarios)

| Scenario | Mode | Snapshots | Alerts | Key Finding |
|----------|------|-----------|--------|-------------|
| Sudden Degradation | batch | 21 | 38 | Drift spike caught instantly at t=5h |
| Linear Decay | streaming | 26 | 18 | Gradual alerts: T1 → drift → error over 24h |
| Recovery | batch | 21 | 27 | Recovery alert fired at t=6.5h after degradation |
| Integration | batch→advisor | 13+4 | 17+4 | Seamless handoff: streaming feeds DuckDB, advisor reads it |

### Highlight: Streaming Mode
```
24 simulated hours completed in 3 real seconds (28800x acceleration)
26 snapshots emitted, 18 alerts:
  t=1.8h  ⚠️  T1 degrading at -7.01 μs/h
  t=5.7h  ⚠️  Drift 0.170 > 0.15 warning
  t=17.1h 🚨 Drift 0.512 > 0.5 critical
  t=19.9h ⚠️  T1 = 94.5μs < 100μs warning
  t=22.7h ⚠️  2Q error 0.031 > 0.03 warning
```

### Highlight: Recovery Detection
```
  t=0-2.5h   Device healthy (no alerts)
  t=3.0h     🚨 Drift spike +1.000, T1 drops to 70μs
  t=3-6h     Sustained critical alerts
  t=6.5h     ℹ️  RECOVERY: drift=0.000, T1=233.7μs, 2Q_err=0.016
```

## Unit Tests
21 tests covering all components:
```
TestHealthTracker       (5)  — sliding window, slopes, jump detection
TestBatchMode           (6)  — stable/sudden/linear, cooldown, recovery, T1 alert
TestStreamingMode       (5)  — start/stop, snapshot collection, callbacks
TestQueries             (4)  — summary, filter by level, last_n, DuckDB persistence
TestIntegration         (1)  — streaming → CalibrationAdvisor handoff
```

## Visualizations
- `alert_timeline.pdf/png` — Alert scatter plot over time (sudden degradation)
- `streaming_linear.pdf/png` — Streaming mode: alerts by level + cumulative alert curve
- `recovery_cycle.pdf/png` — Degradation/recovery zones with alert markers
- `alert_breakdown.pdf/png` — Alert type distribution (3 scenarios, horizontal bars)
- `streaming_architecture.pdf/png` — Architecture: Backend → Stream → Tracker → Alerts

## Files Changed
```
advisor/streaming.py            (new — ~400 lines, streaming advisor + alert engine)
tests/unit/test_streaming.py    (new — 21 tests)
demos/day9/run_day9_demo.py     (4-scenario demo)
demos/day9/gen_figures.py       (5 visualization figures)
demos/day9/results.json         (serialized results)
demos/day9/streaming.duckdb     (shared telemetry data)
demos/day9/integrated.duckdb    (integration demo data)
demos/day9/*.png, *.pdf         (5 figure pairs)
```

## Project Test Suite
**83 tests, all passing** (62 from Day 1-8 + 21 new)

## Architecture So Far

```
User / CLI
    │
    ▼
QuantumAgent (Day 6-7)           CalibrationAdvisor (Day 8)
    │  tool calls                     │  monitor → diagnose → act
    ▼                                 ▼
ToolExecutor ──────────────► StreamingAdvisor (Day 9)  ◄── NEW
    │                              │  │
    ▼                     on_snapshot  on_alert
ShadowBackend ◄────── PropertiesStream ───► HealthTracker
    │                              │              │
    ├── FakeBackendAdapter         │         sliding window
    ├── ReplayBackend              │         trend analysis
    └── SyntheticDriftBackend      │
                                   ▼
                              DuckDB (Day 7)
                              health / probes / alerts / sessions
```

## Next: Day 10
- Install Qiskit Dynamics, run Rabi oscillation tutorial
- First pulse-level simulation (Hamiltonian → pulse → IQ → Rabi curve)
- Lays groundwork for Phase 3 pulse experiment tools
