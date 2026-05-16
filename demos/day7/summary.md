# Day 7: W&B + DuckDB — Experiment Data Pipeline

## Goal
Wire up persistent storage (DuckDB) and experiment tracking (W&B) so every
agent tool call is automatically persisted and queryable.

## What Was Built

### 1. DuckDB Schema (`data/store.py`)
5 tables covering the full experiment lifecycle:

| Table | Columns | Purpose |
|-------|---------|---------|
| `health_snapshots` | backend, T1/T2, errors, drift, calibration age | Device health over time |
| `qubit_properties` | per-qubit T1/T2/readout/gate errors | Qubit-level tracking |
| `circuit_runs` | circuit, fidelity, depth, gates, counts | Circuit execution results |
| `agent_sessions` | model, prompt, tokens, time, tool calls | Agent run metadata |
| `drift_events` | severity, drift score, suggestions | Diagnosis events |

### 2. Instrumented Executor (`data/instrumented.py`)
Wraps the ToolExecutor and auto-persists every tool call result:
- `get_backend_health` → `health_snapshots` table + W&B health/ metrics
- `get_qubit_properties` → `qubit_properties` table
- `run_circuit` → `circuit_runs` table + W&B circuit/ metrics
- `diagnose_and_suggest` → `drift_events` table + W&B diagnosis/ metrics

### 3. Query API (`data/store.py :: DataStore`)
SQL helpers for common queries:
- `recent_runs()` — latest circuit executions
- `fidelity_trend()` — fidelity over time, filterable by circuit
- `health_history()` — backend health snapshots
- `best_qubits()` / `worst_qubits()` — qubit ranking
- `session_summary()` — agent run overview
- `table_counts()` — quick data overview
- `query(sql)` — arbitrary SQL

### 4. W&B Integration
- Offline mode (no login required) — data saved to `demos/day7/wandb/`
- Logs health metrics, circuit fidelity, diagnosis severity per tool call step
- Run summary shows sparkline charts for all tracked metrics
- Can sync to cloud with `wandb sync <run-dir>` when ready

### 5. Agent Integration
- `QuantumAgent` now accepts `db_path` and `wandb_run` parameters
- When `db_path` is set, uses InstrumentedExecutor automatically
- Every `agent.run()` also logs a session record with prompt, answer, tokens, timing

## End-to-End Demo Results (5 tasks via DeepSeek V4-Flash)

| # | Task | Backend | Tool Calls | Tokens | Time | Key Finding |
|---|------|---------|-----------|--------|------|-------------|
| 1 | Health check | FakeBrisbane | 1 | 2,487 | 7.4s | Healthy, fresh calibration |
| 2 | GHZ-5 + qubits | FakeBrisbane | 4 | 7,302 | 12.8s | Fidelity 0.9224, Q3 best T1 |
| 3 | DJ-5 + QFT-4 | FakeBrisbane | 4 | 4,797 | 10.9s | DJ-5: 0.911, QFT-4: 0.999 |
| 4 | Degraded diag | SyntheticDrift(t=8h) | 5 | 16,050 | 18.6s | Drift=1.0, fidelity 0.687 |
| 5 | Slow drift | LinearDecay(t=12h) | 6 | 11,882 | 15.8s | Drift=0.36, still usable |

**Totals: 20 tool calls, 42,518 tokens, 65.5s, 65 DuckDB rows**

## DuckDB Verification

```
Table Counts:
  health_snapshots:  5 rows
  qubit_properties: 46 rows
  circuit_runs:      6 rows
  agent_sessions:    5 rows
  drift_events:      3 rows
  Total:            65 rows

Sample query: SELECT circuit_name, fidelity, backend FROM circuit_runs ORDER BY fidelity
  ghz_5     0.6865  SyntheticDrift(FakeBrisbane)
  ghz_5     0.8625  SyntheticDrift(FakeBrisbane)
  DJ-5      0.9111  FakeBrisbane
  GHZ-5     0.9224  FakeBrisbane
  QFTent-5  0.9489  SyntheticDrift(FakeBrisbane)
  qft_4     0.9987  FakeBrisbane
```

## Visualizations
- `fidelity_by_circuit.png/pdf` — All 6 circuit runs, color-coded by backend
- `health_over_time.png/pdf` — T1, T2, 2Q error, drift across 5 snapshots
- `session_stats.png/pdf` — Tool calls, tokens, time per agent session
- `drift_events.png/pdf` — Severity timeline (nominal → critical → warning)
- `data_pipeline.png/pdf` — Full architecture: Agent → Executor → DuckDB + W&B

## Files Changed
```
data/__init__.py             (new)
data/store.py                (new — 250 lines, DuckDB schema + query API)
data/instrumented.py         (new — 120 lines, auto-logging wrapper)
agent/loop.py                (updated — db_path/wandb_run params, session logging)
demos/day7/run_day7_demo.py  (5-task demo with DuckDB + W&B)
demos/day7/gen_figures.py    (5 visualization figures)
demos/day7/quantumgpt.duckdb (65 rows of experiment data)
demos/day7/wandb/            (W&B offline run)
demos/day7/*.png, *.pdf      (5 figure pairs)
```

## Next: Day 8 (Week 2)
- Multi-step agent chains (monitor → decide → act)
- Temporal reasoning over DuckDB history
- First "autonomous calibration advisor" workflow
