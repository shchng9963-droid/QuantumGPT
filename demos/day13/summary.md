# Day 13: ExperimentRecord Schema — Structured Experiment Memory

## Goal
Design and implement ExperimentRecord (DuckDB table + Python dataclass) as
the agent's structured experiment memory. This replaces simple (circuit, fidelity)
episodic memory with a rich record that captures the full experiment context.

## What Was Done

### 1. ExperimentRecord Dataclass (data/experiment_record.py, ~400 lines)

27-field structured record capturing the complete experiment lifecycle:

| Field Group | Fields | Purpose |
|-------------|--------|---------|
| Identity | id, timestamp, experiment_type | Unique ID + typing |
| Context | backend, circuit_name, num_qubits, problem, tags | What was run |
| Plan | plan (list of steps) | Intended approach |
| Execution | tool_calls, decisions, elapsed_seconds, model, total_tokens | What actually happened |
| Results | fidelity, success, outcome, metrics, raw_data_ref | Quantitative results |
| Fitting | fits (dict) | For Rabi/calibration: pi_amp, freq, R² |
| Backend Snapshot | backend_snapshot (dict) | Device state at experiment time |
| Reflection | summary, lessons, failure_reason | Agent's self-assessment |

Experiment types: circuit_benchmark, health_check, drift_diagnosis,
rabi_tuneup, error_mitigation, multi_step_task, custom.

Outcomes: success (fidelity >= 0.9), partial (>= 0.7), failure (< 0.7), error.

### 2. ExperimentStore — Query API

| Method | Description |
|--------|-------------|
| save() / save_batch() | Persist records |
| get(id) | Retrieve by ID |
| recent(limit) | Most recent experiments |
| by_circuit(name) | Filter by circuit |
| by_backend(name) | Filter by backend |
| by_type(type) | Filter by experiment type |
| search(...) | Multi-filter: circuit, backend, type, fidelity range, success, date |
| stats() | Aggregate: total, avg fidelity, success rate |
| stats_by_circuit() | Per-circuit: runs, avg/min/max fidelity |
| stats_by_backend() | Per-backend: runs, avg fidelity, circuits |
| fidelity_trend() | Time series for a circuit |
| count() | Total records |

DuckDB indexes on: backend, circuit_name, experiment_type, fidelity, timestamp, success.

### 3. Integration

- **DataStore**: `db.experiments` sub-store available on every DataStore instance
- **InstrumentedExecutor**: auto-creates ExperimentRecords for every `run_circuit`
  and `diagnose_and_suggest` call, including backend health snapshot
- **CLI**: `qgpt records` command with filters, stats, and JSON output

### 4. CLI: `qgpt records`

```
qgpt records                       # recent experiments
qgpt records -c ghz_5              # filter by circuit
qgpt records -b FakeBrisbane       # filter by backend
qgpt records --stats               # aggregate statistics
qgpt records --stats -j            # JSON stats output
```

### 5. Tests: 19 new tests, all passing

| Test class | Count | Coverage |
|------------|-------|----------|
| TestExperimentRecord | 4 | Dataclass, serialization, roundtrip |
| TestExperimentStore | 13 | CRUD, search, stats, trends |
| TestDataStoreIntegration | 2 | Schema, sub-store access |

### 6. Full Test Suite: 129 passed, 1 skipped, 0 failures

### 7. Demo: End-to-end (10 circuits × 2 backends + diagnosis)

| Metric | Value |
|--------|-------|
| Total records created | 11 |
| Avg fidelity | 0.9637 |
| Success rate | 100% (11/11) |
| Best circuit | qft_4 (0.9967) |
| Best backend | FakeKyiv (0.9701 avg) |

## Files Created/Modified

| File | Action | Lines |
|------|--------|-------|
| `data/experiment_record.py` | Created | ~400 |
| `data/store.py` | Modified | +6 (import + schema init) |
| `data/instrumented.py` | Modified | +80 (auto-record creation) |
| `data/__init__.py` | Updated | exports |
| `cli.py` | Modified | +170 (records command) |
| `tests/unit/test_experiment_record.py` | Created | ~240 |
| `demos/day13/gen_figures.py` | Created | ~190 |
| `demos/day13/experiment_fidelity_scatter.png` | Generated | — |
| `demos/day13/experiment_stats_summary.png` | Generated | — |
| `demos/day13/demo.duckdb` | Generated | — |

## Design Decisions

1. **Immutable records**: once written, never modified — append-only log
2. **summary field**: one-line human-readable text, designed as RAG embedding target for Phase 3
3. **backend_snapshot**: captures device state at experiment time for drift analysis
4. **fits dict**: extensible for Rabi (pi_amp, rabi_freq) and future calibration experiments
5. **lessons list**: agent can store what it learned, supports future self-improvement
6. **DuckDB indexes**: 6 indexes for fast filtering on common query patterns

## Phase 0 Status: COMPLETE

All Day 1–13 tasks done. Day 14 = buffer/catch-up + collaboration emails.
Phase 0 deliverables:
  [x] BACKENDS.md
  [x] CI tests covering 3 shadow backends (110→129 tests)
  [x] MVP CLI demonstrable (6 commands)
  [x] ExperimentRecord schema defined + functional
  [x] Qiskit Dynamics environment ready (Rabi tutorial passed)
  [x] Physical collaboration emails: pending (Day 14)
