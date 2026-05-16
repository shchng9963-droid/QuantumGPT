# BACKENDS.md — QuantumGPT Backend Architecture

> Last updated: Day 11 (Phase 2, Week 2)

## Overview

QuantumGPT uses a **layered backend architecture** that separates circuit-level
simulation (noisy fake backends) from pulse-level dynamics (Hamiltonian evolution).
All backends implement a common interface so the agent, advisor, and CLI can swap
them transparently.

```
                        ┌─────────────────┐
                        │   User / CLI    │
                        └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼             ▼
            QuantumAgent   CalibrationAdvisor  StreamingAdvisor
                    │            │             │
                    └────────────┼─────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼             ▼
            ShadowBackend   PropertiesStream  dynamics/
             (circuit)       (telemetry)    (pulse-level)
```

---

## 1. Shadow Backends (Circuit-Level)

Shadow backends wrap Qiskit fake backends (IBM hardware models) to provide
realistic noisy simulation without IBM Cloud access. They all implement the
`ShadowBackend` interface defined in `backends/base.py`.

### Common Interface

```python
class ShadowBackend(ABC):
    def get_health(self) -> dict           # T1, T2, error rates, drift score
    def get_qubit_properties(self) -> list  # per-qubit T1, T2, freq, error
    def get_coupling_map(self) -> list      # qubit connectivity edges
    def properties_snapshot(self) -> CalibrationSnapshot  # full snapshot
    def run(circuit, shots) -> SimulationResult  # execute circuit
```

### 1.1 FakeBackendAdapter

**File:** `backends/fake_adapter.py`
**Purpose:** Thin wrapper around `qiskit_ibm_runtime.fake_provider` backends.

| Feature | Detail |
|---------|--------|
| Supported backends | FakeBrisbane (127q), FakeKyiv (127q), FakeSherbrooke (127q), FakeTorino (133q) |
| Noise model | Extracted from backend's calibration data (T1, T2, gate errors) |
| Calibration | Static (real hardware snapshot, no time evolution) |
| Circuit execution | Via `qiskit_aer.AerSimulator` with noise model |

```python
from backends import FakeBackendAdapter

backend = FakeBackendAdapter("FakeBrisbane")
health = backend.get_health()
# → {'n_qubits': 127, 'avg_t1_us': 234.5, 'avg_t2_us': 178.2,
#    'avg_1q_error': 0.00023, 'avg_2q_error': 0.0089, 'drift_score': 0.0}
```

### 1.2 ReplayBackend

**File:** `backends/replay_backend.py`
**Purpose:** Time-travel through a synthetic calibration history. Lets you
simulate how a backend's noise properties change over hours/days.

| Feature | Detail |
|---------|--------|
| Time model | Synthetic drift series from a base snapshot |
| Drift | Linear interpolation between time-stamped snapshots |
| Seek | `set_time(hours)` to jump to any point in the history |
| Noise rebuild | Rebuilds AerSimulator noise model at each time step |

```python
from backends import ReplayBackend

replay = ReplayBackend.from_synthetic_history("FakeBrisbane", hours=24, snapshots=12)
replay.set_time(6.0)   # 6 hours in
health = replay.get_health()
result = replay.run(circuit, shots=8192)

replay.set_time(18.0)  # 18 hours in — more drift
health2 = replay.get_health()
```

### 1.3 SyntheticDriftBackend

**File:** `backends/synthetic_drift.py`
**Purpose:** Programmatic drift profiles (stable, linear decay, sudden degradation,
custom lambda functions). Used by the advisor and streaming systems.

| Feature | Detail |
|---------|--------|
| Profiles | `STABLE`, `LINEAR_DECAY`, `SUDDEN_DEGRADATION`, custom `DriftProfile` |
| Drift model | Lambda functions for T1, T2, error rate, drift score over time |
| Per-qubit | Optional per-qubit overrides for targeted degradation |
| Hot-swap | Change profile at runtime with `set_profile()` |

```python
from backends import SyntheticDriftBackend
from backends.synthetic_drift import STABLE, LINEAR_DECAY, DriftProfile

# Built-in profiles
backend = SyntheticDriftBackend("FakeBrisbane", LINEAR_DECAY)
backend.set_time(12.0)
health = backend.get_health()  # degraded after 12 hours

# Custom profile
custom = DriftProfile(
    t1_drift=lambda t: max(0.3, 1.0 - 0.05*t),
    drift_score_fn=lambda t: min(1.0, 0.02*t),
)
backend = SyntheticDriftBackend("FakeBrisbane", custom)
```

---

## 2. Telemetry Layer

### 2.1 PropertiesStream

**File:** `backends/properties_stream.py`
**Purpose:** Generates a time-series of `CalibrationSnapshot` objects from any
backend. Supports replay (batch) and real-time (streaming) modes.

| Mode | Description |
|------|-------------|
| `replay_all(start, end, interval)` | Generate all snapshots synchronously |
| `stream(interval, accel)` | Background thread emitting snapshots in real-time |

```python
from backends import PropertiesStream

stream = PropertiesStream(backend)
# Batch mode
snapshots = stream.replay_all(start_hours=0, end_hours=24, interval_hours=1)

# Real-time mode (background thread)
stream.start(interval_hours=1, time_acceleration=3600)  # 1h → 1s
stream.on_snapshot(callback)
stream.stop()
```

### 2.2 CalibrationSnapshot

**File:** `backends/calibration_data.py`
**Dataclass fields:**
- `timestamp`: float (hours since epoch)
- `qubit_t1_us`: list[float] — T1 per qubit (microseconds)
- `qubit_t2_us`: list[float] — T2 per qubit
- `qubit_freq_ghz`: list[float] — qubit frequencies
- `qubit_1q_error`: list[float] — single-qubit gate errors
- `coupling_2q_error`: dict[tuple, float] — two-qubit gate errors
- `drift_score`: float — aggregate drift metric (0=pristine, 1=severe)

---

## 3. Pulse-Level Dynamics

### 3.1 Rabi Oscillation Simulator

**File:** `dynamics/rabi.py`
**Engine:** `qiskit-dynamics` 0.6 (Solver with DOP853 integrator)
**Purpose:** Simulate pulse-level qubit control — the physics layer beneath
circuit-level gates.

| Component | Description |
|-----------|-------------|
| `simulate_rabi(amp, config)` | Evolve \|0⟩ under H(t), return P(\|1⟩) time trace |
| `sweep_rabi(config)` | Amplitude sweep → Rabi oscillation curve |
| `estimate_pi_pulse(config)` | Coarse+fine calibration → π-pulse amplitude |

**Physics model** (rotating frame):
```
H(t) = -(Δ/2) σ_z + (Ω(t)/2) σ_x
```
- Δ = drive detuning (GHz)
- Ω(t) = drive envelope × amplitude
- Pulse shapes: square, Gaussian (σ-parameterised)

**Performance:**
- Single simulation: ~10ms (100ns pulse, 0.5ns steps)
- 81-point amplitude sweep: ~1s
- 861-point chevron (21 detunings × 41 amps): ~12s
- Numerical accuracy: matches analytic sin²(Ωt/2) to < 10⁻⁹

```python
from dynamics import simulate_rabi, sweep_rabi, estimate_pi_pulse, RabiConfig

# Single time evolution
config = RabiConfig(pulse_duration_ns=100, pulse_shape="gaussian", sigma_ns=25)
result = simulate_rabi(0.05, config)
print(result.final_p1)  # P(|1⟩) at end of pulse

# Find π-pulse
cal = estimate_pi_pulse(config, refine=True)
print(f"π-amp: {cal['pi_amp']:.4f} GHz, fidelity: {cal['pi_fidelity']:.6f}")
```

---

## 4. Data Persistence

### 4.1 DuckDB Store

**File:** `data/store.py`
**Tables:**
- `health_snapshots` — time-series of backend health metrics
- `probe_results` — circuit probe fidelity measurements
- `alerts` — StreamingAdvisor alert history
- `sessions` — agent session metadata

All backends, advisors, and streaming components write to a shared DuckDB
database, enabling cross-component temporal queries.

---

## 5. Dependency Matrix

| Package | Version | Used By | Notes |
|---------|---------|---------|-------|
| `qiskit` | 1.3.0 | All circuit backends | Pinned by qiskit-dynamics |
| `qiskit-dynamics` | 0.6.0 | `dynamics/` | Requires qiskit ≤ 1.3 |
| `qiskit-ibm-runtime` | 0.29.1 | FakeBackendAdapter, SyntheticDrift | Compatible with qiskit 1.3 |
| `qiskit-aer` | 0.17.2 | Circuit execution (noisy sim) | |
| `duckdb` | 1.5+ | DataStore | Persistence layer |
| `numpy`, `scipy` | latest | Everywhere | Core numerics |
| `matplotlib` | latest | Figures | Visualisation |
| `mqt-bench` | optional | `bench/mqtbench.py` | Benchmark circuits (skipped if absent) |

### Version Pinning Rationale
qiskit-dynamics 0.6.0 caps qiskit at ≤ 1.3. The latest qiskit-ibm-runtime (0.47)
requires qiskit ≥ 2.2. We pin `qiskit-ibm-runtime==0.29.1` which is the last
version compatible with qiskit 1.3. When qiskit-dynamics releases a 2.x-compatible
version, all constraints can be relaxed.

---

## 6. Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| `test_backends.py` | 22 | FakeAdapter, Replay, Synthetic, PropertiesStream, interface compliance |
| `test_replay_backend.py` | 5 | Snapshot extraction, drift series, health, run, seek |
| `test_advisor.py` | 25 | Monitor, diagnose, advise, full cycle, temporal queries |
| `test_streaming.py` | 21 | HealthTracker, batch mode, streaming mode, queries, integration |
| `test_rabi.py` | 20 | Config, envelope, simulate, detuned, sweep, π-calibration |
| `test_mqtbench.py` | 7 | Skipped if mqt-bench not installed |
| **Total** | **100** | **96 pass, 1 skip (mqtbench)** |

---

## 7. Architecture Roadmap

```
Phase 2 (current):
  ✅ Day 1-5:  Shadow backends, drift simulation, data persistence
  ✅ Day 6-7:  Agent loop + tool executor
  ✅ Day 8:    CalibrationAdvisor (monitor → diagnose → act)
  ✅ Day 9:    StreamingAdvisor (real-time alerts)
  ✅ Day 10:   Pulse-level dynamics (Rabi oscillation)
  ✅ Day 11:   BACKENDS.md + CI green (this document)
  ⬜ Day 12:   MVP CLI
  ⬜ Day 13:   ExperimentRecord schema
  ⬜ Day 14:   Phase 2 wrap-up

Phase 3 (next):
  ⬜ Pulse experiment tools (T1/T2/Ramsey via dynamics)
  ⬜ Noise-aware transpilation advisor
  ⬜ Multi-backend comparison workflows
  ⬜ Real IBM Quantum integration (when API access available)
```
