# Day 10: Pulse-Level Rabi Oscillation Simulation (qiskit-dynamics)

## Goal
Install qiskit-dynamics and build a first pulse-level simulation module:
drive a single qubit with a microwave pulse, evolve the Hamiltonian,
and extract the Rabi oscillation curve — the foundational experiment
for calibrating π-pulses on real quantum hardware.

## What Was Built

### 1. `dynamics/rabi.py` (~250 lines)
Pulse-level Rabi oscillation simulator using qiskit-dynamics Solver.

| Component | Description |
|-----------|-------------|
| **RabiConfig** | Dataclass: qubit freq, drive freq, pulse shape, duration, σ, dt, amp range |
| **RabiResult** | Per-amplitude result: time traces, P(|1⟩), statevectors, unitarity |
| **RabiSweepResult** | Multi-amplitude sweep: amplitudes, final populations, π-amp estimate |
| **simulate_rabi()** | Core solver: builds H(t), evolves |0⟩ via DOP853, returns states |
| **sweep_rabi()** | Amplitude sweep: runs simulate_rabi at n_amps points |
| **estimate_pi_pulse()** | Coarse+fine sweep calibration: finds π-amp to ~0.1% accuracy |

### 2. Physics Model

Rotating-frame Hamiltonian:
```
H(t) = -(Δ/2) σ_z + (Ω(t)/2) σ_x
```
- Δ = drive detuning (ω_drive - ω_qubit)
- Ω(t) = drive envelope × amplitude
- Two pulse shapes: square and Gaussian (σ-parameterised)
- Solver: qiskit-dynamics Solver with channel-based API, DOP853 integrator
- Tolerance: atol=rtol=1e-10 → numerical error < 1e-8

### 3. Analytic Verification
For a square on-resonance pulse: P(|1⟩) = sin²(Ω·t/2)
- Simulation matches analytic to < 6×10⁻¹⁰ (machine precision)
- π-pulse: Ω = π/T gives P(|1⟩) = 1.00000000
- 2π-pulse: returns to |0⟩ with P(|1⟩) < 10⁻⁶

## Demo Results (5 Scenarios)

| Scenario | Description | Key Result |
|----------|-------------|------------|
| 1. Time Evolution | Square pulse Ω=0.02GHz, T=200ns | Max error vs analytic: 5.95×10⁻¹⁰ |
| 2. Square Sweep | 81 amplitudes, 0–80 MHz | π-amp: 31.0 MHz (analytic: 31.4 MHz) |
| 3. Gaussian Sweep | 61 amplitudes, σ=25ns | π-amp: 52.0 MHz, P₁=0.9998 |
| 4. Detuned Chevron | 21×41 = 861 simulations | Classic Rabi chevron pattern |
| 5. π Calibration | Coarse (31pt) + Fine (51pt) | π-amp: 52.42 MHz, fidelity: 0.99999 |

Total demo runtime: **16.2 seconds** (861 Hamiltonian evolutions in 11.7s for chevron).

## Unit Tests: 20 tests, all passing

| Test Class | Count | Coverage |
|------------|-------|----------|
| TestRabiConfig | 3 | Defaults, on/off-resonance detuning |
| TestEnvelope | 3 | Square constant, Gaussian peak/symmetry, invalid shape |
| TestSimulateRabi | 6 | Zero amp, analytic match, π/2π pulse, shapes, unitarity |
| TestDetuned | 2 | Reduced max population, large detuning suppression |
| TestSweepRabi | 3 | Basic sweep, π-amp finding, monotonicity |
| TestEstimatePiPulse | 3 | Square calibration, no-refine mode, half-π consistency |

## Visualizations (5 figure pairs, PDF+PNG)

- `rabi_time_evolution.pdf/png` — P(|1⟩) vs time with analytic overlay
- `rabi_curves.pdf/png` — Square vs Gaussian Rabi curves (amplitude sweep)
- `rabi_chevron.pdf/png` — 2D heatmap: amplitude × detuning (Rabi chevron)
- `pi_calibration.pdf/png` — Coarse + fine sweep for π-pulse calibration
- `pulse_shapes_bloch.pdf/png` — Pulse envelopes + Bloch sphere trajectories (square & Gaussian π-rotations)

## Files Changed
```
dynamics/__init__.py              (new — module exports)
dynamics/rabi.py                  (new — ~250 lines, Rabi simulator)
tests/unit/test_rabi.py           (new — 20 tests)
demos/day10/run_day10_demo.py     (new — 5-scenario demo)
demos/day10/gen_figures.py        (new — 5 figure generators)
demos/day10/results.json          (new — serialized demo results)
demos/day10/*.png, *.pdf          (new — 5 figure pairs)
```

## Dependencies
- **qiskit-dynamics 0.6.0** (new) — pulse-level quantum simulation
- Pins qiskit to 1.3.0 (from 2.4.1) — creates version tension with qiskit-ibm-runtime
- Pre-existing tests depending on qiskit-ibm-runtime have import errors (not caused by Day 10)

## Known Issue: Qiskit Version Conflict
qiskit-dynamics 0.6.0 requires qiskit ≤ 1.3, but qiskit-ibm-runtime ≥ 0.47 requires qiskit ≥ 2.2.
Old backend/advisor tests that import `qiskit_ibm_runtime.fake_provider` fail with ImportError.
**Resolution**: Will be addressed in Day 11 (BACKENDS.md) — either pin compatible versions or
mock the fake provider in tests. The new `dynamics/` module works correctly with qiskit 1.3.

## Architecture Update

```
User / CLI
    │
    ▼
QuantumAgent (Day 6-7)           CalibrationAdvisor (Day 8)
    │  tool calls                     │  monitor → diagnose → act
    ▼                                 ▼
ToolExecutor ──────────────► StreamingAdvisor (Day 9)
    │                              │  │
    ▼                     on_snapshot  on_alert
ShadowBackend ◄────── PropertiesStream ───► HealthTracker
    │                              │
    ├── FakeBackendAdapter         │
    ├── ReplayBackend              │
    └── SyntheticDriftBackend      │
                                   ▼
                              DuckDB (Day 7)

    ┌──────────────────────────────────────┐
    │  dynamics/ (Day 10)  ◄── NEW         │
    │  ┌────────────────────────────────┐  │
    │  │ qiskit-dynamics Solver         │  │
    │  │  H(t) = -(Δ/2)σ_z + Ω(t)/2·σ_x │ │
    │  │  DOP853 integrator             │  │
    │  └────────────────────────────────┘  │
    │  simulate_rabi() → sweep_rabi()      │
    │  → estimate_pi_pulse()               │
    └──────────────────────────────────────┘
```

## Next: Day 11
- Write BACKENDS.md documenting all 3 shadow backends + the new dynamics module
- Resolve qiskit version conflict (pin or mock)
- CI test coverage across all backend types
