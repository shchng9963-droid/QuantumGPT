# Day 14: LabBackend + Experiment Protocols + Web Frontend

## Goal
Upgrade from gate-level simulation to pulse-level hardware characterization
with unified LabBackend interface, standard experiment protocols, and web dashboard.

## What Was Done

### 1. LabBackend Abstract Interface (backends/lab_backend.py, 155 lines)
5-primitive protocol: dispatch_pulse, run_circuit, read_measurement, update_calibration, get_device_state.
Supporting data classes: PulseSchedule, RawResult, DeviceSnapshot, CalibrationUpdate, MeasurementSpec.

### 2. DynamicsLabAdapter (backends/dynamics_lab_adapter.py, 195 lines)
Pulse-level backend: single-qubit transmon Hamiltonian, square/gaussian/DRAG pulses,
T1 decoherence, readout error, optimized zero-segment skipping.

### 3. FakeLabAdapter (backends/fake_lab_adapter.py, 140 lines)
Bridges FakeBackendAdapter to LabBackend protocol for gate-level execution.

### 4. Experiment Protocols (experiments/, 4 files)
- RabiExperiment: pi-pulse calibration, cos fit
- T1Experiment: energy relaxation, exponential fit
- RamseyExperiment: T2* dephasing, damped oscillation fit
Each: run() -> analyze() -> visualize()

### 5. FastAPI Backend (api/main.py, 130 lines)
REST: /api/health, /api/backends, /api/experiments/run, /api/experiments/history
WebSocket: /ws/telemetry

### 6. Streamlit Dashboard (web/app.py, 305 lines)
4 pages: Dashboard, Experiment Lab, Device Monitor, History.
Interactive Plotly charts, real-time metrics.

### 7. Tests: 14 new, all passing (34 total with rabi tests)

## Demo Results
| Experiment | Measured | True | R-squared |
|-----------|----------|------|-----------|
| Rabi pi-amp | 5.03 mV | 5.0 mV | 0.999 |
| T1 | 199.6 us | 200 us | 0.989 |
| Ramsey detuning | 2.00 MHz | 2.0 MHz | 0.960 |

## Files Created: 14 new files, ~1700 lines of code
