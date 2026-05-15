"""Quantum GPT - Backends Module

Shadow hardware backends that mimic real quantum devices.
Three concrete implementations:
  - FakeBackendAdapter: wraps Qiskit FakeBackendV2 (static snapshots)
  - ReplayBackend: replays historical IBM calibration data over time
  - SyntheticDriftBackend: injects parameterized T1/T2/gate-error drift
"""

from backends.base import ShadowBackend, BackendHealth, QubitProperties, SimulationResult
from backends.fake_adapter import FakeBackendAdapter
from backends.replay import ReplayBackend
from backends.synthetic_drift import SyntheticDriftBackend, DriftConfig
from backends.calibration_data import (
    CalibrationSnapshot,
    extract_snapshot_from_fake_backend,
    generate_drift_series,
)

__all__ = [
    "ShadowBackend",
    "BackendHealth",
    "QubitProperties",
    "SimulationResult",
    "FakeBackendAdapter",
    "ReplayBackend",
    "SyntheticDriftBackend",
    "DriftConfig",
    "CalibrationSnapshot",
    "extract_snapshot_from_fake_backend",
    "generate_drift_series",
]
