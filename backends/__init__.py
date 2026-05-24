"""Backend implementations for QuantumGPT.

Provides three layers of backends:
- Gate-level: FakeBackendAdapter (Qiskit FakeBackendV2)
- Pulse-level: DynamicsLabAdapter (Hamiltonian simulation)
- Lab-level: FakeLabAdapter (LabBackend protocol over FakeBackend)

All backends implement the LabBackend abstract interface (5 primitives).
"""

from backends.base import BackendHealth, QubitProperties, ShadowBackend, SimulationResult
from backends.lab_backend import (
    CalibrationUpdate,
    DeviceSnapshot,
    LabBackend,
    MeasurementSpec,
    PulseSchedule,
    RawResult,
)
from backends.fake_adapter import FakeBackendAdapter
from backends.dynamics_lab_adapter import DynamicsLabAdapter
from backends.fake_lab_adapter import FakeLabAdapter

__all__ = [
    # Abstract
    "LabBackend",
    "ShadowBackend",
    # Data classes
    "BackendHealth",
    "CalibrationUpdate",
    "DeviceSnapshot",
    "MeasurementSpec",
    "PulseSchedule",
    "QubitProperties",
    "RawResult",
    "SimulationResult",
    # Implementations
    "DynamicsLabAdapter",
    "FakeBackendAdapter",
    "FakeLabAdapter",
]
