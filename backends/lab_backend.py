"""LabBackend — unified abstract interface for quantum hardware.

All backends (fake, simulated, real) implement this protocol.
The agent and experiment protocols interact only through these 5 primitives.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PulseSchedule:
    """A pulse schedule to dispatch to hardware."""

    channels: dict[str, list[dict]]  # channel_name -> list of pulse dicts
    duration_ns: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawResult:
    """Raw measurement result from hardware."""

    iq_data: np.ndarray | None = None  # shape (shots, n_qubits, 2) for I/Q
    counts: dict[str, int] | None = None
    waveform: np.ndarray | None = None  # raw ADC waveform
    timestamps: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceSnapshot:
    """Complete device state at a point in time."""

    backend_name: str
    timestamp: float = field(default_factory=time.time)
    num_qubits: int = 0
    qubit_frequencies_ghz: list[float] = field(default_factory=list)
    t1_us: list[float] = field(default_factory=list)
    t2_us: list[float] = field(default_factory=list)
    readout_errors: list[float] = field(default_factory=list)
    gate_errors_1q: list[float] = field(default_factory=list)
    gate_errors_2q: dict[str, float] = field(default_factory=dict)
    coupling_map: list[tuple[int, int]] = field(default_factory=list)
    calibration_age_minutes: float = 0.0
    drift_score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def avg_t1(self) -> float:
        return float(np.mean(self.t1_us)) if self.t1_us else 0.0

    def avg_t2(self) -> float:
        return float(np.mean(self.t2_us)) if self.t2_us else 0.0

    def avg_1q_error(self) -> float:
        return float(np.mean(self.gate_errors_1q)) if self.gate_errors_1q else 0.0

    def avg_2q_error(self) -> float:
        vals = list(self.gate_errors_2q.values())
        return float(np.mean(vals)) if vals else 0.0

    def avg_readout_error(self) -> float:
        return float(np.mean(self.readout_errors)) if self.readout_errors else 0.0


@dataclass
class CalibrationUpdate:
    """Parameters to write into the device calibration store."""

    qubit_index: int | None = None
    parameter: str = ""  # e.g. "pi_amplitude", "frequency", "readout_threshold"
    value: float = 0.0
    unit: str = ""
    source: str = "quantumgpt"  # who set this
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MeasurementSpec:
    """Specification for what to read from hardware."""

    mode: str = "iq"  # "iq", "counts", "waveform"
    qubits: list[int] = field(default_factory=list)
    shots: int = 1024
    integration_time_ns: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LabBackend(ABC):
    """Abstract base class for all quantum backends.

    Every backend -- fake, pulse-sim, or real hardware -- must implement
    these 5 primitives. The agent and experiment protocols never call
    anything else.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits available."""
        ...

    @property
    def device_type(self) -> str:
        """Device type: 'superconducting', 'ion_trap', 'neutral_atom', 'simulator'."""
        return "simulator"

    @property
    def supports_pulse(self) -> bool:
        """Whether this backend supports pulse-level control."""
        return False

    # --- Primitive 1: dispatch_pulse ---

    @abstractmethod
    def dispatch_pulse(self, schedule: PulseSchedule) -> RawResult:
        """Send a pulse schedule to hardware and return raw measurement."""
        ...

    # --- Primitive 2: run_circuit ---

    @abstractmethod
    def run_circuit(self, circuit: Any, shots: int = 4096, **kwargs: Any) -> RawResult:
        """Execute a quantum circuit and return measurement results."""
        ...

    # --- Primitive 3: read_measurement ---

    @abstractmethod
    def read_measurement(self, spec: MeasurementSpec) -> RawResult:
        """Read measurement data from hardware according to spec."""
        ...

    # --- Primitive 4: update_calibration ---

    @abstractmethod
    def update_calibration(self, update: CalibrationUpdate) -> bool:
        """Write a calibration parameter to the device.

        Returns True if successful, False otherwise.
        """
        ...

    # --- Primitive 5: get_device_state ---

    @abstractmethod
    def get_device_state(self) -> DeviceSnapshot:
        """Return a complete snapshot of current device state."""
        ...

    # --- Convenience methods ---

    def is_healthy(self, t1_threshold: float = 50.0) -> bool:
        """Quick health check based on T1."""
        state = self.get_device_state()
        return state.avg_t1() > t1_threshold

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, qubits={self.num_qubits})"
