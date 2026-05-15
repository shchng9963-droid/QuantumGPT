"""ShadowBackend abstract base class.

All shadow hardware backends must implement this interface.
Designed to be protocol-compatible with Qiskit BackendV2 where possible,
so that switching to a real IBM backend later requires changing only
the backend provider, not the agent or tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from qiskit.circuit import QuantumCircuit


@dataclass
class BackendHealth:
    """Summary of a backend's current state."""
    name: str
    num_qubits: int
    avg_1q_error: float
    avg_2q_error: float
    avg_readout_error: float
    avg_t1_us: float
    avg_t2_us: float
    calibration_age_minutes: float
    drift_score: Optional[float] = None  # 0.0 = stable, 1.0 = severe drift


@dataclass
class QubitProperties:
    """Properties of a single qubit."""
    qubit: int
    t1_us: float
    t2_us: float
    readout_error: float
    gate_errors: dict[str, float] = field(default_factory=dict)  # gate_name -> error rate


@dataclass
class SimulationResult:
    """Result from running a circuit on a shadow backend."""
    counts: dict[str, int]
    shots: int
    fidelity: Optional[float] = None  # vs ideal, if computable
    metadata: dict[str, Any] = field(default_factory=dict)


class ShadowBackend(ABC):
    """Abstract base class for all shadow hardware backends.

    Three concrete implementations:
      - FakeBackendAdapter: wraps Qiskit FakeBackendV2
      - ReplayBackend: replays historical calibration data
      - SyntheticDriftBackend: injects parameterized drift

    The interface is intentionally minimal — it covers what the
    agent's perception and execution layers need.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name (e.g., 'FakeBrisbane')."""
        ...

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits on this backend."""
        ...

    @abstractmethod
    def get_health(self) -> BackendHealth:
        """Get a summary of the backend's current health.
        
        This is what the `get_backend_health` tool calls.
        """
        ...

    @abstractmethod
    def get_qubit_properties(self, qubits: list[int]) -> list[QubitProperties]:
        """Get detailed properties for specific qubits.
        
        This is what the `get_qubit_properties` tool calls.
        """
        ...

    @abstractmethod
    def get_coupling_map(self) -> list[tuple[int, int]]:
        """Return the coupling map as a list of (q_i, q_j) edges."""
        ...

    @abstractmethod
    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 8192,
        **kwargs,
    ) -> SimulationResult:
        """Run a circuit on this backend (transpile + simulate).
        
        The backend handles transpilation internally using its own
        coupling map and noise model.
        """
        ...

    @abstractmethod
    def get_properties_snapshot(self) -> dict[str, Any]:
        """Return the full calibration properties as a dict.
        
        Used by the properties_stream and drift detector.
        The format should match IBM's backend.properties() structure
        as closely as possible.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.name}, {self.num_qubits}q)>"
