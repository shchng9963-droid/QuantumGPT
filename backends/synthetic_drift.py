"""SyntheticDriftBackend — parameterized drift injection for stress testing.

Unlike ReplayBackend (which replays real historical data), this backend
applies controllable drift functions to a baseline snapshot. Useful for:
  - Controlled experiments (exact same drift pattern across runs)
  - Stress testing the agent at specific drift severities
  - Ablation studies (what happens at 2x, 5x, 10x normal drift?)

Drift functions:
  - 'linear': steady degradation over time
  - 'sinusoidal': periodic oscillation (mimics temperature cycles)
  - 'step': sudden jump at a specified time (mimics qubit failure)
  - 'exponential_decay': accelerating degradation
  - 'composite': combination of the above

Usage:
    from backends.synthetic_drift import SyntheticDriftBackend, DriftConfig
    
    config = DriftConfig(
        pattern="step",
        affected_qubits=[3, 7, 12],
        severity=0.5,       # 50% degradation
        step_time=0.5,      # jump at halfway point
    )
    backend = SyntheticDriftBackend("FakeBrisbane", config, duration_seconds=3600)
    result = backend.run(circuit, shots=8192)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)

from backends.base import (
    BackendHealth,
    QubitProperties,
    ShadowBackend,
    SimulationResult,
)
from backends.calibration_data import CalibrationSnapshot, extract_snapshot_from_fake_backend


@dataclass
class DriftConfig:
    """Configuration for synthetic drift injection.
    
    Args:
        pattern: Type of drift: 'linear', 'sinusoidal', 'step', 'exponential_decay', 'composite'
        affected_qubits: Which qubits are affected (None = all)
        severity: Drift magnitude (0.0 = none, 1.0 = severe)
        step_time: For 'step' pattern, when the jump happens (0.0–1.0 fraction of duration)
        period_fraction: For 'sinusoidal', period as fraction of total duration
        parameters: Which parameters drift ('t1', 't2', 'gate_error', 'readout', 'all')
    """
    pattern: str = "linear"
    affected_qubits: Optional[list[int]] = None  # None = all qubits
    severity: float = 0.3
    step_time: float = 0.5
    period_fraction: float = 0.25
    parameters: str = "all"  # 't1', 't2', 'gate_error', 'readout', 'all'


class SyntheticDriftBackend(ShadowBackend):
    """Shadow backend with parameterized drift injection.
    
    Applies a mathematical drift function to a baseline FakeBackend snapshot.
    The drift progresses over a simulated duration, controlled by a speed factor.
    """

    def __init__(
        self,
        base_backend: str = "FakeBrisbane",
        config: Optional[DriftConfig] = None,
        duration_seconds: float = 3600.0,  # 1 hour simulated duration
        speed: float = 1.0,                # real-time by default
    ):
        self._base_snap = extract_snapshot_from_fake_backend(base_backend)
        self._config = config or DriftConfig()
        self._duration = duration_seconds
        self._speed = speed
        self._wall_start = time.time()
        
        # If affected_qubits is None, affect all
        if self._config.affected_qubits is None:
            self._affected = list(range(self._base_snap.num_qubits))
        else:
            self._affected = self._config.affected_qubits
        
        # Cache
        self._cached_progress: Optional[float] = None
        self._cached_sim: Optional[AerSimulator] = None

    @property
    def name(self) -> str:
        return f"SyntheticDrift({self._base_snap.backend_name}, {self._config.pattern})"

    @property
    def num_qubits(self) -> int:
        return self._base_snap.num_qubits

    @property
    def progress(self) -> float:
        """Current progress through the drift duration (0.0 to 1.0)."""
        elapsed = (time.time() - self._wall_start) * self._speed
        return min(1.0, elapsed / self._duration)

    def set_progress(self, p: float):
        """Manually set progress (0.0–1.0). Useful for controlled experiments."""
        p = max(0.0, min(1.0, p))
        # Adjust wall_start so that current time maps to desired progress
        self._wall_start = time.time() - (p * self._duration / self._speed)
        self._cached_progress = None

    def _drift_factor(self, progress: float) -> float:
        """Compute drift factor at given progress (0.0–1.0).
        
        Returns a factor in [0, severity] representing how much
        degradation to apply.
        """
        cfg = self._config
        s = cfg.severity
        
        if cfg.pattern == "linear":
            return s * progress
        
        elif cfg.pattern == "sinusoidal":
            period = cfg.period_fraction
            return s * 0.5 * (1 + np.sin(2 * np.pi * progress / period - np.pi/2))
        
        elif cfg.pattern == "step":
            return s if progress >= cfg.step_time else 0.0
        
        elif cfg.pattern == "exponential_decay":
            # Exponential: starts slow, accelerates
            return s * (np.exp(3 * progress) - 1) / (np.exp(3) - 1)
        
        elif cfg.pattern == "composite":
            # Linear baseline + sinusoidal oscillation + random jumps
            linear = 0.5 * s * progress
            osc = 0.3 * s * 0.5 * (1 + np.sin(2 * np.pi * progress / cfg.period_fraction))
            step_part = 0.2 * s if progress >= cfg.step_time else 0.0
            return linear + osc + step_part
        
        else:
            raise ValueError(f"Unknown drift pattern: {cfg.pattern}")

    def _get_current_snapshot(self) -> CalibrationSnapshot:
        """Apply drift to the base snapshot and return modified version."""
        p = self.progress
        factor = self._drift_factor(p)
        
        snap = self._base_snap
        n = snap.num_qubits
        cfg = self._config
        
        # Copy base values
        t1 = snap.t1_us.copy()
        t2 = snap.t2_us.copy()
        re = snap.readout_error.copy()
        ge1q = snap.gate_error_1q.copy()
        ge2q = dict(snap.gate_error_2q)
        
        for q in self._affected:
            if cfg.parameters in ("t1", "all"):
                t1[q] *= (1.0 - factor)  # T1 degrades
            if cfg.parameters in ("t2", "all"):
                t2[q] *= (1.0 - factor)
            if cfg.parameters in ("readout", "all"):
                re[q] = min(0.5, re[q] * (1.0 + factor * 3))  # readout worsens
            if cfg.parameters in ("gate_error", "all"):
                ge1q[q] = min(0.1, ge1q[q] * (1.0 + factor * 5))  # gate error worsens
        
        # 2Q errors: affect edges connected to affected qubits
        if cfg.parameters in ("gate_error", "all"):
            for edge in ge2q:
                if edge[0] in self._affected or edge[1] in self._affected:
                    ge2q[edge] = min(0.5, ge2q[edge] * (1.0 + factor * 3))
        
        # Enforce T2 <= 2*T1
        for q in range(n):
            t2[q] = min(t2[q], 2 * t1[q])
        
        return CalibrationSnapshot(
            timestamp=time.time(),
            backend_name=snap.backend_name,
            num_qubits=n,
            t1_us=t1,
            t2_us=t2,
            readout_error=re,
            gate_error_1q=ge1q,
            gate_error_2q=ge2q,
            coupling_map=snap.coupling_map.copy(),
        )

    def get_health(self) -> BackendHealth:
        snap = self._get_current_snapshot()
        return BackendHealth(
            name=self.name,
            num_qubits=snap.num_qubits,
            avg_1q_error=float(np.mean(snap.gate_error_1q)),
            avg_2q_error=float(np.mean(list(snap.gate_error_2q.values()))) if snap.gate_error_2q else 0.0,
            avg_readout_error=float(np.mean(snap.readout_error)),
            avg_t1_us=float(np.mean(snap.t1_us)),
            avg_t2_us=float(np.mean(snap.t2_us)),
            calibration_age_minutes=(time.time() - self._wall_start) / 60.0,
            drift_score=self._drift_factor(self.progress),
        )

    def get_qubit_properties(self, qubits: list[int]) -> list[QubitProperties]:
        snap = self._get_current_snapshot()
        return [
            QubitProperties(
                qubit=q,
                t1_us=snap.t1_us[q],
                t2_us=snap.t2_us[q],
                readout_error=snap.readout_error[q],
                gate_errors={"1q": snap.gate_error_1q[q]},
            )
            for q in qubits
        ]

    def get_coupling_map(self) -> list[tuple[int, int]]:
        return self._base_snap.coupling_map

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 8192,
        **kwargs,
    ) -> SimulationResult:
        snap = self._get_current_snapshot()
        sim = self._build_simulator(snap)
        
        transpiled = transpile(circuit, backend=sim)
        job = sim.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        fidelity = self._estimate_fidelity(circuit, counts, shots)
        
        return SimulationResult(
            counts=counts,
            shots=shots,
            fidelity=fidelity,
            metadata={
                "backend": self.name,
                "drift_progress": self.progress,
                "drift_factor": self._drift_factor(self.progress),
                "transpiled_depth": transpiled.depth(),
            },
        )

    def get_properties_snapshot(self) -> dict[str, Any]:
        return self._get_current_snapshot().to_dict()

    def _build_simulator(self, snap: CalibrationSnapshot) -> AerSimulator:
        """Build AerSimulator from snapshot (same logic as ReplayBackend)."""
        noise = NoiseModel()
        
        for q in range(snap.num_qubits):
            t1 = snap.t1_us[q] * 1e-6
            t2 = snap.t2_us[q] * 1e-6
            gate_time = 35e-9
            
            if t1 > 0 and t2 > 0:
                try:
                    thermal_err = thermal_relaxation_error(t1, t2, gate_time)
                    noise.add_quantum_error(thermal_err, ["sx", "x"], [q])
                except Exception:
                    pass
            
            re = snap.readout_error[q]
            if 0 < re < 0.5:
                noise.add_readout_error(ReadoutError([[1-re, re], [re, 1-re]]), [q])
        
        for edge, err in snap.gate_error_2q.items():
            if 0 < err < 1.0:
                try:
                    noise.add_quantum_error(depolarizing_error(err, 2), ["cx", "ecr"], list(edge))
                except Exception:
                    pass
        
        return AerSimulator(noise_model=noise)

    def _estimate_fidelity(self, original, noisy_counts, shots):
        n = original.num_qubits
        if n > 20:
            return None
        try:
            ideal_sim = AerSimulator(method="statevector")
            meas_circ = original.copy()
            if meas_circ.count_ops().get("measure", 0) == 0:
                meas_circ.measure_all()
            ideal_counts = ideal_sim.run(meas_circ, shots=shots).result().get_counts()
            all_keys = set(ideal_counts.keys()) | set(noisy_counts.keys())
            fid = sum(np.sqrt(ideal_counts.get(k, 0)/shots * noisy_counts.get(k, 0)/shots) for k in all_keys)
            return float(fid ** 2)
        except Exception:
            return None
