"""SyntheticDriftBackend — parameterized drift injection for stress testing.

Unlike ReplayBackend (which replays real historical data), this backend
applies controllable drift functions to a baseline snapshot. Useful for:
  - Controlled experiments (exact same drift pattern across runs)
  - Stress testing the agent at specific drift severities
  - Ablation studies

DriftProfile defines how each parameter changes over time as a function
of hours elapsed. Presets: STABLE, LINEAR_DECAY, SUDDEN_DEGRADATION,
DIURNAL_CYCLE.

Usage:
    from backends.synthetic_drift import SyntheticDriftBackend, DriftProfile, LINEAR_DECAY

    backend = SyntheticDriftBackend("FakeBrisbane", LINEAR_DECAY)
    backend.set_time(5)  # advance to 5 hours
    health = backend.get_health()
    result = backend.run(circuit, shots=8192)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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
class DriftProfile:
    """Defines how device parameters drift over simulated time.

    Each function maps time_hours -> multiplicative factor:
      - T1/T2: factor < 1.0 means degraded (shorter coherence times)
      - Gate/readout errors: factor > 1.0 means worse (higher error rates)
      - Factor = 1.0 means no change from baseline

    Args:
        t1_drift: Global T1 drift function. Default: no drift.
        t2_drift: Global T2 drift. None = follows t1_drift.
        gate_error_drift: Gate error multiplier. None = inverse of t1_drift.
        readout_drift: Readout error multiplier. None = no change.
        per_qubit_t1: Per-qubit T1 overrides: {qubit: callable(t) -> factor}.
        per_qubit_t2: Per-qubit T2 overrides.
        drift_score_fn: Explicit drift score(t) -> [0,1]. None = auto-compute.
    """
    t1_drift: Callable[[float], float] = lambda t: 1.0
    t2_drift: Optional[Callable[[float], float]] = None
    gate_error_drift: Optional[Callable[[float], float]] = None
    readout_drift: Optional[Callable[[float], float]] = None

    per_qubit_t1: dict[int, Callable[[float], float]] = field(default_factory=dict)
    per_qubit_t2: dict[int, Callable[[float], float]] = field(default_factory=dict)

    drift_score_fn: Optional[Callable[[float], float]] = None


# ===== Preset Profiles =====

STABLE = DriftProfile(
    t1_drift=lambda t: 1.0,
    drift_score_fn=lambda t: 0.0,
)

LINEAR_DECAY = DriftProfile(
    t1_drift=lambda t: max(0.1, 1.0 - 0.03 * t),
    gate_error_drift=lambda t: 1.0 + 0.05 * t,
    readout_drift=lambda t: 1.0 + 0.02 * t,
    drift_score_fn=lambda t: min(1.0, 0.03 * t),
)

SUDDEN_DEGRADATION = DriftProfile(
    t1_drift=lambda t: 0.3 if t >= 5 else 1.0,
    gate_error_drift=lambda t: 5.0 if t >= 5 else 1.0,
    readout_drift=lambda t: 3.0 if t >= 5 else 1.0,
    drift_score_fn=lambda t: 1.0 if t >= 5 else 0.0,
)

DIURNAL_CYCLE = DriftProfile(
    t1_drift=lambda t: 1.0 - 0.15 * np.sin(2 * np.pi * t / 24),
    gate_error_drift=lambda t: 1.0 + 0.1 * np.sin(2 * np.pi * t / 24),
    drift_score_fn=lambda t: float(abs(np.sin(2 * np.pi * t / 24)) * 0.3),
)


# v2.5 Sprint A: smoother and more violent drift presets to test agent across
# a wider range of drift dynamics. MILD_GRADUAL is a slow ramp (closer to
# real diurnal-style drift), MULTI_SHOCK injects two sudden jumps so the
# agent has to react to a second event after the first recovery.

MILD_GRADUAL = DriftProfile(
    t1_drift=lambda t: max(0.6, 1.0 - 0.015 * t),
    gate_error_drift=lambda t: 1.0 + 0.025 * t,
    readout_drift=lambda t: 1.0 + 0.01 * t,
    drift_score_fn=lambda t: min(0.5, 0.015 * t),
)


def _multi_shock_t1(t: float) -> float:
    if t >= 8:
        return 0.25  # second, harsher shock
    if t >= 4:
        return 0.55
    return 1.0


def _multi_shock_gate(t: float) -> float:
    if t >= 8:
        return 6.0
    if t >= 4:
        return 2.5
    return 1.0


def _multi_shock_readout(t: float) -> float:
    if t >= 8:
        return 4.0
    if t >= 4:
        return 1.8
    return 1.0


def _multi_shock_score(t: float) -> float:
    if t >= 8:
        return 1.0
    if t >= 4:
        return 0.6
    return 0.0


MULTI_SHOCK = DriftProfile(
    t1_drift=_multi_shock_t1,
    gate_error_drift=_multi_shock_gate,
    readout_drift=_multi_shock_readout,
    drift_score_fn=_multi_shock_score,
)


class SyntheticDriftBackend(ShadowBackend):
    """Shadow backend with parameterized drift injection.

    Applies mathematical drift functions from a DriftProfile to a
    FakeBackend baseline. Use set_time(hours) to advance the simulation.

    The .profile property can be hot-swapped to change drift behavior
    mid-experiment without re-creating the backend.
    """

    def __init__(
        self,
        base_backend: str = "FakeBrisbane",
        profile: Optional[DriftProfile] = None,
    ):
        self._base_snap = extract_snapshot_from_fake_backend(base_backend)
        self._profile = profile or STABLE
        self._current_time: float = 0.0

    @property
    def name(self) -> str:
        return f"SyntheticDrift({self._base_snap.backend_name})"

    @property
    def num_qubits(self) -> int:
        return self._base_snap.num_qubits

    @property
    def profile(self) -> DriftProfile:
        return self._profile

    @profile.setter
    def profile(self, p: DriftProfile):
        self._profile = p

    def set_time(self, hours: float):
        """Set the simulated time in hours. Drift functions evaluate at this point."""
        self._current_time = max(0.0, hours)

    # --- Internal drift factor helpers ---

    def _get_t1_factor(self, qubit: int, t: float) -> float:
        if qubit in self._profile.per_qubit_t1:
            return self._profile.per_qubit_t1[qubit](t)
        return self._profile.t1_drift(t)

    def _get_t2_factor(self, qubit: int, t: float) -> float:
        if qubit in self._profile.per_qubit_t2:
            return self._profile.per_qubit_t2[qubit](t)
        if self._profile.t2_drift is not None:
            return self._profile.t2_drift(t)
        # Default: follows T1
        return self._get_t1_factor(qubit, t)

    def _get_gate_error_factor(self, t: float) -> float:
        if self._profile.gate_error_drift is not None:
            return self._profile.gate_error_drift(t)
        # Default: inverse of T1 drift (shorter T1 → higher gate error)
        t1f = self._profile.t1_drift(t)
        return 1.0 / max(0.01, t1f)

    def _get_readout_factor(self, t: float) -> float:
        if self._profile.readout_drift is not None:
            return self._profile.readout_drift(t)
        return 1.0

    def _get_current_snapshot(self) -> CalibrationSnapshot:
        """Apply drift to the base snapshot at the current time."""
        t = self._current_time
        snap = self._base_snap
        n = snap.num_qubits

        t1 = snap.t1_us.copy()
        t2 = snap.t2_us.copy()
        re = snap.readout_error.copy()
        ge1q = snap.gate_error_1q.copy()
        ge2q = dict(snap.gate_error_2q)

        gate_factor = self._get_gate_error_factor(t)
        readout_factor = self._get_readout_factor(t)

        for q in range(n):
            t1_factor = self._get_t1_factor(q, t)
            t2_factor = self._get_t2_factor(q, t)

            t1[q] = max(1.0, t1[q] * t1_factor)
            t2[q] = max(0.5, t2[q] * t2_factor)
            t2[q] = min(t2[q], 2 * t1[q])  # physical constraint T2 <= 2*T1

            ge1q[q] = float(np.clip(ge1q[q] * gate_factor, 1e-5, 0.1))
            re[q] = float(np.clip(re[q] * readout_factor, 0.001, 0.5))

        for edge in ge2q:
            ge2q[edge] = float(np.clip(ge2q[edge] * gate_factor, 1e-4, 0.5))

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

    def _compute_drift_score(self) -> float:
        """Compute drift score at current time."""
        if self._profile.drift_score_fn is not None:
            return float(self._profile.drift_score_fn(self._current_time))
        # Auto-compute: normalized deviation of global T1 factor from 1.0
        factor = self._profile.t1_drift(self._current_time)
        return float(np.clip(abs(1.0 - factor), 0.0, 1.0))

    # --- ShadowBackend interface ---

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
            calibration_age_minutes=self._current_time * 60.0,
            drift_score=self._compute_drift_score(),
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

        transpile_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in {"optimization_level", "initial_layout", "layout_method", "routing_method", "seed_transpiler"}
            and v is not None
        }
        transpiled = transpile(circuit, backend=sim, **transpile_kwargs)
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
                "sim_time_hours": self._current_time,
                "drift_score": self._compute_drift_score(),
                "transpiled_depth": transpiled.depth(),
            },
        )

    def get_properties_snapshot(self) -> dict[str, Any]:
        health = self.get_health()
        return {
            "backend": self.name,
            "timestamp": time.time(),
            "sim_time_hours": self._current_time,
            "num_qubits": self.num_qubits,
            "avg_1q_error": health.avg_1q_error,
            "avg_2q_error": health.avg_2q_error,
            "avg_readout_error": health.avg_readout_error,
            "avg_t1_us": health.avg_t1_us,
            "avg_t2_us": health.avg_t2_us,
            "drift_score": health.drift_score,
        }

    # --- Noise model builder ---

    def _build_simulator(self, snap: CalibrationSnapshot) -> AerSimulator:
        """Build AerSimulator from a drifted calibration snapshot."""
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

            re_val = snap.readout_error[q]
            if 0 < re_val < 0.5:
                noise.add_readout_error(
                    ReadoutError([[1 - re_val, re_val], [re_val, 1 - re_val]]),
                    [q],
                )

        for edge, err in snap.gate_error_2q.items():
            if 0 < err < 1.0:
                try:
                    noise.add_quantum_error(
                        depolarizing_error(err, 2), ["cx", "ecr"], list(edge)
                    )
                except Exception:
                    pass

        return AerSimulator(noise_model=noise)

    def _estimate_fidelity(self, original, noisy_counts, shots):
        """Estimate classical fidelity vs ideal (noiseless) simulation."""
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
            fid = sum(
                np.sqrt(ideal_counts.get(k, 0) / shots * noisy_counts.get(k, 0) / shots)
                for k in all_keys
            )
            return float(fid ** 2)
        except Exception:
            return None
