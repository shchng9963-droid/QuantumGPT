"""ReplayBackend — replays historical calibration data over time.

Simulates real IBM device drift by interpolating between a sequence of
calibration snapshots. Each snapshot modifies qubit T1/T2, gate errors,
and readout errors on top of a base FakeBackend.

Usage:
    from backends.replay_backend import ReplayBackend

    # Create a 7-day replay with realistic drift patterns
    backend = ReplayBackend.from_synthetic_history(
        base_backend="FakeBrisbane",
        duration_hours=168,      # 7 days
        snapshot_interval_hours=1,
    )

    # Advance to a specific point in time
    backend.set_time(hours=42.5)
    result = backend.run(circuit, shots=8192)

    # Or step through time
    for t in range(168):
        backend.set_time(hours=t)
        health = backend.get_health()
        print(f"t={t}h  drift_score={health.drift_score}")
"""

import copy
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

from backends.base import (
    BackendHealth,
    QubitProperties,
    ShadowBackend,
    SimulationResult,
)


@dataclass
class CalibrationSnapshot:
    """A single calibration snapshot at a point in time."""
    timestamp_hours: float
    qubit_t1_us: dict[int, float]       # qubit -> T1 in microseconds
    qubit_t2_us: dict[int, float]       # qubit -> T2 in microseconds
    gate_errors_1q: dict[int, float]    # qubit -> 1Q gate error rate
    gate_errors_2q: dict[tuple[int, int], float]  # (q0, q1) -> 2Q gate error rate
    readout_errors: dict[int, float]    # qubit -> readout error rate


def _generate_drift_series(
    base_value: float,
    num_points: int,
    drift_amplitude: float = 0.15,
    noise_std: float = 0.03,
    jump_prob: float = 0.02,
    jump_magnitude: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a realistic drift time series for a single parameter.

    Combines:
      - Slow sinusoidal drift (thermal cycling, diurnal patterns)
      - Random walk noise (shot-to-shot variation)
      - Occasional sudden jumps (TLS events, cosmic rays)

    All values are expressed as multiplicative factors around 1.0.
    E.g., a factor of 0.8 means the parameter dropped to 80% of baseline.
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0, 2 * np.pi * 3, num_points)  # ~3 full cycles over the series

    # Slow drift: sinusoidal with random phase
    phase = rng.uniform(0, 2 * np.pi)
    slow_drift = drift_amplitude * np.sin(t + phase)

    # Random walk component
    walk = np.cumsum(rng.normal(0, noise_std / np.sqrt(num_points), num_points))

    # Sudden jumps (TLS-like events)
    jumps = np.zeros(num_points)
    for i in range(num_points):
        if rng.random() < jump_prob:
            jump_size = rng.choice([-1, 1]) * jump_magnitude * rng.random()
            jumps[i:] += jump_size

    # Combine and convert to multiplicative factor
    factors = 1.0 + slow_drift + walk + jumps

    # Clamp to reasonable range (parameter can't go negative or explode)
    factors = np.clip(factors, 0.3, 3.0)

    return factors


class ReplayBackend(ShadowBackend):
    """Replays a sequence of calibration snapshots, simulating real device drift.

    The backend maintains a timeline of CalibrationSnapshots. When set_time()
    is called, it interpolates between the nearest snapshots to produce
    the current noise model. The agent sees changing hardware properties
    over time, just like on a real device.
    """

    def __init__(
        self,
        base_backend_name: str,
        snapshots: list[CalibrationSnapshot],
        coupling_map: list[tuple[int, int]],
        num_qubits: int,
    ):
        self._base_name = base_backend_name
        self._snapshots = sorted(snapshots, key=lambda s: s.timestamp_hours)
        self._coupling_map = coupling_map
        self._num_qubits = num_qubits
        self._current_time = 0.0
        self._current_snapshot: Optional[CalibrationSnapshot] = None
        self._sim: Optional[AerSimulator] = None
        self._creation_time = time.time()

        # Initialize to t=0
        self.set_time(0.0)

    @classmethod
    def from_synthetic_history(
        cls,
        base_backend: str = "FakeBrisbane",
        duration_hours: float = 168.0,
        snapshot_interval_hours: float = 1.0,
        seed: int = 42,
        active_qubits: list[int] | None = None,
    ) -> "ReplayBackend":
        """Create a ReplayBackend with synthetically generated drift history.

        Uses a real FakeBackend as the baseline, then generates realistic
        drift patterns for T1/T2/gate-errors/readout-errors over the
        specified duration.

        Args:
            base_backend: Name of the FakeBackend to use as baseline.
            duration_hours: Total duration of the replay in hours.
            snapshot_interval_hours: Time between calibration snapshots.
            seed: Random seed for reproducibility.
            active_qubits: Subset of qubits to track (None = first 20).
        """
        from backends.fake_adapter import FakeBackendAdapter

        adapter = FakeBackendAdapter(base_backend)
        rng = np.random.default_rng(seed)

        if active_qubits is None:
            active_qubits = list(range(min(20, adapter.num_qubits)))

        # Extract baseline properties
        baseline_props = adapter.get_qubit_properties(active_qubits)
        baseline_health = adapter.get_health()
        coupling_map = adapter.get_coupling_map()

        # Filter coupling map to active qubits
        active_set = set(active_qubits)
        active_coupling = [
            (a, b) for a, b in coupling_map
            if a in active_set and b in active_set
        ]

        num_snapshots = int(duration_hours / snapshot_interval_hours) + 1

        # Generate drift series for each qubit's parameters
        drift_t1 = {}
        drift_t2 = {}
        drift_1q_err = {}
        drift_readout = {}
        for qp in baseline_props:
            q = qp.qubit
            drift_t1[q] = _generate_drift_series(
                qp.t1_us, num_snapshots,
                drift_amplitude=0.1, noise_std=0.02, jump_prob=0.01, rng=rng,
            )
            drift_t2[q] = _generate_drift_series(
                qp.t2_us, num_snapshots,
                drift_amplitude=0.12, noise_std=0.025, jump_prob=0.015, rng=rng,
            )
            drift_1q_err[q] = _generate_drift_series(
                1.0, num_snapshots,
                drift_amplitude=0.08, noise_std=0.02, jump_prob=0.01, rng=rng,
            )
            drift_readout[q] = _generate_drift_series(
                1.0, num_snapshots,
                drift_amplitude=0.1, noise_std=0.02, jump_prob=0.01, rng=rng,
            )

        # Generate drift for 2Q gate errors
        drift_2q_err = {}
        for a, b in active_coupling:
            drift_2q_err[(a, b)] = _generate_drift_series(
                1.0, num_snapshots,
                drift_amplitude=0.1, noise_std=0.03, jump_prob=0.02, rng=rng,
            )

        # Build snapshots
        snapshots = []
        for i in range(num_snapshots):
            t = i * snapshot_interval_hours

            qubit_t1 = {}
            qubit_t2 = {}
            gate_1q = {}
            gate_2q = {}
            readout = {}

            for qp in baseline_props:
                q = qp.qubit
                qubit_t1[q] = max(1.0, qp.t1_us * drift_t1[q][i])
                qubit_t2[q] = max(0.5, min(qp.t2_us * drift_t2[q][i], 2 * qubit_t1[q]))
                # Gate errors: apply multiplicative factor, clamp to [1e-5, 0.5]
                base_1q = max(qp.gate_errors.values()) if qp.gate_errors else baseline_health.avg_1q_error
                gate_1q[q] = float(np.clip(base_1q * drift_1q_err[q][i], 1e-5, 0.5))
                readout[q] = float(np.clip(qp.readout_error * drift_readout[q][i], 1e-5, 0.5))

            for (a, b) in active_coupling:
                base_2q = baseline_health.avg_2q_error
                gate_2q[(a, b)] = float(np.clip(base_2q * drift_2q_err[(a, b)][i], 1e-4, 0.5))

            snapshots.append(CalibrationSnapshot(
                timestamp_hours=t,
                qubit_t1_us=qubit_t1,
                qubit_t2_us=qubit_t2,
                gate_errors_1q=gate_1q,
                gate_errors_2q=gate_2q,
                readout_errors=readout,
            ))

        return cls(
            base_backend_name=f"Replay({base_backend})",
            snapshots=snapshots,
            coupling_map=active_coupling,
            num_qubits=len(active_qubits),
        )

    @property
    def name(self) -> str:
        return self._base_name

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def current_time(self) -> float:
        """Current replay time in hours."""
        return self._current_time

    @property
    def total_duration(self) -> float:
        """Total duration of the replay in hours."""
        if not self._snapshots:
            return 0.0
        return self._snapshots[-1].timestamp_hours

    def set_time(self, hours: float) -> None:
        """Advance the replay to a specific time point.

        Interpolates between the two nearest calibration snapshots.
        """
        hours = max(0.0, min(hours, self.total_duration))
        self._current_time = hours
        self._current_snapshot = self._interpolate_snapshot(hours)
        self._sim = self._build_simulator(self._current_snapshot)

    def _interpolate_snapshot(self, hours: float) -> CalibrationSnapshot:
        """Linear interpolation between the two nearest snapshots."""
        if len(self._snapshots) == 1:
            return copy.deepcopy(self._snapshots[0])

        # Find bracketing snapshots
        left = self._snapshots[0]
        right = self._snapshots[-1]
        for i in range(len(self._snapshots) - 1):
            if self._snapshots[i].timestamp_hours <= hours <= self._snapshots[i + 1].timestamp_hours:
                left = self._snapshots[i]
                right = self._snapshots[i + 1]
                break

        if left.timestamp_hours == right.timestamp_hours:
            return copy.deepcopy(left)

        # Interpolation factor
        alpha = (hours - left.timestamp_hours) / (right.timestamp_hours - left.timestamp_hours)

        def lerp(a: float, b: float) -> float:
            return a + alpha * (b - a)

        qubits = set(left.qubit_t1_us.keys()) & set(right.qubit_t1_us.keys())
        return CalibrationSnapshot(
            timestamp_hours=hours,
            qubit_t1_us={q: lerp(left.qubit_t1_us[q], right.qubit_t1_us[q]) for q in qubits},
            qubit_t2_us={q: lerp(left.qubit_t2_us[q], right.qubit_t2_us[q]) for q in qubits},
            gate_errors_1q={q: lerp(left.gate_errors_1q[q], right.gate_errors_1q[q]) for q in qubits},
            gate_errors_2q={
                k: lerp(left.gate_errors_2q.get(k, 0.01), right.gate_errors_2q.get(k, 0.01))
                for k in set(left.gate_errors_2q) | set(right.gate_errors_2q)
            },
            readout_errors={q: lerp(left.readout_errors[q], right.readout_errors[q]) for q in qubits},
        )

    def _build_simulator(self, snap: CalibrationSnapshot) -> AerSimulator:
        """Build a noisy AerSimulator from a calibration snapshot."""
        noise_model = NoiseModel()
        qubits = sorted(snap.qubit_t1_us.keys())

        # 1Q thermal relaxation + depolarizing
        for q in qubits:
            t1 = snap.qubit_t1_us[q] * 1e-6  # us -> seconds
            t2 = snap.qubit_t2_us[q] * 1e-6
            gate_time = 35e-9  # typical 1Q gate time
            err_rate = snap.gate_errors_1q.get(q, 0.001)

            thermal_err = thermal_relaxation_error(t1, t2, gate_time)
            depol_err = depolarizing_error(err_rate, 1)
            combined = thermal_err.compose(depol_err)
            noise_model.add_quantum_error(combined, ["sx", "rz", "x"], [q])

        # 2Q depolarizing
        for (q0, q1), err_rate in snap.gate_errors_2q.items():
            depol_err = depolarizing_error(err_rate, 2)
            noise_model.add_quantum_error(depol_err, ["cx", "ecr"], [q0, q1])

        # Readout errors
        for q in qubits:
            re = snap.readout_errors.get(q, 0.01)
            from qiskit_aer.noise import ReadoutError
            # Symmetric readout error for simplicity
            readout_err = ReadoutError([[1 - re, re], [re, 1 - re]])
            noise_model.add_readout_error(readout_err, [q])

        from qiskit.transpiler import CouplingMap
        cm = CouplingMap(self._coupling_map) if self._coupling_map else None

        sim = AerSimulator(
            noise_model=noise_model,
            coupling_map=cm,
        )
        return sim

    def get_health(self) -> BackendHealth:
        snap = self._current_snapshot
        if snap is None:
            raise RuntimeError("Call set_time() first")

        t1_vals = list(snap.qubit_t1_us.values())
        t2_vals = list(snap.qubit_t2_us.values())
        err_1q = list(snap.gate_errors_1q.values())
        err_2q = list(snap.gate_errors_2q.values())
        readout = list(snap.readout_errors.values())

        return BackendHealth(
            name=self._base_name,
            num_qubits=self._num_qubits,
            avg_1q_error=float(np.mean(err_1q)) if err_1q else 0.0,
            avg_2q_error=float(np.mean(err_2q)) if err_2q else 0.0,
            avg_readout_error=float(np.mean(readout)) if readout else 0.0,
            avg_t1_us=float(np.mean(t1_vals)) if t1_vals else 0.0,
            avg_t2_us=float(np.mean(t2_vals)) if t2_vals else 0.0,
            calibration_age_minutes=self._current_time * 60.0,
            drift_score=self._compute_drift_score(),
        )

    def _compute_drift_score(self) -> float:
        """Compute a drift score by comparing current snapshot to baseline (t=0).

        Score in [0, 1]: 0 = no drift, 1 = severe drift.
        Based on normalized deviation of key parameters from their t=0 values.
        """
        if not self._snapshots or self._current_snapshot is None:
            return 0.0

        baseline = self._snapshots[0]
        current = self._current_snapshot
        deviations = []

        for q in baseline.qubit_t1_us:
            if q in current.qubit_t1_us:
                base_t1 = baseline.qubit_t1_us[q]
                cur_t1 = current.qubit_t1_us[q]
                if base_t1 > 0:
                    deviations.append(abs(cur_t1 - base_t1) / base_t1)

        for q in baseline.gate_errors_1q:
            if q in current.gate_errors_1q:
                base_e = baseline.gate_errors_1q[q]
                cur_e = current.gate_errors_1q[q]
                if base_e > 0:
                    deviations.append(abs(cur_e - base_e) / base_e)

        if not deviations:
            return 0.0

        # Average normalized deviation, clamped to [0, 1]
        raw = float(np.mean(deviations))
        return float(np.clip(raw, 0.0, 1.0))

    def get_qubit_properties(self, qubits: list[int]) -> list[QubitProperties]:
        snap = self._current_snapshot
        if snap is None:
            raise RuntimeError("Call set_time() first")

        results = []
        for q in qubits:
            results.append(QubitProperties(
                qubit=q,
                t1_us=snap.qubit_t1_us.get(q, 0.0),
                t2_us=snap.qubit_t2_us.get(q, 0.0),
                readout_error=snap.readout_errors.get(q, 0.0),
                gate_errors={"sx": snap.gate_errors_1q.get(q, 0.0)},
            ))
        return results

    def get_coupling_map(self) -> list[tuple[int, int]]:
        return list(self._coupling_map)

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 8192,
        **kwargs,
    ) -> SimulationResult:
        if self._sim is None:
            raise RuntimeError("Call set_time() first")

        transpiled = transpile(circuit, backend=self._sim, optimization_level=0)
        job = self._sim.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Estimate fidelity vs ideal
        fidelity = self._estimate_fidelity(circuit, counts, shots)

        return SimulationResult(
            counts=counts,
            shots=shots,
            fidelity=fidelity,
            metadata={
                "backend": self._base_name,
                "replay_time_hours": self._current_time,
                "drift_score": self._compute_drift_score(),
                "transpiled_depth": transpiled.depth(),
            },
        )

    def _estimate_fidelity(
        self,
        original: QuantumCircuit,
        noisy_counts: dict[str, int],
        shots: int,
    ) -> Optional[float]:
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

    def get_properties_snapshot(self) -> dict[str, Any]:
        health = self.get_health()
        return {
            "backend": self._base_name,
            "timestamp": time.time(),
            "replay_time_hours": self._current_time,
            "num_qubits": self._num_qubits,
            "avg_1q_error": health.avg_1q_error,
            "avg_2q_error": health.avg_2q_error,
            "avg_readout_error": health.avg_readout_error,
            "avg_t1_us": health.avg_t1_us,
            "avg_t2_us": health.avg_t2_us,
            "drift_score": health.drift_score,
        }
