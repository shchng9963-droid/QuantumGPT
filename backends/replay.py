"""ReplayBackend — replays historical calibration data over time.

Given a time series of CalibrationSnapshot objects, this backend
"plays back" the noise characteristics as if the real device were
drifting. The agent sees a different noise profile depending on
*when* it queries the backend.

Usage:
    from backends.calibration_data import extract_snapshot_from_fake_backend, generate_drift_series
    from backends.replay import ReplayBackend

    base = extract_snapshot_from_fake_backend("FakeBrisbane")
    snapshots = generate_drift_series(base, num_snapshots=168, seed=42)
    backend = ReplayBackend(snapshots, speed=60.0)  # 60x realtime
    result = backend.run(circuit, shots=8192)
"""

import bisect
import time
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
from backends.calibration_data import CalibrationSnapshot


class ReplayBackend(ShadowBackend):
    """Shadow backend that replays historical calibration snapshots.

    The noise model changes over time, mimicking real device drift.
    A "replay clock" maps wall-clock time to snapshot time, with
    configurable speed multiplier.
    """

    def __init__(
        self,
        snapshots: list[CalibrationSnapshot],
        speed: float = 1.0,
        start_index: int = 0,
    ):
        """
        Args:
            snapshots: Time-ordered list of calibration snapshots
            speed: Replay speed multiplier (60.0 = 1 hour of data per minute)
            start_index: Which snapshot to start from
        """
        if not snapshots:
            raise ValueError("Need at least one snapshot")
        
        self._snapshots = sorted(snapshots, key=lambda s: s.timestamp)
        self._speed = speed
        self._start_index = start_index
        
        # Replay clock: wall_start maps to snapshot[start_index].timestamp
        self._wall_start = time.time()
        self._snap_start = self._snapshots[start_index].timestamp
        
        # Extract timestamps for bisect
        self._timestamps = [s.timestamp for s in self._snapshots]
        
        # Cache: reuse AerSimulator if snapshot hasn't changed
        self._cached_snap_idx: Optional[int] = None
        self._cached_sim: Optional[AerSimulator] = None

    @property
    def name(self) -> str:
        snap = self._current_snapshot()
        return f"Replay({snap.backend_name})"

    @property
    def num_qubits(self) -> int:
        return self._snapshots[0].num_qubits

    @property
    def current_snapshot_index(self) -> int:
        """Which snapshot the replay clock currently points to."""
        return self._find_current_index()

    @property
    def total_snapshots(self) -> int:
        return len(self._snapshots)

    def _replay_time(self) -> float:
        """Convert current wall-clock time to snapshot timeline time."""
        elapsed_wall = time.time() - self._wall_start
        return self._snap_start + elapsed_wall * self._speed

    def _find_current_index(self) -> int:
        """Find the snapshot index for the current replay time."""
        t = self._replay_time()
        idx = bisect.bisect_right(self._timestamps, t) - 1
        return max(0, min(idx, len(self._snapshots) - 1))

    def _current_snapshot(self) -> CalibrationSnapshot:
        return self._snapshots[self._find_current_index()]

    def seek(self, index: int):
        """Jump to a specific snapshot index."""
        index = max(0, min(index, len(self._snapshots) - 1))
        self._wall_start = time.time()
        self._snap_start = self._snapshots[index].timestamp
        self._cached_snap_idx = None

    def get_health(self) -> BackendHealth:
        snap = self._current_snapshot()
        
        return BackendHealth(
            name=f"Replay({snap.backend_name})",
            num_qubits=snap.num_qubits,
            avg_1q_error=float(np.mean(snap.gate_error_1q)) if snap.gate_error_1q else 0.0,
            avg_2q_error=float(np.mean(list(snap.gate_error_2q.values()))) if snap.gate_error_2q else 0.0,
            avg_readout_error=float(np.mean(snap.readout_error)) if snap.readout_error else 0.0,
            avg_t1_us=float(np.mean(snap.t1_us)) if snap.t1_us else 0.0,
            avg_t2_us=float(np.mean(snap.t2_us)) if snap.t2_us else 0.0,
            calibration_age_minutes=0.0,  # "just calibrated" — the snapshot IS the calibration
        )

    def get_qubit_properties(self, qubits: list[int]) -> list[QubitProperties]:
        snap = self._current_snapshot()
        results = []
        for q in qubits:
            results.append(QubitProperties(
                qubit=q,
                t1_us=snap.t1_us[q],
                t2_us=snap.t2_us[q],
                readout_error=snap.readout_error[q],
                gate_errors={"1q": snap.gate_error_1q[q]},
            ))
        return results

    def get_coupling_map(self) -> list[tuple[int, int]]:
        return self._current_snapshot().coupling_map

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 8192,
        **kwargs,
    ) -> SimulationResult:
        idx = self._find_current_index()
        sim = self._get_simulator(idx)
        
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
                "snapshot_index": idx,
                "snapshot_timestamp": self._snapshots[idx].timestamp,
                "transpiled_depth": transpiled.depth(),
                "transpiled_gate_count": transpiled.size(),
            },
        )

    def get_properties_snapshot(self) -> dict[str, Any]:
        snap = self._current_snapshot()
        return snap.to_dict()

    def _get_simulator(self, snap_idx: int) -> AerSimulator:
        """Build (or return cached) AerSimulator with noise from snapshot."""
        if self._cached_snap_idx == snap_idx and self._cached_sim is not None:
            return self._cached_sim
        
        snap = self._snapshots[snap_idx]
        noise_model = self._build_noise_model(snap)
        
        # Build a basic AerSimulator with the noise model
        # Use a subset of qubits for efficiency if circuit is small
        sim = AerSimulator(noise_model=noise_model)
        
        self._cached_snap_idx = snap_idx
        self._cached_sim = sim
        return sim

    def _build_noise_model(self, snap: CalibrationSnapshot) -> NoiseModel:
        """Build a Qiskit NoiseModel from a CalibrationSnapshot."""
        noise = NoiseModel()
        
        n = snap.num_qubits
        
        # 1Q gate errors: depolarizing + thermal relaxation
        for q in range(n):
            t1 = snap.t1_us[q] * 1e-6   # convert to seconds
            t2 = snap.t2_us[q] * 1e-6
            gate_time = 35e-9  # ~35 ns for 1Q gate (typical IBM)
            
            # Thermal relaxation
            if t1 > 0 and t2 > 0:
                try:
                    thermal_err = thermal_relaxation_error(t1, t2, gate_time)
                    noise.add_quantum_error(thermal_err, ["sx", "x"], [q])
                except Exception:
                    pass
            
            # Readout error
            re = snap.readout_error[q]
            if 0 < re < 0.5:
                readout_err = ReadoutError([[1 - re, re], [re, 1 - re]])
                noise.add_readout_error(readout_err, [q])
        
        # 2Q gate errors: depolarizing
        for edge, err in snap.gate_error_2q.items():
            if 0 < err < 1.0:
                try:
                    dep_err = depolarizing_error(err, 2)
                    noise.add_quantum_error(dep_err, ["cx", "ecr"], list(edge))
                except Exception:
                    pass
        
        return noise

    def _estimate_fidelity(
        self,
        original: QuantumCircuit,
        noisy_counts: dict[str, int],
        shots: int,
    ) -> Optional[float]:
        """Estimate classical fidelity vs ideal (noiseless) simulation."""
        n = original.num_qubits
        if n > 20:
            return None
        
        try:
            ideal_sim = AerSimulator(method="statevector")
            meas_circ = original.copy()
            if meas_circ.count_ops().get("measure", 0) == 0:
                meas_circ.measure_all()
            ideal_job = ideal_sim.run(meas_circ, shots=shots)
            ideal_counts = ideal_job.result().get_counts()
            
            all_keys = set(ideal_counts.keys()) | set(noisy_counts.keys())
            fid = 0.0
            for key in all_keys:
                p = ideal_counts.get(key, 0) / shots
                q = noisy_counts.get(key, 0) / shots
                fid += np.sqrt(p * q)
            return float(fid ** 2)
        except Exception:
            return None
