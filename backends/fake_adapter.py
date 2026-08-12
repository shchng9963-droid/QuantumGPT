"""FakeBackendAdapter — wraps Qiskit FakeBackendV2 as a ShadowBackend.

Usage:
    from backends.fake_adapter import FakeBackendAdapter
    backend = FakeBackendAdapter("FakeBrisbane")
    result = backend.run(circuit, shots=8192)
"""

import time
from typing import Any, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator

from backends.base import (
    BackendHealth,
    QubitProperties,
    ShadowBackend,
    SimulationResult,
)

# Registry of available fake backends.
# Qiskit 2.x moved fake backends to qiskit_ibm_runtime.fake_provider
_FAKE_BACKEND_REGISTRY: dict[str, type] = {}


def _populate_registry():
    """Lazily populate the registry of fake backends."""
    global _FAKE_BACKEND_REGISTRY
    if _FAKE_BACKEND_REGISTRY:
        return

    try:
        from qiskit_ibm_runtime.fake_provider import (
            FakeBrisbane,
            FakeKyiv,
            FakeSherbrooke,
            FakeTorino,
        )
        _FAKE_BACKEND_REGISTRY.update({
            "FakeBrisbane": FakeBrisbane,
            "FakeKyiv": FakeKyiv,
            "FakeSherbrooke": FakeSherbrooke,
            "FakeTorino": FakeTorino,
        })
    except ImportError:
        pass


class FakeBackendAdapter(ShadowBackend):
    """Adapts a Qiskit FakeBackendV2 to the ShadowBackend interface.
    
    This gives the agent a static snapshot of a real IBM device,
    including its coupling map, gate errors, T1/T2, and readout errors.
    """

    def __init__(self, backend_name: str = "FakeBrisbane"):
        _populate_registry()
        if backend_name not in _FAKE_BACKEND_REGISTRY:
            available = list(_FAKE_BACKEND_REGISTRY.keys())
            raise ValueError(
                f"Unknown fake backend '{backend_name}'. "
                f"Available: {available}"
            )
        self._fake = _FAKE_BACKEND_REGISTRY[backend_name]()
        self._backend_name = backend_name
        self._sim = AerSimulator.from_backend(self._fake)
        self._creation_time = time.time()

        # Deterministic-mode hooks for paired tests (Plan v2.5 G2 gate).
        # When ``deterministic_seed`` is not None, each run() call advances
        # ``_call_index`` and feeds ``deterministic_seed + _call_index`` to
        # AerSimulator as ``seed_simulator``. Two backends configured with
        # the same seed will therefore produce identical noisy results given
        # an identical sequence of run() invocations. This is required to
        # prove orchestrator≡flat-ReAct equivalence below ZNE shot noise.
        self.deterministic_seed: Optional[int] = None
        self._call_index: int = 0

    @property
    def name(self) -> str:
        return self._backend_name

    @property
    def num_qubits(self) -> int:
        return self._fake.num_qubits

    def get_health(self) -> BackendHealth:
        target = self._fake.target
        
        # Collect gate errors
        gate_errors_1q = []
        gate_errors_2q = []
        for op_name in target.operation_names:
            for qargs, props in target[op_name].items():
                if props is not None and props.error is not None:
                    if len(qargs) == 1:
                        gate_errors_1q.append(props.error)
                    elif len(qargs) == 2:
                        gate_errors_2q.append(props.error)

        # Collect qubit properties
        t1_vals = []
        t2_vals = []
        readout_errors = []
        for q in range(self.num_qubits):
            qp = target.qubit_properties[q] if target.qubit_properties else None
            if qp is not None:
                if qp.t1 is not None:
                    t1_vals.append(qp.t1 * 1e6)  # seconds -> microseconds
                if qp.t2 is not None:
                    t2_vals.append(qp.t2 * 1e6)
            # Readout errors from measure operation
            if "measure" in target.operation_names:
                meas_props = target["measure"].get((q,))
                if meas_props is not None and meas_props.error is not None:
                    readout_errors.append(meas_props.error)

        age_minutes = (time.time() - self._creation_time) / 60.0

        return BackendHealth(
            name=self._backend_name,
            num_qubits=self.num_qubits,
            avg_1q_error=float(np.mean(gate_errors_1q)) if gate_errors_1q else 0.0,
            avg_2q_error=float(np.mean(gate_errors_2q)) if gate_errors_2q else 0.0,
            avg_readout_error=float(np.mean(readout_errors)) if readout_errors else 0.0,
            avg_t1_us=float(np.mean(t1_vals)) if t1_vals else 0.0,
            avg_t2_us=float(np.mean(t2_vals)) if t2_vals else 0.0,
            calibration_age_minutes=age_minutes,
        )

    def get_qubit_properties(self, qubits: list[int]) -> list[QubitProperties]:
        target = self._fake.target
        results = []

        for q in qubits:
            qp = target.qubit_properties[q] if target.qubit_properties else None
            t1 = (qp.t1 * 1e6) if (qp and qp.t1) else 0.0
            t2 = (qp.t2 * 1e6) if (qp and qp.t2) else 0.0

            # Readout error
            readout_err = 0.0
            if "measure" in target.operation_names:
                meas_props = target["measure"].get((q,))
                if meas_props and meas_props.error is not None:
                    readout_err = meas_props.error

            # Gate errors for this qubit
            gate_errors = {}
            for op_name in target.operation_names:
                if op_name == "measure":
                    continue
                props = target[op_name].get((q,))
                if props and props.error is not None:
                    gate_errors[op_name] = props.error

            results.append(QubitProperties(
                qubit=q,
                t1_us=t1,
                t2_us=t2,
                readout_error=readout_err,
                gate_errors=gate_errors,
            ))

        return results

    def get_coupling_map(self) -> list[tuple[int, int]]:
        cm = self._fake.coupling_map
        if cm is None:
            return []
        return list(cm.get_edges())

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 8192,
        **kwargs,
    ) -> SimulationResult:
        # Transpile for this backend's topology
        opt_level = kwargs.get("optimization_level", 1)
        initial_layout = kwargs.get("initial_layout", None)
        transpiled = transpile(
            circuit,
            backend=self._sim,
            optimization_level=opt_level,
            initial_layout=initial_layout,
        )

        # Build run options. In deterministic mode, advance the per-backend
        # call counter so successive run() calls (e.g. ZNE's noise-scale
        # batch) get distinct but reproducible seeds.
        run_kwargs: dict[str, Any] = {"shots": shots}
        # An explicit seed_simulator in kwargs always wins (debug hook).
        explicit_seed = kwargs.get("seed_simulator")
        if explicit_seed is not None:
            run_kwargs["seed_simulator"] = int(explicit_seed)
        elif self.deterministic_seed is not None:
            run_kwargs["seed_simulator"] = int(self.deterministic_seed) + self._call_index
            self._call_index += 1

        # Run on noisy simulator
        job = self._sim.run(transpiled, **run_kwargs)
        result = job.result()
        counts = result.get_counts()

        # Compute fidelity vs ideal if the circuit is small enough
        fidelity = self._estimate_fidelity(circuit, counts, shots)

        return SimulationResult(
            counts=counts,
            shots=shots,
            fidelity=fidelity,
            metadata={
                "backend": self._backend_name,
                "transpiled_depth": transpiled.depth(),
                "transpiled_gate_count": transpiled.size(),
                "seed_simulator": run_kwargs.get("seed_simulator"),
            },
        )

    def get_properties_snapshot(self) -> dict[str, Any]:
        """Return properties as a dict for the drift detector / stream."""
        health = self.get_health()
        return {
            "backend": self._backend_name,
            "timestamp": time.time(),
            "num_qubits": self.num_qubits,
            "avg_1q_error": health.avg_1q_error,
            "avg_2q_error": health.avg_2q_error,
            "avg_readout_error": health.avg_readout_error,
            "avg_t1_us": health.avg_t1_us,
            "avg_t2_us": health.avg_t2_us,
        }

    def _estimate_fidelity(
        self,
        original: QuantumCircuit,
        noisy_counts: dict[str, int],
        shots: int,
    ) -> Optional[float]:
        """Estimate fidelity by comparing noisy output to ideal simulation.
        
        Only works for circuits with <= 20 qubits (statevector feasible).
        """
        n = original.num_qubits
        if n > 20:
            return None

        try:
            from qiskit_aer import AerSimulator as IdealSim
            ideal_sim = IdealSim(method="statevector")
            # Add measurements if not present
            meas_circ = original.copy()
            if meas_circ.count_ops().get("measure", 0) == 0:
                meas_circ.measure_all()
            ideal_job = ideal_sim.run(meas_circ, shots=shots)
            ideal_counts = ideal_job.result().get_counts()

            # Classical fidelity: sum of sqrt(p_i * q_i) squared
            all_keys = set(ideal_counts.keys()) | set(noisy_counts.keys())
            fid = 0.0
            for key in all_keys:
                p = ideal_counts.get(key, 0) / shots
                q = noisy_counts.get(key, 0) / shots
                fid += np.sqrt(p * q)
            return float(fid ** 2)
        except Exception:
            return None
