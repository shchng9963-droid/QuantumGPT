"""FakeLabAdapter — wraps FakeBackendAdapter to implement LabBackend interface.

This bridges the existing FakeBackendAdapter (ShadowBackend) to the new
LabBackend protocol, so all existing functionality keeps working while
new experiment protocols can use the unified interface.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from backends.fake_adapter import FakeBackendAdapter
from backends.lab_backend import (
    CalibrationUpdate,
    DeviceSnapshot,
    LabBackend,
    MeasurementSpec,
    PulseSchedule,
    RawResult,
)


class FakeLabAdapter(LabBackend):
    """LabBackend implementation wrapping Qiskit FakeBackendV2.

    Supports gate-level execution. Pulse dispatch is simulated
    by converting to equivalent gate operations where possible.
    """

    def __init__(self, backend_name: str = "FakeBrisbane"):
        self._adapter = FakeBackendAdapter(backend_name)
        self._calibration_store: dict[str, float] = {}

    @property
    def name(self) -> str:
        return self._adapter.name

    @property
    def num_qubits(self) -> int:
        return self._adapter.num_qubits

    @property
    def device_type(self) -> str:
        return "superconducting"

    @property
    def supports_pulse(self) -> bool:
        return False  # Gate-level only

    def dispatch_pulse(self, schedule: PulseSchedule) -> RawResult:
        """Not supported for fake backends — raises informative error."""
        raise NotImplementedError(
            f"{self.name} is a gate-level simulator. "
            "Use a pulse-level backend (e.g., DynamicsLabAdapter) for pulse dispatch."
        )

    def run_circuit(self, circuit: Any, shots: int = 4096, **kwargs: Any) -> RawResult:
        """Execute circuit via the underlying FakeBackendAdapter."""
        result = self._adapter.run(circuit, shots=shots, **kwargs)
        return RawResult(
            counts=result.counts,
            metadata={
                "fidelity": result.fidelity,
                **result.metadata,
            },
        )

    def read_measurement(self, spec: MeasurementSpec) -> RawResult:
        """Simulate a passive measurement read.

        For fake backends, generates synthetic IQ data based on device noise.
        """
        n_qubits = len(spec.qubits) if spec.qubits else self.num_qubits
        shots = spec.shots

        if spec.mode == "iq":
            # Generate synthetic IQ blobs
            # |0> centered at (1, 0), |1> centered at (-1, 0) with noise
            health = self._adapter.get_health()
            sigma = health.avg_readout_error * 5  # noise proportional to readout error
            # Random ground/excited state assignment
            states = np.random.randint(0, 2, size=(shots, n_qubits))
            iq = np.zeros((shots, n_qubits, 2))
            for q in range(n_qubits):
                for s in range(shots):
                    center = np.array([1.0, 0.0]) if states[s, q] == 0 else np.array([-1.0, 0.0])
                    iq[s, q] = center + np.random.normal(0, sigma, 2)
            return RawResult(iq_data=iq, metadata={"mode": "iq", "qubits": spec.qubits})

        elif spec.mode == "counts":
            # Just run a trivial circuit (identity) and measure
            from qiskit.circuit import QuantumCircuit
            qc = QuantumCircuit(n_qubits)
            qc.measure_all()
            result = self._adapter.run(qc, shots=shots)
            return RawResult(counts=result.counts, metadata={"mode": "counts"})

        else:
            # Waveform mode: generate synthetic time-domain signal
            duration_ns = spec.integration_time_ns or 1000.0
            dt = 1.0  # 1 ns sampling
            t = np.arange(0, duration_ns, dt)
            # Damped oscillation
            freq_ghz = 5.0
            waveform = np.exp(-t / 200.0) * np.cos(2 * np.pi * freq_ghz * t)
            waveform += np.random.normal(0, 0.05, len(t))
            return RawResult(
                waveform=waveform,
                timestamps=t,
                metadata={"mode": "waveform", "sample_rate_ghz": 1.0},
            )

    def update_calibration(self, update: CalibrationUpdate) -> bool:
        """Store calibration parameter (in-memory for fake backends)."""
        key = f"q{update.qubit_index}_{update.parameter}"
        self._calibration_store[key] = update.value
        return True

    def get_device_state(self) -> DeviceSnapshot:
        """Return complete device snapshot."""
        health = self._adapter.get_health()
        target = self._adapter._fake.target

        # Extract per-qubit data
        t1_list = []
        t2_list = []
        freq_list = []
        readout_list = []
        gate_1q_list = []

        for q in range(self.num_qubits):
            qp = target.qubit_properties[q] if target.qubit_properties else None
            t1_list.append((qp.t1 * 1e6) if (qp and qp.t1) else 0.0)
            t2_list.append((qp.t2 * 1e6) if (qp and qp.t2) else 0.0)
            freq_list.append((qp.frequency * 1e-9) if (qp and qp.frequency) else 5.0)

            # Readout error
            if "measure" in target.operation_names:
                meas_props = target["measure"].get((q,))
                readout_list.append(meas_props.error if (meas_props and meas_props.error) else 0.0)
            else:
                readout_list.append(0.0)

            # 1Q gate error (use sx as representative)
            for op in ["sx", "x", "rz"]:
                if op in target.operation_names:
                    props = target[op].get((q,))
                    if props and props.error is not None:
                        gate_1q_list.append(props.error)
                        break
            else:
                gate_1q_list.append(0.0)

        # 2Q gate errors
        gate_2q = {}
        for op_name in target.operation_names:
            for qargs, props in target[op_name].items():
                if len(qargs) == 2 and props and props.error is not None:
                    gate_2q[f"{op_name}_{qargs[0]}_{qargs[1]}"] = props.error

        coupling = self._adapter.get_coupling_map()

        return DeviceSnapshot(
            backend_name=self.name,
            num_qubits=self.num_qubits,
            qubit_frequencies_ghz=freq_list,
            t1_us=t1_list,
            t2_us=t2_list,
            readout_errors=readout_list,
            gate_errors_1q=gate_1q_list,
            gate_errors_2q=gate_2q,
            coupling_map=coupling,
            calibration_age_minutes=health.calibration_age_minutes,
            drift_score=None,
        )
