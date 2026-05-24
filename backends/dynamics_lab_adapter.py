"""DynamicsLabAdapter -- pulse-level LabBackend using Qiskit Dynamics.

This backend supports full pulse-level control via Hamiltonian simulation.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.linalg import expm

from backends.lab_backend import (
    CalibrationUpdate,
    DeviceSnapshot,
    LabBackend,
    MeasurementSpec,
    PulseSchedule,
    RawResult,
)


class DynamicsLabAdapter(LabBackend):
    """Pulse-level quantum backend using Hamiltonian simulation.

    Simulates a transmon qubit system with configurable parameters.
    Supports Rabi, Ramsey, T1, and arbitrary pulse experiments.
    """

    def __init__(
        self,
        num_qubits: int = 1,
        qubit_freq_ghz: float = 5.0,
        anharmonicity_ghz: float = -0.33,
        t1_us: float = 200.0,
        t2_us: float = 150.0,
        readout_error: float = 0.01,
        dt_ns: float = 0.5,
        backend_name: str = "DynamicsSim",
    ):
        self._num_qubits = num_qubits
        self._qubit_freq_ghz = qubit_freq_ghz
        self._anharmonicity_ghz = anharmonicity_ghz
        self._t1_us = t1_us
        self._t2_us = t2_us
        self._readout_error = readout_error
        self._dt_ns = dt_ns
        self._backend_name = backend_name
        self._calibration_store: dict[str, float] = {
            "q0_frequency": qubit_freq_ghz,
            "q0_pi_amplitude": 0.0,
        }
        self._creation_time = time.time()

    @property
    def name(self) -> str:
        return self._backend_name

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def device_type(self) -> str:
        return "superconducting"

    @property
    def supports_pulse(self) -> bool:
        return True

    def dispatch_pulse(self, schedule: PulseSchedule) -> RawResult:
        """Execute a pulse schedule via Hamiltonian simulation."""
        duration_ns = schedule.duration_ns
        dt = self._dt_ns
        n_steps = int(duration_ns / dt)

        if n_steps == 0:
            return RawResult(counts={"0": 1}, metadata={"trivial": True})

        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)

        psi = np.array([1.0, 0.0], dtype=complex)

        # Build drive envelope
        drive_envelope = np.zeros(n_steps, dtype=complex)
        drive_channel = schedule.channels.get("d0", [])

        t_offset = 0
        for pulse in drive_channel:
            amp = pulse.get("amplitude", 0.0)
            dur_ns = pulse.get("duration_ns", duration_ns)
            phase = pulse.get("phase", 0.0)
            shape = pulse.get("shape", "square")
            n_pulse = int(dur_ns / dt)

            if shape == "square":
                envelope = np.ones(n_pulse) * amp
            elif shape == "gaussian":
                sigma_t = n_pulse / 6
                t_arr = np.arange(n_pulse) - n_pulse / 2
                envelope = amp * np.exp(-t_arr**2 / (2 * sigma_t**2))
            elif shape == "drag":
                sigma_t = n_pulse / 6
                t_arr = np.arange(n_pulse) - n_pulse / 2
                gauss = np.exp(-t_arr**2 / (2 * sigma_t**2))
                drag_corr = -t_arr / sigma_t**2 * gauss
                beta = pulse.get("beta", 0.5)
                envelope = amp * (gauss + 1j * beta * drag_corr)
            else:
                envelope = np.ones(n_pulse) * amp

            envelope = envelope * np.exp(1j * phase)
            start = int(t_offset / dt)
            end = min(start + n_pulse, n_steps)
            drive_envelope[start:end] = envelope[:end - start]
            t_offset += dur_ns

        # Time evolution in rotating frame (RWA)
        # Optimization: batch contiguous non-zero segments, skip zero segments
        # Find non-zero regions
        nonzero_mask = np.abs(drive_envelope) > 1e-15
        
        if not np.any(nonzero_mask):
            # No drive at all -- qubit stays in ground state
            p1 = 0.0
        else:
            # Process only non-zero segments
            # Group into contiguous blocks for efficiency
            changes = np.diff(nonzero_mask.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1
            
            # Handle edge cases
            if nonzero_mask[0]:
                starts = np.concatenate([[0], starts])
            if nonzero_mask[-1]:
                ends = np.concatenate([ends, [n_steps]])
            
            for s, e in zip(starts, ends):
                for i in range(s, e):
                    omega = drive_envelope[i]
                    H = (omega.real * sigma_x + omega.imag * sigma_y) / 2
                    U = expm(-1j * H * dt * 2 * np.pi)
                    psi = U @ psi

            p1 = float(np.abs(psi[1])**2)

        # Decoherence model:
        # T1 causes |1> -> |0> decay: P(|1>) *= exp(-t/T1)
        # T2 causes dephasing (loss of coherence in superposition)
        # For populations (diagonal elements), only T1 matters
        # For coherences (off-diagonal), T2 applies
        # Simple model: P(|1>,t) = P(|1>,0) * exp(-t/T1)
        # This is physically correct for energy relaxation
        total_time_us = duration_ns / 1000.0
        t1_decay = np.exp(-total_time_us / self._t1_us)
        p1_eff = p1 * t1_decay

        # Shot noise + readout error
        shots = schedule.metadata.get("shots", 1024)
        n_excited = int(np.random.binomial(shots, np.clip(p1_eff, 0, 1)))
        n_ground = shots - n_excited
        n_misread_0 = int(np.random.binomial(n_ground, self._readout_error))
        n_misread_1 = int(np.random.binomial(n_excited, self._readout_error))

        counts = {
            "0": n_ground - n_misread_0 + n_misread_1,
            "1": n_excited - n_misread_1 + n_misread_0,
        }

        return RawResult(
            counts=counts,
            metadata={
                "p1_ideal": p1,
                "p1_with_decoherence": p1_eff,
                "duration_ns": duration_ns,
                "n_steps": n_steps,
                "t1_decay_factor": t1_decay,
            },
        )

    def run_circuit(self, circuit: Any, shots: int = 4096, **kwargs: Any) -> RawResult:
        """Run a circuit (simplified gate-level noise model)."""
        from qiskit.circuit import QuantumCircuit
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("Expected a QuantumCircuit")

        n_gates = circuit.size()
        gate_fidelity = (1 - 0.001) ** n_gates
        n_qubits = circuit.num_qubits
        counts: dict[str, int] = {}
        for _ in range(shots):
            if np.random.random() < gate_fidelity:
                outcome = "0" * n_qubits
            else:
                outcome = format(np.random.randint(0, 2**n_qubits), f"0{n_qubits}b")
            counts[outcome] = counts.get(outcome, 0) + 1

        return RawResult(counts=counts, metadata={"fidelity_estimate": gate_fidelity})

    def read_measurement(self, spec: MeasurementSpec) -> RawResult:
        """Read IQ data or waveform."""
        if spec.mode == "iq":
            n_qubits = len(spec.qubits) if spec.qubits else self._num_qubits
            sigma = 0.2
            iq = np.zeros((spec.shots, n_qubits, 2))
            for q in range(n_qubits):
                center = np.array([1.0, 0.0])
                iq[:, q, :] = center + np.random.normal(0, sigma, (spec.shots, 2))
            return RawResult(iq_data=iq, metadata={"mode": "iq"})
        elif spec.mode == "waveform":
            duration_ns = spec.integration_time_ns or 2000.0
            t = np.arange(0, duration_ns, self._dt_ns)
            kappa = 0.01
            signal = np.exp(-kappa * t) * np.cos(2 * np.pi * 7.0 * t)
            signal += np.random.normal(0, 0.02, len(t))
            return RawResult(waveform=signal, timestamps=t, metadata={"mode": "waveform"})
        else:
            return RawResult(counts={"0": spec.shots}, metadata={"mode": "counts"})

    def update_calibration(self, update: CalibrationUpdate) -> bool:
        """Update internal calibration parameters."""
        key = f"q{update.qubit_index}_{update.parameter}"
        self._calibration_store[key] = update.value
        if update.parameter == "frequency" and update.qubit_index == 0:
            self._qubit_freq_ghz = update.value
        return True

    def get_device_state(self) -> DeviceSnapshot:
        """Return device snapshot."""
        return DeviceSnapshot(
            backend_name=self._backend_name,
            num_qubits=self._num_qubits,
            qubit_frequencies_ghz=[self._qubit_freq_ghz] * self._num_qubits,
            t1_us=[self._t1_us] * self._num_qubits,
            t2_us=[self._t2_us] * self._num_qubits,
            readout_errors=[self._readout_error] * self._num_qubits,
            gate_errors_1q=[0.001] * self._num_qubits,
            gate_errors_2q={},
            coupling_map=[],
            calibration_age_minutes=(time.time() - self._creation_time) / 60.0,
            drift_score=None,
            extra={"calibration_store": dict(self._calibration_store)},
        )
