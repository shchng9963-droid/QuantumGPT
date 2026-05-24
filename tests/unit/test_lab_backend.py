"""Tests for LabBackend interface and implementations."""

import numpy as np
import pytest

from backends.lab_backend import (
    CalibrationUpdate,
    DeviceSnapshot,
    LabBackend,
    MeasurementSpec,
    PulseSchedule,
    RawResult,
)
from backends.dynamics_lab_adapter import DynamicsLabAdapter


class TestDynamicsLabAdapter:
    """Tests for the pulse-level DynamicsLabAdapter."""

    @pytest.fixture
    def backend(self):
        return DynamicsLabAdapter(
            qubit_freq_ghz=5.0,
            t1_us=200.0,
            t2_us=150.0,
            backend_name="TestSim",
        )

    def test_properties(self, backend):
        assert backend.name == "TestSim"
        assert backend.num_qubits == 1
        assert backend.device_type == "superconducting"
        assert backend.supports_pulse is True

    def test_get_device_state(self, backend):
        state = backend.get_device_state()
        assert isinstance(state, DeviceSnapshot)
        assert state.backend_name == "TestSim"
        assert state.num_qubits == 1
        assert state.t1_us == [200.0]
        assert state.t2_us == [150.0]
        assert len(state.qubit_frequencies_ghz) == 1
        assert state.qubit_frequencies_ghz[0] == 5.0

    def test_dispatch_pulse_ground_state(self, backend):
        """Zero amplitude pulse should leave qubit in ground state."""
        schedule = PulseSchedule(
            channels={"d0": [{"amplitude": 0.0, "duration_ns": 100.0, "shape": "square"}]},
            duration_ns=100.0,
            metadata={"shots": 1000},
        )
        result = backend.dispatch_pulse(schedule)
        assert result.counts is not None
        # Should be mostly in |0>
        p1 = result.counts.get("1", 0) / sum(result.counts.values())
        assert p1 < 0.05  # very low excited population

    def test_dispatch_pulse_pi_rotation(self, backend):
        """Pi-pulse should flip qubit to |1>."""
        # For this backend: amp=0.005, dur=100ns gives pi rotation
        schedule = PulseSchedule(
            channels={"d0": [{"amplitude": 0.005, "duration_ns": 100.0, "shape": "square"}]},
            duration_ns=100.0,
            metadata={"shots": 2000},
        )
        result = backend.dispatch_pulse(schedule)
        p1 = result.counts.get("1", 0) / sum(result.counts.values())
        assert p1 > 0.95  # should be near 1

    def test_dispatch_pulse_half_pi(self, backend):
        """Half-pi pulse should give ~50% population."""
        schedule = PulseSchedule(
            channels={"d0": [{"amplitude": 0.0025, "duration_ns": 100.0, "shape": "square"}]},
            duration_ns=100.0,
            metadata={"shots": 4000},
        )
        result = backend.dispatch_pulse(schedule)
        p1 = result.counts.get("1", 0) / sum(result.counts.values())
        assert 0.3 < p1 < 0.7  # roughly 50%

    def test_dispatch_pulse_decoherence(self, backend):
        """Long delay after pi-pulse should show T1 decay."""
        # Pi-pulse + 400us delay (2*T1)
        schedule = PulseSchedule(
            channels={"d0": [
                {"amplitude": 0.005, "duration_ns": 100.0, "shape": "square"},
                {"amplitude": 0.0, "duration_ns": 400_000.0, "shape": "square"},
            ]},
            duration_ns=400_100.0,
            metadata={"shots": 2000},
        )
        result = backend.dispatch_pulse(schedule)
        p1 = result.counts.get("1", 0) / sum(result.counts.values())
        # After 2*T1, should decay to ~exp(-2) = 0.135
        assert p1 < 0.3

    def test_read_measurement_iq(self, backend):
        """IQ measurement should return proper shape."""
        spec = MeasurementSpec(mode="iq", qubits=[0], shots=100)
        result = backend.read_measurement(spec)
        assert result.iq_data is not None
        assert result.iq_data.shape == (100, 1, 2)

    def test_read_measurement_waveform(self, backend):
        """Waveform measurement should return time series."""
        spec = MeasurementSpec(mode="waveform", integration_time_ns=1000.0)
        result = backend.read_measurement(spec)
        assert result.waveform is not None
        assert result.timestamps is not None
        assert len(result.waveform) == len(result.timestamps)

    def test_update_calibration(self, backend):
        """Calibration update should succeed."""
        update = CalibrationUpdate(
            qubit_index=0,
            parameter="frequency",
            value=5.01,
            unit="GHz",
        )
        assert backend.update_calibration(update) is True
        state = backend.get_device_state()
        assert state.qubit_frequencies_ghz[0] == 5.01

    def test_is_healthy(self, backend):
        assert backend.is_healthy() is True
        assert backend.is_healthy(t1_threshold=300.0) is False


class TestExperimentProtocols:
    """Tests for experiment protocols running on DynamicsLabAdapter."""

    @pytest.fixture
    def backend(self):
        return DynamicsLabAdapter(qubit_freq_ghz=5.0, t1_us=200.0, t2_us=150.0)

    def test_rabi_experiment(self, backend):
        from experiments import RabiExperiment
        exp = RabiExperiment(amp_min=0.0, amp_max=0.02, n_points=15, shots=512)
        result = exp.run(backend)
        result = exp.analyze(result)

        assert result.protocol_name == "rabi"
        assert result.sweep_parameter == "amplitude"
        assert len(result.sweep_values) == 15
        assert len(result.measured_values) == 15
        assert result.fit is not None
        assert result.fit.r_squared > 0.9
        assert "pi_amplitude" in result.fit.parameters

    def test_t1_experiment(self, backend):
        from experiments import T1Experiment
        exp = T1Experiment(delay_max_ns=600_000, n_points=15,
                          pi_amplitude=0.005, shots=512)
        result = exp.run(backend)
        result = exp.analyze(result)

        assert result.protocol_name == "t1"
        assert result.sweep_parameter == "delay_ns"
        assert result.fit is not None
        assert result.fit.r_squared > 0.8
        # T1 should be close to 200us
        t1_measured = result.fit.parameters["T1_us"]
        assert 100 < t1_measured < 400  # within factor of 2

    def test_ramsey_experiment(self, backend):
        from experiments import RamseyExperiment
        exp = RamseyExperiment(delay_max_ns=3000, n_points=20,
                              pi_half_amplitude=0.0025, shots=512,
                              artificial_detuning_mhz=2.0)
        result = exp.run(backend)
        result = exp.analyze(result)

        assert result.protocol_name == "ramsey"
        assert result.sweep_parameter == "delay_ns"
        assert result.fit is not None
        # Detuning should be close to 2 MHz
        if "detuning_mhz" in result.fit.parameters:
            det = result.fit.parameters["detuning_mhz"]
            assert 0.5 < det < 5.0

    def test_experiment_result_summary(self, backend):
        from experiments import RabiExperiment
        exp = RabiExperiment(amp_max=0.02, n_points=10, shots=256)
        result = exp.run(backend)
        result = exp.analyze(result)
        summary = result.summary()
        assert "rabi" in summary
        assert "R" in summary  # R-squared
