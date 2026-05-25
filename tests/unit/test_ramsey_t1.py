"""Unit tests for Ramsey and T1 experiment tools.

Tests the full pipeline: tool call → DynamicsLabAdapter → experiment → analyze → JSON.
"""

import json
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backends.dynamics_lab_adapter import DynamicsLabAdapter
from experiments.ramsey_experiment import RamseyExperiment
from experiments.t1_experiment import T1Experiment
from experiments.base import ExperimentResult


# ═══════════════════════════════════════════════════════
# Ramsey experiment protocol tests
# ═══════════════════════════════════════════════════════


class TestRamseyExperiment:
    """Tests for the RamseyExperiment protocol class."""

    @pytest.fixture
    def backend(self):
        return DynamicsLabAdapter(
            num_qubits=1,
            qubit_freq_ghz=5.0,
            t1_us=200.0,
            t2_us=150.0,
        )

    def test_run_basic(self, backend):
        """Run Ramsey with minimal points and verify output shape."""
        exp = RamseyExperiment(
            qubit=0,
            delay_max_ns=2000.0,
            n_points=10,
            pi_half_amplitude=0.015,
            artificial_detuning_mhz=2.0,
            shots=256,
        )
        result = exp.run(backend)

        assert result.protocol_name == "ramsey"
        assert result.sweep_parameter == "delay_ns"
        assert len(result.sweep_values) == 10
        assert len(result.measured_values) == 10
        assert all(0 <= p <= 1 for p in result.measured_values)
        assert result.raw_counts is not None
        assert len(result.raw_counts) == 10

    def test_analyze_produces_fit(self, backend):
        """Run + analyze should produce a fit with T2* and detuning."""
        exp = RamseyExperiment(
            delay_max_ns=3000.0,
            n_points=30,
            pi_half_amplitude=0.015,
            artificial_detuning_mhz=2.0,
            shots=512,
        )
        result = exp.run(backend)
        result = exp.analyze(result)

        assert result.fit is not None
        assert "T2_star_us" in result.fit.parameters
        assert "T2_star_ns" in result.fit.parameters
        # T2* should be positive and reasonable (> 0.1 us, < 1000 us)
        t2_us = result.fit.parameters["T2_star_us"]
        assert t2_us > 0.1
        assert t2_us < 1000.0

    def test_analyze_fallback_on_bad_data(self):
        """If data is pure noise, analyze should still not crash."""
        rng = np.random.default_rng(42)
        delays = np.linspace(0, 5000, 30)
        pops = rng.uniform(0.3, 0.7, 30)  # random noise, no oscillation

        result = ExperimentResult(
            protocol_name="ramsey",
            sweep_parameter="delay_ns",
            sweep_values=delays,
            measured_values=pops,
            metadata={"qubit": 0},
        )

        exp = RamseyExperiment()
        result = exp.analyze(result)
        # Should produce a fit (possibly fallback) without crashing
        assert result.fit is not None

    def test_name_property(self):
        exp = RamseyExperiment()
        assert exp.name == "ramsey"


# ═══════════════════════════════════════════════════════
# T1 experiment protocol tests
# ═══════════════════════════════════════════════════════


class TestT1Experiment:
    """Tests for the T1Experiment protocol class."""

    @pytest.fixture
    def backend(self):
        return DynamicsLabAdapter(
            num_qubits=1,
            qubit_freq_ghz=5.0,
            t1_us=200.0,
            t2_us=150.0,
        )

    def test_run_basic(self, backend):
        """Run T1 with minimal points and verify output shape."""
        exp = T1Experiment(
            qubit=0,
            delay_max_ns=400_000.0,  # 400 us
            n_points=10,
            pi_amplitude=0.03,
            shots=256,
        )
        result = exp.run(backend)

        assert result.protocol_name == "t1"
        assert result.sweep_parameter == "delay_ns"
        assert len(result.sweep_values) == 10
        assert len(result.measured_values) == 10
        assert all(0 <= p <= 1 for p in result.measured_values)

    def test_analyze_extracts_t1(self, backend):
        """Run + analyze should extract T1 close to the configured value."""
        exp = T1Experiment(
            delay_max_ns=600_000.0,  # 600 us (3x T1)
            n_points=40,
            pi_amplitude=0.03,
            shots=1024,
        )
        result = exp.run(backend)
        result = exp.analyze(result)

        assert result.fit is not None
        assert "T1_us" in result.fit.parameters
        t1_us = result.fit.parameters["T1_us"]
        # T1 should be in a reasonable range around 200 us
        # (the sim has T1=200, but noise introduces variance)
        assert 50.0 < t1_us < 500.0

    def test_decay_trend(self, backend):
        """P(|1>) at start should be higher than at end (decay)."""
        exp = T1Experiment(
            delay_max_ns=400_000.0,
            n_points=20,
            pi_amplitude=0.03,
            shots=512,
        )
        result = exp.run(backend)

        # First point (near 0 delay) should have higher excitation
        # than last point (400 us delay)
        assert result.measured_values[0] > result.measured_values[-1]

    def test_name_property(self):
        exp = T1Experiment()
        assert exp.name == "t1"


# ═══════════════════════════════════════════════════════
# Tool executor integration tests
# ═══════════════════════════════════════════════════════


class TestToolExecutorRamseyT1:
    """Test the ToolExecutor dispatch for ramsey/t1 tools."""

    @pytest.fixture
    def executor(self):
        from backends.base import ShadowBackend
        from tools.quantum_tools import ToolExecutor

        # We need a ShadowBackend for the ToolExecutor, but the
        # ramsey/t1 tools create their own DynamicsLabAdapter internally
        # when the backend isn't pulse-capable.
        from backends.fake_adapter import FakeBackendAdapter
        backend = FakeBackendAdapter("FakeBrisbane")
        return ToolExecutor(backend)

    def test_ramsey_experiment_tool(self, executor):
        """Tool call 'ramsey_experiment' should succeed end-to-end."""
        result_json = executor.execute("ramsey_experiment", {
            "n_points": 15,
            "delay_max_ns": 2000,
            "shots": 256,
        })
        result = json.loads(result_json)

        assert "error" not in result, f"Tool returned error: {result}"
        assert result["experiment"] == "ramsey"
        assert len(result["delays_ns"]) == 15
        assert len(result["populations"]) == 15
        assert result["fit"] is not None
        assert "T2_star_us" in result["fit"]

    def test_fit_ramsey_tool(self, executor):
        """Tool call 'fit_ramsey' should fit synthetic Ramsey data."""
        # Generate synthetic damped cosine
        delays = np.linspace(0, 5000, 40)
        t2_star = 2000.0  # ns
        f_det = 0.002  # GHz (= 2 MHz)
        pops = 0.45 * np.cos(2 * np.pi * f_det * delays) * np.exp(-delays / t2_star) + 0.5
        # Add noise
        rng = np.random.default_rng(123)
        pops += rng.normal(0, 0.02, len(pops))
        pops = np.clip(pops, 0, 1)

        result_json = executor.execute("fit_ramsey", {
            "delays_ns": delays.tolist(),
            "populations": pops.tolist(),
            "artificial_detuning_mhz": 2.0,
        })
        result = json.loads(result_json)

        assert "error" not in result, f"Tool returned error: {result}"
        assert result["fit_successful"] is True
        # T2* should be close to 2.0 us
        assert 1.0 < result["T2_star_us"] < 4.0

    def test_t1_experiment_tool(self, executor):
        """Tool call 't1_experiment' should succeed end-to-end."""
        result_json = executor.execute("t1_experiment", {
            "n_points": 15,
            "delay_max_ns": 400000,
            "shots": 256,
        })
        result = json.loads(result_json)

        assert "error" not in result, f"Tool returned error: {result}"
        assert result["experiment"] == "t1"
        assert len(result["delays_ns"]) == 15
        assert len(result["populations"]) == 15
        assert result["fit"] is not None
        assert "T1_us" in result["fit"]

    def test_fit_t1_tool(self, executor):
        """Tool call 'fit_t1' should fit synthetic exponential decay."""
        delays = np.linspace(0, 600000, 40)  # 0 to 600 us in ns
        t1 = 200000.0  # 200 us in ns
        pops = 0.9 * np.exp(-delays / t1) + 0.05
        rng = np.random.default_rng(456)
        pops += rng.normal(0, 0.01, len(pops))
        pops = np.clip(pops, 0, 1)

        result_json = executor.execute("fit_t1", {
            "delays_ns": delays.tolist(),
            "populations": pops.tolist(),
        })
        result = json.loads(result_json)

        assert "error" not in result, f"Tool returned error: {result}"
        assert result["fit_successful"] is True
        # T1 should be close to 200 us
        assert 100.0 < result["T1_us"] < 400.0

    def test_unknown_tool_raises(self, executor):
        """Unknown tool name should produce an error in JSON."""
        result_json = executor.execute("nonexistent_tool", {})
        result = json.loads(result_json)
        assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
