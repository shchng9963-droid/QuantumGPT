"""Unit tests for dynamics/drag.py — DRAG pulse calibration simulator."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dynamics.drag import (
    DRAGConfig,
    DRAGResult,
    DRAGSweepResult,
    simulate_drag,
    sweep_drag,
    calibrate_drag,
)


# ---------- Config tests ----------

class TestDRAGConfig:
    def test_defaults(self):
        cfg = DRAGConfig()
        assert cfg.anharmonicity_ghz == -0.3
        assert cfg.pulse_shape == "gaussian"
        assert cfg.n_alphas == 21

    def test_custom_range(self):
        cfg = DRAGConfig(alpha_range=(-1.0, 1.0), n_alphas=11)
        assert cfg.alpha_range == (-1.0, 1.0)
        assert cfg.n_alphas == 11


# ---------- Single simulation tests ----------

class TestSimulateDRAG:
    def test_zero_alpha_has_leakage(self):
        """Without DRAG (α=0), Gaussian pulse should leak to |2⟩."""
        cfg = DRAGConfig(
            pi_amp_ghz=0.005,
            pulse_duration_ns=100,
            pulse_shape="gaussian",
            anharmonicity_ghz=-0.3,
            n_pulses=4,  # amplify leakage
            dt_ns=0.5,
        )
        r = simulate_drag(0.0, cfg)
        # Should have non-zero leakage (exact value depends on parameters)
        assert r.leakage > 1e-8, f"Expected leakage > 1e-8 at α=0, got {r.leakage}"

    def test_optimal_alpha_suppresses_leakage(self):
        """At the right α, leakage should be much lower than α=0."""
        cfg = DRAGConfig(
            pi_amp_ghz=0.005,
            pulse_duration_ns=100,
            pulse_shape="gaussian",
            anharmonicity_ghz=-0.3,
            n_pulses=2,
            dt_ns=0.5,
        )
        r0 = simulate_drag(0.0, cfg)
        # Sweep to find rough optimum
        sw = sweep_drag(DRAGConfig(
            pi_amp_ghz=0.005,
            pulse_duration_ns=100,
            pulse_shape="gaussian",
            anharmonicity_ghz=-0.3,
            alpha_range=(-1.0, 1.0),
            n_alphas=11,
            n_pulses=2,
            dt_ns=0.5,
        ))
        r_opt = simulate_drag(sw.optimal_alpha, cfg)
        assert r_opt.leakage < r0.leakage

    def test_square_pulse_no_drag_effect(self):
        """Square pulse has zero derivative, so DRAG should have no effect."""
        cfg = DRAGConfig(
            pi_amp_ghz=0.005,
            pulse_duration_ns=100,
            pulse_shape="square",
            anharmonicity_ghz=-0.3,
            n_pulses=2,
            dt_ns=0.5,
        )
        r0 = simulate_drag(0.0, cfg)
        r1 = simulate_drag(1.0, cfg)
        # Same leakage regardless of α (derivative is zero)
        np.testing.assert_allclose(r0.leakage, r1.leakage, atol=1e-10)

    def test_populations_sum_to_one(self):
        """Total population should be conserved (unitary evolution)."""
        cfg = DRAGConfig(
            pi_amp_ghz=0.005,
            pulse_duration_ns=100,
            pulse_shape="gaussian",
            anharmonicity_ghz=-0.3,
            n_pulses=2,
            dt_ns=0.5,
        )
        for alpha in [-1.0, 0.0, 0.5, 1.0]:
            r = simulate_drag(alpha, cfg)
            total = r.population_0 + r.population_1 + r.leakage
            np.testing.assert_allclose(total, 1.0, atol=1e-8,
                err_msg=f"Population not conserved at α={alpha}")


# ---------- Sweep tests ----------

class TestSweepDRAG:
    def test_sweep_basic(self):
        cfg = DRAGConfig(
            pi_amp_ghz=0.005,
            pulse_duration_ns=100,
            pulse_shape="gaussian",
            anharmonicity_ghz=-0.3,
            alpha_range=(-1.0, 1.0),
            n_alphas=11,
            n_pulses=2,
            dt_ns=1.0,
        )
        sw = sweep_drag(cfg)
        assert len(sw.alphas) == 11
        assert len(sw.leakages) == 11
        assert len(sw.results) == 11
        assert sw.min_leakage >= 0.0

    def test_optimal_alpha_finite(self):
        """Optimal α should be a finite number within range."""
        cfg = DRAGConfig(
            alpha_range=(-2.0, 2.0),
            n_alphas=11,
            n_pulses=2,
            dt_ns=1.0,
        )
        sw = sweep_drag(cfg)
        assert np.isfinite(sw.optimal_alpha)
        assert -2.0 <= sw.optimal_alpha <= 2.0


# ---------- Calibration tests ----------

class TestCalibrateDRAG:
    def test_calibrate_basic(self):
        cfg = DRAGConfig(
            pi_amp_ghz=0.005,
            pulse_duration_ns=100,
            pulse_shape="gaussian",
            anharmonicity_ghz=-0.3,
            alpha_range=(-1.0, 1.0),
            n_alphas=11,
            n_pulses=2,
            dt_ns=1.0,
        )
        result = calibrate_drag(cfg, refine=False)
        assert "optimal_alpha" in result
        assert "min_leakage" in result
        assert "gate_fidelity" in result
        assert result["fine_sweep"] is None
        assert result["gate_fidelity"] > 0.99

    def test_calibrate_with_refinement(self):
        cfg = DRAGConfig(
            pi_amp_ghz=0.005,
            pulse_duration_ns=100,
            pulse_shape="gaussian",
            anharmonicity_ghz=-0.3,
            alpha_range=(-1.0, 1.0),
            n_alphas=11,
            n_pulses=2,
            dt_ns=1.0,
        )
        result = calibrate_drag(cfg, refine=True)
        assert result["fine_sweep"] is not None
        # Refined result should be at least as good as coarse
        assert result["min_leakage"] <= result["coarse_sweep"].min_leakage + 1e-10

    def test_gate_fidelity_bound(self):
        """Gate fidelity should be 1 - leakage."""
        cfg = DRAGConfig(
            alpha_range=(-1.0, 1.0),
            n_alphas=11,
            n_pulses=2,
            dt_ns=1.0,
        )
        result = calibrate_drag(cfg, refine=False)
        np.testing.assert_allclose(
            result["gate_fidelity"],
            1.0 - result["min_leakage"],
            atol=1e-12,
        )


# ---------- Tool executor integration ----------

class TestDRAGToolExecutor:
    def test_drag_tool_via_executor(self):
        """Test DRAG calibration through the ToolExecutor interface."""
        from backends.synthetic_drift import SyntheticDriftBackend, STABLE
        from tools.quantum_tools import ToolExecutor
        import json

        be = SyntheticDriftBackend(profile=STABLE)
        ex = ToolExecutor(be)
        result = json.loads(ex.execute("drag_calibration", {
            "pi_amp_ghz": 0.005,
            "anharmonicity_ghz": -0.3,
            "alpha_min": -1.0,
            "alpha_max": 1.0,
            "n_points": 11,
            "refine": False,
        }))
        assert "optimal_alpha" in result
        assert "min_leakage" in result
        assert "gate_fidelity" in result
        assert "recommendation" in result
        assert result["gate_fidelity"] > 0.99
