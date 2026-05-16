"""Unit tests for dynamics/rabi.py — Rabi oscillation simulator."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dynamics.rabi import (
    RabiConfig,
    RabiResult,
    RabiSweepResult,
    simulate_rabi,
    sweep_rabi,
    estimate_pi_pulse,
    _make_envelope_fn,
)


# ---------- Config tests ----------

class TestRabiConfig:
    def test_defaults(self):
        cfg = RabiConfig()
        assert cfg.qubit_freq_ghz == 5.0
        assert cfg.pulse_shape == "gaussian"
        assert cfg.n_amps == 51

    def test_detuning_on_resonance(self):
        cfg = RabiConfig(qubit_freq_ghz=5.0, drive_freq_ghz=None)
        assert cfg.detuning_ghz == 0.0
        assert cfg.drive_freq == 5.0

    def test_detuning_off_resonance(self):
        cfg = RabiConfig(qubit_freq_ghz=5.0, drive_freq_ghz=5.1)
        assert abs(cfg.detuning_ghz - 0.1) < 1e-10


# ---------- Envelope tests ----------

class TestEnvelope:
    def test_square_constant(self):
        cfg = RabiConfig(pulse_duration_ns=100, pulse_shape="square")
        fn = _make_envelope_fn(cfg, 0.05)
        assert fn(0) == 0.05
        assert fn(50) == 0.05
        assert fn(100) == 0.05

    def test_gaussian_peak(self):
        cfg = RabiConfig(pulse_duration_ns=100, pulse_shape="gaussian", sigma_ns=25)
        fn = _make_envelope_fn(cfg, 0.05)
        # Peak at t=50ns (center)
        assert abs(fn(50) - 0.05) < 1e-14
        # Tails should be < peak
        assert fn(0) < fn(50)
        assert fn(100) < fn(50)
        # Symmetric
        assert abs(fn(25) - fn(75)) < 1e-14

    def test_invalid_shape(self):
        cfg = RabiConfig(pulse_shape="triangle")
        with pytest.raises(ValueError, match="Unknown pulse shape"):
            _make_envelope_fn(cfg, 0.05)


# ---------- Single simulation tests ----------

class TestSimulateRabi:
    def test_zero_amplitude(self):
        """Zero drive → state stays |0⟩."""
        cfg = RabiConfig(pulse_duration_ns=100, pulse_shape="square", dt_ns=1.0)
        r = simulate_rabi(0.0, cfg)
        assert r.final_p1 < 1e-10
        assert r.amplitude == 0.0

    def test_square_analytic(self):
        """Square pulse matches sin²(Ω·t/2) exactly."""
        cfg = RabiConfig(pulse_duration_ns=100, pulse_shape="square", dt_ns=0.5)
        amp = 0.02
        r = simulate_rabi(amp, cfg)
        analytic = np.sin(amp * r.times_ns / 2) ** 2
        np.testing.assert_allclose(r.populations, analytic, atol=1e-8)

    def test_pi_pulse_square(self):
        """Square π-pulse gives P(|1⟩) ≈ 1."""
        T = 100
        pi_amp = np.pi / T
        cfg = RabiConfig(pulse_duration_ns=T, pulse_shape="square", dt_ns=0.1)
        r = simulate_rabi(pi_amp, cfg)
        assert r.final_p1 > 0.9999

    def test_2pi_pulse_returns(self):
        """2π-pulse returns to |0⟩."""
        T = 100
        two_pi_amp = 2 * np.pi / T
        cfg = RabiConfig(pulse_duration_ns=T, pulse_shape="square", dt_ns=0.1)
        r = simulate_rabi(two_pi_amp, cfg)
        assert r.final_p1 < 1e-6

    def test_result_shapes(self):
        """Check output shapes are consistent."""
        cfg = RabiConfig(pulse_duration_ns=50, pulse_shape="square", dt_ns=1.0)
        r = simulate_rabi(0.01, cfg)
        assert r.statevectors.shape[1] == 2
        assert len(r.populations) == len(r.times_ns)
        assert r.statevectors.shape[0] == len(r.times_ns)

    def test_unitarity(self):
        """State stays normalised throughout evolution."""
        cfg = RabiConfig(pulse_duration_ns=100, pulse_shape="gaussian", dt_ns=0.5)
        r = simulate_rabi(0.05, cfg)
        norms = np.abs(r.statevectors[:, 0]) ** 2 + np.abs(r.statevectors[:, 1]) ** 2
        np.testing.assert_allclose(norms, 1.0, atol=1e-8)


# ---------- Detuned drive tests ----------

class TestDetuned:
    def test_detuned_reduces_max_population(self):
        """Off-resonance drive can't fully excite the qubit."""
        T = 100
        pi_amp = np.pi / T  # Would be π on resonance
        cfg_on = RabiConfig(pulse_duration_ns=T, pulse_shape="square", dt_ns=0.5)
        cfg_off = RabiConfig(
            pulse_duration_ns=T, pulse_shape="square", dt_ns=0.5,
            drive_freq_ghz=5.0 + 0.02,  # 20 MHz detuning
        )
        r_on = simulate_rabi(pi_amp, cfg_on)
        r_off = simulate_rabi(pi_amp, cfg_off)
        assert r_off.final_p1 < r_on.final_p1

    def test_large_detuning_suppresses(self):
        """Very large detuning → almost no excitation."""
        cfg = RabiConfig(
            pulse_duration_ns=100, pulse_shape="square", dt_ns=1.0,
            drive_freq_ghz=5.0 + 1.0,  # 1 GHz detuning
        )
        r = simulate_rabi(0.01, cfg)
        assert r.final_p1 < 0.01


# ---------- Sweep tests ----------

class TestSweepRabi:
    def test_sweep_basic(self):
        cfg = RabiConfig(
            pulse_duration_ns=100, pulse_shape="square", dt_ns=1.0,
            amp_range=(0, 0.05), n_amps=11,
        )
        sw = sweep_rabi(cfg)
        assert len(sw.amplitudes) == 11
        assert len(sw.final_populations) == 11
        assert len(sw.results) == 11

    def test_sweep_finds_pi_amp(self):
        """Sweep should find π-amp close to analytic value."""
        T = 100
        cfg = RabiConfig(
            pulse_duration_ns=T, pulse_shape="square", dt_ns=0.5,
            amp_range=(0, 0.05), n_amps=51,
        )
        sw = sweep_rabi(cfg)
        analytic_pi = np.pi / T
        assert abs(sw.pi_amplitude - analytic_pi) < 0.003  # within ~3 MHz

    def test_sweep_monotonic_start(self):
        """At small amplitudes, P(|1⟩) should increase monotonically."""
        cfg = RabiConfig(
            pulse_duration_ns=100, pulse_shape="square", dt_ns=1.0,
            amp_range=(0, 0.015), n_amps=11,
        )
        sw = sweep_rabi(cfg)
        # First few should be increasing (before π-amp)
        for i in range(1, 5):
            assert sw.final_populations[i] >= sw.final_populations[i - 1] - 1e-10


# ---------- Pi-pulse estimation ----------

class TestEstimatePiPulse:
    def test_estimate_square(self):
        T = 100
        cfg = RabiConfig(
            pulse_duration_ns=T, pulse_shape="square", dt_ns=0.5,
            amp_range=(0, 0.05), n_amps=21,
        )
        result = estimate_pi_pulse(cfg, refine=True)
        analytic_pi = np.pi / T
        assert abs(result["pi_amp"] - analytic_pi) < 0.001
        assert result["pi_fidelity"] > 0.999
        assert result["fine_sweep"] is not None

    def test_estimate_no_refine(self):
        cfg = RabiConfig(
            pulse_duration_ns=100, pulse_shape="square", dt_ns=1.0,
            amp_range=(0, 0.05), n_amps=11,
        )
        result = estimate_pi_pulse(cfg, refine=False)
        assert result["fine_sweep"] is None
        assert result["pi_amp"] > 0

    def test_half_pi_is_half(self):
        cfg = RabiConfig(
            pulse_duration_ns=100, pulse_shape="square", dt_ns=1.0,
            amp_range=(0, 0.05), n_amps=11,
        )
        result = estimate_pi_pulse(cfg, refine=False)
        assert abs(result["half_pi_amp"] - result["pi_amp"] / 2) < 1e-14
