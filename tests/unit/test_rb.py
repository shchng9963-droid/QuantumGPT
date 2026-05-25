"""Unit tests for dynamics/rb.py — Randomized Benchmarking simulator."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dynamics.rb import (
    RBConfig,
    RBResult,
    get_cliffords_1q,
    run_rb_sequence,
    run_rb,
)


# ---------- Clifford group tests ----------

class TestCliffords:
    def test_24_cliffords(self):
        """Should generate exactly 24 single-qubit Cliffords."""
        clifs = get_cliffords_1q()
        assert len(clifs) == 24

    def test_all_unitary(self):
        """Every Clifford should be unitary."""
        clifs = get_cliffords_1q()
        for i, U in enumerate(clifs):
            product = U @ U.conj().T
            np.testing.assert_allclose(product, np.eye(2), atol=1e-10,
                err_msg=f"Clifford {i} is not unitary")

    def test_all_distinct(self):
        """All 24 Cliffords should be distinct (modulo global phase)."""
        clifs = get_cliffords_1q()
        # Compare by action on |0��� and |+⟩
        fingerprints = set()
        state_0 = np.array([1, 0], dtype=complex)
        state_p = np.array([1, 1], dtype=complex) / np.sqrt(2)
        for U in clifs:
            out0 = U @ state_0
            outp = U @ state_p
            # Normalize phase
            if abs(out0[0]) > 1e-10:
                out0 = out0 * np.conj(out0[0]) / abs(out0[0])
            fp = tuple(np.round(np.concatenate([out0, outp]), 6))
            fingerprints.add(fp)
        assert len(fingerprints) == 24


# ---------- Config tests ----------

class TestRBConfig:
    def test_defaults(self):
        cfg = RBConfig()
        assert cfg.n_sequences == 20
        assert cfg.error_per_gate == 0.001
        assert len(cfg.sequence_lengths) == 7


# ---------- Single sequence tests ----------

class TestRBSequence:
    def test_perfect_gates_high_survival(self):
        """With no errors, survival should be ~1 (only readout error)."""
        cfg = RBConfig(error_per_gate=0.0, readout_error=0.0, shots=0)
        rng = np.random.default_rng(42)
        prob = run_rb_sequence(10, cfg, rng)
        np.testing.assert_allclose(prob, 1.0, atol=1e-8)

    def test_noisy_gates_reduced_survival(self):
        """With high noise, survival should drop."""
        cfg = RBConfig(error_per_gate=0.1, readout_error=0.0, shots=0)
        rng = np.random.default_rng(42)
        # Average over several sequences
        probs = [run_rb_sequence(32, cfg, rng) for _ in range(20)]
        avg = np.mean(probs)
        # With 10% depolarizing over 33 gates (32 + inverse), survival drops
        assert avg < 0.8, f"Expected reduced survival, got {avg}"


# ---------- Full RB tests ----------

class TestRunRB:
    def test_basic_output_shapes(self):
        cfg = RBConfig(
            sequence_lengths=[1, 4, 16, 64],
            n_sequences=5,
            error_per_gate=0.001,
            seed=42,
            shots=256,
        )
        result = run_rb(cfg)
        assert len(result.sequence_lengths) == 4
        assert len(result.survival_probs) == 4
        assert len(result.survival_stds) == 4
        assert len(result.raw_survivals) == 4
        assert result.raw_survivals[0].shape == (5,)

    def test_epc_reasonable(self):
        """EPC should be in a reasonable range for 0.1% gate error."""
        cfg = RBConfig(
            sequence_lengths=[1, 2, 4, 8, 16, 32, 64],
            n_sequences=20,
            error_per_gate=0.001,
            seed=42,
            shots=4096,
        )
        result = run_rb(cfg)
        assert 0 < result.error_per_clifford < 0.01
        assert result.depolarizing_param > 0.99

    def test_decay_with_length(self):
        """Survival should generally decrease with sequence length."""
        cfg = RBConfig(
            sequence_lengths=[1, 4, 16, 64, 256],
            n_sequences=30,
            error_per_gate=0.005,
            seed=42,
            shots=4096,
        )
        result = run_rb(cfg)
        # First point should be higher than last
        assert result.survival_probs[0] > result.survival_probs[-1]

    def test_fit_quality(self):
        """R² should be reasonable for clean data."""
        cfg = RBConfig(
            sequence_lengths=[1, 2, 4, 8, 16, 32, 64],
            n_sequences=30,
            error_per_gate=0.002,
            seed=42,
            shots=4096,
        )
        result = run_rb(cfg)
        assert result.r_squared > 0.7  # Should have decent fit

    def test_zero_error_high_fidelity(self):
        """With no gate errors, EPC should be very small."""
        cfg = RBConfig(
            sequence_lengths=[1, 4, 16, 64],
            n_sequences=10,
            error_per_gate=0.0,
            readout_error=0.0,
            seed=42,
            shots=0,  # no shot noise
        )
        result = run_rb(cfg)
        # All survivals should be ~1
        np.testing.assert_allclose(result.survival_probs, 1.0, atol=1e-8)

    def test_reproducibility(self):
        """Same seed should give same results."""
        cfg = RBConfig(
            sequence_lengths=[1, 8, 64],
            n_sequences=10,
            error_per_gate=0.001,
            seed=123,
            shots=1024,
        )
        r1 = run_rb(cfg)
        r2 = run_rb(cfg)
        np.testing.assert_array_equal(r1.survival_probs, r2.survival_probs)


# ---------- Tool executor integration ----------

class TestRBToolExecutor:
    def test_rb_tool_via_executor(self):
        """Test RB through the ToolExecutor interface."""
        from backends.synthetic_drift import SyntheticDriftBackend, STABLE
        from tools.quantum_tools import ToolExecutor
        import json

        be = SyntheticDriftBackend(profile=STABLE)
        ex = ToolExecutor(be)
        result = json.loads(ex.execute("randomized_benchmarking", {
            "sequence_lengths": [1, 4, 16],
            "n_sequences": 5,
            "shots": 256,
            "seed": 42,
        }))
        assert "error_per_clifford" in result
        assert "gate_fidelity" in result
        assert "depolarizing_param" in result
        assert "recommendation" in result
        assert result["gate_fidelity"] > 0.9
