"""Unit tests for dynamics/active_learning.py — Active Learning experiment designer."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dynamics.active_learning import (
    ActiveLearningDesigner,
    DesignConfig,
    ExperimentPoint,
    DesignSuggestion,
    create_rabi_designer,
    create_drag_designer,
    create_joint_designer,
)


# ---------- Config tests ----------

class TestDesignConfig:
    def test_defaults(self):
        cfg = DesignConfig(parameter_bounds={"x": (0, 1)})
        assert cfg.objective_name == "fidelity"
        assert cfg.acquisition == "ei"
        assert cfg.n_candidates == 512


# ---------- Designer basic tests ----------

class TestDesignerBasic:
    def test_create_empty(self):
        cfg = DesignConfig(parameter_bounds={"x": (0, 1)})
        d = ActiveLearningDesigner(cfg)
        assert d.n_observations == 0
        assert d.best_observation is None

    def test_add_observation(self):
        cfg = DesignConfig(parameter_bounds={"x": (0, 1)})
        d = ActiveLearningDesigner(cfg)
        d.add_observation({"x": 0.5}, 0.8)
        assert d.n_observations == 1
        assert d.best_observation.objective == 0.8

    def test_best_observation(self):
        cfg = DesignConfig(parameter_bounds={"x": (0, 1)})
        d = ActiveLearningDesigner(cfg)
        d.add_observation({"x": 0.3}, 0.5)
        d.add_observation({"x": 0.7}, 0.9)
        d.add_observation({"x": 0.1}, 0.6)
        assert d.best_observation.parameters["x"] == 0.7

    def test_get_state(self):
        cfg = DesignConfig(parameter_bounds={"x": (0, 1)})
        d = ActiveLearningDesigner(cfg)
        d.add_observation({"x": 0.5}, 0.8)
        state = d.get_state()
        assert state.n_suggestions_made == 0
        assert state.best_objective == 0.8


# ---------- Space-filling (< 3 observations) ----------

class TestSpaceFilling:
    def test_initial_suggestion(self):
        cfg = DesignConfig(parameter_bounds={"x": (0, 1)}, seed=42)
        d = ActiveLearningDesigner(cfg)
        s = d.suggest_next()
        assert isinstance(s, DesignSuggestion)
        assert "x" in s.parameters
        assert 0 <= s.parameters["x"] <= 1
        assert "exploration" in s.rationale.lower()

    def test_multi_param_space_filling(self):
        cfg = DesignConfig(
            parameter_bounds={"a": (-2, 2), "b": (0, 0.01)},
            seed=42,
        )
        d = ActiveLearningDesigner(cfg)
        s = d.suggest_next()
        assert -2 <= s.parameters["a"] <= 2
        assert 0 <= s.parameters["b"] <= 0.01


# ---------- BO suggestions (>= 3 observations) ----------

class TestBOSuggestions:
    def test_bo_suggestion_1d(self):
        """With 4+ observations, BO should kick in."""
        cfg = DesignConfig(parameter_bounds={"x": (0, 1)}, seed=42)
        d = ActiveLearningDesigner(cfg)
        # Add some observations
        for x, y in [(0.1, 0.3), (0.5, 0.9), (0.9, 0.4), (0.3, 0.7)]:
            d.add_observation({"x": x}, y)
        s = d.suggest_next()
        assert 0 <= s.parameters["x"] <= 1
        assert s.predicted_std >= 0  # uncertainty should be non-negative
        assert "BO suggestion" in s.rationale or "Fallback" in s.rationale

    def test_bo_suggestion_2d(self):
        """BO works in 2D parameter space."""
        cfg = DesignConfig(
            parameter_bounds={"alpha": (-2, 2), "amp": (0.001, 0.01)},
            seed=42,
        )
        d = ActiveLearningDesigner(cfg)
        # Synthetic observations
        for alpha, amp, obj in [
            (0, 0.005, 0.9), (1, 0.005, 0.7),
            (-1, 0.005, 0.8), (0, 0.003, 0.85),
        ]:
            d.add_observation({"alpha": alpha, "amp": amp}, obj)
        s = d.suggest_next()
        assert -2 <= s.parameters["alpha"] <= 2
        assert 0.001 <= s.parameters["amp"] <= 0.01

    def test_bo_converges_to_optimum(self):
        """BO should move toward the optimum over iterations."""
        cfg = DesignConfig(parameter_bounds={"x": (0, 1)}, seed=42)
        d = ActiveLearningDesigner(cfg)

        # True optimum at x=0.6
        def true_obj(x):
            return -((x - 0.6) ** 2) + 1.0

        # Initial observations
        for x in [0.1, 0.5, 0.9]:
            d.add_observation({"x": x}, true_obj(x))

        # Run 5 BO iterations
        for _ in range(5):
            s = d.suggest_next()
            x = s.parameters["x"]
            d.add_observation({"x": x}, true_obj(x))

        # Best should be close to 0.6
        best = d.best_observation
        assert abs(best.parameters["x"] - 0.6) < 0.2


# ---------- Convenience constructors ----------

class TestConvenienceConstructors:
    def test_rabi_designer(self):
        d = create_rabi_designer(amp_range=(0, 0.01), seed=42)
        assert "pi_amp_ghz" in d.config.parameter_bounds
        s = d.suggest_next()
        assert 0 <= s.parameters["pi_amp_ghz"] <= 0.01

    def test_drag_designer(self):
        d = create_drag_designer(alpha_range=(-1, 1), seed=42)
        assert "drag_alpha" in d.config.parameter_bounds
        s = d.suggest_next()
        assert -1 <= s.parameters["drag_alpha"] <= 1

    def test_joint_designer(self):
        d = create_joint_designer(seed=42)
        assert len(d.config.parameter_bounds) == 3
        s = d.suggest_next()
        assert "pi_amp_ghz" in s.parameters
        assert "drag_alpha" in s.parameters
        assert "freq_offset_mhz" in s.parameters


# ---------- Tool executor integration ----------

class TestActiveLearningToolExecutor:
    def test_next_best_experiment_tool(self):
        """Test next_best_experiment through the ToolExecutor interface."""
        from backends.synthetic_drift import SyntheticDriftBackend, STABLE
        from tools.quantum_tools import ToolExecutor
        import json

        be = SyntheticDriftBackend(profile=STABLE)
        ex = ToolExecutor(be)
        result = json.loads(ex.execute("next_best_experiment", {
            "parameter_bounds": {"drag_alpha": [-2, 2]},
            "observations": [
                {"parameters": {"drag_alpha": -1.0}, "objective": 0.7},
                {"parameters": {"drag_alpha": 0.0}, "objective": 0.95},
                {"parameters": {"drag_alpha": 1.0}, "objective": 0.8},
                {"parameters": {"drag_alpha": 0.5}, "objective": 0.9},
            ],
            "objective_name": "neg_leakage",
            "seed": 42,
        }))
        assert "suggested_parameters" in result
        assert "drag_alpha" in result["suggested_parameters"]
        assert "predicted_objective" in result
        assert "rationale" in result
        assert "recommendation" in result
        # Suggested α should be in bounds
        assert -2 <= result["suggested_parameters"]["drag_alpha"] <= 2
