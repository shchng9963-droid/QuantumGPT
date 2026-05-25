"""
Active Learning Experiment Designer for quantum calibration.

Uses Bayesian Optimization (via BoTorch / GPyTorch) to decide
what experiment to run next in a tune-up sequence. The designer
maintains a Gaussian Process surrogate over calibration parameter
space and recommends the next measurement point that maximally
reduces uncertainty or improves a target metric (fidelity, leakage, EPC).

Supported experiment types:
  - rabi_sweep      : optimize π-pulse amplitude
  - drag_sweep      : optimize DRAG α parameter
  - freq_sweep      : optimize qubit drive frequency
  - joint_calib     : optimize multiple parameters jointly

Reference: Snoek et al., NIPS 2012 (BO);
           BoTorch: Balandat et al., NeurIPS 2020
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

# Lazy imports for heavy dependencies
_BOTORCH_AVAILABLE = None


def _check_botorch():
    global _BOTORCH_AVAILABLE
    if _BOTORCH_AVAILABLE is None:
        try:
            import botorch
            import gpytorch
            _BOTORCH_AVAILABLE = True
        except ImportError:
            _BOTORCH_AVAILABLE = False
    return _BOTORCH_AVAILABLE


# ---------- Data classes ----------

@dataclass
class ExperimentPoint:
    """A single observed experiment."""
    parameters: dict[str, float]   # e.g. {"drag_alpha": 0.3, "pi_amp": 0.005}
    objective: float               # metric value (higher = better, e.g. fidelity)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DesignConfig:
    """Configuration for the experiment designer.

    Parameters
    ----------
    parameter_bounds : dict[str, tuple[float, float]]
        Bounds for each parameter to optimize.
    objective_name : str
        Name of the metric to maximize. Default "fidelity".
    n_candidates : int
        Number of candidate points to evaluate for acquisition. Default 512.
    seed : int | None
        RNG seed. Default None.
    acquisition : str
        Acquisition function: "ei" (Expected Improvement), "ucb" (Upper Confidence Bound),
        "thompson" (Thompson Sampling). Default "ei".
    exploration_weight : float
        For UCB: trade-off between exploration and exploitation. Default 2.0.
    """

    parameter_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    objective_name: str = "fidelity"
    n_candidates: int = 512
    seed: int | None = None
    acquisition: Literal["ei", "ucb", "thompson"] = "ei"
    exploration_weight: float = 2.0


@dataclass
class DesignSuggestion:
    """Suggested next experiment from the designer."""
    parameters: dict[str, float]
    acquisition_value: float
    predicted_mean: float
    predicted_std: float
    rationale: str


@dataclass
class DesignerState:
    """State of the experiment designer (serializable)."""
    observations: list[ExperimentPoint]
    config: DesignConfig
    n_suggestions_made: int = 0
    best_objective: float = float("-inf")
    best_parameters: dict[str, float] = field(default_factory=dict)


# ---------- Core designer ----------

class ActiveLearningDesigner:
    """Bayesian Optimization-based experiment designer.

    Usage:
        designer = ActiveLearningDesigner(config)
        designer.add_observation(params, objective)
        suggestion = designer.suggest_next()
    """

    def __init__(self, config: DesignConfig):
        self.config = config
        self.observations: list[ExperimentPoint] = []
        self.n_suggestions = 0
        self._param_names = sorted(config.parameter_bounds.keys())
        self._bounds = np.array([config.parameter_bounds[k] for k in self._param_names])
        self._rng = np.random.default_rng(config.seed)

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    @property
    def best_observation(self) -> ExperimentPoint | None:
        if not self.observations:
            return None
        return max(self.observations, key=lambda x: x.objective)

    def add_observation(
        self,
        parameters: dict[str, float],
        objective: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an experimental observation."""
        self.observations.append(ExperimentPoint(
            parameters=parameters,
            objective=objective,
            metadata=metadata or {},
        ))

    def suggest_next(self) -> DesignSuggestion:
        """Suggest the next experiment to run.

        If fewer than 3 observations, uses Latin Hypercube Sampling (space-filling).
        Otherwise, fits a GP and optimizes the acquisition function.
        """
        if self.n_observations < 3:
            return self._suggest_space_filling()

        if _check_botorch():
            return self._suggest_botorch()
        else:
            return self._suggest_fallback()

    def suggest_batch(self, batch_size: int = 3) -> list[DesignSuggestion]:
        """Suggest a batch of experiments (for parallel execution)."""
        suggestions = []
        for _ in range(batch_size):
            s = self.suggest_next()
            suggestions.append(s)
            # Add as phantom observation to diversify batch
            self.add_observation(s.parameters, s.predicted_mean,
                               metadata={"phantom": True})
        # Remove phantom observations
        self.observations = [o for o in self.observations
                           if not o.metadata.get("phantom", False)]
        return suggestions

    def get_state(self) -> DesignerState:
        """Serialize current state."""
        return DesignerState(
            observations=list(self.observations),
            config=self.config,
            n_suggestions_made=self.n_suggestions,
            best_objective=self.best_observation.objective if self.best_observation else float("-inf"),
            best_parameters=self.best_observation.parameters if self.best_observation else {},
        )

    # --- Internal methods ---

    def _params_to_array(self, params: dict[str, float]) -> NDArray:
        """Convert parameter dict to normalized [0,1]^d array."""
        arr = np.array([params[k] for k in self._param_names])
        return (arr - self._bounds[:, 0]) / (self._bounds[:, 1] - self._bounds[:, 0])

    def _array_to_params(self, arr: NDArray) -> dict[str, float]:
        """Convert normalized array back to parameter dict."""
        denorm = arr * (self._bounds[:, 1] - self._bounds[:, 0]) + self._bounds[:, 0]
        return {k: float(v) for k, v in zip(self._param_names, denorm)}

    def _suggest_space_filling(self) -> DesignSuggestion:
        """Space-filling design for initial exploration."""
        d = len(self._param_names)
        # Random point in [0,1]^d
        x = self._rng.random(d)
        params = self._array_to_params(x)

        self.n_suggestions += 1
        return DesignSuggestion(
            parameters=params,
            acquisition_value=0.0,
            predicted_mean=0.0,
            predicted_std=1.0,
            rationale=(
                f"Initial exploration (observation {self.n_observations + 1}/3). "
                "Using space-filling design to build the GP surrogate."
            ),
        )

    def _suggest_botorch(self) -> DesignSuggestion:
        """Use BoTorch GP + acquisition function."""
        import torch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import (
            LogExpectedImprovement,
            UpperConfidenceBound,
        )
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

        # Prepare training data (normalized to [0,1]^d)
        X = torch.from_numpy(np.array(
            [self._params_to_array(o.parameters) for o in self.observations]
        )).to(dtype=torch.double)
        Y = torch.from_numpy(np.array(
            [[o.objective] for o in self.observations]
        )).to(dtype=torch.double)

        # Standardize Y
        y_mean = Y.mean()
        y_std = Y.std().clamp(min=1e-6)
        Y_std = (Y - y_mean) / y_std

        # Fit GP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = SingleTaskGP(X, Y_std)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                pass  # use default hyperparameters if fitting fails

        # Build acquisition function
        best_f = Y_std.max()
        if self.config.acquisition == "ei":
            acqf = LogExpectedImprovement(gp, best_f=best_f)
        elif self.config.acquisition == "ucb":
            acqf = UpperConfidenceBound(gp, beta=self.config.exploration_weight)
        else:  # thompson
            acqf = LogExpectedImprovement(gp, best_f=best_f)  # fallback to EI

        # Optimize acquisition
        d = len(self._param_names)
        bounds = torch.zeros(2, d, dtype=torch.double)
        bounds[1] = 1.0  # normalized bounds

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            candidate, acq_value = optimize_acqf(
                acqf,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=self.config.n_candidates,
            )

        x_best = candidate.squeeze().detach().numpy()
        params = self._array_to_params(x_best)

        # Get GP prediction at the suggested point
        gp.eval()
        with torch.no_grad():
            posterior = gp.posterior(candidate)
            pred_mean_std = posterior.mean.item() * y_std.item() + y_mean.item()
            pred_std = posterior.variance.sqrt().item() * y_std.item()

        self.n_suggestions += 1
        return DesignSuggestion(
            parameters=params,
            acquisition_value=float(acq_value),
            predicted_mean=pred_mean_std,
            predicted_std=pred_std,
            rationale=(
                f"BO suggestion #{self.n_suggestions} using {self.config.acquisition.upper()}. "
                f"GP predicted {self.config.objective_name} = {pred_mean_std:.4f} ± {pred_std:.4f}. "
                f"Current best: {self.best_observation.objective:.4f}."
            ),
        )

    def _suggest_fallback(self) -> DesignSuggestion:
        """Fallback when BoTorch is not available: use random search + exploitation."""
        d = len(self._param_names)
        best = self.best_observation

        # Mix of exploitation (near best) and exploration (random)
        if self._rng.random() < 0.3:
            # Exploration
            x = self._rng.random(d)
        else:
            # Exploitation: perturb best parameters
            x_best = self._params_to_array(best.parameters)
            x = x_best + self._rng.normal(0, 0.1, d)
            x = np.clip(x, 0, 1)

        params = self._array_to_params(x)
        self.n_suggestions += 1
        return DesignSuggestion(
            parameters=params,
            acquisition_value=0.0,
            predicted_mean=best.objective,
            predicted_std=0.0,
            rationale=(
                f"Fallback suggestion #{self.n_suggestions} (BoTorch not available). "
                "Using random perturbation of best known parameters."
            ),
        )


# ---------- Convenience functions ----------

def create_rabi_designer(
    amp_range: tuple[float, float] = (0.0, 0.01),
    seed: int | None = None,
) -> ActiveLearningDesigner:
    """Create a designer for Rabi amplitude optimization."""
    config = DesignConfig(
        parameter_bounds={"pi_amp_ghz": amp_range},
        objective_name="P1_at_target",
        seed=seed,
    )
    return ActiveLearningDesigner(config)


def create_drag_designer(
    alpha_range: tuple[float, float] = (-2.0, 2.0),
    seed: int | None = None,
) -> ActiveLearningDesigner:
    """Create a designer for DRAG α optimization."""
    config = DesignConfig(
        parameter_bounds={"drag_alpha": alpha_range},
        objective_name="neg_leakage",
        seed=seed,
    )
    return ActiveLearningDesigner(config)


def create_joint_designer(
    bounds: dict[str, tuple[float, float]] | None = None,
    seed: int | None = None,
) -> ActiveLearningDesigner:
    """Create a designer for joint multi-parameter calibration."""
    if bounds is None:
        bounds = {
            "pi_amp_ghz": (0.001, 0.01),
            "drag_alpha": (-2.0, 2.0),
            "freq_offset_mhz": (-1.0, 1.0),
        }
    config = DesignConfig(
        parameter_bounds=bounds,
        objective_name="gate_fidelity",
        seed=seed,
    )
    return ActiveLearningDesigner(config)
