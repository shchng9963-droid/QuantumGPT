"""Base class for all experiment protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FitResult:
    """Result of fitting experimental data."""

    parameters: dict[str, float]  # e.g. {"pi_amplitude": 0.03, "rabi_freq_ghz": 0.15}
    uncertainties: dict[str, float] = field(default_factory=dict)
    r_squared: float = 0.0
    model: str = ""  # e.g. "A * cos(2*pi*f*x + phi) * exp(-x/tau) + B"
    residuals: np.ndarray | None = None


@dataclass
class ExperimentResult:
    """Complete result of an experiment run."""

    protocol_name: str
    sweep_parameter: str  # e.g. "amplitude", "delay_ns"
    sweep_values: np.ndarray  # x-axis values
    measured_values: np.ndarray  # y-axis values (e.g. P(|1>))
    measured_errors: np.ndarray | None = None  # standard errors
    fit: FitResult | None = None
    raw_counts: list[dict[str, int]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether the experiment produced a valid fit."""
        return self.fit is not None and self.fit.r_squared > 0.8

    def summary(self) -> str:
        """One-line summary of the result."""
        if self.fit:
            params = ", ".join(f"{k}={v:.4g}" for k, v in self.fit.parameters.items())
            return f"{self.protocol_name}: {params} (R²={self.fit.r_squared:.3f})"
        return f"{self.protocol_name}: no valid fit"


class ExperimentProtocol(ABC):
    """Abstract base class for experiment protocols.

    Subclasses implement the four-step workflow:
        1. configure() - set up sweep parameters
        2. run(backend) - execute on hardware
        3. analyze(result) - fit the data
        4. visualize(result) - generate figure

    Usage:
        exp = RabiExperiment(qubit=0, amp_range=(0, 0.1), n_points=50)
        result = exp.run(backend)
        result = exp.analyze(result)
        fig = exp.visualize(result)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Protocol name (e.g. 'rabi', 'ramsey', 't1')."""
        ...

    @abstractmethod
    def run(self, backend: Any) -> ExperimentResult:
        """Execute the experiment on a LabBackend.

        This method handles the full sweep: for each parameter value,
        dispatch the appropriate pulse and collect measurements.
        """
        ...

    @abstractmethod
    def analyze(self, result: ExperimentResult) -> ExperimentResult:
        """Fit the experimental data and populate result.fit."""
        ...

    @abstractmethod
    def visualize(self, result: ExperimentResult, save_path: str | None = None) -> Any:
        """Generate a visualization of the experiment result.

        Returns a matplotlib Figure.
        """
        ...
