"""Experiment protocols for quantum hardware characterization.

Each protocol follows a unified interface:
    configure() -> parameters
    run(backend) -> raw data
    analyze(raw) -> fit results
    visualize(result) -> figure
"""

from experiments.base import ExperimentProtocol, ExperimentResult, FitResult
from experiments.rabi_experiment import RabiExperiment
from experiments.ramsey_experiment import RamseyExperiment
from experiments.t1_experiment import T1Experiment

__all__ = [
    "ExperimentProtocol",
    "ExperimentResult",
    "FitResult",
    "RabiExperiment",
    "RamseyExperiment",
    "T1Experiment",
]
