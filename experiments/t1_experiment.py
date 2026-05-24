"""T1 relaxation experiment protocol.

Measures energy relaxation time by exciting the qubit with a pi-pulse
and measuring decay as a function of wait time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from experiments.base import ExperimentProtocol, ExperimentResult, FitResult
from backends.lab_backend import LabBackend, PulseSchedule


def _t1_model(t: np.ndarray, A: float, T1: float, B: float) -> np.ndarray:
    """T1 decay model: A * exp(-t/T1) + B"""
    return A * np.exp(-t / T1) + B


@dataclass
class T1Experiment(ExperimentProtocol):
    """T1 energy relaxation experiment.

    Sequence: pi-pulse -- delay(tau) -- measure
    Sweeps tau to observe exponential decay from |1> to |0>.
    """

    qubit: int = 0
    delay_min_ns: float = 0.0
    delay_max_ns: float = 500_000.0  # 500 us
    n_points: int = 50
    pi_amplitude: float = 0.03  # calibrated pi-pulse amplitude
    pi_duration_ns: float = 100.0
    shots: int = 1024

    @property
    def name(self) -> str:
        return "t1"

    def run(self, backend: LabBackend) -> ExperimentResult:
        """Sweep delay after pi-pulse and measure P(|1>)."""
        delays = np.linspace(self.delay_min_ns, self.delay_max_ns, self.n_points)
        populations = np.zeros(self.n_points)
        errors = np.zeros(self.n_points)
        raw_counts_list = []

        for i, tau in enumerate(delays):
            schedule = PulseSchedule(
                channels={
                    "d0": [
                        # Pi-pulse to excite to |1>
                        {
                            "amplitude": float(self.pi_amplitude),
                            "duration_ns": self.pi_duration_ns,
                            "shape": "gaussian",
                            "phase": 0.0,
                        },
                        # Wait (zero amplitude)
                        {
                            "amplitude": 0.0,
                            "duration_ns": float(tau),
                            "shape": "square",
                            "phase": 0.0,
                        },
                    ]
                },
                duration_ns=self.pi_duration_ns + tau,
                metadata={"shots": self.shots},
            )

            result = backend.dispatch_pulse(schedule)
            counts = result.counts or {"0": self.shots}
            raw_counts_list.append(counts)

            n1 = counts.get("1", 0)
            total = sum(counts.values())
            p1 = n1 / total if total > 0 else 0.0
            populations[i] = p1
            errors[i] = np.sqrt(p1 * (1 - p1) / total) if total > 0 else 0.0

        return ExperimentResult(
            protocol_name=self.name,
            sweep_parameter="delay_ns",
            sweep_values=delays,
            measured_values=populations,
            measured_errors=errors,
            raw_counts=raw_counts_list,
            metadata={
                "qubit": self.qubit,
                "pi_amplitude": self.pi_amplitude,
                "pi_duration_ns": self.pi_duration_ns,
                "shots": self.shots,
            },
        )

    def analyze(self, result: ExperimentResult) -> ExperimentResult:
        """Fit exponential decay to extract T1."""
        x = result.sweep_values  # ns
        y = result.measured_values

        # Initial guesses
        A0 = y[0] - y[-1]  # should start high, end low
        B0 = y[-1]
        T1_0 = x[-1] / 3  # guess: decays over 1/3 of range

        try:
            popt, pcov = curve_fit(
                _t1_model, x, y,
                p0=[A0, T1_0, B0],
                bounds=([0, 100, -0.5], [2, 1e7, 1.5]),
                maxfev=5000,
            )
            perr = np.sqrt(np.diag(pcov))

            A, T1_ns, B = popt
            T1_us = T1_ns / 1000.0

            y_fit = _t1_model(x, *popt)
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            fit = FitResult(
                parameters={
                    "T1_us": float(T1_us),
                    "T1_ns": float(T1_ns),
                    "amplitude": float(A),
                    "offset": float(B),
                },
                uncertainties={
                    "T1_ns": float(perr[1]),
                    "T1_us": float(perr[1] / 1000.0),
                    "amplitude": float(perr[0]),
                },
                r_squared=float(r_squared),
                model="A * exp(-t/T1) + B",
                residuals=y - y_fit,
            )
            result.fit = fit

        except (RuntimeError, ValueError):
            # Fallback: estimate from 1/e point
            if y[0] > 0:
                target = y[0] / np.e
                idx = np.argmax(y < target) if np.any(y < target) else len(x) // 3
                T1_est = x[idx] if idx > 0 else x[-1] / 3
            else:
                T1_est = x[-1] / 3

            result.fit = FitResult(
                parameters={"T1_ns": float(T1_est), "T1_us": float(T1_est / 1000)},
                r_squared=0.0,
                model="1/e_fallback",
            )

        return result

    def visualize(self, result: ExperimentResult, save_path: str | None = None) -> Any:
        """Plot T1 decay with fit."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        x = result.sweep_values / 1000.0  # ns -> us
        y = result.measured_values

        # Data
        if result.measured_errors is not None:
            ax.errorbar(x, y, yerr=result.measured_errors,
                       fmt='o', markersize=4, color='#009688',
                       ecolor='#80CBC4', capsize=2, label='Data')
        else:
            ax.plot(x, y, 'o', markersize=4, color='#009688', label='Data')

        # Fit
        if result.fit and result.fit.r_squared > 0.5:
            p = result.fit.parameters
            x_fine_ns = np.linspace(result.sweep_values[0], result.sweep_values[-1], 200)
            x_fine_us = x_fine_ns / 1000.0
            y_fit = _t1_model(x_fine_ns, p['amplitude'], p['T1_ns'], p['offset'])
            ax.plot(x_fine_us, y_fit, '-', color='#E91E63', linewidth=1.5, label='Fit')

            # Mark T1
            ax.axvline(p['T1_us'], color='#FF9800', linestyle='--', linewidth=1,
                      label=f'$T_1$ = {p["T1_us"]:.1f} μs')

            ax.set_title(
                f'$T_1$ Relaxation (Q{result.metadata.get("qubit", 0)}) — '
                f'$T_1$ = {p["T1_us"]:.1f} μs, '
                f'$R^2$ = {result.fit.r_squared:.3f}',
                fontsize=12,
            )
        else:
            ax.set_title(f'$T_1$ Relaxation (Q{result.metadata.get("qubit", 0)})', fontsize=12)

        ax.set_xlabel('Delay (μs)', fontsize=11)
        ax.set_ylabel('P(|1⟩)', fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
