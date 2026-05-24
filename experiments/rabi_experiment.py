"""Rabi oscillation experiment protocol.

Sweeps drive amplitude to find the pi-pulse calibration point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from experiments.base import ExperimentProtocol, ExperimentResult, FitResult
from backends.lab_backend import LabBackend, PulseSchedule


def _rabi_model(x: np.ndarray, A: float, f: float, phi: float, B: float) -> np.ndarray:
    """Rabi oscillation model: A * cos(2*pi*f*x + phi) + B"""
    return A * np.cos(2 * np.pi * f * x + phi) + B


@dataclass
class RabiExperiment(ExperimentProtocol):
    """Rabi oscillation experiment.

    Sweeps the drive amplitude at fixed pulse duration to observe
    oscillations between |0> and |1>. Extracts pi-pulse amplitude.
    """

    qubit: int = 0
    amp_min: float = 0.0
    amp_max: float = 0.1
    n_points: int = 50
    pulse_duration_ns: float = 100.0
    pulse_shape: str = "square"
    shots: int = 1024

    @property
    def name(self) -> str:
        return "rabi"

    def run(self, backend: LabBackend) -> ExperimentResult:
        """Sweep amplitude and measure P(|1>)."""
        amplitudes = np.linspace(self.amp_min, self.amp_max, self.n_points)
        populations = np.zeros(self.n_points)
        errors = np.zeros(self.n_points)
        raw_counts_list = []

        for i, amp in enumerate(amplitudes):
            schedule = PulseSchedule(
                channels={
                    "d0": [{
                        "amplitude": float(amp),
                        "duration_ns": self.pulse_duration_ns,
                        "shape": self.pulse_shape,
                        "phase": 0.0,
                    }]
                },
                duration_ns=self.pulse_duration_ns,
                metadata={"shots": self.shots},
            )

            result = backend.dispatch_pulse(schedule)
            counts = result.counts or {"0": self.shots}
            raw_counts_list.append(counts)

            n1 = counts.get("1", 0)
            total = sum(counts.values())
            p1 = n1 / total if total > 0 else 0.0
            populations[i] = p1
            # Binomial standard error
            errors[i] = np.sqrt(p1 * (1 - p1) / total) if total > 0 else 0.0

        return ExperimentResult(
            protocol_name=self.name,
            sweep_parameter="amplitude",
            sweep_values=amplitudes,
            measured_values=populations,
            measured_errors=errors,
            raw_counts=raw_counts_list,
            metadata={
                "qubit": self.qubit,
                "pulse_duration_ns": self.pulse_duration_ns,
                "pulse_shape": self.pulse_shape,
                "shots": self.shots,
            },
        )

    def analyze(self, result: ExperimentResult) -> ExperimentResult:
        """Fit Rabi oscillation to extract pi-pulse amplitude."""
        x = result.sweep_values
        y = result.measured_values

        # Initial guesses
        A0 = (y.max() - y.min()) / 2
        B0 = y.mean()
        # Estimate frequency from zero crossings
        crossings = np.where(np.diff(np.sign(y - B0)))[0]
        if len(crossings) >= 2:
            period = 2 * np.mean(np.diff(x[crossings]))
            f0 = 1.0 / period if period > 0 else 5.0
        else:
            f0 = 1.0 / (x[-1] - x[0]) if (x[-1] - x[0]) > 0 else 5.0

        try:
            popt, pcov = curve_fit(
                _rabi_model, x, y,
                p0=[A0, f0, 0.0, B0],
                bounds=([-1, 0, -np.pi, -1], [1, 100, np.pi, 2]),
                maxfev=5000,
            )
            perr = np.sqrt(np.diag(pcov))

            A, f, phi, B = popt
            y_fit = _rabi_model(x, *popt)
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Pi-pulse: first maximum of P(|1>)
            # Model: A*cos(2*pi*f*x + phi) + B
            # P(|1>) is maximized when the model is maximized:
            #   If A < 0: max at cos = -1, i.e. 2*pi*f*x + phi = pi
            #   If A > 0: max at cos = +1, i.e. 2*pi*f*x + phi = 0
            # But we want the FIRST positive x where this happens.
            if A < 0:
                # cos = -1 at 2*pi*f*x + phi = pi + 2*n*pi
                # x = (pi - phi) / (2*pi*f) + n/f
                pi_amp = (np.pi - phi) / (2 * np.pi * f)
            else:
                # cos = +1 at 2*pi*f*x + phi = 2*n*pi
                # x = -phi / (2*pi*f) + n/f
                pi_amp = -phi / (2 * np.pi * f)

            # Ensure pi_amp is positive and in sweep range
            while pi_amp < x[0]:
                pi_amp += 1.0 / f  # add one period
            if pi_amp > x[-1]:
                # Fallback: use half-period (works for symmetric oscillation)
                pi_amp = 1.0 / (2 * f)

            fit = FitResult(
                parameters={
                    "pi_amplitude": float(pi_amp),
                    "rabi_frequency": float(f),
                    "amplitude": float(A),
                    "offset": float(B),
                    "phase": float(phi),
                },
                uncertainties={
                    "pi_amplitude": float(perr[1] / (2 * f**2)) if f > 0 else 0.0,
                    "rabi_frequency": float(perr[1]),
                    "amplitude": float(perr[0]),
                },
                r_squared=float(r_squared),
                model="A * cos(2*pi*f*x + phi) + B",
                residuals=y - y_fit,
            )
            result.fit = fit

        except (RuntimeError, ValueError):
            # Fit failed — try simpler approach
            peak_idx = np.argmax(y)
            result.fit = FitResult(
                parameters={"pi_amplitude": float(x[peak_idx])},
                r_squared=0.0,
                model="peak_detection_fallback",
            )

        return result

    def visualize(self, result: ExperimentResult, save_path: str | None = None) -> Any:
        """Plot Rabi oscillation with fit."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        x = result.sweep_values
        y = result.measured_values

        # Data points with error bars
        if result.measured_errors is not None:
            ax.errorbar(x * 1e3, y, yerr=result.measured_errors,
                       fmt='o', markersize=4, color='#2196F3',
                       ecolor='#90CAF9', capsize=2, label='Data')
        else:
            ax.plot(x * 1e3, y, 'o', markersize=4, color='#2196F3', label='Data')

        # Fit curve
        if result.fit and result.fit.r_squared > 0.5:
            x_fine = np.linspace(x[0], x[-1], 200)
            p = result.fit.parameters
            y_fit = _rabi_model(x_fine, p['amplitude'], p['rabi_frequency'],
                               p['phase'], p['offset'])
            ax.plot(x_fine * 1e3, y_fit, '-', color='#F44336', linewidth=1.5, label='Fit')

            # Mark pi-pulse
            pi_amp = p['pi_amplitude']
            ax.axvline(pi_amp * 1e3, color='#4CAF50', linestyle='--', linewidth=1,
                      label=f'$\\pi$-pulse = {pi_amp*1e3:.2f} mV')

            ax.set_title(
                f'Rabi Oscillation (Q{result.metadata.get("qubit", 0)}) — '
                f'$R^2$ = {result.fit.r_squared:.3f}',
                fontsize=12,
            )
        else:
            ax.set_title(f'Rabi Oscillation (Q{result.metadata.get("qubit", 0)})', fontsize=12)

        ax.set_xlabel('Drive Amplitude (mV)', fontsize=11)
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
