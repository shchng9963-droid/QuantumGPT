"""Ramsey fringe experiment protocol.

Measures T2* (dephasing time) by applying two pi/2 pulses separated
by a variable delay, with optional artificial detuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from experiments.base import ExperimentProtocol, ExperimentResult, FitResult
from backends.lab_backend import LabBackend, PulseSchedule


def _ramsey_model(t: np.ndarray, A: float, T2star: float, f_det: float, phi: float, B: float) -> np.ndarray:
    """Ramsey model: A * cos(2*pi*f_det*t + phi) * exp(-t/T2*) + B"""
    return A * np.cos(2 * np.pi * f_det * t + phi) * np.exp(-t / T2star) + B


@dataclass
class RamseyExperiment(ExperimentProtocol):
    """Ramsey fringe experiment.

    Sequence: pi/2 -- delay(tau) -- pi/2 -- measure
    Sweeps tau to observe fringes. Extracts T2* and detuning.
    """

    qubit: int = 0
    delay_min_ns: float = 0.0
    delay_max_ns: float = 5000.0  # 5 us
    n_points: int = 50
    pi_half_amplitude: float = 0.015  # half of pi-pulse amplitude
    pi_half_duration_ns: float = 50.0
    artificial_detuning_mhz: float = 2.0  # adds known oscillation for fitting
    shots: int = 1024

    @property
    def name(self) -> str:
        return "ramsey"

    def run(self, backend: LabBackend) -> ExperimentResult:
        """Sweep delay between two pi/2 pulses."""
        delays = np.linspace(self.delay_min_ns, self.delay_max_ns, self.n_points)
        populations = np.zeros(self.n_points)
        errors = np.zeros(self.n_points)
        raw_counts_list = []

        # Phase increment per delay step for artificial detuning
        # delta_phi = 2*pi * f_det * tau
        f_det_ghz = self.artificial_detuning_mhz * 1e-3  # MHz -> GHz

        for i, tau in enumerate(delays):
            # Second pi/2 pulse gets extra phase from detuning
            phase_2 = 2 * np.pi * f_det_ghz * tau  # tau in ns, f in GHz -> radians

            schedule = PulseSchedule(
                channels={
                    "d0": [
                        # First pi/2 pulse
                        {
                            "amplitude": float(self.pi_half_amplitude),
                            "duration_ns": self.pi_half_duration_ns,
                            "shape": "gaussian",
                            "phase": 0.0,
                        },
                        # Delay (zero amplitude)
                        {
                            "amplitude": 0.0,
                            "duration_ns": float(tau),
                            "shape": "square",
                            "phase": 0.0,
                        },
                        # Second pi/2 pulse with detuning phase
                        {
                            "amplitude": float(self.pi_half_amplitude),
                            "duration_ns": self.pi_half_duration_ns,
                            "shape": "gaussian",
                            "phase": float(phase_2),
                        },
                    ]
                },
                duration_ns=2 * self.pi_half_duration_ns + tau,
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
                "pi_half_amplitude": self.pi_half_amplitude,
                "artificial_detuning_mhz": self.artificial_detuning_mhz,
                "shots": self.shots,
            },
        )

    def analyze(self, result: ExperimentResult) -> ExperimentResult:
        """Fit Ramsey fringes to extract T2* and detuning."""
        x = result.sweep_values  # delay in ns
        y = result.measured_values

        # Convert to microseconds for fitting
        x_us = x / 1000.0

        # Initial guesses
        A0 = (y.max() - y.min()) / 2
        B0 = y.mean()
        T2star_0 = x_us[-1] / 3  # guess: decays over 1/3 of range
        f_det_0 = self.artificial_detuning_mhz * 1e-3  # GHz (for us timescale: MHz)

        try:
            # Fit in ns units
            def model_ns(t_ns, A, T2star_ns, f_det_ghz, phi, B):
                return A * np.cos(2 * np.pi * f_det_ghz * t_ns + phi) * np.exp(-t_ns / T2star_ns) + B

            popt, pcov = curve_fit(
                model_ns, x, y,
                p0=[A0, self.delay_max_ns / 3, self.artificial_detuning_mhz * 1e-3, 0.0, B0],
                bounds=([-1, 100, 0, -np.pi, -1], [1, 1e6, 0.1, np.pi, 2]),
                maxfev=10000,
            )
            perr = np.sqrt(np.diag(pcov))

            A, T2star_ns, f_det_ghz, phi, B = popt
            T2star_us = T2star_ns / 1000.0
            f_det_mhz = f_det_ghz * 1000.0

            y_fit = model_ns(x, *popt)
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            fit = FitResult(
                parameters={
                    "T2_star_us": float(T2star_us),
                    "T2_star_ns": float(T2star_ns),
                    "detuning_mhz": float(f_det_mhz),
                    "amplitude": float(A),
                    "offset": float(B),
                    "phase": float(phi),
                },
                uncertainties={
                    "T2_star_ns": float(perr[1]),
                    "detuning_mhz": float(perr[2] * 1000),
                },
                r_squared=float(r_squared),
                model="A * cos(2*pi*f*t + phi) * exp(-t/T2*) + B",
                residuals=y - y_fit,
            )
            result.fit = fit

        except (RuntimeError, ValueError):
            # Fallback: estimate T2* from envelope decay
            envelope = np.abs(y - y.mean())
            half_idx = np.argmax(envelope < envelope[0] / np.e) if envelope[0] > 0 else len(x) // 3
            T2star_est = x[half_idx] if half_idx > 0 else x[-1] / 3

            result.fit = FitResult(
                parameters={"T2_star_ns": float(T2star_est), "T2_star_us": float(T2star_est / 1000)},
                r_squared=0.0,
                model="envelope_decay_fallback",
            )

        return result

    def visualize(self, result: ExperimentResult, save_path: str | None = None) -> Any:
        """Plot Ramsey fringes with fit."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(9, 7), height_ratios=[3, 1])

        x = result.sweep_values / 1000.0  # convert to us
        y = result.measured_values

        # Main plot
        ax = axes[0]
        if result.measured_errors is not None:
            ax.errorbar(x, y, yerr=result.measured_errors,
                       fmt='o', markersize=3, color='#673AB7',
                       ecolor='#CE93D8', capsize=1.5, label='Data')
        else:
            ax.plot(x, y, 'o', markersize=3, color='#673AB7', label='Data')

        if result.fit and result.fit.r_squared > 0.5:
            p = result.fit.parameters
            x_fine = np.linspace(result.sweep_values[0], result.sweep_values[-1], 500) / 1000.0
            x_fine_ns = x_fine * 1000.0
            T2star_ns = p['T2_star_ns']
            f_det_ghz = p.get('detuning_mhz', 2.0) / 1000.0
            A = p['amplitude']
            B = p['offset']
            phi = p['phase']

            y_fit = A * np.cos(2 * np.pi * f_det_ghz * x_fine_ns + phi) * np.exp(-x_fine_ns / T2star_ns) + B
            ax.plot(x_fine, y_fit, '-', color='#FF5722', linewidth=1.5, label='Fit')

            # Envelope
            y_env_upper = A * np.exp(-x_fine_ns / T2star_ns) + B
            y_env_lower = -A * np.exp(-x_fine_ns / T2star_ns) + B
            ax.plot(x_fine, y_env_upper, '--', color='#FF9800', alpha=0.5, linewidth=1)
            ax.plot(x_fine, y_env_lower, '--', color='#FF9800', alpha=0.5, linewidth=1)

            ax.axvline(p['T2_star_us'], color='#4CAF50', linestyle=':', linewidth=1,
                      label=f'$T_2^*$ = {p["T2_star_us"]:.2f} μs')

            ax.set_title(
                f'Ramsey Fringes (Q{result.metadata.get("qubit", 0)}) — '
                f'$T_2^*$ = {p["T2_star_us"]:.2f} μs, '
                f'Δf = {p.get("detuning_mhz", 0):.2f} MHz, '
                f'$R^2$ = {result.fit.r_squared:.3f}',
                fontsize=11,
            )
        else:
            ax.set_title(f'Ramsey Fringes (Q{result.metadata.get("qubit", 0)})', fontsize=11)

        ax.set_xlabel('Delay (μs)', fontsize=11)
        ax.set_ylabel('P(|1⟩)', fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Residuals
        ax2 = axes[1]
        if result.fit and result.fit.residuals is not None:
            ax2.stem(x, result.fit.residuals, linefmt='#9E9E9E', markerfmt='o',
                    basefmt='k-', label='Residuals')
            ax2.axhline(0, color='k', linewidth=0.5)
            ax2.set_ylabel('Residual', fontsize=10)
        ax2.set_xlabel('Delay (μs)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
