"""
Rabi oscillation simulator using qiskit-dynamics.

Models a single transmon qubit driven by a resonant microwave pulse.
The qubit Hamiltonian in the rotating frame is:

    H(t) = -(Δ/2) σ_z  +  (Ω(t)/2) σ_x

where Ω(t) is the drive envelope (Gaussian or square) and Δ is the detuning.

The Solver evolves |0⟩ under H(t) and we read out P(|1⟩) = |⟨1|ψ(t)⟩|².
A sweep over drive amplitudes produces the classic Rabi oscillation curve
from which the π-pulse amplitude can be extracted.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from qiskit_dynamics import Solver, Signal


# ---------- Pauli matrices (2x2) ----------
_SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


# ---------- Data classes ----------

@dataclass
class RabiConfig:
    """Configuration for a single Rabi simulation or sweep.

    Parameters
    ----------
    qubit_freq_ghz : float
        Qubit transition frequency (GHz). Default 5.0.
    drive_freq_ghz : float | None
        Drive frequency (GHz). None → on resonance (= qubit_freq_ghz).
    pulse_duration_ns : float
        Total pulse duration in nanoseconds. Default 100.
    pulse_shape : Literal["square", "gaussian"]
        Envelope shape. Default "gaussian".
    sigma_ns : float
        Gaussian sigma in ns (only used for gaussian shape). Default 25.
    dt_ns : float
        Integration time-step in ns. Default 0.1.
    amp_range : tuple[float, float]
        (min, max) drive amplitude in GHz for sweeps. Default (0, 0.1).
    n_amps : int
        Number of amplitude points in a sweep. Default 51.
    """

    qubit_freq_ghz: float = 5.0
    drive_freq_ghz: float | None = None
    pulse_duration_ns: float = 100.0
    pulse_shape: Literal["square", "gaussian"] = "gaussian"
    sigma_ns: float = 25.0
    dt_ns: float = 0.1
    amp_range: tuple[float, float] = (0.0, 0.1)
    n_amps: int = 51

    @property
    def drive_freq(self) -> float:
        return self.drive_freq_ghz if self.drive_freq_ghz is not None else self.qubit_freq_ghz

    @property
    def detuning_ghz(self) -> float:
        return self.drive_freq - self.qubit_freq_ghz


@dataclass
class RabiResult:
    """Result of a single Rabi simulation at one amplitude."""

    amplitude: float
    times_ns: NDArray[np.float64]
    populations: NDArray[np.float64]       # P(|1⟩) at each time step
    statevectors: NDArray[np.complex128]   # (n_times, 2) state vectors
    config: RabiConfig = field(repr=False)

    @property
    def final_p1(self) -> float:
        """Excited-state population at the end of the pulse."""
        return float(self.populations[-1])


@dataclass
class RabiSweepResult:
    """Result of a Rabi amplitude sweep."""

    amplitudes: NDArray[np.float64]
    final_populations: NDArray[np.float64]  # P(|1⟩) at end of each amp
    results: list[RabiResult]
    config: RabiConfig = field(repr=False)

    @property
    def pi_amplitude(self) -> float:
        """Amplitude that maximises |1⟩ population (coarse estimate)."""
        return float(self.amplitudes[np.argmax(self.final_populations)])


# ---------- Envelope helpers ----------

def _make_envelope_fn(cfg: RabiConfig, amp: float):
    """Return a callable envelope(t) for the drive pulse."""

    if cfg.pulse_shape == "square":
        def envelope(t):
            return amp
    elif cfg.pulse_shape == "gaussian":
        t_center = cfg.pulse_duration_ns / 2.0
        sigma = cfg.sigma_ns

        def envelope(t):
            return amp * np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    else:
        raise ValueError(f"Unknown pulse shape: {cfg.pulse_shape}")

    return envelope


# ---------- Core simulation ----------

def simulate_rabi(
    amplitude: float,
    config: RabiConfig | None = None,
) -> RabiResult:
    """Simulate a driven qubit at a single drive amplitude.

    Works in the rotating frame of the drive, so the effective
    Hamiltonian is:

        H(t) = -(Δ/2) σ_z  +  (Ω(t)/2) σ_x

    where Δ = ω_drive - ω_qubit (detuning) and Ω(t) is the envelope
    scaled by ``amplitude``.

    Parameters
    ----------
    amplitude : float
        Peak drive amplitude in GHz (angular: 2π × GHz).
    config : RabiConfig, optional
        Simulation parameters. Uses defaults if None.

    Returns
    -------
    RabiResult
        Time traces of population and statevectors.
    """
    cfg = config or RabiConfig()
    envelope_fn = _make_envelope_fn(cfg, amp=amplitude)

    # Build the Hamiltonian in rotating frame
    delta = cfg.detuning_ghz

    # Static Hamiltonian: detuning term (can be zero matrix)
    static_ham = -(delta / 2.0) * _SIGMA_Z

    # Drive operator: (1/2) σ_x  — amplitude comes via the signal
    drive_op = 0.5 * _SIGMA_X

    # Build solver with channel-based API
    solver = Solver(
        static_hamiltonian=static_ham,
        hamiltonian_operators=[drive_op],
        hamiltonian_channels=["d0"],
        channel_carrier_freqs={"d0": 0.0},  # already in rotating frame
        dt=cfg.dt_ns,
    )

    # Initial state |0⟩
    y0 = np.array([1.0 + 0j, 0.0 + 0j])

    # Build the drive signal (callable envelope, list-based per channel)
    drive_signal = Signal(envelope=envelope_fn, carrier_freq=0.0)

    # Evaluation times — subsample for memory efficiency
    n_steps = int(cfg.pulse_duration_ns / cfg.dt_ns)
    times = np.linspace(0, cfg.pulse_duration_ns, n_steps + 1)
    max_eval_points = 500
    if len(times) > max_eval_points:
        eval_indices = np.linspace(0, len(times) - 1, max_eval_points, dtype=int)
        t_eval = times[eval_indices]
    else:
        t_eval = times

    result = solver.solve(
        t_span=[float(times[0]), float(times[-1])],
        y0=y0,
        signals=[drive_signal],
        t_eval=t_eval,
        method="DOP853",
        atol=1e-10,
        rtol=1e-10,
    )

    # result.y shape: (n_times, 2)
    states = np.array(result.y)
    populations = np.abs(states[:, 1]) ** 2  # P(|1⟩)

    return RabiResult(
        amplitude=amplitude,
        times_ns=t_eval,
        populations=populations,
        statevectors=states,
        config=cfg,
    )


def sweep_rabi(config: RabiConfig | None = None) -> RabiSweepResult:
    """Sweep drive amplitude and collect Rabi oscillation data.

    Parameters
    ----------
    config : RabiConfig, optional
        Simulation configuration (includes amp_range and n_amps).

    Returns
    -------
    RabiSweepResult
        Amplitudes, final populations, and per-amplitude RabiResults.
    """
    cfg = config or RabiConfig()
    amps = np.linspace(cfg.amp_range[0], cfg.amp_range[1], cfg.n_amps)
    results: list[RabiResult] = []

    for amp in amps:
        r = simulate_rabi(amp, cfg)
        results.append(r)

    final_pops = np.array([r.final_p1 for r in results])

    return RabiSweepResult(
        amplitudes=amps,
        final_populations=final_pops,
        results=results,
        config=cfg,
    )


def estimate_pi_pulse(
    config: RabiConfig | None = None,
    refine: bool = True,
) -> dict:
    """Estimate the π-pulse amplitude via coarse sweep + optional refinement.

    Returns
    -------
    dict with keys:
        pi_amp : float — estimated π-pulse amplitude (GHz)
        pi_fidelity : float — P(|1⟩) at the π-pulse amplitude
        half_pi_amp : float — estimated π/2-pulse amplitude (GHz)
        coarse_sweep : RabiSweepResult — full coarse sweep data
        fine_sweep : RabiSweepResult | None — refined sweep (if refine=True)
    """
    cfg = config or RabiConfig()

    # Coarse sweep
    coarse = sweep_rabi(cfg)
    pi_amp_coarse = coarse.pi_amplitude

    fine = None
    if refine and cfg.n_amps >= 5:
        # Zoom into ±20% around the coarse estimate
        margin = 0.2 * pi_amp_coarse if pi_amp_coarse > 0 else cfg.amp_range[1] * 0.1
        fine_cfg = dataclasses.replace(
            cfg,
            amp_range=(max(0, pi_amp_coarse - margin), pi_amp_coarse + margin),
            n_amps=51,
        )
        fine = sweep_rabi(fine_cfg)
        pi_amp = fine.pi_amplitude
        pi_fid = float(fine.final_populations[np.argmax(fine.final_populations)])
    else:
        pi_amp = pi_amp_coarse
        pi_fid = float(coarse.final_populations[np.argmax(coarse.final_populations)])

    # Estimate π/2 amplitude ≈ half of π amplitude for small-angle regime
    half_pi_amp = pi_amp / 2.0

    return {
        "pi_amp": pi_amp,
        "pi_fidelity": pi_fid,
        "half_pi_amp": half_pi_amp,
        "coarse_sweep": coarse,
        "fine_sweep": fine,
    }
