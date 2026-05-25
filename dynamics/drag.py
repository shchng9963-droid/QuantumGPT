"""
DRAG (Derivative Removal by Adiabatic Gate) pulse calibration simulator.

Models a 3-level transmon driven by a DRAG-corrected pulse to suppress
leakage to the |2⟩ state.

The transmon Hamiltonian in the rotating frame (drive frame) is:

    H(t) = δ |2⟩⟨2|
         + (Ω_I(t)/2)(|0⟩⟨1| + h.c.) + (√2 Ω_I(t)/2)(|1⟩⟨2| + h.c.)
         + (Ω_Q(t)/2)(-i|0⟩⟨1| + h.c.) + (√2 Ω_Q(t)/2)(-i|1⟩⟨2| + h.c.)

where:
    δ  = anharmonicity (ω_12 - ω_01), typically ~-300 MHz for transmons
    Ω_I(t) = base envelope (Gaussian or square)
    Ω_Q(t) = -α · dΩ_I/dt / δ   (DRAG correction on quadrature)
    α      = DRAG parameter to calibrate

The goal: find optimal α that minimizes |2⟩ leakage after a π-pulse.

Reference: Motzoi et al., PRL 103, 110501 (2009)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


# ---------- 3-level operators ----------

def _sigma_x_01():
    """σ_x acting on |0⟩↔|1⟩ subspace (3x3)."""
    m = np.zeros((3, 3), dtype=complex)
    m[0, 1] = 1.0
    m[1, 0] = 1.0
    return m


def _sigma_y_01():
    """σ_y acting on |0⟩↔|1⟩ subspace (3x3)."""
    m = np.zeros((3, 3), dtype=complex)
    m[0, 1] = -1j
    m[1, 0] = 1j
    return m


def _sigma_x_12():
    """σ_x acting on |1⟩↔|2⟩ subspace (3x3), scaled by √2."""
    m = np.zeros((3, 3), dtype=complex)
    m[1, 2] = np.sqrt(2)
    m[2, 1] = np.sqrt(2)
    return m


def _sigma_y_12():
    """σ_y acting on |1⟩↔|2⟩ subspace (3x3), scaled by √2."""
    m = np.zeros((3, 3), dtype=complex)
    m[1, 2] = -1j * np.sqrt(2)
    m[2, 1] = 1j * np.sqrt(2)
    return m


def _proj_2():
    """|2⟩⟨2| projector (3x3)."""
    m = np.zeros((3, 3), dtype=complex)
    m[2, 2] = 1.0
    return m


# Precompute operators
_SX01 = _sigma_x_01()
_SY01 = _sigma_y_01()
_SX12 = _sigma_x_12()
_SY12 = _sigma_y_12()
_P2 = _proj_2()


# ---------- Data classes ----------

@dataclass
class DRAGConfig:
    """Configuration for DRAG calibration.

    Parameters
    ----------
    qubit_freq_ghz : float
        Qubit 0→1 frequency (GHz). Default 5.0.
    anharmonicity_ghz : float
        Anharmonicity δ = ω_12 - ω_01 (GHz). Negative for transmon. Default -0.3.
    pi_amp_ghz : float
        Pre-calibrated π-pulse amplitude (GHz). Default 0.005 (from Rabi).
    pulse_duration_ns : float
        Duration of the π-pulse in ns. Default 100.
    pulse_shape : Literal["gaussian", "square"]
        Base envelope shape. Default "gaussian".
    sigma_ns : float
        Gaussian sigma (ns). Default 25.
    alpha_range : tuple[float, float]
        Range of DRAG α values to sweep. Default (-2.0, 2.0).
    n_alphas : int
        Number of α points in sweep. Default 21.
    n_pulses : int
        Number of repeated pulses (X-X pairs) for amplifying leakage. Default 2.
    dt_ns : float
        Integration time step (ns). Default 0.5.
    """

    qubit_freq_ghz: float = 5.0
    anharmonicity_ghz: float = -0.3
    pi_amp_ghz: float = 0.005
    pulse_duration_ns: float = 100.0
    pulse_shape: Literal["gaussian", "square"] = "gaussian"
    sigma_ns: float = 25.0
    alpha_range: tuple[float, float] = (-2.0, 2.0)
    n_alphas: int = 21
    n_pulses: int = 2
    dt_ns: float = 0.5


@dataclass
class DRAGResult:
    """Result of a single DRAG simulation at one α value."""

    alpha: float
    leakage: float         # P(|2⟩) after sequence
    population_0: float    # P(|0⟩) after sequence
    population_1: float    # P(|1⟩) after sequence
    config: DRAGConfig = field(repr=False)


@dataclass
class DRAGSweepResult:
    """Result of a DRAG α sweep."""

    alphas: NDArray[np.float64]
    leakages: NDArray[np.float64]       # P(|2⟩) at each α
    populations_1: NDArray[np.float64]  # P(|1⟩) at each α
    results: list[DRAGResult]
    optimal_alpha: float                # α that minimizes leakage
    min_leakage: float                  # minimum P(|2⟩) achieved
    config: DRAGConfig = field(repr=False)


# ---------- Envelope helpers ----------

def _gaussian_envelope(t: float, amp: float, t_center: float, sigma: float) -> float:
    """Gaussian envelope."""
    return amp * np.exp(-0.5 * ((t - t_center) / sigma) ** 2)


def _gaussian_derivative(t: float, amp: float, t_center: float, sigma: float) -> float:
    """Time derivative of Gaussian envelope."""
    return amp * (-(t - t_center) / sigma**2) * np.exp(-0.5 * ((t - t_center) / sigma) ** 2)


# ---------- Core simulation ----------

def simulate_drag(
    alpha: float,
    config: DRAGConfig | None = None,
    initial_state: int = 0,
) -> DRAGResult:
    """Simulate a DRAG pulse sequence at a given α value.

    Applies n_pulses π-pulses (X gates) with DRAG correction.
    For even n_pulses, ideal result is return to |initial_state⟩.
    Leakage to |2⟩ indicates insufficient DRAG correction.

    Parameters
    ----------
    alpha : float
        DRAG parameter.
    config : DRAGConfig, optional
        Simulation parameters.
    initial_state : int
        Initial state index (0 or 1). Default 0.

    Returns
    -------
    DRAGResult
        Populations after the pulse sequence.
    """
    cfg = config or DRAGConfig()
    delta = cfg.anharmonicity_ghz  # GHz (negative for transmon)
    amp = cfg.pi_amp_ghz
    T = cfg.pulse_duration_ns
    dt = cfg.dt_ns

    # Build static Hamiltonian: δ |2⟩⟨2|
    H_static = delta * _P2

    # Drive operators
    drive_x = 0.5 * (_SX01 + _SX12)  # (Ω_I/2)(|0⟩⟨1| + h.c.) + (√2 Ω_I/2)(|1⟩⟨2| + h.c.)
    drive_y = 0.5 * (_SY01 + _SY12)  # quadrature component

    if cfg.pulse_shape == "gaussian":
        t_center = T / 2.0
        sigma = cfg.sigma_ns

        def omega_I(t_local):
            return _gaussian_envelope(t_local, amp, t_center, sigma)

        def d_omega_I(t_local):
            return _gaussian_derivative(t_local, amp, t_center, sigma)

    elif cfg.pulse_shape == "square":
        def omega_I(t_local):
            return amp

        def d_omega_I(t_local):
            return 0.0
    else:
        raise ValueError(f"Unknown pulse shape: {cfg.pulse_shape}")

    # DRAG quadrature: Ω_Q(t) = -α · dΩ_I/dt / δ
    # Guard against δ=0
    if abs(delta) < 1e-12:
        def omega_Q(t_local):
            return 0.0
    else:
        def omega_Q(t_local):
            return -alpha * d_omega_I(t_local) / delta

    # Schrödinger equation: dψ/dt = -i 2π H(t) ψ
    def schrodinger_rhs(t, y):
        psi = y.view(np.complex128)
        t_local = t % T  # Within current pulse
        OI = omega_I(t_local)
        OQ = omega_Q(t_local)
        H = H_static + OI * drive_x + OQ * drive_y
        dpsi = -1j * 2.0 * np.pi * H @ psi
        return dpsi.view(np.float64)

    # Initial state
    psi0 = np.zeros(3, dtype=complex)
    psi0[initial_state] = 1.0
    y0 = psi0.view(np.float64).copy()

    # Total time for n_pulses
    t_total = cfg.n_pulses * T
    n_steps = int(t_total / dt)
    t_span = [0.0, t_total]

    sol = solve_ivp(
        schrodinger_rhs,
        t_span=t_span,
        y0=y0,
        method="DOP853",
        max_step=dt,
        atol=1e-10,
        rtol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    # Final state
    psi_final = sol.y[:, -1].view(np.complex128)
    p0 = float(np.abs(psi_final[0]) ** 2)
    p1 = float(np.abs(psi_final[1]) ** 2)
    p2 = float(np.abs(psi_final[2]) ** 2)

    return DRAGResult(
        alpha=alpha,
        leakage=p2,
        population_0=p0,
        population_1=p1,
        config=cfg,
    )


def sweep_drag(config: DRAGConfig | None = None) -> DRAGSweepResult:
    """Sweep DRAG α parameter and find optimal value.

    Parameters
    ----------
    config : DRAGConfig, optional
        Configuration including alpha_range and n_alphas.

    Returns
    -------
    DRAGSweepResult
        Sweep data with optimal α identified.
    """
    cfg = config or DRAGConfig()
    alphas = np.linspace(cfg.alpha_range[0], cfg.alpha_range[1], cfg.n_alphas)

    results: list[DRAGResult] = []
    for a in alphas:
        r = simulate_drag(a, cfg)
        results.append(r)

    leakages = np.array([r.leakage for r in results])
    pops_1 = np.array([r.population_1 for r in results])

    # Optimal α = minimum leakage
    best_idx = int(np.argmin(leakages))
    optimal_alpha = float(alphas[best_idx])
    min_leakage = float(leakages[best_idx])

    return DRAGSweepResult(
        alphas=alphas,
        leakages=leakages,
        populations_1=pops_1,
        results=results,
        optimal_alpha=optimal_alpha,
        min_leakage=min_leakage,
        config=cfg,
    )


def calibrate_drag(
    config: DRAGConfig | None = None,
    refine: bool = True,
) -> dict:
    """Full DRAG calibration: coarse sweep + optional refinement.

    Parameters
    ----------
    config : DRAGConfig, optional
        Configuration.
    refine : bool
        If True, do a fine sweep around the coarse optimum. Default True.

    Returns
    -------
    dict with keys:
        optimal_alpha : float
        min_leakage : float
        coarse_sweep : DRAGSweepResult
        fine_sweep : DRAGSweepResult | None
        gate_fidelity : float — estimated fidelity (1 - leakage) in computational subspace
    """
    cfg = config or DRAGConfig()

    # Coarse sweep
    coarse = sweep_drag(cfg)

    fine = None
    if refine and cfg.n_alphas >= 5:
        # Zoom into ±20% of range around optimum
        range_width = (cfg.alpha_range[1] - cfg.alpha_range[0])
        margin = 0.1 * range_width
        fine_cfg = dataclasses.replace(
            cfg,
            alpha_range=(coarse.optimal_alpha - margin, coarse.optimal_alpha + margin),
            n_alphas=31,
        )
        fine = sweep_drag(fine_cfg)
        optimal_alpha = fine.optimal_alpha
        min_leakage = fine.min_leakage
    else:
        optimal_alpha = coarse.optimal_alpha
        min_leakage = coarse.min_leakage

    # Gate fidelity estimate: 1 - leakage (simplified)
    gate_fidelity = 1.0 - min_leakage

    return {
        "optimal_alpha": optimal_alpha,
        "min_leakage": min_leakage,
        "gate_fidelity": gate_fidelity,
        "coarse_sweep": coarse,
        "fine_sweep": fine,
    }
