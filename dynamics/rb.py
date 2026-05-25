"""
Randomized Benchmarking (RB) simulator.

Standard single-qubit RB: apply random sequences of Clifford gates
of increasing length, measure survival probability, and fit the
exponential decay to extract the Error Per Clifford (EPC).

The 24 single-qubit Clifford gates are generated as 2×2 unitary matrices.
A simple depolarizing noise model (error rate per gate) is used.

Reference: Knill et al., PRA 77, 012307 (2008);
           Magesan et al., PRL 106, 180504 (2011)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit


# ---------- Clifford group (single qubit, 24 elements) ----------

# Pauli matrices
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _rotation(axis: NDArray, angle: float) -> NDArray:
    """Rotation by `angle` radians about `axis` (Pauli matrix)."""
    return np.cos(angle / 2) * _I - 1j * np.sin(angle / 2) * axis


def _generate_cliffords() -> list[NDArray[np.complex128]]:
    """Generate all 24 single-qubit Clifford unitaries."""
    # Build from standard decomposition:
    # 6 π/2 rotations × 4 Paulis = 24 Cliffords
    #
    # More precisely, the single-qubit Clifford group has 24 elements.
    # We generate them by composing rotations.
    cliffords = set()
    gates = []

    # Basic generators
    basic = [
        _I,
        _rotation(_X, np.pi / 2),   # Rx(π/2)  = SX
        _rotation(_X, -np.pi / 2),  # Rx(-π/2) = SX†
        _rotation(_X, np.pi),       # Rx(π)    = X
        _rotation(_Y, np.pi / 2),   # Ry(π/2)  = SY
        _rotation(_Y, -np.pi / 2),  # Ry(-π/2) = SY†
        _rotation(_Y, np.pi),       # Ry(π)    = Y
        _rotation(_Z, np.pi / 2),   # Rz(π/2)  = S
        _rotation(_Z, -np.pi / 2),  # Rz(-π/2) = S†
        _rotation(_Z, np.pi),       # Rz(π)    = Z
    ]

    # Generate all compositions up to depth 3
    candidates = list(basic)
    for g1 in basic:
        for g2 in basic:
            candidates.append(g1 @ g2)
            for g3 in basic:
                candidates.append(g1 @ g2 @ g3)

    # Deduplicate by comparing action on |0⟩ and |1⟩
    # Two unitaries are equivalent (up to global phase) if they
    # map the same way in projective space.
    seen = set()
    for U in candidates:
        # Normalize global phase so U[0,0] is real and positive
        if abs(U[0, 0]) > 1e-10:
            U = U * np.conj(U[0, 0]) / abs(U[0, 0])
        elif abs(U[0, 1]) > 1e-10:
            U = U * np.conj(U[0, 1]) / abs(U[0, 1])
        elif abs(U[1, 0]) > 1e-10:
            U = U * np.conj(U[1, 0]) / abs(U[1, 0])
        else:
            U = U * np.conj(U[1, 1]) / abs(U[1, 1])

        # Round for hashing
        key = tuple(np.round(U.ravel(), 8))
        if key not in seen:
            seen.add(key)
            gates.append(U.copy())

    assert len(gates) == 24, f"Expected 24 Cliffords, got {len(gates)}"
    return gates


# Cache the Clifford group
_CLIFFORDS_1Q: list[NDArray[np.complex128]] | None = None


def get_cliffords_1q() -> list[NDArray[np.complex128]]:
    """Get the 24 single-qubit Clifford unitaries (cached)."""
    global _CLIFFORDS_1Q
    if _CLIFFORDS_1Q is None:
        _CLIFFORDS_1Q = _generate_cliffords()
    return _CLIFFORDS_1Q


def _inverse_clifford(U: NDArray) -> NDArray:
    """Compute U† (inverse of unitary)."""
    return U.conj().T


# ---------- Data classes ----------

@dataclass
class RBConfig:
    """Configuration for a Randomized Benchmarking experiment.

    Parameters
    ----------
    sequence_lengths : list[int]
        Number of Clifford gates in each sequence. Default [1,2,4,8,16,32,64].
    n_sequences : int
        Number of random sequences per length. Default 20.
    error_per_gate : float
        Depolarizing error rate per Clifford gate. Default 0.001 (0.1%).
    readout_error : float
        Probability of bit-flip on measurement. Default 0.01.
    seed : int | None
        RNG seed for reproducibility. Default None.
    shots : int
        Number of measurement shots per sequence. Default 1024.
    """

    sequence_lengths: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    n_sequences: int = 20
    error_per_gate: float = 0.001
    readout_error: float = 0.01
    seed: int | None = None
    shots: int = 1024


@dataclass
class RBResult:
    """Result of a Randomized Benchmarking experiment."""

    sequence_lengths: NDArray[np.int64]
    survival_probs: NDArray[np.float64]     # mean P(|0⟩) at each length
    survival_stds: NDArray[np.float64]      # std across random sequences
    error_per_clifford: float               # fitted EPC
    depolarizing_param: float               # fitted p (decay parameter)
    fit_a: float                            # A in A·p^m + B
    fit_b: float                            # B offset
    r_squared: float                        # goodness of fit
    raw_survivals: list[NDArray]            # per-length arrays of survival probs
    config: RBConfig = field(repr=False)


# ---------- Noise model ----------

def _apply_depolarizing(rho: NDArray, error_rate: float) -> NDArray:
    """Apply single-qubit depolarizing channel to density matrix.

    ε(ρ) = (1-p)ρ + p/2 · I

    where p = error_rate.
    """
    p = error_rate
    return (1 - p) * rho + (p / 2) * _I


def _apply_readout_error(prob_0: float, readout_error: float) -> float:
    """Apply classical readout error."""
    return (1 - readout_error) * prob_0 + readout_error * (1 - prob_0)


# ---------- Core simulation ----------

def run_rb_sequence(
    length: int,
    config: RBConfig,
    rng: np.random.Generator,
) -> float:
    """Run a single RB sequence of given length and return survival probability.

    Steps:
    1. Pick `length` random Cliffords
    2. Compute their composition
    3. Append the inverse of the composition
    4. Simulate with depolarizing noise
    5. Measure P(|0⟩)
    """
    cliffords = get_cliffords_1q()
    n_clif = len(cliffords)

    # Pick random Clifford indices
    indices = rng.integers(0, n_clif, size=length)

    # Initial state: |0⟩⟨0|
    rho = np.array([[1, 0], [0, 0]], dtype=complex)

    # Apply each Clifford with depolarizing noise
    composition = _I.copy()
    for idx in indices:
        U = cliffords[idx]
        composition = U @ composition  # track ideal composition
        rho = U @ rho @ U.conj().T     # apply gate
        rho = _apply_depolarizing(rho, config.error_per_gate)

    # Append inverse to ideally return to |0⟩
    U_inv = _inverse_clifford(composition)
    rho = U_inv @ rho @ U_inv.conj().T
    rho = _apply_depolarizing(rho, config.error_per_gate)  # inverse gate also has error

    # Measure P(|0⟩)
    prob_0 = float(np.real(rho[0, 0]))
    prob_0 = np.clip(prob_0, 0, 1)

    # Apply readout error
    prob_0 = _apply_readout_error(prob_0, config.readout_error)

    # Simulate shot noise
    if config.shots > 0:
        n_0 = rng.binomial(config.shots, prob_0)
        prob_0 = n_0 / config.shots

    return prob_0


def run_rb(config: RBConfig | None = None) -> RBResult:
    """Run a full Randomized Benchmarking experiment.

    Parameters
    ----------
    config : RBConfig, optional
        Experiment configuration.

    Returns
    -------
    RBResult
        Fitted decay curve and error per Clifford.
    """
    cfg = config or RBConfig()
    rng = np.random.default_rng(cfg.seed)

    lengths = np.array(cfg.sequence_lengths, dtype=np.int64)
    raw_survivals = []
    mean_probs = []
    std_probs = []

    for length in lengths:
        probs = np.array([
            run_rb_sequence(int(length), cfg, rng)
            for _ in range(cfg.n_sequences)
        ])
        raw_survivals.append(probs)
        mean_probs.append(np.mean(probs))
        std_probs.append(np.std(probs))

    mean_probs = np.array(mean_probs)
    std_probs = np.array(std_probs)

    # Fit: f(m) = A · p^m + B
    def rb_model(m, a, p, b):
        return a * p ** m + b

    try:
        # Initial guess: A ≈ 0.5, p ≈ 1-2*EPC, B ≈ 0.5
        p0 = [0.5, 1 - 2 * cfg.error_per_gate, 0.5]
        popt, pcov = curve_fit(
            rb_model, lengths.astype(float), mean_probs,
            p0=p0, bounds=([0, 0, 0], [1, 1, 1]),
            maxfev=5000,
        )
        a_fit, p_fit, b_fit = popt

        # Compute R²
        residuals = mean_probs - rb_model(lengths.astype(float), *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((mean_probs - np.mean(mean_probs)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    except (RuntimeError, ValueError):
        # Fallback: estimate from first and last points
        if mean_probs[0] > mean_probs[-1] and len(lengths) > 1:
            p_fit = (mean_probs[-1] / mean_probs[0]) ** (1.0 / (lengths[-1] - lengths[0]))
        else:
            p_fit = 0.99
        a_fit = 0.5
        b_fit = 0.5
        r_sq = 0.0

    # EPC = (1 - p) · (1 - 1/d) where d = 2 for single qubit
    epc = (1 - p_fit) * (1 - 1 / 2)  # = (1 - p) / 2

    return RBResult(
        sequence_lengths=lengths,
        survival_probs=mean_probs,
        survival_stds=std_probs,
        error_per_clifford=float(epc),
        depolarizing_param=float(p_fit),
        fit_a=float(a_fit),
        fit_b=float(b_fit),
        r_squared=float(r_sq),
        raw_survivals=raw_survivals,
        config=cfg,
    )
