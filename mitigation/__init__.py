"""Error mitigation — Zero Noise Extrapolation (ZNE) without Mitiq.

Implements circuit-level ZNE by:
  1. Folding gates (inserting G G† G pairs) to amplify noise
  2. Running at multiple noise scale factors
  3. Extrapolating to the zero-noise limit via polynomial fit

This avoids the Mitiq dependency (incompatible with Python 3.13).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from scipy.optimize import curve_fit


def fold_gates_global(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """Fold all gates globally: repeat the circuit (scale_factor-1)/2 times
    as circuit + inverse + circuit + ...

    scale_factor must be an odd integer >= 1.
    Measurement gates are stripped before folding and re-appended at the end.
    """
    if scale_factor == 1:
        return circuit.copy()
    if scale_factor % 2 == 0:
        raise ValueError("scale_factor must be odd")

    # Strip measurements
    bare = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    measures = []
    for inst in circuit.data:
        if inst.operation.name == "measure":
            measures.append(inst)
        else:
            bare.append(inst)

    folded = bare.copy()
    n_folds = (scale_factor - 1) // 2
    for _ in range(n_folds):
        folded = folded.compose(bare.inverse())
        folded = folded.compose(bare.copy())

    # Re-append measurements
    for inst in measures:
        folded.append(inst)

    return folded


def zne_extrapolate(
    scale_factors: list[int],
    expectation_values: list[float],
    order: int = 1,
) -> dict:
    """Extrapolate to zero noise using polynomial fit.

    Args:
        scale_factors: noise amplification factors (e.g. [1, 3, 5])
        expectation_values: measured values at each scale
        order: polynomial order (1=linear, 2=quadratic)

    Returns:
        dict with 'zne_value', 'fit_coeffs', 'r_squared'
    """
    x = np.array(scale_factors, dtype=float)
    y = np.array(expectation_values, dtype=float)

    coeffs = np.polyfit(x, y, min(order, len(x) - 1))
    poly = np.poly1d(coeffs)

    # Extrapolate to x=0
    zne_value = float(poly(0))

    # R² goodness of fit
    y_pred = poly(x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "zne_value": np.clip(zne_value, 0.0, 1.0),
        "fit_coeffs": coeffs.tolist(),
        "r_squared": float(r_squared),
        "raw_values": expectation_values,
        "scale_factors": scale_factors,
    }


def run_zne(
    circuit: QuantumCircuit,
    backend,
    shots: int = 4096,
    scale_factors: list[int] | None = None,
    order: int = 2,
) -> dict:
    """Run ZNE end-to-end: fold → execute → extrapolate.

    Args:
        circuit: the original circuit
        backend: a ShadowBackend with .run() method
        shots: shots per scale factor
        scale_factors: noise scales (default [1, 3, 5])
        order: extrapolation polynomial order

    Returns:
        dict with 'mitigated_fidelity', 'unmitigated_fidelity', 'improvement', etc.
    """
    if scale_factors is None:
        scale_factors = [1, 3, 5]

    fidelities = []
    for sf in scale_factors:
        folded = fold_gates_global(circuit, sf)
        result = backend.run(folded, shots=shots)
        fid = result.fidelity if result.fidelity is not None else 0.0
        fidelities.append(fid)

    extrapolation = zne_extrapolate(scale_factors, fidelities, order=order)
    unmitigated = fidelities[0]  # scale=1
    mitigated = extrapolation["zne_value"]

    return {
        "mitigated_fidelity": round(float(mitigated), 4),
        "unmitigated_fidelity": round(float(unmitigated), 4),
        "improvement": round(float(mitigated - unmitigated), 4),
        "extrapolation": extrapolation,
        "method": "ZNE",
        "scale_factors": scale_factors,
        "per_scale_fidelity": [round(f, 4) for f in fidelities],
    }
