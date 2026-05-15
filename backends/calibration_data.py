"""Calibration snapshot generation utilities.

Generates synthetic time-series calibration data that mimics realistic
IBM device behavior: gradual parameter drift, occasional jumps,
and per-qubit variability.

These snapshots feed into ReplayBackend and SyntheticDriftBackend.
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CalibrationSnapshot:
    """A single calibration snapshot at a point in time.
    
    Mirrors the structure of IBM backend.properties() but simplified
    to what the agent and drift detector actually need.
    """
    timestamp: float  # Unix timestamp
    backend_name: str
    num_qubits: int

    # Per-qubit properties: list indexed by qubit number
    t1_us: list[float] = field(default_factory=list)       # T1 in microseconds
    t2_us: list[float] = field(default_factory=list)       # T2 in microseconds
    readout_error: list[float] = field(default_factory=list)
    
    # Per-qubit gate errors (1Q): list indexed by qubit
    gate_error_1q: list[float] = field(default_factory=list)
    
    # Per-edge gate errors (2Q): dict of (q_i, q_j) -> error
    gate_error_2q: dict[tuple[int, int], float] = field(default_factory=dict)
    
    # Coupling map
    coupling_map: list[tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "backend_name": self.backend_name,
            "num_qubits": self.num_qubits,
            "t1_us": self.t1_us,
            "t2_us": self.t2_us,
            "readout_error": self.readout_error,
            "gate_error_1q": self.gate_error_1q,
            "gate_error_2q": {f"{k[0]}-{k[1]}": v for k, v in self.gate_error_2q.items()},
            "coupling_map": self.coupling_map,
        }


def extract_snapshot_from_fake_backend(backend_name: str = "FakeBrisbane") -> CalibrationSnapshot:
    """Extract a CalibrationSnapshot from a Qiskit FakeBackendV2.
    
    This gives us a realistic baseline to drift from.
    """
    from qiskit_ibm_runtime.fake_provider import (
        FakeBrisbane, FakeKyiv, FakeSherbrooke, FakeTorino,
    )
    
    registry = {
        "FakeBrisbane": FakeBrisbane,
        "FakeKyiv": FakeKyiv,
        "FakeSherbrooke": FakeSherbrooke,
        "FakeTorino": FakeTorino,
    }
    
    fake = registry[backend_name]()
    target = fake.target
    n = fake.num_qubits
    
    t1_us = []
    t2_us = []
    readout_error = []
    gate_error_1q = []
    gate_error_2q = {}
    
    for q in range(n):
        qp = target.qubit_properties[q] if target.qubit_properties else None
        t1_us.append((qp.t1 * 1e6) if (qp and qp.t1) else 100.0)
        t2_us.append((qp.t2 * 1e6) if (qp and qp.t2) else 80.0)
        
        # Readout error
        re = 0.02
        if "measure" in target.operation_names:
            mp = target["measure"].get((q,))
            if mp and mp.error is not None:
                re = mp.error
        readout_error.append(re)
        
        # 1Q gate error (use sx gate as representative)
        ge = 0.001
        for gate_name in ["sx", "x", "rz"]:
            if gate_name in target.operation_names:
                gp = target[gate_name].get((q,))
                if gp and gp.error is not None:
                    ge = gp.error
                    break
        gate_error_1q.append(ge)
    
    # 2Q gate errors
    coupling_map = list(fake.coupling_map.get_edges()) if fake.coupling_map else []
    for edge in coupling_map:
        for gate_name in ["ecr", "cx", "cz"]:
            if gate_name in target.operation_names:
                gp = target[gate_name].get(edge)
                if gp and gp.error is not None:
                    gate_error_2q[edge] = gp.error
                    break
        if edge not in gate_error_2q:
            gate_error_2q[edge] = 0.01
    
    return CalibrationSnapshot(
        timestamp=time.time(),
        backend_name=backend_name,
        num_qubits=n,
        t1_us=t1_us,
        t2_us=t2_us,
        readout_error=readout_error,
        gate_error_1q=gate_error_1q,
        gate_error_2q=gate_error_2q,
        coupling_map=coupling_map,
    )


def generate_drift_series(
    base_snapshot: CalibrationSnapshot,
    num_snapshots: int = 168,        # 1 week at hourly intervals
    interval_seconds: float = 3600,  # 1 hour between snapshots
    drift_rate: float = 0.02,        # relative drift per step (std)
    jump_prob: float = 0.03,         # probability of sudden jump per qubit per step
    jump_magnitude: float = 0.3,     # relative magnitude of jumps
    seed: Optional[int] = None,
) -> list[CalibrationSnapshot]:
    """Generate a time series of calibration snapshots with realistic drift.
    
    Drift model:
      - Gradual: Ornstein-Uhlenbeck process (mean-reverting random walk)
      - Jumps: Poisson-distributed sudden parameter changes
      - Correlated: T1 drop often causes T2 drop and gate error increase
    
    Args:
        base_snapshot: Starting calibration state
        num_snapshots: Number of snapshots to generate
        interval_seconds: Time between snapshots
        drift_rate: Std of relative change per step (OU noise)
        jump_prob: Probability per qubit per step of a sudden jump
        jump_magnitude: Relative size of sudden jumps
        seed: Random seed for reproducibility
    
    Returns:
        List of CalibrationSnapshot with increasing timestamps
    """
    rng = np.random.default_rng(seed)
    n = base_snapshot.num_qubits
    
    # Work with arrays for efficiency
    t1 = np.array(base_snapshot.t1_us, dtype=float)
    t2 = np.array(base_snapshot.t2_us, dtype=float)
    re = np.array(base_snapshot.readout_error, dtype=float)
    ge1q = np.array(base_snapshot.gate_error_1q, dtype=float)
    
    edges = list(base_snapshot.gate_error_2q.keys())
    ge2q = np.array([base_snapshot.gate_error_2q[e] for e in edges], dtype=float)
    
    # Store baseline for mean-reversion
    t1_base = t1.copy()
    t2_base = t2.copy()
    re_base = re.copy()
    ge1q_base = ge1q.copy()
    ge2q_base = ge2q.copy()
    
    # OU reversion speed
    theta = 0.05  # mean-reversion rate
    
    snapshots = []
    base_time = base_snapshot.timestamp
    
    for step in range(num_snapshots):
        # --- Ornstein-Uhlenbeck drift ---
        # dX = theta * (mu - X) * dt + sigma * dW
        t1 += theta * (t1_base - t1) + drift_rate * t1_base * rng.standard_normal(n)
        t2 += theta * (t2_base - t2) + drift_rate * t2_base * rng.standard_normal(n)
        re += theta * (re_base - re) + drift_rate * 0.01 * rng.standard_normal(n)
        ge1q += theta * (ge1q_base - ge1q) + drift_rate * 0.001 * rng.standard_normal(n)
        if len(ge2q) > 0:
            ge2q += theta * (ge2q_base - ge2q) + drift_rate * 0.005 * rng.standard_normal(len(ge2q))
        
        # --- Sudden jumps ---
        jump_mask = rng.random(n) < jump_prob
        if jump_mask.any():
            jump_qubits = np.where(jump_mask)[0]
            for q in jump_qubits:
                # T1 drop -> T2 and gate error correlated
                t1_factor = 1.0 - jump_magnitude * rng.random()
                t1[q] *= t1_factor
                t2[q] *= t1_factor * (0.8 + 0.4 * rng.random())  # correlated
                ge1q[q] *= (1.0 + jump_magnitude * rng.random())
                re[q] *= (1.0 + jump_magnitude * 0.5 * rng.random())
        
        # --- Enforce physical bounds ---
        t1 = np.clip(t1, 5.0, 500.0)     # T1: 5–500 us
        t2 = np.minimum(t2, 2 * t1)       # T2 <= 2*T1 (physical constraint)
        t2 = np.clip(t2, 2.0, 400.0)
        re = np.clip(re, 0.001, 0.5)
        ge1q = np.clip(ge1q, 1e-5, 0.1)
        if len(ge2q) > 0:
            ge2q = np.clip(ge2q, 1e-4, 0.5)
        
        snap = CalibrationSnapshot(
            timestamp=base_time + (step + 1) * interval_seconds,
            backend_name=base_snapshot.backend_name,
            num_qubits=n,
            t1_us=t1.tolist(),
            t2_us=t2.tolist(),
            readout_error=re.tolist(),
            gate_error_1q=ge1q.tolist(),
            gate_error_2q={edges[i]: float(ge2q[i]) for i in range(len(edges))},
            coupling_map=base_snapshot.coupling_map.copy(),
        )
        snapshots.append(snap)
    
    return snapshots
