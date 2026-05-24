"""Fidelity Predictor — data-driven circuit fidelity estimation.

Replaces the simple analytical model (product of gate errors) with a trained
XGBoost regressor. The model learns from (circuit_features, backend_state) → fidelity
samples collected across multiple circuits, backends, and drift conditions.

Usage:
    # Collect training data + train + evaluate:
    python eval/fidelity_predictor.py

    # Or use programmatically:
    from eval.fidelity_predictor import FidelityPredictor
    predictor = FidelityPredictor.load("eval/fidelity_model.json")
    pred = predictor.predict(circuit_features, backend_features)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).parent.parent))

from backends.fake_adapter import FakeBackendAdapter
from backends.synthetic_drift import (
    SyntheticDriftBackend,
    DriftProfile,
    STABLE,
    LINEAR_DECAY,
    SUDDEN_DEGRADATION,
    DIURNAL_CYCLE,
)
from bench.circuits import ghz, qft, bernstein_vazirani, vqe_ansatz, qaoa_maxcut


# ═══════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════

@dataclass
class CircuitFeatures:
    """Features extracted from a quantum circuit."""
    name: str
    num_qubits: int
    depth: int
    n_1q_gates: int
    n_2q_gates: int
    n_measurements: int
    gate_density: float       # total_gates / (num_qubits * depth)
    two_qubit_ratio: float    # n_2q / (n_1q + n_2q)
    connectivity: float       # unique qubit pairs used / possible pairs

    def to_array(self) -> NDArray[np.float64]:
        return np.array([
            self.num_qubits,
            self.depth,
            self.n_1q_gates,
            self.n_2q_gates,
            self.n_measurements,
            self.gate_density,
            self.two_qubit_ratio,
            self.connectivity,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "num_qubits", "depth", "n_1q_gates", "n_2q_gates",
            "n_measurements", "gate_density", "two_qubit_ratio", "connectivity",
        ]


@dataclass
class BackendFeatures:
    """Features extracted from backend state."""
    avg_1q_error: float
    avg_2q_error: float
    avg_readout_error: float
    avg_t1_us: float
    avg_t2_us: float
    drift_score: float

    def to_array(self) -> NDArray[np.float64]:
        return np.array([
            self.avg_1q_error,
            self.avg_2q_error,
            self.avg_readout_error,
            self.avg_t1_us,
            self.avg_t2_us,
            self.drift_score,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "avg_1q_error", "avg_2q_error", "avg_readout_error",
            "avg_t1_us", "avg_t2_us", "drift_score",
        ]


def extract_circuit_features(circuit, name: str) -> CircuitFeatures:
    """Extract features from a QuantumCircuit."""
    from qiskit import transpile as qiskit_transpile
    from qiskit_aer import AerSimulator

    # Transpile to get realistic gate counts (IBM Eagle gate set)
    try:
        backend = FakeBackendAdapter("FakeBrisbane")
        fake_be = backend._fake_backend
        transpiled = qiskit_transpile(circuit, backend=fake_be, optimization_level=1)
    except Exception:
        transpiled = circuit

    ops = transpiled.count_ops()
    n_1q = sum(v for k, v in ops.items() if k in ("rz", "sx", "x", "id", "u1", "u2", "u3", "h"))
    n_2q = sum(v for k, v in ops.items() if k in ("cx", "ecr", "cz", "swap"))
    n_meas = ops.get("measure", circuit.num_qubits)
    depth = transpiled.depth()
    n_qubits = transpiled.num_qubits

    total_gates = n_1q + n_2q
    gate_density = total_gates / max(1, n_qubits * depth)
    two_qubit_ratio = n_2q / max(1, total_gates)

    # Connectivity: count unique qubit pairs in 2Q gates
    pairs_used = set()
    for inst in transpiled.data:
        if len(inst.qubits) == 2:
            q0 = transpiled.find_bit(inst.qubits[0]).index
            q1 = transpiled.find_bit(inst.qubits[1]).index
            pairs_used.add((min(q0, q1), max(q0, q1)))
    possible_pairs = n_qubits * (n_qubits - 1) / 2
    connectivity = len(pairs_used) / max(1, possible_pairs)

    return CircuitFeatures(
        name=name,
        num_qubits=n_qubits,
        depth=depth,
        n_1q_gates=n_1q,
        n_2q_gates=n_2q,
        n_measurements=n_meas,
        gate_density=gate_density,
        two_qubit_ratio=two_qubit_ratio,
        connectivity=connectivity,
    )


def extract_backend_features(backend) -> BackendFeatures:
    """Extract features from a ShadowBackend."""
    h = backend.get_health()
    return BackendFeatures(
        avg_1q_error=h.avg_1q_error,
        avg_2q_error=h.avg_2q_error,
        avg_readout_error=h.avg_readout_error,
        avg_t1_us=h.avg_t1_us,
        avg_t2_us=h.avg_t2_us,
        drift_score=h.drift_score if h.drift_score is not None else 0.0,
    )


# ═══════════════════════════════════════════════════════
# Data collection
# ═══════════════════════════════════════════════════════

# All circuits to sweep
CIRCUITS = {
    "ghz_3": lambda: ghz(3),
    "ghz_5": lambda: ghz(5),
    "qft_4": lambda: qft(4),
    "bv_5": lambda: bernstein_vazirani("10110"),
    "vqe_4": lambda: vqe_ansatz(4),
    "qaoa_4": lambda: qaoa_maxcut(4),
}

# Drift conditions to sweep
DRIFT_CONDITIONS = [
    ("stable", STABLE, [0.0]),
    ("linear_decay", LINEAR_DECAY, [0, 2, 5, 10, 15, 20]),
    ("sudden", SUDDEN_DEGRADATION, [0, 3, 5, 6, 8, 10]),
    ("diurnal", DIURNAL_CYCLE, [0, 3, 6, 9, 12, 15, 18, 21, 24]),
]

# Backends
BACKEND_NAMES = ["FakeBrisbane", "FakeKyiv", "FakeSherbrooke"]


def collect_training_data(
    shots: int = 4096,
    repeats: int = 3,
    verbose: bool = True,
) -> list[dict]:
    """Collect (features, fidelity) samples across circuits × backends × drift conditions.

    Returns list of dicts with keys: circuit_features, backend_features, fidelity, metadata.
    """
    samples = []
    total_expected = (
        len(CIRCUITS) * len(BACKEND_NAMES) * sum(len(c[2]) for c in DRIFT_CONDITIONS) * repeats
    )
    count = 0

    # Pre-extract circuit features (only need to do once per circuit)
    circuit_features_cache = {}
    for cname, builder in CIRCUITS.items():
        circ, ideal = builder()
        circuit_features_cache[cname] = extract_circuit_features(circ, cname)

    for backend_name in BACKEND_NAMES:
        for drift_name, profile, time_points in DRIFT_CONDITIONS:
            backend = SyntheticDriftBackend(backend_name, profile)

            for t in time_points:
                backend.set_time(t)
                bf = extract_backend_features(backend)

                for cname, builder in CIRCUITS.items():
                    circ, ideal = builder()
                    cf = circuit_features_cache[cname]

                    for rep in range(repeats):
                        try:
                            result = backend.run(circ, shots=shots)
                            fidelity = result.fidelity if result.fidelity is not None else 0.0
                        except Exception as e:
                            if verbose:
                                print(f"  ERROR: {cname}/{backend_name}/t={t}: {e}")
                            fidelity = 0.0

                        samples.append({
                            "circuit_features": cf.to_array().tolist(),
                            "backend_features": bf.to_array().tolist(),
                            "fidelity": fidelity,
                            "metadata": {
                                "circuit": cname,
                                "backend": backend_name,
                                "drift_profile": drift_name,
                                "time_hours": t,
                                "repeat": rep,
                            },
                        })

                        count += 1
                        if verbose and count % 50 == 0:
                            print(f"  [{count}/{total_expected}] {cname}/{backend_name}"
                                  f" drift={drift_name} t={t}h fidelity={fidelity:.4f}")

    if verbose:
        print(f"\nCollected {len(samples)} samples total.")
    return samples


# ═══════════════════════════════════════════════════════
# Model training
# ═══════════════════════════════════════════════════════

class FidelityPredictor:
    """XGBoost-based fidelity predictor.

    Features: 8 circuit features + 6 backend features = 14 total.
    Target: fidelity ∈ [0, 1].
    """

    def __init__(self):
        self.model = None
        self.feature_names = CircuitFeatures.feature_names() + BackendFeatures.feature_names()
        self.train_stats: dict[str, Any] = {}

    def train(self, samples: list[dict], test_fraction: float = 0.2, verbose: bool = True):
        """Train the XGBoost model on collected samples."""
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

        # Build feature matrix
        X = np.array([s["circuit_features"] + s["backend_features"] for s in samples])
        y = np.array([s["fidelity"] for s in samples])

        if verbose:
            print(f"\nTraining data: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"Fidelity range: [{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=42
        )

        # Train XGBoost
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        self.train_stats = {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
            "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
            "train_r2": float(r2_score(y_train, y_pred_train)),
            "test_r2": float(r2_score(y_test, y_pred_test)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            "feature_importance": dict(zip(
                self.feature_names,
                [float(x) for x in self.model.feature_importances_],
            )),
        }

        if verbose:
            print(f"\n{'='*50}")
            print(f"TRAINING RESULTS")
            print(f"{'='*50}")
            print(f"  Train MAE:  {self.train_stats['train_mae']:.4f}")
            print(f"  Test  MAE:  {self.train_stats['test_mae']:.4f}")
            print(f"  Train R²:   {self.train_stats['train_r2']:.4f}")
            print(f"  Test  R²:   {self.train_stats['test_r2']:.4f}")
            print(f"  Train RMSE: {self.train_stats['train_rmse']:.4f}")
            print(f"  Test  RMSE: {self.train_stats['test_rmse']:.4f}")
            print(f"\nFeature importance (top 5):")
            sorted_fi = sorted(
                self.train_stats["feature_importance"].items(),
                key=lambda x: -x[1]
            )
            for fname, imp in sorted_fi[:5]:
                print(f"  {fname:20s} {imp:.4f}")

        return self.train_stats

    def predict(self, circuit_features: NDArray, backend_features: NDArray) -> float:
        """Predict fidelity for a single sample."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X = np.concatenate([circuit_features, backend_features]).reshape(1, -1)
        return float(self.model.predict(X)[0])

    def predict_batch(self, X: NDArray) -> NDArray:
        """Predict fidelity for a batch."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def save(self, path: str):
        """Save model + stats to disk."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        self.model.save_model(path)
        stats_path = path.replace(".json", "_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.train_stats, f, indent=2)
        print(f"Model saved to {path}")
        print(f"Stats saved to {stats_path}")

    @classmethod
    def load(cls, path: str) -> "FidelityPredictor":
        """Load a trained model from disk."""
        import xgboost as xgb
        predictor = cls()
        predictor.model = xgb.XGBRegressor()
        predictor.model.load_model(path)
        stats_path = path.replace(".json", "_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                predictor.train_stats = json.load(f)
        return predictor


# ═══════════════════════════════════════════════════════
# Comparison: analytical vs ML predictor
# ═══════════════════════════════════════════════════════

def analytical_predict(circuit_features: CircuitFeatures, backend_features: BackendFeatures) -> float:
    """Simple analytical model: F = (1-e1)^n1 * (1-e2)^n2 * (1-er)^nm."""
    f_1q = (1 - backend_features.avg_1q_error) ** circuit_features.n_1q_gates
    f_2q = (1 - backend_features.avg_2q_error) ** circuit_features.n_2q_gates
    f_ro = (1 - backend_features.avg_readout_error) ** circuit_features.n_measurements
    return f_1q * f_2q * f_ro


def compare_predictors(samples: list[dict], predictor: FidelityPredictor, verbose: bool = True):
    """Compare XGBoost vs analytical predictor on the same test set."""
    from sklearn.metrics import mean_absolute_error, r2_score

    y_true = []
    y_analytical = []
    y_xgb = []

    for s in samples:
        cf_arr = np.array(s["circuit_features"])
        bf_arr = np.array(s["backend_features"])

        # Reconstruct feature objects for analytical model
        cf = CircuitFeatures(
            name="", num_qubits=int(cf_arr[0]), depth=int(cf_arr[1]),
            n_1q_gates=int(cf_arr[2]), n_2q_gates=int(cf_arr[3]),
            n_measurements=int(cf_arr[4]), gate_density=cf_arr[5],
            two_qubit_ratio=cf_arr[6], connectivity=cf_arr[7],
        )
        bf = BackendFeatures(
            avg_1q_error=bf_arr[0], avg_2q_error=bf_arr[1],
            avg_readout_error=bf_arr[2], avg_t1_us=bf_arr[3],
            avg_t2_us=bf_arr[4], drift_score=bf_arr[5],
        )

        y_true.append(s["fidelity"])
        y_analytical.append(analytical_predict(cf, bf))
        y_xgb.append(predictor.predict(cf_arr, bf_arr))

    y_true = np.array(y_true)
    y_analytical = np.array(y_analytical)
    y_xgb = np.array(y_xgb)

    results = {
        "analytical": {
            "mae": float(mean_absolute_error(y_true, y_analytical)),
            "r2": float(r2_score(y_true, y_analytical)),
        },
        "xgboost": {
            "mae": float(mean_absolute_error(y_true, y_xgb)),
            "r2": float(r2_score(y_true, y_xgb)),
        },
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"PREDICTOR COMPARISON")
        print(f"{'='*50}")
        print(f"  {'Method':<15} {'MAE':>8} {'R²':>8}")
        print(f"  {'-'*35}")
        print(f"  {'Analytical':<15} {results['analytical']['mae']:>8.4f} {results['analytical']['r2']:>8.4f}")
        print(f"  {'XGBoost':<15} {results['xgboost']['mae']:>8.4f} {results['xgboost']['r2']:>8.4f}")
        improvement = (results['analytical']['mae'] - results['xgboost']['mae']) / results['analytical']['mae'] * 100
        print(f"\n  XGBoost MAE improvement: {improvement:.1f}%")

    return results


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("QuantumGPT Fidelity Predictor — Data Collection + Training")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Collect data
    print("\n[1/4] Collecting training data...")
    print(f"  Circuits: {list(CIRCUITS.keys())}")
    print(f"  Backends: {BACKEND_NAMES}")
    print(f"  Drift profiles: {[d[0] for d in DRIFT_CONDITIONS]}")
    samples = collect_training_data(shots=4096, repeats=3, verbose=True)

    # Save raw data
    data_path = Path(__file__).parent / "fidelity_data.json"
    with open(data_path, "w") as f:
        json.dump(samples, f)
    print(f"\n  Raw data saved to {data_path}")

    # Step 2: Train model
    print("\n[2/4] Training XGBoost model...")
    predictor = FidelityPredictor()
    stats = predictor.train(samples, test_fraction=0.2, verbose=True)

    # Step 3: Save model
    print("\n[3/4] Saving model...")
    model_path = str(Path(__file__).parent / "fidelity_model.json")
    predictor.save(model_path)

    # Step 4: Compare with analytical
    print("\n[4/4] Comparing with analytical baseline...")
    comparison = compare_predictors(samples, predictor, verbose=True)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"{'='*60}")
