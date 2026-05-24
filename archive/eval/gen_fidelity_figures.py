"""Generate figures for the fidelity predictor evaluation.

Produces:
  - eval/fidelity_scatter.png: predicted vs actual fidelity (XGBoost vs analytical)
  - eval/fidelity_importance.png: feature importance bar chart
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.fidelity_predictor import (
    FidelityPredictor, CircuitFeatures, BackendFeatures, analytical_predict
)


def main():
    # Load data
    data_path = Path(__file__).parent / "fidelity_data.json"
    with open(data_path) as f:
        samples = json.load(f)

    # Load model
    model_path = str(Path(__file__).parent / "fidelity_model.json")
    predictor = FidelityPredictor.load(model_path)

    # Compute predictions
    y_true = []
    y_xgb = []
    y_analytical = []
    circuits = []

    for s in samples:
        cf_arr = np.array(s["circuit_features"])
        bf_arr = np.array(s["backend_features"])

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
        y_xgb.append(predictor.predict(cf_arr, bf_arr))
        y_analytical.append(analytical_predict(cf, bf))
        circuits.append(s["metadata"]["circuit"])

    y_true = np.array(y_true)
    y_xgb = np.array(y_xgb)
    y_analytical = np.array(y_analytical)

    # ─── Figure 1: Scatter plot ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)

    # XGBoost
    ax = axes[0]
    ax.scatter(y_true, y_xgb, alpha=0.6, s=20, c="#2563eb", edgecolors="none")
    ax.plot([0.7, 1.0], [0.7, 1.0], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Actual Fidelity", fontsize=10)
    ax.set_ylabel("Predicted Fidelity", fontsize=10)
    ax.set_title("XGBoost Predictor", fontsize=11, fontweight="bold")
    ax.set_xlim(0.7, 1.01)
    ax.set_ylim(0.7, 1.01)
    ax.grid(True, alpha=0.2)

    from sklearn.metrics import mean_absolute_error, r2_score
    mae_xgb = mean_absolute_error(y_true, y_xgb)
    r2_xgb = r2_score(y_true, y_xgb)
    ax.text(0.05, 0.92, f"MAE = {mae_xgb:.4f}\nR² = {r2_xgb:.3f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Analytical
    ax = axes[1]
    ax.scatter(y_true, y_analytical, alpha=0.6, s=20, c="#dc2626", edgecolors="none")
    ax.plot([0, 1.0], [0, 1.0], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Actual Fidelity", fontsize=10)
    ax.set_ylabel("Predicted Fidelity", fontsize=10)
    ax.set_title("Analytical Model", fontsize=11, fontweight="bold")
    ax.set_xlim(0.7, 1.01)
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, alpha=0.2)

    mae_an = mean_absolute_error(y_true, y_analytical)
    r2_an = r2_score(y_true, y_analytical)
    ax.text(0.05, 0.92, f"MAE = {mae_an:.4f}\nR² = {r2_an:.3f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    out1 = Path(__file__).parent / "fidelity_scatter.png"
    plt.savefig(out1, bbox_inches="tight", facecolor="white")
    plt.savefig(str(out1).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.close()

    # ─── Figure 2: Feature importance ─────────────────────────────
    stats_path = Path(__file__).parent / "fidelity_model_stats.json"
    with open(stats_path) as f:
        stats = json.load(f)

    fi = stats["feature_importance"]
    names = list(fi.keys())
    values = list(fi.values())

    # Sort by importance
    sorted_idx = np.argsort(values)[::-1]
    names_sorted = [names[i] for i in sorted_idx]
    values_sorted = [values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    colors = ["#2563eb" if "avg_" in n or "drift" in n else "#059669" for n in names_sorted]
    bars = ax.barh(range(len(names_sorted)), values_sorted, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.set_xlabel("Feature Importance (gain)", fontsize=10)
    ax.set_title("XGBoost Feature Importance", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.2)

    # Legend
    legend_elements = [
        Patch(facecolor="#059669", alpha=0.8, label="Circuit features"),
        Patch(facecolor="#2563eb", alpha=0.8, label="Backend features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    out2 = Path(__file__).parent / "fidelity_importance.png"
    plt.savefig(out2, bbox_inches="tight", facecolor="white")
    plt.savefig(str(out2).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close()

    print("\nDone. Summary:")
    print(f"  XGBoost:    MAE={mae_xgb:.4f}, R²={r2_xgb:.3f}")
    print(f"  Analytical: MAE={mae_an:.4f}, R²={r2_an:.3f}")
    print(f"  Improvement: {(mae_an - mae_xgb) / mae_an * 100:.1f}%")


if __name__ == "__main__":
    main()
