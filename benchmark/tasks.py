"""QC-Agent-Bench — Quantum Computing Agent Benchmark.

60 tasks across 3 tiers:
  Tier 1 (Static, 30 tasks): Standard quantum computing tasks on stable backends
  Tier 2 (Drift, 20 tasks): Tasks under hardware drift conditions
  Tier 3 (Failure, 10 tasks): Tasks requiring fault diagnosis and recovery

Each task is a JSON spec with:
  - id: unique identifier (e.g., "T1-01")
  - tier: 1/2/3
  - name: human-readable name
  - prompt: natural language instruction to the agent
  - backend: which backend to use
  - drift_profile: (tier 2/3) drift condition
  - drift_time: (tier 2/3) simulated time point
  - success_criteria: what constitutes success
  - ground_truth: expected correct answer/behavior
  - metrics: which metrics to evaluate
  - difficulty: easy/medium/hard
  - tags: categorization tags

Usage:
    from benchmark.tasks import TASKS, get_tier, get_task
    tier1 = get_tier(1)  # 30 tasks
    task = get_task("T1-01")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BenchmarkTask:
    """A single benchmark task specification."""
    id: str
    tier: int
    name: str
    prompt: str
    backend: str = "FakeBrisbane"
    drift_profile: Optional[str] = None
    drift_time: Optional[float] = None
    success_criteria: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=lambda: ["success", "fidelity", "tool_calls", "wall_time"])
    difficulty: str = "medium"
    tags: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════
# TIER 1: Static Tasks (30)
# Standard quantum computing on stable backends
# ═══════════════════════════════════════════════════════

TIER_1_TASKS = [
    # --- Health & Monitoring (5) ---
    BenchmarkTask(
        id="T1-01", tier=1, name="Basic Health Check",
        prompt="Check the backend health and report the average gate errors and T1/T2 times.",
        success_criteria={"must_call": ["get_backend_health"], "must_report": ["avg_1q_error", "avg_t1_us"]},
        ground_truth={"avg_1q_error_range": [0.001, 0.01], "avg_t1_us_range": [100, 400]},
        difficulty="easy", tags=["health", "monitoring"],
    ),
    BenchmarkTask(
        id="T1-02", tier=1, name="Qubit Selection",
        prompt="Find the 5 best qubits (lowest error rates) on this backend for running a 5-qubit circuit.",
        success_criteria={"must_call": ["get_qubit_properties"], "must_report": ["selected_qubits"]},
        ground_truth={"n_qubits_selected": 5},
        difficulty="medium", tags=["health", "qubit-selection"],
    ),
    BenchmarkTask(
        id="T1-03", tier=1, name="Backend Comparison",
        prompt="Compare FakeBrisbane and FakeSherbrooke. Which is better for running a 5-qubit GHZ circuit?",
        backend="FakeBrisbane",
        success_criteria={"must_call": ["get_backend_health"], "must_report": ["recommendation"]},
        difficulty="medium", tags=["health", "comparison"],
    ),
    BenchmarkTask(
        id="T1-04", tier=1, name="Calibration Age Check",
        prompt="Check how old the calibration data is. If older than 30 minutes, flag it.",
        success_criteria={"must_call": ["get_backend_health"], "must_report": ["calibration_age"]},
        difficulty="easy", tags=["health", "calibration"],
    ),
    BenchmarkTask(
        id="T1-05", tier=1, name="Coupling Map Analysis",
        prompt="Get the coupling map and identify the most connected qubit (highest degree).",
        success_criteria={"must_call": ["get_coupling_map"], "must_report": ["most_connected_qubit"]},
        difficulty="medium", tags=["health", "topology"],
    ),

    # --- Circuit Execution (10) ---
    BenchmarkTask(
        id="T1-06", tier=1, name="Run GHZ-3",
        prompt="Run the GHZ-3 circuit with 4096 shots and report the fidelity.",
        success_criteria={"must_call": ["run_circuit"], "fidelity_above": 0.85},
        ground_truth={"expected_fidelity_range": [0.88, 0.98]},
        difficulty="easy", tags=["execution", "ghz"],
    ),
    BenchmarkTask(
        id="T1-07", tier=1, name="Run GHZ-5",
        prompt="Run the GHZ-5 circuit with 8192 shots and report the fidelity.",
        success_criteria={"must_call": ["run_circuit"], "fidelity_above": 0.80},
        ground_truth={"expected_fidelity_range": [0.85, 0.95]},
        difficulty="easy", tags=["execution", "ghz"],
    ),
    BenchmarkTask(
        id="T1-08", tier=1, name="Run QFT-4",
        prompt="Run the QFT-4 circuit and report the fidelity.",
        success_criteria={"must_call": ["run_circuit"], "fidelity_above": 0.75},
        ground_truth={"expected_fidelity_range": [0.80, 0.92]},
        difficulty="easy", tags=["execution", "qft"],
    ),
    BenchmarkTask(
        id="T1-09", tier=1, name="Run BV-5",
        prompt="Run the Bernstein-Vazirani circuit (secret=10110) and verify the output.",
        success_criteria={"must_call": ["run_circuit"], "fidelity_above": 0.80},
        ground_truth={"expected_output": "10110"},
        difficulty="easy", tags=["execution", "bv"],
    ),
    BenchmarkTask(
        id="T1-10", tier=1, name="Run VQE-4",
        prompt="Run the VQE-4 hardware-efficient ansatz and report the fidelity.",
        success_criteria={"must_call": ["run_circuit"], "fidelity_above": 0.75},
        difficulty="easy", tags=["execution", "vqe"],
    ),
    BenchmarkTask(
        id="T1-11", tier=1, name="Run QAOA-4",
        prompt="Run the QAOA MaxCut circuit on 4 qubits and report the fidelity.",
        success_criteria={"must_call": ["run_circuit"], "fidelity_above": 0.70},
        difficulty="easy", tags=["execution", "qaoa"],
    ),
    BenchmarkTask(
        id="T1-12", tier=1, name="Multi-Circuit Benchmark",
        prompt="Run ghz_5, qft_4, and bv_5 circuits. Compare their fidelities and rank them.",
        success_criteria={"min_circuits_run": 3, "must_report": ["ranking"]},
        difficulty="medium", tags=["execution", "comparison"],
    ),
    BenchmarkTask(
        id="T1-13", tier=1, name="High-Shot Execution",
        prompt="Run GHZ-5 with 16384 shots for high statistical confidence. Report fidelity with error bars.",
        success_criteria={"must_call": ["run_circuit"], "shots_min": 8192},
        difficulty="easy", tags=["execution", "statistics"],
    ),
    BenchmarkTask(
        id="T1-14", tier=1, name="Circuit on FakeKyiv",
        prompt="Run the GHZ-5 circuit on FakeKyiv backend and compare with FakeBrisbane.",
        backend="FakeKyiv",
        success_criteria={"must_call": ["run_circuit"]},
        difficulty="medium", tags=["execution", "cross-backend"],
    ),
    BenchmarkTask(
        id="T1-15", tier=1, name="All Circuits Sweep",
        prompt="Run all available benchmark circuits and create a fidelity summary table.",
        success_criteria={"min_circuits_run": 5},
        difficulty="medium", tags=["execution", "sweep"],
    ),

    # --- Prediction & Analysis (5) ---
    BenchmarkTask(
        id="T1-16", tier=1, name="Fidelity Prediction",
        prompt="Predict the fidelity of ghz_5 before running it. Then run it and compare.",
        success_criteria={"must_call": ["predict_fidelity", "run_circuit"]},
        ground_truth={"prediction_error_max": 0.1},
        difficulty="medium", tags=["prediction", "verification"],
    ),
    BenchmarkTask(
        id="T1-17", tier=1, name="Predict All Circuits",
        prompt="Predict fidelities for ghz_5, qft_4, bv_5, vqe_4, qaoa_4. Which will perform best?",
        success_criteria={"must_call": ["predict_fidelity"], "min_predictions": 3},
        difficulty="medium", tags=["prediction", "ranking"],
    ),
    BenchmarkTask(
        id="T1-18", tier=1, name="Transpile Analysis",
        prompt="Transpile the QFT-4 circuit at optimization levels 0, 1, and 2. Compare depths.",
        success_criteria={"must_call": ["transpile_circuit"], "min_transpilations": 2},
        difficulty="medium", tags=["transpilation", "optimization"],
    ),
    BenchmarkTask(
        id="T1-19", tier=1, name="Error Budget Analysis",
        prompt="For the GHZ-5 circuit, break down the error sources: 1Q gates, 2Q gates, readout.",
        success_criteria={"must_call": ["predict_fidelity"], "must_report": ["error_breakdown"]},
        difficulty="medium", tags=["analysis", "error-budget"],
    ),
    BenchmarkTask(
        id="T1-20", tier=1, name="Depth vs Fidelity",
        prompt="Run circuits of increasing depth (ghz_3, ghz_5, qft_4) and analyze the depth-fidelity relationship.",
        success_criteria={"min_circuits_run": 3, "must_report": ["trend"]},
        difficulty="hard", tags=["analysis", "scaling"],
    ),

    # --- Mitigation & Diagnosis (5) ---
    BenchmarkTask(
        id="T1-21", tier=1, name="Basic Diagnosis",
        prompt="Run GHZ-5. If fidelity is below 0.95, diagnose the issue.",
        success_criteria={"must_call": ["run_circuit", "diagnose_and_suggest"]},
        difficulty="medium", tags=["diagnosis"],
    ),
    BenchmarkTask(
        id="T1-22", tier=1, name="Error Mitigation (ZNE)",
        prompt="Run GHZ-5 and apply zero-noise extrapolation to improve the fidelity.",
        success_criteria={"must_call": ["apply_mitigation"]},
        ground_truth={"mitigation_improves": True},
        difficulty="medium", tags=["mitigation", "zne"],
    ),
    BenchmarkTask(
        id="T1-23", tier=1, name="Diagnose + Mitigate",
        prompt="Run GHZ-5, diagnose any issues, and apply the recommended mitigation strategy.",
        success_criteria={"must_call": ["diagnose_and_suggest", "apply_mitigation"]},
        difficulty="hard", tags=["diagnosis", "mitigation"],
    ),
    BenchmarkTask(
        id="T1-24", tier=1, name="Mitigation Comparison",
        prompt="Run GHZ-5 with and without error mitigation. Report the improvement.",
        success_criteria={"must_call": ["run_circuit", "apply_mitigation"], "must_report": ["improvement"]},
        difficulty="hard", tags=["mitigation", "comparison"],
    ),
    BenchmarkTask(
        id="T1-25", tier=1, name="Full Diagnostic Pipeline",
        prompt="Check health, run GHZ-5, predict fidelity, diagnose, mitigate, and report final fidelity.",
        success_criteria={"min_tool_calls": 4, "must_call": ["get_backend_health", "run_circuit"]},
        difficulty="hard", tags=["pipeline", "full"],
    ),

    # --- Pulse-Level (5) ---
    BenchmarkTask(
        id="T1-26", tier=1, name="Rabi Oscillation",
        prompt="Run a Rabi oscillation experiment and extract the pi-pulse amplitude.",
        success_criteria={"must_call": ["rabi_experiment"], "must_report": ["pi_amplitude"]},
        ground_truth={"pi_amplitude_range": [4.0, 6.0]},
        difficulty="medium", tags=["pulse", "rabi"],
    ),
    BenchmarkTask(
        id="T1-27", tier=1, name="Rabi + Fit",
        prompt="Run a Rabi experiment, fit the oscillation data, and report the Rabi frequency.",
        success_criteria={"must_call": ["rabi_experiment", "fit_rabi"]},
        ground_truth={"rabi_freq_range": [20, 30]},
        difficulty="medium", tags=["pulse", "rabi", "fitting"],
    ),
    BenchmarkTask(
        id="T1-28", tier=1, name="Rabi with Gaussian Pulse",
        prompt="Run a Rabi experiment using a gaussian pulse shape at 5 GHz qubit frequency.",
        success_criteria={"must_call": ["rabi_experiment"]},
        difficulty="medium", tags=["pulse", "rabi", "gaussian"],
    ),
    BenchmarkTask(
        id="T1-29", tier=1, name="Characterization Suite",
        prompt="Run Rabi, then use the pi-amplitude to characterize the qubit. Report T1 estimate.",
        success_criteria={"must_call": ["rabi_experiment"]},
        difficulty="hard", tags=["pulse", "characterization"],
    ),
    BenchmarkTask(
        id="T1-30", tier=1, name="Cross-Layer Task",
        prompt="First run a Rabi experiment to calibrate, then run GHZ-5 and report if calibration helps.",
        success_criteria={"must_call": ["rabi_experiment", "run_circuit"]},
        difficulty="hard", tags=["pulse", "cross-layer"],
    ),
]


# ═══════════════════════════════════════════════════════
# TIER 2: Drift Tasks (20)
# Tasks under hardware drift conditions
# ═══════════════════════════════════════════════════════

TIER_2_TASKS = [
    # --- Drift Detection (5) ---
    BenchmarkTask(
        id="T2-01", tier=2, name="Detect Linear Drift",
        prompt="Monitor the backend and determine if there is drift. Report the drift score.",
        drift_profile="linear_decay", drift_time=10.0,
        success_criteria={"must_call": ["get_backend_health"], "must_detect_drift": True},
        difficulty="medium", tags=["drift", "detection"],
    ),
    BenchmarkTask(
        id="T2-02", tier=2, name="Detect Sudden Drift",
        prompt="Check if the backend has experienced sudden degradation.",
        drift_profile="sudden", drift_time=6.0,
        success_criteria={"must_detect_drift": True, "drift_score_above": 0.5},
        difficulty="medium", tags=["drift", "detection", "sudden"],
    ),
    BenchmarkTask(
        id="T2-03", tier=2, name="Stable vs Drifted",
        prompt="Compare the backend state now vs its nominal state. Is it drifting?",
        drift_profile="diurnal", drift_time=12.0,
        success_criteria={"must_report": ["drift_status"]},
        difficulty="medium", tags=["drift", "comparison"],
    ),
    BenchmarkTask(
        id="T2-04", tier=2, name="Drift Severity Assessment",
        prompt="Assess the severity of the current drift. Is it mild, moderate, or severe?",
        drift_profile="linear_decay", drift_time=15.0,
        success_criteria={"must_report": ["severity"]},
        difficulty="medium", tags=["drift", "severity"],
    ),
    BenchmarkTask(
        id="T2-05", tier=2, name="Early Drift Warning",
        prompt="Check if drift is beginning. Report even mild drift (score > 0.1).",
        drift_profile="linear_decay", drift_time=3.0,
        success_criteria={"must_report": ["drift_score"]},
        difficulty="hard", tags=["drift", "early-warning"],
    ),

    # --- Execution Under Drift (8) ---
    BenchmarkTask(
        id="T2-06", tier=2, name="GHZ-5 Under Linear Drift",
        prompt="Run GHZ-5 and report the fidelity. Note any degradation from expected.",
        drift_profile="linear_decay", drift_time=10.0,
        success_criteria={"must_call": ["run_circuit"], "must_report": ["fidelity"]},
        ground_truth={"fidelity_degraded": True, "expected_fidelity_range": [0.75, 0.90]},
        difficulty="medium", tags=["drift", "execution"],
    ),
    BenchmarkTask(
        id="T2-07", tier=2, name="GHZ-5 Under Sudden Drift",
        prompt="Run GHZ-5 and report the fidelity.",
        drift_profile="sudden", drift_time=6.0,
        success_criteria={"must_call": ["run_circuit"]},
        ground_truth={"fidelity_degraded": True, "expected_fidelity_range": [0.50, 0.80]},
        difficulty="medium", tags=["drift", "execution", "sudden"],
    ),
    BenchmarkTask(
        id="T2-08", tier=2, name="Multi-Circuit Under Drift",
        prompt="Run ghz_5, qft_4, bv_5 under current conditions. Which is most affected?",
        drift_profile="linear_decay", drift_time=12.0,
        success_criteria={"min_circuits_run": 3, "must_report": ["most_affected"]},
        difficulty="hard", tags=["drift", "execution", "comparison"],
    ),
    BenchmarkTask(
        id="T2-09", tier=2, name="Predict Under Drift",
        prompt="Predict fidelity of GHZ-5 considering current drift, then verify.",
        drift_profile="linear_decay", drift_time=8.0,
        success_criteria={"must_call": ["predict_fidelity", "run_circuit"]},
        difficulty="hard", tags=["drift", "prediction"],
    ),
    BenchmarkTask(
        id="T2-10", tier=2, name="Diurnal Cycle Execution",
        prompt="Run GHZ-5 at the current time point in the diurnal cycle. Report fidelity.",
        drift_profile="diurnal", drift_time=18.0,
        success_criteria={"must_call": ["run_circuit"]},
        difficulty="medium", tags=["drift", "diurnal"],
    ),
    BenchmarkTask(
        id="T2-11", tier=2, name="Drift-Aware Execution",
        prompt="Check for drift first. If drifting, adjust strategy before running GHZ-5.",
        drift_profile="sudden", drift_time=7.0,
        success_criteria={"must_call": ["get_backend_health", "run_circuit"]},
        difficulty="hard", tags=["drift", "adaptive"],
    ),
    BenchmarkTask(
        id="T2-12", tier=2, name="Mitigation Under Drift",
        prompt="Run GHZ-5 under drift conditions and apply error mitigation.",
        drift_profile="linear_decay", drift_time=10.0,
        success_criteria={"must_call": ["run_circuit", "apply_mitigation"]},
        difficulty="hard", tags=["drift", "mitigation"],
    ),
    BenchmarkTask(
        id="T2-13", tier=2, name="Diagnose Drift Cause",
        prompt="The backend is degraded. Diagnose what's causing the performance drop.",
        drift_profile="sudden", drift_time=8.0,
        success_criteria={"must_call": ["diagnose_and_suggest"], "must_report": ["cause"]},
        difficulty="hard", tags=["drift", "diagnosis"],
    ),

    # --- Recovery & Adaptation (7) ---
    BenchmarkTask(
        id="T2-14", tier=2, name="Detect and Report",
        prompt="Check health, detect drift, and provide a summary with recommendations.",
        drift_profile="linear_decay", drift_time=15.0,
        success_criteria={"must_call": ["get_backend_health", "diagnose_and_suggest"]},
        difficulty="medium", tags=["drift", "reporting"],
    ),
    BenchmarkTask(
        id="T2-15", tier=2, name="Adaptive Circuit Selection",
        prompt="Given current drift, which circuit will still achieve >0.8 fidelity? Run it.",
        drift_profile="linear_decay", drift_time=10.0,
        success_criteria={"must_call": ["run_circuit"], "fidelity_above": 0.75},
        difficulty="hard", tags=["drift", "adaptive", "selection"],
    ),
    BenchmarkTask(
        id="T2-16", tier=2, name="Full Pipeline Under Drift",
        prompt="Health check, predict, run, diagnose, mitigate — full pipeline under drift.",
        drift_profile="sudden", drift_time=6.0,
        success_criteria={"min_tool_calls": 4},
        difficulty="hard", tags=["drift", "pipeline"],
    ),
    BenchmarkTask(
        id="T2-17", tier=2, name="Drift Recovery Plan",
        prompt="The backend has drifted. Create a recovery plan: what experiments to run to recalibrate.",
        drift_profile="sudden", drift_time=8.0,
        success_criteria={"must_report": ["recovery_plan"]},
        difficulty="hard", tags=["drift", "recovery", "planning"],
    ),
    BenchmarkTask(
        id="T2-18", tier=2, name="Rabi Under Drift",
        prompt="Run a Rabi experiment under drift conditions. Is the pi-amplitude still valid?",
        drift_profile="linear_decay", drift_time=12.0,
        success_criteria={"must_call": ["rabi_experiment"]},
        difficulty="hard", tags=["drift", "pulse", "rabi"],
    ),
    BenchmarkTask(
        id="T2-19", tier=2, name="Continuous Monitoring",
        prompt="Check health at the current time. Compare with expected nominal values.",
        drift_profile="diurnal", drift_time=6.0,
        success_criteria={"must_call": ["get_backend_health"]},
        difficulty="medium", tags=["drift", "monitoring"],
    ),
    BenchmarkTask(
        id="T2-20", tier=2, name="Worst-Case Drift",
        prompt="Run GHZ-5 under severe drift. Report fidelity and whether the result is usable.",
        drift_profile="sudden", drift_time=10.0,
        success_criteria={"must_call": ["run_circuit"], "must_report": ["usability"]},
        difficulty="hard", tags=["drift", "severe"],
    ),
]


# ═══════════════════════════════════════════════════════
# TIER 3: Failure Tasks (10)
# Tasks requiring fault diagnosis and recovery
# ═══════════════════════════════════════════════════════

TIER_3_TASKS = [
    BenchmarkTask(
        id="T3-01", tier=3, name="Diagnose Sudden Failure",
        prompt="The backend just experienced a sudden failure. Identify what happened and suggest recovery.",
        drift_profile="sudden", drift_time=5.5,
        success_criteria={"must_call": ["get_backend_health", "diagnose_and_suggest"],
                         "must_detect": "sudden_degradation"},
        difficulty="hard", tags=["failure", "diagnosis"],
    ),
    BenchmarkTask(
        id="T3-02", tier=3, name="Qubit Death Detection",
        prompt="One or more qubits may have died (T1 → 0). Identify which ones.",
        drift_profile="sudden", drift_time=7.0,
        success_criteria={"must_call": ["get_qubit_properties"], "must_report": ["dead_qubits"]},
        difficulty="hard", tags=["failure", "qubit-death"],
    ),
    BenchmarkTask(
        id="T3-03", tier=3, name="Graceful Degradation",
        prompt="The backend is severely degraded. Find a circuit that still works acceptably.",
        drift_profile="sudden", drift_time=8.0,
        success_criteria={"must_call": ["run_circuit"], "must_report": ["working_circuit"]},
        difficulty="hard", tags=["failure", "graceful"],
    ),
    BenchmarkTask(
        id="T3-04", tier=3, name="Error Spike Analysis",
        prompt="Gate errors have spiked 5x. Analyze the impact on each benchmark circuit.",
        drift_profile="sudden", drift_time=6.0,
        success_criteria={"must_call": ["predict_fidelity"], "min_predictions": 3},
        difficulty="hard", tags=["failure", "analysis"],
    ),
    BenchmarkTask(
        id="T3-05", tier=3, name="Recovery After Failure",
        prompt="After a sudden failure, the backend partially recovered. Verify current state and run GHZ-5.",
        drift_profile="sudden", drift_time=5.0,
        success_criteria={"must_call": ["get_backend_health", "run_circuit"]},
        difficulty="hard", tags=["failure", "recovery"],
    ),
    BenchmarkTask(
        id="T3-06", tier=3, name="Mitigation Under Failure",
        prompt="The backend is in failure mode. Apply maximum mitigation to salvage GHZ-5 fidelity.",
        drift_profile="sudden", drift_time=7.0,
        success_criteria={"must_call": ["apply_mitigation"]},
        difficulty="hard", tags=["failure", "mitigation"],
    ),
    BenchmarkTask(
        id="T3-07", tier=3, name="Failure Mode Classification",
        prompt="Classify the current failure: is it coherence loss, gate error spike, or readout degradation?",
        drift_profile="sudden", drift_time=6.0,
        success_criteria={"must_call": ["get_backend_health", "diagnose_and_suggest"],
                         "must_report": ["failure_mode"]},
        difficulty="hard", tags=["failure", "classification"],
    ),
    BenchmarkTask(
        id="T3-08", tier=3, name="Safe Operation Check",
        prompt="Determine if it's safe to run experiments on this backend or if it needs recalibration.",
        drift_profile="sudden", drift_time=5.5,
        success_criteria={"must_report": ["safe_to_operate"]},
        difficulty="hard", tags=["failure", "safety"],
    ),
    BenchmarkTask(
        id="T3-09", tier=3, name="Partial Recovery Verification",
        prompt="The backend claims to have recovered. Verify by running a diagnostic suite.",
        drift_profile="sudden", drift_time=4.0,
        success_criteria={"must_call": ["get_backend_health", "run_circuit"]},
        difficulty="hard", tags=["failure", "verification"],
    ),
    BenchmarkTask(
        id="T3-10", tier=3, name="Full Failure Pipeline",
        prompt="Detect failure, diagnose cause, attempt mitigation, report final assessment.",
        drift_profile="sudden", drift_time=6.0,
        success_criteria={"min_tool_calls": 4, "must_call": ["diagnose_and_suggest"]},
        difficulty="hard", tags=["failure", "pipeline"],
    ),
]


# ═══════════════════════════════════════════════════════
# Combined task list
# ═══════════════════════════════════════════════════════

TASKS: list[BenchmarkTask] = TIER_1_TASKS + TIER_2_TASKS + TIER_3_TASKS


def get_tier(tier: int) -> list[BenchmarkTask]:
    """Get all tasks for a specific tier."""
    return [t for t in TASKS if t.tier == tier]


def get_task(task_id: str) -> BenchmarkTask:
    """Get a specific task by ID."""
    for t in TASKS:
        if t.id == task_id:
            return t
    raise ValueError(f"Task {task_id} not found")


def summary():
    """Print benchmark summary."""
    print(f"QC-Agent-Bench: {len(TASKS)} tasks")
    print(f"  Tier 1 (Static):  {len(TIER_1_TASKS)} tasks")
    print(f"  Tier 2 (Drift):   {len(TIER_2_TASKS)} tasks")
    print(f"  Tier 3 (Failure): {len(TIER_3_TASKS)} tasks")
    print(f"\nDifficulty distribution:")
    for d in ["easy", "medium", "hard"]:
        n = sum(1 for t in TASKS if t.difficulty == d)
        print(f"  {d}: {n}")
    print(f"\nTag distribution:")
    from collections import Counter
    tags = Counter(tag for t in TASKS for tag in t.tags)
    for tag, count in tags.most_common(10):
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    summary()
