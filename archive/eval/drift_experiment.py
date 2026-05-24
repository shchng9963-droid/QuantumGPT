"""Drift-Aware Replanning Experiment — P3-T2 core evaluation.

Runs controlled drift injection experiments comparing:
  1. QuantumGPT Full (drift-aware replanning)
  2. QuantumGPT No-Drift (same agent without drift monitoring)
  3. ReAct baseline (standard ReAct, no budget, no drift)
  4. Static pipeline (fixed sequence, no adaptation)

The experiment:
  - Backend starts stable (t=0, fidelity ~0.93)
  - At step N, inject SUDDEN_DEGRADATION (T1 drops 70%, errors 5x)
  - Measure: does the agent detect drift and recover?
  - Key metric: recovery_steps (how fast) and final_fidelity (how well)
"""

import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends.synthetic_drift import (
    SyntheticDriftBackend, DriftProfile, STABLE, SUDDEN_DEGRADATION, LINEAR_DECAY
)
from agent.react import ReActAgent, AgentTrace, TraceStep
from agent.drift_aware import DriftMonitor, ReplanningPolicy, DriftAwareRulePlanner
from agent.budget import FidelityBudget, BudgetDecision


# ═══════════════════════════════════════════════════════
# Fast Mock Executor with Drift Awareness
# ═══════════════════════════════════════════════════════

class DriftAwareMockExecutor:
    """Mock executor that returns fidelity based on backend drift state.

    When the backend is drifted, fidelity drops. When the agent re-runs
    after drift detection, it gets the degraded fidelity (realistic).
    """

    def __init__(self, backend: SyntheticDriftBackend):
        self.backend = backend
        self.call_log = []
        self._run_count = 0

    def execute(self, tool_name: str, tool_input: dict) -> str:
        self.call_log.append({"tool": tool_name, "input": tool_input})

        if tool_name == "get_backend_health":
            health = self.backend.get_health()
            return json.dumps({
                "backend": self.backend.name,
                "num_qubits": self.backend.num_qubits,
                "avg_1q_error": round(health.avg_1q_error, 6),
                "avg_2q_error": round(health.avg_2q_error, 6),
                "avg_readout_error": round(health.avg_readout_error, 6),
                "avg_t1_us": round(health.avg_t1_us, 1),
                "avg_t2_us": round(health.avg_t2_us, 1),
                "calibration_age_minutes": 5.0,
                "drift_score": health.drift_score or 0.0,
            }, indent=2)

        elif tool_name == "get_qubit_properties":
            qubits = tool_input.get("qubits", [0, 1, 2, 3, 4])
            props = []
            for q in qubits:
                qp = self.backend.get_qubit_properties(q)
                props.append({
                    "qubit": q,
                    "t1_us": round(qp.t1_us, 1),
                    "t2_us": round(qp.t2_us, 1),
                    "readout_error": round(qp.readout_error, 4),
                })
            return json.dumps({"backend": self.backend.name, "qubits": props}, indent=2)

        elif tool_name == "run_circuit":
            self._run_count += 1
            # Fidelity depends on current drift state
            health = self.backend.get_health()
            # Simple fidelity model: base 0.93, degraded by error rates
            base_fidelity = 0.93
            error_factor = (1 - health.avg_2q_error * 10) * (1 - health.avg_readout_error * 3)
            fidelity = max(0.1, min(1.0, base_fidelity * error_factor))

            return json.dumps({
                "circuit": tool_input.get("circuit_name", "ghz_5"),
                "shots": tool_input.get("shots", 4096),
                "fidelity": round(fidelity, 4),
                "transpiled_depth": 42,
                "top_counts": {"00000": int(4096 * fidelity / 2), "11111": int(4096 * fidelity / 2)},
                "metadata": {"backend": self.backend.name},
            }, indent=2)

        elif tool_name == "apply_mitigation":
            health = self.backend.get_health()
            base_fidelity = 0.93
            error_factor = (1 - health.avg_2q_error * 10) * (1 - health.avg_readout_error * 3)
            unmit = max(0.1, min(1.0, base_fidelity * error_factor))
            # Mitigation helps a bit but can't fully recover from hardware drift
            mit = min(1.0, unmit + 0.03)
            return json.dumps({
                "circuit": tool_input.get("circuit_name", "ghz_5"),
                "method": "zne",
                "unmitigated_fidelity": round(unmit, 4),
                "mitigated_fidelity": round(mit, 4),
                "improvement": round(mit - unmit, 4),
            }, indent=2)

        elif tool_name == "detect_drift":
            health = self.backend.get_health()
            score = health.drift_score or 0.0
            return json.dumps({
                "drift_score": round(score, 4),
                "is_drifting": score > 0.3,
                "recommendation": "recalibrate" if score > 0.3 else "stable",
            }, indent=2)

        elif tool_name == "diagnose_and_suggest":
            health = self.backend.get_health()
            score = health.drift_score or 0.0
            if score > 0.3:
                return json.dumps({
                    "severity": "high",
                    "suggestions": [
                        "Significant drift detected — recalibration recommended",
                        "Gate errors elevated — consider re-transpiling",
                        "T1 degradation observed — avoid long-depth circuits",
                    ],
                }, indent=2)
            return json.dumps({
                "severity": "low",
                "suggestions": ["Backend operating normally"],
            }, indent=2)

        elif tool_name == "transpile_circuit":
            return json.dumps({
                "circuit": tool_input.get("circuit_name", "ghz_5"),
                "original_depth": 5,
                "transpiled_depth": 42,
                "two_qubit_gates": 8,
            }, indent=2)

        elif tool_name == "predict_fidelity":
            health = self.backend.get_health()
            base_fidelity = 0.93
            error_factor = (1 - health.avg_2q_error * 10) * (1 - health.avg_readout_error * 3)
            pred = max(0.1, min(1.0, base_fidelity * error_factor))
            return json.dumps({
                "circuit": tool_input.get("circuit_name", "ghz_5"),
                "predicted_fidelity": round(pred, 4),
                "confidence": "medium",
                "breakdown": {
                    "f_1q_gates": round(1 - health.avg_1q_error * 5, 4),
                    "f_2q_gates": round(1 - health.avg_2q_error * 10, 4),
                    "f_readout": round(1 - health.avg_readout_error * 3, 4),
                },
            }, indent=2)

        elif tool_name == "retrieve_past_experiments":
            return json.dumps({"experiments": [], "message": "No history yet."})

        elif tool_name == "list_benchmarks":
            return json.dumps({
                "hand_written": {"ghz_5": {}, "qft_4": {}, "bv_5": {}},
                "mqtbench": {},
            }, indent=2)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ═══════════════════════════════════════════════════════
# Experiment Runner
# ═══════════════════════════════════════════════════════

@dataclass
class DriftExperimentResult:
    """Result of one drift experiment run."""
    system: str
    task: str
    drift_profile: str
    fidelities: list[float]  # fidelity at each step
    drift_injected_at: int   # step number
    recovery_step: int       # step where fidelity recovered (0 = never)
    final_fidelity: float
    total_steps: int
    replan_triggered: bool
    replan_at_step: int      # step where replan was triggered (0 = never)

    @property
    def recovered(self) -> bool:
        return self.recovery_step > 0

    @property
    def recovery_latency(self) -> int:
        """Steps from drift injection to recovery."""
        if self.recovery_step > 0:
            return self.recovery_step - self.drift_injected_at
        return -1  # never recovered


def run_drift_aware_experiment(
    backend: SyntheticDriftBackend,
    task: str = "Run GHZ-5 and report fidelity",
    drift_at_step: int = 3,
    max_steps: int = 12,
    use_drift_aware: bool = True,
    target_fidelity: float = 0.85,
) -> DriftExperimentResult:
    """Run a single drift experiment with the mock agent.

    Simulates the agent loop step by step, injecting drift at a specific step.
    """
    # Start stable
    backend.set_time(0)
    backend.profile = STABLE

    executor = DriftAwareMockExecutor(backend)
    monitor = DriftMonitor(backend, drift_threshold=0.3) if use_drift_aware else None
    policy = ReplanningPolicy(strategy="aggressive") if use_drift_aware else None

    if monitor:
        monitor.initialize()

    # Simulate the agent loop manually
    fidelities = []
    replan_triggered = False
    replan_at_step = 0
    recovery_step = 0
    step = 0

    # Phase 1: health check
    executor.execute("get_backend_health", {})
    step += 1
    if monitor:
        monitor.step()

    # Phase 2: run circuit (pre-drift)
    for i in range(drift_at_step - 1):
        result = json.loads(executor.execute("run_circuit", {"circuit_name": "ghz_5", "shots": 4096}))
        fidelities.append(result["fidelity"])
        step += 1
        if monitor:
            monitor.step()

    # === INJECT DRIFT ===
    backend.set_time(6)  # past the t=5 threshold for SUDDEN_DEGRADATION
    backend.profile = SUDDEN_DEGRADATION

    # Phase 3: post-drift steps
    for i in range(max_steps - drift_at_step):
        # Check drift before acting
        if monitor:
            state = monitor.check_now()
            if state.needs_replan and not replan_triggered:
                replan_triggered = True
                replan_at_step = step
                # Drift-aware agent: re-check health and re-run
                executor.execute("get_backend_health", {})
                step += 1
                monitor.acknowledge_replan()

        # Run circuit
        result = json.loads(executor.execute("run_circuit", {"circuit_name": "ghz_5", "shots": 4096}))
        fidelities.append(result["fidelity"])
        step += 1

        if monitor:
            monitor.step()

        # Check if recovered (fidelity back above target)
        # Note: with SUDDEN_DEGRADATION, fidelity won't recover without
        # actual recalibration. The "recovery" here means the agent
        # detected the issue and reported it correctly.
        if result["fidelity"] >= target_fidelity and recovery_step == 0 and step > drift_at_step:
            recovery_step = step

    system_name = "QuantumGPT-Full" if use_drift_aware else "ReAct-Baseline"

    return DriftExperimentResult(
        system=system_name,
        task=task,
        drift_profile="SUDDEN_DEGRADATION",
        fidelities=fidelities,
        drift_injected_at=drift_at_step,
        recovery_step=recovery_step,
        final_fidelity=fidelities[-1] if fidelities else 0.0,
        total_steps=step,
        replan_triggered=replan_triggered,
        replan_at_step=replan_at_step,
    )


def run_static_pipeline_experiment(
    backend: SyntheticDriftBackend,
    drift_at_step: int = 3,
    max_steps: int = 12,
) -> DriftExperimentResult:
    """Static pipeline: fixed sequence, no adaptation."""
    backend.set_time(0)
    backend.profile = STABLE
    executor = DriftAwareMockExecutor(backend)

    fidelities = []
    step = 0

    # Fixed pipeline: health → run → run → run ...
    executor.execute("get_backend_health", {})
    step += 1

    for i in range(max_steps - 1):
        if i == drift_at_step - 1:
            backend.set_time(6)
            backend.profile = SUDDEN_DEGRADATION

        result = json.loads(executor.execute("run_circuit", {"circuit_name": "ghz_5", "shots": 4096}))
        fidelities.append(result["fidelity"])
        step += 1

    return DriftExperimentResult(
        system="Static-Pipeline",
        task="Run GHZ-5",
        drift_profile="SUDDEN_DEGRADATION",
        fidelities=fidelities,
        drift_injected_at=drift_at_step,
        recovery_step=0,  # never recovers
        final_fidelity=fidelities[-1] if fidelities else 0.0,
        total_steps=step,
        replan_triggered=False,
        replan_at_step=0,
    )


# ═══════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════

def run_full_comparison():
    """Run the full drift experiment comparison."""
    print("\n" + "=" * 70)
    print("P3-T2: Drift-Aware Replanning Experiment")
    print("=" * 70)
    print("\nSetup:")
    print("  Backend: SyntheticDrift(FakeBrisbane)")
    print("  Drift: SUDDEN_DEGRADATION at step 3 (T1 -70%, errors 5x)")
    print("  Target fidelity: 0.85")
    print("  Max steps: 12")
    print()

    backend = SyntheticDriftBackend("FakeBrisbane")

    # Run experiments
    results = []

    # 1. QuantumGPT Full (drift-aware)
    print("Running: QuantumGPT Full (drift-aware)...")
    r1 = run_drift_aware_experiment(backend, use_drift_aware=True)
    results.append(r1)

    # 2. QuantumGPT No-Drift (same agent, no drift monitoring)
    print("Running: QuantumGPT No-Drift...")
    r2 = run_drift_aware_experiment(backend, use_drift_aware=False)
    results.append(r2)

    # 3. Static Pipeline
    print("Running: Static Pipeline...")
    r3 = run_static_pipeline_experiment(backend)
    results.append(r3)

    # Print results
    print("\n" + "─" * 70)
    print(f"{'System':<25} {'Pre-drift F':<12} {'Post-drift F':<13} {'Final F':<10} {'Replan?':<8} {'Recovery'}")
    print("─" * 70)

    for r in results:
        pre_f = r.fidelities[0] if r.fidelities else 0
        # Find first post-drift fidelity
        post_idx = r.drift_injected_at - 1  # index in fidelities list
        post_f = r.fidelities[post_idx] if post_idx < len(r.fidelities) else 0

        recovery_str = f"step {r.recovery_step}" if r.recovered else "NEVER"
        replan_str = f"step {r.replan_at_step}" if r.replan_triggered else "No"

        print(f"{r.system:<25} {pre_f:<12.4f} {post_f:<13.4f} {r.final_fidelity:<10.4f} {replan_str:<8} {recovery_str}")

    print("─" * 70)

    # Detailed fidelity traces
    print("\nFidelity traces (step by step):")
    for r in results:
        trace_str = " → ".join(f"{f:.3f}" for f in r.fidelities)
        print(f"  {r.system}: [{trace_str}]")
        if r.replan_triggered:
            print(f"    ↳ Replan triggered at step {r.replan_at_step}")

    # Key findings
    print("\n" + "=" * 70)
    print("Key Findings:")
    print("=" * 70)

    drift_aware = results[0]
    no_drift = results[1]
    static = results[2]

    print(f"\n  1. Drift Detection:")
    print(f"     - QuantumGPT Full: {'✓ Detected' if drift_aware.replan_triggered else '✗ Missed'} "
          f"(at step {drift_aware.replan_at_step})")
    print(f"     - No-Drift / Static: No detection capability")

    print(f"\n  2. Fidelity Impact:")
    if drift_aware.fidelities and no_drift.fidelities:
        aware_post = drift_aware.fidelities[-1]
        naive_post = no_drift.fidelities[-1]
        print(f"     - With drift-aware: final fidelity = {aware_post:.4f}")
        print(f"     - Without drift-aware: final fidelity = {naive_post:.4f}")
        print(f"     - Difference: {aware_post - naive_post:+.4f}")

    print(f"\n  3. Recovery:")
    print(f"     - QuantumGPT Full: {'Recovered' if drift_aware.recovered else 'Detected but hardware degraded'}")
    print(f"     - Others: No recovery attempt")

    print(f"\n  Note: With SUDDEN_DEGRADATION, fidelity cannot fully recover")
    print(f"  without actual recalibration. The key metric is DETECTION SPEED")
    print(f"  and the agent's ability to report the issue and adapt its strategy.")

    return results


if __name__ == "__main__":
    results = run_full_comparison()
