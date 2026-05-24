"""Test P3-T2: Drift-Aware Replanning.

Verifies:
  1. DriftMonitor detects drift correctly
  2. ReplanningPolicy makes correct decisions
  3. DriftAwareRulePlanner overrides base planner on drift
  4. End-to-end drift experiment produces expected results
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends.synthetic_drift import (
    SyntheticDriftBackend, DriftProfile, STABLE, SUDDEN_DEGRADATION
)
from agent.drift_aware import (
    DriftMonitor, DriftState, ReplanningPolicy, ReplanDecision,
    DriftAwareRulePlanner,
)


def test_drift_monitor_stable():
    """Monitor should report no drift when backend is stable."""
    backend = SyntheticDriftBackend("FakeBrisbane", STABLE)
    backend.set_time(0)

    monitor = DriftMonitor(backend, drift_threshold=0.3)
    monitor.initialize()

    # Step multiple times — should stay stable
    for _ in range(5):
        state = monitor.step()

    assert not state.is_drifting
    assert state.drift_score < 0.3
    print(f"  ✓ Stable: drift_score={state.drift_score:.4f}, is_drifting={state.is_drifting}")


def test_drift_monitor_detects_sudden():
    """Monitor should detect sudden degradation."""
    backend = SyntheticDriftBackend("FakeBrisbane", STABLE)
    backend.set_time(0)

    monitor = DriftMonitor(backend, drift_threshold=0.3)
    monitor.initialize()

    # Stable phase
    state = monitor.check_now()
    assert not state.is_drifting

    # Inject drift
    backend.set_time(6)
    backend.profile = SUDDEN_DEGRADATION

    # Check again
    state = monitor.check_now()
    assert state.is_drifting, f"Expected drifting, got score={state.drift_score}"
    assert state.drift_score > 0.3
    assert len(state.invalidated_results) > 0
    print(f"  ✓ Detected: drift_score={state.drift_score:.4f}, invalidated={state.invalidated_results}")


def test_drift_monitor_acknowledge():
    """After acknowledge_replan, invalidation should clear."""
    backend = SyntheticDriftBackend("FakeBrisbane", STABLE)
    backend.set_time(0)

    monitor = DriftMonitor(backend, drift_threshold=0.3)
    monitor.initialize()

    # Inject drift
    backend.set_time(6)
    backend.profile = SUDDEN_DEGRADATION
    state = monitor.check_now()
    assert state.is_drifting

    # Acknowledge
    monitor.acknowledge_replan()
    assert len(monitor.state.invalidated_results) == 0
    print("  ✓ Acknowledge clears invalidation")


def test_replanning_policy_aggressive():
    """Aggressive policy should replan immediately on drift."""
    policy = ReplanningPolicy(strategy="aggressive")

    # No drift
    state = DriftState(is_drifting=False, drift_score=0.1)
    decision = policy.decide(state)
    assert not decision.should_replan

    # Drift detected
    state = DriftState(
        is_drifting=True,
        drift_score=0.5,
        invalidated_results=["transpile", "run_circuit"],
    )
    decision = policy.decide(state)
    assert decision.should_replan
    assert "transpile" in decision.invalidated
    print(f"  ✓ Aggressive: replan on drift (reason: {decision.reason[:50]})")


def test_replanning_policy_conservative():
    """Conservative policy should wait for persistent drift."""
    policy = ReplanningPolicy(strategy="conservative", persistence_threshold=3)

    # First drift check
    state = DriftState(
        is_drifting=True,
        drift_score=0.5,
        consecutive_drift_checks=1,
        invalidated_results=["transpile"],
    )
    decision = policy.decide(state)
    assert not decision.should_replan, "Should wait for persistence"

    # After 3 consecutive checks
    state.consecutive_drift_checks = 3
    decision = policy.decide(state)
    assert decision.should_replan
    print(f"  ✓ Conservative: waits for persistence ({state.consecutive_drift_checks} checks)")


def test_drift_context_generation():
    """Monitor should generate useful context string for LLM."""
    backend = SyntheticDriftBackend("FakeBrisbane", STABLE)
    backend.set_time(0)

    monitor = DriftMonitor(backend, drift_threshold=0.3)
    monitor.initialize()

    # No drift — empty context
    ctx = monitor.get_drift_context()
    assert ctx == ""

    # Inject drift
    backend.set_time(6)
    backend.profile = SUDDEN_DEGRADATION
    monitor.check_now()

    ctx = monitor.get_drift_context()
    assert "[DRIFT ALERT]" in ctx
    assert "ACTION REQUIRED" in ctx
    print(f"  ✓ Drift context generated:")
    for line in ctx.split("\n")[:3]:
        print(f"    {line}")


def test_end_to_end_drift_experiment():
    """Full drift experiment should show detection and adaptation."""
    from eval.drift_experiment import run_drift_aware_experiment

    backend = SyntheticDriftBackend("FakeBrisbane")

    # With drift awareness
    r_aware = run_drift_aware_experiment(
        backend, use_drift_aware=True, drift_at_step=3, max_steps=10
    )
    assert r_aware.replan_triggered, "Drift-aware agent should detect drift"
    assert r_aware.replan_at_step == 3, f"Should detect at step 3, got {r_aware.replan_at_step}"

    # Without drift awareness
    r_naive = run_drift_aware_experiment(
        backend, use_drift_aware=False, drift_at_step=3, max_steps=10
    )
    assert not r_naive.replan_triggered, "Naive agent should not detect drift"

    print(f"  ✓ End-to-end: aware detected at step {r_aware.replan_at_step}, naive missed")
    print(f"    Aware final fidelity: {r_aware.final_fidelity:.4f}")
    print(f"    Naive final fidelity: {r_naive.final_fidelity:.4f}")


def test_enhanced_drift_experiment():
    """Enhanced experiment should show >= 15% improvement."""
    from eval.drift_experiment_enhanced import run_adaptive_experiment

    backend = SyntheticDriftBackend("FakeBrisbane")

    r_full = run_adaptive_experiment(backend, use_drift_aware=True, use_mitigation=True)
    r_base = run_adaptive_experiment(backend, use_drift_aware=False)

    # Compute average post-drift fidelity
    full_post = r_full.fidelities[r_full.drift_injected_at - 1:]
    base_post = r_base.fidelities[r_base.drift_injected_at - 1:]

    avg_full = sum(full_post) / len(full_post) if full_post else 0
    avg_base = sum(base_post) / len(base_post) if base_post else 0

    improvement_pct = (avg_full - avg_base) / max(avg_base, 0.01) * 100

    assert improvement_pct >= 15.0, f"Expected >= 15% improvement, got {improvement_pct:.1f}%"
    print(f"  ✓ Enhanced experiment: +{improvement_pct:.1f}% (target: >= 15%)")
    print(f"    Full avg post-drift: {avg_full:.4f}")
    print(f"    Base avg post-drift: {avg_base:.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("P3-T2: Drift-Aware Replanning Tests")
    print("=" * 60)

    tests = [
        ("Monitor stable", test_drift_monitor_stable),
        ("Monitor detects sudden", test_drift_monitor_detects_sudden),
        ("Monitor acknowledge", test_drift_monitor_acknowledge),
        ("Policy aggressive", test_replanning_policy_aggressive),
        ("Policy conservative", test_replanning_policy_conservative),
        ("Drift context", test_drift_context_generation),
        ("End-to-end experiment", test_end_to_end_drift_experiment),
        ("Enhanced experiment (>=15%)", test_enhanced_drift_experiment),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
