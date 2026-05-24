"""Test P3-T1: ExperimentRecord integration with ReAct Agent.

Uses mock mode (rule-based planner) which doesn't require actual circuit
simulation. The ToolExecutor is patched to return fast mock results.

Verifies:
  1. Agent auto-saves ExperimentRecord after each run
  2. Memory retrieval works
  3. Memory context injection works
  4. Ablation switch (use_memory=False) disables recording
  5. retrieve_past_experiments tool works
"""

import os
import sys
import json
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends.fake_adapter import FakeBackendAdapter
from agent.react import ReActAgent, AgentTrace, TraceStep
from agent.memory import AgentMemory
from data.store import DataStore
from data.experiment_record import ExperimentRecord, ExperimentStore


class FastMockExecutor:
    """Fast executor that returns canned results without simulation."""

    def __init__(self, backend):
        self.backend = backend
        self.call_log = []

    def execute(self, tool_name: str, tool_input: dict) -> str:
        self.call_log.append({"tool": tool_name, "input": tool_input})

        if tool_name == "get_backend_health":
            return json.dumps({
                "backend": "FakeBrisbane",
                "num_qubits": 127,
                "avg_1q_error": 0.006,
                "avg_2q_error": 0.009,
                "avg_readout_error": 0.029,
                "avg_t1_us": 224.5,
                "avg_t2_us": 144.9,
                "calibration_age_minutes": 5.0,
                "drift_score": 0.12,
            }, indent=2)

        elif tool_name == "get_qubit_properties":
            qubits = tool_input.get("qubits", [0, 1, 2, 3, 4])
            return json.dumps({
                "backend": "FakeBrisbane",
                "qubits": [
                    {"qubit": q, "t1_us": 220 + q, "t2_us": 140 + q,
                     "readout_error": 0.02 + q * 0.001}
                    for q in qubits
                ],
            }, indent=2)

        elif tool_name == "run_circuit":
            circ = tool_input.get("circuit_name", "ghz_5")
            return json.dumps({
                "circuit": circ,
                "shots": tool_input.get("shots", 4096),
                "fidelity": 0.927,
                "transpiled_depth": 42,
                "top_counts": {"00000": 1900, "11111": 1800, "00001": 150},
                "metadata": {"backend": "FakeBrisbane"},
            }, indent=2)

        elif tool_name == "apply_mitigation":
            return json.dumps({
                "circuit": tool_input.get("circuit_name", "ghz_5"),
                "method": "zne",
                "unmitigated_fidelity": 0.927,
                "mitigated_fidelity": 0.961,
                "improvement": 0.034,
            }, indent=2)

        elif tool_name == "diagnose_and_suggest":
            return json.dumps({
                "severity": "low",
                "suggestions": ["Consider ZNE mitigation", "Qubit 3 has elevated readout error"],
            }, indent=2)

        elif tool_name == "rabi_experiment":
            return json.dumps({
                "pulse_shape": "square",
                "pulse_duration_ns": 100,
                "qubit_freq_ghz": 5.0,
                "pi_amplitude_mhz": 42.3,
                "max_population": 0.98,
                "amplitudes": [0.0, 0.01, 0.02, 0.03, 0.04],
                "populations": [0.0, 0.25, 0.7, 0.98, 0.7],
            }, indent=2)

        elif tool_name == "fit_rabi":
            return json.dumps({
                "pi_amplitude_ghz": 0.0423,
                "pi_amplitude_mhz": 42.3,
                "half_pi_amplitude_ghz": 0.0212,
                "r_squared": 0.997,
                "rabi_frequency_mhz": 42.3,
                "fit_successful": True,
            }, indent=2)

        elif tool_name == "predict_fidelity":
            return json.dumps({
                "circuit": tool_input.get("circuit_name", "ghz_5"),
                "predicted_fidelity": 0.91,
                "confidence": "medium",
                "breakdown": {"f_1q_gates": 0.97, "f_2q_gates": 0.94, "f_readout": 0.99},
            }, indent=2)

        elif tool_name == "transpile_circuit":
            return json.dumps({
                "circuit": tool_input.get("circuit_name", "ghz_5"),
                "original_depth": 5,
                "transpiled_depth": 42,
                "two_qubit_gates": 8,
            }, indent=2)

        elif tool_name == "list_benchmarks":
            return json.dumps({
                "hand_written": {"ghz_5": {}, "qft_4": {}, "bv_5": {}, "vqe_4": {}, "qaoa_4": {}},
                "mqtbench": {},
            }, indent=2)

        elif tool_name == "retrieve_past_experiments":
            return json.dumps({"experiments": [], "message": "No matching experiments."})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})


class LowFidelityExecutor(FastMockExecutor):
    """Fast executor variant that records consistently low circuit fidelity."""

    def execute(self, tool_name: str, tool_input: dict) -> str:
        result = super().execute(tool_name, tool_input)
        data = json.loads(result)
        if tool_name == "run_circuit":
            data["fidelity"] = 0.42
        elif tool_name == "apply_mitigation":
            data["unmitigated_fidelity"] = 0.42
            data["mitigated_fidelity"] = 0.48
            data["improvement"] = 0.06
        else:
            return result
        return json.dumps(data, indent=2)


class LoggingExecutor(FastMockExecutor):
    """Fast executor variant used to inspect the second-run plan."""



def _make_agent(db_path, use_memory=True, executor_cls=FastMockExecutor):
    """Create a ReActAgent with fast mock executor."""
    backend = FakeBackendAdapter()
    agent = ReActAgent(
        backend=backend,
        use_mock=True,
        verbose=False,
        db_path=db_path,
        use_memory=use_memory,
        max_turns=10,
        max_seconds=30.0,
    )
    # Replace the executor with our fast mock
    agent.executor = executor_cls(backend)
    return agent


def test_memory_auto_recording():
    """Agent should auto-save an ExperimentRecord after each run."""
    db_path = tempfile.mktemp(suffix=".duckdb")
    try:
        agent = _make_agent(db_path)

        # Run a simple task
        trace = agent.run("Run GHZ-5 and report fidelity")

        # Check that an experiment record was saved
        assert agent.memory is not None, "Memory should be initialized"
        records = agent.memory.store.recent(limit=5)
        assert len(records) >= 1, f"Expected at least 1 record, got {len(records)}"

        rec = records[0]
        assert rec.backend == "FakeBrisbane"
        assert rec.problem == "Run GHZ-5 and report fidelity"
        assert rec.fidelity is not None
        assert len(rec.tool_calls) > 0
        assert len(rec.plan) > 0
        print(f"  ✓ Auto-recorded: {rec.summary}")
        print(f"    fidelity={rec.fidelity}, tools={len(rec.tool_calls)}, outcome={rec.outcome}")

    finally:
        for ext in ["", ".wal"]:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass


def test_memory_retrieval():
    """Agent memory should be retrievable by circuit/backend."""
    db_path = tempfile.mktemp(suffix=".duckdb")
    try:
        agent = _make_agent(db_path)

        # Run multiple tasks to build history
        agent.run("Run GHZ-5 and report fidelity")
        agent.run("Run QFT-4 and report fidelity")
        agent.run("Run GHZ-5 again with mitigation")

        # Retrieve by circuit
        results = agent.memory.retrieve_similar(circuit_name="ghz_5")
        assert len(results) >= 2, f"Expected >= 2 GHZ-5 records, got {len(results)}"
        print(f"  ✓ Retrieved {len(results)} records for ghz_5")

        # Retrieve by backend
        results = agent.memory.retrieve_similar(backend="FakeBrisbane")
        assert len(results) >= 3, f"Expected >= 3 records, got {len(results)}"
        print(f"  ✓ Retrieved {len(results)} records for FakeBrisbane")

    finally:
        for ext in ["", ".wal"]:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass


def test_memory_context_injection():
    """Memory context should be generated for prompt injection."""
    db_path = tempfile.mktemp(suffix=".duckdb")
    try:
        agent = _make_agent(db_path)

        # First run — no history yet
        ctx = agent.memory.get_context_summary("Run GHZ-5", "FakeBrisbane")
        assert ctx == "", "Should be empty with no history"

        # Run a task
        agent.run("Run GHZ-5 and report fidelity")

        # Now context should be non-empty
        ctx = agent.memory.get_context_summary("Run GHZ-5", "FakeBrisbane")
        assert "[MEMORY]" in ctx, f"Expected [MEMORY] prefix, got: {ctx[:100]}"
        assert "fidelity" in ctx.lower()
        print(f"  ✓ Context injection works:")
        for line in ctx.strip().split("\n")[:3]:
            print(f"    {line}")

    finally:
        for ext in ["", ".wal"]:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass


def test_memory_ablation_switch():
    """use_memory=False should disable all memory operations."""
    db_path = tempfile.mktemp(suffix=".duckdb")
    try:
        agent = _make_agent(db_path, use_memory=False)

        # Memory should be None
        assert agent.memory is None, "Memory should be None when disabled"

        # Run should still work
        trace = agent.run("Run GHZ-5 and report fidelity")
        assert trace.final_answer != ""
        print("  ✓ Agent works with memory disabled")

    finally:
        for ext in ["", ".wal"]:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass


def test_memory_learning_effect():
    """After multiple runs, memory should accumulate stats."""
    db_path = tempfile.mktemp(suffix=".duckdb")
    try:
        agent = _make_agent(db_path)

        # Run 5 times
        for i in range(5):
            agent.run("Run GHZ-5 and report fidelity")

        # Check stats
        stats = agent.memory.store.stats()
        assert stats["total_experiments"] == 5
        assert stats["avg_fidelity"] is not None
        print(f"  ✓ Memory stats after 5 runs:")
        print(f"    total={stats['total_experiments']}, avg_fidelity={stats['avg_fidelity']:.3f}")
        print(f"    successes={stats['successes']}")

        # Check stats summary
        summary = agent.memory.get_stats_summary()
        assert "5 experiments" in summary
        print(f"  ✓ Stats summary: {summary}")

    finally:
        for ext in ["", ".wal"]:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass


def test_retrieve_past_experiments_tool():
    """The retrieve_past_experiments tool should work via _execute_tool."""
    db_path = tempfile.mktemp(suffix=".duckdb")
    try:
        agent = _make_agent(db_path)

        # Seed some history
        agent.run("Run GHZ-5 and report fidelity")
        agent.run("Run QFT-4 and report fidelity")

        # Call the tool directly
        result = agent._execute_tool("retrieve_past_experiments", {
            "circuit_name": "ghz_5",
            "limit": 3,
        })
        data = json.loads(result)
        assert "experiments" in data
        assert len(data["experiments"]) >= 1
        print(f"  ✓ retrieve_past_experiments tool returned {data['count']} results")

        # Test with no matches
        result = agent._execute_tool("retrieve_past_experiments", {
            "circuit_name": "nonexistent_circuit",
        })
        data = json.loads(result)
        assert data["experiments"] == []
        print("  ✓ Empty query returns empty list")

    finally:
        for ext in ["", ".wal"]:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass


def test_low_fidelity_history_influences_next_plan():
    """Low-fidelity memory should trigger diagnosis before repeating a circuit run."""
    db_path = tempfile.mktemp(suffix=".duckdb")
    try:
        first_agent = _make_agent(db_path, executor_cls=LowFidelityExecutor)
        first_agent.run("Run GHZ-5 and report fidelity")

        second_agent = _make_agent(db_path, executor_cls=LoggingExecutor)
        trace = second_agent.run("Run GHZ-5 and report fidelity")
        actions = [step.action for step in trace.steps if step.action]

        assert "diagnose_and_suggest" in actions
        assert actions.index("diagnose_and_suggest") < actions.index("run_circuit")
        assert trace.state is not None
        assert trace.state.memory_context
        assert trace.state.memory_context[0]["circuit_name"] == "ghz_5"
        assert trace.state.memory_context[0]["fidelity"] < 0.85

    finally:
        for ext in ["", ".wal"]:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass


def test_use_memory_false_trace_has_no_memory_context():
    """use_memory=False should not leak memory_context into trace exports."""
    db_path = tempfile.mktemp(suffix=".duckdb")
    try:
        agent = _make_agent(db_path, use_memory=False)
        trace = agent.run("Run GHZ-5 and report fidelity")
        trace_data = trace.to_dict(include_observations=True)

        assert "memory_context" not in json.dumps(trace_data)
        assert trace.state is not None
        assert trace.state.memory_context == []

    finally:
        for ext in ["", ".wal"]:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass


def test_rabi_experiment_memory():
    """Rabi experiments should be recorded with fit data."""
    db_path = tempfile.mktemp(suffix=".duckdb")
    try:
        agent = _make_agent(db_path)

        # Run a Rabi task
        agent.run("Run a Rabi oscillation experiment to characterize qubit 0")

        records = agent.memory.store.recent(limit=5)
        rabi_records = [r for r in records if r.experiment_type == "rabi_tuneup"]
        assert len(rabi_records) >= 1, f"Expected rabi record, got types: {[r.experiment_type for r in records]}"

        rec = rabi_records[0]
        assert rec.fits, f"Expected fits data, got: {rec.fits}"
        print(f"  ✓ Rabi experiment recorded with fits: {rec.fits}")

    finally:
        for ext in ["", ".wal"]:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("P3-T1: ExperimentRecord Integration Tests")
    print("=" * 60)

    tests = [
        ("Auto-recording", test_memory_auto_recording),
        ("Retrieval", test_memory_retrieval),
        ("Context injection", test_memory_context_injection),
        ("Ablation switch", test_memory_ablation_switch),
        ("Learning effect", test_memory_learning_effect),
        ("Tool execution", test_retrieve_past_experiments_tool),
        ("Rabi memory", test_rabi_experiment_memory),
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
