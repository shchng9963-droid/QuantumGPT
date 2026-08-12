"""Unit tests for MQTBench integration."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest
from qiskit.circuit import QuantumCircuit

mqt = pytest.importorskip("mqt.bench", reason="mqt-bench not installed")

from bench.mqtbench import (
    get_mqtbench_circuits,
    list_mqtbench_available,
    MQTBenchCircuit,
    DEFAULT_MQTBENCH_SELECTION,
    EXTENDED_MQTBENCH_SELECTION,
    ORACLE_REQUIRED_LABELS,
    assert_pool_matches_oracle,
)


def test_list_available():
    """Verify we can list all MQTBench benchmarks."""
    names = list_mqtbench_available()
    assert len(names) >= 30  # MQTBench has 34 benchmarks
    assert "ghz" in names
    assert "qft" in names
    assert "vqe_su2" in names


def test_default_selection():
    """Default selection generates 5 circuits."""
    circuits = get_mqtbench_circuits()
    assert len(circuits) == 5
    labels = {c.label for c in circuits}
    assert labels == {"GHZ-5", "DJ-5", "GraphState-5", "QFTent-5", "VQE_SU2-4"}


def test_circuit_properties():
    """Each circuit has valid properties."""
    circuits = get_mqtbench_circuits()
    for c in circuits:
        assert isinstance(c, MQTBenchCircuit)
        assert isinstance(c.circuit, QuantumCircuit)
        assert c.num_qubits >= 4
        assert c.depth > 0
        assert c.total_gates > 0
        assert c.gateset == "ibm_eagle"


def test_native_gate_set():
    """Circuits should only contain IBM Eagle native gates + measure."""
    allowed = {"ecr", "rz", "sx", "x", "measure", "barrier"}
    circuits = get_mqtbench_circuits()
    for c in circuits:
        ops = set(c.circuit.count_ops().keys())
        unexpected = ops - allowed
        assert not unexpected, (
            f"{c.label} has unexpected gates: {unexpected}"
        )


def test_ecr_count():
    """ECR count should match the circuit ops."""
    circuits = get_mqtbench_circuits()
    for c in circuits:
        ops = dict(c.circuit.count_ops())
        assert c.ecr_count == ops.get("ecr", 0)


def test_custom_selection():
    """Custom selection with a single circuit."""
    selection = [("ghz", 3, "GHZ-3", "Small GHZ")]
    circuits = get_mqtbench_circuits(selection=selection)
    assert len(circuits) == 1
    assert circuits[0].label == "GHZ-3"
    assert circuits[0].num_qubits == 3


def test_runnable_on_aer():
    """Circuits should be runnable on AerSimulator."""
    from qiskit_aer import AerSimulator
    from qiskit.compiler import transpile

    circuits = get_mqtbench_circuits()
    sim = AerSimulator()
    for c in circuits:
        tc = transpile(c.circuit, backend=sim)
        job = sim.run(tc, shots=100)
        counts = job.result().get_counts()
        assert sum(counts.values()) == 100, f"{c.label} shot count mismatch"


def test_oracle_required_labels_match_extended():
    """ORACLE_REQUIRED_LABELS must equal EXTENDED labels — single source of truth."""
    extended_labels = {label for _, _, label, _ in EXTENDED_MQTBENCH_SELECTION}
    assert ORACLE_REQUIRED_LABELS == extended_labels, (
        "ORACLE_REQUIRED_LABELS drifted from EXTENDED_MQTBENCH_SELECTION; "
        "fix bench/mqtbench.py so they remain in sync."
    )


def test_assert_pool_matches_oracle_passes_for_extended():
    """assert_pool_matches_oracle accepts the EXTENDED pool."""
    extended_labels = {label for _, _, label, _ in EXTENDED_MQTBENCH_SELECTION}
    # No exception expected.
    assert_pool_matches_oracle(extended_labels)


def test_assert_pool_matches_oracle_rejects_default_only():
    """Default-only pool must be rejected: this was the v3_main_agent pollution."""
    default_labels = {label for _, _, label, _ in DEFAULT_MQTBENCH_SELECTION}
    with pytest.raises(AssertionError) as excinfo:
        assert_pool_matches_oracle(default_labels)
    msg = str(excinfo.value)
    # Should name the missing EXTENDED-only labels.
    for missing_label in ("Adder-6", "HHL-3", "RealAmpRandom-5", "WState-5", "QAOA-6"):
        assert missing_label in msg, f"Expected {missing_label} in error message"


def test_tool_executor_pool_matches_oracle():
    """The agent's ToolExecutor must mount the full Oracle pool at construction."""
    from backends.synthetic_drift import SyntheticDriftBackend
    from tools.quantum_tools import ToolExecutor

    backend = SyntheticDriftBackend("FakeBrisbane")
    executor = ToolExecutor(backend)
    pool = executor._get_mqt_circuits()
    # Should not raise; pool should contain everything the Oracle expects.
    assert_pool_matches_oracle(set(pool.keys()))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
