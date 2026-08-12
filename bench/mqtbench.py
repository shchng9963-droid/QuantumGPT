"""MQTBench integration — gate-level benchmark circuits.

Wraps MQT Bench to provide pre-selected benchmark circuits at
NATIVEGATES level (IBM Eagle gate set: ECR, RZ, SX, X).

These circuits complement the hand-written circuits in bench/circuits.py
with industry-standard benchmarks that are already decomposed into
physical gate sets.
"""

from dataclasses import dataclass
from typing import Optional

from qiskit.circuit import QuantumCircuit


# v2.5 Sprint A circuit-pool consistency contract.
# When the agent toolset and the deterministic Oracle disagree on which
# circuit labels are reachable, the agent silently fails on every "extra"
# circuit ("Unknown circuit 'Adder-6'") while the Oracle quietly evaluates
# them, producing a benchmark gap that is purely a registration bug rather
# than a capability gap. The pollution observed in v3_main_agent (132/450
# polluted rows) traced back to exactly this drift.
#
# Anything that mounts circuits for the agent should call this contract
# before evaluation, so any future divergence fails loudly at startup.
ORACLE_REQUIRED_LABELS: set[str] = set()  # populated below after EXTENDED_MQTBENCH_SELECTION


def assert_pool_matches_oracle(agent_pool_labels: set[str]) -> None:
    """Raise if the agent's MQTBench pool is missing labels the Oracle expects.

    Args:
        agent_pool_labels: labels currently exposed to the agent (e.g. keys of
            ToolExecutor._mqt_cache).

    Raises:
        AssertionError: when ``ORACLE_REQUIRED_LABELS - agent_pool_labels`` is
            non-empty.
    """
    missing = ORACLE_REQUIRED_LABELS - set(agent_pool_labels)
    if missing:
        raise AssertionError(
            "Agent MQTBench pool is missing circuits the Oracle expects: "
            f"{sorted(missing)}. Either extend the agent selection (see "
            "EXTENDED_MQTBENCH_SELECTION) or shrink the Oracle benchmark set."
        )


@dataclass
class MQTBenchCircuit:
    """A benchmark circuit from MQT Bench with metadata."""
    name: str
    label: str
    circuit: QuantumCircuit
    num_qubits: int
    depth: int
    ecr_count: int
    total_gates: int
    description: str
    gateset: str


# Default selection: 5 representative circuits
DEFAULT_MQTBENCH_SELECTION = [
    ("ghz",          5, "GHZ-5",        "Entanglement benchmark"),
    ("dj",           5, "DJ-5",         "Deutsch-Jozsa oracle"),
    ("graphstate",   5, "GraphState-5", "Graph state preparation"),
    ("qftentangled", 5, "QFTent-5",     "QFT + entanglement"),
    ("vqe_su2",      4, "VQE_SU2-4",    "Variational SU(2) ansatz"),
]


# v2.5 Sprint A — extended candidate pool for "decisive-region" circuit screening.
# These circuits have distinct depth / 2q-count profiles to span the noise-sensitivity spectrum.
EXTENDED_MQTBENCH_SELECTION = DEFAULT_MQTBENCH_SELECTION + [
    ("qaoa",                     6, "QAOA-6",         "6-qubit QAOA MaxCut, depth ~60"),
    ("vqe_real_amp",             5, "RealAmpRandom-5", "Hardware-efficient ansatz, depth ~30"),
    ("wstate",                   5, "WState-5",       "W-state preparation, shallow"),
    ("hhl",                      3, "HHL-3",          "HHL 3-qubit, long-depth fixed"),
    ("cdkm_ripple_carry_adder",  6, "Adder-6",        "CDKM ripple-carry adder, depth ~100"),
]


# Oracle / multi_shock evaluator binds against the EXTENDED pool, so the agent
# must too. Treat this as the source of truth.
ORACLE_REQUIRED_LABELS = {label for _, _, label, _ in EXTENDED_MQTBENCH_SELECTION}


def get_mqtbench_circuits(
    selection: Optional[list[tuple[str, int, str, str]]] = None,
    gateset: str = "ibm_eagle",
) -> list[MQTBenchCircuit]:
    """Generate MQTBench circuits at NATIVEGATES level.

    Args:
        selection: List of (mqt_name, num_qubits, label, description) tuples.
                   Default: DEFAULT_MQTBENCH_SELECTION (5 circuits).
        gateset: Target gate set. Default: "ibm_eagle" (ECR, RZ, SX, X).

    Returns:
        List of MQTBenchCircuit objects.
    """
    from mqt.bench import get_benchmark, BenchmarkLevel
    from mqt.bench.targets import get_target_for_gateset

    if selection is None:
        selection = DEFAULT_MQTBENCH_SELECTION

    results = []
    for mqt_name, n_qubits, label, description in selection:
        target = get_target_for_gateset(gateset, num_qubits=n_qubits)
        circ = get_benchmark(
            mqt_name,
            level=BenchmarkLevel.NATIVEGATES,
            circuit_size=n_qubits,
            target=target,
        )
        ops = dict(circ.count_ops())
        # Some MQTBench circuits (e.g. QAOA) ship without measurements; the agent
        # and oracle backends both rely on counts, so add measure_all when no
        # measurements are present. Preserves the partial-measurement design of
        # algorithms like HHL that intentionally measure only a subset.
        if ops.get("measure", 0) == 0:
            circ.measure_all()
            ops = dict(circ.count_ops())
        ecr = ops.get("ecr", 0)
        total = sum(v for k, v in ops.items() if k not in ("measure", "barrier"))

        results.append(MQTBenchCircuit(
            name=mqt_name,
            label=label,
            circuit=circ,
            num_qubits=circ.num_qubits,
            depth=circ.depth(),
            ecr_count=ecr,
            total_gates=total,
            description=description,
            gateset=gateset,
        ))

    return results


def list_mqtbench_available() -> list[str]:
    """List all available benchmark names in MQT Bench."""
    from mqt.bench.benchmarks import get_available_benchmark_names
    return sorted(get_available_benchmark_names())
