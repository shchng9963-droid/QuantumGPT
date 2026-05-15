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
