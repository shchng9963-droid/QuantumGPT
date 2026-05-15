"""Standard benchmark circuits for Phase 0–2 evaluation.

5 representative tasks covering different circuit structures:
  1. GHZ-5: entanglement, shallow depth
  2. QFT-4: Quantum Fourier Transform, medium depth
  3. BV-5: Bernstein-Vazirani oracle, structured
  4. VQE-4: variational ansatz (hardware-efficient), parameterized
  5. QAOA-MaxCut-4: combinatorial optimization

Each function returns (circuit_with_measurements, ideal_distribution)
where ideal_distribution is the noiseless output for fidelity calculation.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter


def ghz(n: int = 5) -> tuple[QuantumCircuit, dict[str, float]]:
    """GHZ state: (|00...0> + |11...1>) / sqrt(2)"""
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()

    ideal = {"0" * n: 0.5, "1" * n: 0.5}
    return qc, ideal


def qft(n: int = 4) -> tuple[QuantumCircuit, dict[str, float]]:
    """Quantum Fourier Transform on |0...0>.
    
    QFT on |0> gives uniform superposition, so ideal output is uniform.
    """
    qc = QuantumCircuit(n)
    
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            qc.cp(angle, j, i)
    
    # Swap qubits for correct ordering
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    
    qc.measure_all()

    # QFT|0> = uniform superposition
    num_states = 2 ** n
    ideal = {format(i, f"0{n}b"): 1.0 / num_states for i in range(num_states)}
    return qc, ideal


def bernstein_vazirani(secret: str = "10110") -> tuple[QuantumCircuit, dict[str, float]]:
    """Bernstein-Vazirani algorithm to find the secret string.
    
    The ideal output is the secret string with probability 1.
    """
    n = len(secret)
    qc = QuantumCircuit(n + 1, n)  # n data + 1 ancilla

    # Initialize ancilla in |->
    qc.x(n)
    qc.h(n)

    # Hadamard on data qubits
    for i in range(n):
        qc.h(i)

    # Oracle: CX for each '1' in the secret
    for i, bit in enumerate(secret):
        if bit == "1":
            qc.cx(i, n)

    # Hadamard on data qubits
    for i in range(n):
        qc.h(i)

    # Measure data qubits only
    for i in range(n):
        qc.measure(i, i)

    # Ideal: secret string with probability 1
    # Qiskit bit ordering: LSB is qubit 0
    ideal = {secret[::-1]: 1.0}  # reverse for Qiskit convention
    return qc, ideal


def vqe_ansatz(n: int = 4, layers: int = 2) -> tuple[QuantumCircuit, dict[str, float]]:
    """Hardware-efficient VQE ansatz with fixed (optimized) parameters.
    
    Uses RY + CX layers. Parameters set to approximate H2 ground state.
    """
    qc = QuantumCircuit(n)

    # "Optimized" parameters (representative, not actual H2 solution)
    params = [
        [0.3, -0.5, 0.7, -0.2],  # layer 1
        [0.1, 0.4, -0.3, 0.6],   # layer 2
    ]

    for layer in range(layers):
        for q in range(n):
            qc.ry(params[layer][q], q)
        for q in range(n - 1):
            qc.cx(q, q + 1)

    qc.measure_all()

    # Ideal distribution computed from noiseless simulation
    # We'll compute it dynamically rather than hardcode
    ideal = None  # will be computed by the evaluator
    return qc, ideal


def qaoa_maxcut(n: int = 4, p: int = 1) -> tuple[QuantumCircuit, dict[str, float]]:
    """QAOA for MaxCut on a simple ring graph.
    
    Ring graph: 0-1-2-3-0
    Uses fixed "optimized" gamma and beta parameters.
    """
    qc = QuantumCircuit(n)

    # Edges of ring graph
    edges = [(i, (i + 1) % n) for i in range(n)]

    # "Optimized" QAOA parameters
    gamma = [0.6]  # cost layer parameter
    beta = [0.4]   # mixer layer parameter

    # Initial superposition
    for q in range(n):
        qc.h(q)

    for layer in range(p):
        # Cost layer: exp(-i * gamma * C)
        for i, j in edges:
            qc.cx(i, j)
            qc.rz(2 * gamma[layer], j)
            qc.cx(i, j)

        # Mixer layer: exp(-i * beta * B)
        for q in range(n):
            qc.rx(2 * beta[layer], q)

    qc.measure_all()

    ideal = None  # computed by evaluator
    return qc, ideal


# Registry of all benchmark circuits
BENCHMARKS = {
    "ghz_5": {"builder": lambda: ghz(5), "description": "GHZ-5 entanglement state"},
    "qft_4": {"builder": lambda: qft(4), "description": "4-qubit Quantum Fourier Transform"},
    "bv_5": {"builder": lambda: bernstein_vazirani("10110"), "description": "Bernstein-Vazirani (secret=10110)"},
    "vqe_4": {"builder": lambda: vqe_ansatz(4, 2), "description": "4-qubit VQE hardware-efficient ansatz"},
    "qaoa_4": {"builder": lambda: qaoa_maxcut(4, 1), "description": "4-qubit QAOA MaxCut (ring graph, p=1)"},
}


def get_benchmark(name: str) -> tuple[QuantumCircuit, dict[str, float] | None]:
    """Get a benchmark circuit by name.
    
    Returns (circuit, ideal_distribution).
    ideal_distribution may be None for variational circuits.
    """
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark '{name}'. Available: {list(BENCHMARKS.keys())}")
    return BENCHMARKS[name]["builder"]()


def list_benchmarks() -> dict[str, str]:
    """List all available benchmarks with descriptions."""
    return {k: v["description"] for k, v in BENCHMARKS.items()}
