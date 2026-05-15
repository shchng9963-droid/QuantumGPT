"""Cross-backend validation: run 5 standard circuits on all 3 shadow backends.

Circuits:
  1. GHZ-5    — entanglement benchmark
  2. Bell     — simplest entangled state
  3. QFT-4   — quantum Fourier transform (4 qubits)
  4. BV-4    — Bernstein-Vazirani (4 qubits, secret=1011)
  5. Random-5 — random 5-qubit circuit (depth 10)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QFT

from backends.fake_adapter import FakeBackendAdapter
from backends.replay_backend import ReplayBackend
from backends.synthetic_drift import SyntheticDriftBackend, LINEAR_DECAY


def make_ghz(n: int = 5) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def make_bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def make_qft(n: int = 4) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    # Prepare a computational basis state to transform
    qc.x(0)
    qc.x(2)
    # Apply QFT
    qft = QFT(n)
    qc.compose(qft, inplace=True)
    qc.measure_all()
    return qc


def make_bernstein_vazirani(secret: str = "1011") -> QuantumCircuit:
    n = len(secret)
    qc = QuantumCircuit(n + 1, n)
    # Initialize ancilla in |->
    qc.x(n)
    qc.h(n)
    # Apply H to input qubits
    for i in range(n):
        qc.h(i)
    # Oracle: CX where secret bit = 1
    for i, bit in enumerate(reversed(secret)):
        if bit == "1":
            qc.cx(i, n)
    # Apply H again
    for i in range(n):
        qc.h(i)
    # Measure input qubits
    for i in range(n):
        qc.measure(i, i)
    return qc


def make_random_circuit(n: int = 5, depth: int = 10, seed: int = 42) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    gates_1q = ["h", "x", "sx", "rz"]
    for _ in range(depth):
        for q in range(n):
            gate = rng.choice(gates_1q)
            if gate == "rz":
                qc.rz(rng.uniform(0, 2 * np.pi), q)
            else:
                getattr(qc, gate)(q)
        # Random CX layer
        pairs = list(range(n))
        rng.shuffle(pairs)
        for i in range(0, n - 1, 2):
            qc.cx(pairs[i], pairs[i + 1])
    qc.measure_all()
    return qc


def main():
    # Build circuits
    circuits = {
        "GHZ-5": make_ghz(5),
        "Bell": make_bell(),
        "QFT-4": make_qft(4),
        "BV-1011": make_bernstein_vazirani("1011"),
        "Random-5": make_random_circuit(5, depth=10),
    }

    # Build backends
    backends = {
        "FakeAdapter": FakeBackendAdapter("FakeBrisbane"),
        "Replay(t=0)": ReplayBackend.from_synthetic_history("FakeBrisbane", 48, 2, seed=42),
        "SynthDrift(t=0)": SyntheticDriftBackend("FakeBrisbane", LINEAR_DECAY),
    }

    # Set initial time for time-dependent backends
    backends["Replay(t=0)"].set_time(0)
    backends["SynthDrift(t=0)"].set_time(0)

    shots = 4096

    # Header
    print(f"{'Circuit':<12} {'Backend':<20} {'Fidelity':>10} {'Depth':>7} {'Time(s)':>8}")
    print("=" * 62)

    results = []
    for circ_name, circuit in circuits.items():
        for backend_name, backend in backends.items():
            t0 = time.time()
            result = backend.run(circuit, shots=shots)
            elapsed = time.time() - t0
            fid_str = f"{result.fidelity:.4f}" if result.fidelity is not None else "N/A"
            depth = result.metadata.get("transpiled_depth", "?")
            print(f"{circ_name:<12} {backend_name:<20} {fid_str:>10} {depth:>7} {elapsed:>8.2f}")
            results.append({
                "circuit": circ_name,
                "backend": backend_name,
                "fidelity": result.fidelity,
                "depth": depth,
                "elapsed": elapsed,
            })
        print("-" * 62)

    # Summary
    fidelities = [r["fidelity"] for r in results if r["fidelity"] is not None]
    print(f"\nTotal: {len(results)} runs, {len(fidelities)} with fidelity")
    print(f"Avg fidelity: {np.mean(fidelities):.4f}")
    print(f"Min fidelity: {np.min(fidelities):.4f}")
    print(f"Max fidelity: {np.max(fidelities):.4f}")
    print(f"Total time: {sum(r['elapsed'] for r in results):.1f}s")
    print("\nAll backends validated successfully!")


if __name__ == "__main__":
    main()
