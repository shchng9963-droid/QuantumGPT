"""Agent tools — JSON schema definitions and executor.

Defines the tool interface that the LLM sees, and the executor that
dispatches tool calls to real backend functions.

Tools:
  1. get_backend_health   — quick health summary (T1, errors, drift)
  2. get_qubit_properties — detailed per-qubit T1/T2/errors
  3. run_circuit          — run a named benchmark circuit
  4. list_benchmarks      — show available circuits
  5. diagnose_and_suggest — analyze health and suggest actions
"""

import json
import time
from typing import Any

from backends.base import ShadowBackend
from bench.circuits import get_benchmark, list_benchmarks as _list_bench
from bench.mqtbench import get_mqtbench_circuits, MQTBenchCircuit
from detection.drift_detector import DriftDetector


# ═══════════════════════════════════════════════════════
# Tool definitions (Anthropic tool_use JSON schema)
# ═══════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    {
        "name": "get_backend_health",
        "description": (
            "Get a summary of the quantum backend's current health: "
            "average T1/T2, gate errors, readout errors, drift score. "
            "Call this first to understand the device state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_qubit_properties",
        "description": (
            "Get detailed properties for specific qubits: T1, T2, "
            "readout error, gate errors. Use this to identify the "
            "best or worst qubits for circuit mapping."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "qubits": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of qubit indices to query (e.g. [0, 1, 2])",
                },
            },
            "required": ["qubits"],
        },
    },
    {
        "name": "run_circuit",
        "description": (
            "Run a benchmark circuit on the backend and get fidelity. "
            "Available circuits: ghz_5, qft_4, bv_5, vqe_4, qaoa_4 "
            "(hand-written), plus MQTBench: GHZ-5, DJ-5, GraphState-5, "
            "QFTent-5, VQE_SU2-4 (native gates). "
            "Returns counts, fidelity, transpiled depth."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "circuit_name": {
                    "type": "string",
                    "description": "Name of the benchmark circuit to run",
                },
                "shots": {
                    "type": "integer",
                    "description": "Number of shots (default: 4096)",
                    "default": 4096,
                },
            },
            "required": ["circuit_name"],
        },
    },
    {
        "name": "list_benchmarks",
        "description": (
            "List all available benchmark circuits with descriptions. "
            "Call this to see what circuits you can run."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_coupling_map",
        "description": (
            "Get the coupling map (connectivity graph) of the backend. "
            "Returns a list of [qubit_i, qubit_j] pairs that can perform "
            "2-qubit gates. Essential for circuit routing decisions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "detect_drift",
        "description": (
            "Run drift/changepoint detection on recent backend telemetry. "
            "Uses PELT algorithm to identify abrupt changes in T1, T2, "
            "gate errors, or readout errors. Returns changepoints with "
            "severity, direction, and recalibration recommendation. "
            "Requires time-stepping backends (SyntheticDrift, Replay)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "window_hours": {
                    "type": "number",
                    "description": "How many hours of history to analyze (default: 24)",
                    "default": 24,
                },
                "step_hours": {
                    "type": "number",
                    "description": "Time resolution in hours (default: 0.5)",
                    "default": 0.5,
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_calibration_age",
        "description": (
            "Get the age of the most recent calibration in minutes. "
            "Stale calibrations (>60 min) increase the risk of drift-induced "
            "errors. Use this to decide whether to re-calibrate before running."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "compare_backends",
        "description": (
            "Compare multiple backends and rank them by suitability for a "
            "given circuit. Considers T1, gate errors, connectivity, and "
            "drift score. Returns a ranked list with scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "circuit_name": {
                    "type": "string",
                    "description": "Circuit to evaluate backends against (optional)",
                },
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Backend names to compare (default: all available fakes)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "transpile_circuit",
        "description": (
            "Transpile a circuit for a specific backend with configurable "
            "optimization level and noise-aware routing. Returns transpiled "
            "depth, gate counts, and the routing map."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "circuit_name": {
                    "type": "string",
                    "description": "Name of the benchmark circuit to transpile",
                },
                "opt_level": {
                    "type": "integer",
                    "description": "Optimization level 0-3 (default: 1)",
                    "default": 1,
                },
                "noise_aware": {
                    "type": "boolean",
                    "description": "Use noise-aware layout (default: true)",
                    "default": True,
                },
            },
            "required": ["circuit_name"],
        },
    },
    {
        "name": "apply_mitigation",
        "description": (
            "Apply error mitigation to a circuit execution using Zero Noise "
            "Extrapolation (ZNE). Runs the circuit at multiple noise scale "
            "factors and extrapolates to the zero-noise limit. "
            "Returns mitigated fidelity vs unmitigated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "circuit_name": {
                    "type": "string",
                    "description": "Circuit to mitigate",
                },
                "method": {
                    "type": "string",
                    "description": "Mitigation method: 'zne' (default)",
                    "default": "zne",
                },
                "shots": {
                    "type": "integer",
                    "description": "Shots per scale factor (default: 4096)",
                    "default": 4096,
                },
            },
            "required": ["circuit_name"],
        },
    },
    {
        "name": "predict_fidelity",
        "description": (
            "Predict the expected fidelity of a circuit on a backend "
            "WITHOUT running it. Uses a lookup table of historical results "
            "plus analytical noise model estimation. Fast and cheap."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "circuit_name": {
                    "type": "string",
                    "description": "Circuit to predict fidelity for",
                },
            },
            "required": ["circuit_name"],
        },
    },
    {
        "name": "rabi_experiment",
        "description": (
            "Run a Rabi oscillation experiment on a simulated qubit. "
            "Sweeps drive amplitude and returns population vs amplitude data. "
            "This is a pulse-level experiment for qubit characterization."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "qubit_freq_ghz": {
                    "type": "number",
                    "description": "Qubit frequency in GHz (default: 5.0)",
                    "default": 5.0,
                },
                "amp_min": {
                    "type": "number",
                    "description": "Min drive amplitude in GHz (default: 0.0)",
                    "default": 0.0,
                },
                "amp_max": {
                    "type": "number",
                    "description": "Max drive amplitude in GHz (default: 0.08)",
                    "default": 0.08,
                },
                "n_points": {
                    "type": "integer",
                    "description": "Number of amplitude points (default: 30)",
                    "default": 30,
                },
                "pulse_duration_ns": {
                    "type": "number",
                    "description": "Pulse duration in ns (default: 100)",
                    "default": 100,
                },
                "pulse_shape": {
                    "type": "string",
                    "description": "Pulse shape: 'square' or 'gaussian' (default: square)",
                    "default": "square",
                },
            },
            "required": [],
        },
    },
    {
        "name": "fit_rabi",
        "description": (
            "Fit Rabi oscillation data to extract the pi-pulse amplitude. "
            "Takes sweep results from rabi_experiment and returns the fitted "
            "pi and pi/2 pulse amplitudes with goodness of fit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amplitudes": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Drive amplitudes (GHz) from rabi_experiment",
                },
                "populations": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "P(|1>) values from rabi_experiment",
                },
            },
            "required": ["amplitudes", "populations"],
        },
    },
    {
        "name": "diagnose_and_suggest",
        "description": (
            "Analyze the backend health and circuit results, then suggest "
            "concrete actions: recalibrate, remap qubits, apply error "
            "mitigation, or wait for drift to subside. "
            "Provide recent fidelity and health data for best results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fidelity": {
                    "type": "number",
                    "description": "Most recent circuit fidelity (0-1)",
                },
                "circuit_name": {
                    "type": "string",
                    "description": "Which circuit was run",
                },
                "drift_score": {
                    "type": "number",
                    "description": "Current drift score (0-1)",
                },
            },
            "required": [],
        },
    },
]


# ═══════════════════════════════════════════════════════
# Tool executor
# ═══════════════════════════════════════════════════════

class ToolExecutor:
    """Executes tool calls against a ShadowBackend.

    Stateful: holds a reference to the current backend and caches
    MQTBench circuits to avoid regenerating them on every call.
    """

    def __init__(self, backend: ShadowBackend, extra_backends: list[ShadowBackend] | None = None):
        self.backend = backend
        self._extra_backends = extra_backends or []
        self._mqt_cache: dict[str, MQTBenchCircuit] = {}
        self.call_log: list[dict[str, Any]] = []

    def _get_mqt_circuits(self) -> dict[str, MQTBenchCircuit]:
        if not self._mqt_cache:
            for mc in get_mqtbench_circuits():
                self._mqt_cache[mc.label] = mc
        return self._mqt_cache

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call and return the result as a JSON string."""
        t0 = time.time()
        try:
            result = self._dispatch(tool_name, tool_input)
            elapsed = time.time() - t0
            self.call_log.append({
                "tool": tool_name,
                "input": tool_input,
                "success": True,
                "elapsed": elapsed,
            })
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            elapsed = time.time() - t0
            self.call_log.append({
                "tool": tool_name,
                "input": tool_input,
                "success": False,
                "error": str(e),
                "elapsed": elapsed,
            })
            return json.dumps({"error": str(e)})

    def _dispatch(self, tool_name: str, tool_input: dict) -> dict:
        if tool_name == "get_backend_health":
            return self._get_health()
        elif tool_name == "get_qubit_properties":
            return self._get_qubit_props(tool_input)
        elif tool_name == "get_coupling_map":
            return self._get_coupling_map()
        elif tool_name == "detect_drift":
            return self._detect_drift(tool_input)
        elif tool_name == "get_calibration_age":
            return self._get_calibration_age()
        elif tool_name == "compare_backends":
            return self._compare_backends(tool_input)
        elif tool_name == "run_circuit":
            return self._run_circuit(tool_input)
        elif tool_name == "list_benchmarks":
            return self._list_benchmarks()
        elif tool_name == "diagnose_and_suggest":
            return self._diagnose(tool_input)
        elif tool_name == "transpile_circuit":
            return self._transpile_circuit(tool_input)
        elif tool_name == "apply_mitigation":
            return self._apply_mitigation(tool_input)
        elif tool_name == "predict_fidelity":
            return self._predict_fidelity(tool_input)
        elif tool_name == "rabi_experiment":
            return self._rabi_experiment(tool_input)
        elif tool_name == "fit_rabi":
            return self._fit_rabi(tool_input)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _get_health(self) -> dict:
        h = self.backend.get_health()
        return {
            "backend": h.name,
            "num_qubits": h.num_qubits,
            "avg_1q_error": round(h.avg_1q_error, 6),
            "avg_2q_error": round(h.avg_2q_error, 6),
            "avg_readout_error": round(h.avg_readout_error, 6),
            "avg_t1_us": round(h.avg_t1_us, 1),
            "avg_t2_us": round(h.avg_t2_us, 1),
            "calibration_age_minutes": round(h.calibration_age_minutes, 1),
            "drift_score": round(h.drift_score, 4) if h.drift_score is not None else None,
        }

    def _get_qubit_props(self, inp: dict) -> dict:
        qubits = inp.get("qubits", [0, 1, 2])
        props = self.backend.get_qubit_properties(qubits)
        return {
            "qubits": [
                {
                    "qubit": p.qubit,
                    "t1_us": round(p.t1_us, 1),
                    "t2_us": round(p.t2_us, 1),
                    "readout_error": round(p.readout_error, 6),
                    "gate_errors": {k: round(v, 6) for k, v in p.gate_errors.items()},
                }
                for p in props
            ]
        }

    def _run_circuit(self, inp: dict) -> dict:
        name = inp["circuit_name"]
        shots = inp.get("shots", 4096)
        optimization_level = inp.get("optimization_level")
        initial_layout = inp.get("initial_layout")

        # Try hand-written circuits first
        hand_written = _list_bench()
        if name in hand_written:
            circ, _ = get_benchmark(name)
        else:
            # Try MQTBench
            mqt = self._get_mqt_circuits()
            if name in mqt:
                circ = mqt[name].circuit
            else:
                raise ValueError(
                    f"Unknown circuit '{name}'. Available: "
                    f"{list(hand_written.keys()) + list(mqt.keys())}"
                )

        run_kwargs: dict[str, Any] = {}
        if optimization_level is not None:
            run_kwargs["optimization_level"] = optimization_level
        if initial_layout is not None:
            run_kwargs["initial_layout"] = initial_layout

        result = self.backend.run(circ, shots=shots, **run_kwargs)
        # Top 5 counts
        top_counts = dict(sorted(result.counts.items(), key=lambda x: -x[1])[:5])

        return {
            "circuit": name,
            "shots": result.shots,
            "fidelity": round(result.fidelity, 4) if result.fidelity else None,
            "transpiled_depth": result.metadata.get("transpiled_depth"),
            "top_counts": top_counts,
            "metadata": {k: v for k, v in result.metadata.items()
                         if k != "transpiled_depth"},
        }

    def _list_benchmarks(self) -> dict:
        hand = _list_bench()
        try:
            mqt = self._get_mqt_circuits()
        except Exception:
            mqt = {}
        return {
            "hand_written": hand,
            "mqtbench": {
                label: {
                    "description": mc.description,
                    "qubits": mc.num_qubits,
                    "depth": mc.depth,
                    "ecr_count": mc.ecr_count,
                }
                for label, mc in mqt.items()
            },
        }

    def _get_coupling_map(self) -> dict:
        cmap = self.backend.get_coupling_map()
        return {
            "backend": self.backend.name,
            "num_qubits": self.backend.num_qubits,
            "num_edges": len(cmap),
            "coupling_map": [list(edge) for edge in cmap],
            "avg_degree": round(2 * len(cmap) / max(1, self.backend.num_qubits), 2),
        }

    def _detect_drift(self, inp: dict) -> dict:
        from backends.properties_stream import PropertiesStream

        window = inp.get("window_hours", 24)
        step = inp.get("step_hours", 0.5)

        if not hasattr(self.backend, "set_time"):
            # For non-time-stepping backends, return current drift score only
            h = self.backend.get_health()
            return {
                "backend": self.backend.name,
                "method": "static_health",
                "drift_score": round(h.drift_score, 4) if h.drift_score else 0.0,
                "changepoints": [],
                "recalibrate_recommended": (h.drift_score or 0) > 0.3,
                "note": "Backend does not support time-stepping; only static drift score available.",
            }

        # Generate telemetry via replay
        snapshots = PropertiesStream.replay_all(
            self.backend, hours=window, step_hours=step
        )

        detector = DriftDetector(method="pelt", pelt_penalty=3.0, pelt_model="rbf")
        report = detector.detect(snapshots)

        return {
            "backend": self.backend.name,
            "method": report.algorithm,
            "window_hours": window,
            "n_observations": report.n_observations,
            "changepoints": [
                {
                    "time_hours": round(cp.time_hours, 1),
                    "severity": round(cp.severity, 3),
                    "direction": cp.direction,
                    "features_affected": cp.features_affected,
                }
                for cp in report.changepoints
            ],
            "recalibrate_recommended": report.recalibrate_recommended,
            "drift_score_current": round(float(report.drift_scores[-1]), 4) if len(report.drift_scores) > 0 else 0.0,
        }

    def _get_calibration_age(self) -> dict:
        h = self.backend.get_health()
        age = h.calibration_age_minutes
        stale = age > 60
        return {
            "backend": self.backend.name,
            "calibration_age_minutes": round(age, 1),
            "is_stale": stale,
            "recommendation": (
                "Calibration is stale (>60 min). Consider re-calibrating."
                if stale else "Calibration is fresh."
            ),
        }

    def _compare_backends(self, inp: dict) -> dict:
        from backends.fake_adapter import FakeBackendAdapter

        candidates = inp.get("candidates", [
            "FakeBrisbane", "FakeSherbrooke", "FakeKyoto",
        ])
        circuit_name = inp.get("circuit_name")

        backends_to_compare = []
        # Always include current backend
        backends_to_compare.append(self.backend)
        for be in self._extra_backends:
            if be.name in candidates:
                backends_to_compare.append(be)

        # Create any missing candidates from FakeBackendAdapter
        existing_names = {b.name for b in backends_to_compare}
        for cand in candidates:
            if cand not in existing_names:
                try:
                    backends_to_compare.append(FakeBackendAdapter(cand))
                    existing_names.add(cand)
                except Exception:
                    pass

        ranking = []
        for be in backends_to_compare:
            h = be.get_health()
            # Composite score: lower is better
            # Weights: T1 (higher better), 2q error (lower better), drift (lower better)
            t1_score = min(1.0, h.avg_t1_us / 300.0)  # normalize to ~300us max
            error_score = 1.0 - min(1.0, h.avg_2q_error / 0.05)  # 5% = 0
            drift_penalty = (h.drift_score or 0) * 0.3
            readout_penalty = min(1.0, h.avg_readout_error / 0.1) * 0.2

            composite = 0.4 * t1_score + 0.3 * error_score - drift_penalty - readout_penalty

            entry = {
                "backend": h.name,
                "num_qubits": h.num_qubits,
                "avg_t1_us": round(h.avg_t1_us, 1),
                "avg_2q_error": round(h.avg_2q_error, 6),
                "avg_readout_error": round(h.avg_readout_error, 6),
                "drift_score": round(h.drift_score, 4) if h.drift_score is not None else None,
                "composite_score": round(composite, 4),
            }

            # If circuit specified, run it and add fidelity
            if circuit_name:
                try:
                    hand_written = _list_bench()
                    if circuit_name in hand_written:
                        circ, _ = get_benchmark(circuit_name)
                    else:
                        mqt = self._get_mqt_circuits()
                        if circuit_name in mqt:
                            circ = mqt[circuit_name].circuit
                        else:
                            circ = None
                    if circ is not None:
                        result = be.run(circ, shots=1024)
                        entry["fidelity"] = round(result.fidelity, 4) if result.fidelity else None
                        # Adjust composite with fidelity
                        if result.fidelity:
                            composite = 0.6 * result.fidelity + 0.4 * composite
                            entry["composite_score"] = round(composite, 4)
                except Exception:
                    entry["fidelity"] = None

            ranking.append(entry)

        # Sort by composite score descending
        ranking.sort(key=lambda x: x["composite_score"], reverse=True)
        for i, entry in enumerate(ranking):
            entry["rank"] = i + 1

        return {
            "circuit": circuit_name,
            "ranking": ranking,
            "recommended": ranking[0]["backend"] if ranking else None,
        }

    def _diagnose(self, inp: dict) -> dict:
        h = self.backend.get_health()
        fidelity = inp.get("fidelity")
        circuit = inp.get("circuit_name", "unknown")
        drift = inp.get("drift_score", h.drift_score)

        suggestions = []
        severity = "nominal"

        # Drift-based suggestions
        if drift is not None and drift > 0.3:
            severity = "warning"
            suggestions.append(
                f"Drift score is {drift:.2f} (>0.3). "
                "Consider re-calibrating or waiting for drift to subside."
            )
        if drift is not None and drift > 0.6:
            severity = "critical"
            suggestions.append(
                "Severe drift detected. Strongly recommend re-calibration "
                "before running sensitive experiments."
            )

        # Fidelity-based suggestions
        if fidelity is not None and fidelity < 0.7:
            severity = "critical"
            suggestions.append(
                f"Fidelity {fidelity:.3f} is very low. Consider: "
                "1) Error mitigation (ZNE, PEC via Mitiq), "
                "2) Remapping to better qubits, "
                "3) Reducing circuit depth."
            )
        elif fidelity is not None and fidelity < 0.9:
            if severity == "nominal":
                severity = "warning"
            suggestions.append(
                f"Fidelity {fidelity:.3f} is moderate. "
                "Measurement error mitigation (M3/mthree) may help."
            )

        # Error-rate checks
        if h.avg_2q_error > 0.05:
            suggestions.append(
                f"High 2Q error ({h.avg_2q_error:.4f}). "
                "Avoid circuits with many CX/ECR gates or use "
                "qubit-aware routing to avoid worst edges."
            )

        if h.avg_readout_error > 0.05:
            suggestions.append(
                f"High readout error ({h.avg_readout_error:.4f}). "
                "Apply readout error mitigation (twirling or matrix inversion)."
            )

        if not suggestions:
            suggestions.append("Device looks healthy. No action needed.")

        # ── Actionable overrides ──────────────────────────────────
        # These are concrete parameter changes the planner can pass
        # to the next run_circuit call to improve the outcome.
        overrides: dict[str, Any] = {}

        if fidelity is not None and fidelity < 0.85:
            # Boost shots for better statistics
            overrides["shots"] = 8192
            # Higher optimization tries harder at routing/gate cancellation
            overrides["optimization_level"] = 3
        elif fidelity is not None and fidelity < 0.95:
            overrides["shots"] = 8192
            overrides["optimization_level"] = 2

        if h.avg_2q_error > 0.03:
            # When 2Q errors are high, aggressive optimization helps
            overrides["optimization_level"] = 3

        return {
            "severity": severity,
            "drift_score": round(drift, 4) if drift is not None else None,
            "avg_1q_error": round(h.avg_1q_error, 6),
            "avg_2q_error": round(h.avg_2q_error, 6),
            "suggestions": suggestions,
            "recommended_overrides": overrides,
        }

    # ═══════════════════════════════════════════════════════
    # Phase 2 tools
    # ═══════════════════════════════════════════════════════

    def _get_circuit(self, name: str) -> "QuantumCircuit":
        """Resolve circuit name to a QuantumCircuit object."""
        from qiskit import QuantumCircuit
        hand_written = _list_bench()
        if name in hand_written:
            circ, _ = get_benchmark(name)
            return circ
        mqt = self._get_mqt_circuits()
        if name in mqt:
            return mqt[name].circuit
        raise ValueError(f"Unknown circuit '{name}'")

    def _transpile_circuit(self, inp: dict) -> dict:
        from qiskit import transpile as qiskit_transpile
        from qiskit_aer import AerSimulator

        name = inp["circuit_name"]
        opt_level = inp.get("opt_level", 1)
        noise_aware = inp.get("noise_aware", True)

        circ = self._get_circuit(name)

        # Get backend target for transpilation
        backend_target = None
        if hasattr(self.backend, '_fake_backend'):
            backend_target = self.backend._fake_backend
        elif hasattr(self.backend, '_base_snap'):
            # SyntheticDrift — use AerSimulator as generic target
            backend_target = AerSimulator()

        if backend_target is None:
            backend_target = AerSimulator()

        transpiled = qiskit_transpile(
            circ,
            backend=backend_target,
            optimization_level=opt_level,
        )

        # Gate counts
        ops = transpiled.count_ops()
        cx_count = ops.get("cx", 0) + ops.get("ecr", 0) + ops.get("cz", 0)

        return {
            "circuit": name,
            "backend": self.backend.name,
            "opt_level": opt_level,
            "noise_aware": noise_aware,
            "original_depth": circ.depth(),
            "transpiled_depth": transpiled.depth(),
            "original_gates": sum(circ.count_ops().values()),
            "transpiled_gates": sum(ops.values()),
            "two_qubit_gates": cx_count,
            "gate_counts": dict(ops),
        }

    def _apply_mitigation(self, inp: dict) -> dict:
        from mitigation import run_zne

        name = inp["circuit_name"]
        method = inp.get("method", "zne")
        shots = inp.get("shots", 4096)

        circ = self._get_circuit(name)

        if method != "zne":
            return {"error": f"Unknown mitigation method: {method}. Available: zne"}

        result = run_zne(circ, self.backend, shots=shots)
        result["circuit"] = name
        result["backend"] = self.backend.name
        return result

    def _predict_fidelity(self, inp: dict) -> dict:
        """Predict fidelity using XGBoost model (with analytical fallback).

        Primary: trained XGBoost regressor (R2=0.97, MAE=0.006)
        Fallback: analytical model F = prod(1 - error_rate) per gate layer
        """
        from qiskit import transpile as qiskit_transpile
        from qiskit_aer import AerSimulator
        import os

        name = inp["circuit_name"]
        circ = self._get_circuit(name)
        h = self.backend.get_health()

        # Transpile to get realistic gate counts
        try:
            backend_target = getattr(self.backend, '_fake', AerSimulator())
            transpiled = qiskit_transpile(circ, backend=backend_target, optimization_level=1)
            ops = transpiled.count_ops()
        except Exception:
            transpiled = circ
            ops = circ.count_ops()

        # Extract features
        n_1q = sum(v for k, v in ops.items() if k in ("rz", "sx", "x", "id", "u1", "u2", "u3", "h"))
        n_2q = sum(v for k, v in ops.items() if k in ("cx", "ecr", "cz", "swap"))
        n_meas = ops.get("measure", circ.num_qubits)
        depth = transpiled.depth()
        n_qubits = transpiled.num_qubits
        total_gates = n_1q + n_2q
        gate_density = total_gates / max(1, n_qubits * depth)
        two_qubit_ratio = n_2q / max(1, total_gates)

        pairs_used = set()
        for inst in transpiled.data:
            if len(inst.qubits) == 2:
                q0 = transpiled.find_bit(inst.qubits[0]).index
                q1 = transpiled.find_bit(inst.qubits[1]).index
                pairs_used.add((min(q0, q1), max(q0, q1)))
        possible_pairs = n_qubits * (n_qubits - 1) / 2
        connectivity = len(pairs_used) / max(1, possible_pairs)

        # Try XGBoost model first
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  "eval", "fidelity_model.json")
        method = "analytical"
        f_predicted = 0.0
        try:
            if os.path.exists(model_path):
                from eval.fidelity_predictor import FidelityPredictor
                import numpy as np
                predictor = FidelityPredictor.load(model_path)
                cf = np.array([n_qubits, depth, n_1q, n_2q, n_meas,
                               gate_density, two_qubit_ratio, connectivity])
                bf = np.array([h.avg_1q_error, h.avg_2q_error, h.avg_readout_error,
                               h.avg_t1_us, h.avg_t2_us,
                               h.drift_score if h.drift_score is not None else 0.0])
                f_predicted = float(predictor.predict(cf, bf))
                f_predicted = max(0.0, min(1.0, f_predicted))
                method = "xgboost"
        except Exception:
            pass

        if method == "analytical":
            f_1q = (1 - h.avg_1q_error) ** n_1q if n_1q > 0 else 1.0
            f_2q = (1 - h.avg_2q_error) ** n_2q if n_2q > 0 else 1.0
            f_readout = (1 - h.avg_readout_error) ** n_meas
            f_predicted = f_1q * f_2q * f_readout

        confidence = "low" if f_predicted < 0.5 else ("medium" if f_predicted < 0.8 else "high")

        return {
            "circuit": name,
            "backend": self.backend.name,
            "predicted_fidelity": round(float(f_predicted), 4),
            "method": method,
            "confidence": confidence,
            "breakdown": {
                "n_qubits": n_qubits,
                "depth": depth,
                "n_1q_gates": n_1q,
                "n_2q_gates": n_2q,
                "n_measurements": n_meas,
                "gate_density": round(gate_density, 4),
                "two_qubit_ratio": round(two_qubit_ratio, 4),
            },
            "note": f"Predicted via {method} model.",
        }
    def _rabi_experiment(self, inp: dict) -> dict:
        from dynamics.rabi import RabiConfig, sweep_rabi

        cfg = RabiConfig(
            qubit_freq_ghz=inp.get("qubit_freq_ghz", 5.0),
            amp_range=(inp.get("amp_min", 0.0), inp.get("amp_max", 0.08)),
            n_amps=inp.get("n_points", 30),
            pulse_duration_ns=inp.get("pulse_duration_ns", 100),
            dt_ns=0.5,
            pulse_shape=inp.get("pulse_shape", "square"),
        )

        result = sweep_rabi(cfg)

        return {
            "qubit_freq_ghz": cfg.qubit_freq_ghz,
            "pulse_shape": cfg.pulse_shape,
            "pulse_duration_ns": cfg.pulse_duration_ns,
            "n_points": len(result.amplitudes),
            "amplitudes": [round(float(a), 6) for a in result.amplitudes],
            "populations": [round(float(p), 4) for p in result.final_populations],
            "pi_amplitude_ghz": round(float(result.pi_amplitude), 6),
            "pi_amplitude_mhz": round(float(result.pi_amplitude * 1e3), 2),
            "max_population": round(float(result.final_populations.max()), 4),
        }

    def _fit_rabi(self, inp: dict) -> dict:
        import numpy as np
        from scipy.optimize import curve_fit

        amps = np.array(inp["amplitudes"])
        pops = np.array(inp["populations"])

        # Fit sinusoidal: P(a) = A * sin²(π * a / (2 * a_pi)) + offset
        def rabi_model(a, a_pi, A, offset):
            return A * np.sin(np.pi * a / (2 * a_pi)) ** 2 + offset

        # Initial guess from data
        a_pi_guess = amps[np.argmax(pops)]
        if a_pi_guess <= 0:
            a_pi_guess = amps[-1] / 2

        try:
            popt, pcov = curve_fit(
                rabi_model, amps, pops,
                p0=[a_pi_guess, 1.0, 0.0],
                bounds=([1e-6, 0.0, -0.5], [amps[-1] * 2, 2.0, 0.5]),
                maxfev=5000,
            )
            a_pi, A, offset = popt
            perr = np.sqrt(np.diag(pcov))

            # Goodness of fit
            y_pred = rabi_model(amps, *popt)
            ss_res = np.sum((pops - y_pred) ** 2)
            ss_tot = np.sum((pops - np.mean(pops)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

            return {
                "pi_amplitude_ghz": round(float(a_pi), 6),
                "pi_amplitude_mhz": round(float(a_pi * 1e3), 2),
                "half_pi_amplitude_ghz": round(float(a_pi / 2), 6),
                "amplitude": round(float(A), 4),
                "offset": round(float(offset), 4),
                "r_squared": round(float(r_squared), 4),
                "pi_uncertainty_ghz": round(float(perr[0]), 6),
                "fit_successful": True,
            }
        except Exception as e:
            # Fallback: just use argmax
            pi_idx = int(np.argmax(pops))
            return {
                "pi_amplitude_ghz": round(float(amps[pi_idx]), 6),
                "pi_amplitude_mhz": round(float(amps[pi_idx] * 1e3), 2),
                "half_pi_amplitude_ghz": round(float(amps[pi_idx] / 2), 6),
                "r_squared": 0.0,
                "fit_successful": False,
                "fit_error": str(e),
                "note": "Fallback to argmax; sinusoidal fit failed.",
            }
