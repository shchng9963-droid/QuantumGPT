"""OpenAI-compatible tool schemas exposed by QuantumGPT.

This module is intentionally declarative. Runtime dispatch metadata lives in
``tools.registry`` and execution logic lives in ``tools.quantum_tools``.
"""

from __future__ import annotations

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
    {
        "name": "ramsey_experiment",
        "description": (
            "Run a Ramsey fringe experiment on a simulated qubit. "
            "Applies two pi/2 pulses separated by a variable delay with "
            "artificial detuning. Returns the decay envelope and oscillation data. "
            "Use fit_ramsey to extract T2* and detuning from the results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "qubit": {
                    "type": "integer",
                    "description": "Qubit index (default: 0)",
                    "default": 0,
                },
                "delay_max_ns": {
                    "type": "number",
                    "description": "Maximum delay between pi/2 pulses in ns (default: 5000)",
                    "default": 5000.0,
                },
                "n_points": {
                    "type": "integer",
                    "description": "Number of delay points (default: 50)",
                    "default": 50,
                },
                "artificial_detuning_mhz": {
                    "type": "number",
                    "description": "Artificial detuning frequency in MHz (default: 2.0)",
                    "default": 2.0,
                },
                "shots": {
                    "type": "integer",
                    "description": "Shots per point (default: 1024)",
                    "default": 1024,
                },
            },
            "required": [],
        },
    },
    {
        "name": "fit_ramsey",
        "description": (
            "Fit Ramsey fringe data to extract T2* (dephasing time) and detuning. "
            "Takes delay values and P(|1>) populations from ramsey_experiment. "
            "Returns T2* in microseconds, detuning in MHz, and fit quality."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "delays_ns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Delay values in nanoseconds",
                },
                "populations": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "P(|1>) values at each delay",
                },
                "artificial_detuning_mhz": {
                    "type": "number",
                    "description": "Detuning used in experiment for initial guess (default: 2.0)",
                    "default": 2.0,
                },
            },
            "required": ["delays_ns", "populations"],
        },
    },
    {
        "name": "t1_experiment",
        "description": (
            "Run a T1 energy relaxation experiment on a simulated qubit. "
            "Applies a pi-pulse to excite the qubit, then measures decay "
            "as a function of wait time. Returns population vs delay data. "
            "Use fit_t1 to extract the T1 relaxation time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "qubit": {
                    "type": "integer",
                    "description": "Qubit index (default: 0)",
                    "default": 0,
                },
                "delay_max_ns": {
                    "type": "number",
                    "description": "Maximum delay after pi-pulse in ns (default: 500000 = 500 us)",
                    "default": 500000.0,
                },
                "n_points": {
                    "type": "integer",
                    "description": "Number of delay points (default: 50)",
                    "default": 50,
                },
                "shots": {
                    "type": "integer",
                    "description": "Shots per point (default: 1024)",
                    "default": 1024,
                },
            },
            "required": [],
        },
    },
    {
        "name": "fit_t1",
        "description": (
            "Fit T1 relaxation data to extract the energy relaxation time T1. "
            "Takes delay values and P(|1>) populations from t1_experiment. "
            "Returns T1 in microseconds, amplitude, offset, and fit quality."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "delays_ns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Delay values in nanoseconds",
                },
                "populations": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "P(|1>) values at each delay",
                },
            },
            "required": ["delays_ns", "populations"],
        },
    },
    {
        "name": "drag_calibration",
        "description": (
            "Calibrate the DRAG (Derivative Removal by Adiabatic Gate) parameter "
            "to suppress leakage to the |2⟩ state of a transmon qubit. "
            "Sweeps the DRAG α parameter and finds the value that minimizes leakage. "
            "Requires a pre-calibrated π-pulse amplitude (from rabi_experiment/fit_rabi). "
            "Returns optimal α, minimum leakage, and gate fidelity estimate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "qubit": {
                    "type": "integer",
                    "description": "Qubit index (used to fetch anharmonicity from backend). Default 0.",
                    "default": 0,
                },
                "pi_amp_ghz": {
                    "type": "number",
                    "description": "Pre-calibrated π-pulse amplitude in GHz (from Rabi). Default 0.005.",
                    "default": 0.005,
                },
                "anharmonicity_ghz": {
                    "type": "number",
                    "description": "Transmon anharmonicity δ = ω_12 - ω_01 in GHz (negative). Default -0.3.",
                    "default": -0.3,
                },
                "pulse_duration_ns": {
                    "type": "number",
                    "description": "π-pulse duration in ns (default: 100).",
                    "default": 100,
                },
                "pulse_shape": {
                    "type": "string",
                    "description": "Envelope shape: 'gaussian' or 'square' (default: gaussian).",
                    "default": "gaussian",
                },
                "alpha_min": {
                    "type": "number",
                    "description": "Minimum DRAG α to sweep (default: -2.0).",
                    "default": -2.0,
                },
                "alpha_max": {
                    "type": "number",
                    "description": "Maximum DRAG α to sweep (default: 2.0).",
                    "default": 2.0,
                },
                "n_points": {
                    "type": "integer",
                    "description": "Number of α sweep points (default: 21).",
                    "default": 21,
                },
                "refine": {
                    "type": "boolean",
                    "description": "Whether to do a fine sweep around the coarse optimum (default: true).",
                    "default": True,
                },
            },
            "required": [],
        },
    },
    {
        "name": "randomized_benchmarking",
        "description": (
            "Run a single-qubit Randomized Benchmarking (RB) experiment. "
            "Applies random sequences of Clifford gates of increasing length, "
            "measures survival probability decay, and extracts the Error Per "
            "Clifford (EPC). This is the gold-standard metric for single-qubit "
            "gate quality. Returns EPC, depolarizing parameter, and survival curve."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "qubit": {
                    "type": "integer",
                    "description": "Qubit index (default: 0).",
                    "default": 0,
                },
                "sequence_lengths": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Clifford sequence lengths (default: [1,2,4,8,16,32,64]).",
                },
                "n_sequences": {
                    "type": "integer",
                    "description": "Random sequences per length (default: 20).",
                    "default": 20,
                },
                "error_per_gate": {
                    "type": "number",
                    "description": "Depolarizing error per gate (default: uses backend gate error).",
                },
                "shots": {
                    "type": "integer",
                    "description": "Shots per sequence (default: 1024).",
                    "default": 1024,
                },
                "seed": {
                    "type": "integer",
                    "description": "RNG seed for reproducibility (optional).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "next_best_experiment",
        "description": (
            "Use Bayesian Optimization to suggest the next most informative "
            "calibration experiment. Maintains a Gaussian Process surrogate "
            "over the parameter space and uses Expected Improvement to pick "
            "the point that maximally improves the objective. "
            "Pass previous observations as (parameters, objective) pairs. "
            "Returns suggested parameter values with predicted outcome and rationale."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "parameter_bounds": {
                    "type": "object",
                    "description": (
                        "Bounds for each parameter, e.g. "
                        "{\"drag_alpha\": [-2, 2], \"pi_amp_ghz\": [0, 0.01]}."
                    ),
                },
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "parameters": {
                                "type": "object",
                                "description": "Parameter values for this observation.",
                            },
                            "objective": {
                                "type": "number",
                                "description": "Objective value (higher = better).",
                            },
                        },
                    },
                    "description": "List of previous observations [{parameters: {...}, objective: float}].",
                },
                "objective_name": {
                    "type": "string",
                    "description": "Name of the objective metric (default: fidelity).",
                    "default": "fidelity",
                },
                "acquisition": {
                    "type": "string",
                    "description": "Acquisition function: 'ei', 'ucb', or 'thompson' (default: ei).",
                    "default": "ei",
                },
                "seed": {
                    "type": "integer",
                    "description": "RNG seed (optional).",
                },
            },
            "required": ["parameter_bounds", "observations"],
        },
    },
]
