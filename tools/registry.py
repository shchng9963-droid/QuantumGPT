"""Single source of truth for tool runtime wiring.

LLM-facing JSON schemas live in ``tools.schemas`` and are re-exported by
``tools.quantum_tools``. This registry owns
the execution handler and state-artifact metadata that previously lived in
separate if/elif and mapping blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ToolRuntimeSpec:
    handler_name: str
    accepts_input: bool = True
    artifact_type: str | None = None
    drift_features: tuple[str, ...] = ()


NOISE_DRIFT_FEATURES = (
    "avg_t1_us",
    "avg_t2_us",
    "avg_readout_error",
    "avg_1q_error",
    "avg_2q_error",
)
STRUCTURAL_DRIFT_FEATURES = ("coupling_map", "backend_available")
EXECUTION_DRIFT_FEATURES = NOISE_DRIFT_FEATURES + STRUCTURAL_DRIFT_FEATURES


TOOL_RUNTIME_SPECS: Mapping[str, ToolRuntimeSpec] = {
    "get_backend_health": ToolRuntimeSpec("_get_health", False, "backend_snapshot"),
    "get_qubit_properties": ToolRuntimeSpec(
        "_get_qubit_props", True, "qubit_properties", NOISE_DRIFT_FEATURES
    ),
    "run_circuit": ToolRuntimeSpec(
        "_run_circuit", True, "circuit_result", EXECUTION_DRIFT_FEATURES
    ),
    "list_benchmarks": ToolRuntimeSpec("_list_benchmarks", False),
    "get_coupling_map": ToolRuntimeSpec(
        "_get_coupling_map", False, "coupling_map", STRUCTURAL_DRIFT_FEATURES
    ),
    "detect_drift": ToolRuntimeSpec("_detect_drift", True, "drift_report"),
    "get_calibration_age": ToolRuntimeSpec("_get_calibration_age", False),
    "compare_backends": ToolRuntimeSpec("_compare_backends"),
    "transpile_circuit": ToolRuntimeSpec(
        "_transpile_circuit", True, "transpiled_circuit", EXECUTION_DRIFT_FEATURES
    ),
    "apply_mitigation": ToolRuntimeSpec(
        "_apply_mitigation", True, "mitigation_result", EXECUTION_DRIFT_FEATURES
    ),
    "predict_fidelity": ToolRuntimeSpec(
        "_predict_fidelity", True, "predicted_fidelity", EXECUTION_DRIFT_FEATURES
    ),
    "rabi_experiment": ToolRuntimeSpec(
        "_rabi_experiment", True, "lab_experiment_result", NOISE_DRIFT_FEATURES
    ),
    "fit_rabi": ToolRuntimeSpec("_fit_rabi", True, "fit_result", NOISE_DRIFT_FEATURES),
    "diagnose_and_suggest": ToolRuntimeSpec("_diagnose"),
    "ramsey_experiment": ToolRuntimeSpec("_ramsey_experiment", True, "lab_experiment_result", NOISE_DRIFT_FEATURES),
    "fit_ramsey": ToolRuntimeSpec("_fit_ramsey", True, "fit_result", NOISE_DRIFT_FEATURES),
    "t1_experiment": ToolRuntimeSpec("_t1_experiment", True, "lab_experiment_result", NOISE_DRIFT_FEATURES),
    "fit_t1": ToolRuntimeSpec("_fit_t1", True, "fit_result", NOISE_DRIFT_FEATURES),
    "drag_calibration": ToolRuntimeSpec("_drag_calibration"),
    "randomized_benchmarking": ToolRuntimeSpec("_randomized_benchmarking"),
    "next_best_experiment": ToolRuntimeSpec("_next_best_experiment"),
}


def get_tool_runtime_spec(tool_name: str) -> ToolRuntimeSpec | None:
    return TOOL_RUNTIME_SPECS.get(tool_name)


def validate_tool_registry(tool_definitions: Iterable[dict]) -> None:
    """Fail fast when LLM-visible tools and executable tools diverge."""
    definition_names = {str(item["name"]) for item in tool_definitions}
    runtime_names = set(TOOL_RUNTIME_SPECS)
    if definition_names != runtime_names:
        missing_runtime = sorted(definition_names - runtime_names)
        missing_schema = sorted(runtime_names - definition_names)
        raise RuntimeError(
            "Tool registry mismatch: "
            f"missing_runtime={missing_runtime}, missing_schema={missing_schema}"
        )


__all__ = [
    "TOOL_RUNTIME_SPECS",
    "ToolRuntimeSpec",
    "get_tool_runtime_spec",
    "NOISE_DRIFT_FEATURES",
    "STRUCTURAL_DRIFT_FEATURES",
    "EXECUTION_DRIFT_FEATURES",
    "validate_tool_registry",
]
