"""Instrumented ToolExecutor that auto-logs results to DuckDB + W&B.

Wraps the base ToolExecutor and intercepts results to persist them.
"""

import json
import os
import time
from typing import Any, Optional

from tools.quantum_tools import ToolExecutor, TOOL_DEFINITIONS
from data.store import DataStore
from data.experiment_record import (
    ExperimentRecord, ExperimentType, ExperimentOutcome,
)

# Optional W&B
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class InstrumentedExecutor:
    """Wraps ToolExecutor, auto-logs every call to DuckDB and optionally W&B.

    Usage:
        from data.instrumented import InstrumentedExecutor
        executor = InstrumentedExecutor(backend)
        executor.execute("get_backend_health", {})
        # Results automatically saved to DuckDB
    """

    def __init__(self, backend, db: Optional[DataStore] = None,
                 wandb_run=None, session_id: Optional[str] = None):
        self.inner = ToolExecutor(backend)
        self.db = db or DataStore()
        self.wandb_run = wandb_run
        self.session_id = session_id
        self._step = 0
        self._last_health: Optional[dict] = None  # cache for ExperimentRecord

    @property
    def call_log(self):
        return self.inner.call_log

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Execute tool and auto-log result."""
        result_str = self.inner.execute(tool_name, tool_input)
        self._step += 1

        try:
            result = json.loads(result_str)
        except json.JSONDecodeError:
            result = {"raw": result_str}

        # Skip logging errors
        if "error" in result:
            return result_str

        # Auto-persist based on tool type
        try:
            self._log_to_db(tool_name, tool_input, result)
        except Exception as e:
            pass  # Don't break the agent on logging failures

        try:
            self._log_to_wandb(tool_name, tool_input, result)
        except Exception:
            pass

        return result_str

    def _log_to_db(self, tool_name: str, tool_input: dict, result: dict):
        if tool_name == "get_backend_health":
            self.db.log_health(result)
            self._last_health = result

        elif tool_name == "get_qubit_properties":
            backend = result.get("qubits", [{}])[0].get("backend", "unknown")
            # Infer backend from the inner executor
            backend = self.inner.backend.name
            self.db.log_qubit_properties(backend, result.get("qubits", []))

        elif tool_name == "run_circuit":
            self.db.log_circuit_run(result, session_id=self.session_id)
            self._log_experiment_record(tool_name, tool_input, result)

        elif tool_name == "diagnose_and_suggest":
            event = {
                "backend": self.inner.backend.name,
                "drift_score": result.get("drift_score"),
                "severity": result.get("severity"),
                "fidelity_after": tool_input.get("fidelity"),
                "suggestions": result.get("suggestions", []),
            }
            self.db.log_drift_event(event)
            self._log_experiment_record(tool_name, tool_input, result)

    def _log_experiment_record(self, tool_name: str, tool_input: dict,
                                result: dict):
        """Create and save an ExperimentRecord for significant operations."""
        backend_name = self.inner.backend.name

        if tool_name == "run_circuit":
            fidelity = result.get("fidelity")
            circuit = result.get("circuit", tool_input.get("circuit_name", "unknown"))
            success = fidelity is not None and fidelity >= 0.7
            if fidelity is None:
                outcome = ExperimentOutcome.ERROR.value
            elif fidelity >= 0.9:
                outcome = ExperimentOutcome.SUCCESS.value
            elif fidelity >= 0.7:
                outcome = ExperimentOutcome.PARTIAL.value
            else:
                outcome = ExperimentOutcome.FAILURE.value

            record = ExperimentRecord(
                experiment_type=ExperimentType.CIRCUIT_BENCHMARK.value,
                backend=backend_name,
                circuit_name=circuit,
                num_qubits=result.get("metadata", {}).get("num_qubits"),
                problem=f"Run {circuit} with {tool_input.get('shots', 4096)} shots",
                plan=["run_circuit"],
                tool_calls=[{"tool": tool_name, "input": tool_input,
                             "result_length": len(json.dumps(result))}],
                fidelity=fidelity,
                success=success,
                outcome=outcome,
                metrics={
                    "transpiled_depth": result.get("transpiled_depth", 0),
                    "transpiled_gates": result.get("metadata", {}).get(
                        "transpiled_gate_count", 0),
                    "shots": result.get("shots", 0),
                },
                backend_snapshot=self._last_health or {},
                summary=(
                    f"{circuit} on {backend_name}: "
                    f"fidelity={fidelity:.4f}, "
                    f"depth={result.get('transpiled_depth', '?')}, "
                    f"outcome={outcome}"
                ) if fidelity else f"{circuit} on {backend_name}: no fidelity",
                model=self._model_name(),
            )
            self.db.experiments.save(record)

        elif tool_name == "diagnose_and_suggest":
            severity = result.get("severity", "unknown")
            record = ExperimentRecord(
                experiment_type=ExperimentType.DRIFT_DIAGNOSIS.value,
                backend=backend_name,
                problem="Diagnose backend health and suggest mitigations",
                plan=["diagnose_and_suggest"],
                tool_calls=[{"tool": tool_name, "input": tool_input,
                             "result_length": len(json.dumps(result))}],
                success=(severity == "nominal"),
                outcome=(ExperimentOutcome.SUCCESS.value if severity == "nominal"
                         else ExperimentOutcome.PARTIAL.value),
                metrics={
                    "drift_score": result.get("drift_score", 0) or 0,
                    "avg_1q_error": result.get("avg_1q_error", 0),
                    "avg_2q_error": result.get("avg_2q_error", 0),
                },
                backend_snapshot=self._last_health or {},
                summary=(
                    f"Diagnosis on {backend_name}: severity={severity}, "
                    f"suggestions={len(result.get('suggestions', []))}"
                ),
                lessons=result.get("suggestions", []),
                model=self._model_name(),
            )
            self.db.experiments.save(record)

    def _model_name(self) -> str:
        """Best-effort model name for records."""
        return getattr(self, '_agent_model', 'unknown')

    def _log_to_wandb(self, tool_name: str, tool_input: dict, result: dict):
        if not self.wandb_run:
            return

        log_data = {"step": self._step, "tool": tool_name}

        if tool_name == "get_backend_health":
            log_data.update({
                "health/avg_1q_error": result.get("avg_1q_error"),
                "health/avg_2q_error": result.get("avg_2q_error"),
                "health/avg_readout_error": result.get("avg_readout_error"),
                "health/avg_t1_us": result.get("avg_t1_us"),
                "health/avg_t2_us": result.get("avg_t2_us"),
                "health/drift_score": result.get("drift_score"),
                "health/calibration_age_min": result.get("calibration_age_minutes"),
            })

        elif tool_name == "run_circuit":
            log_data.update({
                "circuit/name": result.get("circuit"),
                "circuit/fidelity": result.get("fidelity"),
                "circuit/transpiled_depth": result.get("transpiled_depth"),
                "circuit/shots": result.get("shots"),
            })

        elif tool_name == "diagnose_and_suggest":
            log_data.update({
                "diagnosis/severity": result.get("severity"),
                "diagnosis/drift_score": result.get("drift_score"),
                "diagnosis/avg_2q_error": result.get("avg_2q_error"),
            })

        self.wandb_run.log(log_data)
