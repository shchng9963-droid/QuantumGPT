"""Agent Memory — ExperimentRecord integration for the ReAct agent.

Provides:
  1. Auto-recording: after each agent run, save a structured ExperimentRecord
  2. Retrieval tool: `retrieve_past_experiments` for the agent to query history
  3. Context injection: before planning, inject relevant past experiments
  4. Ablation switch: `use_memory=True/False` for controlled experiments

This is the key differentiator for the paper: the agent learns from its own
history and makes better decisions over time.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from data.experiment_record import (
    ExperimentRecord,
    ExperimentStore,
    ExperimentType,
    ExperimentOutcome,
)
from data.store import DataStore


class AgentMemory:
    """Memory layer that wraps ExperimentStore for agent use.

    Responsibilities:
      - Convert AgentTrace → ExperimentRecord (auto-recording)
      - Provide retrieval methods the agent can call as a tool
      - Generate context summaries for injection into prompts
    """

    def __init__(self, db: DataStore, enabled: bool = True):
        self.db = db
        self.store = db.experiments
        self.enabled = enabled

    # ─── Auto-recording ──────────────────────────────────────────

    def record_from_trace(self, trace, user_prompt: str, backend_name: str) -> Optional[str]:
        """Convert an AgentTrace into an ExperimentRecord and save it.

        Returns the record ID, or None if memory is disabled.
        """
        if not self.enabled:
            return None

        # Determine experiment type from the tools used
        exp_type = self._infer_experiment_type(trace)

        # Extract circuit name from tool calls
        circuit_name = self._extract_circuit_name(trace)

        # Extract fidelity (best observed)
        fidelity = self._extract_best_fidelity(trace)

        # Determine success
        success = fidelity is not None and fidelity >= 0.85
        if fidelity is None:
            outcome = ExperimentOutcome.PARTIAL.value
        elif success:
            outcome = ExperimentOutcome.SUCCESS.value
        else:
            outcome = ExperimentOutcome.PARTIAL.value

        # Build tool_calls list
        tool_calls = []
        for step in trace.steps:
            if step.action:
                tool_calls.append({
                    "tool": step.action,
                    "input": step.action_input,
                    "fidelity": step.fidelity_observed,
                })

        # Build plan (sequence of tool names)
        plan = [tc["tool"] for tc in tool_calls]

        # Extract fits if Rabi experiment
        fits = self._extract_fits(trace)

        # Generate summary
        summary = self._generate_summary(
            exp_type, circuit_name, backend_name, fidelity, plan
        )

        # Extract lessons from final answer
        lessons = self._extract_lessons(trace.final_answer)

        record = ExperimentRecord(
            experiment_type=exp_type,
            backend=backend_name,
            circuit_name=circuit_name,
            problem=user_prompt,
            plan=plan,
            tool_calls=tool_calls,
            fidelity=fidelity,
            success=success,
            outcome=outcome,
            elapsed_seconds=trace.elapsed_seconds,
            model=trace.model,
            total_tokens=trace.total_tokens,
            fits=fits,
            summary=summary,
            lessons=lessons,
            metrics={
                "num_tool_calls": trace.num_tool_calls,
                "elapsed_seconds": trace.elapsed_seconds,
            },
        )

        return self.store.save(record)

    # ─── Retrieval (for agent tool use) ──────────────────────────

    def retrieve_similar(
        self,
        circuit_name: Optional[str] = None,
        backend: Optional[str] = None,
        experiment_type: Optional[str] = None,
        min_fidelity: Optional[float] = None,
        limit: int = 5,
    ) -> list[dict]:
        """Retrieve past experiments matching the query.

        Returns a list of compact summaries suitable for LLM context.
        """
        if not self.enabled:
            return []

        records = self.store.search(
            circuit_name=circuit_name,
            backend=backend,
            experiment_type=experiment_type,
            min_fidelity=min_fidelity,
            limit=limit,
        )

        return [self._record_to_context(r) for r in records]

    def retrieve_for_task(self, user_prompt: str, backend: str) -> list[dict]:
        """Smart retrieval: infer what's relevant from the user prompt.

        Used for automatic context injection before planning.
        """
        if not self.enabled:
            return []

        # Try to extract circuit name from prompt
        circuit_name = self._parse_circuit_from_prompt(user_prompt)

        # Search with available filters
        records = self.store.search(
            circuit_name=circuit_name,
            backend=backend,
            limit=5,
        )

        # If no circuit-specific results, get recent experiments on this backend
        if not records:
            records = self.store.by_backend(backend, limit=3)

        return [self._record_to_context(r) for r in records]

    def get_context_summary(self, user_prompt: str, backend: str) -> str:
        """Generate a text summary of relevant past experiments for prompt injection.

        Returns empty string if no relevant history or memory disabled.
        """
        if not self.enabled:
            return ""

        past = self.retrieve_for_task(user_prompt, backend)
        if not past:
            return ""

        lines = ["[MEMORY] Relevant past experiments:"]
        for i, exp in enumerate(past, 1):
            lines.append(
                f"  {i}. {exp['summary']} "
                f"(fidelity={exp.get('fidelity', '?')}, "
                f"tools={exp.get('num_tools', '?')}, "
                f"outcome={exp.get('outcome', '?')})"
            )
            if exp.get("lessons"):
                for lesson in exp["lessons"][:2]:
                    lines.append(f"     → {lesson}")

        lines.append("")
        return "\n".join(lines)

    def get_stats_summary(self) -> str:
        """Get overall memory stats for display."""
        if not self.enabled:
            return "[Memory disabled]"

        stats = self.store.stats()
        if not stats or stats.get("total_experiments", 0) == 0:
            return "[Memory empty — no past experiments]"

        return (
            f"[Memory: {stats['total_experiments']} experiments, "
            f"{stats['unique_circuits']} circuits, "
            f"avg fidelity={stats.get('avg_fidelity', 0):.3f}, "
            f"success rate={stats.get('successes', 0)}/{stats['total_experiments']}]"
        )

    # ─── Tool definition (for MCP/agent tool list) ───────────────

    @staticmethod
    def tool_definition() -> dict:
        """Return the tool schema for retrieve_past_experiments."""
        return {
            "name": "retrieve_past_experiments",
            "description": (
                "Search the agent's experiment memory for past runs. "
                "Returns summaries of similar experiments including fidelity, "
                "tools used, and lessons learned. Use this to inform decisions "
                "based on historical data."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "circuit_name": {
                        "type": "string",
                        "description": "Filter by circuit name (e.g. 'ghz_5', 'vqe_4')",
                    },
                    "backend": {
                        "type": "string",
                        "description": "Filter by backend name",
                    },
                    "experiment_type": {
                        "type": "string",
                        "description": "Filter by type: circuit_benchmark, rabi_tuneup, drift_diagnosis, error_mitigation",
                    },
                    "min_fidelity": {
                        "type": "number",
                        "description": "Only return experiments with fidelity >= this value",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 5)",
                    },
                },
                "required": [],
            },
        }

    # ─── Internal helpers ────────────────────────────────────────

    def _infer_experiment_type(self, trace) -> str:
        """Infer experiment type from the tools used."""
        tools_used = {s.action for s in trace.steps if s.action}

        if "rabi_experiment" in tools_used or "fit_rabi" in tools_used:
            return ExperimentType.RABI_TUNEUP.value
        if "apply_mitigation" in tools_used:
            return ExperimentType.ERROR_MITIGATION.value
        if "detect_drift" in tools_used or "diagnose_and_suggest" in tools_used:
            return ExperimentType.DRIFT_DIAGNOSIS.value
        if "run_circuit" in tools_used:
            return ExperimentType.CIRCUIT_BENCHMARK.value
        if "get_backend_health" in tools_used and len(tools_used) <= 2:
            return ExperimentType.HEALTH_CHECK.value
        return ExperimentType.MULTI_STEP_TASK.value

    def _extract_circuit_name(self, trace) -> Optional[str]:
        """Extract circuit name from tool calls."""
        for step in trace.steps:
            if step.action == "run_circuit" and step.action_input:
                return step.action_input.get("circuit_name")
            if step.action == "transpile_circuit" and step.action_input:
                return step.action_input.get("circuit_name")
        return None

    def _extract_best_fidelity(self, trace) -> Optional[float]:
        """Get the best fidelity observed during the run."""
        fidelities = [
            s.fidelity_observed for s in trace.steps
            if s.fidelity_observed is not None
        ]
        return max(fidelities) if fidelities else None

    def _extract_fits(self, trace) -> dict:
        """Extract fitting results from Rabi experiments."""
        for step in trace.steps:
            if step.action == "fit_rabi" and step.observation:
                try:
                    r = json.loads(step.observation)
                    return {
                        "rabi_freq_mhz": r.get("rabi_frequency_mhz"),
                        "pi_amp_mhz": r.get("pi_amplitude_mhz"),
                        "r_squared": r.get("r_squared"),
                    }
                except (json.JSONDecodeError, TypeError):
                    pass
        return {}

    def _generate_summary(
        self,
        exp_type: str,
        circuit_name: Optional[str],
        backend: str,
        fidelity: Optional[float],
        plan: list[str],
    ) -> str:
        """Generate a one-line summary for RAG embedding."""
        parts = [exp_type.replace("_", " ").title()]
        if circuit_name:
            parts.append(f"circuit={circuit_name}")
        parts.append(f"on {backend}")
        if fidelity is not None:
            parts.append(f"fidelity={fidelity:.3f}")
        parts.append(f"steps={len(plan)}")
        return " | ".join(parts)

    def _extract_lessons(self, final_answer: str) -> list[str]:
        """Extract actionable lessons from the final answer.

        Simple heuristic: look for sentences with key phrases.
        """
        if not final_answer:
            return []

        lessons = []
        indicators = [
            "suggest", "recommend", "should", "could improve",
            "mitigation", "drift", "recalibrat", "retry",
        ]
        sentences = final_answer.replace("\n", " ").split(".")
        for sent in sentences:
            sent = sent.strip()
            if any(ind in sent.lower() for ind in indicators) and len(sent) > 20:
                lessons.append(sent[:200])
                if len(lessons) >= 3:
                    break
        return lessons

    def _record_to_context(self, record: ExperimentRecord) -> dict:
        """Convert a record to a compact dict for LLM context."""
        return {
            "summary": record.summary,
            "experiment_type": record.experiment_type,
            "circuit_name": record.circuit_name,
            "backend": record.backend,
            "fidelity": record.fidelity,
            "outcome": record.outcome,
            "num_tools": len(record.plan),
            "plan": record.plan[:5],  # truncate for context
            "lessons": record.lessons[:3],
            "fits": record.fits if record.fits else None,
            "elapsed_seconds": record.elapsed_seconds,
            "timestamp": record.timestamp,
        }

    @staticmethod
    def _parse_circuit_from_prompt(prompt: str) -> Optional[str]:
        """Try to extract a circuit name from a natural language prompt."""
        import re
        prompt_lower = prompt.lower()
        patterns = [
            (r"ghz[_\-]?5", "ghz_5"),
            (r"qft[_\-]?4", "qft_4"),
            (r"bv[_\-]?5", "bv_5"),
            (r"vqe[_\-]?4", "vqe_4"),
            (r"qaoa[_\-]?4", "qaoa_4"),
        ]
        for pattern, name in patterns:
            if re.search(pattern, prompt_lower):
                return name
        return None
