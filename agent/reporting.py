"""Human-readable reporting utilities for QuantumGPT agent traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.react import AgentTrace, TraceStep


class TraceReporter:
    """Render an AgentTrace as a report, raw trace bundle, and demo talk track."""

    def __init__(self, trace: AgentTrace):
        self.trace = trace

    def to_markdown(self) -> str:
        lines: list[str] = [
            "# QuantumGPT Agent Report",
            "",
            "## Request",
            self.trace.user_prompt,
            "",
            "## Run summary",
            f"- Backend: {self.trace.backend}",
            f"- Provider: {self.trace.provider}",
            f"- Model: {self.trace.model}",
            f"- Tool calls: {self.trace.num_tool_calls}",
            f"- Elapsed seconds: {self.trace.elapsed_seconds:.3f}",
            "",
            "## Key results",
        ]

        key_results = self._key_results()
        if key_results:
            for key, value in key_results.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- No structured numeric result was found in observations.")

        lines.extend(["", "## Trace steps"])
        for step in self.trace.steps:
            lines.extend(self._format_step(step))

        tool_issues = self._tool_issues()
        if tool_issues:
            lines.extend(["", "## Tool issues"])
            for issue in tool_issues:
                lines.append(f"- {issue}")

        unparsed = self._unparsed_observations()
        if unparsed:
            lines.extend(["", "## Unparsed observations"])
            for item in unparsed:
                lines.append(f"- {item}")

        reliability = self._reliability_warnings()
        if reliability:
            lines.extend(["", "## Reliability warnings"])
            for item in reliability:
                lines.append(f"- {item}")

        state_lines = self._runtime_state_lines()
        if state_lines:
            lines.extend(["", "## Runtime state"])
            lines.extend(state_lines)

        memory_lines = self._memory_context_lines()
        if memory_lines:
            lines.extend(["", "## Memory context"])
            lines.extend(memory_lines)

        safety_lines = self._safety_lines()
        if safety_lines:
            lines.extend(["", "## Safety and dry-run"])
            lines.extend(safety_lines)

        lines.extend([
            "",
            "## Diagnostics",
        ])
        for key, value in self.trace.diagnostics.to_dict().items():
            lines.append(f"- {key}: {value}")

        lines.extend([
            "",
            "## Final answer",
            self.trace.final_answer or "No final answer recorded.",
            "",
        ])
        return "\n".join(lines)

    def to_talk_track(self) -> str:
        key_results = self._key_results()
        pi = key_results.get("pi_amplitude_mhz")
        r2 = key_results.get("r_squared")
        pi_text = f" It extracted a pi-pulse amplitude of {pi} MHz." if pi is not None else ""
        r2_text = f" The fit quality was R²={r2}." if r2 is not None else ""
        tools = ", ".join(dict.fromkeys(s.action for s in self.trace.steps if s.action))
        return "\n".join([
            f"I asked QuantumGPT to run a physics-facing Rabi calibration workflow on {self.trace.backend}.",
            f"The agent produced a traceable sequence of tool calls: {tools or 'no tools'}.",
            f"It first generated Rabi sweep data, then fit the oscillation.{pi_text}{r2_text}",
            "The important point is that the final recommendation is backed by a readable trace, raw JSON, and explicit diagnostics rather than an opaque chat answer.",
        ])

    def write_bundle(self, output_dir: str | Path, prefix: str = "agent") -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "report": out / f"{prefix}_report.md",
            "trace": out / f"{prefix}_trace.json",
            "talk_track": out / f"{prefix}_talk_track.txt",
        }
        paths["report"].write_text(self.to_markdown(), encoding="utf-8")
        paths["trace"].write_text(
            json.dumps(self.trace.to_dict(include_observations=True), indent=2, default=str),
            encoding="utf-8",
        )
        paths["talk_track"].write_text(self.to_talk_track(), encoding="utf-8")
        return paths

    def _format_step(self, step: TraceStep) -> list[str]:
        lines = [f"### Step {step.step_num}"]
        if step.thought:
            lines.append(f"- Thought: {step.thought}")
        if step.action:
            lines.append(f"- Action: `{step.action}`")
        if step.action_input:
            lines.append(f"- Input: `{json.dumps(step.action_input, sort_keys=True, default=str)}`")
        extracted = self._extract_observation_fields(step.observation)
        if extracted:
            lines.append("- Observation highlights:")
            for key, value in extracted.items():
                lines.append(f"  - {key}: {value}")
        elif step.observation:
            lines.append(f"- Observation length: {len(step.observation)} chars")
        return lines

    def _tool_issues(self) -> list[str]:
        issues: list[str] = []
        for step in self.trace.steps:
            parsed = self._parse_observation(step.observation)
            if isinstance(parsed, dict) and parsed.get("error"):
                detail = f"{step.action}: {parsed.get('error')}"
                if parsed.get("reason"):
                    detail += f" ({parsed.get('reason')})"
                safety = parsed.get("safety")
                if isinstance(safety, dict) and safety.get("status"):
                    detail += f" [safety={safety.get('status')}]"
                issues.append(detail)
        return issues

    def _unparsed_observations(self) -> list[str]:
        items: list[str] = []
        for step in self.trace.steps:
            if step.observation and self._parse_observation(step.observation) is None:
                items.append(f"{step.action or 'step ' + str(step.step_num)}: observation was not valid JSON")
        return items

    def _reliability_warnings(self) -> list[str]:
        diagnostics = self.trace.diagnostics.to_dict()
        warning_keys = [
            "no_final_answer",
            "invalid_tool_call_count",
            "malformed_json_count",
            "repeated_tool_count",
            "max_turns_exceeded",
            "safety_block_count",
            "safety_dry_run_count",
        ]
        warnings: list[str] = []
        for key in warning_keys:
            value = diagnostics.get(key)
            if value not in (False, 0, None, [], ""):
                warnings.append(f"{key}: {value}")
        return warnings

    def _runtime_state_lines(self) -> list[str]:
        state = self.trace.state
        if state is None:
            return []
        summary = state.summary_for_prompt()
        lines = [
            f"- Task id: {summary.get('task_id')}",
            f"- Backend snapshot id: {summary.get('backend_snapshot_id')}",
            f"- Current step: {summary.get('current_step')}",
        ]
        artifact_counts = summary.get("artifact_counts") or {}
        if artifact_counts:
            lines.append("- Artifact counts:")
            for key, value in sorted(artifact_counts.items()):
                lines.append(f"  - {key}: {value}")
        artifact_status_counts = summary.get("artifact_status_counts") or {}
        if artifact_status_counts:
            lines.append("- Artifact status counts:")
            for key, value in sorted(artifact_status_counts.items()):
                lines.append(f"  - {key}: {value}")
        budget_state = summary.get("budget_state") or {}
        if budget_state:
            lines.append("- Budget state:")
            for key, value in sorted(budget_state.items()):
                lines.append(f"  - {key}: {value}")
        drift_state = summary.get("drift_state") or {}
        if drift_state:
            lines.append("- Drift state:")
            for key, value in sorted(drift_state.items()):
                lines.append(f"  - {key}: {value}")
        return lines

    def _memory_context_lines(self) -> list[str]:
        state = self.trace.state
        if state is None:
            return []
        memory_context = list(state.memory_context or [])
        if not memory_context:
            return ["- No memory entries were used for this run."]
        lines: list[str] = []
        for idx, entry in enumerate(memory_context, start=1):
            if isinstance(entry, dict):
                summary_parts = [f"{key}={value}" for key, value in entry.items()]
                lines.append(f"- Memory {idx}: " + "; ".join(summary_parts))
            else:
                lines.append(f"- Memory {idx}: {entry}")
        return lines

    def _safety_lines(self) -> list[str]:
        lines: list[str] = []
        state = self.trace.state
        if state is not None and state.safety_status:
            lines.append("- State safety status:")
            for key, value in sorted(state.safety_status.items()):
                lines.append(f"  - {key}: {value}")
        diagnostics = self.trace.diagnostics.to_dict()
        for key in ("safety_block_count", "safety_dry_run_count"):
            lines.append(f"- {key}: {diagnostics.get(key, 0)}")
        return lines

    def _key_results(self) -> dict[str, Any]:
        preferred = [
            "pi_amplitude_mhz",
            "pi_amplitude_ghz",
            "half_pi_amplitude_ghz",
            "r_squared",
            "fit_successful",
            "max_population",
            "fidelity",
            "predicted_fidelity",
            "mitigated_fidelity",
        ]
        results: dict[str, Any] = {}
        for step in self.trace.steps:
            fields = self._extract_observation_fields(step.observation)
            for key in preferred:
                if key in fields:
                    results[key] = fields[key]
        return results

    @staticmethod
    def _parse_observation(observation: str | None) -> Any | None:
        if not observation:
            return {}
        try:
            return json.loads(observation)
        except Exception:
            return None

    @staticmethod
    def _extract_observation_fields(observation: str | None) -> dict[str, Any]:
        parsed = TraceReporter._parse_observation(observation)
        if not isinstance(parsed, dict):
            return {}
        interesting = {}
        for key, value in parsed.items():
            if key in {
                "pi_amplitude_mhz",
                "pi_amplitude_ghz",
                "half_pi_amplitude_ghz",
                "r_squared",
                "fit_successful",
                "max_population",
                "fidelity",
                "predicted_fidelity",
                "mitigated_fidelity",
                "backend",
                "circuit",
                "error",
            }:
                interesting[key] = value
        return interesting
