"""QuantumGPT ReAct Agent — Thought → Action → Observation loop.

Upgrades the simple while-loop agent with:
  1. Explicit Thought/Action/Observation trace (ReAct pattern)
  2. Fidelity-budget integration — budget status injected each turn
  3. Structured trace logging for ablation analysis
  4. Works with all providers: OpenAI-compatible, Anthropic, or rule-based mock

The key difference from the vanilla loop agent:
  - Each LLM turn is prompted with budget state
  - The system prompt instructs the model to emit Thought: before tool calls
  - All traces are captured in a structured AgentTrace for analysis
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

from backends.base import ShadowBackend
from tools.quantum_tools import TOOL_DEFINITIONS, ToolExecutor
from tools.registry import get_tool_runtime_spec
from agent.budget import FidelityBudget, BudgetDecision
from agent.planners.rule import ReActRulePlanner
from agent.state import AgentState, ArtifactType, ArtifactStatus, InvalidationReason
from agent.trace import AgentTrace, TraceDiagnostics, TraceStep

# Optional instrumented executor
try:
    from data.instrumented import InstrumentedExecutor
    from data.store import DataStore
    HAS_INSTRUMENTED = True
except ImportError:
    HAS_INSTRUMENTED = False

# Optional memory module
try:
    from agent.memory import AgentMemory
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════
# ReAct system prompt
# ═══════════════════════════════════════════════════════

REACT_SYSTEM_PROMPT = """\
You are QuantumGPT, an AI agent that operates quantum computing hardware.
You use the ReAct (Reason + Act) framework: before every action, state your
reasoning as a Thought.

Your capabilities:
- Monitor device health (T1/T2, gate errors, drift)
- Run and transpile quantum circuits
- Apply error mitigation (ZNE)
- Predict fidelity before running circuits
- Run Rabi oscillation experiments for qubit characterization
- Run Ramsey fringe experiments to measure T2* (dephasing time) and detuning
- Run T1 relaxation experiments to measure energy relaxation time
- Diagnose problems and suggest concrete actions

Available experiment tools:
- rabi_experiment / fit_rabi: drive amplitude sweep → pi-pulse calibration
- ramsey_experiment / fit_ramsey: delay sweep with artificial detuning → T2* and frequency offset
- t1_experiment / fit_t1: post-excitation delay sweep → T1 relaxation time

WORKFLOW:
1. Thought: reason about what information you need and why
2. Action: call the appropriate tool
3. Observation: analyze the result
4. Repeat until you have enough information to answer

FIDELITY BUDGET:
You will be given a fidelity budget status before each turn. Use it to:
- Stop when the target is met
- Try mitigation when fidelity is below target
- Wrap up when budget is exhausted

Always be concise and include concrete numbers in your answers.
When you have gathered enough data, provide a comprehensive final answer.
"""

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_TURNS = 20


# ═══════════════════════════════════════════════════════
# Trace data structures
# ═══════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════
# ReAct Rule Planner (mock mode)
# ═══════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════

class ReActAgent:
    """ReAct agent with fidelity-budget integration.

    Supports: OpenAI-compatible API, Anthropic API, rule-based mock.
    """

    def __init__(
        self,
        backend: ShadowBackend,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str = "auto",
        max_turns: int = MAX_TURNS,
        verbose: bool = True,
        use_mock: bool = False,
        target_fidelity: float = 0.85,
        max_tool_calls: int = 20,
        max_seconds: float = 120.0,
        db_path: str | None = None,
        wandb_run=None,
        use_memory: bool = True,
        use_drift_aware: bool = False,
        drift_threshold: float = 0.3,
        safety_policy=None,
    ):
        self.backend = backend
        self.model = model
        self.max_turns = max_turns
        self.verbose = verbose
        self.requested_provider = provider
        self.use_mock = use_mock or (provider == "mock")
        self.client = None
        self.provider = provider
        self.use_memory = use_memory
        self.use_drift_aware = use_drift_aware
        self.drift_threshold = drift_threshold

        self.budget = FidelityBudget(
            target_fidelity=target_fidelity,
            max_tool_calls=max_tool_calls,
            max_seconds=max_seconds,
        )

        # Set up executor
        if HAS_INSTRUMENTED and db_path is not None:
            self._db = DataStore(db_path)
            self.executor = InstrumentedExecutor(backend, db=self._db, wandb_run=wandb_run)
        else:
            self._db = None
            self.executor = ToolExecutor(backend)

        self.safety_policy = safety_policy
        if safety_policy is not None:
            from agent.safety import SafetyAwareExecutor
            self.executor = SafetyAwareExecutor(self.executor, safety_policy)

        # Initialize memory
        if HAS_MEMORY and self._db is not None and use_memory:
            self.memory = AgentMemory(self._db, enabled=True)
        elif HAS_MEMORY and use_memory:
            # Create a temporary DB for memory if no db_path given
            import tempfile, uuid
            from data.store import DataStore as _DS
            _tmp_path = db_path or f"/tmp/qgpt_mem_{uuid.uuid4().hex[:8]}.duckdb"
            self._memory_db = _DS(_tmp_path)
            self.memory = AgentMemory(self._memory_db, enabled=True)
        else:
            self.memory = None

        # Initialize drift monitor
        self.drift_monitor = None
        if use_drift_aware:
            try:
                from agent.drift_aware import DriftMonitor, ReplanningPolicy
                self.drift_monitor = DriftMonitor(
                    backend, drift_threshold=drift_threshold
                )
                self.drift_monitor.initialize()
                self._replan_policy = ReplanningPolicy(strategy="aggressive")
            except ImportError:
                pass

        if not self.use_mock:
            # ── DeepSeek (first-class: OpenAI-compatible API) ──────────
            if provider in ("deepseek", "auto"):
                ds_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
                ds_base = base_url or "https://api.deepseek.com"
                if ds_key:
                    try:
                        from openai import OpenAI
                        self.client = OpenAI(
                            api_key=ds_key,
                            base_url=ds_base,
                        )
                        self.provider = "deepseek"
                        if model == DEFAULT_MODEL:
                            self.model = "deepseek-chat"
                    except ImportError:
                        pass

            # ── OpenAI ─────────────────────────────────────────────────
            if self.client is None and provider in ("openai", "auto"):
                oai_key = api_key or os.environ.get("OPENAI_API_KEY")
                oai_base = base_url or os.environ.get("OPENAI_BASE_URL")
                if oai_key:
                    try:
                        from openai import OpenAI
                        kwargs = {"api_key": oai_key}
                        if oai_base:
                            kwargs["base_url"] = oai_base
                        self.client = OpenAI(**kwargs)
                        self.provider = "openai"
                    except ImportError:
                        pass

            # ── Anthropic ──────────────────────────────────────────────
            if self.client is None and provider in ("anthropic", "auto"):
                anth_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                if anth_key:
                    try:
                        import anthropic
                        self.client = anthropic.Anthropic(api_key=anth_key)
                        self.provider = "anthropic"
                    except ImportError:
                        pass

        if self.client is None:
            self.use_mock = True
            self.provider = "mock"
            self.model = "react-rule-planner-v1"

        self.planner = None
        if self.use_mock:
            base_planner = ReActRulePlanner()
            if self.drift_monitor is not None and hasattr(self, "_replan_policy"):
                try:
                    from agent.drift_aware import DriftAwareRulePlanner
                    self.planner = DriftAwareRulePlanner(
                        base_planner,
                        self.drift_monitor,
                        self._replan_policy,
                    )
                except ImportError:
                    self.planner = base_planner
            else:
                self.planner = base_planner

    def _new_trace(self, user_prompt: str) -> AgentTrace:
        trace = AgentTrace(
            user_prompt=user_prompt,
            model=self.model,
            provider=self.provider,
            backend=self.backend.name,
            state=AgentState(
                user_prompt=user_prompt,
                backend_name=self.backend.name,
            ),
        )
        trace.diagnostics.requested_provider = self.requested_provider
        trace.diagnostics.resolved_provider = self.provider
        trace.diagnostics.model = self.model
        return trace

    def _finalize_trace(self, trace: AgentTrace, reached_max_turns: bool) -> None:
        if reached_max_turns:
            trace.diagnostics.max_turns_exceeded = True
        trace.diagnostics.final_answer_length = len(trace.final_answer or "")
        trace.diagnostics.no_final_answer = not bool((trace.final_answer or "").strip())

    def _known_tool_names(self) -> set[str]:
        names = {td["name"] for td in TOOL_DEFINITIONS}
        if self.memory is not None:
            names.add("retrieve_past_experiments")
        return names

    def _record_tool_diagnostics(
        self,
        trace: AgentTrace,
        tool_name: str,
        tool_input: dict,
        seen_calls: set[tuple[str, str]],
    ) -> None:
        if tool_name not in self._known_tool_names():
            trace.diagnostics.invalid_tool_call_count += 1
            if tool_name not in trace.diagnostics.hallucinated_tool_names:
                trace.diagnostics.hallucinated_tool_names.append(tool_name)

        key = (tool_name, json.dumps(tool_input, sort_keys=True, default=str))
        if key in seen_calls:
            trace.diagnostics.repeated_tool_count += 1
        seen_calls.add(key)

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call, handling memory tools internally."""
        if tool_name == "retrieve_past_experiments" and self.memory is not None:
            results = self.memory.retrieve_similar(
                circuit_name=tool_input.get("circuit_name"),
                backend=tool_input.get("backend"),
                experiment_type=tool_input.get("experiment_type"),
                min_fidelity=tool_input.get("min_fidelity"),
                limit=tool_input.get("limit", 5),
            )
            if not results:
                return json.dumps({"experiments": [], "message": "No matching experiments in memory."})
            return json.dumps({"experiments": results, "count": len(results)}, default=str)
        return self.executor.execute(tool_name, tool_input)

    def _artifact_type_for_tool(self, tool_name: str) -> ArtifactType | None:
        """Resolve state-artifact metadata from the canonical tool registry."""
        spec = get_tool_runtime_spec(tool_name)
        if spec is None or spec.artifact_type is None:
            return None
        return ArtifactType(spec.artifact_type)

    def _hash_payload(self, payload: Any) -> str:
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _current_backend_snapshot_hash(self, state: AgentState) -> str | None:
        if not state.backend_snapshot_id:
            return None
        snapshot = state.artifacts.get(state.backend_snapshot_id)
        if snapshot is None:
            return None
        value = snapshot.metadata.get("snapshot_hash")
        return str(value) if value is not None else None

    def _record_state_after_tool(
        self,
        trace: AgentTrace,
        step: TraceStep,
        tool_name: str,
        tool_input: dict,
        result_str: str,
    ) -> None:
        """Update AgentState and attach a compact state snapshot to the step."""
        if trace.state is None:
            return

        safety_status = self._safety_status_from_result(result_str)
        if safety_status in {"blocked", "dry_run"}:
            trace.state.current_step = step.step_num
            trace.state.safety_status = {
                "last_tool": tool_name,
                "status": safety_status,
            }
            trace.state.observations.append(f"{tool_name} -> safety_{safety_status}")

            # Register a SAFETY_VIOLATION artifact so it shows up in
            # trace exports (visible to physics teachers reviewing the run).
            if safety_status == "blocked":
                try:
                    parsed = json.loads(result_str)
                    reason = parsed.get("reason", "unknown")
                except Exception:
                    reason = "unknown"
                trace.state.artifacts.add_artifact(
                    artifact_type=ArtifactType.SAFETY_VIOLATION,
                    created_at_step=step.step_num,
                    metadata={
                        "tool": tool_name,
                        "input": tool_input,
                        "reason": reason,
                        "status": safety_status,
                    },
                    artifact_id=f"safety-block-{step.step_num}",
                )

            step.state_summary = trace.state.summary_for_prompt()
            return

        state = trace.state
        state.current_step = step.step_num
        runtime_spec = get_tool_runtime_spec(tool_name)
        artifact_type = self._artifact_type_for_tool(tool_name)
        if artifact_type is None:
            step.state_summary = state.summary_for_prompt()
            return

        try:
            parsed_result = json.loads(result_str)
        except Exception:
            parsed_result = {"raw_result": result_str}

        metadata = {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "result_keys": sorted(parsed_result.keys()) if isinstance(parsed_result, dict) else [],
        }
        if runtime_spec is not None:
            metadata["drift_features"] = list(runtime_spec.drift_features)
        metadata["backend_name"] = state.backend_name
        metadata["evidence_snapshot_id"] = state.backend_snapshot_id
        circuit_name = tool_input.get("circuit_name")
        if circuit_name is None and isinstance(parsed_result, dict):
            circuit_name = parsed_result.get("circuit")
        if circuit_name is not None:
            metadata["circuit_name"] = str(circuit_name)
        tool_error = parsed_result.get("error") if isinstance(parsed_result, dict) else None
        if tool_error:
            metadata["error"] = tool_error
            if parsed_result.get("reason"):
                metadata["error_reason"] = parsed_result.get("reason")
        depends_on: list[str] = []
        backend_snapshot_hash: str | None = None

        if artifact_type is ArtifactType.BACKEND_SNAPSHOT:
            snapshot_hash = self._hash_payload(parsed_result)
            metadata["snapshot_hash"] = snapshot_hash
            backend_snapshot_hash = snapshot_hash
        elif state.backend_snapshot_id:
            depends_on = [state.backend_snapshot_id]
            backend_snapshot_hash = self._current_backend_snapshot_hash(state)

        artifact = state.artifacts.add_artifact(
            artifact_type=artifact_type,
            created_at_step=step.step_num,
            depends_on=depends_on,
            backend_snapshot_hash=backend_snapshot_hash,
            payload_ref=f"tool:{tool_name}:step:{step.step_num}",
            metadata=metadata,
        )
        if tool_error:
            artifact.status = ArtifactStatus.FAILED
            artifact.invalidation_reason = InvalidationReason.TOOL_FAILURE
            artifact.invalidation_detail = str(tool_error)
        if artifact_type is ArtifactType.BACKEND_SNAPSHOT:
            state.backend_snapshot_id = artifact.artifact_id
        state.observations.append(f"{tool_name} -> {artifact.artifact_type.value}")
        step.state_summary = state.summary_for_prompt()

    def _record_drift_after_tool(self, trace: AgentTrace, step: TraceStep) -> None:
        """Poll drift monitor and invalidate stale state artifacts if drift appears.

        Also surfaces a one-shot DRIFT ALERT into trace._pending_drift_alert
        so the next LLM turn (in _run_openai/_run_anthropic) can see the
        device parameter shift and replan. Counts each stable→drifting
        transition in trace.diagnostics.drift_alert_count, and sets
        trace.diagnostics.replan_triggered the first time drift is observed.
        """
        if self.drift_monitor is None:
            return

        was_drifting = bool(getattr(self.drift_monitor.state, "is_drifting", False))
        drift_state = self.drift_monitor.step()

        if trace.state is not None:
            state = trace.state
            state.drift_state = {
                "is_drifting": bool(getattr(drift_state, "is_drifting", False)),
                "drift_score": float(getattr(drift_state, "drift_score", 0.0) or 0.0),
                "affected_features": list(getattr(drift_state, "affected_features", []) or []),
                "feature_changes": dict(getattr(drift_state, "feature_changes", {}) or {}),
                "consecutive_drift_checks": int(getattr(drift_state, "consecutive_drift_checks", 0) or 0),
                "invalidated_results": list(getattr(drift_state, "invalidated_results", []) or []),
                "recovery_steps": int(getattr(drift_state, "recovery_steps", 0) or 0),
            }

            if (
                state.drift_state["is_drifting"]
                and not was_drifting
                and state.backend_snapshot_id
            ):
                candidates = [
                    artifact for artifact in state.artifacts.descendants(state.backend_snapshot_id)
                    if artifact.status is ArtifactStatus.VALID
                ]
                invalidated = state.artifacts.invalidate_for_drift(
                    state.backend_snapshot_id,
                    affected_features=state.drift_state["affected_features"],
                    reason=InvalidationReason.BACKEND_DRIFT,
                    detail=f"drift_score={state.drift_state['drift_score']:.3f}",
                    status=ArtifactStatus.STALE,
                )
                invalidated_ids = {artifact.artifact_id for artifact in invalidated}
                preserved = [
                    artifact for artifact in candidates
                    if artifact.artifact_id not in invalidated_ids
                ]
                state.drift_state["invalidated_artifact_ids"] = sorted(invalidated_ids)
                state.drift_state["preserved_artifact_ids"] = sorted(
                    artifact.artifact_id for artifact in preserved
                )
                trace.diagnostics.artifact_invalidation_count += len(invalidated)
                trace.diagnostics.artifact_preservation_count += len(preserved)
                if candidates and not invalidated:
                    trace.diagnostics.drift_without_relevant_artifact_count += 1
                if invalidated:
                    invalidated_types = sorted({a.artifact_type.value for a in invalidated})
                    state.observations.append(
                        "backend drift invalidated " + ", ".join(invalidated_types)
                    )

            step.state_summary = state.summary_for_prompt()

        is_drifting = bool(getattr(drift_state, "is_drifting", False))
        if is_drifting and not was_drifting:
            trace.diagnostics.drift_alert_count += 1
            trace.diagnostics.drift_exposure_count += 1
            for feature in list(getattr(drift_state, "affected_features", []) or []):
                counts = trace.diagnostics.drift_affected_feature_counts
                counts[str(feature)] = counts.get(str(feature), 0) + 1
            trace.diagnostics.replan_triggered = True
            trace.diagnostics.drift_alert_steps.append(int(getattr(step, "step_num", 0)))
            invalidated = list(getattr(drift_state, "invalidated_results", []) or [])
            score = float(getattr(drift_state, "drift_score", 0.0) or 0.0)
            alert_lines = [
                "[DRIFT ALERT] Device parameters have shifted since your last action.",
                f"  drift_score={score:.3f} (threshold={getattr(self.drift_monitor, 'drift_threshold', 0.3)})",
            ]
            if invalidated:
                alert_lines.append(
                    "  Invalidated artifacts: " + ", ".join(str(x) for x in invalidated)
                )
                alert_lines.append(
                    "  ACTION REQUIRED: re-check backend health, retranspile, and re-run the affected circuit before trusting earlier results."
                )
            trace._pending_drift_alert = "\n".join(alert_lines)

    def _safety_status_from_result(self, result_str: str) -> str | None:
        try:
            parsed = json.loads(result_str)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        safety = parsed.get("safety")
        if isinstance(safety, dict):
            status = safety.get("status")
            return str(status) if status is not None else None
        return None

    def _record_safety_diagnostics(self, trace: AgentTrace, result_str: str) -> None:
        status = self._safety_status_from_result(result_str)
        if status == "blocked":
            trace.diagnostics.safety_block_count += 1
        elif status == "dry_run":
            trace.diagnostics.safety_dry_run_count += 1

    def _extract_fidelity(self, result_str: str) -> float | None:
        """Try to extract fidelity from a tool result string."""
        try:
            r = json.loads(result_str)
            # Direct fidelity field
            if "fidelity" in r:
                return float(r["fidelity"])
            # Mitigated fidelity from ZNE
            if "mitigated_fidelity" in r:
                return float(r["mitigated_fidelity"])
            # Predicted fidelity
            if "predicted_fidelity" in r:
                return float(r["predicted_fidelity"])
        except Exception:
            pass
        return None

    def _budget_stop_step(self, step_num: int, thought: str = "") -> TraceStep:
        return TraceStep(
            step_num=step_num,
            thought=thought,
            budget_decision=BudgetDecision.STOP.name,
            timestamp=time.time(),
        )

    def _maybe_stop_for_budget(self, trace: AgentTrace, step_num: int, thought: str = "") -> bool:
        if self.budget.remaining_calls > 0 and self.budget.remaining_seconds > 0:
            return False
        if not trace.final_answer:
            trace.final_answer = "[Budget exhausted before executing additional tools]"
        if not trace.steps or trace.steps[-1].budget_decision != BudgetDecision.STOP.name:
            trace.steps.append(self._budget_stop_step(step_num, thought=thought))
        return True

    def run(self, user_prompt: str, memory_context: str = "") -> AgentTrace:
        """Run the ReAct agent loop.

        Args:
            user_prompt: The user's request.
            memory_context: Optional externally-injected memory context.
                If provided, it overrides the auto-retrieved context from
                self.memory. Useful for demos and testing.
        """
        self.budget.start()

        # Inject memory context — prefer explicit parameter, fallback to self.memory
        if not memory_context and self.memory is not None:
            memory_context = self.memory.get_context_summary(
                user_prompt, self.backend.name
            )
            if self.verbose and memory_context:
                print(f"  🧠 Memory: {self.memory.get_stats_summary()}")

        if self.use_mock:
            trace = self._run_mock(user_prompt, memory_context=memory_context)
        elif self.provider in ("openai", "deepseek"):
            trace = self._run_openai(user_prompt, memory_context=memory_context)
        else:
            trace = self._run_anthropic(user_prompt, memory_context=memory_context)

        trace.budget_summary = self.budget.summary()

        # Log to DB
        if self._db is not None:
            self._db.log_agent_session({
                "ts_start": _now_iso(),
                "ts_end": _now_iso(),
                "model": trace.model,
                "provider": self.provider,
                "backend": self.backend.name,
                "user_prompt": user_prompt,
                "final_answer": trace.final_answer[:2000],
                "num_tool_calls": trace.num_tool_calls,
                "total_tokens": trace.total_tokens,
                "elapsed_seconds": trace.elapsed_seconds,
                "tool_calls": trace.tool_calls_made,
            })

        # Save to experiment memory
        if self.memory is not None:
            record_id = self.memory.record_from_trace(
                trace, user_prompt, self.backend.name
            )
            if self.verbose and record_id:
                print(f"  💾 Saved to memory: {record_id}")

        return trace

    def _parse_memory_context(self, memory_context: str) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for line in memory_context.splitlines():
            if not line.strip().startswith(tuple(str(i) + "." for i in range(1, 10))):
                continue
            entry: dict[str, Any] = {"raw": line.strip()}
            circuit_match = re.search(r"circuit=([a-zA-Z0-9_\-]+)", line)
            fidelity_match = re.search(r"fidelity=([0-9]*\.?[0-9]+)", line)
            outcome_match = re.search(r"outcome=([^\)]+)\)", line)
            if circuit_match:
                entry["circuit_name"] = circuit_match.group(1).lower().replace("-", "_")
            if fidelity_match:
                entry["fidelity"] = float(fidelity_match.group(1))
            if outcome_match:
                entry["outcome"] = outcome_match.group(1)
            entries.append(entry)
        return entries

    def _run_mock(self, user_prompt: str, memory_context: str = "") -> AgentTrace:
        """ReAct loop with rule-based planner."""
        t0 = time.time()
        trace = self._new_trace(user_prompt)
        if memory_context and trace.state is not None:
            trace.state.memory_context = self._parse_memory_context(memory_context)
        history: list[dict] = []
        seen_calls: set[tuple[str, str]] = set()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"QuantumGPT ReAct Agent ({self.model})")
            print(f"Backend: {self.backend.name} ({self.backend.num_qubits}q)")
            print(f"Target fidelity: {self.budget.target_fidelity}")
            print(f"{'='*60}")
            print(f"\nUser: {user_prompt}\n")

        for step_num in range(self.max_turns):
            if self._maybe_stop_for_budget(trace, step_num):
                break

            thought, tool_calls, final_text = self.planner.plan(
                user_prompt, history, self.budget, memory_context=memory_context
            )

            step = TraceStep(step_num=step_num, thought=thought, timestamp=time.time())

            if self.verbose and thought:
                print(f"  💭 Thought: {thought}")

            if not tool_calls:
                step.budget_decision = self.budget.decide().name
                trace.steps.append(step)
                trace.final_answer = final_text or ""
                break

            # Execute tool calls
            for call in tool_calls:
                if self._maybe_stop_for_budget(trace, step_num, thought=thought):
                    break

                tool_name = call["name"]
                tool_input = call["input"]
                self._record_tool_diagnostics(trace, tool_name, tool_input, seen_calls)

                step.action = tool_name
                step.action_input = tool_input
                step.budget_decision = self.budget.decide().name

                if self.verbose:
                    print(f"  🔧 Action: {tool_name}({json.dumps(tool_input, default=str)})")

                result_str = self._execute_tool(tool_name, tool_input)
                step.observation = result_str
                self._record_safety_diagnostics(trace, result_str)
                self._record_state_after_tool(trace, step, tool_name, tool_input, result_str)
                self._record_drift_after_tool(trace, step)

                # Track fidelity
                fidelity = self._extract_fidelity(result_str)
                step.fidelity_observed = fidelity
                self.budget.record_tool_call(tool_name, fidelity)

                if self.verbose:
                    display = result_str[:200] + "..." if len(result_str) > 200 else result_str
                    print(f"  👁 Observation: {display}")
                    if fidelity is not None:
                        print(f"  📊 Fidelity: {fidelity:.4f} (target: {self.budget.target_fidelity})")
                    print(f"  📋 Budget: {self.budget.remaining_calls} calls left, {self.budget.remaining_seconds:.0f}s left\n")

                history.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "result": result_str,
                })

                trace.steps.append(step)
                step = TraceStep(step_num=step_num, timestamp=time.time())

        trace.elapsed_seconds = time.time() - t0
        reached_max_turns = not trace.final_answer
        if reached_max_turns:
            trace.final_answer = "[Agent reached maximum turns]"
        self._finalize_trace(trace, reached_max_turns)

        if self.verbose:
            print(f"\nAgent:\n{trace.final_answer}\n")
            print(f"{'─'*60}")
            print(f"Completed in {trace.elapsed_seconds:.1f}s, {trace.num_tool_calls} tool calls")
            print(f"Best fidelity: {self.budget.best_fidelity}")

        return trace

    def _run_openai(self, user_prompt: str, memory_context: str = "") -> AgentTrace:
        """ReAct loop with OpenAI-compatible API."""
        t0 = time.time()
        trace = self._new_trace(user_prompt)

        # Build tool list (include memory tool if enabled)
        all_tool_defs = list(TOOL_DEFINITIONS)
        if self.memory is not None:
            all_tool_defs.append(AgentMemory.tool_definition())

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": td["name"],
                    "description": td["description"],
                    "parameters": td["input_schema"],
                },
            }
            for td in all_tool_defs
        ]

        sys_content = REACT_SYSTEM_PROMPT
        if memory_context:
            sys_content += "\n\n" + memory_context

        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_prompt},
        ]
        history: list[dict] = []
        seen_calls: set[tuple[str, str]] = set()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"QuantumGPT ReAct Agent ({self.model}) via OpenAI API")
            print(f"Backend: {self.backend.name} ({self.backend.num_qubits}q)")
            print(f"Target fidelity: {self.budget.target_fidelity}")
            print(f"{'='*60}")
            print(f"\nUser: {user_prompt}\n")

        for step_num in range(self.max_turns):
            if self._maybe_stop_for_budget(trace, step_num):
                break

            # Inject budget status
            budget_msg = self.budget.budget_prompt_insert()
            messages.append({"role": "system", "content": budget_msg})
            pending_alert = getattr(trace, "_pending_drift_alert", "")
            if pending_alert:
                messages.append({"role": "system", "content": pending_alert})

            api_t0 = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=2048,
                tools=openai_tools,
                messages=messages,
            )
            api_latency_ms = (time.time() - api_t0) * 1000

            # Remove the ephemeral budget injection (and drift alert if any)
            if pending_alert:
                messages.pop()
                trace._pending_drift_alert = ""
            messages.pop()

            choice = response.choices[0]
            usage = response.usage
            step_prompt_tokens = 0
            step_completion_tokens = 0
            if usage:
                step_prompt_tokens = usage.prompt_tokens or 0
                step_completion_tokens = usage.completion_tokens or 0
                trace.total_tokens += step_prompt_tokens + step_completion_tokens
                trace.prompt_tokens_total += step_prompt_tokens
                trace.completion_tokens_total += step_completion_tokens

            msg = choice.message
            thought_text = (msg.content or "").strip()

            if self.verbose and thought_text:
                print(f"  💭 {thought_text}\n")

            if not msg.tool_calls:
                trace.final_answer = thought_text
                trace.steps.append(TraceStep(
                    step_num=step_num, thought=thought_text,
                    budget_decision=self.budget.decide().name,
                    timestamp=time.time(),
                    prompt_tokens=step_prompt_tokens,
                    completion_tokens=step_completion_tokens,
                    api_latency_ms=api_latency_ms,
                ))
                break

            messages.append(msg.model_dump())

            for tc in msg.tool_calls:
                if self._maybe_stop_for_budget(trace, step_num, thought=thought_text):
                    break

                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    trace.diagnostics.malformed_json_count += 1
                    tool_input = {}
                self._record_tool_diagnostics(trace, tool_name, tool_input, seen_calls)

                step = TraceStep(
                    step_num=step_num, thought=thought_text,
                    action=tool_name, action_input=tool_input,
                    timestamp=time.time(),
                    prompt_tokens=step_prompt_tokens,
                    completion_tokens=step_completion_tokens,
                    api_latency_ms=api_latency_ms,
                )

                if self.verbose:
                    print(f"  🔧 {tool_name}({json.dumps(tool_input, default=str)})")

                result_str = self._execute_tool(tool_name, tool_input)
                step.observation = result_str
                self._record_safety_diagnostics(trace, result_str)
                self._record_state_after_tool(trace, step, tool_name, tool_input, result_str)
                self._record_drift_after_tool(trace, step)

                fidelity = self._extract_fidelity(result_str)
                step.fidelity_observed = fidelity
                self.budget.record_tool_call(tool_name, fidelity)
                step.budget_decision = self.budget.decide().name

                if self.verbose:
                    display = result_str[:200] + "..." if len(result_str) > 200 else result_str
                    print(f"  👁 {display}")
                    if fidelity is not None:
                        print(f"  📊 Fidelity: {fidelity:.4f}")
                    print()

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
                history.append({"tool": tool_name, "input": tool_input, "result": result_str})
                trace.steps.append(step)
                thought_text = ""  # Only first call gets the thought

        trace.elapsed_seconds = time.time() - t0
        reached_max_turns = not trace.final_answer
        if reached_max_turns:
            trace.final_answer = "[Agent reached maximum turns]"
        self._finalize_trace(trace, reached_max_turns)

        if self.verbose:
            print(f"\nAgent:\n{trace.final_answer}\n")
            print(f"{'─'*60}")
            print(f"Completed in {trace.elapsed_seconds:.1f}s, {trace.num_tool_calls} tool calls, "
                  f"{trace.total_tokens} tokens")

        return trace

    def _run_anthropic(self, user_prompt: str, memory_context: str = "") -> AgentTrace:
        """ReAct loop with Anthropic API."""
        import anthropic as anth

        t0 = time.time()
        trace = self._new_trace(user_prompt)

        messages = [{"role": "user", "content": user_prompt}]
        history: list[dict] = []
        seen_calls: set[tuple[str, str]] = set()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"QuantumGPT ReAct Agent ({self.model}) via Anthropic API")
            print(f"Backend: {self.backend.name} ({self.backend.num_qubits}q)")
            print(f"Target fidelity: {self.budget.target_fidelity}")
            print(f"{'='*60}")
            print(f"\nUser: {user_prompt}\n")

        for step_num in range(self.max_turns):
            if self._maybe_stop_for_budget(trace, step_num):
                break

            sys_prompt = REACT_SYSTEM_PROMPT
            if memory_context:
                sys_prompt += "\n\n" + memory_context
            sys_prompt += "\n\n" + self.budget.budget_prompt_insert()
            pending_alert = getattr(trace, "_pending_drift_alert", "")
            if pending_alert:
                sys_prompt += "\n\n" + pending_alert
                trace._pending_drift_alert = ""

            api_t0 = time.time()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=sys_prompt,
                tools=list(TOOL_DEFINITIONS) + ([AgentMemory.tool_definition()] if self.memory else []),
                messages=messages,
            )
            api_latency_ms = (time.time() - api_t0) * 1000
            step_prompt_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
            step_completion_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)
            trace.total_tokens += step_prompt_tokens + step_completion_tokens
            trace.prompt_tokens_total += step_prompt_tokens
            trace.completion_tokens_total += step_completion_tokens

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]
            thought_text = "\n".join(tb.text for tb in text_blocks).strip()

            if self.verbose and thought_text:
                print(f"  💭 {thought_text}\n")

            if not tool_use_blocks:
                trace.final_answer = thought_text
                trace.steps.append(TraceStep(
                    step_num=step_num, thought=thought_text,
                    budget_decision=self.budget.decide().name,
                    timestamp=time.time(),
                    prompt_tokens=step_prompt_tokens,
                    completion_tokens=step_completion_tokens,
                    api_latency_ms=api_latency_ms,
                ))
                break

            # Build assistant message
            assistant_content = []
            for b in response.content:
                if b.type == "text":
                    assistant_content.append({"type": "text", "text": b.text})
                elif b.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use", "id": b.id,
                        "name": b.name, "input": b.input,
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results_content = []
            for block in tool_use_blocks:
                if self._maybe_stop_for_budget(trace, step_num, thought=thought_text):
                    break

                tool_name = block.name
                tool_input = block.input
                self._record_tool_diagnostics(trace, tool_name, tool_input, seen_calls)

                step = TraceStep(
                    step_num=step_num, thought=thought_text,
                    action=tool_name, action_input=tool_input,
                    timestamp=time.time(),
                    prompt_tokens=step_prompt_tokens,
                    completion_tokens=step_completion_tokens,
                    api_latency_ms=api_latency_ms,
                )

                if self.verbose:
                    print(f"  🔧 {tool_name}({json.dumps(tool_input, default=str)})")

                result_str = self._execute_tool(tool_name, tool_input)
                step.observation = result_str
                self._record_safety_diagnostics(trace, result_str)
                self._record_state_after_tool(trace, step, tool_name, tool_input, result_str)
                self._record_drift_after_tool(trace, step)

                fidelity = self._extract_fidelity(result_str)
                step.fidelity_observed = fidelity
                self.budget.record_tool_call(tool_name, fidelity)
                step.budget_decision = self.budget.decide().name

                if self.verbose:
                    display = result_str[:200] + "..." if len(result_str) > 200 else result_str
                    print(f"  👁 {display}")
                    if fidelity is not None:
                        print(f"  📊 Fidelity: {fidelity:.4f}")
                    print()

                tool_results_content.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })
                history.append({"tool": tool_name, "input": tool_input, "result": result_str})
                trace.steps.append(step)
                thought_text = ""

            messages.append({"role": "user", "content": tool_results_content})

        trace.elapsed_seconds = time.time() - t0
        reached_max_turns = not trace.final_answer
        if reached_max_turns:
            trace.final_answer = "[Agent reached maximum turns]"
        self._finalize_trace(trace, reached_max_turns)

        if self.verbose:
            print(f"\nAgent:\n{trace.final_answer}\n")
            print(f"{'─'*60}")
            print(f"Completed in {trace.elapsed_seconds:.1f}s, {trace.num_tool_calls} tool calls, "
                  f"{trace.total_tokens} tokens")

        return trace
