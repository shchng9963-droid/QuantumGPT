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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from backends.base import ShadowBackend
from tools.quantum_tools import TOOL_DEFINITIONS, ToolExecutor
from agent.budget import FidelityBudget, BudgetDecision
from agent.state import AgentState, ArtifactType, ArtifactStatus, InvalidationReason
from agent.planner import AgentPlanner, ReActRulePlanner

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
- Diagnose problems and suggest concrete actions

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

@dataclass
class TraceStep:
    """One step in the ReAct trace."""
    step_num: int
    thought: str | None = None
    action: str | None = None
    action_input: dict | None = None
    observation: str | None = None
    fidelity_observed: float | None = None
    budget_decision: str | None = None
    state_summary: dict[str, Any] | None = None
    timestamp: float = 0.0

    def to_dict(self, include_observation: bool = False) -> dict:
        data = {k: v for k, v in {
            "step": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation_length": len(self.observation) if self.observation else 0,
            "fidelity_observed": self.fidelity_observed,
            "budget_decision": self.budget_decision,
            "state_summary": self.state_summary,
            "timestamp": round(self.timestamp, 3),
        }.items() if v is not None}
        if include_observation and self.observation is not None:
            data["observation"] = self.observation
        return data


@dataclass
class TraceDiagnostics:
    """Structured reliability diagnostics for one agent run."""
    requested_provider: str = "mock"
    resolved_provider: str = "mock"
    model: str = ""
    final_answer_length: int = 0
    no_final_answer: bool = False
    invalid_tool_call_count: int = 0
    malformed_json_count: int = 0
    repeated_tool_count: int = 0
    max_turns_exceeded: bool = False
    hallucinated_tool_names: list[str] = field(default_factory=list)
    safety_block_count: int = 0
    safety_dry_run_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_provider": self.requested_provider,
            "resolved_provider": self.resolved_provider,
            "model": self.model,
            "final_answer_length": self.final_answer_length,
            "no_final_answer": self.no_final_answer,
            "invalid_tool_call_count": self.invalid_tool_call_count,
            "malformed_json_count": self.malformed_json_count,
            "repeated_tool_count": self.repeated_tool_count,
            "max_turns_exceeded": self.max_turns_exceeded,
            "hallucinated_tool_names": self.hallucinated_tool_names,
            "safety_block_count": self.safety_block_count,
            "safety_dry_run_count": self.safety_dry_run_count,
        }


@dataclass
class AgentTrace:
    """Full trace of a ReAct agent run."""
    user_prompt: str
    model: str
    provider: str
    backend: str
    steps: list[TraceStep] = field(default_factory=list)
    final_answer: str = ""
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    budget_summary: dict = field(default_factory=dict)
    diagnostics: TraceDiagnostics = field(default_factory=TraceDiagnostics)
    state: AgentState | None = None

    @property
    def num_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.action is not None)

    @property
    def tool_calls_made(self) -> list[dict]:
        return [
            {"turn": s.step_num, "tool": s.action, "input": s.action_input}
            for s in self.steps if s.action is not None
        ]

    def to_dict(self, include_observations: bool = False) -> dict[str, Any]:
        """Serialize trace with compact steps and structured diagnostics."""
        data = {
            "user_prompt": self.user_prompt,
            "model": self.model,
            "provider": self.provider,
            "backend": self.backend,
            "final_answer": self.final_answer,
            "total_tokens": self.total_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "budget_summary": self.budget_summary,
            "diagnostics": self.diagnostics.to_dict(),
            "steps": [
                s.to_dict(include_observation=include_observations)
                for s in self.steps
            ],
        }
        if self.state is not None:
            data["state_summary"] = self.state.summary_for_prompt()
            data["artifacts"] = self.state.artifacts.to_dict()
        return data

    def to_agent_result(self):
        """Convert to legacy AgentResult for compatibility."""
        from agent.loop import AgentResult, AgentTurn
        turns = [AgentTurn(role="user", content=self.user_prompt, timestamp=0)]
        for s in self.steps:
            if s.thought:
                turns.append(AgentTurn(role="assistant", content=f"Thought: {s.thought}", timestamp=s.timestamp))
            if s.action:
                turns.append(AgentTurn(
                    role="assistant",
                    content=f"[tool] {s.action}",
                    tool_calls=[{"name": s.action, "input": s.action_input}],
                    timestamp=s.timestamp,
                ))
        turns.append(AgentTurn(role="assistant", content=self.final_answer, timestamp=time.time()))

        return AgentResult(
            final_answer=self.final_answer,
            turns=turns,
            tool_calls_made=self.tool_calls_made,
            total_tokens=self.total_tokens,
            elapsed_seconds=self.elapsed_seconds,
            model=self.model,
        )


# ReActRulePlanner moved to agent/planner.py in P1-4 (see AgentPlanner ABC).

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
            if provider in ("openai", "auto"):
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

            if self.client is None and provider in ("anthropic", "auto"):
                anth_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                if anth_key:
                    try:
                        import anthropic
                        self.client = anthropic.Anthropic(api_key=anth_key)
                        self.provider = "anthropic"
                    except ImportError:
                        pass

            # DeepSeek fallback
            if self.client is None and provider in ("deepseek", "auto"):
                ds_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
                if ds_key:
                    try:
                        from openai import OpenAI
                        self.client = OpenAI(
                            api_key=ds_key,
                            base_url="https://api.deepseek.com",
                        )
                        self.provider = "openai"
                        if model == DEFAULT_MODEL:
                            self.model = "deepseek-chat"
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
        """Map tool names to state artifact types."""
        return {
            "get_backend_health": ArtifactType.BACKEND_SNAPSHOT,
            "get_qubit_properties": ArtifactType.QUBIT_PROPERTIES,
            "get_coupling_map": ArtifactType.COUPLING_MAP,
            "transpile_circuit": ArtifactType.TRANSPILED_CIRCUIT,
            "predict_fidelity": ArtifactType.PREDICTED_FIDELITY,
            "run_circuit": ArtifactType.CIRCUIT_RESULT,
            "apply_mitigation": ArtifactType.MITIGATION_RESULT,
            "detect_drift": ArtifactType.DRIFT_REPORT,
            "rabi_experiment": ArtifactType.LAB_EXPERIMENT_RESULT,
            "fit_rabi": ArtifactType.FIT_RESULT,
        }.get(tool_name)

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
            step.state_summary = trace.state.summary_for_prompt()
            return

        state = trace.state
        state.current_step = step.step_num
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
        """Poll drift monitor and invalidate stale state artifacts if drift appears."""
        if trace.state is None or self.drift_monitor is None:
            return

        drift_state = self.drift_monitor.step()
        state = trace.state
        state.drift_state = {
            "is_drifting": bool(getattr(drift_state, "is_drifting", False)),
            "drift_score": float(getattr(drift_state, "drift_score", 0.0) or 0.0),
            "consecutive_drift_checks": int(getattr(drift_state, "consecutive_drift_checks", 0) or 0),
            "invalidated_results": list(getattr(drift_state, "invalidated_results", []) or []),
            "recovery_steps": int(getattr(drift_state, "recovery_steps", 0) or 0),
        }

        if state.drift_state["is_drifting"] and state.backend_snapshot_id:
            invalidated = state.artifacts.invalidate_dependents(
                state.backend_snapshot_id,
                reason=InvalidationReason.BACKEND_DRIFT,
                detail=f"drift_score={state.drift_state['drift_score']:.3f}",
                status=ArtifactStatus.STALE,
            )
            if invalidated:
                invalidated_types = sorted({a.artifact_type.value for a in invalidated})
                state.observations.append(
                    "backend drift invalidated " + ", ".join(invalidated_types)
                )

        step.state_summary = state.summary_for_prompt()

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

    def run(self, user_prompt: str) -> AgentTrace:
        """Run the ReAct agent loop."""
        self.budget.start()

        # Inject memory context if available
        memory_context = ""
        if self.memory is not None:
            memory_context = self.memory.get_context_summary(
                user_prompt, self.backend.name
            )
            if self.verbose and memory_context:
                print(f"  🧠 Memory: {self.memory.get_stats_summary()}")

        if self.use_mock:
            trace = self._run_mock(user_prompt, memory_context=memory_context)
        elif self.provider == "openai":
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
            # Inject budget status
            budget_msg = self.budget.budget_prompt_insert()
            messages.append({"role": "system", "content": budget_msg})

            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=2048,
                tools=openai_tools,
                messages=messages,
            )

            # Remove the ephemeral budget injection
            messages.pop()

            choice = response.choices[0]
            usage = response.usage
            if usage:
                trace.total_tokens += (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)

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
                ))
                break

            messages.append(msg.model_dump())

            for tc in msg.tool_calls:
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
            sys_prompt = REACT_SYSTEM_PROMPT
            if memory_context:
                sys_prompt += "\n\n" + memory_context
            sys_prompt += "\n\n" + self.budget.budget_prompt_insert()

            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=sys_prompt,
                tools=list(TOOL_DEFINITIONS) + ([AgentMemory.tool_definition()] if self.memory else []),
                messages=messages,
            )
            trace.total_tokens += response.usage.input_tokens + response.usage.output_tokens

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
                tool_name = block.name
                tool_input = block.input
                self._record_tool_diagnostics(trace, tool_name, tool_input, seen_calls)

                step = TraceStep(
                    step_num=step_num, thought=thought_text,
                    action=tool_name, action_input=tool_input,
                    timestamp=time.time(),
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
