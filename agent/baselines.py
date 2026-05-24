"""Baseline systems for ablation comparison.

Four systems for the paper:
  1. StaticPipeline     — fixed-sequence execution, no LLM, no reasoning
  2. LLMSingleShot     — one LLM call to plan all steps, then execute
  3. ReActNoBudget      — ReAct loop but no fidelity-budget gating
  4. ReActFull          — full ReAct + budget (imported from agent.react)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from backends.base import ShadowBackend
from tools.quantum_tools import ToolExecutor
from agent.react import AgentTrace, TraceStep


# ═══════════════════════════════════════════════════════
# 1. Static Pipeline
# ═══════════════════════════════════════════════════════

class StaticPipeline:
    """Fixed-sequence pipeline: health → run → report.

    No reasoning, no adaptation. Represents the "scripted automation" baseline.
    """

    def __init__(self, backend: ShadowBackend, verbose: bool = True):
        self.backend = backend
        self.executor = ToolExecutor(backend)
        self.verbose = verbose

    def run(self, user_prompt: str) -> AgentTrace:
        t0 = time.time()
        trace = AgentTrace(
            user_prompt=user_prompt,
            model="static-pipeline-v1",
            provider="none",
            backend=self.backend.name,
        )

        # Determine which circuits to run from prompt
        circuits = self._extract_circuits(user_prompt) or ["ghz_5"]
        is_rabi = any(kw in user_prompt.lower() for kw in ["rabi", "pi pulse", "characteriz"])

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Static Pipeline (no reasoning)")
            print(f"Backend: {self.backend.name}")
            print(f"{'='*60}\n")

        # Step 1: always health check
        step = self._exec_tool("get_backend_health", {}, 0)
        trace.steps.append(step)

        if is_rabi:
            # Rabi path: experiment → fit
            step = self._exec_tool("rabi_experiment", {}, 1)
            trace.steps.append(step)
            if step.observation:
                try:
                    r = json.loads(step.observation)
                    step2 = self._exec_tool("fit_rabi", {
                        "amplitudes": r["amplitudes"],
                        "populations": r["populations"],
                    }, 2)
                    trace.steps.append(step2)
                except Exception:
                    pass
        else:
            # Circuit path: run each circuit
            for i, circ in enumerate(circuits):
                step = self._exec_tool("run_circuit", {
                    "circuit_name": circ, "shots": 4096
                }, i + 1)
                trace.steps.append(step)

        trace.elapsed_seconds = time.time() - t0
        trace.final_answer = self._synthesize(trace)

        if self.verbose:
            print(f"\n{trace.final_answer}")
            print(f"\n{'─'*60}")
            print(f"Completed in {trace.elapsed_seconds:.1f}s, {trace.num_tool_calls} tool calls")

        return trace

    def _exec_tool(self, name: str, inp: dict, step_num: int) -> TraceStep:
        step = TraceStep(step_num=step_num, action=name, action_input=inp, timestamp=time.time())
        if self.verbose:
            print(f"  [static] {name}({json.dumps(inp, default=str)[:100]})")
        result_str = self.executor.execute(name, inp)
        step.observation = result_str
        # Extract fidelity
        try:
            r = json.loads(result_str)
            if "fidelity" in r:
                step.fidelity_observed = float(r["fidelity"])
            elif "mitigated_fidelity" in r:
                step.fidelity_observed = float(r["mitigated_fidelity"])
        except Exception:
            pass
        if self.verbose:
            display = result_str[:150] + "..." if len(result_str) > 150 else result_str
            print(f"  <- {display}\n")
        return step

    def _extract_circuits(self, prompt: str) -> list[str]:
        import re
        prompt_lower = prompt.lower()
        found = []
        patterns = [
            (r"ghz[_-]?5", "ghz_5"), (r"qft[_-]?4", "qft_4"),
            (r"bv[_-]?5", "bv_5"), (r"vqe[_-]?4", "vqe_4"),
            (r"qaoa[_-]?4", "qaoa_4"),
        ]
        for pattern, name in patterns:
            if re.search(pattern, prompt_lower):
                found.append(name)
        return found

    def _synthesize(self, trace: AgentTrace) -> str:
        sections = ["## Static Pipeline Report\n"]
        for s in trace.steps:
            if s.observation:
                try:
                    r = json.loads(s.observation)
                    if s.action == "get_backend_health":
                        sections.append(f"Backend: {r.get('backend')} — 1Q err={r.get('avg_1q_error')}, "
                                       f"2Q err={r.get('avg_2q_error')}")
                    elif s.action == "run_circuit":
                        sections.append(f"Circuit {r.get('circuit')}: fidelity={r.get('fidelity')}")
                    elif s.action == "rabi_experiment":
                        sections.append(f"Rabi: π-amp={r.get('pi_amplitude_mhz')} MHz")
                    elif s.action == "fit_rabi":
                        sections.append(f"Rabi fit: π={r.get('pi_amplitude_mhz')} MHz, R²={r.get('r_squared')}")
                except Exception:
                    pass
        return "\n".join(sections)


# ═══════════════════════════════════════════════════════
# 2. LLM Single-Shot
# ═══════════════════════════════════════════════════════

class LLMSingleShot:
    """Single LLM call to plan, then blind execution.

    The LLM sees the user prompt + tool definitions and outputs a plan.
    Then all planned tools are executed sequentially without any
    intermediate reasoning or adaptation.
    """

    def __init__(self, backend: ShadowBackend, verbose: bool = True):
        self.backend = backend
        self.executor = ToolExecutor(backend)
        self.verbose = verbose
        self.planner = _SingleShotPlanner()

    def run(self, user_prompt: str) -> AgentTrace:
        t0 = time.time()
        trace = AgentTrace(
            user_prompt=user_prompt,
            model="llm-single-shot-v1",
            provider="none",
            backend=self.backend.name,
        )

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"LLM Single-Shot (plan once, execute blind)")
            print(f"Backend: {self.backend.name}")
            print(f"{'='*60}\n")

        # Plan: single pass
        plan = self.planner.plan(user_prompt)

        if self.verbose:
            print(f"  Plan: {[p['name'] for p in plan]}\n")

        # Execute all steps blindly (no feedback between steps)
        for i, call in enumerate(plan):
            step = TraceStep(
                step_num=i,
                thought=f"Executing step {i+1} of plan: {call['name']}",
                action=call["name"],
                action_input=call["input"],
                timestamp=time.time(),
            )

            if self.verbose:
                print(f"  [{i+1}/{len(plan)}] {call['name']}({json.dumps(call['input'], default=str)[:100]})")

            result_str = self.executor.execute(call["name"], call["input"])
            step.observation = result_str

            try:
                r = json.loads(result_str)
                if "fidelity" in r:
                    step.fidelity_observed = float(r["fidelity"])
                elif "mitigated_fidelity" in r:
                    step.fidelity_observed = float(r["mitigated_fidelity"])
            except Exception:
                pass

            if self.verbose:
                display = result_str[:150] + "..." if len(result_str) > 150 else result_str
                print(f"  <- {display}\n")

            trace.steps.append(step)

            # Feed Rabi data to fit_rabi if applicable
            if call["name"] == "rabi_experiment" and i + 1 < len(plan) and plan[i+1]["name"] == "fit_rabi":
                try:
                    r = json.loads(result_str)
                    plan[i+1]["input"] = {
                        "amplitudes": r["amplitudes"],
                        "populations": r["populations"],
                    }
                except Exception:
                    pass

        trace.elapsed_seconds = time.time() - t0
        trace.final_answer = self._synthesize(trace)

        if self.verbose:
            print(f"\n{trace.final_answer}")
            print(f"\n{'─'*60}")
            print(f"Completed in {trace.elapsed_seconds:.1f}s, {trace.num_tool_calls} tool calls")

        return trace

    def _synthesize(self, trace: AgentTrace) -> str:
        sections = ["## LLM Single-Shot Report\n"]
        best_fid = None
        for s in trace.steps:
            if s.fidelity_observed is not None:
                if best_fid is None or s.fidelity_observed > best_fid:
                    best_fid = s.fidelity_observed
            if s.observation:
                try:
                    r = json.loads(s.observation)
                    if s.action == "run_circuit":
                        sections.append(f"Circuit {r.get('circuit')}: fidelity={r.get('fidelity')}")
                    elif s.action == "rabi_experiment":
                        sections.append(f"Rabi: π-amp={r.get('pi_amplitude_mhz')} MHz")
                    elif s.action == "fit_rabi":
                        sections.append(f"Fit: π={r.get('pi_amplitude_mhz')} MHz R²={r.get('r_squared')}")
                    elif s.action == "apply_mitigation":
                        sections.append(f"ZNE: {r.get('unmitigated_fidelity')} → {r.get('mitigated_fidelity')}")
                except Exception:
                    pass
        if best_fid:
            sections.append(f"\nBest fidelity: {best_fid}")
        return "\n".join(sections)


class _SingleShotPlanner:
    """Rule-based single-shot planner — one plan, no iteration."""

    def plan(self, prompt: str) -> list[dict]:
        import re
        prompt_lower = prompt.lower()
        steps = []

        # Always health check
        steps.append({"name": "get_backend_health", "input": {}})

        # Rabi path
        if any(kw in prompt_lower for kw in ["rabi", "pi pulse", "characteriz"]):
            params = {}
            freq_m = re.search(r'(\d+\.?\d*)\s*ghz', prompt_lower)
            if freq_m:
                params["qubit_freq_ghz"] = float(freq_m.group(1))
            if "gaussian" in prompt_lower:
                params["pulse_shape"] = "gaussian"
            steps.append({"name": "rabi_experiment", "input": params})
            steps.append({"name": "fit_rabi", "input": {}})  # will be filled in
            return steps

        # Circuit path
        circuits = []
        patterns = [
            (r"ghz[_-]?5", "ghz_5"), (r"qft[_-]?4", "qft_4"),
            (r"bv[_-]?5", "bv_5"), (r"vqe[_-]?4", "vqe_4"),
            (r"qaoa[_-]?4", "qaoa_4"),
        ]
        for pattern, name in patterns:
            if re.search(pattern, prompt_lower):
                circuits.append(name)
        if not circuits and any(kw in prompt_lower for kw in ["run", "circuit", "fidelity"]):
            circuits = ["ghz_5"]

        for circ in circuits:
            steps.append({"name": "run_circuit", "input": {"circuit_name": circ, "shots": 4096}})

        # If user asks for mitigation, add it for the first circuit
        if any(kw in prompt_lower for kw in ["mitigat", "zne", "error correct"]):
            circ = circuits[0] if circuits else "ghz_5"
            steps.append({"name": "apply_mitigation", "input": {"circuit_name": circ}})

        # Diagnose if requested
        if any(kw in prompt_lower for kw in ["diagnos", "suggest", "problem"]):
            steps.append({"name": "diagnose_and_suggest", "input": {}})

        return steps


# ═══════════════════════════════════════════════════════
# 3. ReAct No Budget (uses ReAct agent with budget disabled)
# ═══════════════════════════════════════════════════════

class ReActNoBudget:
    """ReAct agent without fidelity budget — ablation variant.

    Uses the same ReAct loop but with essentially infinite budget,
    so it never triggers MITIGATE or STOP decisions.
    """

    def __init__(self, backend: ShadowBackend, verbose: bool = True, **kwargs):
        from agent.react import ReActAgent
        self._agent = ReActAgent(
            backend,
            provider="mock",
            verbose=verbose,
            target_fidelity=0.0,    # never triggers GOAL_MET
            max_tool_calls=100,     # never triggers STOP
            max_seconds=9999.0,     # never triggers STOP
            **kwargs,
        )
        self._agent.model = "react-no-budget-v1"

    def run(self, user_prompt: str) -> AgentTrace:
        trace = self._agent.run(user_prompt)
        trace.model = "react-no-budget-v1"
        return trace


# ═══════════════════════════════════════════════════════
# Convenience: get all systems
# ═══════════════════════════════════════════════════════

def get_all_systems(backend: ShadowBackend, verbose: bool = False, target_fidelity: float = 0.85):
    """Return a dict of {name: system} for all 4 baselines."""
    from agent.react import ReActAgent

    return {
        "static": StaticPipeline(backend, verbose=verbose),
        "single_shot": LLMSingleShot(backend, verbose=verbose),
        "react_no_budget": ReActNoBudget(backend, verbose=verbose),
        "react_full": ReActAgent(
            backend, provider="mock", verbose=verbose,
            target_fidelity=target_fidelity,
        ),
    }
