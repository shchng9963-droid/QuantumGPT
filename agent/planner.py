"""Agent planner abstractions.

Extracted from agent/react.py in P1-4 to break the monolith and allow
multiple planner backends (rule-based, LLM-driven, drift-aware) to share
a common interface.

The abstract base class ``AgentPlanner`` formalizes the
``plan(user_prompt, history, budget, memory_context) -> (thought, tool_calls, final_text)``
contract that ``ReActAgent`` already depends on via duck typing.

P1-4 scope (this commit):
  - Define ``AgentPlanner`` ABC.
  - Move ``ReActRulePlanner`` here verbatim (now inherits ``AgentPlanner``).
  - Provide a stub ``LLMPlanner`` placeholder.

P1-4.2 (future): extract the LLM-driven planning loop currently embedded
in ``ReActAgent._run_openai`` / ``_run_anthropic`` into ``LLMPlanner.plan``.
"""

from __future__ import annotations

import abc
import json
import re
from typing import Any, Optional

from agent.budget import FidelityBudget, BudgetDecision


# ═══════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════


class AgentPlanner(abc.ABC):
    """Abstract base class for agent planners.

    A planner decides, given the current conversation/tool history and budget
    state, what the agent should do next: emit a thought + one or more tool
    calls, or stop and produce a final answer.
    """

    @abc.abstractmethod
    def plan(
        self,
        user_prompt: str,
        history: list[dict],
        budget: FidelityBudget,
        memory_context: str = "",
    ) -> tuple[Optional[str], list[dict], Optional[str]]:
        """Return ``(thought, tool_calls, final_text)``.

        - If ``tool_calls`` is non-empty, the agent executes those tools and
          ``final_text`` should be ``None``.
        - If ``tool_calls`` is empty, ``final_text`` is the agent's final
          synthesized answer.
        """
        raise NotImplementedError


# ═══════════════════════════════════════════════════════
# ReAct Rule Planner (mock mode)
# ═══════════════════════════════════════════════════════

class ReActRulePlanner(AgentPlanner):
    """Deterministic ReAct planner — generates structured Thought/Action pairs.

    Enhanced over the basic RulePlanner with:
    - Budget awareness
    - Phase 2 tool support (transpile, mitigation, predict, rabi, fit)
    - Structured thoughts
    - Multi-circuit planning
    - Keyword-driven tool selection
    """

    # All known benchmark circuits
    ALL_CIRCUITS = ["ghz_3", "ghz_5", "qft_4", "bv_5", "vqe_4", "qaoa_4"]

    def plan(
        self,
        user_prompt: str,
        history: list[dict],
        budget: FidelityBudget,
        memory_context: str = "",
    ) -> tuple[str | None, list[dict], str | None]:
        """Return (thought, tool_calls, final_text).

        If tool_calls non-empty → execute them; final_text is None.
        If tool_calls empty → final_text is the answer.
        """
        prompt_lower = user_prompt.lower()
        past_tools = [h["tool"] for h in history]
        decision = budget.decide()

        # ─── Intent detection ─────────────────────────────────────
        intents = self._detect_intents(prompt_lower)

        # ─── Budget says stop ─────────────────────────────────────
        if decision == BudgetDecision.STOP:
            return (
                "Budget exhausted. Summarizing findings.",
                [],
                self._synthesize(user_prompt, history, budget),
            )

        # ─── Budget says goal met — check pending work ────────────
        if decision == BudgetDecision.GOAL_MET:
            pending = self._get_pending_actions(prompt_lower, intents, history, budget)
            if not pending:
                return (
                    "Fidelity target reached. Summarizing results.",
                    [],
                    self._synthesize(user_prompt, history, budget),
                )
            # else fall through to continue

        # ─── Phase 1: health check ───────────────────────────────
        if "get_backend_health" not in past_tools:
            thought = "First, I need to check the backend health to understand current device state."
            calls = [{"name": "get_backend_health", "input": {}}]

            # Also get qubit properties if relevant. Lab characterization tasks
            # such as Rabi should proceed to the experiment after the health
            # snapshot instead of detouring into generic qubit-property scans.
            if intents.get("qubit_properties") and not intents.get("rabi"):
                thought += " Also getting detailed qubit properties."
                calls.append({"name": "get_qubit_properties", "input": {"qubits": [0, 1, 2, 3, 4]}})

            # Also get coupling map if relevant
            if intents.get("coupling_map"):
                thought += " Getting the coupling map for topology analysis."
                calls.append({"name": "get_coupling_map", "input": {}})

            return thought, calls, None

        # ─── Phase: coupling map (if not yet done) ────────────────
        if intents.get("coupling_map") and "get_coupling_map" not in past_tools:
            thought = "Getting the coupling map to analyze qubit connectivity."
            return thought, [{"name": "get_coupling_map", "input": {}}], None

        # ─── Phase: qubit properties (if not yet done) ────────────
        if intents.get("qubit_properties") and not intents.get("rabi") and "get_qubit_properties" not in past_tools:
            thought = "Getting detailed qubit properties."
            # Get more qubits for selection tasks
            n_qubits = 20 if "best" in prompt_lower or "select" in prompt_lower else 5
            qubits = list(range(n_qubits))
            return thought, [{"name": "get_qubit_properties", "input": {"qubits": qubits}}], None

        # ─── Phase: Rabi experiment ───────────────────────────────
        if intents.get("rabi"):
            if "rabi_experiment" not in past_tools:
                thought = "Running a Rabi oscillation experiment for qubit characterization."
                params = {}
                freq_match = re.search(r'(\d+\.?\d*)\s*ghz', prompt_lower)
                if freq_match:
                    params["qubit_freq_ghz"] = float(freq_match.group(1))
                dur_match = re.search(r'(\d+)\s*ns', prompt_lower)
                if dur_match:
                    params["pulse_duration_ns"] = float(dur_match.group(1))
                if "gaussian" in prompt_lower:
                    params["pulse_shape"] = "gaussian"
                return thought, [{"name": "rabi_experiment", "input": params}], None

            if "fit_rabi" not in past_tools:
                for h in reversed(history):
                    if h["tool"] == "rabi_experiment" and h.get("result"):
                        try:
                            r = json.loads(h["result"])
                            thought = "Fitting the Rabi data to extract π-pulse amplitude."
                            return thought, [{"name": "fit_rabi", "input": {
                                "amplitudes": r["amplitudes"],
                                "populations": r["populations"],
                            }}], None
                        except Exception:
                            pass

        # ─── Phase: memory-informed preflight ─────────────────────
        low_fidelity_circuit = self._low_fidelity_circuit_from_memory(
            memory_context, prompt_lower, budget.target_fidelity
        )
        if (
            low_fidelity_circuit
            and "get_backend_health" in past_tools
            and "diagnose_and_suggest" not in past_tools
        ):
            thought = (
                f"Memory shows prior low fidelity for {low_fidelity_circuit}; "
                "diagnosing likely causes before repeating the run."
            )
            return thought, [{"name": "diagnose_and_suggest", "input": {"circuit_name": low_fidelity_circuit}}], None

        # ─── Phase: transpile ─────────────────────────────────────
        if intents.get("transpile"):
            circ_name = self._extract_circuit_name(prompt_lower) or "ghz_5"
            if not any(h["tool"] == "transpile_circuit" for h in history):
                thought = f"Transpiling {circ_name} to see the compiled depth and gate count."
                opt_level = 1
                # Check if multiple optimization levels requested
                if "level" in prompt_lower and any(str(i) in prompt_lower for i in range(4)):
                    levels = [i for i in range(4) if str(i) in prompt_lower]
                    opt_level = levels[0] if levels else 1
                return thought, [{"name": "transpile_circuit", "input": {
                    "circuit_name": circ_name, "optimization_level": opt_level
                }}], None
            # If multiple levels requested, try next level
            if "level" in prompt_lower or "optimiz" in prompt_lower:
                done_levels = set()
                for h in history:
                    if h["tool"] == "transpile_circuit":
                        done_levels.add(h.get("input", {}).get("optimization_level", 1))
                for lvl in [0, 1, 2, 3]:
                    if lvl not in done_levels and str(lvl) in prompt_lower:
                        thought = f"Transpiling {circ_name} at optimization level {lvl}."
                        return thought, [{"name": "transpile_circuit", "input": {
                            "circuit_name": circ_name, "optimization_level": lvl
                        }}], None

        # ─── Phase: predict fidelity ──────────────────────────────
        if intents.get("predict"):
            all_circuits = self._get_circuits_to_predict(prompt_lower)
            for circ_name in all_circuits:
                already_predicted = any(
                    h["tool"] == "predict_fidelity" and
                    h.get("input", {}).get("circuit_name") == circ_name
                    for h in history
                )
                if not already_predicted:
                    thought = f"Predicting fidelity for {circ_name} without running it."
                    return thought, [{"name": "predict_fidelity", "input": {"circuit_name": circ_name}}], None

        # ─── Phase: list benchmarks ───────────────────────────────
        if intents.get("list_benchmarks") and "list_benchmarks" not in past_tools:
            thought = "Listing available benchmark circuits."
            return thought, [{"name": "list_benchmarks", "input": {}}], None

        # ─── Phase: run circuit(s) ────────────────────────────────
        all_circuits = self._get_circuits_to_run(prompt_lower, history)
        for circ_name in all_circuits:
            failed_attempts = self._failed_tool_count(history, "run_circuit", circuit_name=circ_name)
            already_succeeded = any(
                h["tool"] == "run_circuit"
                and h.get("input", {}).get("circuit_name") == circ_name
                and not self._result_has_error(h.get("result"))
                for h in history
            )
            if not already_succeeded:
                if failed_attempts >= 2:
                    if not any(h["tool"] == "diagnose_and_suggest" for h in history):
                        thought = f"{circ_name} failed repeatedly. Diagnosing instead of looping on the same tool."
                        return thought, [{"name": "diagnose_and_suggest", "input": {"circuit_name": circ_name}}], None
                    continue
                retry_note = "Retrying" if failed_attempts else "Running"
                thought = f"{retry_note} {circ_name} circuit to measure fidelity."
                return thought, [{"name": "run_circuit", "input": {"circuit_name": circ_name, "shots": 4096}}], None

        # ─── Phase: mitigation ────────────────────────────────────
        if intents.get("mitigate") and not budget.mitigation_attempted:
            circ_name = self._last_circuit(history) or "ghz_5"
            thought = f"Applying ZNE error mitigation to {circ_name}."
            budget.mitigation_attempted = True
            return thought, [{"name": "apply_mitigation", "input": {"circuit_name": circ_name, "shots": 4096}}], None

        # Budget-driven mitigation (fidelity below target)
        if decision == BudgetDecision.MITIGATE and not budget.mitigation_attempted:
            circ_name = self._last_circuit(history) or "ghz_5"
            thought = f"Fidelity below target. Applying ZNE error mitigation to {circ_name}."
            budget.mitigation_attempted = True
            return thought, [{"name": "apply_mitigation", "input": {"circuit_name": circ_name, "shots": 4096}}], None

        # ─── Phase: diagnose ──────────────────────────────────────
        if intents.get("diagnose") and "diagnose_and_suggest" not in past_tools:
            thought = "Diagnosing potential issues with the backend."
            inp = {}
            for h in reversed(history):
                if h["tool"] == "run_circuit" and h.get("result"):
                    try:
                        r = json.loads(h["result"])
                        inp["fidelity"] = r.get("fidelity")
                        inp["circuit_name"] = r.get("circuit")
                    except Exception:
                        pass
                    break
            return thought, [{"name": "diagnose_and_suggest", "input": inp}], None

        # ─── Phase: drift check ───────────────────────────────────
        if intents.get("drift") and "detect_drift" not in past_tools:
            thought = "Checking for hardware drift."
            return thought, [{"name": "detect_drift", "input": {}}], None

        # ─── Final synthesis ──────────────────────────────────────
        return (
            "I have gathered enough information. Preparing the final report.",
            [],
            self._synthesize(user_prompt, history, budget),
        )

    def _detect_intents(self, prompt_lower: str) -> dict[str, bool]:
        """Detect user intents from the prompt."""
        return {
            "coupling_map": any(kw in prompt_lower for kw in [
                "coupling", "topology", "connected", "degree", "coupling map"
            ]),
            "qubit_properties": any(kw in prompt_lower for kw in [
                "qubit", "t1", "t2", "best", "worst", "select", "properties"
            ]),
            "rabi": any(kw in prompt_lower for kw in [
                "rabi", "pi pulse", "pi-pulse", "characteriz"
            ]) or ("pulse" in prompt_lower and "pi" in prompt_lower),
            "transpile": any(kw in prompt_lower for kw in [
                "transpil", "compile", "optimiz"
            ]) and "level" in prompt_lower or "transpil" in prompt_lower,
            "predict": any(kw in prompt_lower for kw in [
                "predict", "estimat", "error budget", "break down", "error source",
                "breakdown", "impact", "analyze the impact", "assess"
            ]),
            "mitigate": any(kw in prompt_lower for kw in [
                "mitigat", "zne", "error correct", "zero-noise", "zero noise"
            ]),
            "diagnose": any(kw in prompt_lower for kw in [
                "diagnose", "wrong", "suggest", "problem", "issue", "cause",
                "what happened", "failure", "recommend", "recovery", "recalib"
            ]),
            "drift": any(kw in prompt_lower for kw in [
                "drift", "degrad", "monitor", "stable", "drifting", "detect",
                "spike", "sudden"
            ]),
            "list_benchmarks": any(kw in prompt_lower for kw in [
                "list", "available", "all circuit", "all benchmark", "sweep"
            ]),
            "compare": any(kw in prompt_lower for kw in [
                "compare", "rank", "which", "best", "worst"
            ]),
            "all_circuits": any(kw in prompt_lower for kw in [
                "all circuit", "all available", "all benchmark", "every circuit",
                "each circuit", "sweep"
            ]),
        }

    def _get_pending_actions(self, prompt_lower, intents, history, budget):
        """Check if there are still pending actions after goal is met."""
        pending = []
        past_tools = [h["tool"] for h in history]

        # Unrun circuits
        all_circuits = self._get_circuits_to_run(prompt_lower, history)
        for c in all_circuits:
            if not any(h["tool"] == "run_circuit" and h.get("input", {}).get("circuit_name") == c for h in history):
                pending.append(f"run_{c}")

        # Pending mitigation
        if intents.get("mitigate") and not budget.mitigation_attempted:
            pending.append("mitigate")

        # Pending diagnosis
        if intents.get("diagnose") and "diagnose_and_suggest" not in past_tools:
            pending.append("diagnose")

        # Pending prediction
        if intents.get("predict") and "predict_fidelity" not in past_tools:
            pending.append("predict")

        # Pending coupling map
        if intents.get("coupling_map") and "get_coupling_map" not in past_tools:
            pending.append("coupling_map")

        # Pending transpile
        if intents.get("transpile") and "transpile_circuit" not in past_tools:
            pending.append("transpile")

        # Pending rabi
        if intents.get("rabi") and "rabi_experiment" not in past_tools:
            pending.append("rabi")

        return pending

    def _low_fidelity_circuit_from_memory(
        self,
        memory_context: str,
        prompt_lower: str,
        target_fidelity: float,
    ) -> str | None:
        if not memory_context:
            return None
        requested = self._extract_circuit_name(prompt_lower)
        for line in memory_context.splitlines():
            circuit_match = re.search(r"circuit=([a-z0-9_\-]+)", line, flags=re.IGNORECASE)
            fidelity_match = re.search(r"fidelity=([0-9]*\.?[0-9]+)", line, flags=re.IGNORECASE)
            if not circuit_match or not fidelity_match:
                continue
            circuit = circuit_match.group(1).lower().replace("-", "_")
            if requested and circuit != requested:
                continue
            try:
                fidelity = float(fidelity_match.group(1))
            except ValueError:
                continue
            if fidelity < target_fidelity:
                return circuit
        return None

    def _result_has_error(self, result: str | None) -> bool:
        if not result:
            return False
        try:
            parsed = json.loads(result)
        except Exception:
            return False
        return isinstance(parsed, dict) and bool(parsed.get("error"))

    def _failed_tool_count(
        self,
        history: list[dict],
        tool_name: str,
        circuit_name: str | None = None,
    ) -> int:
        count = 0
        for h in history:
            if h.get("tool") != tool_name:
                continue
            if circuit_name is not None and h.get("input", {}).get("circuit_name") != circuit_name:
                continue
            if self._result_has_error(h.get("result")):
                count += 1
        return count

    def _get_circuits_to_run(self, prompt_lower: str, history: list[dict]) -> list[str]:
        """Determine which circuits to run based on prompt."""
        # Check for "all circuits" intent
        if any(kw in prompt_lower for kw in [
            "all circuit", "all available", "all benchmark", "every circuit",
            "each circuit", "sweep"
        ]):
            # If we already listed benchmarks, use that list
            for h in history:
                if h["tool"] == "list_benchmarks" and h.get("result"):
                    try:
                        r = json.loads(h["result"])
                        circuits = []
                        for key in ["hand_written", "circuits"]:
                            if key in r:
                                if isinstance(r[key], dict):
                                    circuits.extend(r[key].keys())
                                elif isinstance(r[key], list):
                                    circuits.extend(r[key])
                        if circuits:
                            return circuits
                    except Exception:
                        pass
            # Fallback: all known circuits
            return list(self.ALL_CIRCUITS)

        # Extract explicitly named circuits
        found = self._extract_all_circuits(prompt_lower)

        # A lab characterization request such as "run a Rabi experiment" is not
        # a circuit benchmark request. For physics-facing demos, do not append a
        # default GHZ run unless the user explicitly names a circuit.
        is_lab_characterization = any(kw in prompt_lower for kw in [
            "rabi", "ramsey", "t1", "pi pulse", "pi-pulse", "characteriz"
        ])
        if is_lab_characterization and not found:
            return []

        # If prompt mentions running/fidelity but no specific circuit, default to ghz_5
        if not found and any(kw in prompt_lower for kw in ["run", "circuit", "fidelity", "execute"]):
            found = ["ghz_5"]

        return found

    def _get_circuits_to_predict(self, prompt_lower: str) -> list[str]:
        """Determine which circuits to predict fidelity for."""
        if any(kw in prompt_lower for kw in ["all", "each", "every"]):
            return list(self.ALL_CIRCUITS)

        found = self._extract_all_circuits(prompt_lower)
        if not found:
            found = ["ghz_5"]
        return found

    def _extract_circuit_name(self, prompt_lower: str) -> str | None:
        """Extract first matching circuit name from prompt."""
        for pattern, name in self._circuit_patterns():
            if re.search(pattern, prompt_lower):
                return name
        return None

    def _extract_all_circuits(self, prompt_lower: str) -> list[str]:
        """Extract all matching circuit names from prompt."""
        found = []
        for pattern, name in self._circuit_patterns():
            if re.search(pattern, prompt_lower):
                found.append(name)
        return found

    @staticmethod
    def _circuit_patterns():
        return [
            (r"ghz[_\-]?3", "ghz_3"), (r"ghz[_\-]?5", "ghz_5"),
            (r"qft[_\-]?4", "qft_4"),
            (r"bv[_\-]?5|bernstein.vazirani", "bv_5"),
            (r"vqe[_\-]?4", "vqe_4"),
            (r"qaoa[_\-]?4|maxcut", "qaoa_4"),
        ]

    def _last_circuit(self, history: list[dict]) -> str | None:
        for h in reversed(history):
            if h["tool"] == "run_circuit" and h.get("input"):
                return h["input"].get("circuit_name")
        return None

    def _synthesize(self, user_prompt: str, history: list[dict], budget: FidelityBudget) -> str:
        sections = ["## QuantumGPT ReAct Analysis Report\n"]
        for h in history:
            if not h.get("result"):
                continue
            try:
                r = json.loads(h["result"])
            except Exception:
                continue

            if r.get("error"):
                sections.append(f"**Tool issue: {h['tool']}**")
                sections.append(f"  error: {r.get('error')}")
                if r.get("reason"):
                    sections.append(f"  reason: {r.get('reason')}")
                if isinstance(r.get("safety"), dict):
                    sections.append(f"  safety_status: {r['safety'].get('status')}")
                sections.append("")
                continue

            if h["tool"] == "get_backend_health":
                sections.append(f"**Backend: {r.get('backend', '?')}** ({r.get('num_qubits')} qubits)")
                sections.append(f"  T1={r.get('avg_t1_us')}μs  T2={r.get('avg_t2_us')}μs")
                sections.append(f"  1Q err={r.get('avg_1q_error')}  2Q err={r.get('avg_2q_error')}  readout={r.get('avg_readout_error')}")
                sections.append(f"  drift={r.get('drift_score')}\n")

            elif h["tool"] == "get_qubit_properties":
                sections.append("**Qubit Properties:**")
                for q in r.get("qubits", []):
                    sections.append(f"  Q{q['qubit']}: T1={q['t1_us']}μs T2={q['t2_us']}μs readout_err={q['readout_error']}")
                sections.append("")

            elif h["tool"] == "get_coupling_map":
                sections.append("**Coupling Map:**")
                edges = r.get("edges", [])
                sections.append(f"  {r.get('num_qubits', '?')} qubits, {len(edges)} edges")
                # Find most connected qubit
                from collections import Counter
                degree = Counter()
                for e in edges:
                    degree[e[0]] += 1
                    degree[e[1]] += 1
                if degree:
                    top = degree.most_common(3)
                    sections.append(f"  Most connected: Q{top[0][0]} (degree {top[0][1]})")
                sections.append("")

            elif h["tool"] == "run_circuit":
                sections.append(f"**Circuit: {r.get('circuit')}**")
                sections.append(f"  Fidelity: {r.get('fidelity')}  depth: {r.get('transpiled_depth')}  shots: {r.get('shots')}")
                tc = r.get("top_counts", {})
                if tc:
                    top3 = sorted(tc.items(), key=lambda x: -x[1])[:3]
                    sections.append(f"  Top counts: {dict(top3)}")
                sections.append("")

            elif h["tool"] == "transpile_circuit":
                sections.append(f"**Transpile: {r.get('circuit')}**")
                sections.append(f"  Depth: {r.get('original_depth')} → {r.get('transpiled_depth')}  2Q gates: {r.get('two_qubit_gates')}")
                sections.append("")

            elif h["tool"] == "apply_mitigation":
                sections.append(f"**Error Mitigation (ZNE): {r.get('circuit', '?')}**")
                sections.append(f"  Unmitigated: {r.get('unmitigated_fidelity')}  Mitigated: {r.get('mitigated_fidelity')}  Δ: {r.get('improvement')}")
                sections.append("")

            elif h["tool"] == "predict_fidelity":
                sections.append(f"**Predicted Fidelity: {r.get('circuit')}**")
                sections.append(f"  F_predicted={r.get('predicted_fidelity')} ({r.get('confidence')})")
                b = r.get("breakdown", {})
                sections.append(f"  1Q={b.get('f_1q_gates')} 2Q={b.get('f_2q_gates')} readout={b.get('f_readout')}")
                sections.append("")

            elif h["tool"] == "rabi_experiment":
                sections.append(f"**Rabi Experiment** ({r.get('pulse_shape')} pulse, {r.get('pulse_duration_ns')}ns)")
                sections.append(f"  Qubit freq: {r.get('qubit_freq_ghz')} GHz")
                sections.append(f"  π-pulse amplitude: {r.get('pi_amplitude_mhz')} MHz")
                sections.append(f"  Max P(|1⟩): {r.get('max_population')}")
                sections.append("")

            elif h["tool"] == "fit_rabi":
                sections.append(f"**Rabi Fit Result**")
                sections.append(f"  π-amplitude: {r.get('pi_amplitude_mhz')} MHz ± {r.get('pi_uncertainty_ghz', 'N/A')} GHz")
                sections.append(f"  π/2-amplitude: {r.get('half_pi_amplitude_ghz')} GHz")
                sections.append(f"  R²: {r.get('r_squared')}")
                sections.append("")

            elif h["tool"] == "diagnose_and_suggest":
                sections.append(f"**Diagnosis:** severity={r.get('severity')}")
                for s in r.get("suggestions", []):
                    sections.append(f"  • {s}")
                sections.append("")

            elif h["tool"] == "list_benchmarks":
                hw = r.get("hand_written", {})
                mqt = r.get("mqtbench", {})
                if isinstance(hw, dict):
                    sections.append(f"**Circuits:** {len(hw)} hand-written + {len(mqt)} MQTBench\n")
                else:
                    sections.append(f"**Circuits:** {len(r.get('circuits', []))} available\n")

            elif h["tool"] == "detect_drift":
                sections.append(f"**Drift Check:**")
                sections.append(f"  Drift detected: {r.get('drift_detected')}")
                sections.append(f"  Drift score: {r.get('drift_score')}")
                sections.append("")

        # Budget summary
        bs = budget.summary()
        sections.append(f"---\n**Budget:** {bs['tool_calls_used']} calls, {bs['elapsed_seconds']}s elapsed")
        sections.append(f"**Best fidelity:** {bs['best_fidelity']}")
        sections.append(f"**Target:** {bs['target_fidelity']}  **Status:** {bs['decision']}")

        return "\n".join(sections)


# ═══════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════
# LLM-driven planner (placeholder for P1-4.2)
# ═══════════════════════════════════════════════════════


class LLMPlanner(AgentPlanner):
    """LLM-driven planner placeholder.

    P1-4.2 will lift the bodies of ``ReActAgent._run_openai`` and
    ``ReActAgent._run_anthropic`` into this class so that the agent runtime
    no longer cares which provider produced the next tool call. Until then
    this class is a constructor-only stub so callers can wire it up in
    advance.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.client = client
        self.api_key = api_key
        self.base_url = base_url

    def plan(
        self,
        user_prompt: str,
        history: list[dict],
        budget: FidelityBudget,
        memory_context: str = "",
    ) -> tuple[Optional[str], list[dict], Optional[str]]:
        raise NotImplementedError(
            "LLMPlanner.plan is a P1-4.2 stub. The active LLM planning loops "
            "still live in agent.react.ReActAgent._run_openai / _run_anthropic."
        )
