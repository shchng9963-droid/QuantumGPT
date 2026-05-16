"""QuantumGPT Agent — minimal while-loop agent with tool calling.

Two modes:
  1. Anthropic API mode (when valid API key available)
  2. Rule-based mock mode (for offline testing / no API key)

The simplest possible agent: a while loop that sends messages to an LLM,
receives tool_use blocks, executes them, and feeds results back until
the model produces a final text response.

Usage:
    from agent.loop import QuantumAgent
    from backends.fake_adapter import FakeBackendAdapter

    backend = FakeBackendAdapter("FakeBrisbane")
    agent = QuantumAgent(backend)
    result = agent.run("Check the backend health and run a GHZ-5 circuit")
    print(result.final_answer)
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from backends.base import ShadowBackend
from tools.quantum_tools import TOOL_DEFINITIONS, ToolExecutor

# Optional instrumented executor
try:
    from data.instrumented import InstrumentedExecutor
    from data.store import DataStore
    HAS_INSTRUMENTED = True
except ImportError:
    HAS_INSTRUMENTED = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SYSTEM_PROMPT = """\
You are QuantumGPT, an AI agent that operates quantum computing hardware.

You have access to a shadow quantum backend (a realistic simulator that mimics
real IBM quantum devices, including noise, drift, and calibration data).

Your job is to:
1. Monitor device health (T1/T2, gate errors, drift)
2. Run quantum circuits and evaluate fidelity
3. Diagnose problems and suggest mitigations
4. Make decisions about when to run, when to wait, and what to fix

Always start by checking the backend health before running circuits.
When reporting results, include concrete numbers (fidelity, error rates, etc.).
Be concise and actionable in your suggestions.
"""

# Default model — can be overridden
DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_TURNS = 15


@dataclass
class AgentTurn:
    """A single turn in the agent conversation."""
    role: str  # "user", "assistant", "tool_result"
    content: Any
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class AgentResult:
    """Result of an agent run."""
    final_answer: str
    turns: list[AgentTurn]
    tool_calls_made: list[dict]
    total_tokens: int
    elapsed_seconds: float
    model: str


# ═══════════════════════════════════════════════════════
# Rule-based planner (mock LLM for offline testing)
# ═══════════════════════════════════════════════════════

class RulePlanner:
    """Deterministic planner that mimics LLM tool-calling behavior.

    Parses the user prompt for keywords and generates a sequence of
    tool calls, then synthesizes a final answer from the results.
    """

    def plan(self, user_prompt: str, history: list[dict]) -> tuple[list[dict], Optional[str]]:
        """Given a prompt + past tool results, return (tool_calls, final_text).

        If tool_calls is non-empty, final_text is None (need more turns).
        If tool_calls is empty, final_text is the synthesized answer.
        """
        prompt_lower = user_prompt.lower()
        past_tools = [h["tool"] for h in history]

        # Phase 1: always start with health check if not done
        if "get_backend_health" not in past_tools:
            calls = [{"name": "get_backend_health", "input": {}}]
            # Also check qubit properties if user mentions "qubit" or "T1"
            if any(kw in prompt_lower for kw in ["qubit", "t1", "t2", "best", "worst"]):
                calls.append({"name": "get_qubit_properties", "input": {"qubits": [0, 1, 2, 3, 4]}})
            return calls, None

        # Phase 2: list benchmarks if user asks to list/compare
        if any(kw in prompt_lower for kw in ["list", "available", "compare"]):
            if "list_benchmarks" not in past_tools:
                return [{"name": "list_benchmarks", "input": {}}], None

        # Phase 3: run circuits
        circuits_to_run = []
        # Detect circuit names in user prompt
        circuit_patterns = [
            (r"ghz[_-]?5", "ghz_5"),
            (r"qft[_-]?4", "qft_4"),
            (r"bv[_-]?5", "bv_5"),
            (r"vqe[_-]?4", "vqe_4"),
            (r"qaoa[_-]?4", "qaoa_4"),
            (r"dj[_-]?5", "DJ-5"),
            (r"graphstate[_-]?5", "GraphState-5"),
            (r"qftent[_-]?5", "QFTent-5"),
            (r"vqe.su2[_-]?4", "VQE_SU2-4"),
        ]
        for pattern, name in circuit_patterns:
            if re.search(pattern, prompt_lower):
                circuits_to_run.append(name)

        # Default: if user says "run" or "benchmark" but no specific circuit, run ghz_5
        if not circuits_to_run and any(kw in prompt_lower for kw in ["run", "benchmark", "circuit", "fidelity"]):
            circuits_to_run = ["ghz_5"]

        # Run circuits we haven't run yet
        for circ in circuits_to_run:
            if not any(h["tool"] == "run_circuit" and h.get("input", {}).get("circuit_name") == circ
                       for h in history):
                return [{"name": "run_circuit", "input": {"circuit_name": circ, "shots": 4096}}], None

        # Phase 4: diagnose if user asks for diagnosis or the backend is degraded
        if any(kw in prompt_lower for kw in ["diagnose", "wrong", "poorly", "suggest", "problem", "issue"]):
            if "diagnose_and_suggest" not in past_tools:
                # Get fidelity from the most recent run
                fidelity = None
                circuit_name = "unknown"
                for h in reversed(history):
                    if h["tool"] == "run_circuit" and h.get("result"):
                        try:
                            r = json.loads(h["result"])
                            fidelity = r.get("fidelity")
                            circuit_name = r.get("circuit", "unknown")
                        except:
                            pass
                        break
                inp = {}
                if fidelity is not None:
                    inp["fidelity"] = fidelity
                    inp["circuit_name"] = circuit_name
                return [{"name": "diagnose_and_suggest", "input": inp}], None

        # Phase 5: synthesize final answer from all collected results
        return [], self._synthesize(user_prompt, history)

    def _synthesize(self, user_prompt: str, history: list[dict]) -> str:
        """Build a final answer from collected tool results."""
        sections = []
        sections.append("## QuantumGPT Analysis Report\n")

        for h in history:
            if h["tool"] == "get_backend_health" and h.get("result"):
                try:
                    r = json.loads(h["result"])
                    sections.append(f"**Backend Health: {r.get('backend', 'Unknown')}**")
                    sections.append(f"- Qubits: {r.get('num_qubits')}")
                    sections.append(f"- Avg T1: {r.get('avg_t1_us')} μs | T2: {r.get('avg_t2_us')} μs")
                    sections.append(f"- Avg 1Q error: {r.get('avg_1q_error')} | 2Q error: {r.get('avg_2q_error')}")
                    sections.append(f"- Readout error: {r.get('avg_readout_error')}")
                    sections.append(f"- Drift score: {r.get('drift_score')}")
                    sections.append("")
                except:
                    pass

            elif h["tool"] == "get_qubit_properties" and h.get("result"):
                try:
                    r = json.loads(h["result"])
                    sections.append("**Qubit Properties (qubits 0-4):**")
                    best_t1 = (-1, 0)
                    for q in r.get("qubits", []):
                        sections.append(
                            f"  Q{q['qubit']}: T1={q['t1_us']}μs, T2={q['t2_us']}μs, "
                            f"readout_err={q['readout_error']}"
                        )
                        if q['t1_us'] > best_t1[1]:
                            best_t1 = (q['qubit'], q['t1_us'])
                    sections.append(f"  → Best T1: qubit {best_t1[0]} ({best_t1[1]} μs)")
                    sections.append("")
                except:
                    pass

            elif h["tool"] == "run_circuit" and h.get("result"):
                try:
                    r = json.loads(h["result"])
                    sections.append(f"**Circuit: {r.get('circuit')}**")
                    sections.append(f"- Fidelity: {r.get('fidelity')}")
                    sections.append(f"- Transpiled depth: {r.get('transpiled_depth')}")
                    sections.append(f"- Top counts: {r.get('top_counts')}")
                    sections.append("")
                except:
                    pass

            elif h["tool"] == "diagnose_and_suggest" and h.get("result"):
                try:
                    r = json.loads(h["result"])
                    sections.append(f"**Diagnosis: severity={r.get('severity')}**")
                    for s in r.get("suggestions", []):
                        sections.append(f"  • {s}")
                    sections.append("")
                except:
                    pass

            elif h["tool"] == "list_benchmarks" and h.get("result"):
                try:
                    r = json.loads(h["result"])
                    hw = r.get("hand_written", {})
                    mqt = r.get("mqtbench", {})
                    sections.append(f"**Available circuits:** {len(hw)} hand-written + {len(mqt)} MQTBench")
                    sections.append("")
                except:
                    pass

        if not sections or len(sections) <= 1:
            sections.append("Analysis complete. No further action required.")

        return "\n".join(sections)


# ═══════════════════════════════════════════════════════
# Main Agent class
# ═══════════════════════════════════════════════════════

class QuantumAgent:
    """Minimal while-loop agent with Anthropic tool calling.

    Architecture:
        User prompt
            → LLM / RulePlanner (with tool definitions)
            → tool_use block(s) returned
            → ToolExecutor runs them
            → results fed back
            → repeat until final text (no more tool calls)

    Falls back to RulePlanner when no API key is available.
    """

    def __init__(
        self,
        backend: ShadowBackend,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "auto",
        max_turns: int = MAX_TURNS,
        verbose: bool = True,
        use_mock: bool = False,
        db_path: Optional[str] = None,
        wandb_run=None,
    ):
        """
        Args:
            provider: "openai", "anthropic", "auto" (try openai first), or "mock"
            base_url: OpenAI-compatible base URL (e.g. "https://api2.tabcode.cc/openai/pro")
            api_key: API key for the provider
        """
        self.backend = backend
        self.model = model
        self.max_turns = max_turns
        self.verbose = verbose
        self.use_mock = use_mock or (provider == "mock")
        self.client = None
        self.provider = provider  # "openai", "anthropic", "mock"

        # Set up executor — instrumented if available
        if HAS_INSTRUMENTED and db_path is not None:
            self._db = DataStore(db_path)
            self.executor = InstrumentedExecutor(
                backend, db=self._db, wandb_run=wandb_run)
        else:
            self._db = None
            self.executor = ToolExecutor(backend)

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

        if self.client is None:
            self.use_mock = True
            self.provider = "mock"
            self.model = "rule-planner-v1"

        self.planner = RulePlanner() if self.use_mock else None

    def _log_session(self, user_prompt: str, result: 'AgentResult'):
        """Log completed agent session to DuckDB."""
        if self._db is None:
            return
        self._db.log_agent_session({
            "ts_start": datetime.fromtimestamp(result.turns[0].timestamp, tz=timezone.utc).isoformat()
                if result.turns else _now_iso(),
            "ts_end": _now_iso(),
            "model": result.model,
            "provider": self.provider,
            "backend": self.backend.name,
            "user_prompt": user_prompt,
            "final_answer": result.final_answer[:2000],
            "num_tool_calls": len(result.tool_calls_made),
            "total_tokens": result.total_tokens,
            "elapsed_seconds": result.elapsed_seconds,
            "tool_calls": result.tool_calls_made,
        })

    def run(self, user_prompt: str) -> AgentResult:
        """Run the agent loop on a user prompt."""
        if self.use_mock:
            result = self._run_mock(user_prompt)
        elif self.provider == "openai":
            result = self._run_openai(user_prompt)
        else:
            result = self._run_api(user_prompt)
        self._log_session(user_prompt, result)
        return result

    def _run_mock(self, user_prompt: str) -> AgentResult:
        """Run with the rule-based planner (no API needed)."""
        t0 = time.time()
        turns: list[AgentTurn] = [
            AgentTurn(role="user", content=user_prompt, timestamp=time.time())
        ]
        all_tool_calls: list[dict] = []
        history: list[dict] = []  # {tool, input, result}

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"QuantumGPT Agent ({self.model})")
            print(f"Backend: {self.backend.name} ({self.backend.num_qubits}q)")
            print(f"{'='*60}")
            print(f"\nUser: {user_prompt}\n")

        for turn_num in range(self.max_turns):
            tool_calls, final_text = self.planner.plan(user_prompt, history)

            if not tool_calls:
                # Final answer
                turns.append(AgentTurn(
                    role="assistant", content=final_text, timestamp=time.time(),
                ))
                elapsed = time.time() - t0
                if self.verbose:
                    print(f"Agent:\n{final_text}\n")
                    print(f"\n{'─'*60}")
                    print(f"Completed in {elapsed:.1f}s, {len(all_tool_calls)} tool calls")

                return AgentResult(
                    final_answer=final_text,
                    turns=turns,
                    tool_calls_made=all_tool_calls,
                    total_tokens=0,
                    elapsed_seconds=elapsed,
                    model=self.model,
                )

            # Execute tool calls
            for call in tool_calls:
                tool_name = call["name"]
                tool_input = call["input"]

                if self.verbose:
                    print(f"  [tool] {tool_name}({json.dumps(tool_input, default=str)})")

                result_str = self.executor.execute(tool_name, tool_input)

                if self.verbose:
                    display = result_str[:300] + "..." if len(result_str) > 300 else result_str
                    print(f"  <- {display}\n")

                history.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "result": result_str,
                })
                all_tool_calls.append({
                    "turn": turn_num,
                    "tool": tool_name,
                    "input": tool_input,
                    "result_length": len(result_str),
                })

            turns.append(AgentTurn(
                role="assistant",
                content=f"[Called {len(tool_calls)} tool(s)]",
                tool_calls=tool_calls,
                timestamp=time.time(),
            ))

        elapsed = time.time() - t0
        return AgentResult(
            final_answer="[Agent reached maximum turns without a final answer]",
            turns=turns,
            tool_calls_made=all_tool_calls,
            total_tokens=0,
            elapsed_seconds=elapsed,
            model=self.model,
        )

    def _run_openai(self, user_prompt: str) -> AgentResult:
        """Run with OpenAI-compatible API (supports tool calling)."""
        t0 = time.time()

        # Convert our tool definitions to OpenAI format
        openai_tools = []
        for td in TOOL_DEFINITIONS:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": td["name"],
                    "description": td["description"],
                    "parameters": td["input_schema"],
                },
            })

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        turns: list[AgentTurn] = [
            AgentTurn(role="user", content=user_prompt, timestamp=time.time())
        ]
        all_tool_calls: list[dict] = []
        total_tokens = 0

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"QuantumGPT Agent ({self.model}) via OpenAI API")
            print(f"Backend: {self.backend.name} ({self.backend.num_qubits}q)")
            print(f"{'='*60}")
            print(f"\nUser: {user_prompt}\n")

        for turn_num in range(self.max_turns):
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=2048,
                tools=openai_tools,
                messages=messages,
            )

            choice = response.choices[0]
            usage = response.usage
            if usage:
                total_tokens += (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)

            msg = choice.message

            # Print text content
            if self.verbose and msg.content:
                print(f"Agent: {msg.content.strip()}\n")

            # Check for tool calls
            if not msg.tool_calls:
                # Final answer
                final_text = (msg.content or "").strip()
                turns.append(AgentTurn(role="assistant", content=final_text, timestamp=time.time()))
                elapsed = time.time() - t0
                if self.verbose:
                    print(f"\n{'─'*60}")
                    print(f"Completed in {elapsed:.1f}s, {len(all_tool_calls)} tool calls, "
                          f"{total_tokens} tokens")
                return AgentResult(
                    final_answer=final_text, turns=turns,
                    tool_calls_made=all_tool_calls, total_tokens=total_tokens,
                    elapsed_seconds=elapsed, model=self.model,
                )

            # Process tool calls
            # Add assistant message with tool calls to history
            messages.append(msg.model_dump())

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}

                if self.verbose:
                    print(f"  [tool] {tool_name}({json.dumps(tool_input, default=str)})")

                result_str = self.executor.execute(tool_name, tool_input)

                if self.verbose:
                    display = result_str[:300] + "..." if len(result_str) > 300 else result_str
                    print(f"  <- {display}\n")

                all_tool_calls.append({
                    "turn": turn_num, "tool": tool_name,
                    "input": tool_input, "result_length": len(result_str),
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

            turns.append(AgentTurn(
                role="assistant",
                content=f"[Called {len(msg.tool_calls)} tool(s)]",
                tool_calls=[{"name": tc.function.name, "input": tc.function.arguments}
                            for tc in msg.tool_calls],
                timestamp=time.time(),
            ))

        elapsed = time.time() - t0
        return AgentResult(
            final_answer="[Agent reached maximum turns without a final answer]",
            turns=turns, tool_calls_made=all_tool_calls,
            total_tokens=total_tokens, elapsed_seconds=elapsed, model=self.model,
        )

    def _run_api(self, user_prompt: str) -> AgentResult:
        """Run with real Anthropic API."""
        import anthropic as anth

        t0 = time.time()
        messages = [{"role": "user", "content": user_prompt}]
        turns: list[AgentTurn] = [
            AgentTurn(role="user", content=user_prompt, timestamp=time.time())
        ]
        all_tool_calls: list[dict] = []
        total_tokens = 0

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"QuantumGPT Agent ({self.model})")
            print(f"Backend: {self.backend.name} ({self.backend.num_qubits}q)")
            print(f"{'='*60}")
            print(f"\nUser: {user_prompt}\n")

        for turn_num in range(self.max_turns):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
            total_tokens += response.usage.input_tokens + response.usage.output_tokens

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if self.verbose:
                for tb in text_blocks:
                    if tb.text.strip():
                        print(f"Agent: {tb.text.strip()}\n")

            if not tool_use_blocks:
                final_text = "\n".join(tb.text for tb in text_blocks).strip()
                turns.append(AgentTurn(role="assistant", content=final_text, timestamp=time.time()))
                elapsed = time.time() - t0
                if self.verbose:
                    print(f"\n{'─'*60}")
                    print(f"Completed in {elapsed:.1f}s, {len(all_tool_calls)} tool calls, "
                          f"{total_tokens} tokens")
                return AgentResult(
                    final_answer=final_text, turns=turns,
                    tool_calls_made=all_tool_calls, total_tokens=total_tokens,
                    elapsed_seconds=elapsed, model=self.model,
                )

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
                if self.verbose:
                    print(f"  [tool] {tool_name}({json.dumps(tool_input, default=str)})")
                result_str = self.executor.execute(tool_name, tool_input)
                if self.verbose:
                    display = result_str[:300] + "..." if len(result_str) > 300 else result_str
                    print(f"  <- {display}\n")
                all_tool_calls.append({
                    "turn": turn_num, "tool": tool_name,
                    "input": tool_input, "result_length": len(result_str),
                })
                tool_results_content.append({
                    "type": "tool_result", "tool_use_id": block.id, "content": result_str,
                })
            messages.append({"role": "user", "content": tool_results_content})
            turns.append(AgentTurn(
                role="assistant", content=assistant_content,
                tool_calls=[{"name": b.name, "input": b.input} for b in tool_use_blocks],
                timestamp=time.time(),
            ))

        elapsed = time.time() - t0
        return AgentResult(
            final_answer="[Agent reached maximum turns without a final answer]",
            turns=turns, tool_calls_made=all_tool_calls,
            total_tokens=total_tokens, elapsed_seconds=elapsed, model=self.model,
        )
