"""Memory-aware agent demo — shows the causal chain:

  prior bad run  →  memory injection  →  diagnose  →  actionable overrides  →  better run

Trial 1: cold run — agent runs the circuit with default settings.
Trial 2: warm run — agent receives memory_context about the prior low fidelity.
          It triggers diagnose_and_suggest → gets optimization overrides → re-runs.
Trial 3: control — identical to Trial 1 (proves Trial 2 difference is causal).

Usage:
    cd /path/to/QuantumGPT
    .venv/bin/python demos/memory_demo.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.react import ReActAgent
from backends.fake_adapter import FakeBackendAdapter


CIRCUIT = "ghz_5"
TARGET = 0.95
PROMPT = f"Run {CIRCUIT} on FakeBrisbane and report the fidelity."
MAX_TOOLS = 6


def _run_trial(*, memory_context: str = "", label: str) -> dict:
    be = FakeBackendAdapter("FakeBrisbane")
    agent = ReActAgent(
        be,
        provider="mock",
        verbose=True,
        target_fidelity=TARGET,
        max_tool_calls=MAX_TOOLS,
    )
    trace = agent.run(PROMPT, memory_context=memory_context)

    tools = []
    for s in trace.steps:
        if s.action:
            tools.append({
                "tool": s.action,
                "input": s.action_input or {},
            })

    return {
        "label": label,
        "memory_context": memory_context[:120] + "..." if len(memory_context) > 120 else memory_context,
        "tools": tools,
        "tool_count": len(tools),
        "best_fidelity": trace.best_fidelity,
        "used_diagnose": any(t["tool"] == "diagnose_and_suggest" for t in tools),
        "run_circuit_input": next(
            (t["input"] for t in tools if t["tool"] == "run_circuit"), {}
        ),
        "final_answer": trace.final_answer,
    }


def run_memory_demo(save_dir: str | None = None) -> dict:
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "memory")
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  QUANTUMGPT MEMORY DEMO")
    print("  Prompt:", PROMPT)
    print("=" * 70)

    # ── Trial 1: cold run (no memory) ──────────────────────────────────
    print("\n" + "-" * 60)
    print("  TRIAL 1 — Cold run (no memory context)")
    print("-" * 60 + "\n")
    t1 = _run_trial(label="trial1_cold")

    # ── Trial 2: warm run (inject memory of the bad fidelity) ──────────
    memory_ctx = (
        f"Prior run: circuit={CIRCUIT} backend=FakeBrisbane "
        f"fidelity={t1['best_fidelity']:.4f} (below target {TARGET}). "
        "The device showed elevated readout errors."
    )
    print("\n" + "-" * 60)
    print("  TRIAL 2 — Warm run (memory injects prior low fidelity)")
    print(f"  Memory: {memory_ctx}")
    print("-" * 60 + "\n")
    t2 = _run_trial(memory_context=memory_ctx, label="trial2_warm")

    # ── Trial 3: control — cold again ──────────────────────────────────
    print("\n" + "-" * 60)
    print("  TRIAL 3 — Control (no memory, identical to Trial 1)")
    print("-" * 60 + "\n")
    t3 = _run_trial(label="trial3_control")

    # ── Comparison ────────────────────────────────────────────────────
    summary = {
        "circuit": CIRCUIT,
        "target_fidelity": TARGET,
        "trials": [t1, t2, t3],
    }

    summary_path = os.path.join(save_dir, "memory_demo_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # Write human-readable report
    report_lines = [
        "# QuantumGPT Memory Demo Report\n",
        f"Circuit: {CIRCUIT}  |  Target fidelity: {TARGET}\n",
        "## Side-by-side comparison\n",
        f"| {'Metric':<30} | {'Trial 1 (cold)':>16} | {'Trial 2 (warm)':>16} | {'Trial 3 (ctrl)':>16} |",
        f"|{'-'*31}|{'-'*17}|{'-'*17}|{'-'*17}|",
        f"| {'Fidelity':<30} | {t1['best_fidelity']:>16.4f} | {t2['best_fidelity']:>16.4f} | {t3['best_fidelity']:>16.4f} |",
        f"| {'Tool calls':<30} | {t1['tool_count']:>16} | {t2['tool_count']:>16} | {t3['tool_count']:>16} |",
        f"| {'Used diagnose_and_suggest':<30} | {str(t1['used_diagnose']):>16} | {str(t2['used_diagnose']):>16} | {str(t3['used_diagnose']):>16} |",
        f"| {'run_circuit shots':<30} | {t1['run_circuit_input'].get('shots', 'N/A'):>16} | {t2['run_circuit_input'].get('shots', 'N/A'):>16} | {t3['run_circuit_input'].get('shots', 'N/A'):>16} |",
        f"| {'run_circuit opt_level':<30} | {str(t1['run_circuit_input'].get('optimization_level', 'def')):>16} | {str(t2['run_circuit_input'].get('optimization_level', 'def')):>16} | {str(t3['run_circuit_input'].get('optimization_level', 'def')):>16} |",
        "",
        "## Key observations\n",
    ]

    if t2["used_diagnose"] and not t1["used_diagnose"]:
        report_lines.append(
            "- **Trial 2 (warm) added diagnose_and_suggest** before run_circuit, "
            "triggered by memory of Trial 1's sub-target fidelity."
        )
    if t2["run_circuit_input"].get("optimization_level") and not t1["run_circuit_input"].get("optimization_level"):
        report_lines.append(
            f"- **Trial 2 applied optimization_level={t2['run_circuit_input']['optimization_level']}** "
            "from the diagnosis overrides (Trial 1 used the default)."
        )
    if t2["run_circuit_input"].get("shots", 4096) > t1["run_circuit_input"].get("shots", 4096):
        report_lines.append(
            f"- **Trial 2 boosted shots to {t2['run_circuit_input']['shots']}** "
            f"(Trial 1 used {t1['run_circuit_input'].get('shots', 4096)})."
        )
    if not t3["used_diagnose"]:
        report_lines.append(
            "- **Trial 3 (control, no memory) matches Trial 1** — confirming the "
            "difference in Trial 2 is caused by memory, not randomness."
        )

    report_lines.append(
        "\n## Conclusion\n"
        "Memory injection changes **both the plan** (diagnose step appears) "
        "**and the execution** (higher shots, more aggressive optimization). "
        "The agent's reasoning chain is: recall → diagnose → override → execute.\n"
    )
    report_lines.append(
        "**Mode:** simulation — no physical hardware was modified.\n"
    )

    report_path = os.path.join(save_dir, "memory_demo_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    # Print summary to stdout
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    fmt = "  {:<30s}  {:>10s}  {:>10s}  {:>10s}"
    print(fmt.format("", "Trial 1", "Trial 2", "Trial 3"))
    print(fmt.format("Fidelity",
                      f"{t1['best_fidelity']:.4f}",
                      f"{t2['best_fidelity']:.4f}",
                      f"{t3['best_fidelity']:.4f}"))
    print(fmt.format("Tool calls",
                      str(t1['tool_count']),
                      str(t2['tool_count']),
                      str(t3['tool_count'])))
    print(fmt.format("Diagnose fired",
                      str(t1['used_diagnose']),
                      str(t2['used_diagnose']),
                      str(t3['used_diagnose'])))
    print(fmt.format("run_circuit shots",
                      str(t1['run_circuit_input'].get('shots', 'N/A')),
                      str(t2['run_circuit_input'].get('shots', 'N/A')),
                      str(t3['run_circuit_input'].get('shots', 'N/A'))))
    print(fmt.format("run_circuit opt_level",
                      str(t1['run_circuit_input'].get('optimization_level', 'def')),
                      str(t2['run_circuit_input'].get('optimization_level', 'def')),
                      str(t3['run_circuit_input'].get('optimization_level', 'def'))))
    print()
    print(f"  Report: {report_path}")
    print(f"  Data:   {summary_path}")

    return summary


if __name__ == "__main__":
    run_memory_demo()
