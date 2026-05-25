#!/usr/bin/env python3
"""F4: LLM vs Mock comparison on 10 standardized prompts.

Runs each prompt with both mock (rule-based) and deepseek (LLM) providers,
then produces a comparison table.
"""

import sys
import os
import json
import time
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from backends.fake_adapter import FakeBackendAdapter
from agent.react import ReActAgent


# ── 10 standardized prompts ──────────────────────────────
PROMPTS = [
    # 1. Basic perception
    "Check the health of the backend and report the average gate errors and T1 time.",

    # 2. Circuit execution
    "Run a 5-qubit GHZ circuit and report the fidelity.",

    # 3. Drift detection
    "Check if there is any calibration drift on the backend.",

    # 4. Qubit properties
    "What are the T1, T2, and readout error for qubits 0, 1, and 2?",

    # 5. Transpilation + mitigation
    "Transpile a 3-qubit QFT circuit for this backend, run it, and apply ZNE mitigation if fidelity is below 0.9.",

    # 6. Rabi experiment
    "Run a Rabi experiment to find the pi-pulse amplitude for qubit 0.",

    # 7. T2* measurement
    "Measure the T2* dephasing time of qubit 0 using a Ramsey experiment.",

    # 8. T1 measurement
    "Measure the T1 relaxation time of qubit 0.",

    # 9. Full characterization
    "Characterize qubit 0: measure the pi-pulse amplitude, T2*, and T1. Report all values.",

    # 10. Diagnosis
    "Diagnose qubit 0 and suggest improvements if any parameters are suboptimal.",
]


def run_single(prompt: str, provider: str, backend) -> dict:
    """Run one prompt and return summary dict."""
    max_turns = 12 if provider == "deepseek" else 20
    agent = ReActAgent(
        backend=backend,
        provider=provider,
        model="deepseek-chat" if provider == "deepseek" else None,
        verbose=False,
        max_turns=max_turns,
        target_fidelity=0.85,
        max_tool_calls=12,
        max_seconds=120.0,
        use_memory=False,
    )
    try:
        trace = agent.run(prompt)
        return {
            "provider": provider,
            "prompt_idx": None,  # filled later
            "success": not trace.final_answer.startswith("Error"),
            "tool_calls": trace.num_tool_calls,
            "tokens": trace.total_tokens,
            "elapsed_s": round(trace.elapsed_seconds, 1),
            "tools_used": [s.action for s in trace.steps if s.action],
            "answer_len": len(trace.final_answer),
            "error": None,
        }
    except Exception as e:
        return {
            "provider": provider,
            "prompt_idx": None,
            "success": False,
            "tool_calls": 0,
            "tokens": 0,
            "elapsed_s": 0,
            "tools_used": [],
            "answer_len": 0,
            "error": str(e)[:200],
        }


def main():
    backend = FakeBackendAdapter("FakeBrisbane")

    results = []

    for i, prompt in enumerate(PROMPTS):
        short = prompt[:60] + ("..." if len(prompt) > 60 else "")
        print(f"\n{'='*70}")
        print(f"Prompt {i+1}/10: {short}")
        print(f"{'='*70}")

        # Run mock
        print(f"  Running MOCK...", end="", flush=True)
        r_mock = run_single(prompt, "mock", backend)
        r_mock["prompt_idx"] = i + 1
        results.append(r_mock)
        print(f" done ({r_mock['elapsed_s']}s, {r_mock['tool_calls']} calls)")

        # Run DeepSeek
        print(f"  Running DEEPSEEK...", end="", flush=True)
        r_ds = run_single(prompt, "deepseek", backend)
        r_ds["prompt_idx"] = i + 1
        results.append(r_ds)
        print(f" done ({r_ds['elapsed_s']}s, {r_ds['tool_calls']} calls, {r_ds['tokens']} tokens)")

    # ── Print comparison table ────────────────────────────
    print("\n\n" + "=" * 90)
    print("F4 COMPARISON: MOCK vs DEEPSEEK")
    print("=" * 90)

    header = f"{'#':>2} | {'Provider':>8} | {'OK?':>3} | {'Tools':>5} | {'Tokens':>6} | {'Time':>6} | {'Ans':>5} | Tools Used"
    print(header)
    print("-" * 90)

    for r in results:
        ok = "✓" if r["success"] else "✗"
        tools_str = ", ".join(r["tools_used"][:5])
        if len(r["tools_used"]) > 5:
            tools_str += f" +{len(r['tools_used'])-5}"
        print(f"{r['prompt_idx']:>2} | {r['provider']:>8} | {ok:>3} | {r['tool_calls']:>5} | {r['tokens']:>6} | {r['elapsed_s']:>5.1f}s | {r['answer_len']:>5} | {tools_str}")

    # ── Summary stats ─────────────────────────────────────
    mock_results = [r for r in results if r["provider"] == "mock"]
    ds_results = [r for r in results if r["provider"] == "deepseek"]

    mock_ok = sum(1 for r in mock_results if r["success"])
    ds_ok = sum(1 for r in ds_results if r["success"])
    mock_tools = sum(r["tool_calls"] for r in mock_results)
    ds_tools = sum(r["tool_calls"] for r in ds_results)
    mock_time = sum(r["elapsed_s"] for r in mock_results)
    ds_time = sum(r["elapsed_s"] for r in ds_results)
    ds_tokens = sum(r["tokens"] for r in ds_results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'':>15} | {'Mock':>8} | {'DeepSeek':>8}")
    print(f"  {'Success rate':>15} | {mock_ok}/10    | {ds_ok}/10")
    print(f"  {'Total tools':>15} | {mock_tools:>8} | {ds_tools:>8}")
    print(f"  {'Total time':>15} | {mock_time:>7.1f}s | {ds_time:>7.1f}s")
    print(f"  {'Total tokens':>15} | {'N/A':>8} | {ds_tokens:>8}")
    print(f"  {'Avg tools/task':>15} | {mock_tools/10:>8.1f} | {ds_tools/10:>8.1f}")
    print(f"  {'Avg time/task':>15} | {mock_time/10:>7.1f}s | {ds_time/10:>7.1f}s")

    # Save raw results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f4_mock_vs_deepseek.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to: {out_path}")


if __name__ == "__main__":
    main()
