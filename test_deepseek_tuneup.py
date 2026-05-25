#!/usr/bin/env python3
"""DeepSeek agent: full tune-up (Rabi → T2* → T1) in one prompt."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends.fake_adapter import FakeBackendAdapter
from agent.react import ReActAgent


def main():
    backend = FakeBackendAdapter("FakeBrisbane")

    agent = ReActAgent(
        backend=backend,
        provider="deepseek",
        model="deepseek-chat",
        verbose=True,
        max_turns=12,
        target_fidelity=0.85,
        max_tool_calls=12,
        max_seconds=180.0,
        use_memory=False,
    )

    prompt = (
        "Characterize qubit 0: first find the pi-pulse amplitude with a Rabi experiment, "
        "then measure T2* with Ramsey fringes, and finally measure T1 relaxation. "
        "Report all three values."
    )

    trace = agent.run(prompt)

    print("\n" + "=" * 60)
    print(f"Tool calls: {trace.num_tool_calls}")
    print(f"Tokens: {trace.total_tokens}")
    print(f"Elapsed: {trace.elapsed_seconds:.1f}s")
    tools_used = [s.action for s in trace.steps if s.action]
    print(f"Tools used: {tools_used}")


if __name__ == "__main__":
    main()
