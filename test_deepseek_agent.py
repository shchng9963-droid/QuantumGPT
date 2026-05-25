#!/usr/bin/env python3
"""Quick smoke test: DeepSeek API agent on a T2* measurement task."""

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
        max_turns=10,
        target_fidelity=0.85,
        max_tool_calls=10,
        max_seconds=120.0,
        use_memory=False,
    )

    print(f"Provider resolved: {agent.provider}")
    print(f"Model: {agent.model}")
    print(f"Client: {type(agent.client).__name__ if agent.client else 'None'}")
    print(f"Use mock: {agent.use_mock}")
    print()

    if agent.use_mock:
        print("ERROR: Agent fell back to mock mode! DeepSeek API key not working.")
        sys.exit(1)

    trace = agent.run("Measure the T2* dephasing time of qubit 0 using a Ramsey experiment.")

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(trace.final_answer)
    print(f"\nTool calls: {trace.num_tool_calls}")
    print(f"Tokens: {trace.total_tokens}")
    print(f"Elapsed: {trace.elapsed_seconds:.1f}s")
    print(f"Provider: {trace.provider}")
    print(f"Model: {trace.model}")


if __name__ == "__main__":
    main()
