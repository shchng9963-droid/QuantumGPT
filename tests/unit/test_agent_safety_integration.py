"""Integration tests for SafetyPolicy inside the ReActAgent loop."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.react import ReActAgent
from agent.safety import SafetyPolicy
from backends.fake_adapter import FakeBackendAdapter


def _rabi_prompt() -> str:
    return "Run a Rabi experiment on a 5.0 GHz qubit and fit the pi pulse."


def test_react_agent_blocks_unsafe_rabi_tool_call_and_records_trace_diagnostics():
    backend = FakeBackendAdapter("FakeBrisbane")
    policy = SafetyPolicy(max_rabi_amp_ghz=0.01)
    agent = ReActAgent(
        backend,
        provider="mock",
        use_memory=False,
        verbose=False,
        safety_policy=policy,
        max_turns=4,
    )

    trace = agent.run(_rabi_prompt())

    rabi_steps = [s for s in trace.steps if s.action == "rabi_experiment"]
    assert rabi_steps, "agent should attempt the requested Rabi experiment"
    blocked = json.loads(rabi_steps[0].observation)
    assert blocked["error"] == "blocked_by_safety_policy"
    assert blocked["safety"]["status"] == "blocked"
    assert trace.diagnostics.safety_block_count == 1
    assert trace.diagnostics.safety_dry_run_count == 0
    assert policy.audit_log[-1]["decision"] == "block"
    assert "blocked" in trace.final_answer.lower() or "safety" in trace.final_answer.lower()


def test_react_agent_dry_run_does_not_record_successful_lab_result_artifact():
    backend = FakeBackendAdapter("FakeBrisbane")
    policy = SafetyPolicy(dry_run=True)
    agent = ReActAgent(
        backend,
        provider="mock",
        use_memory=False,
        verbose=False,
        safety_policy=policy,
        max_turns=4,
    )

    trace = agent.run(_rabi_prompt())

    rabi_steps = [s for s in trace.steps if s.action == "rabi_experiment"]
    assert rabi_steps
    dry_run = json.loads(rabi_steps[0].observation)
    assert dry_run["safety"]["status"] == "dry_run"
    assert dry_run["safety"]["would_execute"] is True
    assert trace.diagnostics.safety_dry_run_count >= 1
    assert trace.diagnostics.safety_block_count == 0

    artifacts = trace.to_dict(include_observations=True).get("artifacts", {})
    lab_artifacts = [
        a for a in artifacts.values()
        if a.get("artifact_type") == "lab_experiment_result"
    ]
    assert lab_artifacts == []
    assert policy.audit_log[-1]["decision"] == "dry_run"
