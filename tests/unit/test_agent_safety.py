"""Tests for minimal safety policy around tool execution."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.safety import SafetyAwareExecutor, SafetyPolicy


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, tool_name, tool_input):
        self.calls.append((tool_name, tool_input))
        return json.dumps({"ok": True, "tool": tool_name})


def test_safety_policy_dry_run_returns_plan_without_calling_executor():
    inner = RecordingExecutor()
    executor = SafetyAwareExecutor(inner, SafetyPolicy(dry_run=True))

    result = json.loads(executor.execute("rabi_experiment", {"amp_max": 0.02}))

    assert result["safety"]["status"] == "dry_run"
    assert result["safety"]["would_execute"] is True
    assert result["tool"] == "rabi_experiment"
    assert inner.calls == []


def test_safety_policy_audits_allowed_tool_execution():
    inner = RecordingExecutor()
    policy = SafetyPolicy(dry_run=False, audit_enabled=True)
    executor = SafetyAwareExecutor(inner, policy)

    result = json.loads(executor.execute("rabi_experiment", {"amp_max": 0.02, "n_points": 5}))

    assert result["ok"] is True
    assert inner.calls == [("rabi_experiment", {"amp_max": 0.02, "n_points": 5})]
    assert len(policy.audit_log) == 1
    assert policy.audit_log[0]["tool"] == "rabi_experiment"
    assert policy.audit_log[0]["decision"] == "allow"


def test_safety_policy_blocks_risky_pulse_amplitude_before_execution():
    inner = RecordingExecutor()
    policy = SafetyPolicy(max_rabi_amp_ghz=0.05)
    executor = SafetyAwareExecutor(inner, policy)

    result = json.loads(executor.execute("rabi_experiment", {"amp_max": 0.2}))

    assert result["error"] == "blocked_by_safety_policy"
    assert "amp_max" in result["reason"]
    assert inner.calls == []
    assert policy.audit_log[-1]["decision"] == "block"


def test_safety_policy_blocks_too_many_shots():
    policy = SafetyPolicy(max_shots=8192)

    decision = policy.assess("run_circuit", {"circuit_name": "ghz_5", "shots": 100000})

    assert decision.allowed is False
    assert "shots" in decision.reason
