"""Agent failure-recovery and grounding tests."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.react import ReActAgent
from agent.state import ArtifactStatus, ArtifactType, InvalidationReason
from backends.fake_adapter import FakeBackendAdapter


class BrokenFitExecutor:
    def __init__(self, inner):
        self.inner = inner

    def execute(self, tool_name, tool_input):
        if tool_name == "fit_rabi":
            return json.dumps({"error": "fit_failed", "reason": "synthetic optimizer failure"})
        return self.inner.execute(tool_name, tool_input)


def test_agent_final_answer_reports_failed_fit_instead_of_claiming_success():
    agent = ReActAgent(
        FakeBackendAdapter("FakeBrisbane"),
        provider="mock",
        use_memory=False,
        verbose=False,
        max_turns=5,
    )
    agent.executor = BrokenFitExecutor(agent.executor)

    trace = agent.run("Run a Rabi experiment on a 5.0 GHz qubit and fit the pi pulse.")

    assert "Tool issue: fit_rabi" in trace.final_answer
    assert "fit_failed" in trace.final_answer
    assert "synthetic optimizer failure" in trace.final_answer
    assert "**Rabi Fit Result**" not in trace.final_answer


def test_failed_tool_result_is_tracked_as_failed_state_artifact():
    agent = ReActAgent(
        FakeBackendAdapter("FakeBrisbane"),
        provider="mock",
        use_memory=False,
        verbose=False,
        max_turns=5,
    )
    agent.executor = BrokenFitExecutor(agent.executor)

    trace = agent.run("Run a Rabi experiment on a 5.0 GHz qubit and fit the pi pulse.")

    failed_fit_artifacts = [
        artifact for artifact in trace.state.artifacts.artifacts.values()
        if artifact.artifact_type is ArtifactType.FIT_RESULT
        and artifact.status is ArtifactStatus.FAILED
    ]
    assert len(failed_fit_artifacts) == 1
    assert failed_fit_artifacts[0].invalidation_reason is InvalidationReason.TOOL_FAILURE
    assert failed_fit_artifacts[0].metadata["error"] == "fit_failed"
