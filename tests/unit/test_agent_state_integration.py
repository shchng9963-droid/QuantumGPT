"""Integration tests for AgentState inside ReActAgent traces."""

from backends.fake_adapter import FakeBackendAdapter
from agent.react import ReActAgent
from agent.drift_aware import DriftAwareRulePlanner, ReplanningPolicy
from agent.state import ArtifactStatus, ArtifactType, InvalidationReason


def test_mock_agent_trace_exports_state_summary_and_artifacts():
    agent = ReActAgent(
        FakeBackendAdapter("FakeBrisbane"),
        provider="mock",
        use_memory=False,
        verbose=False,
    )

    trace = agent.run("Check backend health and run ghz_5 fidelity")
    data = trace.to_dict(include_observations=True)

    assert trace.state is not None
    assert data["state_summary"]["backend_name"] == "FakeBrisbane"
    assert data["state_summary"]["artifact_counts"]["backend_snapshot"] == 1
    assert data["state_summary"]["artifact_counts"]["circuit_result"] == 1
    assert "artifacts" in data

    artifact_types = {
        artifact["artifact_type"]
        for artifact in data["artifacts"].values()
    }
    assert ArtifactType.BACKEND_SNAPSHOT.value in artifact_types
    assert ArtifactType.CIRCUIT_RESULT.value in artifact_types


def test_tool_artifacts_depend_on_latest_backend_snapshot():
    agent = ReActAgent(
        FakeBackendAdapter("FakeBrisbane"),
        provider="mock",
        use_memory=False,
        verbose=False,
    )

    trace = agent.run("Check backend health, predict ghz_5 fidelity, and run ghz_5")

    snapshot_artifacts = trace.state.artifacts.by_type(ArtifactType.BACKEND_SNAPSHOT)
    prediction_artifacts = trace.state.artifacts.by_type(ArtifactType.PREDICTED_FIDELITY)
    result_artifacts = trace.state.artifacts.by_type(ArtifactType.CIRCUIT_RESULT)

    assert len(snapshot_artifacts) == 1
    assert len(prediction_artifacts) == 1
    assert len(result_artifacts) == 1
    snapshot = snapshot_artifacts[0]

    assert trace.state.backend_snapshot_id == snapshot.artifact_id
    assert prediction_artifacts[0].depends_on == [snapshot.artifact_id]
    assert result_artifacts[0].depends_on == [snapshot.artifact_id]
    assert prediction_artifacts[0].backend_snapshot_hash == snapshot.metadata["snapshot_hash"]
    assert result_artifacts[0].backend_snapshot_hash == snapshot.metadata["snapshot_hash"]
    assert prediction_artifacts[0].status is ArtifactStatus.VALID
    assert result_artifacts[0].status is ArtifactStatus.VALID


def test_trace_steps_include_state_summary_after_tool_execution():
    agent = ReActAgent(
        FakeBackendAdapter("FakeBrisbane"),
        provider="mock",
        use_memory=False,
        verbose=False,
    )

    trace = agent.run("Check backend health and qubit properties")
    tool_steps = [step for step in trace.steps if step.action]

    assert len(tool_steps) >= 2
    assert tool_steps[0].state_summary["artifact_counts"] == {"backend_snapshot": 1}
    assert tool_steps[1].state_summary["artifact_counts"] == {
        "backend_snapshot": 1,
        "qubit_properties": 1,
    }


class _FakeDriftState:
    def __init__(self, is_drifting=False, drift_score=0.0, invalidated_results=None):
        self.is_drifting = is_drifting
        self.drift_score = drift_score
        self.invalidated_results = list(invalidated_results or [])
        self.consecutive_drift_checks = 1 if is_drifting else 0
        self.recovery_steps = 0


class _FakeDriftMonitor:
    def __init__(self):
        self.calls = 0
        self.acknowledged = False
        self.state = _FakeDriftState()

    def step(self):
        self.calls += 1
        if self.acknowledged:
            return self.state
        if self.calls >= 2:
            self.state = _FakeDriftState(
                is_drifting=True,
                drift_score=0.91,
                invalidated_results=["transpile", "predict_fidelity", "run_circuit"],
            )
        else:
            self.state = _FakeDriftState()
        return self.state

    def acknowledge_replan(self):
        self.acknowledged = True
        self.state.invalidated_results = []


def test_drift_monitor_invalidates_state_artifacts_during_react_loop():
    agent = ReActAgent(
        FakeBackendAdapter("FakeBrisbane"),
        provider="mock",
        use_memory=False,
        verbose=False,
        use_drift_aware=False,
    )
    monitor = _FakeDriftMonitor()
    agent.drift_monitor = monitor

    trace = agent.run("Check backend health and run ghz_5")

    assert trace.state.drift_state["is_drifting"] is True
    assert trace.state.drift_state["drift_score"] == 0.91
    assert monitor.acknowledged is False

    stale = trace.state.artifacts.by_status(ArtifactStatus.STALE)
    assert stale, "drift should mark old snapshot-dependent artifacts stale"
    assert any(a.artifact_type is ArtifactType.CIRCUIT_RESULT for a in stale)
    assert all(a.invalidation_reason is InvalidationReason.BACKEND_DRIFT for a in stale)

    final_tool_step = [step for step in trace.steps if step.action][-1]
    assert final_tool_step.state_summary["drift_state"]["is_drifting"] is True
    assert final_tool_step.state_summary["artifact_status_counts"]["stale"] >= 1


def test_use_drift_aware_wraps_mock_rule_planner_for_replanning():
    agent = ReActAgent(
        FakeBackendAdapter("FakeBrisbane"),
        provider="mock",
        use_memory=False,
        verbose=False,
        use_drift_aware=True,
    )

    assert agent.drift_monitor is not None
    assert isinstance(agent.planner, DriftAwareRulePlanner)


class _RecordingExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, tool_name: str, tool_input: dict) -> str:
        self.calls.append((tool_name, dict(tool_input)))
        if tool_name == "get_backend_health":
            return '{"backend":"FakeBrisbane","avg_2q_error":0.01,"drift_score":0.0}'
        if tool_name == "run_circuit":
            return '{"circuit":"ghz_5","fidelity":0.91,"shots":4096}'
        return "{}"


def test_drift_aware_mock_agent_forces_health_recheck_and_rerun_after_drift():
    agent = ReActAgent(
        FakeBackendAdapter("FakeBrisbane"),
        provider="mock",
        use_memory=False,
        verbose=False,
        use_drift_aware=False,
        max_turns=6,
    )
    monitor = _FakeDriftMonitor()
    agent.drift_monitor = monitor
    agent.planner = DriftAwareRulePlanner(
        agent.planner,
        monitor,
        ReplanningPolicy(strategy="aggressive"),
    )
    agent.executor = _RecordingExecutor()

    trace = agent.run("Check backend health and run ghz_5")
    tool_names = [step.action for step in trace.steps if step.action]

    assert tool_names.count("get_backend_health") >= 2
    assert tool_names.count("run_circuit") >= 2
    assert trace.state.drift_state["is_drifting"] is True
    assert agent.planner.replan_count == 1


def test_rabi_only_task_does_not_run_unrequested_circuit_benchmark():
    agent = ReActAgent(
        FakeBackendAdapter("FakeBrisbane"),
        provider="mock",
        use_memory=False,
        verbose=False,
        target_fidelity=0.0,
        max_tool_calls=10,
    )

    trace = agent.run(
        "Run a Rabi oscillation experiment at 5.0 GHz and fit the pi-pulse amplitude"
    )
    tool_names = [step.action for step in trace.steps if step.action]

    assert "rabi_experiment" in tool_names
    assert "fit_rabi" in tool_names
    assert "run_circuit" not in tool_names
