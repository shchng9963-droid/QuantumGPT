"""Tests for human-readable agent trace reporting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.react import AgentTrace, TraceDiagnostics, TraceStep
from agent.reporting import TraceReporter
from agent.state import AgentState, ArtifactType


def _sample_trace() -> AgentTrace:
    trace = AgentTrace(
        user_prompt="Run a Rabi experiment and fit pi pulse",
        model="react-rule-planner-v1",
        provider="mock",
        backend="FakeBrisbane",
    )
    trace.elapsed_seconds = 1.234
    trace.final_answer = "The pi pulse amplitude is 5.0 MHz."
    trace.diagnostics = TraceDiagnostics(
        requested_provider="mock",
        resolved_provider="mock",
        model="react-rule-planner-v1",
        final_answer_length=len(trace.final_answer),
    )
    trace.steps.append(
        TraceStep(
            step_num=0,
            thought="Need pulse sweep data first.",
            action="rabi_experiment",
            action_input={"qubit_freq_ghz": 5.0, "n_points": 7},
            observation='{"pi_amplitude_mhz": 5.0, "max_population": 0.99}',
            timestamp=0.1,
        )
    )
    trace.steps.append(
        TraceStep(
            step_num=1,
            thought="Fit the oscillation.",
            action="fit_rabi",
            action_input={"amplitudes": [0.0, 0.005], "populations": [0.0, 1.0]},
            observation='{"pi_amplitude_mhz": 5.0, "r_squared": 0.998, "fit_successful": true}',
            timestamp=0.2,
        )
    )
    return trace


def test_trace_reporter_generates_markdown_summary_with_physics_numbers():
    report = TraceReporter(_sample_trace()).to_markdown()

    assert "# QuantumGPT Agent Report" in report
    assert "Run a Rabi experiment" in report
    assert "Backend: FakeBrisbane" in report
    assert "rabi_experiment" in report
    assert "fit_rabi" in report
    assert "pi_amplitude_mhz: 5.0" in report
    assert "r_squared: 0.998" in report
    assert "Final answer" in report


def test_trace_reporter_generates_short_talk_track_for_demo():
    talk_track = TraceReporter(_sample_trace()).to_talk_track()

    assert len(talk_track.splitlines()) >= 3
    assert "I asked QuantumGPT to run" in talk_track
    assert "Rabi" in talk_track
    assert "5.0 MHz" in talk_track
    assert "traceable" in talk_track.lower()


def test_trace_reporter_writes_report_and_trace_json(tmp_path):
    trace = _sample_trace()
    reporter = TraceReporter(trace)

    paths = reporter.write_bundle(tmp_path, prefix="unit_rabi")

    assert paths["report"].name == "unit_rabi_report.md"
    assert paths["trace"].name == "unit_rabi_trace.json"
    assert paths["talk_track"].name == "unit_rabi_talk_track.txt"
    assert "QuantumGPT Agent Report" in paths["report"].read_text()
    assert "pi_amplitude_mhz" in paths["trace"].read_text()
    assert "traceable" in paths["talk_track"].read_text().lower()


def test_trace_reporter_highlights_tool_errors_and_missing_final_answer():
    trace = AgentTrace(user_prompt="unsafe request", model="m", provider="mock", backend="Fake")
    trace.steps.append(
        TraceStep(
            step_num=0,
            action="rabi_experiment",
            observation='{"error": "blocked_by_safety_policy", "reason": "amp_max too high", "safety": {"status": "blocked"}}',
        )
    )
    trace.diagnostics.no_final_answer = True

    report = TraceReporter(trace).to_markdown()

    assert "Tool issues" in report
    assert "blocked_by_safety_policy" in report
    assert "amp_max too high" in report
    assert "No final answer" in report


def test_trace_reporter_marks_malformed_observation_as_unparsed():
    trace = AgentTrace(user_prompt="broken tool", model="m", provider="mock", backend="Fake")
    trace.steps.append(TraceStep(step_num=0, action="fit_rabi", observation="not-json"))

    report = TraceReporter(trace).to_markdown()

    assert "Unparsed observations" in report
    assert "fit_rabi" in report


def test_trace_reporter_surfaces_reliability_diagnostics():
    trace = _sample_trace()
    trace.diagnostics.repeated_tool_count = 2
    trace.diagnostics.invalid_tool_call_count = 1
    trace.diagnostics.safety_block_count = 1

    report = TraceReporter(trace).to_markdown()

    assert "Reliability warnings" in report
    assert "repeated_tool_count: 2" in report
    assert "invalid_tool_call_count: 1" in report
    assert "safety_block_count: 1" in report


def test_trace_reporter_surfaces_state_memory_and_safety_context():
    trace = _sample_trace()
    trace.state = AgentState(
        task_id="demo-rabi",
        user_prompt=trace.user_prompt,
        backend_name="FakeBrisbane",
        backend_snapshot_id="snapshot-123",
        current_step=2,
        memory_context=[{
            "id": "mem-low-fidelity",
            "task": "previous GHZ run",
            "outcome": "low fidelity after drift",
        }],
        safety_status={
            "mode": "dry_run",
            "last_status": "dry_run",
            "dry_run_count": 1,
        },
        budget_state={"calls_used": 2, "max_calls": 10},
    )
    trace.state.artifacts.add_artifact(
        ArtifactType.BACKEND_SNAPSHOT,
        created_at_step=0,
        artifact_id="backend-snapshot",
    )
    trace.diagnostics.safety_dry_run_count = 1

    report = TraceReporter(trace).to_markdown()

    assert "## Runtime state" in report
    assert "backend_snapshot: 1" in report
    assert "snapshot-123" in report
    assert "## Memory context" in report
    assert "low fidelity after drift" in report
    assert "## Safety and dry-run" in report
    assert "dry_run_count: 1" in report
    assert "safety_dry_run_count: 1" in report
