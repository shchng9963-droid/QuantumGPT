"""Tests for QC-Agent-Bench runner engineering controls."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark import runner


class DummyBackend:
    name = "FakeTest"
    num_qubits = 5


class DummyTrace:
    num_tool_calls = 1
    steps = [SimpleNamespace(action="get_backend_health", fidelity_observed=None, observation="{}")]
    final_answer = "done"

    diagnostics = SimpleNamespace(
        requested_provider="deepseek",
        resolved_provider="openai",
        model="deepseek-chat",
        final_answer_length=4,
        no_final_answer=False,
        invalid_tool_call_count=0,
        malformed_json_count=0,
        repeated_tool_count=1,
        max_turns_exceeded=False,
        hallucinated_tool_names=[],
        to_dict=lambda: {
            "requested_provider": "deepseek",
            "resolved_provider": "openai",
            "model": "deepseek-chat",
            "final_answer_length": 4,
            "no_final_answer": False,
            "invalid_tool_call_count": 0,
            "malformed_json_count": 0,
            "repeated_tool_count": 1,
            "max_turns_exceeded": False,
            "hallucinated_tool_names": [],
        },
    )

    def to_dict(self, include_observations=False):
        return {
            "user_prompt": "p",
            "diagnostics": self.diagnostics.to_dict(),
            "steps": [{"action": "get_backend_health"}],
        }


class DummySystem:
    def run(self, prompt):
        return DummyTrace()


def test_make_systems_passes_real_llm_configuration_to_react_agent(monkeypatch):
    captured = []

    class CapturingAgent:
        provider = "deepseek"

        def __init__(self, backend, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(runner, "ReActAgent", CapturingAgent)

    systems = runner.make_systems(
        DummyBackend(),
        ["react_drift"],
        provider="deepseek",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        verbose=False,
    )

    assert list(systems) == ["react_drift"]
    assert captured == [
        {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
            "use_memory": True,
            "use_drift_aware": True,
            "verbose": False,
        }
    ]


def test_make_systems_rejects_silent_mock_fallback_for_real_provider(monkeypatch):
    class FallbackAgent:
        provider = "mock"

        def __init__(self, backend, **kwargs):
            pass

    monkeypatch.setattr(runner, "ReActAgent", FallbackAgent)

    try:
        runner.make_systems(DummyBackend(), ["react_drift"], provider="deepseek")
    except RuntimeError as exc:
        assert "requested provider 'deepseek'" in str(exc)
        assert "fell back to mock" in str(exc)
    else:
        raise AssertionError("Expected non-mock provider fallback to fail loudly")


def test_run_benchmark_filters_task_ids_and_max_tasks(monkeypatch):
    monkeypatch.setattr(runner, "make_backend", lambda task: DummyBackend())
    monkeypatch.setattr(runner, "make_systems", lambda backend, systems_list, **kwargs: {"react_drift": DummySystem()})

    results = runner.run_benchmark(
        tiers=[1],
        systems_list=["react_drift"],
        task_ids=["T1-01", "T1-06"],
        max_tasks=1,
        provider="mock",
        verbose=False,
    )

    assert len(results) == 1
    assert results[0].task_id == "T1-01"
    assert results[0].system == "react_drift"
    assert results[0].success is True


def test_run_benchmark_copies_trace_diagnostics_and_optional_trace(monkeypatch):
    monkeypatch.setattr(runner, "make_backend", lambda task: DummyBackend())
    monkeypatch.setattr(runner, "make_systems", lambda backend, systems_list, **kwargs: {"react_drift": DummySystem()})

    results = runner.run_benchmark(
        tiers=[1],
        systems_list=["react_drift"],
        task_ids=["T1-01"],
        provider="deepseek",
        model="deepseek-chat",
        verbose=False,
        save_traces=True,
    )

    result = results[0]
    assert result.requested_provider == "deepseek"
    assert result.resolved_provider == "openai"
    assert result.model == "deepseek-chat"
    assert result.final_answer_length == 4
    assert result.no_final_answer is False
    assert result.invalid_tool_call_count == 0
    assert result.malformed_json_count == 0
    assert result.repeated_tool_count == 1
    assert result.max_turns_exceeded is False
    assert result.hallucinated_tool_names == []
    assert result.trace["diagnostics"]["requested_provider"] == "deepseek"


def test_cli_passes_engineering_controls_to_run_benchmark(monkeypatch, tmp_path):
    captured = {}

    def fake_run_benchmark(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(runner, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(runner, "generate_report", lambda results: "REPORT")
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner.py",
            "--systems", "react_drift",
            "--tiers", "1,2",
            "--provider", "deepseek",
            "--model", "deepseek-chat",
            "--base-url", "https://api.deepseek.com",
            "--task-ids", "T1-01,T1-06",
            "--max-tasks", "1",
            "--save-traces",
            "--output", str(tmp_path / "results.json"),
        ],
    )

    runner.main()

    assert captured == {
        "tiers": [1, 2],
        "systems_list": ["react_drift"],
        "task_ids": ["T1-01", "T1-06"],
        "max_tasks": 1,
        "provider": "deepseek",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "verbose": False,
        "save_traces": True,
    }
