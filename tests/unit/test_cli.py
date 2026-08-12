"""Tests for the qgpt CLI (Day 12 MVP)."""

import json
import pytest
from click.testing import CliRunner

from agent.run_persistence import load_run
from cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tmp_runs_dir(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setenv("QUANTUMGPT_RUNS_DIR", str(root))
    return root


class TestVersion:
    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestHealth:
    def test_health_default(self, runner):
        result = runner.invoke(main, ["health"])
        assert result.exit_code == 0
        assert "FakeBrisbane" in result.output
        assert "T1" in result.output

    def test_health_json(self, runner):
        result = runner.invoke(main, ["health", "-j"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["backend"] == "FakeBrisbane"
        assert data["num_qubits"] == 127
        assert "avg_t1_us" in data


class TestList:
    def test_list_circuits(self, runner):
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        assert "ghz_5" in result.output
        assert "qft_4" in result.output

    def test_list_json(self, runner):
        result = runner.invoke(main, ["list", "-j"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "hand_written" in data
        assert "ghz_5" in data["hand_written"]


class TestSimulate:
    def test_simulate_ghz(self, runner):
        result = runner.invoke(main, ["simulate", "ghz", "--shots", "1024"])
        assert result.exit_code == 0
        assert "Fidelity" in result.output
        assert "ghz_5" in result.output

    def test_simulate_json(self, runner):
        result = runner.invoke(main, ["simulate", "ghz", "-s", "1024", "-j"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["circuit"] == "ghz_5"
        assert data["shots"] == 1024
        assert 0 < data["fidelity"] <= 1.0

    def test_simulate_alias_qft(self, runner):
        result = runner.invoke(main, ["simulate", "qft", "-s", "1024", "-j"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["circuit"] == "qft_4"

    def test_simulate_full_name(self, runner):
        result = runner.invoke(main, ["simulate", "bv_5", "-s", "1024", "-j"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["circuit"] == "bv_5"

    def test_simulate_unknown_circuit(self, runner):
        result = runner.invoke(main, ["simulate", "nonexistent", "-s", "1024"])
        assert result.exit_code != 0


class TestDiagnose:
    def test_diagnose_default(self, runner):
        result = runner.invoke(main, ["diagnose"])
        assert result.exit_code == 0
        assert "Severity" in result.output

    def test_diagnose_json(self, runner):
        result = runner.invoke(main, ["diagnose", "-j"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "severity" in data
        assert "suggestions" in data


class TestAgent:
    def test_agent_mock(self, runner):
        result = runner.invoke(main, ["agent", "Check health"])
        assert result.exit_code == 0
        assert "QuantumGPT" in result.output

    def test_agent_json(self, runner):
        result = runner.invoke(main, ["agent", "Check health", "-j"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "final_answer" in data
        assert data["model"] == "rule-planner-v1"


class TestRunOrchestrator:
    def test_run_orchestrator_persists_graph_state(self, runner, tmp_runs_dir):
        result = runner.invoke(
            main,
            [
                "run",
                "ghz_5",
                "--system",
                "QuantumGPT-Orchestrator",
                "--provider",
                "mock",
                "-j",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        run = load_run(payload["task_id"])

        assert run["result"]["system"] == "QuantumGPT-Orchestrator"
        assert "post_state" in run
        assert run["post_state"]["done"] is True
        assert "verification" in run["post_state"]
        assert run["result"]["termination_reason"] == run["post_state"]["termination_reason"]

    def test_run_orch_noverifier_uses_stub_verifier(self, runner, tmp_runs_dir):
        result = runner.invoke(
            main,
            [
                "run",
                "ghz_5",
                "--system",
                "QuantumGPT-Orch-NoVerifier",
                "--provider",
                "mock",
                "-j",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        run = load_run(payload["task_id"])

        assert "post_state" in run
        assert run["post_state"]["verification"]["reason"] == "stub verifier always accepts"
        assert run["result"]["plan_revision"] >= 1

    def test_show_displays_orchestrator_verifier_metadata(self, runner, tmp_runs_dir):
        run_result = runner.invoke(
            main,
            [
                "run",
                "ghz_5",
                "--system",
                "QuantumGPT-Orch-NoVerifier",
                "--provider",
                "mock",
                "-j",
            ],
        )
        assert run_result.exit_code == 0
        task_id = json.loads(run_result.output)["task_id"]

        show_result = runner.invoke(main, ["show", task_id])

        assert show_result.exit_code == 0
        assert "termination reason" in show_result.output
        assert "verifier reason" in show_result.output
        assert "plan revision" in show_result.output
        assert "stub verifier always accepts" in show_result.output
        assert "satisfied" in show_result.output

    def test_show_surfaces_available_b10_artifacts(self, runner, tmp_runs_dir):
        from agent.run_persistence import TaskSpec, save_task_spec, run_dir

        spec = TaskSpec(circuit="ghz_5", backend="FakeBrisbane", system="QuantumGPT-Orchestrator")
        task_id, _ = save_task_spec(spec)
        d = run_dir(task_id)
        (d / "pre_graph_state.json").write_text(json.dumps({"phase": "pre"}))
        (d / "post_graph_state.json").write_text(json.dumps({"phase": "post"}))
        (d / "oracle_row.json").write_text(json.dumps({"profile": "severe_sudden"}))

        show_result = runner.invoke(main, ["show", task_id])

        assert show_result.exit_code == 0
        assert "available artifacts" in show_result.output
        assert "pre_state" in show_result.output
        assert "post_state" in show_result.output
        assert "oracle_row" in show_result.output

    def test_runs_json_exposes_orchestrator_summary_fields(self, runner, tmp_runs_dir):
        run_result = runner.invoke(
            main,
            [
                "run",
                "ghz_5",
                "--system",
                "QuantumGPT-Orch-NoVerifier",
                "--provider",
                "mock",
                "-j",
            ],
        )
        assert run_result.exit_code == 0
        task_id = json.loads(run_result.output)["task_id"]

        runs_result = runner.invoke(main, ["runs", "-j"])

        assert runs_result.exit_code == 0
        rows = json.loads(runs_result.output)
        row = next(r for r in rows if r["task_id"] == task_id)
        summary = row["summary"]

        assert summary["verifier_reason"] == "stub verifier always accepts"
        assert summary["termination_reason"] == "satisfied"
        assert summary["plan_revision"] >= 1
