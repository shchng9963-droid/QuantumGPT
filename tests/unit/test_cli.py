"""Tests for the qgpt CLI (Day 12 MVP)."""

import json
import pytest
from click.testing import CliRunner
from cli import main


@pytest.fixture
def runner():
    return CliRunner()


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
