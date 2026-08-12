from __future__ import annotations

from eval.benchmark_pilot import (
    DEFAULT_CIRCUITS,
    DEFAULT_ORACLE_JSONL,
    build_pilot_command,
    load_env_file,
)


def test_default_oracle_jsonl_points_to_oracle_results_file():
    assert DEFAULT_ORACLE_JSONL.endswith("oracle_results.jsonl")


def test_default_circuits_use_score_focused_oracle_canonical_names():
    assert DEFAULT_CIRCUITS == [
        "QAOA-6",
        "qft_4",
        "VQE_SU2-4",
        "GraphState-5",
        "vqe_4",
    ]


def test_load_env_file_reads_key_value_pairs(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("DEEPSEEK_API_KEY=test-key\nOTHER_VAR=hello\n# comment\n")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OTHER_VAR", raising=False)

    loaded = load_env_file(env_path)

    assert loaded["DEEPSEEK_API_KEY"] == "test-key"
    assert loaded["OTHER_VAR"] == "hello"


def test_load_env_file_does_not_override_existing_env(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("DEEPSEEK_API_KEY=file-key\n")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "existing-key")

    loaded = load_env_file(env_path)

    assert loaded["DEEPSEEK_API_KEY"] == "existing-key"


def test_build_pilot_command_uses_calibrated_b10_defaults(tmp_path):
    out_dir = tmp_path / "pilot_out"
    cmd = build_pilot_command(
        oracle_jsonl="eval/results/v3_oracle_full/oracle_results.jsonl",
        out_dir=out_dir,
    )

    joined = " ".join(cmd)
    assert "python" in cmd[0]
    assert cmd[1].endswith("eval/benchmark_pilot.py") is False
    assert cmd[1].endswith("eval/run_orchestrator_b10.py")
    assert "--systems" in cmd
    assert "QuantumGPT-Orchestrator,QuantumGPT-Orch-NoVerifier" in cmd
    assert "--profiles" in cmd
    assert "mild_gradual" in joined
    assert "--circuits" in cmd
    assert "QAOA-6,qft_4,VQE_SU2-4,GraphState-5,vqe_4" in cmd
    assert "--seeds" in cmd
    assert "0,1,2,3,4" in cmd
    assert "--max-tool-calls" in cmd
    assert "6" in cmd
    assert "--max-seconds" in cmd
    assert "60" in cmd
    assert "--max-usd" in cmd
    assert "2.0" in cmd
    assert str(out_dir) in cmd


def test_build_pilot_command_supports_small_smoke_override(tmp_path):
    out_dir = tmp_path / "pilot_smoke"
    cmd = build_pilot_command(
        oracle_jsonl="oracle.jsonl",
        out_dir=out_dir,
        circuits=["ghz_5", "qaoa_4"],
        profiles=["severe_sudden"],
        seeds=[0],
        max_tasks=5,
        max_usd=0.5,
    )

    assert "ghz_5,qaoa_4" in cmd
    assert "severe_sudden" in cmd
    assert "0" in cmd
    assert "--max-tasks" in cmd
    assert "5" in cmd
    assert "0.5" in cmd
