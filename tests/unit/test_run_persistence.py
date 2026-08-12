"""Tests for task_id and run persistence (P0-1)."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from agent.run_persistence import (
    TaskSpec,
    compute_task_id,
    runs_root,
    run_dir,
    save_task_spec,
    save_result,
    append_trace,
    save_artifact,
    list_runs,
    load_run,
    load_spec,
    git_commit_short,
)


# ──────────────────────────────────────────────────────────────────────
# Use a tmp dir to avoid clobbering the user's actual ~/.quantumgpt/
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _tmp_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUMGPT_RUNS_DIR", str(tmp_path / "runs"))
    yield


def test_runs_root_respects_env_var(tmp_path):
    expected = (tmp_path / "runs").resolve()
    assert runs_root() == expected


def test_compute_task_id_deterministic():
    """Same spec + same commit → same id."""
    spec = TaskSpec(circuit="ghz_5", backend="FakeBrisbane", target_fidelity=0.85)
    a = compute_task_id(spec, code_commit="abc123")
    b = compute_task_id(spec, code_commit="abc123")
    assert a == b
    assert len(a) == 12


def test_compute_task_id_changes_with_circuit():
    spec1 = TaskSpec(circuit="ghz_5", backend="FakeBrisbane", target_fidelity=0.85)
    spec2 = TaskSpec(circuit="qft_4", backend="FakeBrisbane", target_fidelity=0.85)
    assert compute_task_id(spec1, "x") != compute_task_id(spec2, "x")


def test_compute_task_id_changes_with_commit():
    spec = TaskSpec(circuit="ghz_5", backend="FakeBrisbane", target_fidelity=0.85)
    assert compute_task_id(spec, "abc") != compute_task_id(spec, "def")


def test_compute_task_id_extras_order_independent():
    s1 = TaskSpec(circuit="ghz_5", extras={"a": 1, "b": 2})
    s2 = TaskSpec(circuit="ghz_5", extras={"b": 2, "a": 1})
    assert compute_task_id(s1, "x") == compute_task_id(s2, "x")


def test_save_task_spec_creates_dir_and_files(tmp_path):
    spec = TaskSpec(circuit="ghz_5", backend="FakeBrisbane", target_fidelity=0.85)
    task_id, d = save_task_spec(spec)
    assert d.exists()
    assert (d / "task_spec.json").exists()
    assert (d / "metadata.json").exists()
    saved_spec = json.loads((d / "task_spec.json").read_text())
    assert saved_spec["circuit"] == "ghz_5"
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["task_id"] == task_id
    assert meta["status"] == "running"
    assert "started_at" in meta


def test_save_task_spec_records_git_dirty_metadata(monkeypatch):
    monkeypatch.setattr("agent.run_persistence.git_commit_short", lambda: "abc12345")
    monkeypatch.setattr(
        "agent.run_persistence.git_working_tree_status",
        lambda: {
            "git_dirty": True,
            "git_diff_summary": ["M agent/run_persistence.py", "?? eval/pilot.py"],
        },
    )

    task_id, d = save_task_spec(TaskSpec(circuit="ghz_5"))
    meta = json.loads((d / "metadata.json").read_text())

    assert meta["task_id"] == task_id
    assert meta["code_commit"] == "abc12345"
    assert meta["git_dirty"] is True
    assert meta["git_diff_summary"] == ["M agent/run_persistence.py", "?? eval/pilot.py"]


def test_save_result_updates_metadata_status():
    spec = TaskSpec(circuit="ghz_5", target_fidelity=0.85)
    task_id, d = save_task_spec(spec)
    save_result(task_id, {
        "best_score": 0.87,
        "target_fidelity": 0.85,
        "tool_calls": 4,
        "total_cost_usd": 0.018,
    })
    assert (d / "result.json").exists()
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["status"] == "completed"
    assert meta["summary"]["best_score"] == 0.87
    assert "finished_at" in meta
    assert meta["walltime_seconds"] >= 0


def test_save_result_failed_status():
    spec = TaskSpec(circuit="ghz_5")
    task_id, _ = save_task_spec(spec)
    save_result(task_id, {"error": "boom"}, status="failed")
    meta = json.loads((run_dir(task_id) / "metadata.json").read_text())
    assert meta["status"] == "failed"


def test_append_trace_appends_jsonl():
    spec = TaskSpec(circuit="ghz_5")
    task_id, d = save_task_spec(spec)
    append_trace(task_id, {"step": 1, "action": "get_backend_health"})
    append_trace(task_id, {"step": 2, "action": "run_circuit"})
    lines = (d / "trace.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 1
    assert json.loads(lines[1])["action"] == "run_circuit"


def test_save_artifact_writes_named_json():
    spec = TaskSpec(circuit="ghz_5")
    task_id, d = save_task_spec(spec)
    save_artifact(task_id, "pre_state", {"plan": ["a", "b"]})
    assert (d / "pre_state.json").exists()
    loaded = json.loads((d / "pre_state.json").read_text())
    assert loaded["plan"] == ["a", "b"]


def test_load_run_round_trip():
    spec = TaskSpec(circuit="qft_4", backend="FakeKyiv", target_fidelity=0.9)
    task_id, _ = save_task_spec(spec)
    append_trace(task_id, {"step": 1})
    save_artifact(task_id, "pre_state", {"x": 1})
    save_artifact(task_id, "post_state", {"x": 2})
    save_result(task_id, {"best_score": 0.91})

    loaded = load_run(task_id)
    assert loaded["task_id"] == task_id
    assert loaded["spec"]["circuit"] == "qft_4"
    assert loaded["spec"]["backend"] == "FakeKyiv"
    assert loaded["result"]["best_score"] == 0.91
    assert len(loaded["trace"]) == 1
    assert loaded["pre_state"]["x"] == 1
    assert loaded["post_state"]["x"] == 2
    assert "git_dirty" in loaded["metadata"]
    assert "git_diff_summary" in loaded["metadata"]


def test_load_run_accepts_b10_graph_state_aliases():
    spec = TaskSpec(circuit="ghz_5", backend="FakeBrisbane", target_fidelity=0.85)
    task_id, d = save_task_spec(spec)
    (d / "pre_graph_state.json").write_text(json.dumps({"phase": "pre"}))
    (d / "post_graph_state.json").write_text(json.dumps({"phase": "post"}))
    (d / "oracle_row.json").write_text(json.dumps({"profile": "severe_sudden"}))

    loaded = load_run(task_id)

    assert loaded["pre_state"]["phase"] == "pre"
    assert loaded["post_state"]["phase"] == "post"
    assert loaded["oracle_row"]["profile"] == "severe_sudden"


def test_load_spec_round_trip():
    spec = TaskSpec(
        circuit="ghz_5",
        backend="FakeBrisbane",
        target_fidelity=0.85,
        seed=7,
        max_tool_calls=20,
        extras={"foo": "bar"},
    )
    task_id, _ = save_task_spec(spec)
    loaded = load_spec(task_id)
    assert loaded.circuit == "ghz_5"
    assert loaded.seed == 7
    assert loaded.max_tool_calls == 20
    assert loaded.extras == {"foo": "bar"}


def test_list_runs_filters():
    s1 = TaskSpec(circuit="ghz_5", backend="FakeBrisbane")
    s2 = TaskSpec(circuit="qft_4", backend="FakeBrisbane")
    s3 = TaskSpec(circuit="ghz_5", backend="FakeKyiv")
    save_task_spec(s1)
    save_task_spec(s2)
    save_task_spec(s3)

    all_runs = list_runs()
    assert len(all_runs) == 3

    # filter by backend
    brisbane = list_runs(backend="FakeBrisbane")
    assert len(brisbane) == 2
    assert all(r["spec"]["backend"] == "FakeBrisbane" for r in brisbane)

    # filter by circuit
    ghz = list_runs(circuit="ghz_5")
    assert len(ghz) == 2

    # filter by both
    both = list_runs(backend="FakeBrisbane", circuit="ghz_5")
    assert len(both) == 1


def test_list_runs_status_filter():
    s1 = TaskSpec(circuit="ghz_5", backend="FakeBrisbane")
    s2 = TaskSpec(circuit="qft_4", backend="FakeBrisbane")
    tid1, _ = save_task_spec(s1)
    tid2, _ = save_task_spec(s2)

    save_result(tid1, {"best_score": 0.9})  # → completed
    # tid2 stays "running"

    completed = list_runs(status="completed")
    assert len(completed) == 1
    assert completed[0]["task_id"] == tid1

    running = list_runs(status="running")
    assert len(running) == 1
    assert running[0]["task_id"] == tid2


def test_list_runs_sorted_newest_first():
    """Save 3 specs, force their started_at, ensure sorted desc."""
    tids = []
    for i in range(3):
        spec = TaskSpec(circuit=f"c{i}")
        tid, d = save_task_spec(spec)
        # Manually rewrite metadata with monotone increasing timestamp
        meta_path = d / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["started_at"] = 1000 + i
        meta_path.write_text(json.dumps(meta))
        tids.append(tid)

    runs = list_runs()
    assert [r["task_id"] for r in runs] == tids[::-1]  # newest first


def test_list_runs_limit():
    for i in range(5):
        save_task_spec(TaskSpec(circuit=f"c{i}"))
    assert len(list_runs(limit=2)) == 2


def test_list_runs_empty_dir():
    """No runs/ subdirectory yet → empty list, not crash."""
    assert list_runs() == []


def test_load_run_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_run("does_not_exist")


def test_git_commit_short_returns_str():
    """In a git repo, returns 8-char hex; otherwise returns ''."""
    sha = git_commit_short()
    # Either we're in a repo (8 hex chars) or we're not (empty)
    assert sha == "" or (len(sha) == 8 and all(c in "0123456789abcdef" for c in sha))
