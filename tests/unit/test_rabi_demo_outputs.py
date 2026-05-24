"""Smoke tests for physics-facing demo artifacts."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demos.rabi_demo import run_rabi_demo


def test_rabi_demo_writes_report_trace_and_talk_track(tmp_path):
    result = run_rabi_demo(
        freq_ghz=5.0,
        pulse_shape="square",
        duration_ns=100.0,
        save_dir=str(tmp_path),
    )

    report_path = Path(result["report_path"])
    trace_path = Path(result["trace_path"])
    talk_track_path = Path(result["talk_track_path"])

    assert report_path.exists()
    assert trace_path.exists()
    assert talk_track_path.exists()
    assert "QuantumGPT Agent Report" in report_path.read_text()
    report_text = report_path.read_text()
    assert "QuantumGPT Agent Report" in report_text
    assert "Runtime state" in report_text
    assert "Memory context" in report_text
    assert "Safety and dry-run" in report_text
    assert "safety_dry_run_count" in report_text
    assert "traceable" in talk_track_path.read_text().lower()

    trace = json.loads(trace_path.read_text())
    assert trace["diagnostics"]["resolved_provider"] == "mock"
    assert trace["steps"]
    assert "rabi_experiment" in {s.get("action") for s in trace["steps"]}
    assert "fit_rabi" in {s.get("action") for s in trace["steps"]}
