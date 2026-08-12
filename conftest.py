"""Project-wide pytest configuration.

Detects when pytest is being invoked with the wrong Python interpreter
(e.g., system miniforge instead of the project .venv) and fails loudly
with a clear remediation message instead of producing 4 baffling
"collection errors".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"


def _running_in_project_venv() -> bool:
    """Return True if sys.executable lives inside .venv/."""
    try:
        exe = Path(sys.executable).resolve()
        return VENV_DIR in exe.parents
    except Exception:
        return False


def _critical_imports_ok() -> tuple[bool, list[str]]:
    """Try to import packages the test suite cannot work without."""
    missing = []
    for mod in ["langgraph", "qiskit", "click"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return (not missing), missing


def pytest_configure(config: pytest.Config) -> None:
    """Early-exit with friendly message if the wrong Python is in use.

    Without this, missing langgraph causes 4 'collection errors' which
    look like real test failures. Fail-fast with a one-line fix instead.
    """
    if os.environ.get("QGPT_SKIP_VENV_CHECK"):
        return

    in_venv = _running_in_project_venv()
    ok, missing = _critical_imports_ok()

    if in_venv and ok:
        return  # Healthy.

    msg_lines = [
        "",
        "=" * 70,
        "QuantumGPT pytest environment check FAILED.",
        "=" * 70,
        f"  sys.executable = {sys.executable}",
        f"  in .venv?      = {in_venv}",
        f"  missing deps   = {missing or 'none'}",
        "",
        "Likely cause: pytest is being run from a system Python that does",
        "not have the project dependencies installed (e.g. langgraph).",
        "",
        "Fix:",
        f"  cd {PROJECT_ROOT}",
        "  source .venv/bin/activate    # or: make test",
        "  pytest tests/",
        "",
        "Or set QGPT_SKIP_VENV_CHECK=1 to bypass this check (not recommended).",
        "=" * 70,
        "",
    ]
    pytest.exit("\n".join(msg_lines), returncode=2)
