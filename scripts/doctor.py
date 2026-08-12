#!/usr/bin/env python3
"""Diagnose whether the current interpreter can safely run QuantumGPT."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODULES = {
    "qiskit": "qiskit",
    "qiskit_aer": "qiskit-aer",
    "numpy": "numpy",
    "scipy": "scipy",
    "click": "click",
    "rich": "rich",
    "langgraph": "langgraph",
}
OPTIONAL_KEYS = (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
)


def _inside_project(path: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def run_checks() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if sys.version_info[:2] != (3, 11):
        errors.append(f"Python 3.11 required; current interpreter is {sys.version.split()[0]}")

    expected_python = ROOT / ".venv" / "bin" / "python"
    if not _inside_project(sys.executable):
        errors.append(f"Interpreter is outside project .venv: {sys.executable}")
    elif Path(sys.executable).resolve() != expected_python.resolve():
        warnings.append(f"Interpreter differs from expected path: {sys.executable}")

    for module_name, distribution in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
            version = importlib.metadata.version(distribution)
            print(f"[ok] {distribution} {version}")
        except Exception as exc:  # diagnosis should report all missing dependencies
            errors.append(f"Cannot import {module_name}: {exc}")

    try:
        from agent.planners.rule import ReActRulePlanner as CanonicalPlanner
        from agent.react import AgentTrace, ReActRulePlanner
        from agent.trace import AgentTrace as CanonicalTrace
        from tools.quantum_tools import TOOL_DEFINITIONS as PublicSchemas
        from tools.registry import TOOL_RUNTIME_SPECS, validate_tool_registry
        from tools.schemas import TOOL_DEFINITIONS

        validate_tool_registry(TOOL_DEFINITIONS)
        if PublicSchemas is not TOOL_DEFINITIONS:
            errors.append("tools.quantum_tools does not re-export canonical TOOL_DEFINITIONS")
        if AgentTrace is not CanonicalTrace:
            errors.append("agent.react does not re-export canonical AgentTrace")
        if ReActRulePlanner is not CanonicalPlanner:
            errors.append("agent.react does not re-export canonical ReActRulePlanner")
        print(f"[ok] tool registry: {len(TOOL_DEFINITIONS)} schemas / {len(TOOL_RUNTIME_SPECS)} runtimes")
    except Exception as exc:
        errors.append(f"Runtime boundary check failed: {exc}")

    qgpt = ROOT / ".venv" / "bin" / "qgpt"
    if not qgpt.exists():
        errors.append(".venv/bin/qgpt is missing; run `make install`")
    elif shutil.which(str(qgpt)) is None:
        warnings.append(f"qgpt exists but is not executable: {qgpt}")

    if not any(os.environ.get(key) for key in OPTIONAL_KEYS):
        warnings.append("No LLM API key detected; offline mock mode remains available")

    runs_dir = os.environ.get("QUANTUMGPT_RUNS_DIR")
    if runs_dir:
        target = Path(runs_dir).expanduser()
        parent = target if target.exists() else target.parent
        if not parent.exists() or not os.access(parent, os.W_OK):
            errors.append(f"QUANTUMGPT_RUNS_DIR is not writable: {target}")

    return errors, warnings


def main() -> int:
    print("=== QuantumGPT environment doctor ===")
    print(f"project: {ROOT}")
    print(f"python:  {sys.executable} ({sys.version.split()[0]})")
    errors, warnings = run_checks()
    for item in warnings:
        print(f"[warn] {item}")
    for item in errors:
        print(f"[error] {item}")
    if errors:
        print(f"FAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"PASSED: 0 errors, {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
