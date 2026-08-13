"""Executable handoff contracts for packaging and developer entry points."""

from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_handoff_files_and_make_targets_exist():
    required_files = [
        ".env.example",
        "INSTALL.md",
        "docs/HANDOFF.md",
        "docs/ARCHITECTURE.md",
        "scripts/doctor.py",
        "scripts/smoke_test.py",
    ]
    for relative in required_files:
        assert (ROOT / relative).is_file(), relative

    makefile = (ROOT / "Makefile").read_text()
    targets = {match.group(1) for match in re.finditer(r"^([a-z][a-z0-9-]*):", makefile, re.MULTILINE)}
    assert {"setup", "doctor", "smoke-test", "test-handoff", "test-fast", "test", "lint"} <= targets


def test_package_discovery_includes_runtime_subpackages():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    setuptools = config["tool"]["setuptools"]
    assert "packages" in setuptools
    patterns = set(setuptools["packages"]["find"]["include"])
    assert {"agent*", "tools*", "backends*", "benchmark*"} <= patterns


def test_env_example_contains_names_but_no_credentials():
    entries = {}
    for raw_line in (ROOT / ".env.example").read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        entries[key] = value
    assert {"DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"} <= entries.keys()
    assert all(value in {"", "0"} for value in entries.values())


def test_active_python_sources_do_not_hardcode_server_checkout():
    roots = [
        "agent", "advisor", "backends", "bench", "benchmark", "data",
        "dynamics", "eval", "tools", "tests", "web",
    ]
    candidates = [ROOT / "cli.py", ROOT / "conftest.py"]
    for directory in roots:
        candidates.extend((ROOT / directory).rglob("*.py"))
    forbidden_checkout = "/home/" + "wangshuchang/quantumgpt"
    offenders = [
        str(path.relative_to(ROOT))
        for path in candidates
        if forbidden_checkout in path.read_text(errors="ignore")
    ]
    assert offenders == []


def test_tracked_markdown_is_minimal():
    git_metadata = ROOT / ".git"
    if not git_metadata.exists():
        return
    import subprocess

    tracked = subprocess.check_output(
        ["git", "ls-files", "*.md"], cwd=ROOT, text=True
    ).splitlines()
    assert set(tracked) == {
        "BACKENDS.md",
        "INSTALL.md",
        "README.md",
        "docs/ARCHITECTURE.md",
        "docs/CCF_B_RESEARCH_TASKS_CN.md",
        "docs/HANDOFF.md",
        "docs/README.md",
        "web/README.md",
    }
