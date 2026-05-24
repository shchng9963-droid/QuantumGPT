# Day 12: MVP CLI — `qgpt` One-Liner Interface

## Goal
Build a production-quality CLI so users can run quantum simulations with a
single command: `qgpt simulate ghz --shots 8192 --backend FakeBrisbane`.

## What Was Done

### 1. pyproject.toml + Editable Install
Created `pyproject.toml` with `[project.scripts]` entry point. After
`pip install -e .`, the `qgpt` command is available system-wide.

### 2. CLI Commands (cli.py, ~300 lines)

| Command | Description | Example |
|---------|-------------|---------|
| `qgpt simulate <circuit>` | Run circuit, get fidelity | `qgpt simulate ghz --shots 8192` |
| `qgpt health` | Show backend T1/T2/errors/drift | `qgpt health -b FakeKyiv` |
| `qgpt list` | List available benchmark circuits | `qgpt list` |
| `qgpt diagnose` | Run diagnostics + suggestions | `qgpt diagnose` |
| `qgpt agent "<prompt>"` | Natural-language agent mode | `qgpt agent "Check health and run GHZ-5"` |

Features:
- **Short aliases**: `ghz`, `qft`, `bv`, `vqe`, `qaoa` → auto-resolve
- **4 backends**: FakeBrisbane, FakeKyiv, FakeSherbrooke, FakeTorino
- **Rich output**: color-coded fidelity (green ≥0.9, yellow ≥0.7, red <0.7)
- **JSON mode**: `-j` flag for all commands → machine-readable output
- **Click framework**: proper help text, validation, version display

### 3. Test Suite (14 tests, all passing)

| Test class | Count | Coverage |
|------------|-------|----------|
| TestVersion | 1 | --version flag |
| TestHealth | 2 | rich + JSON output |
| TestList | 2 | rich + JSON output |
| TestSimulate | 5 | alias, full name, JSON, unknown circuit |
| TestDiagnose | 2 | rich + JSON output |
| TestAgent | 2 | mock mode + JSON output |

### 4. Full Test Suite: 110 passed, 1 skipped, 0 failures

### 5. Demo Figures

| Figure | Description |
|--------|-------------|
| `fidelity_comparison.png` | 5 circuits on FakeBrisbane: GHZ=0.924, QFT=0.999, BV=0.946, VQE=0.980, QAOA=0.957 |
| `backend_comparison.png` | GHZ-5 across 4 backends: Brisbane=0.927, Kyiv=0.949, Sherbrooke=0.926, Torino=0.916 |

## Files Created/Modified

| File | Action | Lines |
|------|--------|-------|
| `pyproject.toml` | Created | 32 |
| `cli.py` | Created | ~300 |
| `tests/unit/test_cli.py` | Created | ~100 |
| `demos/day12/gen_figures.py` | Created | ~120 |
| `demos/day12/fidelity_comparison.png` | Generated | — |
| `demos/day12/backend_comparison.png` | Generated | — |

## Key Design Decisions
- **Click over argparse**: cleaner subcommand structure, built-in help generation
- **Rich for terminal output**: tables and color without external deps
- **JSON mode on every command**: enables piping into `jq` or downstream scripts
- **Alias system**: physicists type `qgpt simulate ghz`, not `ghz_5`
- **Agent subcommand**: preserves the full agentic loop as a CLI option
