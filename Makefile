# QuantumGPT developer entry points. Every Python command uses project .venv.

VENV       := .venv
PYTHON     := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
QGPT       := $(VENV)/bin/qgpt
PROJECT_PY := python3.11

.PHONY: help setup install doctor smoke-test test-handoff test test-fast qgpt-help agent lint clean demo-rabi demo-drift demo-memory orchestrator-test

help:
	@echo "QuantumGPT developer commands"
	@echo "  make setup         Create/refresh .venv and install the project"
	@echo "  make doctor        Validate interpreter, dependencies and runtime boundaries"
	@echo "  make smoke-test    Run one deterministic offline Agent loop"
	@echo "  make test-handoff  Run fast handoff/architecture contract tests"
	@echo "  make test-fast     Run all unit tests"
	@echo "  make test          Run the full test suite"
	@echo "  make lint          Compile core Python modules"
	@echo "  make agent ARGS="Check health""
	@echo "All commands use $(VENV), never the system Python."

setup:
	@if [ ! -x "$(PYTHON)" ]; then 		if command -v conda >/dev/null 2>&1; then 			conda create -p "$(VENV)" python=3.11 -y; 		elif command -v "$(PROJECT_PY)" >/dev/null 2>&1; then 			"$(PROJECT_PY)" -m venv "$(VENV)"; 		else 			echo "Python 3.11 or conda is required"; exit 1; 		fi; 	fi
	@$(MAKE) install

install:
	@if [ ! -x "$(PYTHON)" ]; then echo "Missing $(PYTHON); run: make setup"; exit 1; fi
	$(PIP) install -e . --no-build-isolation
	$(PYTHON) scripts/doctor.py

doctor:
	$(PYTHON) scripts/doctor.py

smoke-test: doctor
	$(PYTHON) scripts/smoke_test.py

test-handoff: doctor
	$(PYTHON) -m pytest -q --tb=short 		tests/unit/test_handoff_contract.py 		tests/unit/test_tool_registry.py 		tests/unit/test_agent_planner_behavior.py

test: doctor
	$(PYTHON) -m pytest tests/ -v --tb=short

test-fast: doctor
	$(PYTHON) -m pytest tests/unit/ -v --tb=short

qgpt-help:
	$(QGPT) --help

agent:
	$(QGPT) agent "$(ARGS)" --backend FakeBrisbane --provider mock

orchestrator-test:
	$(PYTHON) -c "from agent.orchestrator.graph import build_graph; print('orchestrator graph build OK:', build_graph())"

demo-v2.5:
	bash showcase/run_v2.5_demos.sh

demo-rabi:
	PYTHONPATH=. $(PYTHON) -c "from demos.rabi_demo import run_rabi_demo; run_rabi_demo(save_dir='showcase/v2.5_demo_output/rabi')"

demo-drift:
	PYTHONPATH=. $(PYTHON) demos/drift_demo.py

demo-memory:
	PYTHONPATH=. $(PYTHON) demos/memory_demo.py

lint:
	$(PYTHON) -m compileall -q agent advisor backends bench benchmark data dynamics eval tools scripts cli.py
	@echo "Python compilation passed"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned Python caches"
