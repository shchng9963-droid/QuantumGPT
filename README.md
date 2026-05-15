# Quantum GPT

A device-aware closed-loop LLM agent for NISQ quantum computing workflows.

## Overview

Quantum GPT is an agentic system that operates quantum computing tasks under realistic hardware drift conditions. It provides:

- **Hardware perception**: real-time monitoring of backend health, qubit properties, and drift detection
- **Tool-based execution**: 14+ MCP-compatible tools spanning gate-level and pulse-level operations
- **Closed-loop learning**: ExperimentRecord-based memory enabling drift-aware replanning
- **QC-Agent-Bench**: a benchmark suite (65 tasks) for evaluating quantum-LLM-agents

## Project Structure

```
quantum-gpt/
├── backends/       # Shadow hardware: FakeBackend, Replay, SyntheticDrift
├── tools/          # MCP tool implementations (perception, gate, pulse, analysis)
├── agent/          # Agent loop, planner, memory
├── bench/          # QC-Agent-Bench task definitions & ground truth
├── eval/           # Evaluation scripts & metrics
├── configs/        # Backend configs, agent configs
├── scripts/        # CLI entry points, utilities
├── tests/          # Unit & integration tests
├── data/           # Calibration data, circuit data, results
├── notebooks/      # Exploration & demo notebooks
└── docs/           # Documentation & related work survey
```

## Setup

```bash
# Create conda environment
conda create -p .venv python=3.11 -y
conda activate .venv

# Install dependencies
pip install -r requirements.txt
```

## Target

ICLR 2027 Main Track

## Status

Phase 0 — Foundation (W1–W2)
