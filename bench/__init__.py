"""Quantum GPT - Bench Module

QC-Agent-Bench: benchmark suite for quantum-LLM-agent evaluation.
Tiers:
  - Tier 1: 30 static tasks (fixed backend snapshot)
  - Tier 2: 20 drift tasks (ReplayBackend)
  - Tier 3: 10 failure tasks (injected anomalies)
  - Tier 4: 5 device-level tasks (Rabi tune-up)
"""
