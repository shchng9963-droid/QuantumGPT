"""Baseline implementations for the v2.5 paired benchmark.

These are *non-LLM*, deterministic comparators that share the public
PublicAgentRunResult schema with the agent rows. Each baseline reads a
pre-computed oracle row (from eval/public_mqtbench_oracle.py) and emits
one result row, so they can be paired (circuit, seed) with agent rows for
Wilcoxon / McNemar / bootstrap CI analysis.
"""
