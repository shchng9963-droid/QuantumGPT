"""Quantum GPT - Backends Module

Shadow hardware backends that mimic real quantum devices.
Three concrete implementations:
  - FakeBackendAdapter: wraps Qiskit FakeBackendV2 (static snapshots)
  - ReplayBackend: replays historical IBM calibration data over time
  - SyntheticDriftBackend: injects parameterized T1/T2/gate-error drift
"""
