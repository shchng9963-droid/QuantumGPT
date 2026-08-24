"""Append-only Measurement v3.1 construct delta for runtime terminal closure."""

from __future__ import annotations

from types import MappingProxyType

from .measurement_spec_v3 import (
    CONSTRUCT_REGISTRY as V3_CONSTRUCT_REGISTRY,
    EVIDENCE_STATE_RULES,
    TASK_REQUIRED_SLOTS,
)


MEASUREMENT_SPEC_VERSION = "reliabilitybench-q/measurement-constructs-3.1"

PRIMARY_UTILITY_WEIGHTS = MappingProxyType(
    {
        "correct": 1.0,
        "wrong": 1.0,
        "unnecessary_abstain": 0.5,
        "execution_failure": 1.0,
        "cost": 0.01,
    }
)

CONSTRUCT_REGISTRY = V3_CONSTRUCT_REGISTRY + (
    {
        "construct": "runtime_execution_failure",
        "required_observables": (
            "runtime state transition; immutable candidate hash; failure reason; "
            "complete token/tool/latency/cost trace"
        ),
        "evaluator": "RuntimeTerminalEvaluator",
        "ambiguity_condition": "workflow terminates before an append-only trace can close",
        "status": "primary ITT outcome",
        "paper_claim": (
            "safe failure lowers coverage and utility and remains scoreable rather "
            "than disappearing as unscorable"
        ),
    },
)


__all__ = [
    "CONSTRUCT_REGISTRY",
    "EVIDENCE_STATE_RULES",
    "MEASUREMENT_SPEC_VERSION",
    "PRIMARY_UTILITY_WEIGHTS",
    "TASK_REQUIRED_SLOTS",
]
