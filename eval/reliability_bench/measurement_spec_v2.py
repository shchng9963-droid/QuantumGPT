"""Frozen construct registry for the executable ReliabilityBench-Q measures.

This module is evaluator-facing research protocol.  Controllers and runtime
guards must not import it.  Keeping task requirements here prevents prompt or
method code from acquiring hidden evaluation predicates.
"""

from __future__ import annotations

from types import MappingProxyType


MEASUREMENT_SPEC_VERSION = "reliabilitybench-q/measurement-constructs-2.0"

# Req(d): evidence slots required to support a terminal decision of each task.
TASK_REQUIRED_SLOTS = MappingProxyType(
    {
        "backend_selection": ("backend_ranking",),
        "qubit_mapping": ("qubit_properties",),
        "transpilation": ("compilation",),
        "fidelity_claim": ("circuit_result",),
        "mitigation_decision": ("mitigation_estimate",),
        "unreachable_target": ("backend_capacity",),
    }
)

# Pre-registered primary utility.  Alternative weights, if reported, are
# sensitivity analyses and must never replace this primary estimand post hoc.
PRIMARY_UTILITY_WEIGHTS = MappingProxyType(
    {
        "correct": 1.0,
        "wrong": 1.0,
        "unnecessary_abstain": 0.5,
        "cost": 0.01,
    }
)

EVIDENCE_STATE_RULES = MappingProxyType(
    {
        "safe": (
            "every Req(d) slot is covered by cited, current, valid evidence; "
            "all cited evidence IDs resolve; and no cited evidence is stale"
        ),
        "unsafe": "at least one cited evidence node is explicitly stale",
        "unknown": (
            "not unsafe, but support is inadequate or provenance is incomplete"
        ),
    }
)

CONSTRUCT_REGISTRY = (
    {
        "construct": "task_predicate",
        "required_observables": "terminal decision; frozen episode predicate",
        "evaluator": "TaskPredicateEvaluator",
        "ambiguity_condition": "predicate does not define a set-valued acceptable action",
        "status": "primary",
        "paper_claim": "current-snapshot decision correctness",
    },
    {
        "construct": "evidence_safety_state",
        "required_observables": "support IDs; evidence graph; validity; Req(d)",
        "evaluator": "TraceEvidenceEvaluator",
        "ambiguity_condition": "missing provenance produces unknown, never unsafe",
        "status": "primary",
        "paper_claim": "safe/unsafe/unknown decision support",
    },
    {
        "construct": "definite_stale_dependence",
        "required_observables": "cited evidence IDs; explicit drift invalidation",
        "evaluator": "TraceEvidenceEvaluator",
        "ambiguity_condition": "uncited queried evidence is not dependence",
        "status": "primary",
        "paper_claim": "Evidence Layer reduces stale evidence use",
    },
    {
        "construct": "support_adequacy",
        "required_observables": "Req(d); cited valid evidence slot coverage",
        "evaluator": "TraceEvidenceEvaluator",
        "ambiguity_condition": "unknown evidence type cannot cover a required slot",
        "status": "primary",
        "paper_claim": "decisions are supported by current evidence",
    },
    {
        "construct": "required_revalidation",
        "required_observables": "invalidated initial slots; successful current calls",
        "evaluator": "TraceEvidenceEvaluator",
        "ambiguity_condition": "task requirement absent from Req(d)",
        "status": "primary",
        "paper_claim": "selective recovery restores only required evidence",
    },
    {
        "construct": "intervention_stages",
        "required_observables": "requested policy; matching call; response; verification evidence",
        "evaluator": "TraceEvidenceEvaluator",
        "ambiguity_condition": "policy identity cannot be matched across request and call",
        "status": "primary",
        "paper_claim": "mitigation is requested, attempted, executed, and verified",
    },
    {
        "construct": "coverage_and_selective_risk",
        "required_observables": "terminal status; task predicate result; feasibility",
        "evaluator": "TaskPredicateEvaluator",
        "ambiguity_condition": "task feasibility is not executable",
        "status": "primary",
        "paper_claim": "reliability is not obtained by universal abstention",
    },
    {
        "construct": "actual_cost",
        "required_observables": "accepted calls; frozen prices; observed latency",
        "evaluator": "TraceEvidenceEvaluator",
        "ambiguity_condition": "provider-side unlogged work is outside the estimand",
        "status": "primary",
        "paper_claim": "selective revalidation reduces observed recovery cost",
    },
    {
        "construct": "semantic_failure_stage",
        "required_observables": "full trace and human codebook",
        "evaluator": "Semantic Diagnostic Classifier or human analysis",
        "ambiguity_condition": "multiple plausible primary failure stages",
        "status": "secondary",
        "paper_claim": "qualitative diagnosis only",
    },
)


__all__ = [
    "CONSTRUCT_REGISTRY",
    "EVIDENCE_STATE_RULES",
    "MEASUREMENT_SPEC_VERSION",
    "PRIMARY_UTILITY_WEIGHTS",
    "TASK_REQUIRED_SLOTS",
]
