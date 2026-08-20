"""Minimum-cost evidence-slot recovery planner, separate from ActionGuard."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


RECOVERY_PLANNER_VERSION = "reliabilitybench-q/recovery-planner-1.0"


@dataclass(frozen=True)
class RecoveryCandidate:
    action_id: str
    restores_slots: tuple[str, ...]
    cost_units: int

    def __post_init__(self) -> None:
        if self.cost_units < 0:
            raise ValueError("recovery cost must be non-negative")


@dataclass(frozen=True)
class RecoveryPlan:
    action_ids: tuple[str, ...]
    restored_slots: tuple[str, ...]
    total_cost_units: int
    feasible: bool


def minimum_cost_recovery_plan(
    required_slots: tuple[str, ...],
    candidates: tuple[RecoveryCandidate, ...],
) -> RecoveryPlan:
    """Exact small-instance set-cover solver over abstract recovery actions."""
    required = set(required_slots)
    if not required:
        return RecoveryPlan((), (), 0, True)
    best: tuple[int, int, tuple[str, ...], set[str]] | None = None
    for size in range(1, len(candidates) + 1):
        for selected in combinations(candidates, size):
            restored = {slot for item in selected for slot in item.restores_slots}
            if not required.issubset(restored):
                continue
            action_ids = tuple(sorted(item.action_id for item in selected))
            key = (sum(item.cost_units for item in selected), size, action_ids, restored)
            if best is None or key[:3] < best[:3]:
                best = key
    if best is None:
        return RecoveryPlan((), (), 0, False)
    return RecoveryPlan(best[2], tuple(sorted(best[3])), best[0], True)


__all__ = [
    "RECOVERY_PLANNER_VERSION",
    "RecoveryCandidate",
    "RecoveryPlan",
    "minimum_cost_recovery_plan",
]
