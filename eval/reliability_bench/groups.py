"""Pre-registered component matrix and fairness contract for experiment arms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class ExperimentGroup(str, Enum):
    REACT = "react"
    MONITOR_ONLY = "monitor_only"
    LEDGER_ONLY = "ledger_only"
    GLOBAL_REVALIDATE = "global_revalidate"
    FULL_SELECTIVE = "full_selective"
    PROMPT_ORACLE = "prompt_oracle"
    ORACLE = "oracle"


@dataclass(frozen=True)
class GroupConfig:
    sees_neutral_drift_event: bool
    evidence_ledger: bool
    invalidation_mode: str
    revalidation_policy: str
    prompt_oracle: bool = False


GROUP_CONFIGS: Mapping[ExperimentGroup, GroupConfig] = {
    ExperimentGroup.REACT: GroupConfig(False, False, "none", "llm"),
    ExperimentGroup.MONITOR_ONLY: GroupConfig(True, False, "none", "llm"),
    ExperimentGroup.LEDGER_ONLY: GroupConfig(True, True, "none", "llm"),
    ExperimentGroup.GLOBAL_REVALIDATE: GroupConfig(
        True, True, "global", "global"
    ),
    ExperimentGroup.FULL_SELECTIVE: GroupConfig(
        True, True, "dependency_scoped", "selective"
    ),
    ExperimentGroup.PROMPT_ORACLE: GroupConfig(
        True, False, "none", "llm", prompt_oracle=True
    ),
    ExperimentGroup.ORACLE: GroupConfig(True, True, "oracle", "oracle"),
}


@dataclass(frozen=True)
class FairnessContract:
    model: str = "locked-at-pilot"
    temperature: float = 0.0
    max_tokens: int = 4096
    max_turns: int = 12
    max_tool_calls: int = 10
    neutral_event_fields: tuple[str, ...] = (
        "event_id",
        "phase",
        "changed_features",
        "change_magnitude",
        "magnitude_quantile",
    )


def validate_group_matrix() -> None:
    if set(GROUP_CONFIGS) != set(ExperimentGroup):
        raise ValueError("every experiment group must have exactly one config")
    main_groups = set(ExperimentGroup) - {ExperimentGroup.REACT}
    if not all(GROUP_CONFIGS[group].sees_neutral_drift_event for group in main_groups):
        raise ValueError("all Study B groups must receive the same neutral event")
    configs = list(GROUP_CONFIGS.values())
    if len(configs) != len(set(configs)):
        raise ValueError("experiment group configurations must be distinct")
    forbidden = ("rerun", "revalidate", "invalid", "stale", "recommended_action")
    for field_name in FairnessContract().neutral_event_fields:
        if any(token in field_name.lower() for token in forbidden):
            raise ValueError("neutral event fields must not reveal a required action")


__all__ = [
    "ExperimentGroup",
    "FairnessContract",
    "GROUP_CONFIGS",
    "GroupConfig",
    "validate_group_matrix",
]
