"""Authoritative JSONL schema for paper-grade QuantumGPT runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


ALLOWED_EXECUTION_MODES = {"real", "hardware", "simulator"}


def assert_paper_grade_execution(*, execution_mode: str, provider: str, use_mock: bool) -> None:
    """Fail loudly if a paper-grade agentic run is contaminated by mock mode."""

    if execution_mode == "mock" or provider == "mock" or use_mock:
        raise ValueError(
            "paper-grade evaluation forbids mock/rule-planner execution; "
            f"got execution_mode={execution_mode!r}, provider={provider!r}, use_mock={use_mock!r}"
        )


@dataclass
class PaperGradeResultRow:
    """One authoritative row in a paper-grade result_summary.jsonl file."""

    run_id: str
    system: str
    task: str
    circuit: str
    drift_profile: str
    seed: int
    execution_mode: str
    provider: str
    model: str
    use_mock: bool
    trace_file: str
    target_fidelity: float
    final_score: float | None
    best_score: float | None
    target_success: bool
    tool_calls: int
    max_tool_calls: int
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        missing = [
            name for name in [
                "run_id",
                "system",
                "task",
                "circuit",
                "drift_profile",
                "execution_mode",
                "provider",
                "model",
                "trace_file",
            ]
            if not getattr(self, name)
        ]
        if missing:
            raise ValueError(f"Missing required paper-grade result fields: {', '.join(missing)}")
        if self.execution_mode not in ALLOWED_EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {sorted(ALLOWED_EXECUTION_MODES)}, "
                f"got {self.execution_mode!r}"
            )
        assert_paper_grade_execution(
            execution_mode=self.execution_mode,
            provider=self.provider,
            use_mock=self.use_mock,
        )
        if self.tool_calls < 0 or self.max_tool_calls < 0:
            raise ValueError("tool call counts must be non-negative")
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.total_cost_usd < 0:
            raise ValueError("total_cost_usd must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def write_result_jsonl(path: str | Path, rows: Iterable[PaperGradeResultRow]) -> None:
    """Write paper-grade rows to JSONL after validating every row."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), sort_keys=True, default=str) + "\n")
