"""Task ID + run persistence for QuantumGPT.

Every agent invocation (`qgpt run`, `qgpt agent`, eval scripts) computes a
**deterministic task_id** from the (circuit, backend, target, budget, seed,
code_commit) tuple, and persists everything under

    ~/.quantumgpt/runs/<task_id>/
        task_spec.json   # the input spec
        metadata.json    # task_id, started_at, code_commit, status, ...
        trace.jsonl      # agent trace lines (jsonl)
        result.json      # final_answer, best_score, verifier_verdict, cost
        pre_state.json   # graph state at pre-drift (orchestrator only)
        post_state.json  # graph state at post-drift (orchestrator only)
        agent_log.txt    # plain text agent stdout/stderr

The task_id is the first 12 hex chars of sha256(canonical_json(spec) +
git_commit). Same inputs + same code → same id. Useful for:

  * `qgpt rerun <task_id>` — load saved spec, replay agent
  * `qgpt show <task_id>`  — pretty-print metadata + result
  * `qgpt runs`            — list all runs (filter by date / backend / circuit)
  * dashboards / PDF reports — every artifact addressable by task_id

Public API:
    compute_task_id(spec)  -> str
    runs_root()            -> Path
    run_dir(task_id)       -> Path
    save_task_spec(...)    -> Path  (writes task_spec.json + metadata.json)
    save_result(...)       -> Path  (writes result.json)
    append_trace(...)      -> None  (appends to trace.jsonl)
    list_runs(...)         -> list[dict]
    load_run(task_id)      -> dict
    git_commit_short()     -> str
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


# ──────────────────────────────────────────────────────────────────────
# Spec dataclass — single source of truth for what defines a "task"
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskSpec:
    """A runnable task. task_id is a deterministic hash of these fields."""

    # What to run
    circuit: str                       # circuit name (e.g. 'ghz_5') OR qasm filename
    backend: str = "FakeBrisbane"      # backend identifier (FakeBrisbane / ibm:brisbane / ...)

    # Goals
    target_fidelity: float = 0.85
    shots: int = 4096

    # Budget (None = no cap)
    max_tool_calls: int = 12
    max_seconds: int = 60

    # Reproducibility
    seed: int = 0
    drift_profile: str = "stable"      # stable / mild_gradual / multi_shock / severe_sudden

    # Agent variant
    system: str = "QuantumGPT-Orchestrator"   # which agent system to run
    model: str = "rule-planner-v1"     # LLM model (or 'rule-planner-v1' for mock)
    provider: str = "auto"

    # Free-form extras (sorted dict serialized into hash)
    extras: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Hashing
# ──────────────────────────────────────────────────────────────────────

def _canonical_spec_payload(spec: TaskSpec) -> str:
    """Stable serialization for hashing.

    Uses json.dumps(sort_keys=True, default=str) so dict ordering and
    numeric floats serialize the same way each time. Bool/None pass through.
    """
    payload = asdict(spec)
    # Sort the extras dict deterministically
    payload["extras"] = dict(sorted(payload["extras"].items()))
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_task_id(spec: TaskSpec, code_commit: Optional[str] = None) -> str:
    """Return a 12-char hex task_id derived from the spec + code commit.

    Same inputs → same id. The commit is included so that a code change
    does not silently produce identical task_id with different agent
    behavior. Pass `code_commit=""` to disable code pinning.
    """
    if code_commit is None:
        code_commit = git_commit_short() or "no-git"
    payload = _canonical_spec_payload(spec) + "|" + code_commit
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def git_commit_short() -> str:
    """Return short git SHA, or empty string if not a git repo / git missing."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def git_working_tree_status() -> dict[str, Any]:
    """Return whether the repo is dirty plus a compact status summary.

    This is for reproducibility metadata only. It does not affect task_id
    hashing because we still want the same spec+commit to address the same
    run family, while making dirty-state explicit in metadata.
    """
    repo_root = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode("utf-8")
    except Exception:
        return {"git_dirty": False, "git_diff_summary": []}

    lines = [ln.rstrip() for ln in out.splitlines() if ln.strip()]
    return {
        "git_dirty": bool(lines),
        "git_diff_summary": lines[:50],
    }


# ──────────────────────────────────────────────────────────────────────
# Filesystem layout
# ──────────────────────────────────────────────────────────────────────

def runs_root() -> Path:
    """~/.quantumgpt/runs/ — all task records live here."""
    root = os.environ.get("QUANTUMGPT_RUNS_DIR")
    if root:
        return Path(root).expanduser().resolve()
    return Path("~/.quantumgpt/runs").expanduser().resolve()


def run_dir(task_id: str) -> Path:
    """Directory holding all artifacts for one task_id."""
    return runs_root() / task_id


# ──────────────────────────────────────────────────────────────────────
# Save / load
# ──────────────────────────────────────────────────────────────────────

def save_task_spec(spec: TaskSpec, task_id: Optional[str] = None) -> tuple[str, Path]:
    """Persist task_spec.json + metadata.json. Returns (task_id, run_dir).

    Idempotent — if the same task_id directory already exists, it is
    overwritten in place (you typically want this to update metadata).
    """
    if task_id is None:
        task_id = compute_task_id(spec)
    d = run_dir(task_id)
    d.mkdir(parents=True, exist_ok=True)

    (d / "task_spec.json").write_text(
        json.dumps(asdict(spec), indent=2, sort_keys=True, default=str)
    )

    git_status = git_working_tree_status()
    meta = {
        "task_id": task_id,
        "started_at": time.time(),
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "code_commit": git_commit_short(),
        "hostname": socket.gethostname(),
        "status": "running",
        "spec": asdict(spec),
        **git_status,
    }
    (d / "metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str)
    )
    return task_id, d


def save_result(task_id: str, result: dict, status: str = "completed") -> Path:
    """Persist result.json and update metadata status + finished_at."""
    d = run_dir(task_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str)
    )

    meta_path = d / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"task_id": task_id, "spec": {}}
    meta["finished_at"] = time.time()
    meta["finished_at_iso"] = time.strftime(
        "%Y-%m-%dT%H:%M:%S%z", time.localtime()
    )
    meta["status"] = status
    if "started_at" in meta:
        meta["walltime_seconds"] = round(
            meta["finished_at"] - meta["started_at"], 3
        )
    # Cache headline numbers in metadata for cheap listing
    for key in (
        "best_score",
        "target_fidelity",
        "verifier_satisfied",
        "verifier_hallucinated",
        "total_cost_usd",
        "tool_calls",
        "termination_reason",
        "verifier_reason",
        "plan_revision",
    ):
        if key in result:
            meta.setdefault("summary", {})[key] = result[key]
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
    return d / "result.json"


def append_trace(task_id: str, entry: dict) -> None:
    """Append one JSON line to trace.jsonl."""
    d = run_dir(task_id)
    d.mkdir(parents=True, exist_ok=True)
    with (d / "trace.jsonl").open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def save_artifact(task_id: str, name: str, content: Any) -> Path:
    """Persist an arbitrary JSON artifact (e.g. pre_graph_state)."""
    d = run_dir(task_id)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    path.write_text(json.dumps(content, indent=2, sort_keys=True, default=str))
    return path


# ──────────────────────────────────────────────────────────────────────
# Discovery / listing
# ──────────────────────────────────────────────────────────────────────

def list_runs(
    backend: Optional[str] = None,
    circuit: Optional[str] = None,
    since_epoch: Optional[float] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Return list of run metadata dicts, sorted newest-first.

    Each entry has keys: task_id, started_at, started_at_iso, status,
    code_commit, hostname, spec (input dict), summary (cached headline).
    """
    root = runs_root()
    if not root.exists():
        return []

    rows = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        # Filters
        if status and meta.get("status") != status:
            continue
        if since_epoch and meta.get("started_at", 0) < since_epoch:
            continue
        spec = meta.get("spec") or {}
        if backend and spec.get("backend") != backend:
            continue
        if circuit and spec.get("circuit") != circuit:
            continue
        rows.append(meta)

    rows.sort(key=lambda r: r.get("started_at", 0), reverse=True)
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_run(task_id: str) -> dict:
    """Load all artifacts for a task_id. Returns a dict with keys:
    metadata, spec, result (if present), trace (list of entries),
    pre_state, post_state, oracle_row.
    """
    d = run_dir(task_id)
    if not d.exists():
        raise FileNotFoundError(f"Run directory not found: {d}")
    out: dict = {"task_id": task_id, "run_dir": str(d)}
    if (d / "metadata.json").exists():
        out["metadata"] = json.loads((d / "metadata.json").read_text())
    if (d / "task_spec.json").exists():
        out["spec"] = json.loads((d / "task_spec.json").read_text())
    if (d / "result.json").exists():
        out["result"] = json.loads((d / "result.json").read_text())
    trace_path = d / "trace.jsonl"
    if trace_path.exists():
        out["trace"] = [
            json.loads(ln) for ln in trace_path.read_text().splitlines() if ln.strip()
        ]

    artifact_aliases = {
        "pre_state": ["pre_state.json", "pre_graph_state.json"],
        "post_state": ["post_state.json", "post_graph_state.json"],
        "oracle_row": ["oracle_row.json"],
    }
    for key, candidates in artifact_aliases.items():
        for filename in candidates:
            path = d / filename
            if not path.exists():
                continue
            try:
                out[key] = json.loads(path.read_text())
                break
            except Exception:
                continue
    return out


def load_spec(task_id: str) -> TaskSpec:
    """Reconstruct a TaskSpec from a saved run (for `qgpt rerun`)."""
    d = run_dir(task_id)
    spec_path = d / "task_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"task_spec.json missing in {d}")
    raw = json.loads(spec_path.read_text())
    # Filter to only fields TaskSpec accepts (in case schema evolves)
    allowed = {f for f in TaskSpec.__dataclass_fields__}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    return TaskSpec(**filtered)
