"""ExperimentRecord — structured experiment memory for QuantumGPT.

Replaces simple (circuit, fidelity) episodic memory with a rich record that
captures the full context of every experiment the agent runs. This is a key
differentiator for the paper: the agent can retrieve past experiments via
semantic search and learn from its own history.

Schema design:
  - Each record = one "experiment" the agent ran (could be a single circuit
    execution, a Rabi tune-up, or a multi-step diagnostic session)
  - Records are immutable once written
  - Rich metadata supports both gate-level and pulse-level experiments
  - Designed for RAG retrieval (summary field is the embedding target)

Usage:
    from data.experiment_record import ExperimentRecord, ExperimentStore

    rec = ExperimentRecord(
        experiment_type="circuit_benchmark",
        backend="FakeBrisbane",
        circuit_name="ghz_5",
        problem="Run GHZ-5 and evaluate fidelity",
        plan=["check_health", "run_circuit", "diagnose"],
        tool_calls=[...],
        fidelity=0.927,
        success=True,
        summary="GHZ-5 on FakeBrisbane: fidelity 0.927, nominal health",
    )

    store = ExperimentStore(db)
    store.save(rec)
    similar = store.search(circuit_name="ghz_5", min_fidelity=0.9)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import duckdb


# ── Enums ────────────────────────────────────────────────────────

class ExperimentType(str, Enum):
    """Categories of experiments the agent can run."""
    CIRCUIT_BENCHMARK = "circuit_benchmark"
    HEALTH_CHECK = "health_check"
    DRIFT_DIAGNOSIS = "drift_diagnosis"
    RABI_TUNEUP = "rabi_tuneup"
    ERROR_MITIGATION = "error_mitigation"
    MULTI_STEP_TASK = "multi_step_task"
    CUSTOM = "custom"


class ExperimentOutcome(str, Enum):
    """Outcome of an experiment."""
    SUCCESS = "success"
    PARTIAL = "partial"       # completed but below threshold
    FAILURE = "failure"
    ERROR = "error"           # runtime error, not physics failure


# ── Dataclass ────────────────────────────────────────────────────

@dataclass
class ExperimentRecord:
    """A single experiment record — the agent's structured memory unit.

    Fields are grouped by purpose:
      Identity:  id, timestamp, experiment_type
      Context:   backend, circuit_name, problem, tags
      Plan:      plan (list of intended steps)
      Execution: tool_calls, decisions, elapsed_seconds
      Results:   fidelity, success, outcome, raw_data_ref
      Fitting:   fits (for Rabi/calibration experiments)
      Reflection: summary, lessons, failure_reason
    """

    # ── Identity ──
    experiment_type: str = ExperimentType.CIRCUIT_BENCHMARK.value
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Context ──
    backend: str = ""
    circuit_name: Optional[str] = None
    num_qubits: Optional[int] = None
    problem: str = ""                   # natural-language task description
    tags: list[str] = field(default_factory=list)

    # ── Plan ──
    plan: list[str] = field(default_factory=list)  # intended tool sequence

    # ── Execution ──
    tool_calls: list[dict] = field(default_factory=list)  # actual calls made
    decisions: list[dict] = field(default_factory=list)    # agent decision points
    elapsed_seconds: Optional[float] = None
    model: Optional[str] = None
    total_tokens: Optional[int] = None

    # ── Results ──
    fidelity: Optional[float] = None
    success: bool = False
    outcome: str = ExperimentOutcome.SUCCESS.value
    metrics: dict[str, float] = field(default_factory=dict)  # extra metrics
    raw_data_ref: Optional[str] = None  # path to raw data (counts, IQ, etc.)

    # ── Fitting (Rabi / calibration) ──
    fits: dict[str, Any] = field(default_factory=dict)
    # e.g. {"rabi_freq_mhz": 42.3, "pi_amp": 0.312, "r_squared": 0.997}

    # ── Backend snapshot at experiment time ──
    backend_snapshot: dict[str, Any] = field(default_factory=dict)
    # e.g. {"avg_t1_us": 224.5, "avg_2q_error": 0.009, "drift_score": 0.15}

    # ── Reflection ──
    summary: str = ""          # one-line summary (RAG embedding target)
    lessons: list[str] = field(default_factory=list)  # things learned
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to dict (JSON-safe)."""
        d = asdict(self)
        # Ensure nested structures are JSON strings for DuckDB
        for key in ("tags", "plan", "tool_calls", "decisions",
                     "metrics", "fits", "backend_snapshot", "lessons"):
            if isinstance(d[key], (list, dict)):
                d[key] = json.dumps(d[key])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentRecord:
        """Deserialize from dict (as returned by DuckDB query)."""
        for key in ("tags", "plan", "tool_calls", "decisions",
                     "metrics", "fits", "backend_snapshot", "lessons"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


# ── DuckDB Table Schema ─────────────────────────────────────────

EXPERIMENT_RECORDS_SQL = """
CREATE TABLE IF NOT EXISTS experiment_records (
    id                  VARCHAR PRIMARY KEY,
    timestamp           TIMESTAMP NOT NULL,
    experiment_type     VARCHAR NOT NULL,

    -- Context
    backend             VARCHAR NOT NULL,
    circuit_name        VARCHAR,
    num_qubits          INTEGER,
    problem             VARCHAR,
    tags                VARCHAR,          -- JSON array

    -- Plan
    plan                VARCHAR,          -- JSON array

    -- Execution
    tool_calls          VARCHAR,          -- JSON array of dicts
    decisions           VARCHAR,          -- JSON array of dicts
    elapsed_seconds     DOUBLE,
    model               VARCHAR,
    total_tokens        INTEGER,

    -- Results
    fidelity            DOUBLE,
    success             BOOLEAN,
    outcome             VARCHAR,
    metrics             VARCHAR,          -- JSON dict
    raw_data_ref        VARCHAR,

    -- Fitting
    fits                VARCHAR,          -- JSON dict

    -- Backend snapshot
    backend_snapshot    VARCHAR,          -- JSON dict

    -- Reflection
    summary             VARCHAR,
    lessons             VARCHAR,          -- JSON array
    failure_reason      VARCHAR
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_er_backend ON experiment_records(backend);
CREATE INDEX IF NOT EXISTS idx_er_circuit ON experiment_records(circuit_name);
CREATE INDEX IF NOT EXISTS idx_er_type ON experiment_records(experiment_type);
CREATE INDEX IF NOT EXISTS idx_er_fidelity ON experiment_records(fidelity);
CREATE INDEX IF NOT EXISTS idx_er_ts ON experiment_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_er_success ON experiment_records(success);
"""


# ── ExperimentStore ──────────────────────────────────────────────

class ExperimentStore:
    """Persistence layer for ExperimentRecord — DuckDB backed.

    Provides:
      - save() / save_batch()
      - get() by ID
      - search() with flexible filters
      - recent() / by_circuit() / by_backend() convenience methods
      - stats() for aggregate summaries
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        self._init_schema()

    def _init_schema(self):
        self.conn.execute(EXPERIMENT_RECORDS_SQL)

    # ── Write ────────────────────────────────────────────

    def save(self, record: ExperimentRecord) -> str:
        """Save a single experiment record. Returns the record ID."""
        d = record.to_dict()
        cols = list(d.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        self.conn.execute(
            f"INSERT INTO experiment_records ({col_names}) VALUES ({placeholders})",
            list(d.values())
        )
        return record.id

    def save_batch(self, records: list[ExperimentRecord]) -> list[str]:
        """Save multiple records. Returns list of IDs."""
        return [self.save(r) for r in records]

    # ── Read ─────────────────────────────────────────────

    def get(self, record_id: str) -> Optional[ExperimentRecord]:
        """Get a single record by ID."""
        rows = self._query(
            "SELECT * FROM experiment_records WHERE id = ?", [record_id])
        return ExperimentRecord.from_dict(rows[0]) if rows else None

    def recent(self, limit: int = 20) -> list[ExperimentRecord]:
        """Get most recent experiments."""
        rows = self._query(
            "SELECT * FROM experiment_records ORDER BY timestamp DESC LIMIT ?",
            [limit])
        return [ExperimentRecord.from_dict(r) for r in rows]

    def by_circuit(self, circuit_name: str,
                   limit: int = 50) -> list[ExperimentRecord]:
        """Get experiments for a specific circuit."""
        rows = self._query(
            "SELECT * FROM experiment_records "
            "WHERE circuit_name = ? ORDER BY timestamp DESC LIMIT ?",
            [circuit_name, limit])
        return [ExperimentRecord.from_dict(r) for r in rows]

    def by_backend(self, backend: str,
                   limit: int = 50) -> list[ExperimentRecord]:
        """Get experiments on a specific backend."""
        rows = self._query(
            "SELECT * FROM experiment_records "
            "WHERE backend = ? ORDER BY timestamp DESC LIMIT ?",
            [backend, limit])
        return [ExperimentRecord.from_dict(r) for r in rows]

    def by_type(self, experiment_type: str,
                limit: int = 50) -> list[ExperimentRecord]:
        """Get experiments of a specific type."""
        rows = self._query(
            "SELECT * FROM experiment_records "
            "WHERE experiment_type = ? ORDER BY timestamp DESC LIMIT ?",
            [experiment_type, limit])
        return [ExperimentRecord.from_dict(r) for r in rows]

    # ── Search ───────────────────────────────────────────

    def search(
        self,
        circuit_name: Optional[str] = None,
        backend: Optional[str] = None,
        experiment_type: Optional[str] = None,
        min_fidelity: Optional[float] = None,
        max_fidelity: Optional[float] = None,
        success_only: bool = False,
        since: Optional[str] = None,       # ISO timestamp
        limit: int = 50,
    ) -> list[ExperimentRecord]:
        """Flexible search with multiple optional filters."""
        conditions = []
        params = []

        if circuit_name is not None:
            conditions.append("circuit_name = ?")
            params.append(circuit_name)
        if backend is not None:
            conditions.append("backend = ?")
            params.append(backend)
        if experiment_type is not None:
            conditions.append("experiment_type = ?")
            params.append(experiment_type)
        if min_fidelity is not None:
            conditions.append("fidelity >= ?")
            params.append(min_fidelity)
        if max_fidelity is not None:
            conditions.append("fidelity <= ?")
            params.append(max_fidelity)
        if success_only:
            conditions.append("success = true")
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = (f"SELECT * FROM experiment_records "
               f"WHERE {where} ORDER BY timestamp DESC LIMIT ?")
        params.append(limit)

        rows = self._query(sql, params)
        return [ExperimentRecord.from_dict(r) for r in rows]

    # ── Stats ────────────────────────────────────────────

    def stats(self) -> dict:
        """Aggregate statistics across all experiment records."""
        rows = self._query("""
            SELECT
                COUNT(*) as total_experiments,
                COUNT(DISTINCT backend) as unique_backends,
                COUNT(DISTINCT circuit_name) as unique_circuits,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
                AVG(fidelity) as avg_fidelity,
                MIN(fidelity) as min_fidelity,
                MAX(fidelity) as max_fidelity,
                AVG(elapsed_seconds) as avg_elapsed_seconds,
                SUM(total_tokens) as total_tokens_used
            FROM experiment_records
        """)
        return rows[0] if rows else {}

    def stats_by_circuit(self) -> list[dict]:
        """Per-circuit aggregate stats."""
        return self._query("""
            SELECT
                circuit_name,
                COUNT(*) as runs,
                AVG(fidelity) as avg_fidelity,
                MIN(fidelity) as min_fidelity,
                MAX(fidelity) as max_fidelity,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
                AVG(elapsed_seconds) as avg_elapsed_s
            FROM experiment_records
            WHERE circuit_name IS NOT NULL
            GROUP BY circuit_name
            ORDER BY runs DESC
        """)

    def stats_by_backend(self) -> list[dict]:
        """Per-backend aggregate stats."""
        return self._query("""
            SELECT
                backend,
                COUNT(*) as runs,
                AVG(fidelity) as avg_fidelity,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
                COUNT(DISTINCT circuit_name) as unique_circuits
            FROM experiment_records
            GROUP BY backend
            ORDER BY runs DESC
        """)

    def fidelity_trend(self, circuit_name: str,
                       backend: Optional[str] = None) -> list[dict]:
        """Get fidelity over time for a circuit (optionally on a specific backend)."""
        conditions = ["circuit_name = ?"]
        params: list = [circuit_name]
        if backend:
            conditions.append("backend = ?")
            params.append(backend)
        where = " AND ".join(conditions)
        return self._query(
            f"SELECT timestamp, fidelity, backend, outcome "
            f"FROM experiment_records WHERE {where} ORDER BY timestamp",
            params)

    def count(self) -> int:
        """Total number of experiment records."""
        rows = self._query("SELECT COUNT(*) as cnt FROM experiment_records")
        return rows[0]["cnt"] if rows else 0

    # ── Internal ─────────────────────────────────────────

    def _query(self, sql: str, params: list = None) -> list[dict]:
        result = self.conn.execute(sql, params or [])
        cols = [desc[0] for desc in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]
