"""QuantumGPT data store — DuckDB schema + logger + query API.

Persists every tool call, circuit run, health check, and agent session
into a local DuckDB file for offline analysis and W&B sync.

Usage:
    from data.store import DataStore
    db = DataStore()                          # default: data/quantumgpt.duckdb
    db.log_health(backend, health_dict)
    db.log_circuit_run(backend, run_dict)
    db.log_agent_session(session_dict)
    results = db.query("SELECT * FROM circuit_runs WHERE fidelity < 0.8")
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import duckdb

from data.experiment_record import (
    ExperimentRecord, ExperimentStore, EXPERIMENT_RECORDS_SQL,
)


DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "quantumgpt.duckdb")

SCHEMA_SQL = """
-- Backend health snapshots
CREATE TABLE IF NOT EXISTS health_snapshots (
    id              VARCHAR PRIMARY KEY,
    ts              TIMESTAMP DEFAULT current_timestamp,
    backend         VARCHAR NOT NULL,
    num_qubits      INTEGER,
    avg_1q_error    DOUBLE,
    avg_2q_error    DOUBLE,
    avg_readout_error DOUBLE,
    avg_t1_us       DOUBLE,
    avg_t2_us       DOUBLE,
    calibration_age_min DOUBLE,
    drift_score     DOUBLE,
    raw_json        VARCHAR
);

-- Per-qubit properties
CREATE TABLE IF NOT EXISTS qubit_properties (
    id              VARCHAR PRIMARY KEY,
    ts              TIMESTAMP DEFAULT current_timestamp,
    health_id       VARCHAR,  -- FK to health_snapshots
    backend         VARCHAR NOT NULL,
    qubit           INTEGER NOT NULL,
    t1_us           DOUBLE,
    t2_us           DOUBLE,
    readout_error   DOUBLE,
    gate_errors_json VARCHAR
);

-- Circuit execution results
CREATE TABLE IF NOT EXISTS circuit_runs (
    id              VARCHAR PRIMARY KEY,
    ts              TIMESTAMP DEFAULT current_timestamp,
    session_id      VARCHAR,  -- FK to agent_sessions
    backend         VARCHAR NOT NULL,
    circuit_name    VARCHAR NOT NULL,
    shots           INTEGER,
    fidelity        DOUBLE,
    transpiled_depth INTEGER,
    transpiled_gates INTEGER,
    top_counts_json VARCHAR,
    metadata_json   VARCHAR
);

-- Agent sessions
CREATE TABLE IF NOT EXISTS agent_sessions (
    id              VARCHAR PRIMARY KEY,
    ts_start        TIMESTAMP,
    ts_end          TIMESTAMP,
    model           VARCHAR,
    provider        VARCHAR,
    backend         VARCHAR,
    user_prompt     VARCHAR,
    final_answer    VARCHAR,
    num_tool_calls  INTEGER,
    total_tokens    INTEGER,
    elapsed_seconds DOUBLE,
    tool_calls_json VARCHAR
);

-- Drift events (detected state changes)
CREATE TABLE IF NOT EXISTS drift_events (
    id              VARCHAR PRIMARY KEY,
    ts              TIMESTAMP DEFAULT current_timestamp,
    backend         VARCHAR NOT NULL,
    drift_score     DOUBLE,
    severity        VARCHAR,
    fidelity_before DOUBLE,
    fidelity_after  DOUBLE,
    suggestions_json VARCHAR
);
"""


def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DataStore:
    """DuckDB-backed data store for QuantumGPT experiment data."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute(SCHEMA_SQL)
        self.conn.execute(EXPERIMENT_RECORDS_SQL)
        self.experiments = ExperimentStore(self.conn)

    def close(self):
        self.conn.close()

    # ─── Health snapshots ────────────────────────────────

    def log_health(self, health: dict) -> str:
        """Log a health check result. Returns the record ID."""
        rid = _uid()
        self.conn.execute("""
            INSERT INTO health_snapshots
            (id, ts, backend, num_qubits, avg_1q_error, avg_2q_error,
             avg_readout_error, avg_t1_us, avg_t2_us, calibration_age_min,
             drift_score, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            rid, _now(),
            health.get("backend", "unknown"),
            health.get("num_qubits"),
            health.get("avg_1q_error"),
            health.get("avg_2q_error"),
            health.get("avg_readout_error"),
            health.get("avg_t1_us"),
            health.get("avg_t2_us"),
            health.get("calibration_age_minutes"),
            health.get("drift_score"),
            json.dumps(health),
        ])
        return rid

    # ─── Qubit properties ────────────────────────────────

    def log_qubit_properties(self, backend: str, qubits: list[dict],
                              health_id: Optional[str] = None) -> list[str]:
        """Log per-qubit properties. Returns list of record IDs."""
        ids = []
        for q in qubits:
            rid = _uid()
            self.conn.execute("""
                INSERT INTO qubit_properties
                (id, ts, health_id, backend, qubit, t1_us, t2_us,
                 readout_error, gate_errors_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                rid, _now(), health_id, backend,
                q.get("qubit"),
                q.get("t1_us"),
                q.get("t2_us"),
                q.get("readout_error"),
                json.dumps(q.get("gate_errors", {})),
            ])
            ids.append(rid)
        return ids

    # ─── Circuit runs ────────────────────────────────────

    def log_circuit_run(self, run: dict, session_id: Optional[str] = None) -> str:
        """Log a circuit execution result. Returns the record ID."""
        rid = _uid()
        self.conn.execute("""
            INSERT INTO circuit_runs
            (id, ts, session_id, backend, circuit_name, shots, fidelity,
             transpiled_depth, transpiled_gates, top_counts_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            rid, _now(), session_id,
            run.get("metadata", {}).get("backend", "unknown"),
            run.get("circuit", "unknown"),
            run.get("shots"),
            run.get("fidelity"),
            run.get("transpiled_depth"),
            run.get("metadata", {}).get("transpiled_gate_count"),
            json.dumps(run.get("top_counts", {})),
            json.dumps(run.get("metadata", {})),
        ])
        return rid

    # ─── Agent sessions ──────────────────────────────────

    def log_agent_session(self, session: dict) -> str:
        """Log an agent session. Returns the record ID."""
        rid = _uid()
        self.conn.execute("""
            INSERT INTO agent_sessions
            (id, ts_start, ts_end, model, provider, backend, user_prompt,
             final_answer, num_tool_calls, total_tokens, elapsed_seconds,
             tool_calls_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            rid,
            session.get("ts_start", _now()),
            session.get("ts_end", _now()),
            session.get("model"),
            session.get("provider"),
            session.get("backend"),
            session.get("user_prompt"),
            session.get("final_answer"),
            session.get("num_tool_calls"),
            session.get("total_tokens"),
            session.get("elapsed_seconds"),
            json.dumps(session.get("tool_calls", [])),
        ])
        return rid

    # ─── Drift events ────────────────────────────────────

    def log_drift_event(self, event: dict) -> str:
        """Log a drift / diagnosis event. Returns the record ID."""
        rid = _uid()
        self.conn.execute("""
            INSERT INTO drift_events
            (id, ts, backend, drift_score, severity,
             fidelity_before, fidelity_after, suggestions_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            rid, _now(),
            event.get("backend", "unknown"),
            event.get("drift_score"),
            event.get("severity"),
            event.get("fidelity_before"),
            event.get("fidelity_after"),
            json.dumps(event.get("suggestions", [])),
        ])
        return rid

    # ─── Query API ───────────────────────────────────────

    def query(self, sql: str, params: list = None) -> list[dict]:
        """Run a SQL query and return results as list of dicts."""
        result = self.conn.execute(sql, params or [])
        cols = [desc[0] for desc in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def recent_runs(self, limit: int = 10) -> list[dict]:
        return self.query(
            "SELECT * FROM circuit_runs ORDER BY ts DESC LIMIT ?", [limit])

    def fidelity_trend(self, circuit_name: str = None, limit: int = 50) -> list[dict]:
        if circuit_name:
            return self.query(
                "SELECT ts, circuit_name, fidelity, transpiled_depth, backend "
                "FROM circuit_runs WHERE circuit_name = ? ORDER BY ts DESC LIMIT ?",
                [circuit_name, limit])
        return self.query(
            "SELECT ts, circuit_name, fidelity, transpiled_depth, backend "
            "FROM circuit_runs ORDER BY ts DESC LIMIT ?", [limit])

    def health_history(self, backend: str = None, limit: int = 50) -> list[dict]:
        if backend:
            return self.query(
                "SELECT * FROM health_snapshots WHERE backend = ? ORDER BY ts DESC LIMIT ?",
                [backend, limit])
        return self.query(
            "SELECT * FROM health_snapshots ORDER BY ts DESC LIMIT ?", [limit])

    def best_qubits(self, backend: str, top_n: int = 10) -> list[dict]:
        """Find qubits with best T1 in most recent snapshot."""
        return self.query("""
            SELECT qubit, t1_us, t2_us, readout_error
            FROM qubit_properties
            WHERE backend = ?
            ORDER BY ts DESC, t1_us DESC
            LIMIT ?
        """, [backend, top_n])

    def worst_qubits(self, backend: str, top_n: int = 10) -> list[dict]:
        """Find qubits with worst readout error."""
        return self.query("""
            SELECT qubit, t1_us, t2_us, readout_error
            FROM qubit_properties
            WHERE backend = ?
            ORDER BY ts DESC, readout_error DESC
            LIMIT ?
        """, [backend, top_n])

    def session_summary(self) -> list[dict]:
        return self.query("""
            SELECT id, model, backend, num_tool_calls, total_tokens,
                   elapsed_seconds, substr(user_prompt, 1, 80) as prompt_preview
            FROM agent_sessions
            ORDER BY ts_start DESC
            LIMIT 20
        """)

    def table_counts(self) -> dict:
        """Quick overview of how much data is stored."""
        tables = ["health_snapshots", "qubit_properties", "circuit_runs",
                   "agent_sessions", "drift_events"]
        counts = {}
        for t in tables:
            r = self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
            counts[t] = r[0]
        return counts
