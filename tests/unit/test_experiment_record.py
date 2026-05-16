"""Tests for ExperimentRecord and ExperimentStore (Day 13)."""

import json
import os
import tempfile

import duckdb
import pytest

from data.experiment_record import (
    ExperimentRecord,
    ExperimentStore,
    ExperimentType,
    ExperimentOutcome,
    EXPERIMENT_RECORDS_SQL,
)


@pytest.fixture
def conn():
    """In-memory DuckDB connection."""
    c = duckdb.connect(":memory:")
    c.execute(EXPERIMENT_RECORDS_SQL)
    yield c
    c.close()


@pytest.fixture
def store(conn):
    return ExperimentStore(conn)


# ── ExperimentRecord dataclass ───────────────────────────────

class TestExperimentRecord:
    def test_defaults(self):
        r = ExperimentRecord()
        assert r.experiment_type == "circuit_benchmark"
        assert r.success is False
        assert r.outcome == "success"
        assert len(r.id) == 16
        assert r.fidelity is None
        assert r.tool_calls == []

    def test_full_record(self):
        r = ExperimentRecord(
            experiment_type=ExperimentType.CIRCUIT_BENCHMARK.value,
            backend="FakeBrisbane",
            circuit_name="ghz_5",
            num_qubits=5,
            problem="Run GHZ-5 with 4096 shots",
            tags=["benchmark", "ghz"],
            plan=["check_health", "run_circuit"],
            tool_calls=[{"tool": "run_circuit", "input": {"circuit_name": "ghz_5"}}],
            fidelity=0.927,
            success=True,
            outcome=ExperimentOutcome.SUCCESS.value,
            metrics={"transpiled_depth": 23, "shots": 4096},
            backend_snapshot={"avg_t1_us": 224.5},
            summary="GHZ-5 on FakeBrisbane: fidelity=0.927",
            lessons=["Good fidelity, no action needed"],
        )
        assert r.fidelity == 0.927
        assert r.backend == "FakeBrisbane"
        assert "benchmark" in r.tags

    def test_to_dict_serializes_json(self):
        r = ExperimentRecord(
            tags=["a", "b"],
            plan=["step1"],
            tool_calls=[{"x": 1}],
            metrics={"m": 0.5},
            fits={"pi_amp": 0.3},
            backend_snapshot={"t1": 200},
            lessons=["learned something"],
        )
        d = r.to_dict()
        # JSON fields should be strings
        assert isinstance(d["tags"], str)
        assert json.loads(d["tags"]) == ["a", "b"]
        assert isinstance(d["metrics"], str)
        assert isinstance(d["fits"], str)

    def test_from_dict_roundtrip(self):
        r = ExperimentRecord(
            backend="FakeKyiv",
            circuit_name="qft_4",
            fidelity=0.999,
            tags=["qft", "test"],
            metrics={"depth": 15},
        )
        d = r.to_dict()
        r2 = ExperimentRecord.from_dict(d)
        assert r2.backend == "FakeKyiv"
        assert r2.fidelity == 0.999
        assert r2.tags == ["qft", "test"]
        assert r2.metrics == {"depth": 15}


# ── ExperimentStore ──────────────────────────────────────────

class TestExperimentStore:
    def test_save_and_get(self, store):
        r = ExperimentRecord(
            backend="FakeBrisbane",
            circuit_name="ghz_5",
            fidelity=0.92,
            success=True,
            summary="test save",
        )
        rid = store.save(r)
        assert rid == r.id

        fetched = store.get(rid)
        assert fetched is not None
        assert fetched.backend == "FakeBrisbane"
        assert fetched.fidelity == 0.92

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent_id") is None

    def test_recent(self, store):
        for i in range(5):
            store.save(ExperimentRecord(
                backend="FakeBrisbane",
                circuit_name=f"circ_{i}",
                fidelity=0.9 + i * 0.01,
            ))
        recs = store.recent(limit=3)
        assert len(recs) == 3

    def test_by_circuit(self, store):
        store.save(ExperimentRecord(backend="FB", circuit_name="ghz_5", fidelity=0.9))
        store.save(ExperimentRecord(backend="FB", circuit_name="qft_4", fidelity=0.99))
        store.save(ExperimentRecord(backend="FB", circuit_name="ghz_5", fidelity=0.91))

        ghz = store.by_circuit("ghz_5")
        assert len(ghz) == 2
        assert all(r.circuit_name == "ghz_5" for r in ghz)

    def test_by_backend(self, store):
        store.save(ExperimentRecord(backend="FakeBrisbane", circuit_name="ghz_5"))
        store.save(ExperimentRecord(backend="FakeKyiv", circuit_name="ghz_5"))
        store.save(ExperimentRecord(backend="FakeBrisbane", circuit_name="qft_4"))

        fb = store.by_backend("FakeBrisbane")
        assert len(fb) == 2

    def test_by_type(self, store):
        store.save(ExperimentRecord(
            experiment_type=ExperimentType.CIRCUIT_BENCHMARK.value,
            backend="FB", circuit_name="ghz_5"))
        store.save(ExperimentRecord(
            experiment_type=ExperimentType.DRIFT_DIAGNOSIS.value,
            backend="FB"))
        store.save(ExperimentRecord(
            experiment_type=ExperimentType.RABI_TUNEUP.value,
            backend="FB"))

        benchmarks = store.by_type(ExperimentType.CIRCUIT_BENCHMARK.value)
        assert len(benchmarks) == 1

    def test_search_filters(self, store):
        store.save(ExperimentRecord(
            backend="FakeBrisbane", circuit_name="ghz_5",
            fidelity=0.95, success=True))
        store.save(ExperimentRecord(
            backend="FakeBrisbane", circuit_name="ghz_5",
            fidelity=0.7, success=True))
        store.save(ExperimentRecord(
            backend="FakeKyiv", circuit_name="ghz_5",
            fidelity=0.85, success=True))

        # min fidelity
        high = store.search(min_fidelity=0.9)
        assert len(high) == 1
        assert high[0].fidelity == 0.95

        # backend + circuit
        kyiv_ghz = store.search(backend="FakeKyiv", circuit_name="ghz_5")
        assert len(kyiv_ghz) == 1

        # success only
        all_success = store.search(success_only=True)
        assert len(all_success) == 3

    def test_save_batch(self, store):
        recs = [
            ExperimentRecord(backend="FB", circuit_name=f"c{i}", fidelity=0.9)
            for i in range(4)
        ]
        ids = store.save_batch(recs)
        assert len(ids) == 4
        assert store.count() == 4

    def test_stats(self, store):
        store.save(ExperimentRecord(
            backend="FB", circuit_name="ghz_5", fidelity=0.9, success=True))
        store.save(ExperimentRecord(
            backend="FB", circuit_name="qft_4", fidelity=0.99, success=True))
        store.save(ExperimentRecord(
            backend="FK", circuit_name="ghz_5", fidelity=0.8, success=True))

        s = store.stats()
        assert s["total_experiments"] == 3
        assert s["unique_backends"] == 2
        assert s["unique_circuits"] == 2
        assert s["successes"] == 3
        assert 0.89 < s["avg_fidelity"] < 0.91  # (0.9+0.99+0.8)/3

    def test_stats_by_circuit(self, store):
        store.save(ExperimentRecord(backend="FB", circuit_name="ghz_5", fidelity=0.9))
        store.save(ExperimentRecord(backend="FB", circuit_name="ghz_5", fidelity=0.95))
        store.save(ExperimentRecord(backend="FB", circuit_name="qft_4", fidelity=0.99))

        by_c = store.stats_by_circuit()
        assert len(by_c) == 2
        ghz_stats = next(r for r in by_c if r["circuit_name"] == "ghz_5")
        assert ghz_stats["runs"] == 2
        assert 0.92 < ghz_stats["avg_fidelity"] < 0.93

    def test_stats_by_backend(self, store):
        store.save(ExperimentRecord(backend="FakeBrisbane", circuit_name="g", fidelity=0.9))
        store.save(ExperimentRecord(backend="FakeKyiv", circuit_name="g", fidelity=0.95))

        by_b = store.stats_by_backend()
        assert len(by_b) == 2

    def test_fidelity_trend(self, store):
        for i in range(5):
            store.save(ExperimentRecord(
                backend="FB", circuit_name="ghz_5",
                fidelity=0.9 + i * 0.01))
        trend = store.fidelity_trend("ghz_5")
        assert len(trend) == 5
        # Should be ordered by timestamp
        fids = [r["fidelity"] for r in trend]
        assert fids == sorted(fids)  # ascending

    def test_count(self, store):
        assert store.count() == 0
        store.save(ExperimentRecord(backend="FB"))
        assert store.count() == 1


# ── Integration with DataStore ───────────────────────────────

class TestDataStoreIntegration:
    def test_datastore_has_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.duckdb")
            from data.store import DataStore
            ds = DataStore(db_path)
            assert hasattr(ds, "experiments")
            assert isinstance(ds.experiments, ExperimentStore)

            # Save via experiments sub-store
            ds.experiments.save(ExperimentRecord(
                backend="FB", circuit_name="ghz_5", fidelity=0.9))
            assert ds.experiments.count() == 1
            ds.close()

    def test_table_exists_in_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.duckdb")
            from data.store import DataStore
            ds = DataStore(db_path)
            tables = ds.query(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name = 'experiment_records'")
            assert len(tables) == 1
            ds.close()
