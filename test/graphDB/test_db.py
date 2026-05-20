"""Tests for pyrigi.graphDB.db (DatabaseManager) and GraphRepository."""

import pytest
from pyrigi.graphDB.db import DatabaseManager
from pyrigi.graphDB.repositories.graph_repo import GraphRepository


@pytest.fixture
def db():
    mgr = DatabaseManager(":memory:")
    mgr.connect()
    mgr.bootstrap()
    yield mgr
    mgr.close()


class TestDatabaseManager:
    def test_bootstrap_creates_tables(self, db):
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "graphs" in tables
        assert "column_registry" in tables

    def test_bootstrap_is_idempotent(self, db):
        db.bootstrap()  # second call must not raise
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "graphs" in tables

    def test_default_columns_in_registry(self, db):
        rows = db.execute("SELECT name FROM column_registry").fetchall()
        names = {r[0] for r in rows}
        expected = {
            "graph",
            "num_vertices",
            "num_edges",
            "min_degree",
            "max_degree",
            "rigidity",
            "min_rigidity",
            "global_rigidity",
        }
        assert expected.issubset(names)

    def test_add_column(self, db):
        db.add_column("my_col", "REAL")
        cols = db._existing_columns()
        assert "my_col" in cols

    def test_add_column_idempotent(self, db):
        db.add_column("dup_col", "INTEGER")
        db.add_column("dup_col", "INTEGER")  # should not raise
        cols = db._existing_columns()
        assert "dup_col" in cols

    def test_add_column_invalid_identifier_raises(self, db):
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            db.add_column("bad-name", "INTEGER")

    def test_context_manager(self):
        with DatabaseManager(":memory:") as mgr:
            mgr.bootstrap()
            tables = {
                row[0]
                for row in mgr.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "graphs" in tables


class TestGraphRepository:
    def test_add_column_invalid_identifier_raises(self, db):
        repo = GraphRepository(db)
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            repo._assert_known_column("bad-name!")

    def test_insert_batch_empty_is_noop(self, db):
        repo = GraphRepository(db)
        inserted, skipped = repo.insert_batch([])
        assert inserted == 0 and skipped == 0

    def test_iter_all_with_specific_columns(self, db):
        repo = GraphRepository(db)
        repo.insert_batch(
            [
                {
                    "graph": "Bw",
                    "num_vertices": 3,
                    "num_edges": 3,
                    "min_degree": 2,
                    "max_degree": 2,
                }
            ]
        )
        rows = list(repo.iter_all(columns=["graph", "num_vertices"]))
        assert len(rows) == 1
        assert set(rows[0].keys()) == {"graph", "num_vertices"}
