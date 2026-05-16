"""Tests for pyrigi.graphDB.db (DatabaseManager)."""

import pytest
from pyrigi.graphDB.db import DatabaseManager


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
