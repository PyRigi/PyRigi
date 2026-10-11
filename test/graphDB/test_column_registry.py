"""Tests for pyrigi.graphDB.repositories.column_registry (ColumnRegistryRepo)."""

import pytest

from pyrigi.graphDB.db import DatabaseManager
from pyrigi.graphDB.models import ColumnDef
from pyrigi.graphDB.repositories.column_registry import ColumnRegistryRepo


@pytest.fixture
def registry():
    db = DatabaseManager(":memory:")
    db.connect()
    db.bootstrap()
    return ColumnRegistryRepo(db)


class TestRead:
    def test_list_all_contains_defaults(self, registry):
        names = [c.name for c in registry.list_all()]
        for expected in (
            "rigidity",
            "min_rigidity",
            "global_rigidity",
            "num_vertices",
            "num_edges",
        ):
            assert expected in names

    def test_list_all_defaults_sorted_first(self, registry):
        cols = registry.list_all()
        flags = [c.is_default for c in cols]
        # All True values precede any False value
        seen_false = False
        for f in flags:
            if not f:
                seen_false = True
            assert not (seen_false and f), "default column appeared after a custom one"

    def test_list_custom_empty_initially(self, registry):
        assert registry.list_custom() == []

    def test_list_custom_returns_registered(self, registry):
        registry.register(ColumnDef("mymetric", "REAL"))
        names = [c.name for c in registry.list_custom()]
        assert "mymetric" in names

    def test_list_custom_excludes_defaults(self, registry):
        registry.register(ColumnDef("extra", "INTEGER"))
        for col in registry.list_custom():
            assert not col.is_default

    def test_get_known_column(self, registry):
        col = registry.get("rigidity")
        assert col is not None
        assert col.name == "rigidity"

    def test_get_unknown_column_returns_none(self, registry):
        assert registry.get("nonexistent") is None

    def test_exists_true_for_default(self, registry):
        assert registry.exists("num_vertices") is True

    def test_exists_false_for_unknown(self, registry):
        assert registry.exists("not_there") is False

    def test_column_names_sorted(self, registry):
        names = registry.column_names()
        assert isinstance(names, list)
        assert names == sorted(names)
        assert "rigidity" in names


class TestWrite:
    def test_register_new_column(self, registry):
        registry.register(ColumnDef("score", "INTEGER", description="A score"))
        col = registry.get("score")
        assert col is not None
        assert col.data_type == "INTEGER"
        assert col.description == "A score"

    def test_register_upserts_existing(self, registry):
        registry.register(ColumnDef("x", "INTEGER", description="old"))
        registry.register(ColumnDef("x", "REAL", description="new"))
        col = registry.get("x")
        assert col.data_type == "REAL"
        assert col.description == "new"

    def test_register_preserves_refs(self, registry):
        registry.register(
            ColumnDef(
                "tagged",
                "TEXT",
                populator_ref="mod:pop",
                fetch_ref="mod:fetch",
            )
        )
        col = registry.get("tagged")
        assert col.populator_ref == "mod:pop"
        assert col.fetch_ref == "mod:fetch"

    def test_delete_custom_column(self, registry):
        registry.register(ColumnDef("temp", "REAL"))
        assert registry.exists("temp")
        registry.delete("temp")
        assert not registry.exists("temp")

    def test_delete_default_column_is_noop(self, registry):
        registry.delete("rigidity")  # protected by is_default = 1 guard
        assert registry.exists("rigidity")

    def test_row_to_def_is_default_flag(self, registry):
        col = registry.get("num_edges")
        assert col.is_default is True

    def test_row_to_def_description_empty_string_on_null(self, registry):
        registry.register(ColumnDef("nodesc", "INTEGER"))
        col = registry.get("nodesc")
        assert col.description == ""

    def test_register_persists_valid_operators(self, registry):
        ops = frozenset({"=", "IN", "IS NULL", "IS NOT NULL"})
        registry.register(ColumnDef("restricted", "INTEGER", valid_operators=ops))
        col = registry.get("restricted")
        assert col is not None
        assert col.valid_operators == ops

    def test_register_persists_null_valid_operators(self, registry):
        registry.register(ColumnDef("unrestricted", "INTEGER", valid_operators=None))
        col = registry.get("unrestricted")
        assert col is not None
        assert col.valid_operators is None

    def test_register_overwrites_valid_operators_on_upsert(self, registry):
        from pyrigi.graphDB.models import ColumnDef

        registry.register(ColumnDef("col", "INTEGER", valid_operators=frozenset({"="})))
        assert registry.get("col").valid_operators == frozenset({"="})
        registry.register(
            ColumnDef("col", "INTEGER", valid_operators=frozenset({"=", "IN"}))
        )
        assert registry.get("col").valid_operators == frozenset({"=", "IN"})

    def test_default_rigidity_columns_have_valid_operators(self, registry):
        for name in ("rigidity", "min_rigidity", "global_rigidity"):
            col = registry.get(name)
            assert col is not None
            assert col.valid_operators == frozenset(
                {"=", "IN", "IS NULL", "IS NOT NULL"}
            )
