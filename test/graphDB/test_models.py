"""Tests for pyrigi.graphDB.models."""

import pytest
import networkx as nx
import pyrigi.graphDB.defaults.populators as default_populators
from pyrigi.graphDB.models import (
    AndExpr,
    ColumnDef,
    IngestStats,
    NotExpr,
    OrExpr,
    PopulateStats,
    QueryFilter,
    _default_fetch_strategy,
    all_of,
    any_of,
    not_,
)


class TestQueryFilter:
    def test_valid_operators(self):
        for op in [
            "=",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
            "IN",
            "BETWEEN",
            "LIKE",
            "IS NULL",
            "IS NOT NULL",
        ]:
            qf = QueryFilter("col", op, None)
            assert qf.operator == op.upper()

    def test_invalid_operator_raises(self):
        with pytest.raises(ValueError, match="Unsupported operator"):
            QueryFilter("col", "~~", 5)

    def test_operator_normalised_to_upper(self):
        qf = QueryFilter("col", "like", "%foo%")
        assert qf.operator == "LIKE"


class TestDefaultFetchStrategy:
    def test_simple_equal(self):
        sql, params = _default_fetch_strategy("col", "=", 5)
        assert sql == "col = ?"
        assert params == [5]

    def test_in_operator(self):
        sql, params = _default_fetch_strategy("col", "IN", [1, 2, 3])
        assert sql == "col IN (?, ?, ?)"
        assert params == [1, 2, 3]

    def test_between_operator(self):
        sql, params = _default_fetch_strategy("col", "BETWEEN", (3, 8))
        assert sql == "col BETWEEN ? AND ?"
        assert params == [3, 8]

    def test_is_null(self):
        sql, params = _default_fetch_strategy("col", "IS NULL", None)
        assert sql == "col IS NULL"
        assert params == []

    def test_is_not_null(self):
        sql, params = _default_fetch_strategy("col", "IS NOT NULL", None)
        assert sql == "col IS NOT NULL"
        assert params == []


class TestColumnDef:
    def test_resolve_runtime_populator(self):
        def fn(_):  # noqa: U101
            return 42

        col = ColumnDef("x", populator=fn)
        assert col.resolve_populator() is fn

    def test_resolve_ref_populator(self):
        col = ColumnDef(
            "x",
            populator_ref="pyrigi.graphDB.defaults.populators:_compute_num_vertices",
        )
        resolved = col.resolve_populator()
        assert callable(resolved)

    def test_no_populator_returns_none(self):
        col = ColumnDef("x")
        assert col.resolve_populator() is None

    def test_default_fetch_strategy_is_passthrough(self):
        col = ColumnDef("x")
        strategy = col.resolve_fetch_strategy()
        sql, params = strategy("x", "=", 7)
        assert "x" in sql
        assert params == [7]


class TestStats:
    def test_ingest_stats_defaults(self):
        s = IngestStats()
        assert s.inserted == 0 and s.skipped == 0 and s.errors == 0

    def test_populate_stats_defaults(self):
        s = PopulateStats(column="rigidity")
        assert s.column == "rigidity"
        assert s.processed == 0


class TestDefaultPopulators:
    def test_structural_populators_decode_once_per_row(self, monkeypatch):
        call_count = {"n": 0}
        original = nx.from_graph6_bytes

        def counted_decode(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(nx, "from_graph6_bytes", counted_decode)
        row = {"graph": "Bw"}

        assert default_populators._compute_num_vertices(row) == 3
        assert default_populators._compute_num_edges(row) == 3
        assert default_populators._compute_min_degree(row) == 2
        assert default_populators._compute_max_degree(row) == 2
        assert call_count["n"] == 1


class TestQueryExpressions:
    def test_and_expr_requires_children(self):
        with pytest.raises(ValueError, match="at least one"):
            AndExpr([])

    def test_or_expr_requires_children(self):
        with pytest.raises(ValueError, match="at least one"):
            OrExpr([])

    def test_expression_helpers(self):
        f1 = QueryFilter("num_vertices", "=", 5)
        f2 = QueryFilter("num_vertices", "=", 7)
        and_expr = all_of(f1, f2)
        or_expr = any_of(f1, f2)
        not_expr = not_(f1)

        assert isinstance(and_expr, AndExpr)
        assert isinstance(or_expr, OrExpr)
        assert isinstance(not_expr, NotExpr)
