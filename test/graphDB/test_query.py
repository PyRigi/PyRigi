"""Tests for pyrigi.graphDB.query (QueryBuilder + CompiledQuery)."""

import pytest
from pyrigi.graphDB.db import DatabaseManager
from pyrigi.graphDB.models import AndExpr, NotExpr, OrExpr, QueryFilter
from pyrigi.graphDB.query import QueryBuilder
from pyrigi.graphDB.repositories.column_registry import ColumnRegistryRepo


@pytest.fixture
def registry():
    db = DatabaseManager(":memory:")
    db.connect()
    db.bootstrap()
    return ColumnRegistryRepo(db)


class TestQueryBuilder:
    def test_select_all_default(self, registry):
        qb = QueryBuilder(registry)
        compiled = qb.compile()
        assert compiled.sql.startswith("SELECT * FROM graphs")

    def test_select_specific_columns(self, registry):
        qb = QueryBuilder(registry).select(["graph", "num_vertices"])
        compiled = qb.compile()
        assert "graph, num_vertices" in compiled.sql

    def test_where_single_filter(self, registry):
        qb = QueryBuilder(registry).where([QueryFilter("num_vertices", "=", 7)])
        compiled = qb.compile()
        assert "WHERE" in compiled.sql
        assert "num_vertices = ?" in compiled.sql
        assert compiled.params == [7]

    def test_where_multiple_filters(self, registry):
        qb = QueryBuilder(registry).where(
            [
                QueryFilter("num_vertices", "=", 7),
                QueryFilter("min_rigidity", "=", 3),
            ]
        )
        compiled = qb.compile()
        # num_vertices filter + min_rigidity expanded by custom strategy
        assert "num_vertices = ?" in compiled.sql
        assert "min_rigidity = ?" in compiled.sql
        assert compiled.params == [7, 3, -3]

    def test_where_in_operator(self, registry):
        qb = QueryBuilder(registry).where(
            [QueryFilter("num_vertices", "IN", [5, 6, 7])]
        )
        compiled = qb.compile()
        assert "IN (?, ?, ?)" in compiled.sql
        assert compiled.params == [5, 6, 7]

    def test_where_between_operator(self, registry):
        qb = QueryBuilder(registry).where(
            [QueryFilter("num_edges", "BETWEEN", (3, 10))]
        )
        compiled = qb.compile()
        assert "BETWEEN ? AND ?" in compiled.sql
        assert compiled.params == [3, 10]

    def test_where_expr_grouped_or_and(self, registry):
        expr = AndExpr(
            [
                OrExpr(
                    [
                        QueryFilter("num_vertices", "=", 5),
                        QueryFilter("num_vertices", "=", 7),
                    ]
                ),
                OrExpr(
                    [
                        QueryFilter("num_edges", ">=", 5),
                        QueryFilter("num_edges", "IS NULL", None),
                    ]
                ),
            ]
        )
        compiled = QueryBuilder(registry).where_expr(expr).compile()
        assert "(num_vertices = ? OR num_vertices = ?)" in compiled.sql
        assert "(num_edges >= ? OR num_edges IS NULL)" in compiled.sql
        assert compiled.params == [5, 7, 5]

    def test_where_any_helper_composes_with_and(self, registry):
        compiled = (
            QueryBuilder(registry)
            .where_any(
                [
                    QueryFilter("num_vertices", "=", 5),
                    QueryFilter("num_vertices", "=", 7),
                ]
            )
            .where_any(
                [
                    QueryFilter("num_edges", "=", 4),
                    QueryFilter("num_edges", "=", 5),
                ]
            )
            .compile()
        )
        assert "(num_vertices = ? OR num_vertices = ?)" in compiled.sql
        assert "(num_edges = ? OR num_edges = ?)" in compiled.sql
        assert " AND " in compiled.sql
        assert compiled.params == [5, 7, 4, 5]

    def test_where_expr_not(self, registry):
        compiled = (
            QueryBuilder(registry)
            .where_expr(NotExpr(QueryFilter("num_vertices", "=", 5)))
            .compile()
        )
        assert "WHERE (NOT num_vertices = ?)" in compiled.sql
        assert compiled.params == [5]

    def test_order_by_asc(self, registry):
        compiled = QueryBuilder(registry).order_by("num_vertices").compile()
        assert "ORDER BY num_vertices ASC" in compiled.sql

    def test_order_by_desc(self, registry):
        compiled = QueryBuilder(registry).order_by("num_edges", asc=False).compile()
        assert "ORDER BY num_edges DESC" in compiled.sql

    def test_limit(self, registry):
        compiled = QueryBuilder(registry).limit(10).compile()
        assert "LIMIT 10" in compiled.sql

    def test_limit_and_offset(self, registry):
        compiled = QueryBuilder(registry).limit(5).offset(10).compile()
        assert "LIMIT 5" in compiled.sql
        assert "OFFSET 10" in compiled.sql

    def test_chaining(self, registry):
        compiled = (
            QueryBuilder(registry)
            .select(["graph", "num_vertices"])
            .where([QueryFilter("num_vertices", "=", 5)])
            .order_by("num_edges")
            .limit(20)
            .compile()
        )
        assert "graph, num_vertices" in compiled.sql
        assert "WHERE" in compiled.sql
        assert "LIMIT 20" in compiled.sql

    def test_fetch_without_repo_raises(self, registry):
        qb = QueryBuilder(registry, repo=None)
        with pytest.raises(RuntimeError, match="fetch\\(\\)"):
            qb.fetch()

    def test_iter_fetch_without_repo_raises(self, registry):
        qb = QueryBuilder(registry, repo=None)
        with pytest.raises(RuntimeError, match="iter_fetch\\(\\)"):
            list(qb.iter_fetch())

    def test_pretty_helpers_without_repo_raise(self, registry):
        qb = QueryBuilder(registry, repo=None)
        with pytest.raises(RuntimeError, match="iter_fetch\\(\\)"):
            qb.format_results()
        with pytest.raises(RuntimeError, match="iter_fetch\\(\\)"):
            qb.pretty_print()

    def test_filter_shorthand(self, registry):
        compiled = QueryBuilder(registry).filter("num_vertices", "=", 6).compile()
        assert "num_vertices = ?" in compiled.sql
        assert compiled.params == [6]
