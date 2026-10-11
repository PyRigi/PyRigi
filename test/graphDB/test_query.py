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
        assert compiled.params == (7,)

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
        assert compiled.params == (7, 3, -3)

    def test_where_in_operator(self, registry):
        qb = QueryBuilder(registry).where(
            [QueryFilter("num_vertices", "IN", [5, 6, 7])]
        )
        compiled = qb.compile()
        assert "IN (?, ?, ?)" in compiled.sql
        assert compiled.params == (5, 6, 7)

    def test_where_between_operator(self, registry):
        qb = QueryBuilder(registry).where(
            [QueryFilter("num_edges", "BETWEEN", (3, 10))]
        )
        compiled = qb.compile()
        assert "BETWEEN ? AND ?" in compiled.sql
        assert compiled.params == (3, 10)

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
        assert compiled.params == (5, 7, 5)

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
        assert compiled.params == (5, 7, 4, 5)

    def test_where_expr_not(self, registry):
        compiled = (
            QueryBuilder(registry)
            .where_expr(NotExpr(QueryFilter("num_vertices", "=", 5)))
            .compile()
        )
        assert "WHERE (NOT num_vertices = ?)" in compiled.sql
        assert compiled.params == (5,)

    def test_order_by_asc(self, registry):
        compiled = QueryBuilder(registry).order_by("num_vertices").compile()
        assert "ORDER BY num_vertices ASC" in compiled.sql

    def test_order_by_desc(self, registry):
        compiled = (
            QueryBuilder(registry).order_by("num_edges", ascending=False).compile()
        )
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
        assert compiled.params == (6,)

    # ------------------------------------------------------------------
    # compile_delete
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # valid_operators validation
    # ------------------------------------------------------------------

    def test_compile_raises_for_invalid_operator_on_restricted_column(self, registry):
        from pyrigi.graphDB.models import ColumnDef

        registry.register(
            ColumnDef("restricted", "INTEGER", valid_operators=frozenset({"=", "IN"}))
        )
        qb = QueryBuilder(registry).filter("restricted", ">", 1)
        with pytest.raises(ValueError, match="not supported"):
            qb.compile()

    def test_compile_passes_for_valid_operator_on_restricted_column(self, registry):
        from pyrigi.graphDB.models import ColumnDef

        registry.register(
            ColumnDef("restricted", "INTEGER", valid_operators=frozenset({"=", "IN"}))
        )
        compiled = QueryBuilder(registry).filter("restricted", "=", 1).compile()
        assert "restricted = ?" in compiled.sql

    def test_compile_no_restriction_on_unrestricted_column(self, registry):
        compiled = QueryBuilder(registry).filter("num_vertices", ">", 3).compile()
        assert "num_vertices > ?" in compiled.sql

    def test_compile_raises_for_invalid_operator_inside_and_expr(self, registry):
        from pyrigi.graphDB.models import ColumnDef

        registry.register(
            ColumnDef("restricted", "INTEGER", valid_operators=frozenset({"="}))
        )
        qb = QueryBuilder(registry).where_expr(
            AndExpr(
                children=(
                    QueryFilter("num_vertices", "=", 5),
                    QueryFilter("restricted", ">", 1),
                )
            )
        )
        with pytest.raises(ValueError, match="not supported"):
            qb.compile()

    def test_compile_raises_for_invalid_operator_inside_or_expr(self, registry):
        from pyrigi.graphDB.models import ColumnDef

        registry.register(
            ColumnDef("restricted", "INTEGER", valid_operators=frozenset({"="}))
        )
        qb = QueryBuilder(registry).where_expr(
            OrExpr(
                children=(
                    QueryFilter("num_vertices", "=", 5),
                    QueryFilter("restricted", ">=", 1),
                )
            )
        )
        with pytest.raises(ValueError, match="not supported"):
            qb.compile()

    def test_compile_delete_rejects_invalid_operator_on_restricted_column(
        self, registry
    ):
        from pyrigi.graphDB.models import ColumnDef

        registry.register(
            ColumnDef("restricted", "INTEGER", valid_operators=frozenset({"="}))
        )
        qb = QueryBuilder(registry).filter("restricted", ">", 1)
        with pytest.raises(ValueError, match="not supported"):
            qb.compile_delete()

    def test_compile_delete_no_filters(self, registry):
        compiled = QueryBuilder(registry).compile_delete()
        assert compiled.sql == "DELETE FROM graphs"
        assert compiled.params == ()

    def test_compile_delete_with_filter(self, registry):
        compiled = (
            QueryBuilder(registry).filter("num_vertices", "=", 6).compile_delete()
        )
        assert compiled.sql == "DELETE FROM graphs WHERE num_vertices = ?"
        assert compiled.params == (6,)

    def test_compile_delete_with_expr(self, registry):
        compiled = (
            QueryBuilder(registry)
            .where_expr(
                AndExpr(
                    children=(
                        QueryFilter("num_vertices", "=", 5),
                        QueryFilter("num_edges", ">", 3),
                    )
                )
            )
            .compile_delete()
        )
        assert "DELETE FROM graphs WHERE" in compiled.sql
        assert "num_vertices = ?" in compiled.sql
        assert "num_edges > ?" in compiled.sql
        assert compiled.params == (5, 3)


# ---------------------------------------------------------------------------
# Column-name validation and compile_count
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    db = DatabaseManager(":memory:")
    db.connect()
    db.bootstrap()
    return db


@pytest.fixture
def checked_builder(db):
    """Builder that also checks columns against the real ``graphs`` table."""

    def make():
        return QueryBuilder(ColumnRegistryRepo(db), table_columns=db.table_columns)

    return make


class TestColumnValidation:
    @pytest.mark.parametrize(
        "compile_method", ["compile", "compile_delete", "compile_count"]
    )
    def test_malformed_filter_column_raises(self, registry, compile_method):
        qb = QueryBuilder(registry).filter("1=1 OR num_vertices", "=", 1)
        with pytest.raises(ValueError, match="Invalid column name"):
            getattr(qb, compile_method)()

    @pytest.mark.parametrize("column", ["COUNT(*)", "graph num_vertices", "graph, id"])
    def test_malformed_select_raises(self, registry, column):
        qb = QueryBuilder(registry).select([column])
        with pytest.raises(ValueError, match="Invalid column name"):
            qb.compile()

    def test_malformed_order_by_raises(self, registry):
        qb = QueryBuilder(registry).order_by("num_edges DESC, graph")
        with pytest.raises(ValueError, match="Invalid column name"):
            qb.compile()

    def test_malformed_column_nested_in_expression_raises(self, registry):
        expr = AndExpr(
            [
                QueryFilter("num_vertices", "=", 5),
                OrExpr(
                    [
                        QueryFilter("num_edges", "=", 4),
                        NotExpr(QueryFilter("x; --", "=", 1)),
                    ]
                ),
            ]
        )
        with pytest.raises(ValueError, match="Invalid column name"):
            QueryBuilder(registry).where_expr(expr).compile()

    def test_star_select_still_compiles(self, registry):
        assert (
            QueryBuilder(registry)
            .select(["*"])
            .compile()
            .sql.startswith("SELECT * FROM graphs")
        )

    def test_unknown_column_raises_key_error_with_table_columns(self, checked_builder):
        with pytest.raises(KeyError, match="nope"):
            checked_builder().select(["nope"]).compile()
        with pytest.raises(KeyError, match="nope"):
            checked_builder().filter("nope", "=", 1).compile_delete()
        with pytest.raises(KeyError, match="nope"):
            checked_builder().order_by("nope").compile()

    def test_existing_columns_pass_case_insensitively(self, checked_builder):
        compiled = (
            checked_builder()
            .select(["NUM_VERTICES", "id"])
            .filter("Num_Edges", ">", 3)
            .order_by("ID")
            .compile()
        )
        assert "NUM_VERTICES, id" in compiled.sql

    def test_without_table_columns_only_syntax_is_checked(self, registry):
        # unknown but well-formed names are not rejected by a bare builder
        compiled = QueryBuilder(registry).select(["not_a_column"]).compile()
        assert "not_a_column" in compiled.sql

    def test_delete_and_count_ignore_stale_select_and_order_by(self, checked_builder):
        # select/order_by are not part of DELETE or COUNT, so they are not checked
        qb = checked_builder().select(["nope"]).order_by("nope")
        assert qb.compile_delete().sql == "DELETE FROM graphs"
        assert qb.compile_count().sql == "SELECT COUNT(*) FROM graphs"


class TestCompileCount:
    def test_count_without_filters(self, registry):
        compiled = QueryBuilder(registry).compile_count()
        assert compiled.sql == "SELECT COUNT(*) FROM graphs"
        assert compiled.params == ()

    def test_count_with_filter(self, registry):
        compiled = QueryBuilder(registry).filter("num_vertices", "=", 6).compile_count()
        assert compiled.sql == "SELECT COUNT(*) FROM graphs WHERE num_vertices = ?"
        assert compiled.params == (6,)

    def test_count_ignores_select_order_limit_offset(self, registry):
        compiled = (
            QueryBuilder(registry)
            .select(["graph"])
            .filter("num_edges", ">", 3)
            .order_by("num_edges", ascending=False)
            .limit(5)
            .offset(2)
            .compile_count()
        )
        assert compiled.sql == "SELECT COUNT(*) FROM graphs WHERE num_edges > ?"
        assert compiled.params == (3,)

    def test_count_uses_rigidity_fetch_strategy(self, registry):
        compiled = QueryBuilder(registry).filter("rigidity", "=", 2).compile_count()
        assert "rigidity >= ?" in compiled.sql
        assert "rigidity = -1" in compiled.sql
