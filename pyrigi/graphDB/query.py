"""
pyrigi.graphDB.query
~~~~~~~~~~~~~~~~~~~~
Fluent ``QueryBuilder`` and the ``CompiledQuery`` it produces.

The builder never touches the database itself; it only assembles SQL
from user-provided column names and :class:`~pyrigi.graphDB.models.filters.QueryFilter`
objects.  Raw SQL access to the database is delegated entirely to
:class:`~pyrigi.graphDB.repositories.graph_repo.GraphRepository`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Optional, TextIO

from pyrigi.graphDB.constants.identifiers import IDENTIFIER_RE
from pyrigi.graphDB.models import AndExpr, NotExpr, OrExpr, QueryExpr, QueryFilter
from pyrigi.graphDB.utils.pretty import format_result_table, pretty_print_table

if TYPE_CHECKING:
    from pyrigi.graphDB.repositories.column_registry import ColumnRegistryRepo


# ---------------------------------------------------------------------------
# Compiled output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledQuery:
    """Immutable SQL statement and its bound parameters.

    Produced by :meth:`QueryBuilder.compile` and consumed by
    :meth:`~pyrigi.graphDB.repositories.graph_repo.GraphRepository.fetch`.
    """

    sql: str
    params: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.params, tuple):
            object.__setattr__(self, "params", tuple(self.params))


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class QueryBuilder:
    """Fluent builder for SELECT queries against the ``graphs`` table.

    All methods return ``self`` so calls can be chained::

        query = (
            store.query()
                 .select(["graph", "num_vertices", "min_rigidity"])
                 .where([
                     QueryFilter("num_vertices", "=", 7),
                     QueryFilter("min_rigidity", "=", 3),
                 ])
                 .order_by("num_edges")
                 .limit(50)
        )
        rows = query.fetch()

    Parameters
    ----------
    registry:
        Used to resolve per-column fetch strategies.
    repo:
        Used only when :meth:`fetch` is called directly on the builder.
    fetch_strategy_resolver:
        Optional callable returning a runtime fetch strategy for a column,
        or ``None`` to fall back to the registry.
    table_columns:
        Optional callable returning the column names of the ``graphs`` table.
        When given, every column used in a query must exist in the table
        (compared case-insensitively, as in SQLite); otherwise only the
        syntax of column names is checked.
    """

    def __init__(
        self,
        registry: "ColumnRegistryRepo",
        repo=None,  # GraphRepository, avoided circular import via string
        fetch_strategy_resolver: Optional[
            Callable[[str], Optional[Callable[[str, str, Any], tuple[str, list]]]]
        ] = None,
        table_columns: Optional[Callable[[], set[str]]] = None,
    ) -> None:
        self._registry = registry
        self._repo = repo
        self._fetch_strategy_resolver = fetch_strategy_resolver
        self._table_columns = table_columns
        self._select: list[str] = ["*"]
        self._filters: list[QueryFilter] = []
        self._exprs: list[QueryExpr] = []
        self._order_col: Optional[str] = None
        self._order_asc: bool = True
        self._limit_val: Optional[int] = None
        self._offset_val: Optional[int] = None

    # ------------------------------------------------------------------
    # Fluent setters
    # ------------------------------------------------------------------

    def select(self, columns: list[str]) -> "QueryBuilder":
        """Choose which columns to include in the result.

        Parameters
        ----------
        columns:
            Column names only (no SQL expressions).  Pass ``["*"]`` for all
            columns (default).  Names are validated when the query is compiled.
        """
        self._select = list(columns)
        return self

    def where(self, filters: list[QueryFilter]) -> "QueryBuilder":
        """Add WHERE predicates.  Multiple calls accumulate (AND logic)."""
        self._filters.extend(filters)
        return self

    def where_expr(self, expr: QueryExpr) -> "QueryBuilder":
        """Add a grouped boolean predicate expression."""
        self._exprs.append(expr)
        return self

    def where_any(self, filters: list[QueryFilter]) -> "QueryBuilder":
        """Add an OR-group of filters.

        Multiple calls are combined with AND at the top level.
        """
        self._exprs.append(OrExpr(filters))
        return self

    def filter(self, column: str, operator: str, value: Any = None) -> "QueryBuilder":
        """Convenience single-filter shorthand."""
        self._filters.append(QueryFilter(column=column, operator=operator, value=value))
        return self

    def order_by(self, column: str, ascending: bool = True) -> "QueryBuilder":
        """Set ORDER BY clause.

        Parameters
        ----------
        column:
            A single column name.  It is validated when the query is compiled.
        ascending:
            Sort ascending if ``True`` (default), descending if ``False``.
        """
        self._order_col = column
        self._order_asc = ascending
        return self

    def limit(self, n: int) -> "QueryBuilder":
        """Set LIMIT."""
        self._limit_val = n
        return self

    def offset(self, n: int) -> "QueryBuilder":
        """Set OFFSET (requires a LIMIT to be set as well)."""
        self._offset_val = n
        return self

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile(self) -> CompiledQuery:
        """Assemble the SQL string and parameter list.

        Returns
        -------
        CompiledQuery
            Immutable (sql, params) pair ready for the repository.

        Raises
        ------
        ValueError
            If a selected, ordering or filter column name is malformed.
        KeyError
            If such a column does not exist in the ``graphs`` table.
        """
        projection = [c for c in self._select if c != "*"]
        if self._order_col:
            projection.append(self._order_col)
        self._validate_columns([*projection, *self._filter_columns()])

        col_list = ", ".join(self._select)
        sql_parts = [f"SELECT {col_list} FROM graphs"]

        where_sql, params = self._compile_where()
        if where_sql:
            sql_parts.append(where_sql)

        if self._order_col:
            direction = "ASC" if self._order_asc else "DESC"
            sql_parts.append(f"ORDER BY {self._order_col} {direction}")

        if self._limit_val is not None:
            sql_parts.append(f"LIMIT {int(self._limit_val)}")
            if self._offset_val is not None:
                sql_parts.append(f"OFFSET {int(self._offset_val)}")

        return CompiledQuery(sql=" ".join(sql_parts), params=params)

    def compile_delete(self) -> CompiledQuery:
        """Assemble a ``DELETE FROM graphs`` statement from the current filters.

        ORDER BY, LIMIT, and OFFSET are ignored because they are not valid in a
        DELETE statement.  If no filters are set, all rows are deleted.

        Returns
        -------
        CompiledQuery
            Immutable (sql, params) pair ready for the repository.

        Raises
        ------
        ValueError
            If a filter column name is malformed.
        KeyError
            If a filter column does not exist in the ``graphs`` table.
        """
        self._validate_columns(self._filter_columns())
        sql_parts = ["DELETE FROM graphs"]

        where_sql, params = self._compile_where()
        if where_sql:
            sql_parts.append(where_sql)

        return CompiledQuery(sql=" ".join(sql_parts), params=params)

    def compile_count(self) -> CompiledQuery:
        """Assemble a ``SELECT COUNT(*)`` statement from the current filters.

        SQLite counts the matching rows itself, so no rows are loaded into
        memory.  The selected columns, ORDER BY, LIMIT, and OFFSET are
        ignored.  If no filters are set, all rows are counted.

        Returns
        -------
        CompiledQuery
            Immutable (sql, params) pair ready for the repository.

        Raises
        ------
        ValueError
            If a filter column name is malformed.
        KeyError
            If a filter column does not exist in the ``graphs`` table.
        """
        self._validate_columns(self._filter_columns())
        sql_parts = ["SELECT COUNT(*) FROM graphs"]

        where_sql, params = self._compile_where()
        if where_sql:
            sql_parts.append(where_sql)

        return CompiledQuery(sql=" ".join(sql_parts), params=params)

    def _filter_columns(self) -> list[str]:
        """Return the column names used by all filters and expressions."""
        columns: list[str] = []

        def walk(expr: QueryExpr) -> None:
            if isinstance(expr, QueryFilter):
                columns.append(expr.column)
            elif isinstance(expr, (AndExpr, OrExpr)):
                for child in expr.children:
                    walk(child)
            elif isinstance(expr, NotExpr):
                walk(expr.child)

        for expr in [*self._filters, *self._exprs]:
            walk(expr)
        return columns

    def _validate_columns(self, columns: Iterable[str]) -> None:
        """Check that every column name is well formed and, if known, exists.

        Column names cannot be bound as SQL parameters, so they are checked
        here before any of them is placed into the SQL string.

        Raises
        ------
        ValueError
            If a name is not a plain identifier.
        KeyError
            If ``table_columns`` was given and a name is not a column of the
            ``graphs`` table (compared case-insensitively).
        """
        names = list(columns)
        for name in names:
            if not isinstance(name, str) or not IDENTIFIER_RE.fullmatch(name):
                raise ValueError(
                    f"Invalid column name {name!r}: only plain column names "
                    "(letters, digits and underscores) are allowed."
                )
        if self._table_columns is None or not names:
            return
        known = {c.casefold() for c in self._table_columns()}
        for name in names:
            if name.casefold() not in known:
                raise KeyError(f"Unknown column: {name!r}")

    def _compile_where(self) -> tuple[str, list[Any]]:
        """Compile the filters and expressions into a WHERE clause.

        Returns an empty string and empty parameter list when no filters or
        expressions are set, so callers can omit the clause entirely.
        """
        clauses: list[str] = []
        params: list[Any] = []
        for f in self._filters:
            fragment, fparams = self._compile_filter(f)
            clauses.append(fragment)
            params.extend(fparams)
        for expr in self._exprs:
            fragment, eparams = self._compile_expr(expr)
            clauses.append(fragment)
            params.extend(eparams)
        if not clauses:
            return "", []
        return "WHERE " + " AND ".join(clauses), params

    def _resolve_filter_strategy(
        self,
        column: str,
    ) -> Callable[[str, str, Any], tuple[str, list]]:
        strategy = None
        if self._fetch_strategy_resolver is not None:
            strategy = self._fetch_strategy_resolver(column)

        if strategy is not None:
            return strategy

        col_def = self._registry.get(column)
        if col_def is not None:
            return col_def.resolve_fetch_strategy()

        # Unknown column, fall back to pass-through
        from pyrigi.graphDB.models.resolvers import _default_fetch_strategy

        return _default_fetch_strategy

    def _compile_filter(self, filt: QueryFilter) -> tuple[str, list[Any]]:
        col_def = self._registry.get(filt.column)
        if col_def is not None and col_def.valid_operators is not None:
            if filt.operator not in col_def.valid_operators:
                allowed = sorted(col_def.valid_operators)
                raise ValueError(
                    f"Operator {filt.operator!r} is not supported for column "
                    f"{filt.column!r}. Allowed operators: {allowed}"
                )
        strategy = self._resolve_filter_strategy(filt.column)
        return strategy(filt.column, filt.operator, filt.value)

    def _compile_expr(self, expr: QueryExpr) -> tuple[str, list[Any]]:
        if isinstance(expr, QueryFilter):
            return self._compile_filter(expr)

        if isinstance(expr, AndExpr):
            fragments: list[str] = []
            params: list[Any] = []
            for child in expr.children:
                frag, child_params = self._compile_expr(child)
                fragments.append(frag)
                params.extend(child_params)
            return "(" + " AND ".join(fragments) + ")", params

        if isinstance(expr, OrExpr):
            fragments = []
            params: list[Any] = []
            for child in expr.children:
                frag, child_params = self._compile_expr(child)
                fragments.append(frag)
                params.extend(child_params)
            return "(" + " OR ".join(fragments) + ")", params

        if isinstance(expr, NotExpr):
            frag, params = self._compile_expr(expr.child)
            return f"(NOT {frag})", params

        raise TypeError(f"Unsupported query expression: {type(expr)!r}")

    # ------------------------------------------------------------------
    # Terminal operation (requires repo)
    # ------------------------------------------------------------------

    def fetch(self) -> list[dict]:
        """Compile and execute the query, returning rows as dicts.

        Only available when the builder was created by
        :meth:`~pyrigi.graphDB.service.GraphStoreService.query` (which injects
        the repository).
        """
        if self._repo is None:
            raise RuntimeError(
                "fetch() is only available on builders created by "
                "GraphStoreService.query(). Use service.fetch() instead, or pass "
                "the compiled query to GraphRepository.fetch(compiled_query)."
            )
        return list(self.iter_fetch())

    def iter_fetch(
        self,
        mapper: Optional[Callable[[dict], Any]] = None,
    ) -> Iterator[Any]:
        """Compile and execute the query, yielding rows lazily.

        Parameters
        ----------
        mapper:
            Optional callable transforming each row dict to another object.
            If omitted, raw row dicts are yielded.
        """
        if self._repo is None:
            raise RuntimeError(
                "iter_fetch() is only available on builders created by "
                "GraphStoreService.query(). Use service.iter_fetch() instead, or pass "
                "the compiled query to GraphRepository.iter_fetch(compiled_query)."
            )
        for row in self._repo.iter_fetch(self.compile()):
            yield mapper(row) if mapper is not None else row

    def format_results(
        self,
        *,
        columns: Optional[list[str]] = None,
        max_rows: Optional[int] = 20,
        max_col_width: int = 48,
        show_index: bool = False,
    ) -> str:
        """Return this query's results as a formatted ASCII table string."""
        return format_result_table(
            self.iter_fetch(),
            columns=columns,
            max_rows=max_rows,
            max_col_width=max_col_width,
            show_index=show_index,
        )

    def pretty_print(
        self,
        *,
        columns: Optional[list[str]] = None,
        max_rows: Optional[int] = 20,
        max_col_width: int = 48,
        show_index: bool = False,
        file: Optional[TextIO] = None,
    ) -> str:
        """Execute and print this query's results as an ASCII table."""
        return pretty_print_table(
            self.iter_fetch(),
            columns=columns,
            max_rows=max_rows,
            max_col_width=max_col_width,
            show_index=show_index,
            file=file,
        )
