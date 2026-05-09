"""
pyrigi.graphDB.service
~~~~~~~~~~~~~~~~
User-facing facade: :class:`GraphStoreService`.

Typical usage::

    from pyrigi.graphDB import GraphStoreService, QueryFilter

    store = GraphStoreService("outputs/graph_store.db")
    store.init()

    stats = store.ingest("outputs/g6")
    print(stats)                            # IngestStats(inserted=853, ...)

    # Add a custom column (runtime callable, not persisted across sessions)
    store.add_column("density", "REAL",
                     populator=lambda row: row["num_edges"] / (row["num_vertices"] * (row["num_vertices"] - 1) / 2))
    store.populate_column("density")

    # Query
    rows = store.fetch(
        select=["graph", "num_vertices", "density"],
        filters=[QueryFilter("num_vertices", "=", 7)],
        order_by="density",
        limit=10,
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TextIO, cast

from pyrigi.graphDB.db import DatabaseManager
from pyrigi.graphDB.defaults.columns import DEFAULT_COLUMNS
from pyrigi.graphDB.ingestion import DefaultColumnComputer, G6Reader, GraphParser
from pyrigi.graphDB.models import ColumnDef, IngestStats, PopulateStats, QueryExpr, QueryFilter
from pyrigi.graphDB.query import QueryBuilder
from pyrigi.graphDB.repositories.column_registry import ColumnRegistryRepo
from pyrigi.graphDB.repositories.graph_repo import GraphRepository
from pyrigi.graphDB.utils.pretty import format_result_table, pretty_print_table

log = logging.getLogger(__name__)
_UNSET = object()


class GraphStoreService:
    """User-friendly facade for the graph database.

    Parameters
    ----------
    db_path:
        Filesystem path for the SQLite database file.
        Defaults to ``"outputs/graph_store.db"``.
        Use ``":memory:"`` for an in-memory database (useful for testing).
    batch_size:
        Default batch size for ingestion inserts.
    """

    def __init__(
        self,
        db_path: str | Path = "outputs/graph_store.db",
        batch_size: int = 500,
    ) -> None:
        self._db = DatabaseManager(db_path)
        self._batch_size = batch_size
        self._registry: Optional[ColumnRegistryRepo] = None
        self._repo: Optional[GraphRepository] = None
        self._populator_cache: dict[str, Callable[[dict], Any]] = {}
        self._fetch_cache: dict[str, Callable[[str, str, Any], tuple[str, list]]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> "GraphStoreService":
        """Open the database and bootstrap the schema.

        Must be called before any other method.  Safe to call multiple
        times (idempotent).
        """
        self._db.connect()
        self._db.bootstrap()
        self._registry = ColumnRegistryRepo(self._db)
        self._repo = GraphRepository(self._db)
        return self

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "GraphStoreService":
        return self.init()

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        source: str | Path,
        batch_size: Optional[int] = None,
    ) -> IngestStats:
        """Ingest graphs from a ``.g6`` file or directory of ``.g6`` files.

        Ingestion is idempotent: graphs already in the database (matched
        by the unique ``graph`` column) are silently skipped.

        Parameters
        ----------
        source:
            Path to a ``.g6`` / ``.g6.gz`` file or a directory
            containing such files.
        batch_size:
            Override the instance-level default batch size.

        Returns
        -------
        IngestStats:
            Summary of inserted, skipped, and errored rows.
        """
        self._require_init()
        reader = G6Reader(source)
        parser = GraphParser(strict=False)
        computer = DefaultColumnComputer()
        bs = batch_size or self._batch_size

        stats = IngestStats()
        batch: list[dict] = []

        for path in reader.files():
            stats.files_processed += 1
            for g6 in G6Reader._read_file(path):
                graph = parser.parse(g6)
                if graph is None:
                    stats.errors += 1
                    continue
                if graph.number_of_nodes() < 2:
                    log.debug("Skipping graph with < 2 vertices: %r", g6[:20])
                    stats.skipped += 1
                    continue
                batch.append(computer.compute(g6, graph))
                if len(batch) >= bs:
                    ins, sk = self._repo.insert_batch(batch)
                    stats.inserted += ins
                    stats.skipped += sk
                    batch.clear()

        if batch:
            ins, sk = self._repo.insert_batch(batch)
            stats.inserted += ins
            stats.skipped += sk

        log.info(
            "Ingest complete: files=%d inserted=%d skipped=%d errors=%d",
            stats.files_processed, stats.inserted, stats.skipped, stats.errors,
        )
        return stats

    # ------------------------------------------------------------------
    # Column management
    # ------------------------------------------------------------------

    def add_column(
        self,
        name: str,
        data_type: str = "INTEGER",
        description: str = "",
        *,
        populator: Optional[Callable[[dict], Any]] = None,
        populator_ref: Optional[str] = None,
        fetch_strategy: Optional[Callable[[str, str, Any], tuple[str, list]]] = None,
        fetch_ref: Optional[str] = None,
    ) -> ColumnDef:
        """Register a new column and add it to the ``graphs`` table.

        Parameters
        ----------
        name:
            Column name (valid SQL identifier).
        data_type:
            SQLite type affinity: ``"INTEGER"``, ``"REAL"``, ``"TEXT"``,
            ``"BLOB"``.
        description:
            Human-readable description stored in ``column_registry``.
        populator:
            Runtime callable ``(row: dict) -> scalar``.  Not persisted.
        populator_ref:
            Importable path ``"package.module:func"``.  Persisted in DB.
        fetch_strategy:
            Runtime callable ``(col, op, val) -> (sql, params)``.
        fetch_ref:
            Importable path for the fetch strategy.  Persisted in DB.

        Returns
        -------
        ColumnDef:
            The registered column definition.

        Raises
        ------
        ValueError:
            If *name* conflicts with an existing default column name.
        """
        self._require_init()

        default_names = {c.name for c in DEFAULT_COLUMNS}
        if name in default_names:
            raise ValueError(
                f"Column {name!r} is a built-in default column. "
                "To update its populator, use update_column_populator()."
            )

        col = ColumnDef(
            name=name,
            data_type=data_type,
            description=description,
            is_default=False,
            populator=populator,
            populator_ref=populator_ref,
            fetch_strategy=fetch_strategy,
            fetch_ref=fetch_ref,
        )
        self._registry.register(col)
        self._db.add_column(name, data_type)
        # Cache runtime callables so they survive the SQLite round-trip.
        if populator is not None:
            self._populator_cache[name] = populator
        if fetch_strategy is not None:
            self._fetch_cache[name] = fetch_strategy
        log.info("Registered custom column: %s (%s)", name, data_type)
        return col

    def update_column_populator(
        self,
        name: str,
        populator: Optional[Callable[[dict], Any]] = None,
        populator_ref: Optional[str] | object = _UNSET,
    ) -> None:
        """Replace the populator for any column (including defaults).

        Use this to attach your rigidity solver to the ``rigidity``,
        ``min_rigidity``, or ``global_rigidity`` columns::

            store.update_column_populator(
                "rigidity",
                populator=my_solver.compute_max_rigid_dim,
            )

        Parameters
        ----------
        name:
            Column name.
        populator:
            New runtime callable.
        populator_ref:
            New importable path (replaces the stored reference).
            If omitted, the existing persisted reference is kept.
            Pass ``None`` explicitly to clear the stored reference.
        """
        self._require_init()
        col = self._registry.get(name)
        if col is None:
            raise KeyError(f"Column {name!r} not found in registry.")
        col.populator = populator
        if populator_ref is not _UNSET:
            col.populator_ref = cast(Optional[str], populator_ref)
        self._registry.register(col)
        # Keep the runtime callable in the cache regardless of populator_ref
        if populator is not None:
            self._populator_cache[name] = populator
        elif name in self._populator_cache:
            del self._populator_cache[name]

    def list_columns(self) -> list[ColumnDef]:
        """Return all registered columns (defaults + custom)."""
        self._require_init()
        return self._registry.list_all()

    def get_column(self, name: str) -> Optional[ColumnDef]:
        """Return the :class:`ColumnDef` for *name*, or ``None``."""
        self._require_init()
        return self._registry.get(name)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def populate_column(
        self,
        column: str,
        *,
        populator: Optional[Callable[[dict], Any]] = None,
        batch_size: Optional[int] = None,
        all_rows: bool = False,
    ) -> PopulateStats:
        """Compute and store values for *column* across all matching rows.

        Parameters
        ----------
        column:
            Column name to populate.
        populator:
            Override the registered populator for this call only.
        batch_size:
            How many rows to fetch per iteration (default: instance default).
        all_rows:
            If ``True``, re-populate every row (overwrite existing values).
            If ``False`` (default), only rows where *column* IS NULL are
            updated.

        Returns
        -------
        PopulateStats:
            Summary of processed rows and any errors.

        Raises
        ------
        KeyError:
            If *column* is not registered.
        RuntimeError:
            If no populator is available for *column*.
        """
        self._require_init()
        col = self._registry.get(column)
        if col is None:
            raise KeyError(f"Column {column!r} not found in registry.")

        # Resolution order:
        #   1. explicit override passed to this call
        #   2. in-memory cache (survives SQLite round-trip for lambdas)
        #   3. populator_ref resolved via importlib (persisted across sessions)
        fn = populator or self._populator_cache.get(column) or col.resolve_populator()
        if fn is None:
            raise RuntimeError(
                f"No populator available for column {column!r}. "
                "Provide one via populate_column(..., populator=fn) or "
                "store.update_column_populator(name, populator=fn)."
            )

        stats = PopulateStats(column=column)
        iterator = self._repo.iter_all() if all_rows else self._repo.iter_unpopulated(column)

        for row in iterator:
            try:
                value = fn(row)
                self._repo.update_column(column, row["id"], value)
                stats.processed += 1
            except Exception as exc:
                log.error(
                    "Populator error for column=%s row_id=%s: %s",
                    column, row.get("id"), exc,
                )
                stats.errors += 1

        log.info(
            "Population complete: column=%s processed=%d errors=%d",
            column, stats.processed, stats.errors,
        )
        return stats

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self) -> QueryBuilder:
        """Return a fresh :class:`~pyrigi.graphDB.query.QueryBuilder` bound to this store.

        Use for fluent, composable queries::

            rows = (
                store.query()
                     .select(["graph", "num_vertices", "min_rigidity"])
                     .where([QueryFilter("num_vertices", "=", 7)])
                     .order_by("num_edges")
                     .limit(50)
                     .fetch()
            )
        """
        self._require_init()
        return QueryBuilder(
            registry=self._registry,
            repo=self._repo,
            fetch_strategy_resolver=self._resolve_runtime_fetch_strategy,
        )

    def fetch(
        self,
        select: Optional[list[str]] = None,
        filters: Optional[list[QueryFilter]] = None,
        expr: Optional[QueryExpr] = None,
        order_by: Optional[str] = None,
        asc: bool = True,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        mapper: Optional[Callable[[dict], Any]] = None,
    ) -> list[Any]:
        """Execute a query and return rows as a list of dicts.

        A convenience wrapper around :meth:`query` for one-liner usage.

        Parameters
        ----------
        select:
            Column names to return.  ``None`` means all columns (``*``).
        filters:
            List of :class:`~pyrigi.graphDB.models.QueryFilter` predicates.
        expr:
            Optional grouped boolean predicate expression.
        order_by:
            Column name to sort by.
        asc:
            Sort ascending if ``True`` (default), descending if ``False``.
        limit:
            Maximum number of rows to return.
        offset:
            Number of rows to skip (requires *limit*).
        mapper:
            Optional callable transforming each row dict to another object.
            If omitted, rows are returned as dicts.

        Returns
        -------
        list[Any]:
            Materialized query results.
        """
        return list(
            self.iter_fetch(
                select=select,
                filters=filters,
                expr=expr,
                order_by=order_by,
                asc=asc,
                limit=limit,
                offset=offset,
                mapper=mapper,
            )
        )

    def iter_fetch(
        self,
        select: Optional[list[str]] = None,
        filters: Optional[list[QueryFilter]] = None,
        expr: Optional[QueryExpr] = None,
        order_by: Optional[str] = None,
        asc: bool = True,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        mapper: Optional[Callable[[dict], Any]] = None,
    ) -> Iterator[Any]:
        """Execute a query and lazily yield results.

        Parameters are the same as :meth:`fetch`.
        """
        builder = self.query()
        if select:
            builder.select(select)
        if filters:
            builder.where(filters)
        if expr is not None:
            builder.where_expr(expr)
        if order_by:
            builder.order_by(order_by, asc=asc)
        if limit is not None:
            builder.limit(limit)
        if offset is not None:
            builder.offset(offset)
        return builder.iter_fetch(mapper=mapper)

    # ------------------------------------------------------------------
    # Stats / info
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of graphs in the database."""
        self._require_init()
        return self._repo.count()

    def count_unpopulated(self, column: str) -> int:
        """Return the number of rows where *column* is NULL."""
        self._require_init()
        return self._repo.count_where(column, is_null=True)

    def info(self) -> dict:
        """Return a summary dict for quick inspection."""
        self._require_init()
        cols = self._registry.list_all()
        return {
            "total_graphs": self.count(),
            "columns": [
                {
                    "name": c.name,
                    "type": c.data_type,
                    "default": c.is_default,
                    "description": c.description,
                    "has_populator": c.resolve_populator() is not None,
                }
                for c in cols
            ],
        }

    # ------------------------------------------------------------------
    # Pretty print helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_results(
        rows: Iterator[Any] | list[Any],
        *,
        columns: Optional[list[str]] = None,
        max_rows: Optional[int] = 20,
        max_col_width: int = 48,
        show_index: bool = False,
    ) -> str:
        """Format rows as an ASCII table string."""
        return format_result_table(
            rows,
            columns=columns,
            max_rows=max_rows,
            max_col_width=max_col_width,
            show_index=show_index,
        )

    @staticmethod
    def pretty_print_results(
        rows: Iterator[Any] | list[Any],
        *,
        columns: Optional[list[str]] = None,
        max_rows: Optional[int] = 20,
        max_col_width: int = 48,
        show_index: bool = False,
        file: Optional[TextIO] = None,
    ) -> str:
        """Pretty-print rows as an ASCII table and return the printed text."""
        return pretty_print_table(
            rows,
            columns=columns,
            max_rows=max_rows,
            max_col_width=max_col_width,
            show_index=show_index,
            file=file,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _require_init(self) -> None:
        if self._registry is None or self._repo is None:
            raise RuntimeError(
                "GraphStoreService is not initialised. Call init() first."
            )

    def _resolve_runtime_fetch_strategy(
        self,
        column: str,
    ) -> Optional[Callable[[str, str, Any], tuple[str, list]]]:
        """Return an in-memory fetch strategy for *column*, if available."""
        return self._fetch_cache.get(column)
