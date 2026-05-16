"""
pyrigi.graphDB.repositories.graph_repo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CRUD and query operations for the ``graphs`` table.

All SQL lives here.  No business logic.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from pyrigi.graphDB.db import DatabaseManager
    from pyrigi.graphDB.query import CompiledQuery


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class GraphRepository:
    """Data-access layer for the ``graphs`` table.

    Parameters
    ----------
    db:
        An already-connected :class:`~pyrigi.graphDB.db.DatabaseManager`.
    """

    def __init__(self, db: "DatabaseManager") -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Bulk insert
    # ------------------------------------------------------------------

    def insert_batch(self, rows: list[dict]) -> tuple[int, int]:
        """Insert a batch of graph rows, skipping duplicates.

        Parameters
        ----------
        rows:
            Each dict must have keys matching the default column names
            (``graph``, ``num_vertices``, ``num_edges``, ``min_degree``,
            ``max_degree``).

        Returns
        -------
        (inserted, skipped):
            Counts of new rows and rows already present.
        """
        if not rows:
            return 0, 0

        keys = list(rows[0].keys())
        for key in keys:
            self._assert_known_column(key)
        placeholders = ", ".join("?" * len(keys))
        col_list = ", ".join(keys)

        before = self.count()
        with self._db.connection:
            self._db.executemany(
                f"INSERT OR IGNORE INTO graphs ({col_list}) VALUES ({placeholders})",
                [tuple(r[k] for k in keys) for r in rows],
            )
        after = self.count()
        inserted = after - before
        skipped = len(rows) - inserted
        return inserted, skipped

    # ------------------------------------------------------------------
    # Single-column update
    # ------------------------------------------------------------------

    def update_column(self, column: str, row_id: int, value: Any) -> None:
        """Set ``graphs.{column} = value`` for the row with ``id = row_id``."""
        self._assert_known_column(column)
        with self._db.connection:
            self._db.execute(
                f"UPDATE graphs SET {column} = ? WHERE id = ?",
                (value, row_id),
            )

    def update_column_batch(self, column: str, updates: list[tuple[Any, int]]) -> None:
        """Apply multiple ``(value, id)`` updates in a single transaction."""
        self._assert_known_column(column)
        with self._db.connection:
            self._db.executemany(
                f"UPDATE graphs SET {column} = ? WHERE id = ?",
                updates,
            )

    # ------------------------------------------------------------------
    # Iteration helpers for population
    # ------------------------------------------------------------------

    def iter_all(self, columns: list[str] | None = None) -> Iterator[dict]:
        """Iterate over *all* rows, yielding plain dicts.

        Parameters
        ----------
        columns:
            List of column names to include.  Defaults to ``*`` (all
            columns) so populators can reference any existing field.
        """
        if columns:
            for column in columns:
                self._assert_known_column(column)
            cols = ", ".join(columns)
        else:
            cols = "*"
        cur = self._db.execute(f"SELECT {cols} FROM graphs ORDER BY id ASC")
        for row in cur:
            yield dict(row)

    def iter_unpopulated(self, column: str) -> Iterator[dict]:
        """Iterate over rows where *column* is ``NULL``.

        Yields full row dicts (all columns) so populators can reference
        any existing field (e.g. ``num_edges``, ``num_vertices``).
        """
        self._assert_known_column(column)
        cur = self._db.execute(
            f"SELECT * FROM graphs WHERE {column} IS NULL ORDER BY id ASC"
        )
        for row in cur:
            yield dict(row)

    # ------------------------------------------------------------------
    # Fetch / query
    # ------------------------------------------------------------------

    def fetch(self, query: "CompiledQuery") -> list[dict]:
        """Execute a compiled query and return rows as a list of dicts.

        Parameters
        ----------
        query:
            A :class:`~pyrigi.graphDB.query.CompiledQuery` produced by
            :class:`~pyrigi.graphDB.query.QueryBuilder`.
        """
        return list(self.iter_fetch(query))

    def iter_fetch(self, query: "CompiledQuery") -> Iterator[dict]:
        """Execute a compiled query and yield row dicts lazily."""
        cur = self._db.execute(query.sql, query.params)
        for row in cur:
            yield dict(row)

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of rows in ``graphs``."""
        cur = self._db.execute("SELECT COUNT(*) FROM graphs")
        return cur.fetchone()[0]

    def count_where(self, column: str, is_null: bool = True) -> int:
        """Count rows where *column* IS NULL or IS NOT NULL."""
        self._assert_known_column(column)
        pred = "IS NULL" if is_null else "IS NOT NULL"
        cur = self._db.execute(f"SELECT COUNT(*) FROM graphs WHERE {column} {pred}")
        return cur.fetchone()[0]

    def _assert_known_column(self, name: str) -> None:
        if not _IDENTIFIER_RE.fullmatch(name):
            raise ValueError(f"Invalid SQL identifier: {name!r}")
        if name not in self._existing_columns():
            raise KeyError(f"Unknown column: {name!r}")

    def _existing_columns(self) -> set[str]:
        cur = self._db.execute("PRAGMA table_info(graphs)")
        return {row["name"] for row in cur.fetchall()}
