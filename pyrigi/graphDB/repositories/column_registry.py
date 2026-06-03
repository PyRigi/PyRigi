"""
pyrigi.graphDB.repositories.column_registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CRUD operations for the ``column_registry`` table.

This repository is the single source of truth for which columns exist,
their types, import references, and whether they are default or custom.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from pyrigi.graphDB.models import ColumnDef

if TYPE_CHECKING:
    from pyrigi.graphDB.db import DatabaseManager


class ColumnRegistryRepo:
    """Data-access layer for ``column_registry``.

    Parameters
    ----------
    db:
        An already-connected :class:`~pyrigi.graphDB.db.DatabaseManager`.
    """

    def __init__(self, db: "DatabaseManager") -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_all(self) -> list[ColumnDef]:
        """Return every registered column, defaults first."""
        cur = self._db.execute(
            "SELECT * FROM column_registry ORDER BY is_default DESC, name ASC"
        )
        return [self._row_to_def(row) for row in cur.fetchall()]

    def list_custom(self) -> list[ColumnDef]:
        """Return only user-added (non-default) columns."""
        cur = self._db.execute(
            "SELECT * FROM column_registry WHERE is_default = 0 ORDER BY name ASC"
        )
        return [self._row_to_def(row) for row in cur.fetchall()]

    def get(self, name: str) -> Optional[ColumnDef]:
        """Return the :class:`ColumnDef` for *name*, or ``None`` if not found."""
        cur = self._db.execute("SELECT * FROM column_registry WHERE name = ?", (name,))
        row = cur.fetchone()
        return self._row_to_def(row) if row else None

    def exists(self, name: str) -> bool:
        cur = self._db.execute("SELECT 1 FROM column_registry WHERE name = ?", (name,))
        return cur.fetchone() is not None

    def column_names(self) -> list[str]:
        cur = self._db.execute("SELECT name FROM column_registry ORDER BY name ASC")
        return [row[0] for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def register(self, col: ColumnDef) -> None:
        """Upsert a :class:`ColumnDef` into the registry.

        If a column with the same name already exists the record is
        updated; otherwise a new row is inserted.
        """
        now = datetime.now(tz=timezone.utc).isoformat()
        valid_ops_str = (
            "|".join(sorted(col.valid_operators))
            if col.valid_operators is not None
            else None
        )
        with self._db.connection:
            self._db.execute(
                """
                INSERT INTO column_registry
                    (name, data_type, description, populator_ref, fetch_ref,
                     valid_operators, is_default, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    data_type       = excluded.data_type,
                    description     = excluded.description,
                    populator_ref   = excluded.populator_ref,
                    fetch_ref       = excluded.fetch_ref,
                    valid_operators = excluded.valid_operators
                """,
                (
                    col.name,
                    col.data_type,
                    col.description,
                    col.populator_ref,
                    col.fetch_ref,
                    valid_ops_str,
                    int(col.is_default),
                    now,
                ),
            )

    def delete(self, name: str) -> None:
        """Remove a custom column registration from ``column_registry``.

        .. note::
            This removes only the registry row.  Dropping the corresponding
            SQL column from the ``graphs`` table is handled by the service
            layer (:meth:`GraphStoreService.drop_column`), which calls
            :meth:`DatabaseManager.drop_column` after this method.
        """
        with self._db.connection:
            self._db.execute(
                "DELETE FROM column_registry WHERE name = ? AND is_default = 0",
                (name,),
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_def(row) -> ColumnDef:
        raw = row["valid_operators"]
        valid_ops = frozenset(raw.split("|")) if raw else None
        return ColumnDef(
            name=row["name"],
            data_type=row["data_type"],
            description=row["description"] or "",
            populator_ref=row["populator_ref"],
            fetch_ref=row["fetch_ref"],
            valid_operators=valid_ops,
            is_default=bool(row["is_default"]),
        )
