"""
pyrigi.graphDB.db
~~~~~~~~~~~
Database tier: manages the SQLite connection and schema lifecycle.

Responsibilities
----------------
* Open / close the sqlite3 connection.
* Bootstrap the core ``graphs`` and ``column_registry`` tables.
* Apply ``ALTER TABLE`` migrations when new custom columns are added.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from pyrigi.graphDB.constants.schema import _GRAPHS_DDL, _REGISTRY_DDL
from pyrigi.graphDB.defaults.columns import DEFAULT_COLUMNS

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DatabaseManager:
    """Manages a single SQLite connection and schema lifecycle.

    Parameters
    ----------
    db_path:
        Filesystem path to the ``.db`` file.  Use ``":memory:"`` for
        in-memory testing.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        return self._conn

    def connect(self) -> sqlite3.Connection:
        """Open the connection and enable WAL mode + foreign keys."""
        if self._conn is not None:
            return self._conn
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        self._conn = conn
        return conn

    def close(self) -> None:
        """Close the connection gracefully."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DatabaseManager":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def bootstrap(self) -> None:
        """Create core tables and seed the default registry rows.

        Idempotent — safe to call on an already-initialised database.
        """
        conn = self.connection
        with conn:
            conn.execute(_GRAPHS_DDL)
            conn.execute(_REGISTRY_DDL)
            defaults_data = [
                (c.name, c.data_type, c.description, c.populator_ref, c.fetch_ref, int(c.is_default))
                for c in DEFAULT_COLUMNS
            ]
            conn.executemany(
                """
                INSERT OR IGNORE INTO column_registry
                    (name, data_type, description, populator_ref, fetch_ref, is_default)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                defaults_data,
            )

    # ------------------------------------------------------------------
    # Schema migrations
    # ------------------------------------------------------------------

    def add_column(self, name: str, data_type: str) -> None:
        """Add a new column to ``graphs`` if it does not already exist.

        Parameters
        ----------
        name:
            Column name (must be a valid SQLite identifier).
        data_type:
            SQLite type affinity, e.g. ``"REAL"``, ``"INTEGER"``, ``"TEXT"``.
        """
        self._validate_identifier(name)
        existing = self._existing_columns()
        if name in existing:
            return  # idempotent
        with self.connection:
            self.connection.execute(
                f"ALTER TABLE graphs ADD COLUMN {name} {data_type}"
            )

    @staticmethod
    def _validate_identifier(name: str) -> None:
        if not _IDENTIFIER_RE.fullmatch(name):
            raise ValueError(f"Invalid SQL identifier: {name!r}")

    def _existing_columns(self) -> set[str]:
        """Return the set of column names currently on the ``graphs`` table."""
        cur = self.connection.execute("PRAGMA table_info(graphs)")
        return {row["name"] for row in cur.fetchall()}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        """Run a statement and return the cursor (outside of a transaction)."""
        return self.connection.execute(sql, params)

    def executemany(self, sql: str, params_seq) -> sqlite3.Cursor:
        return self.connection.executemany(sql, params_seq)
