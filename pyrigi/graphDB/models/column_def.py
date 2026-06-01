"""
pyrigi.graphDB.models.column_def
~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`ColumnDef` — describes one column in the ``graphs`` table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pyrigi.graphDB.models.resolvers import _default_fetch_strategy, _import_callable


@dataclass
class ColumnDef:
    """Describes one column in the graphs table.

    Parameters
    ----------
    name:
        SQL column name (also the key in result dicts).
    data_type:
        SQLite type affinity string, e.g. ``"INTEGER"``, ``"REAL"``,
        ``"TEXT"``.
    description:
        Human-readable explanation of what the column stores.
    is_default:
        ``True`` for the built-in columns.  Custom columns added by
        users are ``False``.
    populator:
        Runtime callable ``(row: dict) -> scalar``.  Takes a row dict (which
        always contains at least ``id`` and ``graph``) and returns the value
        to be written.  Not persisted across sessions.
    populator_ref:
        Importable dotted path ``"package.module:function"``.  Resolved
        lazily and persisted in the ``column_registry`` table so the
        populator survives across sessions.
    fetch_strategy:
        Runtime callable ``(column, operator, value) -> (sql_fragment,
        params)``.  Lets columns override how a ``QueryFilter`` is
        translated to SQL.  Defaults to a simple ``column op ?`` pass-
        through if ``None``.
    fetch_ref:
        Importable dotted path for ``fetch_strategy``, persisted in DB.
    """

    name: str
    data_type: str = "INTEGER"
    description: str = ""
    is_default: bool = False

    # ---- populator --------------------------------------------------------
    populator: Optional[Callable[[dict], Any]] = field(default=None, repr=False)
    populator_ref: Optional[str] = None

    # ---- fetch strategy ---------------------------------------------------
    fetch_strategy: Optional[Callable[[str, str, Any], tuple[str, list]]] = field(
        default=None, repr=False
    )
    fetch_ref: Optional[str] = None

    # ---- operator whitelist -----------------------------------------------
    valid_operators: Optional[frozenset[str]] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def resolve_populator(self) -> Optional[Callable[[dict], Any]]:
        """Return the populator callable, resolving ``populator_ref`` if needed."""
        if self.populator is not None:
            return self.populator
        if self.populator_ref:
            return _import_callable(self.populator_ref)
        return None

    def resolve_fetch_strategy(
        self,
    ) -> Callable[[str, str, Any], tuple[str, list]]:
        """Return the fetch-strategy callable, falling back to a default pass-through."""
        if self.fetch_strategy is not None:
            return self.fetch_strategy
        if self.fetch_ref:
            return _import_callable(self.fetch_ref)
        return _default_fetch_strategy
