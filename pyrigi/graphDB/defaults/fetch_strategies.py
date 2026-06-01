"""
pyrigi.graphDB.defaults.fetch_strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Custom fetch-strategy callables for the built-in rigidity columns.

Each function has the signature::

    strategy(column: str, operator: str, value: Any) -> tuple[str, list]

and is registered on the corresponding :class:`~pyrigi.graphDB.models.ColumnDef`
via both ``fetch_strategy`` (runtime) and ``fetch_ref`` (persisted in the DB).

Rigidity / global-rigidity encoding
-------------------------------------
The stored value is the *maximum* d such that G is d-rigid (or ``-1``
for complete graphs, which are rigid in all dimensions).  Because
complete graphs are rigid in every dimension, they must appear in the
results of any ``= d`` or ``IN [...]`` query.  The ``=`` and ``IN``
operators therefore expand to include the ``-1`` sentinel:

* ``= d``       →  ``(column = d OR column = -1)``
* ``IN [...]``  →  ``(column IN (...) OR column = -1)``

``IS NULL`` and ``IS NOT NULL`` pass through unchanged (``NULL`` means
the column has not been populated yet).

Minimal-rigidity encoding
--------------------------
The stored value uses a three-way encoding:

* ``-(n-1)``  — complete graph on n vertices (minimally d-rigid for all d ≥ n-1)
* ``d``       — non-complete graph that is minimally d-rigid
* ``0``       — not minimally rigid for any d

A graph is minimally d-rigid iff ``stored = d`` OR
``(stored < 0 AND stored >= -d)``  (the complete-graph case).
The ``=`` operator expands to cover both branches.  The ``IN``
operator expands analogously: for ``IN [d1, ..., dk]`` the complete-graph
branch uses ``stored >= -max(d1, ..., dk)`` as the lower bound.
"""

from __future__ import annotations

from typing import Any

from pyrigi.graphDB.models.resolvers import _default_fetch_strategy


def _rigidity_fetch_strategy(
    column: str, operator: str, value: Any
) -> tuple[str, list]:
    """Fetch strategy for ``rigidity`` and ``global_rigidity`` columns.

    Expands ``=`` and ``IN`` to include complete graphs (stored as ``-1``).
    ``IS NULL`` and ``IS NOT NULL`` fall back to the default pass-through.
    """
    op = operator.upper()
    if op == "=":
        return f"({column} = ? OR {column} = -1)", [value]
    if op == "IN":
        placeholders = ", ".join("?" * len(value))
        return f"({column} IN ({placeholders}) OR {column} = -1)", list(value)
    return _default_fetch_strategy(column, operator, value)


def _min_rigidity_fetch_strategy(
    column: str, operator: str, value: Any
) -> tuple[str, list]:
    """Fetch strategy for the ``min_rigidity`` column.

    Expands ``=`` and ``IN`` to cover both the non-complete case and the
    complete-graph case.

    * ``= d``        → ``(stored = d OR (stored < 0 AND stored >= -d))``
    * ``IN [d1,...]``→ ``(stored IN (...) OR (stored < 0 AND stored >= -max_d))``

    All other operators fall back to the default pass-through.
    """
    op = operator.upper()
    if op == "=":
        return (
            f"({column} = ? OR ({column} < 0 AND {column} >= ?))",
            [value, -value],
        )
    if op == "IN":
        placeholders = ", ".join("?" * len(value))
        max_d = max(value)
        return (
            f"({column} IN ({placeholders}) OR ({column} < 0 AND {column} >= ?))",
            list(value) + [-max_d],
        )
    return _default_fetch_strategy(column, operator, value)
