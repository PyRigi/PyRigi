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
The stored value is the *maximum* d such that G is d-rigid (or ``NULL``
for complete graphs, which are rigid in all dimensions).  A graph is
d-rigid iff ``d ≤ stored_value``, so "give me all 2-rigid graphs"
translates to ``rigidity >= 2 OR rigidity IS NULL``.  The ``>=`` and
``>`` operators therefore need to include the ``NULL`` case; all other
operators pass through to the default strategy.

Minimal-rigidity encoding
--------------------------
The stored value uses a three-way encoding:

* ``-(n-1)``  — complete graph on n vertices (minimally d-rigid for all d ≥ n-1)
* ``d``       — non-complete graph that is minimally d-rigid
* ``0``       — not minimally rigid for any d

A graph is minimally d-rigid iff ``stored = d`` OR
``(stored < 0 AND stored >= -d)``  (the complete-graph case).
The ``=`` operator is therefore expanded to cover both branches; all
other operators pass through to the default strategy.
"""

from __future__ import annotations

from typing import Any

from pyrigi.graphDB.models.resolvers import _default_fetch_strategy


def _rigidity_fetch_strategy(
    column: str, operator: str, value: Any
) -> tuple[str, list]:
    """Fetch strategy for ``rigidity`` and ``global_rigidity`` columns.

    Handles ``>=`` and ``>`` to include complete graphs (stored as ``NULL``).
    All other operators fall back to the default pass-through.
    """
    op = operator.upper()
    if op == ">=":
        return f"({column} >= ? OR {column} IS NULL)", [value]
    if op == ">":
        return f"({column} > ? OR {column} IS NULL)", [value]
    return _default_fetch_strategy(column, operator, value)


def _min_rigidity_fetch_strategy(
    column: str, operator: str, value: Any
) -> tuple[str, list]:
    """Fetch strategy for the ``min_rigidity`` column.

    Expands the ``=`` operator to cover both the non-complete case
    (``stored = d``) and the complete-graph case
    (``stored < 0 AND stored >= -d``).
    All other operators fall back to the default pass-through.
    """
    op = operator.upper()
    if op == "=":
        return (
            f"({column} = ? OR ({column} < 0 AND {column} >= ?))",
            [value, -value],
        )
    return _default_fetch_strategy(column, operator, value)
