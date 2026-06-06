"""
pyrigi.graphDB.models.filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`QueryFilter` — a single WHERE-clause predicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyrigi.graphDB.constants.operators import VALID_OPERATORS


@dataclass
class QueryFilter:
    """A single WHERE-clause predicate.

    Parameters
    ----------
    column:
        Name of the column to filter on.
    operator:
        One of ``=``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``IN``,
        ``BETWEEN``, ``LIKE``, ``IS NULL``, ``IS NOT NULL``.
    value:
        The right-hand side value.  For ``IN`` pass a list/tuple; for
        ``BETWEEN`` pass a 2-tuple ``(lo, hi)``; for ``IS NULL`` /
        ``IS NOT NULL`` pass ``None``.
    """

    column: str
    operator: str
    value: Any = None

    def __post_init__(self) -> None:
        op = self.operator.upper()
        if op not in VALID_OPERATORS:
            raise ValueError(
                f"Unsupported operator {self.operator!r}. "
                f"Valid: {sorted(VALID_OPERATORS)}"
            )
        self.operator = op
