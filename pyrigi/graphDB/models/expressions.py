"""
pyrigi.graphDB.models.expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Boolean expression nodes for grouped query predicates.

These are composed from :class:`~pyrigi.graphDB.models.filters.QueryFilter`
leaves and used by :class:`~pyrigi.graphDB.query.QueryBuilder.where_expr`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pyrigi.graphDB.models.filters import QueryFilter


@dataclass(frozen=True)
class AndExpr:
    """Logical AND over one or more child expressions."""

    children: tuple["QueryExpr", ...]

    def __init__(self, children: Iterable["QueryExpr"]) -> None:
        items = tuple(children)
        if not items:
            raise ValueError("AndExpr requires at least one child expression.")
        object.__setattr__(self, "children", items)


@dataclass(frozen=True)
class OrExpr:
    """Logical OR over one or more child expressions."""

    children: tuple["QueryExpr", ...]

    def __init__(self, children: Iterable["QueryExpr"]) -> None:
        items = tuple(children)
        if not items:
            raise ValueError("OrExpr requires at least one child expression.")
        object.__setattr__(self, "children", items)


@dataclass(frozen=True)
class NotExpr:
    """Logical NOT over one child expression."""

    child: "QueryExpr"


QueryExpr = QueryFilter | AndExpr | OrExpr | NotExpr


def all_of(*exprs: QueryExpr) -> AndExpr:
    """Convenience helper to build an :class:`AndExpr`."""
    return AndExpr(exprs)


def any_of(*exprs: QueryExpr) -> OrExpr:
    """Convenience helper to build an :class:`OrExpr`."""
    return OrExpr(exprs)


def not_(expr: QueryExpr) -> NotExpr:
    """Convenience helper to build a :class:`NotExpr`."""
    return NotExpr(expr)
