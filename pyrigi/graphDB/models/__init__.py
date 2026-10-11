"""
pyrigi.graphDB.models
~~~~~~~~~~~~~~~
Pure data classes shared across all layers.

Sub-modules
-----------
column_def  :class:`ColumnDef`
filters     :class:`QueryFilter` + ``VALID_OPERATORS``
expressions : grouped boolean nodes (``AndExpr``, ``OrExpr``, ``NotExpr``)
stats       :class:`IngestStats`, :class:`PopulateStats`
resolvers   ``_import_callable``, ``_default_fetch_strategy``

All public names (and the private helpers used in tests) are re-exported
here so that existing ``from pyrigi.graphDB.models import ...`` calls continue
to work without any changes.
"""

from __future__ import annotations

from pyrigi.graphDB.models.column_def import ColumnDef
from pyrigi.graphDB.models.expressions import (
    AndExpr,
    NotExpr,
    OrExpr,
    QueryExpr,
    all_of,
    any_of,
    not_,
)
from pyrigi.graphDB.models.filters import QueryFilter
from pyrigi.graphDB.models.resolvers import _default_fetch_strategy, _import_callable
from pyrigi.graphDB.models.stats import IngestStats, PopulateStats

# Also re-export VALID_OPERATORS so callers can do
# ``from pyrigi.graphDB.models import VALID_OPERATORS`` if needed.
from pyrigi.graphDB.constants.operators import VALID_OPERATORS

__all__ = [
    "ColumnDef",
    "QueryFilter",
    "QueryExpr",
    "AndExpr",
    "OrExpr",
    "NotExpr",
    "all_of",
    "any_of",
    "not_",
    "IngestStats",
    "PopulateStats",
    "VALID_OPERATORS",
    # private helpers re-exported for backwards compatibility
    "_import_callable",
    "_default_fetch_strategy",
]
