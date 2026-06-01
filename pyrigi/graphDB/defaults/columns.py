"""
pyrigi.graphDB.defaults.columns
~~~~~~~~~~~~~~~~~~~~~~~~~~
Declaration of the built-in :class:`~pyrigi.graphDB.models.ColumnDef`
objects.

Populator functions live in :mod:`pyrigi.graphDB.defaults.populators` so they
can be referenced by the importable path
``"pyrigi.graphDB.defaults.populators:<function_name>"`` and thus survive
across Python sessions when stored in ``column_registry.populator_ref``.
"""

from __future__ import annotations

from pyrigi.graphDB.models import ColumnDef
from pyrigi.graphDB.defaults.fetch_strategies import (
    _min_rigidity_fetch_strategy,
    _rigidity_fetch_strategy,
)
from pyrigi.graphDB.defaults.populators import (
    _compute_num_vertices,
    _compute_num_edges,
    _compute_min_degree,
    _compute_max_degree,
    _compute_rigidity,
    _compute_min_rigidity,
    _compute_global_rigidity,
)

# ---------------------------------------------------------------------------
# Default column registry
# ---------------------------------------------------------------------------

_REF = "pyrigi.graphDB.defaults.populators"
_FETCH_REF = "pyrigi.graphDB.defaults.fetch_strategies"
_RIGIDITY_VALID_OPS = frozenset({"=", "IN", "IS NULL", "IS NOT NULL"})

DEFAULT_COLUMNS: list[ColumnDef] = [
    ColumnDef(
        name="graph",
        data_type="TEXT",
        description="Graph6-encoded graph string (unique identifier)",
        is_default=True,
        populator=None,  # populated by ingestion, not by populator
        populator_ref=None,
    ),
    ColumnDef(
        name="num_vertices",
        data_type="INTEGER",
        description="Number of vertices in the graph",
        is_default=True,
        populator=_compute_num_vertices,
        populator_ref=f"{_REF}:_compute_num_vertices",
    ),
    ColumnDef(
        name="num_edges",
        data_type="INTEGER",
        description="Number of edges in the graph",
        is_default=True,
        populator=_compute_num_edges,
        populator_ref=f"{_REF}:_compute_num_edges",
    ),
    ColumnDef(
        name="min_degree",
        data_type="INTEGER",
        description="Minimum vertex degree",
        is_default=True,
        populator=_compute_min_degree,
        populator_ref=f"{_REF}:_compute_min_degree",
    ),
    ColumnDef(
        name="max_degree",
        data_type="INTEGER",
        description="Maximum vertex degree",
        is_default=True,
        populator=_compute_max_degree,
        populator_ref=f"{_REF}:_compute_max_degree",
    ),
    ColumnDef(
        name="rigidity",
        data_type="INTEGER",
        description=(
            "Maximum d such that G is d-rigid. " "G is d-rigid iff d ≤ stored value."
        ),
        is_default=True,
        populator=_compute_rigidity,
        populator_ref=f"{_REF}:_compute_rigidity",
        fetch_strategy=_rigidity_fetch_strategy,
        fetch_ref=f"{_FETCH_REF}:_rigidity_fetch_strategy",
        valid_operators=_RIGIDITY_VALID_OPS,
    ),
    ColumnDef(
        name="min_rigidity",
        data_type="INTEGER",
        description=(
            "Encoded d_min for minimal d-rigidity: "
            "-(|V|-1) if complete, d if minimally d-rigid, 0 otherwise."
        ),
        is_default=True,
        populator=_compute_min_rigidity,
        populator_ref=f"{_REF}:_compute_min_rigidity",
        fetch_strategy=_min_rigidity_fetch_strategy,
        fetch_ref=f"{_FETCH_REF}:_min_rigidity_fetch_strategy",
        valid_operators=_RIGIDITY_VALID_OPS,
    ),
    ColumnDef(
        name="global_rigidity",
        data_type="INTEGER",
        description=(
            "Maximum d such that G is globally d-rigid. "
            "G is globally d-rigid iff d ≤ stored value."
        ),
        is_default=True,
        populator=_compute_global_rigidity,
        populator_ref=f"{_REF}:_compute_global_rigidity",
        fetch_strategy=_rigidity_fetch_strategy,
        fetch_ref=f"{_FETCH_REF}:_rigidity_fetch_strategy",
        valid_operators=_RIGIDITY_VALID_OPS,
    ),
]

# Convenience lookup by name
DEFAULT_COLUMN_MAP: dict[str, ColumnDef] = {c.name: c for c in DEFAULT_COLUMNS}
