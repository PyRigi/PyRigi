"""
pyrigi.graphDB.defaults.populators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Populator functions for the built-in
:class:`~pyrigi.graphDB.models.column_def.ColumnDef` objects.

Each function takes a row dict (at minimum ``{"id": ..., "graph": ...}``)
and returns the computed value for its column.

the fast path during initial ingestion is
:class:`~pyrigi.graphDB.ingestion.default_computer.DefaultColumnComputer`, which
already has the decoded :class:`networkx.Graph` in memory.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Structural column populators
# ---------------------------------------------------------------------------


def _decoded_graph(row: dict):
    """Decode graph6 once per row and cache it for sibling populators."""
    cached = row.get("_decoded_graph")
    if cached is not None:
        return cached
    import networkx as nx

    graph = nx.from_graph6_bytes(row["graph"].encode("ascii"))
    row["_decoded_graph"] = graph
    return graph


def _compute_num_vertices(row: dict) -> int:
    g = _decoded_graph(row)
    return g.number_of_nodes()


def _compute_num_edges(row: dict) -> int:
    g = _decoded_graph(row)
    return g.number_of_edges()


def _compute_min_degree(row: dict) -> int:
    g = _decoded_graph(row)
    degrees = [d for _, d in g.degree()]
    return min(degrees) if degrees else 0


def _compute_max_degree(row: dict) -> int:
    g = _decoded_graph(row)
    degrees = [d for _, d in g.degree()]
    return max(degrees) if degrees else 0


# ---------------------------------------------------------------------------
# Rigidity populators
# ---------------------------------------------------------------------------


def _compute_rigidity(row: dict):
    """Maximum d such that G is d-rigid. Returns -1 for complete graphs (infinite)."""
    from pyrigi.graph._rigidity.generic import max_rigid_dimension
    import sympy

    result = max_rigid_dimension(_decoded_graph(row))
    return -1 if result == sympy.oo else int(result)


def _compute_min_rigidity(row: dict):
    """Encoded d_min for minimal rigidity per the database encoding convention."""
    from pyrigi.graphDB.small_graphs import _min_rigidity_dimension_encoding

    return _min_rigidity_dimension_encoding(_decoded_graph(row))


def _compute_global_rigidity(row: dict):
    """Maximum d such that G is globally d-rigid. Returns -1 for complete graphs."""
    from pyrigi.graph._rigidity.global_ import max_globally_rigid_dimension
    import sympy

    result = max_globally_rigid_dimension(_decoded_graph(row))
    return -1 if result == sympy.oo else int(result)
