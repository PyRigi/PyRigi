"""Row mappers that decode a fetched row's ``graph`` column into graph objects.

Pass one of these as the ``mapper`` argument of
:meth:`~pyrigi.graphDB.service.GraphStoreService.fetch` or ``iter_fetch`` to get graph
objects back instead of the stored graph6 strings::

    from pyrigi.graphDB import to_networkx, to_pyrigi

    graphs = store.fetch(select=["graph"], mapper=to_pyrigi)

Both read the fixed ``graph`` identifier column, so the query must select it (include
``"graph"`` in ``select``, or use ``select=None``).
"""

from __future__ import annotations

import networkx as nx


def to_networkx(row: dict) -> nx.Graph:
    """Fetch mapper: decode a row's ``graph`` (graph6) column to a networkx Graph."""
    if "graph" not in row:
        raise KeyError(
            "row has no 'graph' column to decode; add 'graph' to the query "
            "select (or use select=None)"
        )
    return nx.from_graph6_bytes(row["graph"].encode("ascii"))


def to_pyrigi(row: dict):
    """Fetch mapper: decode a row's ``graph`` (graph6) column to a pyrigi Graph."""
    from pyrigi.graph import Graph  # lazy import to avoid import-time cycles

    return Graph(to_networkx(row))
