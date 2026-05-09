"""
pyrigi.graphDB.ingestion.default_computer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`DefaultColumnComputer` — computes the always-populated default
columns in a single pass over a decoded networkx Graph.
"""
from __future__ import annotations

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "networkx is required for graph parsing. "
        "Install it with: pip install networkx"
    ) from exc


class DefaultColumnComputer:
    """Computes the values for the default columns from a networkx Graph.

    The always-populated columns are computed in a single pass over
    the graph.
    """

    def compute(self, g6: str, graph: nx.Graph) -> dict:
        """Return a dict ready for insertion into the ``graphs`` table.

        Parameters
        ----------
        g6:
            The raw graph6 string (stored verbatim).
        graph:
            The decoded networkx Graph.
        """
        degrees = [d for _, d in graph.degree()]
        return {
            "graph":        g6,
            "num_vertices": graph.number_of_nodes(),
            "num_edges":    graph.number_of_edges(),
            "min_degree":   min(degrees) if degrees else 0,
            "max_degree":   max(degrees) if degrees else 0,
        }
