"""
The module checks if the given coloring is a NAC-coloring.
The algorithm is based on :prf:ref:`lem-color-components`.
"""

from typing import Collection, Iterable

import networkx as nx

from pyrigi.graph._flexibility.nac.core import (
    IntEdge,
    NACColoring,
    can_have_NAC_coloring,
)


def _check_for_almost_red_cycles(
    adj: list[list[int]],
    red_edges: Iterable[IntEdge],
    blue_edges: Iterable[IntEdge],
) -> bool:
    """
    Check if there is an almost cycle in the graph with the given coloring.

    `True` is returned if the coloring has no almost red cycles with a single blue edge.
    It is not checked whether the coloring is surjective.

    Parameters
    ----------
    adj:
        Pre-allocated adjacency list of length n (number of vertices, indexed 0..n-1).
        Cleared and rebuilt in-place from ``red_edges`` on each call.
    red_edges:
        Edges in the red color - used to create components.
    blue_edges:
        Edges in the blue color - used to check for almost cycles.
    """
    n = len(adj)
    for lst in adj:
        lst.clear()
    for u, v in red_edges:
        adj[u].append(v)
        adj[v].append(u)

    comp = [-1] * n
    cid = 0
    stack = []

    for i in range(n):
        if comp[i] != -1:
            continue

        comp[i] = cid
        stack.append(i)

        while stack:
            u = stack.pop()
            for v in adj[u]:
                if comp[v] == -1:
                    comp[v] = cid
                    stack.append(v)
        cid += 1

    for e1, e2 in blue_edges:
        if comp[e1] == comp[e2]:
            return False

    return True


def _is_NAC_coloring_impl(
    graph: nx.Graph,
    coloring: NACColoring,
) -> bool:
    """
    Check if the coloring given is a :prf:ref:`NAC-coloring <def-nac>`
    by using algorithm described in :prf:ref:`lem-color-components`.

    The algorithm checks if all the edges are in the same component.
    This is an internal implementation, so some properties like injectivity
    are not checked for performance reasons - we only search for the cycles.

    Parameters
    ----------
    graph:
    coloring:
        The coloring to check if it is a NAC coloring.
    """
    red, blue = coloring

    nodes = list(graph.nodes)
    n = len(nodes)
    index = {v: i for i, v in enumerate(nodes)}
    red_i = [(index[u], index[v]) for u, v in red]
    blue_i = [(index[u], index[v]) for u, v in blue]
    adj: list[list[int]] = [[] for _ in range(n)]

    return _check_for_almost_red_cycles(
        adj, red_i, blue_i
    ) and _check_for_almost_red_cycles(adj, blue_i, red_i)


# public facing interface
def is_NAC_coloring(
    graph: nx.Graph,
    coloring: NACColoring | dict[str, Collection[IntEdge]],
) -> bool:
    """
    Check if the coloring given is a :prf:ref:`NAC-coloring <def-nac>`
    by using algorithm described in :prf:ref:`lem-color-components`.

    The algorithm checks if all the edges are in the same component.
    This is an internal implementation, so some properties like injectivity
    are not checked for performance reasons - we only search for the cycles.

    Parameters
    ----------
    graph:
    coloring:
        The coloring to check if it is a NAC coloring.
    """
    red: Collection[IntEdge]
    blue: Collection[IntEdge]

    if isinstance(coloring, dict):
        red, blue = coloring["red"], coloring["blue"]
    else:
        red, blue = coloring

    if not can_have_NAC_coloring(graph):
        return False

    # Both colors have to be used
    if len(red) == 0 or len(blue) == 0:  # this is faster than *
        return False

    if len(red) + len(blue) != len(graph.edges):
        return False

    if isinstance(red, set) and len(red.intersection(blue)) != 0:
        return False
    else:
        # Yes, this is slower - in case you care, use sets
        for e in red:
            if e in blue:
                return False

    return _is_NAC_coloring_impl(graph, (red, blue))
