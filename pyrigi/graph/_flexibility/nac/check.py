"""
The module checks if the coloring given is a NAC-coloring.
Algorithm based on :prf:ref:`<lem-color-components>` is used.
"""

from typing import Collection, Iterable

import networkx as nx

from pyrigi.graph._flexibility.nac.core import (
    IntEdge,
    NACColoring,
    can_have_NAC_coloring,
)


def _check_for_almost_red_cycles(
    G: nx.Graph,
    red_edges: Iterable[IntEdge],
    blue_edges: Iterable[IntEdge],
) -> bool:
    """
    Checks if there is an almost cycle in the graph given with the given coloring.
    Does not check if the coloring is surjective.

    Parameters
    ----------
    G:
        The graph to check.
    red_edges:
        Edges in the red color - used to create components.
    blue_edges:
        Edges in the blue color - used to check for almost cycles.

    Returns
    -------
    `True` if the coloring has no almost cycles with a single blue edge.

    Suggested Improvement
    ----------------------
    Keep cached graph with no edges for multiple runs,
    this has a potential of a big performance gain.
    """
    G.clear_edges()
    G.add_edges_from(red_edges)

    component_mapping: dict[int, int] = {}
    vertices: Iterable[int]
    for i, vertices in enumerate(nx.components.connected_components(G)):
        for v in vertices:
            component_mapping[v] = i

    for e1, e2 in blue_edges:
        if component_mapping[e1] == component_mapping[e2]:
            return False
    return True


def _is_NAC_coloring_impl(
    graph: nx.Graph,
    coloring: NACColoring,
) -> bool:
    """
    Check if the coloring given is a :prf:ref:`NAC-coloring <def-nac>`
    by using algorithm described in :prf:ref:`<lem-color-components>`.

    The algorithm checks if all the edges are in the same component.
    This is an internal implementation, so some properties like injectivity
    are not checked for performance reasons - we only search for the cycles.

    Parameters:
    ----------
    coloring:
        The coloring to check if it is a NAC coloring.
    allow_non_surjective: if `True`, return `True` also
        For non-surjective (monochromatic) colorings.
        This can be useful for checking subgraphs.
    """
    red, blue = coloring

    # This improves performance as it takes significantly longer to create
    # the graph when edges are added whole vertices are missing.
    # This approaches shares the vertices among multiple runs.
    # The performance can be improved even more if this graph is cached
    # for each graph as usually this check is run multiple times
    # on the same graph for many colorings.
    # This caching causes memory leaks unless the temporary graph is deleted
    # manually or the original graph is cleared.
    # Performance gain was ~40% percent in my tests half a year ago.
    G = nx.Graph()
    G.add_nodes_from(graph.nodes)

    return _check_for_almost_red_cycles(G, red, blue) and _check_for_almost_red_cycles(
        G, blue, red
    )


# public facing interface
def is_NAC_coloring(
    graph: nx.Graph,
    coloring: NACColoring | dict[str, Collection[IntEdge]],
) -> bool:
    """
    Check if the coloring given is a :prf:ref:`NAC-coloring <def-nac>`
    by using algorithm described in :prf:ref:`<lem-color-components>`.

    The algorithm checks if all the edges are in the same component.
    This is an internal implementation, so some properties like injectivity
    are not checked for performance reasons - we only search for the cycles.

    Parameters:
    ----------
    coloring:
        The coloring to check if it is a NAC coloring.
    allow_non_surjective: if `True`, return `True` also
        For non-surjective (monochromatic) colorings.
        This can be useful for checking subgraphs.
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
