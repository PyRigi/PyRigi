"""
The module checks if the coloring given is a NAC coloring.
The main entry point is _is_NAC_coloring_impl
"""

from typing import *

import networkx as nx

from pyrigi.data_type import Edge
from pyrigi.graph.flexibility.nac.existence import check_NAC_constrains

NACColoring: TypeAlias = Tuple[Collection[Edge], Collection[Edge]]


def _check_for_almost_red_cycles(
    G: nx.Graph,
    red_edges: Iterable[Edge],
    blue_edges: Iterable[Edge],
) -> bool:
    """
    Checks if there is an almost cycle in the graph given with the given coloring.
    Does not check if the coloring is surjective.
    Returns true if the coloring has no such cycles..
    """
    G.clear_edges()
    G.add_edges_from(red_edges)

    component_mapping: Dict[int, int] = {}
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
    Check if the coloring given is a NAC coloring.
    The algorithm checks if all the edges are in the same component.

    This is an internal implementation, so some properties like injectivity
    are not checked for performance reasons - we only search for the cycles.

    Parameters:
    ----------
        coloring: the coloring to check if it is a NAC coloring.
        allow_non_surjective: if True, allows the coloring to be non-surjective.
            This can be useful for checking subgraphs - the can use only one color.
    ----------


    (TODO format)
    """
    red, blue = coloring

    # TODO NAC reimplement advanced graph vertices caching in PyRigi
    # # 43% speedup (from base solution, current work around was not yet compared)
    # # !!! make sure !!! this graph is cleared before every run
    # # this also makes the whole NAC coloring search thread insecure
    # # things will break if you add vertices while another search
    # # is still running

    # G = graph._graph_is_NAC_coloring

    # # somehow this if is faster than setting things outside once
    # if G is None:
    #     G = nx.Graph()
    #     G.add_nodes_from(graph.nodes)
    #     graph._graph_is_NAC_coloring = G

    # Workaround
    G = nx.Graph()
    G.add_nodes_from(graph.nodes)

    return _check_for_almost_red_cycles(G, red, blue) and _check_for_almost_red_cycles(
        G, blue, red
    )


def is_NAC_coloring(
    graph: nx.Graph,
    coloring: NACColoring | Dict[str, Collection[Edge]],
) -> bool:
    """
    Check if the coloring given is a NAC coloring.
    The algorithm checks if all the edges are in the same component.

    Parameters:
    ----------
        coloring: the coloring to check if it is a NAC coloring.
    ----------


    (TODO format)
    """
    red: Collection[Edge]
    blue: Collection[Edge]

    if type(coloring) == dict:
        red, blue = coloring["red"], coloring["blue"]
    else:
        red, blue = coloring
    assert type(red) == type(blue)

    if not check_NAC_constrains(graph):
        return False

    # Both colors have to be used
    if len(red) == 0 or len(blue) == 0:  # this is faster than *
        return False

    if len(red) + len(blue) != len(graph.edges):
        return False

    if type(red) == set and len(red.intersection(blue)) != 0:
        return False
    else:
        # Yes, this is slower - in case you care, use sets
        for e in red:
            if e in blue:
                return False

    # graph._graph_is_NAC_coloring = None

    return _is_NAC_coloring_impl(graph, (red, blue))
