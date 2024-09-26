"""
This is a module for providing common types of graphs.
"""

import networkx as nx
from pyrigi.graph import Graph


def Cycle(n: int) -> Graph:
    """Return the cycle graph on n vertices."""
    return Graph(nx.cycle_graph(n))


def Complete(n: int) -> Graph:
    """Return the complete graph on n vertices."""
    return Graph(nx.complete_graph(n))


def Path(n: int) -> Graph:
    """Return the path graph with n vertices."""
    return Graph(nx.path_graph(n))


def CompleteBipartite(m: int, n: int) -> Graph:
    """Return the complete bipartite graph on m+n vertices."""
    return Graph(nx.complete_multipartite_graph(m, n))


def K33plusEdge() -> Graph:
    """Return the complete bipartite graph on 3+3 vertices with an extra edge."""
    G = CompleteBipartite(3, 3)
    G.add_edge(0, 1)
    return G


def Diamond() -> Graph:
    """Return the complete graph on 4 vertices minus an edge."""
    return Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])


def ThreePrism() -> Graph:
    """Return the 3-prism graph."""
    return Graph(
        [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (0, 3), (1, 4), (2, 5)]
    )


def ThreePrismPlusEdge() -> Graph:
    """Return the 3-prism graph with one extra edge."""
    return Graph(
        [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (0, 3), (1, 4), (2, 5), (0, 5)]
    )


def CubeWithDiagonal() -> Graph:
    """Return the graph given by the skeleton of the cube with a main diagonal."""
    return Graph(
        [(0, 1), (1, 2), (2, 3), (0, 3)]
        + [(4, 5), (5, 6), (6, 7), (4, 7)]
        + [(0, 4), (1, 5), (2, 6), (3, 7)]
        + [(0, 6)]
    )


def DoubleBanana(d: int = 3) -> Graph:
    """Return the d-dimensional double banana graph."""
    K = Complete(d + 2)
    K1 = K.edge_list()
    K2 = [[e[0] + d, e[1] + d] for e in K1]
    DB = Graph(K1 + K2)
    DB.delete_edge([d, d + 1])
    return DB
