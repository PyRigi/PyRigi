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


def CubeWithDiagonal():
    """Return the graph given by the skeleton of the cube with a main diagonal."""
    return Graph(
        [(0, 1), (1, 2), (2, 3), (0, 3)]
        + [(4, 5), (5, 6), (6, 7), (4, 7)]
        + [(0, 4), (1, 5), (2, 6), (3, 7)]
        + [(0, 6)]
    )


def CompleteMinusOne(n: int) -> Graph:
    """Return the complete graph on n vertices minus one edge."""
    G = Complete(n)
    G.delete_edge((0, 1))
    return G


def Octahedral():
    """Return the graph given by the skeleton of an octahedron."""
    return Graph(
        [
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 4),
            (2, 5),
            (3, 4),
            (3, 5),
        ]
    )


def K66MinusPerfectMatching():
    """Create a Complete bipartite graph minus a perfect matching
    by deleting six non-incident edges."""
    G = CompleteBipartite(6, 6)
    G.delete_edges([(i, i + 6) for i in range(0, 6)])
    return G


def C8FlexibleWithFixedVertex():
    """Create C_8-symmetric graph with a fixed vertex which satisfies the
    expected symmetry-adapted Laman count for rotation but is infinitessimally
    flexible."""
    return Graph(
        [
            (0, 1),
            (0, 3),
            (0, 5),
            (0, 7),
            (1, 6),
            (1, 4),
            (1, 2),
            (2, 5),
            (2, 7),
            (2, 3),
            (3, 6),
            (3, 4),
            (4, 7),
            (4, 5),
            (5, 6),
            (6, 7),
            (8, 9),
            (8, 10),
            (8, 11),
            (8, 12),
            (8, 14),
            (8, 15),
            (8, 16),
            (9, 0),
            (9, 11),
            (9, 15),
            (10, 12),
            (10, 1),
            (10, 16),
            (11, 2),
            (11, 13),
            (12, 3),
            (12, 14),
            (13, 4),
            (13, 15),
            (14, 5),
            (14, 16),
            (15, 6),
            (16, 7),
        ]
    )


def ThreeConnectedR3Circuit():
    """Create a 3-connected R_3-circuit that is hypothesized to be the smallest."""
    return Graph(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 5),
            (0, 6),
            (0, 8),
            (0, 9),
            (0, 11),
            (0, 12),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 10),
            (1, 11),
            (1, 12),
            (2, 3),
            (2, 4),
            (3, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (5, 6),
            (5, 7),
            (6, 7),
            (7, 8),
            (7, 9),
            (7, 10),
            (8, 9),
            (8, 10),
            (9, 10),
            (10, 11),
            (10, 12),
            (11, 12),
        ]
    )
