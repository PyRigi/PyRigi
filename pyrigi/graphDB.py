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


def DoubleBanana(d: int = 3, t: int = 2) -> Graph:
    r"""
    Return the d-dimensional double banana graph.

    Parameters
    ----------
    d: integer, must be at least 3
    t: integer, must be 2 <= t <= d-1

    Definitions
    -----
    :prf:ref:`Generalized Double Banana <def-generalized-double-banana>`

    Examples
    --------
    >>> DoubleBanana()
    Graph with vertices [0, 1, 2, 3, 4, 5, 6, 7] and edges [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [2, 3], [2, 4], [3, 4], [5, 6], [5, 7], [6, 7]]
    >>> DoubleBanana(d = 4)
    Graph with vertices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] and edges [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5], [6, 7], [6, 8], [6, 9], [7, 8], [7, 9], [8, 9]]
    """  # noqa: E501
    if d < 3:
        raise ValueError(f"The parameter d must be at least 3, instead it is {d}.")
    if t < 2 or t > d - 1:
        raise ValueError(f"The parameter t must be 2 <= t <= {d-1}, instead it is {t}.")
    r = (d + 2) - t
    K = Complete(t)
    K1 = K.copy()
    for i in range(t, d + 2):
        K1.add_edges([[i, v] for v in K1.nodes])
    K2 = K.copy()
    for i in range(d + 2, d + 2 + r):
        K2.add_edges([[i, v] for v in K2.nodes])
    return K1.sum_t(K2, [0, 1], t)


def CompleteMinusOne(n: int) -> Graph:
    """Return the complete graph on n vertices minus one edge."""
    G = Complete(n)
    G.delete_edge((0, 1))
    return G


def Octahedral() -> Graph:
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


def Frustum(n: int) -> Graph:
    """Return the :prf:ref:`n-Frustum graph <def-n-frustum>`"""
    return Graph(
        [(j, (j + 1) % n) for j in range(0, n)]
        + [(j, (j + 1 - n) % n + n) for j in range(n, 2 * n)]
        + [(j, j + n) for j in range(0, n)]
    )


def K66MinusPerfectMatching():
    """
    Return a complete bipartite graph minus a perfect matching.

    A matching is formed by six non-incident edges.

    TODO
    ----
    use in tests
    """
    G = CompleteBipartite(6, 6)
    G.delete_edges([(i, i + 6) for i in range(0, 6)])
    return G


def CnSymmetricFourRegular(n: int = 8) -> Graph:
    """
    Return a C_n-symmetric graph.

    TODO
    ----
    use in tests

    Definitions
    -----------
    * :prf:ref:`Example with a free group action <def-Cn-symmetric>`

    """
    if not n % 2 == 0 or n < 8:
        raise ValueError(
            "To generate this graph, the cyclical group "
            + "needs to have an even order of at least 8!"
        )
    G = Graph()
    G.add_edges([(0, n - 1), (n - 3, 0), (n - 2, 1), (n - 1, 2)])
    for i in range(1, n):
        G.add_edges([(i, i - 1)])
    for i in range(n - 3):
        G.add_edge(i, i + 3)
    return G


def CnSymmetricFourRegularWithFixedVertex(n: int = 8) -> Graph:
    """
    Return a C_n-symmetric graph with a fixed vertex.
    The cyclical group C_n needs to have even order of at least 8.

    The returned graph satisfies the expected symmetry-adapted Laman
    count for rotation but is infinitesimally flexible.

    TODO
    ----
    use in tests

    Definitions
    -----------
    * :prf:ref:`Example with joint at origin <def-Cn-symmetric-joint-at-origin>`
    """
    if not n % 2 == 0 or n < 8:
        raise ValueError(
            "To generate this graph, the cyclical group "
            + "needs to have an even order of at least 8!"
        )
    G = CnSymmetricFourRegular(n)
    G.add_edges([(0, n), (n, 2 * n), (n + 1, 2 * n - 1), (n, 2 * n - 2)])
    for i in range(1, n):
        G.add_edges([(i, i + n), (2 * n, i + n), ((i + 1) + n, (i + 1) + n - 2)])
    return G


def ThreeConnectedR3Circuit():
    """
    Return a 3-connected R_3-circuit.

    The returned graph is hypothesized to be the smallest 3-connected R_3-circuit.

    TODO
    ----
    use in tests
    """
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
