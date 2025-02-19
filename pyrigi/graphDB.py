"""
This is a module for providing common types of graphs.
"""

import networkx as nx
from pyrigi.graph import Graph
import pyrigi._input_check as _input_check


def Cycle(n: int) -> Graph:
    """Return the cycle graph on ``n`` vertices."""
    return Graph(nx.cycle_graph(n))


def Complete(n: int) -> Graph:
    """Return the complete graph on ``n`` vertices."""
    return Graph(nx.complete_graph(n))


def Path(n: int) -> Graph:
    """Return the path graph with ``n`` vertices."""
    return Graph(nx.path_graph(n))


def CompleteBipartite(m: int, n: int) -> Graph:
    """Return the complete bipartite graph on ``m+n`` vertices."""
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


def DoubleBanana(dim: int = 3, t: int = 2) -> Graph:
    r"""
    Return the `dim`-dimensional double banana graph.

    Parameters
    ----------
    dim:
        An integer greater or equal 3.
    t:
        An integer such that ``2 <= t <= dim-1``.

    Definitions
    -----
    :prf:ref:`Generalized Double Banana <def-generalized-double-banana>`

    Examples
    --------
    >>> DoubleBanana()
    Graph with vertices [0, 1, 2, 3, 4, 5, 6, 7] and edges [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [2, 3], [2, 4], [3, 4], [5, 6], [5, 7], [6, 7]]
    >>> DoubleBanana(dim = 4)
    Graph with vertices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] and edges [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5], [6, 7], [6, 8], [6, 9], [7, 8], [7, 9], [8, 9]]
    """  # noqa: E501
    _input_check.greater_equal(dim, 3, "dimension")
    _input_check.greater_equal(t, 2, "parameter t")
    _input_check.smaller_equal(t, dim - 1, "parameter t", "dim - 1")

    r = (dim + 2) - t
    K = Complete(t)
    K1 = K.copy()
    for i in range(t, dim + 2):
        K1.add_edges([[i, v] for v in K1.nodes])
    K2 = K.copy()
    for i in range(dim + 2, dim + 2 + r):
        K2.add_edges([[i, v] for v in K2.nodes])
    return K1.sum_t(K2, [0, 1], t)


def CompleteMinusOne(n: int) -> Graph:
    """Return the complete graph on ``n`` vertices minus one edge."""
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


def Icosahedral() -> Graph:
    """Return the graph given by the skeleton of an icosahedron."""
    return Graph(nx.icosahedral_graph().edges)


def Dodecahedral() -> Graph:
    """Return the graph given by the skeleton of a dodecahedron."""
    return Graph(
        [
            (0, 8),
            (0, 12),
            (0, 16),
            (1, 8),
            (1, 13),
            (1, 18),
            (2, 10),
            (2, 12),
            (2, 17),
            (3, 9),
            (3, 14),
            (3, 16),
            (4, 10),
            (4, 13),
            (4, 19),
            (5, 9),
            (5, 15),
            (5, 18),
            (6, 11),
            (6, 14),
            (6, 17),
            (7, 11),
            (7, 15),
            (7, 19),
            (8, 9),
            (10, 11),
            (12, 13),
            (14, 15),
            (16, 17),
            (18, 19),
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
    """
    G = CompleteBipartite(6, 6)
    G.delete_edges([(i, i + 6) for i in range(0, 6)])
    return G


def CnSymmetricFourRegular(n: int = 8) -> Graph:
    """
    Return a $C_n$-symmetric graph.

    Definitions
    -----------
    * :prf:ref:`Example with a free group action <def-Cn-symmetric>`
    """
    if not n % 2 == 0 or n < 8:
        raise ValueError(
            "To generate this graph, the cyclic group "
            + "must have an even order of at least 8!"
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
    Return a $C_n$-symmetric graph with a fixed vertex.

    The value ``n`` must be even and at least 8.

    The returned graph satisfies the expected symmetry-adapted Laman
    count for rotation but is infinitesimally flexible.

    Definitions
    -----------
    * :prf:ref:`Example with joint at origin <def-Cn-symmetric-joint-at-origin>`
    """
    if not n % 2 == 0 or n < 8:
        raise ValueError(
            "To generate this graph, the cyclic group "
            + "must have an even order of at least 8!"
        )
    G = CnSymmetricFourRegular(n)
    G.add_edges([(0, n), (n, 2 * n), (n + 1, 2 * n - 1), (n, 2 * n - 2)])
    for i in range(1, n):
        G.add_edges([(i, i + n), (2 * n, i + n), ((i + 1) + n, (i + 1) + n - 2)])
    return G


def ThreeConnectedR3Circuit():
    """
    Return a 3-connected $R_3$-circuit.

    The returned graph is hypothesized to be the smallest 3-connected $R_3$-circuit.
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
