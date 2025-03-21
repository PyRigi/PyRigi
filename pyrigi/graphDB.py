"""
This is a module for providing common types of graphs.
"""

import networkx as nx

import pyrigi._input_check as _input_check
from pyrigi.graph import Graph
from pyrigi.data_type import Vertex, Sequence
from itertools import combinations


def Cycle(n: int) -> Graph:
    """Return the cycle graph on ``n`` vertices."""
    return Graph(nx.cycle_graph(n))


def Complete(n: int = None, vertices: Sequence[Vertex] = None) -> Graph:
    """
    Return the complete graph on ``n`` vertices.

    The vertex labels can also be specified explicitly via
    the keyword ``vertices``.

    Parameters
    ----------
    n:
        The number of vertices.
    vertices:
        An optional parameter for the vertices.

    Examples
    --------
    >>> print(Complete(5))
    Graph with vertices [0, 1, 2, 3, 4] and edges [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    >>> print(Complete(5, [0, 1, 2, 3, 4]))
    Graph with vertices [0, 1, 2, 3, 4] and edges [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    >>> print(Complete(vertices=['a', 'b', 'c', 'd']))
    Graph with vertices ['a', 'b', 'c', 'd'] and edges [['a', 'b'], ['a', 'c'], ['a', 'd'], ['b', 'c'], ['b', 'd'], ['c', 'd']]
    """  # noqa: E501
    if vertices is None:
        _input_check.integrality_and_range(n, "number of vertices n", min_val=0)
        return Graph(nx.complete_graph(n))
    if n is None:
        n = len(vertices)
    _input_check.equal(len(vertices), n, "number of `vertices`", "the parameter `n`")
    edges = list(combinations(vertices, 2))
    return Graph.from_vertices_and_edges(vertices, edges)


def Path(n: int) -> Graph:
    """Return the path graph with ``n`` vertices."""
    _input_check.integrality_and_range(n, "number of vertices n", min_val=0)
    return Graph(nx.path_graph(n))


def CompleteBipartite(n1: int, n2: int) -> Graph:
    """Return the complete bipartite graph on ``n1+n2`` vertices."""
    _input_check.integrality_and_range(n1, "number of vertices n1", min_val=1)
    _input_check.integrality_and_range(n2, "number of vertices n2", min_val=1)
    return Graph(nx.complete_multipartite_graph(n1, n2))


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
    Return the ``dim``-dimensional double banana graph.

    Definitions
    -----
    :prf:ref:`Generalized Double Banana <def-generalized-double-banana>`

    Parameters
    ----------
    dim:
        An integer greater or equal 3.
    t:
        An integer such that ``2 <= t <= dim-1``.

    Examples
    --------
    >>> print(DoubleBanana())
    Graph with vertices [0, 1, 2, 3, 4, 5, 6, 7] and edges [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [2, 3], [2, 4], [3, 4], [5, 6], [5, 7], [6, 7]]
    >>> print(DoubleBanana(dim = 4))
    Graph with vertices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] and edges [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5], [6, 7], [6, 8], [6, 9], [7, 8], [7, 9], [8, 9]]
    """  # noqa: E501
    _input_check.integrality_and_range(dim, "dimension dim", min_val=3)
    _input_check.integrality_and_range(t, "parameter t", min_val=2)
    _input_check.smaller_equal(t, dim - 1, "parameter t", "dim - 1")

    r = (dim + 2) - t
    Kt = Complete(t)
    Kt1 = Kt.copy()
    for i in range(t, dim + 2):
        Kt1.add_edges([[i, v] for v in Kt1.nodes])
    Kt2 = Kt.copy()
    for i in range(dim + 2, dim + 2 + r):
        Kt2.add_edges([[i, v] for v in Kt2.nodes])
    return Kt1.sum_t(Kt2, [0, 1], t)


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
    """
    Return the ``n``-Frustum graph.

    Definitions
    -----------
    :prf:ref:`n-Frustum graph <def-n-frustum>`
    """
    return Graph(
        [(j, (j + 1) % n) for j in range(0, n)]
        + [(j, (j + 1 - n) % n + n) for j in range(n, 2 * n)]
        + [(j, j + n) for j in range(0, n)]
    )


def K66MinusPerfectMatching() -> Graph:
    """
    Return the complete bipartite graph minus a perfect matching.

    A matching is formed by six non-incident edges.
    """
    G = CompleteBipartite(6, 6)
    G.delete_edges([(i, i + 6) for i in range(0, 6)])
    return G


def CnSymmetricFourRegular(n: int = 8) -> Graph:
    """
    Return a $C_n$-symmetric 4-regular graph.

    The value ``n`` must be even and at least 8.

    Definitions
    -----------
    * :prf:ref:`Example with a free group action <def-Cn-symmetric>`
    """
    _input_check.integrality_and_range(n, "number of vertices n", min_val=8)
    if not n % 2 == 0:
        raise ValueError(
            "To generate this graph, the cyclic group " + "must have an even order!"
        )
    G = Graph()
    G.add_edges([(0, n - 1), (n - 3, 0), (n - 2, 1), (n - 1, 2)])
    for i in range(1, n):
        G.add_edges([(i, i - 1)])
    for i in range(n - 3):
        G.add_edge(i, i + 3)
    return G


def CnSymmetricWithFixedVertex(n: int = 8) -> Graph:
    """
    Return a $C_n$-symmetric graph with a fixed vertex.

    The value ``n`` must be even and at least 8.

    The returned graph satisfies the expected symmetry-adapted Laman
    count for rotation but is (generically) infinitesimally flexible.

    Definitions
    -----------
    :prf:ref:`Example with joint at origin <def-Cn-symmetric-joint-at-origin>`
    """
    _input_check.integrality_and_range(n, "order of cyclic group n", min_val=8)
    if not n % 2 == 0:
        raise ValueError(
            "To generate this graph, the cyclic group " + "must have an even order!"
        )
    G = CnSymmetricFourRegular(n)
    G.add_edges([(0, n), (n, 2 * n), (n + 1, 2 * n - 1), (n, 2 * n - 2)])
    for i in range(1, n):
        G.add_edges([(i, i + n), (2 * n, i + n), ((i + 1) + n, (i + 1) + n - 2)])
    return G


def ThreeConnectedR3Circuit() -> Graph:
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


def Wheel(n: int) -> Graph:
    """
    Create the wheel graph on ``n+1`` vertices.
    """
    _input_check.integrality_and_range(n + 1, "number of vertices n+1", min_val=4)
    G = Cycle(n)
    G.add_edges([(i, n) for i in range(n)])
    return G
