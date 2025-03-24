from random import randint

import math
import matplotlib.pyplot as plt

from itertools import combinations, product
import networkx as nx
import pytest
from sympy import Matrix

import pyrigi.graphDB as graphs
import pyrigi.misc as misc
from pyrigi.graph import Graph
from pyrigi.exception import LoopError, NotSupportedValueError
from pyrigi.warning import RandomizedAlgorithmWarning


def relabeled_inc(graph: Graph, increment: int = None) -> Graph:
    """
    Return the graph with each vertex label incremented by a given number.

    Note that ``graph`` must have integer vertex labels.
    """
    if increment is None:
        increment = graph.number_of_nodes()
    return nx.relabel_nodes(graph, {i: i + increment for i in graph.nodes()}, copy=True)


def test__add__():
    G = Graph([[0, 1], [1, 2], [2, 0]])
    H = Graph([[0, 1], [1, 3], [3, 0]])
    assert G + H == Graph([[0, 1], [1, 2], [2, 0], [1, 3], [3, 0]])
    G = Graph([[0, 1], [1, 2], [2, 0]])
    H = Graph([[3, 4], [4, 5], [5, 3]])
    assert G + H == Graph([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]])
    G = Graph.from_vertices_and_edges([0, 1, 2, 3], [[0, 1], [1, 2]])
    H = Graph.from_vertices_and_edges([0, 1, 2, 4], [[0, 1]])
    assert G + H == Graph.from_vertices_and_edges([0, 1, 2, 3, 4], [[0, 1], [1, 2]])


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Complete(3),
        graphs.Complete(4),
        graphs.CompleteBipartite(3, 3),
        graphs.CompleteBipartite(3, 4),
        graphs.CompleteBipartite(4, 4),
        graphs.Diamond(),
        graphs.K33plusEdge(),
        graphs.ThreePrism(),
        graphs.ThreePrismPlusEdge(),
        graphs.K66MinusPerfectMatching(),
    ],
)
def test_is_rigid_d2(graph):
    assert graph.is_rigid(dim=2, algorithm="default")
    assert graph.is_rigid(dim=2, algorithm="sparsity")
    assert graph.is_rigid(dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Path(3),
        graphs.Path(4),
        graphs.Cycle(4),
        graphs.Cycle(5),
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
    ],
)
def test_not_is_rigid_d2(graph):
    assert not graph.is_rigid(dim=2, algorithm="default")
    assert not graph.is_rigid(dim=2, algorithm="sparsity")
    assert not graph.is_rigid(dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.Cycle(4),
        graphs.Cycle(5),
        graphs.Path(3),
        graphs.Path(4),
    ],
)
def test_is_not_rigid_d2(graph):
    assert not graph.is_rigid(dim=2, algorithm="sparsity")
    assert not graph.is_rigid(dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Diamond(),
        graphs.K33plusEdge(),
        graphs.ThreePrism(),
        graphs.ThreePrismPlusEdge(),
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.Cycle(4),
        graphs.Path(3),
        graphs.Dodecahedral(),
    ],
)
def test_is_rigid_d1(graph):
    assert graph.is_rigid(dim=1, algorithm="default")
    assert graph.is_rigid(dim=1, algorithm="graphic")
    assert graph.is_rigid(dim=1, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [Graph.from_vertices(range(3)), Graph([[0, 1], [2, 3]])],
)
def test_is_not_rigid_d1(graph):
    assert not graph.is_rigid(dim=1, algorithm="sparsity")
    assert not graph.is_rigid(dim=1, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.K66MinusPerfectMatching(), 3],
        pytest.param(graphs.Icosahedral(), 3, marks=pytest.mark.long_local),
    ]
    + [[graphs.Complete(n), d] for d in range(1, 5) for n in range(1, d + 2)],
)
def test_is_rigid(graph, dim):
    assert graph.is_rigid(dim, algorithm="sparsity" if (dim < 3) else "randomized")
    assert graph.is_rigid(dim, algorithm="default")


@pytest.mark.parametrize(
    "graph",
    [
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.CompleteBipartite(3, 3),
        graphs.Complete(2),
        graphs.Complete(3),
        graphs.Cycle(4),
        graphs.Cycle(5),
        graphs.Diamond(),
        graphs.Path(3),
        graphs.Path(4),
        graphs.ThreePrism(),
    ],
)
def test_is_2_3_sparse(graph):
    assert graph.is_kl_sparse(2, 3, algorithm="subgraph")
    assert graph.is_kl_sparse(2, 3, algorithm="pebble")
    assert graph.is_sparse()


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(4),
        graphs.CompleteBipartite(3, 4),
        graphs.CompleteBipartite(4, 4),
        graphs.K33plusEdge(),
        graphs.ThreePrismPlusEdge(),
    ],
)
def test_is_not_2_3_sparse(graph):
    assert not graph.is_kl_sparse(2, 3, algorithm="subgraph")
    assert not graph.is_kl_sparse(2, 3, algorithm="pebble")
    assert not graph.is_sparse()


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Complete(3),
        graphs.CompleteBipartite(3, 3),
        graphs.Diamond(),
        graphs.ThreePrism(),
    ],
)
def test_is_2_3_tight(graph):
    assert graph.is_kl_tight(2, 3, algorithm="pebble")
    assert graph.is_kl_tight(2, 3, algorithm="subgraph")
    assert graph.is_tight()


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(4),
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.CompleteBipartite(3, 4),
        graphs.CompleteBipartite(4, 4),
        graphs.Cycle(4),
        graphs.Cycle(5),
        graphs.K33plusEdge(),
        graphs.Path(3),
        graphs.Path(4),
        graphs.ThreePrismPlusEdge(),
    ],
)
def test_is_not_2_3_tight(graph):
    assert not graph.is_kl_tight(2, 3, algorithm="subgraph")
    assert not graph.is_kl_tight(2, 3, algorithm="pebble")
    assert not graph.is_tight()


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.CompleteBipartite(1, 3),
        graphs.Path(3),
        Graph.from_int(102),  # a tree on 5 vertices
    ],
)
def test_is_min_rigid_d1(graph):
    assert graph.is_min_rigid(dim=1, algorithm="graphic")
    assert graph.is_min_rigid(dim=1, algorithm="extension_sequence")
    assert graph.is_min_rigid(dim=1, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [
        Graph.from_vertices(range(3)),
        Graph([[0, 1], [2, 3]]),
        graphs.Complete(3),
        graphs.Diamond(),
        graphs.K33plusEdge(),
        graphs.ThreePrism(),
        graphs.ThreePrismPlusEdge(),
        graphs.CompleteBipartite(2, 3),
        graphs.Cycle(4),
    ],
)
def test_is_not_min_rigid_d1(graph):
    assert not graph.is_min_rigid(dim=1, algorithm="sparsity")
    assert not graph.is_min_rigid(dim=1, algorithm="extension_sequence")
    assert not graph.is_min_rigid(dim=1, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Complete(3),
        graphs.CompleteBipartite(3, 3),
        graphs.Diamond(),
        graphs.ThreePrism(),
    ],
)
def test_is_min_rigid_d2(graph):
    assert graph.is_min_rigid(dim=2, algorithm="sparsity")
    assert graph.is_min_rigid(dim=2, algorithm="extension_sequence")
    assert graph.is_min_rigid(dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(4),
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.CompleteBipartite(3, 4),
        graphs.CompleteBipartite(4, 4),
        graphs.Cycle(4),
        graphs.Cycle(5),
        graphs.K33plusEdge(),
        graphs.Path(3),
        graphs.Path(4),
        graphs.ThreePrismPlusEdge(),
        pytest.param(graphs.Dodecahedral(), marks=pytest.mark.long_local),
    ],
)
def test_is_not_min_rigid_d2(graph):
    assert not graph.is_min_rigid(dim=2, algorithm="sparsity")
    assert not graph.is_min_rigid(dim=2, algorithm="extension_sequence")
    assert not graph.is_min_rigid(dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [
        Graph.from_vertices_and_edges([0, 1], [[0, 1]]),
        graphs.Complete(3),
        graphs.Complete(4),
        graphs.Octahedral(),
        pytest.param(graphs.K66MinusPerfectMatching(), marks=pytest.mark.slow_main),
        pytest.param(graphs.Icosahedral(), marks=pytest.mark.long_local),
    ],
)
def test_is_min_rigid_d3(graph):
    assert graph.is_min_rigid(dim=3, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(5),
        graphs.CubeWithDiagonal(),
        graphs.CompleteBipartite(5, 5),
        graphs.DoubleBanana(dim=3),
        pytest.param(graphs.ThreeConnectedR3Circuit(), marks=pytest.mark.long_local),
        graphs.Dodecahedral(),
    ],
)
def test_is_not_min_rigid_d3(graph):
    assert not graph.is_min_rigid(dim=3, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Complete(3),
        graphs.Complete(4),
        graphs.CompleteBipartite(3, 4),
        graphs.CompleteBipartite(4, 4),
        graphs.K33plusEdge(),
        graphs.ThreePrismPlusEdge(),
    ],
)
def test_is_globally_rigid_d2(graph):
    assert graph.is_globally_rigid(dim=2)


def read_random_from_graph6(filename):
    file_ = nx.read_graph6(filename)
    if isinstance(file_, list):
        return Graph(file_[randint(0, len(file_) - 1)])
    else:
        return Graph(file_)


def read_globally(d_v_):
    return read_random_from_graph6("test/input_graphs/globally_rigid/" + d_v_ + ".g6")


def read_redundantly(d_v_):
    return read_random_from_graph6(
        "test/input_graphs/redundantly_rigid/" + d_v_ + ".g6"
    )


def read_sparsity(filename):
    return Graph(nx.read_sparse6("test/input_graphs/sparsity/" + filename + ".s6"))


# Examples of globally rigid graphs taken from:
# Grasegger, G. (2022). Dataset of globally rigid graphs [Data set].
# Zenodo. https://doi.org/10.5281/zenodo.7473052
@pytest.mark.parametrize(
    "graph, gdim",
    [
        [graphs.Complete(2), 3],
        [graphs.Complete(2), 6],
        [read_globally("D3V4"), 3],
        [read_globally("D3V5"), 3],
        [read_globally("D3V6"), 3],
        [read_globally("D3V7"), 3],
        [read_globally("D3V8"), 3],
        [read_globally("D4V5"), 4],
        [read_globally("D4V6"), 4],
        [read_globally("D4V7"), 4],
        [read_globally("D4V8"), 4],
        [read_globally("D4V9"), 4],
        [read_globally("D6V7"), 6],
        [read_globally("D6V8"), 6],
        [read_globally("D6V9"), 6],
        [read_globally("D6V10"), 6],
        [read_globally("D10V11"), 10],
        [read_globally("D10V12"), 10],
        [read_globally("D10V13"), 10],
        [read_globally("D10V14"), 10],
        [read_globally("D19V20"), 19],
        pytest.param(read_globally("D19V21"), 19, marks=pytest.mark.slow_main),
        pytest.param(read_globally("D19V22"), 19, marks=pytest.mark.slow_main),
        pytest.param(read_globally("D19V23"), 19, marks=pytest.mark.long_local),
    ],
)
def test_is_globally_rigid(graph, gdim):
    assert graph.is_globally_rigid(dim=gdim)


@pytest.mark.parametrize(
    "graph, gdim",
    [
        [graphs.Diamond(), 3],
        [graphs.Path(3), 3],
        [graphs.ThreePrism(), 3],
        [graphs.Cycle(5), 3],
        [graphs.CompleteMinusOne(4), 3],
        [graphs.CompleteMinusOne(5), 3],
        [graphs.CompleteBipartite(1, 3), 3],
        [graphs.CompleteBipartite(2, 3), 3],
        [graphs.Diamond(), 4],
        [graphs.Path(4), 4],
        [graphs.ThreePrism(), 4],
        [graphs.Cycle(4), 4],
        [graphs.CompleteMinusOne(4), 4],
        [graphs.CompleteMinusOne(5), 4],
        [graphs.CompleteBipartite(2, 3), 4],
        [graphs.CompleteBipartite(3, 3), 4],
        [graphs.Diamond(), 6],
        [graphs.Path(4), 6],
        [graphs.ThreePrism(), 6],
        [graphs.Cycle(5), 6],
        [graphs.CompleteMinusOne(4), 6],
        [graphs.CompleteMinusOne(5), 6],
        [graphs.CompleteBipartite(1, 3), 6],
        [graphs.CompleteBipartite(3, 3), 6],
        [graphs.Diamond(), 10],
        [graphs.Path(4), 10],
        [graphs.ThreePrism(), 10],
        [graphs.Cycle(5), 10],
        [graphs.CompleteMinusOne(4), 10],
        [graphs.CompleteMinusOne(5), 10],
        [graphs.CompleteBipartite(2, 3), 10],
        [graphs.CompleteBipartite(3, 3), 10],
        [graphs.Diamond(), 19],
        [graphs.Path(4), 19],
        [graphs.ThreePrism(), 19],
        [graphs.Cycle(5), 19],
        [graphs.CompleteMinusOne(4), 19],
        [graphs.CompleteMinusOne(5), 19],
        [graphs.CompleteBipartite(1, 3), 19],
        [graphs.CompleteBipartite(2, 3), 19],
    ],
)
def test_is_not_globally_rigid(graph, gdim):
    assert not graph.is_globally_rigid(dim=gdim)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.CompleteBipartite(3, 3),
        graphs.Cycle(4),
        graphs.Cycle(5),
        graphs.Diamond(),
        graphs.Path(3),
        graphs.Path(4),
        graphs.ThreePrism(),
    ],
)
def test_is_not_globally_d2(graph):
    assert not graph.is_globally_rigid(dim=2)


def test_rigid_in_d2():
    graph = read_sparsity("K4")
    assert graph.is_rigid(dim=2, algorithm="sparsity")

    # (2,3)-tight graph on 1000 vertices and 1997 edges
    graph = read_sparsity("huge_tight_2_3")
    assert graph.is_kl_tight(K=2, L=3, algorithm="pebble")


@pytest.mark.parametrize(
    "graph, K, L",
    [
        # (6,8)-tight graph on 50 vertices and 292 edges
        pytest.param(read_sparsity("tight_6_8"), 6, 8, marks=pytest.mark.slow_main),
        # (7,3)-tight graph on 70 vertices and 487 edges
        pytest.param(read_sparsity("tight_7_3"), 7, 3, marks=pytest.mark.slow_main),
        # (5,9)-tight graph on 40 vertices and 191 edges
        pytest.param(read_sparsity("tight_5_9"), 5, 9, marks=pytest.mark.slow_main),
        # (13,14)-tight graph on 20 vertices and 246 edges
        pytest.param(read_sparsity("tight_13_14"), 13, 14, marks=pytest.mark.slow_main),
        # (2,3)-tight graph on 1000 vertices and 1997 edges
        pytest.param(
            read_sparsity("huge_tight_2_3"), 2, 3, marks=pytest.mark.slow_main
        ),
    ],
)
def test_big_random_tight_graphs(graph, K, L):
    assert graph.is_kl_tight(K, L, algorithm="pebble")


@pytest.mark.parametrize(
    "graph, K, L",
    [
        # Dense graph on 20 vertices
        pytest.param(
            read_sparsity("not_sparse_5_2"), 5, 2, marks=pytest.mark.slow_main
        ),
        # (7,7)-tight graph plus one edge on 40 vertices (274 edges)
        pytest.param(
            read_sparsity("not_sparse_7_7"), 7, 7, marks=pytest.mark.slow_main
        ),
        # few edges in graph on 30 vertices, but has a (3,5)-connected circle
        pytest.param(
            read_sparsity("not_sparse_3_5"), 3, 5, marks=pytest.mark.slow_main
        ),
        # random large graph on 70 vertices, not sparse
        pytest.param(
            read_sparsity("not_sparse_6_6"), 6, 6, marks=pytest.mark.slow_main
        ),
    ],
)
def test_big_random_not_sparse_graphs(graph, K, L):
    assert not graph.is_kl_sparse(K, L, algorithm="pebble")


@pytest.mark.parametrize(
    "graph",
    [
        pytest.param(read_sparsity("K4"), marks=pytest.mark.slow_main),
        pytest.param(read_sparsity("circle_5_8"), marks=pytest.mark.slow_main),
        pytest.param(read_sparsity("circle_10_18"), marks=pytest.mark.slow_main),
        pytest.param(read_sparsity("circle_20_38"), marks=pytest.mark.slow_main),
        pytest.param(read_sparsity("circle_30_58"), marks=pytest.mark.slow_main),
    ],
)
def test_Rd_circuit_graphs_d2(graph):
    assert graph.is_Rd_circuit(dim=2, algorithm="sparsity")


@pytest.mark.parametrize(
    "graph",
    [
        pytest.param(read_sparsity("not_circle_5_7"), marks=pytest.mark.slow_main),
        pytest.param(read_sparsity("not_circle_10_18"), marks=pytest.mark.slow_main),
        pytest.param(read_sparsity("not_circle_20_39"), marks=pytest.mark.slow_main),
        pytest.param(read_sparsity("not_circle_30_58"), marks=pytest.mark.slow_main),
    ],
)
def test_Rd_not_circuit_graphs_d2(graph):
    assert not graph.is_Rd_circuit(dim=2, algorithm="sparsity")


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        read_globally("D2V4"),
        read_globally("D2V5"),
        read_globally("D2V6"),
        read_globally("D2V7"),
        read_globally("D2V8"),
    ],
)
def test_is_weakly_globally_linked_for_globally_rigid_graphs(graph):
    # in a globally rigid graph, each pair of vertices should be weakly globally linked
    for u, v in list(combinations(graph.nodes, 2)):
        assert graph.is_weakly_globally_linked(u, v)


@pytest.mark.parametrize(
    "graph",
    [
        read_redundantly("D2V4"),
        read_redundantly("D2V5"),
        read_redundantly("D2V6"),
        read_redundantly("D2V7"),
        read_redundantly("D2V8"),
        read_redundantly("D2V9"),
    ],
)
def test_is_weakly_globally_linked_for_redundantly_rigid_graphs(graph):
    # graph is redundantly rigid, i.e., if we remove any edge, it is rigid
    for u, v in graph.edges:
        H = graph.copy()
        H.remove_edge(u, v)
        # now H is surely a rigid graph
        if H.is_globally_rigid():
            return test_is_weakly_globally_linked_for_globally_rigid_graphs(H)
        else:
            # if H is rigid but it is not globally rigid, then we know that there must
            # be at least one pair of vertices that is not weakly globally linked in
            # the graph, so we set the counter and we do a for loop that ends when a
            # not weakly globally linked pair of vertices is found
            counter = 0
            for a, b in list(combinations(H.nodes, 2)):
                if not H.is_weakly_globally_linked(a, b):
                    counter = 1
                    break
            assert counter


@pytest.mark.parametrize(
    "graph, u, v",
    [
        # The following two examples are Figure 2 and Figure 5
        # of the article :cite:p:`Jordan2024`
        [
            Graph(
                [
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [0, 6],
                    [1, 2],
                    [1, 3],
                    [1, 4],
                    [2, 4],
                    [2, 5],
                    [3, 5],
                    [4, 6],
                ]
            ),
            3,
            4,
        ],
        [
            Graph(
                [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4]]
            ),
            3,
            4,
        ],
        [
            Graph(
                [
                    [0, 1],
                    [0, 5],
                    [0, 7],
                    [1, 2],
                    [1, 3],
                    [1, 7],
                    [2, 3],
                    [2, 4],
                    [3, 4],
                    [4, 5],
                    [4, 8],
                    [4, 11],
                    [5, 6],
                    [5, 8],
                    [5, 13],
                    [6, 10],
                    [6, 11],
                    [6, 12],
                    [7, 8],
                    [7, 14],
                    [8, 12],
                    [9, 10],
                    [9, 14],
                    [10, 13],
                    [11, 12],
                    [13, 14],
                ]
            ),
            0,
            11,
        ],
    ],
)
def test_is_weakly_globally_linked_articles_graphs(graph, u, v):
    assert graph.is_weakly_globally_linked(u, v)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.CompleteMinusOne(5),
        graphs.Complete(5),
        Graph.from_int(7679),
        Graph([["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"], ["b", "d"]]),
    ],
)
def test_is_vertex_redundantly_rigid_d2(graph):
    assert graph.is_vertex_redundantly_rigid(dim=2)
    assert graph.is_vertex_redundantly_rigid(dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Cycle(4), 1],
        [Graph.from_int(3294), 1],
        [graphs.CompleteMinusOne(5), 2],
        [
            Graph(
                [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"], ["b", "d"]]
            ),
            2,
        ],
        [Graph.from_int(16351), 3],
    ],
)
def test_is_k_vertex_redundantly_rigid_d1(graph, k):
    assert graph.is_k_vertex_redundantly_rigid(k, dim=1)
    assert graph.is_k_vertex_redundantly_rigid(k, dim=1, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Complete(3), 1],
        [Graph.from_int(7679), 1],
        [
            Graph(
                [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"], ["b", "d"]]
            ),
            1,
        ],
        [graphs.CompleteMinusOne(6), 2],
        [graphs.Complete(6), 2],
        [graphs.CompleteMinusOne(7), 3],
    ],
)
def test_is_k_vertex_redundantly_rigid_d2(graph, k):
    assert graph.is_k_vertex_redundantly_rigid(k, dim=2)
    assert graph.is_k_vertex_redundantly_rigid(k, dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.CompleteMinusOne(6), 1],
        [
            Graph(
                [
                    ["a", "b"],
                    ["a", "c"],
                    ["a", "d"],
                    ["a", "e"],
                    ["b", "c"],
                    ["b", "d"],
                    ["b", "e"],
                    ["c", "d"],
                    ["c", "e"],
                    ["d", "e"],
                ]
            ),
            1,
        ],
    ],
)
def test_is_k_vertex_redundantly_rigid_d3(graph, k):
    assert graph.is_k_vertex_redundantly_rigid(k, dim=3, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [
        graphs.CompleteBipartite(3, 3),
        graphs.ThreePrismPlusEdge(),
        graphs.K33plusEdge(),
        Graph([["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"]]),
    ],
)
def test_is_not_vertex_redundantly_rigid_d2(graph):
    assert not graph.is_vertex_redundantly_rigid(dim=2)
    assert not graph.is_vertex_redundantly_rigid(dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(101739), 1],
        [Graph.from_int(255567), 2],
        [Graph.from_int(515576), 3],
        [Graph([["a", "b"], ["b", "c"], ["c", "a"], ["d", "a"], ["e", "d"]]), 1],
    ],
)
def test_is_not_k_vertex_redundantly_rigid_d1(graph, k):
    assert not graph.is_k_vertex_redundantly_rigid(k, dim=1)
    assert not graph.is_k_vertex_redundantly_rigid(k, dim=1, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.ThreePrism(), 1],
        [Graph.from_int(8191), 2],
        [Graph.from_int(1048059), 3],
        [Graph([["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"]]), 1],
        [graphs.Diamond(), 3],
        [graphs.Diamond(), 2],
        [graphs.Diamond(), 1],
    ],
)
def test_is_not_k_vertex_redundantly_rigid_d2(graph, k):
    assert not graph.is_k_vertex_redundantly_rigid(k, dim=2)
    assert not graph.is_k_vertex_redundantly_rigid(k, dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(16351), 1],
        [
            Graph(
                [
                    ["a", "b"],
                    ["a", "c"],
                    ["a", "d"],
                    ["a", "e"],
                    ["b", "c"],
                    ["b", "d"],
                    ["b", "e"],
                    ["c", "d"],
                    ["c", "e"],
                ]
            ),
            2,
        ],
    ],
)
def test_is_not_k_vertex_redundantly_rigid_d3(graph, k):
    assert not graph.is_k_vertex_redundantly_rigid(k, dim=3, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Cycle(4), 1],
        [Graph.from_int(222), 1],
        [graphs.CompleteBipartite(3, 3), 2],
        [
            Graph(
                [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"], ["b", "d"]]
            ),
            2,
        ],
        [Graph.from_int(16350), 3],
    ],
)
def test_is_min_k_vertex_redundantly_rigid_d1(graph, k):
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=1)
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=1, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Complete(3), 1],
        [Graph.from_int(7679), 1],
        [
            Graph(
                [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"], ["b", "d"]]
            ),
            1,
        ],
        [Graph.from_int(16383), 2],
        [Graph.from_int(1048575), 3],
    ],
)
def test_is_min_k_vertex_redundantly_rigid_d2(graph, k):
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=2)
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(507903), 1],
        [Graph.from_int(1048575), 2],
    ],
)
def test_is_min_k_vertex_redundantly_rigid_d3(graph, k):
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=3, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(101739), 1],
        [Graph.from_int(223), 1],
        [graphs.CompleteMinusOne(5), 2],
        [Graph.from_int(16351), 3],
    ],
)
def test_is_not_min_k_vertex_redundantly_rigid_d1(graph, k):
    assert not graph.is_min_k_vertex_redundantly_rigid(k, dim=1)
    assert not graph.is_min_k_vertex_redundantly_rigid(k, dim=1, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.ThreePrism(), 1],
        [Graph.from_int(8191), 1],
        [Graph.from_int(32767), 2],
        [Graph.from_int(2097151), 3],
    ],
)
def test_is_not_min_k_vertex_redundantly_rigid_d2(graph, k):
    assert not graph.is_min_k_vertex_redundantly_rigid(k, dim=2)
    assert not graph.is_min_k_vertex_redundantly_rigid(k, dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(16351), 1],
        [Graph.from_int(32767), 1],
        [Graph.from_int(2097151), 2],
        [
            Graph(
                [
                    ["a", "b"],
                    ["a", "c"],
                    ["a", "d"],
                    ["a", "e"],
                    ["b", "c"],
                    ["b", "d"],
                    ["b", "e"],
                    ["c", "d"],
                    ["c", "e"],
                ]
            ),
            2,
        ],
    ],
)
def test_is_not_min_k_vertex_redundantly_rigid_d3(graph, k):
    assert not graph.is_min_k_vertex_redundantly_rigid(k, dim=3, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(4),
        graphs.CompleteBipartite(3, 4),
        graphs.CompleteBipartite(4, 4),
        graphs.K33plusEdge(),
        graphs.ThreePrismPlusEdge(),
        Graph([["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"], ["b", "d"]]),
        pytest.param(graphs.Complete(7), marks=pytest.mark.slow_main),
    ],
)
def test_is_redundantly_rigid_d2(graph):
    assert graph.is_redundantly_rigid(dim=2)
    assert graph.is_redundantly_rigid(dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Diamond(), 1],
        [graphs.Complete(4), 1],
        [
            Graph(
                [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"], ["b", "d"]]
            ),
            2,
        ],
        [Graph.from_int(222), 1],
        [Graph.from_int(507), 2],
        [graphs.CompleteMinusOne(5), 2],
        [graphs.Complete(5), 3],
    ],
)
def test_is_k_redundantly_rigid_d1(graph, k):
    assert graph.is_k_redundantly_rigid(k, dim=1)
    assert graph.is_k_redundantly_rigid(k, dim=1, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Complete(4), 1],
        [
            Graph(
                [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"], ["b", "d"]]
            ),
            1,
        ],
        [graphs.Complete(5), 2],
        [graphs.Octahedral(), 2],
        [graphs.Complete(6), 2],
        pytest.param(graphs.Complete(6), 3, marks=pytest.mark.slow_main),
        # [Graph.from_int(1048059), 3],
        # [Graph.from_int(2097151), 3],
    ],
)
def test_is_k_redundantly_rigid_d2(graph, k):
    assert graph.is_k_redundantly_rigid(k, dim=2)
    assert graph.is_k_redundantly_rigid(k, dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Complete(5), 1],
        [Graph.from_int(16351), 1],
        [
            Graph(
                [
                    ["a", "b"],
                    ["a", "c"],
                    ["a", "d"],
                    ["a", "e"],
                    ["b", "c"],
                    ["b", "d"],
                    ["b", "e"],
                    ["c", "d"],
                    ["c", "e"],
                    ["d", "e"],
                ]
            ),
            1,
        ],
    ],
)
def test_is_k_redundantly_rigid_d3(graph, k):
    assert graph.is_k_redundantly_rigid(k, dim=3, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Complete(3),
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.CompleteBipartite(3, 3),
        graphs.Cycle(4),
        graphs.Cycle(5),
        graphs.Diamond(),
        graphs.Path(3),
        graphs.Path(4),
        graphs.ThreePrism(),
        Graph([["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"]]),
    ],
)
def test_is_not_redundantly_rigid_d2(graph):
    assert not graph.is_redundantly_rigid(dim=2)
    assert not graph.is_redundantly_rigid(dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(15), 1],
        [graphs.Diamond(), 2],
        [Graph([["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"]]), 3],
    ],
)
def test_is_not_k_redundantly_rigid_d1(graph, k):
    assert not graph.is_k_redundantly_rigid(k, dim=1)
    assert not graph.is_k_redundantly_rigid(k, dim=1, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [
            Graph(
                [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (0, 2), (1, 3)]
            ),
            1,
        ],
        [Graph.from_int(255), 1],
        [Graph.from_int(507), 2],
        [Graph.from_int(14917374), 2],
        [
            Graph(
                [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"], ["b", "d"]]
            ),
            2,
        ],
    ],
)
def test_is_not_k_redundantly_rigid_d2(graph, k):
    assert not graph.is_k_redundantly_rigid(k, dim=2)
    assert not graph.is_k_redundantly_rigid(k, dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Cycle(14), 2],
        [graphs.CnSymmetricWithFixedVertex(14), 2],
        [graphs.DoubleBanana(), 0],
        [graphs.DoubleBanana(), 1],
        [Graph.from_int(7679), 1],
        [Graph.from_int(16351), 2],
        [Graph.from_int(1048575), 3],
        [
            Graph(
                [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"], ["b", "d"]]
            ),
            1,
        ],
    ],
)
def test_is_not_k_redundantly_rigid_d3(graph, k):
    assert not graph.is_k_redundantly_rigid(k, dim=3, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Cycle(4), 1],
        [
            Graph(
                [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"], ["b", "d"]]
            ),
            2,
        ],
        [Graph.from_int(222), 1],
        [Graph.from_int(507), 2],
        [graphs.Complete(5), 3],
    ],
)
def test_is_min_k_redundantly_rigid_d1(graph, k):
    assert graph.is_min_k_redundantly_rigid(k, dim=1)
    assert graph.is_min_k_redundantly_rigid(k, dim=1, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(7915), 1],
        [
            Graph(
                [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"], ["b", "d"]]
            ),
            1,
        ],
        [Graph.from_int(16350), 2],
        [Graph.from_int(507851), 2],
        # [Graph.from_int(1048059), 3],
    ],
)
def test_is_min_k_redundantly_rigid_d2(graph, k):
    assert graph.is_min_k_redundantly_rigid(k, dim=2)
    assert graph.is_min_k_redundantly_rigid(k, dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Complete(5), 1],
        [Graph.from_int(16351), 1],
        [Graph.from_int(32767), 2],
    ],
)
def test_is_min_k_redundantly_rigid_d3(graph, k):
    assert graph.is_min_k_redundantly_rigid(k, dim=3, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(15), 1],
        [Graph.from_int(223), 1],
        [graphs.Diamond(), 2],
        [Graph.from_int(7679), 2],
        [Graph([["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"]]), 3],
        [Graph.from_int(16351), 3],
    ],
)
def test_is_not_min_k_redundantly_rigid_d1(graph, k):
    assert not graph.is_min_k_redundantly_rigid(k, dim=1)
    assert not graph.is_min_k_redundantly_rigid(k, dim=1, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.ThreePrism(), 1],
        [Graph.from_int(8191), 1],
        [graphs.Complete(7), 1],
        [Graph.from_int(16351), 2],
        # [Graph.from_int(1048063), 3],
    ],
)
def test_is_not_min_k_redundantly_rigid_d2(graph, k):
    assert not graph.is_min_k_redundantly_rigid(k, dim=2)
    assert not graph.is_min_k_redundantly_rigid(k, dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(7679), 1],
        [Graph.from_int(16383), 1],
        [Graph.from_int(16351), 2],
        pytest.param(Graph.from_int(1048063), 2, marks=pytest.mark.slow_main),
        # [Graph.from_int(1048575), 3],
        # [Graph.from_int(134201311), 3],
    ],
)
def test_is_not_min_k_redundantly_rigid_d3(graph, k):
    assert not graph.is_min_k_redundantly_rigid(k, dim=3, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, components, dim",
    [
        [graphs.Path(6), [[0, 1, 2, 3, 4, 5]], 1],
        [graphs.Path(3) + relabeled_inc(graphs.Path(3), 3), [[0, 1, 2], [3, 4, 5]], 1],
        [graphs.Path(5), [[i, i + 1] for i in range(4)], 2],
        [
            graphs.CompleteBipartite(3, 3) + Graph([(0, "a"), (0, "b"), ("a", "b")]),
            [[0, "a", "b"], [0, 1, 2, 3, 4, 5]],
            2,
        ],
        [graphs.Cycle(3) + relabeled_inc(graphs.Cycle(3)), [[0, 1, 2], [3, 4, 5]], 2],
        [
            graphs.Cycle(3) + relabeled_inc(graphs.Cycle(3), 2),
            [[0, 1, 2], [2, 3, 4]],
            2,
        ],
        [graphs.Complete(3) + Graph.from_vertices([3]), [[0, 1, 2], [3]], 2],
        [graphs.ThreePrism(), [[i for i in range(6)]], 2],
        [graphs.DoubleBanana(), [[0, 1, 2, 3, 4], [0, 1, 5, 6, 7]], 3],
        [
            graphs.Diamond() + relabeled_inc(graphs.Diamond()) + Graph([[2, 6]]),
            [[0, 1, 2, 3], [4, 5, 6, 7], [2, 6]],
            2,
        ],
        [
            # graphs.ThreeConnectedR3Circuit with 0 removed
            # and then each vertex label decreased by 1
            Graph.from_int(64842845087398392615),
            [[0, 1, 2, 3], [0, 9, 10, 11], [3, 4, 5, 6], [6, 7, 8, 9]],
            2,
        ],
    ],
)
def test_rigid_components(graph, components, dim):
    def to_sets(comps):
        return set([frozenset(comp) for comp in comps])

    comps_set = to_sets(components)

    if dim == 1:
        assert (
            to_sets(graph.rigid_components(dim=dim, algorithm="graphic")) == comps_set
        )
    elif dim == 2:
        assert to_sets(graph.rigid_components(dim=dim, algorithm="pebble")) == comps_set
        if graph.number_of_nodes() <= 8:  # since it runs through all subgraphs
            assert (
                to_sets(graph.rigid_components(dim=dim, algorithm="subgraphs-pebble"))
                == comps_set
            )

    # randomized algorithm is tested for all dimensions for graphs
    # with at most 8 vertices (since it runs through all subgraphs)
    if graph.number_of_nodes() <= 8:
        assert (
            to_sets(graph.rigid_components(dim=dim, algorithm="randomized"))
            == comps_set
        )


@pytest.mark.parametrize(
    "graph",
    [
        Graph(nx.gnp_random_graph(20, 0.1)),
        Graph(nx.gnm_random_graph(30, 62)),
        pytest.param(Graph(nx.gnm_random_graph(25, 46)), marks=pytest.mark.slow_main),
        pytest.param(Graph(nx.gnm_random_graph(40, 80)), marks=pytest.mark.slow_main),
        pytest.param(
            Graph(nx.gnm_random_graph(100, 230)), marks=pytest.mark.long_local
        ),
        pytest.param(
            Graph(nx.gnm_random_graph(100, 190)), marks=pytest.mark.long_local
        ),
    ],
)
def test_rigid_components_pebble_random_graphs(graph):
    rigid_components = graph.rigid_components(dim=2, algorithm="pebble")

    # Check that all components are rigid
    for c in rigid_components:
        new_graph = graph.subgraph(c)
        assert new_graph.is_rigid(dim=2, algorithm="sparsity")

    # Check that vertex-pairs that are not in a component are not in a rigid component
    # check every vertex pairs in the graph
    for u, v in list(combinations(graph.nodes, 2)):
        # if there is no component from rigid components that contains u and v together
        # the edge u,v can be added
        if not any([u in c and v in c for c in rigid_components]):
            graph._build_pebble_digraph(2, 3)
            assert graph._pebble_digraph.can_add_edge_between_vertices(u, v)


def test__str__():
    G = Graph([[2, 1], [2, 3]])
    assert str(G) == "Graph with vertices [1, 2, 3] and edges [[1, 2], [2, 3]]"
    G = Graph([(chr(i + 67), i + 1) for i in range(3)] + [(i, i + 1) for i in range(3)])
    assert str(G) == (
        "Graph with vertices ['C', 1, 'D', 2, 'E', 3, 0] "
        "and edges [('C', 1), (1, 0), (1, 2), ('D', 2), (2, 3), ('E', 3)]"
    )
    G = Graph.from_vertices(["C", 1, "D", 2, "E", 3, 0])
    assert str(G) == "Graph with vertices ['C', 1, 'D', 2, 'E', 3, 0] and edges []"


def test__repr__():
    assert (
        repr(Graph([[2, 1], [2, 3]]))
        == "Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)])"
    )
    assert (
        repr(Graph.from_vertices_and_edges([1, 2, 3], [(1, 2)]))
        == "Graph.from_vertices_and_edges([1, 2, 3], [(1, 2)])"
    )


def test_vertex_and_edge_lists():
    G = Graph([[2, 1], [2, 3]])
    assert G.vertex_list() == [1, 2, 3]
    assert G.edge_list() == [[1, 2], [2, 3]]
    G = Graph([(chr(i + 67), i + 1) for i in range(3)] + [(i, i + 1) for i in range(3)])
    assert set(G.vertex_list()) == {"C", 1, "D", 2, "E", 3, 0}
    assert set(G.edge_list()) == {("C", 1), (1, 0), (1, 2), ("D", 2), (2, 3), ("E", 3)}
    G = Graph.from_vertices(["C", 1, "D", 2, "E", 3, 0])
    assert set(G.vertex_list()) == {"C", 2, "E", 1, "D", 3, 0}
    assert G.edge_list() == []


def test_adjacency_matrix():
    G = Graph()
    assert G.adjacency_matrix() == Matrix([])
    G = Graph([[2, 1], [2, 3]])
    assert G.adjacency_matrix() == Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert G.adjacency_matrix(vertex_order=[2, 3, 1]) == Matrix(
        [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
    )
    assert graphs.Complete(4).adjacency_matrix() == Matrix.ones(4) - Matrix.diag(
        [1, 1, 1, 1]
    )
    G = Graph.from_vertices(["C", 1, "D"])
    assert G.adjacency_matrix() == Matrix.zeros(3)
    G = Graph.from_vertices_and_edges(["C", 1, "D"], [[1, "D"], ["C", "D"]])
    assert G.adjacency_matrix(vertex_order=["C", 1, "D"]) == Matrix(
        [[0, 0, 1], [0, 0, 1], [1, 1, 0]]
    )
    M = Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert G.from_adjacency_matrix(M).adjacency_matrix() == M
    M = Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert G.from_adjacency_matrix(M).adjacency_matrix() == M


@pytest.mark.parametrize(
    "graph, gint",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 7],
        [graphs.Complete(4), 63],
        [graphs.CompleteBipartite(3, 4), 507840],
        [graphs.CompleteBipartite(4, 4), 31965120],
        [graphs.ThreePrism(), 29327],
    ],
)
def test_integer_representation(graph, gint):
    assert graph.to_int() == gint
    assert Graph.from_int(gint).is_isomorphic(graph)
    assert Graph.from_int(gint).to_int() == gint
    assert Graph.from_int(graph.to_int()).is_isomorphic(graph)


def test_integer_representation_error():
    with pytest.raises(ValueError):
        Graph([]).to_int()
    with pytest.raises(ValueError):
        M = Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        G = Graph.from_adjacency_matrix(M)
        G.to_int()
    with pytest.raises(ValueError):
        Graph.from_int(0)
    with pytest.raises(TypeError):
        Graph.from_int(1 / 2)
    with pytest.raises(TypeError):
        Graph.from_int(1.2)
    with pytest.raises(ValueError):
        Graph.from_int(-1)


@pytest.mark.parametrize(
    "method, params",
    [
        ["is_rigid", []],
        ["is_min_rigid", []],
        ["is_redundantly_rigid", []],
        ["is_vertex_redundantly_rigid", []],
        ["is_k_vertex_redundantly_rigid", [2]],
        ["is_k_redundantly_rigid", [2]],
        ["is_globally_rigid", []],
        ["is_Rd_dependent", []],
        ["is_Rd_independent", []],
        ["is_Rd_circuit", []],
        ["is_Rd_closed", []],
        ["rigid_components", []],
        ["_input_check_no_loop", []],
        ["k_extension", [0, [1, 2], []]],
        ["zero_extension", [[1, 2], []]],
        ["one_extension", [[1, 2, 3], [1, 2]]],
    ],
)
def test_loop_error(method, params):
    with pytest.raises(LoopError):
        G = Graph([[1, 2], [1, 1], [2, 3], [1, 3]])
        func = getattr(G, method)
        func(*params)
    with pytest.raises(LoopError):
        G = Graph([[1, 1]])
        func = getattr(G, method)
        func(*params)


@pytest.mark.parametrize(
    "method, params",
    [
        ["all_k_extensions", [1]],
    ],
)
def test_iterator_loop_error(method, params):
    with pytest.raises(LoopError):
        G = Graph([[1, 2], [1, 1], [2, 3], [1, 3]])
        func = getattr(G, method)
        next(func(*params))
    with pytest.raises(LoopError):
        G = Graph([[1, 1]])
        func = getattr(G, method)
        next(func(*params))


@pytest.mark.parametrize(
    "method, params",
    [
        ["extension_sequence", [1.1]],
        ["is_Rd_circuit", [2.1]],
        ["is_Rd_closed", [3.2]],
        ["is_Rd_dependent", [3 / 2]],
        ["is_Rd_independent", [1.2]],
        ["is_globally_rigid", [3.1]],
        ["is_k_redundantly_rigid", [2, 3.7]],
        ["is_k_vertex_redundantly_rigid", [2, 2.3]],
        ["is_min_k_redundantly_rigid", [2, 3.7]],
        ["is_min_k_vertex_redundantly_rigid", [2, 2.3]],
        ["is_min_redundantly_rigid", [2.6]],
        ["is_min_vertex_redundantly_rigid", [3.2]],
        ["is_min_rigid", [1.2]],
        ["is_rigid", [1.1]],
        ["is_redundantly_rigid", [math.log(2)]],
        ["is_vertex_redundantly_rigid", [4.8]],
        ["k_extension", [0, [1, 2], [], 4, 2.6]],
        ["one_extension", [[1, 2, 3], [1, 2], 4, 2.6]],
        ["random_framework", [1.1]],
        ["rigid_components", [3.7]],
        ["zero_extension", [[1, 2], 4, 2.6]],
    ],
)
def test_dimension_type_error(method, params):
    with pytest.raises(TypeError):
        G = Graph([[1, 2], [1, 3], [2, 3]])
        func = getattr(G, method)
        func(*params)


@pytest.mark.parametrize(
    "method, params",
    [
        ["extension_sequence", [0]],
        ["extension_sequence", [-2]],
        ["is_Rd_circuit", [0]],
        ["is_Rd_circuit", [-1]],
        ["is_Rd_closed", [0]],
        ["is_Rd_closed", [-2]],
        ["is_Rd_dependent", [0]],
        ["is_Rd_dependent", [-2]],
        ["is_Rd_independent", [0]],
        ["is_Rd_independent", [-1]],
        ["is_globally_rigid", [0]],
        ["is_globally_rigid", [-2]],
        ["is_k_redundantly_rigid", [2, 0]],
        ["is_k_redundantly_rigid", [2, -4]],
        ["is_k_vertex_redundantly_rigid", [2, 0]],
        ["is_k_vertex_redundantly_rigid", [2, -7]],
        ["is_min_k_redundantly_rigid", [2, 0]],
        ["is_min_k_redundantly_rigid", [2, -4]],
        ["is_min_k_vertex_redundantly_rigid", [2, 0]],
        ["is_min_k_vertex_redundantly_rigid", [2, -7]],
        ["is_min_redundantly_rigid", [0]],
        ["is_min_redundantly_rigid", [-2]],
        ["is_min_vertex_redundantly_rigid", [0]],
        ["is_min_vertex_redundantly_rigid", [-4]],
        ["is_min_rigid", [0]],
        ["is_min_rigid", [-3]],
        ["is_rigid", [0]],
        ["is_rigid", [-2]],
        ["is_redundantly_rigid", [0]],
        ["is_redundantly_rigid", [-2]],
        ["is_vertex_redundantly_rigid", [0]],
        ["is_vertex_redundantly_rigid", [-3]],
        ["k_extension", [0, [1, 2], [], 4, 0]],
        ["k_extension", [0, [1, 2], [], 4, -3]],
        ["one_extension", [[1, 2, 3], [1, 2], 4, 0]],
        ["one_extension", [[1, 2, 3], [1, 2], 4, -3]],
        ["random_framework", [0]],
        ["random_framework", [-2]],
        ["rigid_components", [0]],
        ["rigid_components", [-4]],
        ["zero_extension", [[1, 2], 4, 0]],
        ["zero_extension", [[1, 2], 4, -3]],
    ],
)
def test_dimension_value_error(method, params):
    with pytest.raises(ValueError):
        G = Graph([[1, 2], [1, 3], [2, 3]])
        func = getattr(G, method)
        func(*params)


@pytest.mark.parametrize(
    "method, params",
    [
        ["all_k_extensions", [1, 2.1]],
    ],
)
def test_iterator_dimension_type_error(method, params):
    with pytest.raises(TypeError):
        G = Graph([[1, 2], [1, 3], [2, 3]])
        func = getattr(G, method)
        next(func(*params))


@pytest.mark.parametrize(
    "method, params",
    [
        ["all_k_extensions", [1, 0]],
        ["all_k_extensions", [2, -1]],
    ],
)
def test_iterator_dimension_value_error(method, params):
    with pytest.raises(ValueError):
        G = Graph([[1, 2], [1, 3], [2, 3]])
        func = getattr(G, method)
        next(func(*params))


@pytest.mark.parametrize(
    "method, params",
    [
        ["is_k_redundantly_rigid", [2.4, 3]],
        ["is_k_vertex_redundantly_rigid", [3.7, 2]],
        ["is_min_k_redundantly_rigid", [2.5, 3]],
        ["is_min_k_vertex_redundantly_rigid", [2 / 3, 2]],
        ["k_extension", [0.3, [1, 2], [], 4, 2]],
    ],
)
def test_parameter_type_error(method, params):
    with pytest.raises(TypeError):
        G = Graph([[1, 2], [1, 3], [2, 3]])
        func = getattr(G, method)
        func(*params)


@pytest.mark.parametrize(
    "method, params",
    [
        ["is_k_redundantly_rigid", [-1, 3]],
        ["is_k_redundantly_rigid", [-2, 4]],
        ["is_k_vertex_redundantly_rigid", [-1, 2]],
        ["is_k_vertex_redundantly_rigid", [-3, 7]],
        ["is_min_k_redundantly_rigid", [-1, 3]],
        ["is_min_k_redundantly_rigid", [-2, 4]],
        ["is_min_k_vertex_redundantly_rigid", [-1, 2]],
        ["is_min_k_vertex_redundantly_rigid", [-3, 7]],
        ["k_extension", [-1, [1, 2], [], 4, 2]],
        ["k_extension", [-2, [1, 2], [], 4, 3]],
    ],
)
def test_parameter_value_error(method, params):
    with pytest.raises(ValueError):
        G = Graph([[1, 2], [1, 3], [2, 3]])
        func = getattr(G, method)
        func(*params)


@pytest.mark.parametrize(
    "method, params",
    [
        ["all_k_extensions", [1.1, 2]],
    ],
)
def test_iterator_parameter_type_error(method, params):
    with pytest.raises(TypeError):
        G = Graph([[1, 2], [1, 3], [2, 3]])
        func = getattr(G, method)
        next(func(*params))


@pytest.mark.parametrize(
    "method, params",
    [
        ["all_k_extensions", [-1, 2]],
        ["all_k_extensions", [-2, 1]],
    ],
)
def test_iterator_parameter_value_error(method, params):
    with pytest.raises(ValueError):
        G = Graph([[1, 2], [1, 3], [2, 3]])
        func = getattr(G, method)
        next(func(*params))


@pytest.mark.parametrize(
    "method, params",
    [
        ["is_min_rigid", [3]],
        ["is_rigid", [3]],
    ],
)
def test_dimension_sparsity_error(method, params):
    with pytest.raises(ValueError):
        G = graphs.DoubleBanana()
        func = getattr(G, method)
        func(*params, algorithm="sparsity")


def test_k_extension():
    assert graphs.Complete(2).zero_extension([0, 1]) == graphs.Complete(3)
    assert graphs.Complete(2).zero_extension([1], dim=1) == graphs.Path(3)
    assert graphs.Complete(4).one_extension([0, 1, 2], (0, 1)) == Graph(
        [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
    )
    assert graphs.CompleteBipartite(3, 2).one_extension(
        [0, 1, 2, 3, 4], (0, 3), dim=4
    ) == Graph(
        [
            (0, 4),
            (0, 5),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 5),
            (4, 5),
        ]
    )

    assert graphs.CompleteBipartite(3, 2).k_extension(
        2, [0, 1, 3], [(0, 3), (1, 3)], dim=1
    ) == Graph([(0, 4), (0, 5), (1, 4), (1, 5), (2, 3), (2, 4), (3, 5)])
    assert graphs.CompleteBipartite(3, 2).k_extension(
        2, [0, 1, 3, 4], [(0, 3), (1, 3)]
    ) == Graph([(0, 4), (0, 5), (1, 4), (1, 5), (2, 3), (2, 4), (3, 5), (4, 5)])
    assert graphs.Cycle(6).k_extension(
        4, [0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4)], dim=1
    ) == Graph([(0, 5), (0, 6), (1, 6), (2, 6), (3, 6), (4, 5), (4, 6)])


def test_all_k_extensions():
    for extension in graphs.Complete(4).all_k_extensions(1, 1):
        assert extension in [
            Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3]]),
            Graph([[0, 1], [0, 3], [0, 4], [1, 2], [1, 3], [2, 3], [2, 4]]),
            Graph([[0, 1], [0, 2], [0, 4], [1, 2], [1, 3], [2, 3], [3, 4]]),
            Graph([[0, 1], [0, 2], [0, 3], [1, 3], [1, 4], [2, 3], [2, 4]]),
            Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 3], [3, 4]]),
            Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 4], [3, 4]]),
        ]
    for extension in graphs.Complete(4).all_k_extensions(
        2, 2, only_non_isomorphic=True
    ):
        assert extension in [
            Graph([[0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]),
            Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 4], [3, 4]]),
        ]
    all_diamond_0_2 = list(
        graphs.Diamond().all_k_extensions(0, 2, only_non_isomorphic=True)
    )
    assert (
        len(all_diamond_0_2) == 3
        and all_diamond_0_2[0]
        == Graph([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 4], [2, 3]])
        and all_diamond_0_2[1]
        == Graph([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [2, 4]])
        and all_diamond_0_2[2]
        == Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 3], [3, 4]])
    )
    all_diamond_1_2 = graphs.Diamond().all_k_extensions(1, 2, only_non_isomorphic=True)
    assert next(all_diamond_1_2) == Graph(
        [[0, 2], [0, 3], [0, 4], [1, 2], [1, 4], [2, 3], [2, 4]]
    ) and next(all_diamond_1_2) == Graph(
        [[0, 2], [0, 3], [0, 4], [1, 2], [1, 4], [2, 3], [3, 4]]
    )


@pytest.mark.parametrize(
    "graph, k, dim, sol",
    [
        [Graph.from_int(254), 1, 2, [3934, 4011, 6891, 7672, 7916]],
        [graphs.Diamond(), 0, 2, [223, 239, 254]],
        [graphs.Complete(4), 0, 3, [511]],
        [graphs.CompleteMinusOne(5), 0, 1, [1535, 8703]],
        [
            Graph.from_int(16350),
            2,
            3,
            [257911, 260603, 376807, 384943, 1497823, 1973983],
        ],
        [graphs.CompleteMinusOne(5), 2, 3, [4095, 7679, 7935, 8187]],
    ],
)
def test_all_k_extensions2(graph, k, dim, sol):
    assert misc.is_isomorphic_graph_list(
        list(graph.all_k_extensions(k, dim, only_non_isomorphic=True)),
        [Graph.from_int(igraph) for igraph in sol],
    )


@pytest.mark.parametrize(
    "graph, k, vertices, edges, dim",
    [
        [graphs.Complete(6), 2, [0, 1, 2], [[0, 1], [0, 2]], -1],
        [graphs.Complete(6), 2, [0, 1, 6], [[0, 1], [0, 6]], 1],
        [graphs.Complete(6), 2, [0, 1, 2], [[0, 1]], 1],
        [graphs.Complete(3), -1, [0], [], 2],
        [graphs.CompleteBipartite(2, 3), 2, [0, 1, 2], [[0, 1], [0, 2]], 1],
    ],
)
def test_k_extension_dim_error(graph, k, vertices, edges, dim):
    with pytest.raises(ValueError):
        graph.k_extension(k, vertices, edges, dim)


@pytest.mark.parametrize(
    "graph, k, vertices, edges",
    [
        [
            Graph.from_vertices_and_edges([1, 2, 3], [[1, 2], [2, 3], [3, 3]]),
            1,
            [1, 2, 3],
            [[3, 3]],
        ],
        [graphs.Complete(6), 2, [1, 2, 3, 4], [[1, 2], [1, 2]]],
        [graphs.Complete(6), 2, [1, 2, 3, 4], [[1, 2], [2, 1]]],
        [graphs.Complete(6), 2, [1, 2, 3, 4], [(1, 2), [1, 2]]],
        [graphs.Complete(6), 2, [1, 2, 3, 4], [(1, 2), [2, 1]]],
        [graphs.Complete(6), 2, [1, 2, 3, 4], [(1, 2), (1, 2)]],
        [graphs.Complete(6), 2, [1, 2, 3, 4], [(1, 2), (2, 1)]],
        [graphs.Complete(6), 3, [1, 2, 3, 4, 5], [[1, 2], [2, 3], [1, 2]]],
    ],
)
def test_k_extension_error(graph, k, vertices, edges):
    with pytest.raises(ValueError):
        graph.k_extension(k, vertices, edges)


def test_all_k_extension_error():
    with pytest.raises(ValueError):
        list(Graph.from_vertices([0, 1, 2]).all_k_extensions(1, 1))


@pytest.mark.parametrize(
    "graph, dim, sol",
    [
        [Graph.from_int(254), 2, [3326, 3934, 4011, 6891, 7672, 7916, 10479, 12511]],
        [graphs.Diamond(), 2, [223, 239, 254]],
        [graphs.Complete(4), 3, [511]],
        [graphs.Complete(1), 1, [1]],
        [graphs.CompleteMinusOne(5), 1, [1535, 8703]],
        [
            Graph.from_int(16350),
            3,
            [257911, 260603, 376807, 384943, 515806, 981215, 1497823, 1973983],
        ],
        [graphs.CompleteMinusOne(5), 3, [4095, 7679, 7935, 8187, 16350]],
    ],
)
def test_all_extensions(graph, dim, sol):
    assert misc.is_isomorphic_graph_list(
        list(graph.all_extensions(dim, only_non_isomorphic=True)),
        [Graph.from_int(igraph) for igraph in sol],
    )


@pytest.mark.parametrize(
    "graph, dim",
    [
        [Graph.from_int(254), 2],
        [graphs.Diamond(), 2],
        [graphs.ThreePrism(), 2],
        [graphs.Complete(2), 1],
        [graphs.Complete(1), 1],
        [graphs.CompleteMinusOne(5), 1],
        pytest.param(Graph.from_int(16350), 3, marks=pytest.mark.slow_main),
        [graphs.CompleteMinusOne(5), 3],
    ],
)
def test_all_extensions_single(graph, dim):
    for k in range(0, dim):
        assert misc.is_isomorphic_graph_list(
            list(graph.all_extensions(dim, only_non_isomorphic=True, k_min=k, k_max=k)),
            list(graph.all_k_extensions(k, dim, only_non_isomorphic=True)),
        )
        assert misc.is_isomorphic_graph_list(
            list(graph.all_extensions(dim, k_min=k, k_max=k)),
            list(graph.all_k_extensions(k, dim)),
        )


@pytest.mark.parametrize(
    "graph, dim, k_min, k_max",
    [
        [graphs.Diamond(), 2, -1, 0],
        [graphs.ThreePrism(), 2, 0, -1],
        [graphs.Diamond(), 2, 2, 1],
        [graphs.Diamond(), 2, 3, None],
        [graphs.Complete(4), 3, -2, -1],
        [graphs.CompleteMinusOne(5), 1, 5, 4],
        [graphs.Complete(3), 3, 5, None],
    ],
)
def test_all_extensions_value_error(graph, dim, k_min, k_max):
    with pytest.raises(ValueError):
        list(graph.all_extensions(dim=dim, k_min=k_min, k_max=k_max))


@pytest.mark.parametrize(
    "graph, dim, k_min, k_max",
    [
        [graphs.Diamond(), 2, 0, 1.4],
        [graphs.Diamond(), 2, 0.2, 2],
        [graphs.Diamond(), 1.2, 2, 1],
        [graphs.Diamond(), "2", 2, 1],
        [graphs.Diamond(), 1, 2, "1"],
        [graphs.Diamond(), 2, 3 / 2, None],
        [graphs.Diamond(), 2, "2", None],
        [graphs.Diamond(), None, 2, 1],
        [graphs.Diamond(), 1, None, 1],
    ],
)
def test_all_extensions_type_error(graph, dim, k_min, k_max):
    with pytest.raises(TypeError):
        list(graph.all_extensions(dim=dim, k_min=k_min, k_max=k_max))


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Complete(3),
        graphs.CompleteBipartite(3, 3),
        graphs.Diamond(),
        graphs.ThreePrism(),
        graphs.CubeWithDiagonal(),
        Graph.from_int(6462968),
        Graph.from_int(69380589),
        Graph.from_int(19617907),
        Graph.from_int(170993054),
        Graph.from_int(173090142),
    ],
)
def test_has_extension_sequence(graph):
    assert graph.has_extension_sequence()


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Path(3),
        graphs.CompleteBipartite(1, 2),
        graphs.Complete(4),
        graphs.Cycle(6),
        graphs.K33plusEdge(),
        graphs.ThreePrismPlusEdge(),
        Graph.from_int(2269176),
        Graph.from_int(19650659),
        Graph.from_vertices([0]),
        Graph.from_vertices([]),
    ],
)
def test_has_not_extension_sequence(graph):
    assert not graph.has_extension_sequence()


def test_extension_sequence_solution():
    assert graphs.Complete(2).extension_sequence(return_type="graphs") == [
        Graph([[0, 1]]),
    ]

    assert graphs.Complete(3).extension_sequence(return_type="graphs") == [
        Graph([[1, 2]]),
        Graph([[0, 1], [0, 2], [1, 2]]),
    ]

    solution = [
        Graph([[3, 4]]),
        Graph([[2, 3], [2, 4], [3, 4]]),
        Graph([[1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]),
        Graph([[1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4]]),
        Graph(
            [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]],
        ),
    ]
    assert (
        graphs.CompleteBipartite(3, 3).extension_sequence(return_type="graphs")
        == solution
    )

    solution_ext = [
        [0, [3, 4], [], 2],  # k, vertices, edges, new_vertex
        [0, [3, 4], [], 1],
        [0, [1, 2], [], 5],
        [1, [3, 4, 5], [(3, 4)], 0],
    ]
    G = Graph([[3, 4]])
    for i in range(len(solution)):
        assert solution[i] == G
        if i < len(solution_ext):
            G.k_extension(*solution_ext[i], dim=2, inplace=True)

    assert graphs.Diamond().extension_sequence(return_type="graphs") == [
        Graph([[2, 3]]),
        Graph([[0, 2], [0, 3], [2, 3]]),
        Graph([[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]]),
    ]

    result = graphs.ThreePrism().extension_sequence(return_type="graphs")
    solution = [
        Graph([[4, 5]]),
        Graph([[3, 4], [3, 5], [4, 5]]),
        Graph([[1, 3], [1, 4], [3, 4], [3, 5], [4, 5]]),
        Graph([[1, 2], [1, 3], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]]),
        Graph(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]],
        ),
    ]
    assert solution == result
    solution_ext = [
        [0, [4, 5], [], 3],  # k, vertices, edges, new_vertex
        [0, [3, 4], [], 1],
        [0, [1, 5], [], 2],
        [1, [1, 2, 3], [(1, 3)], 0],
    ]
    G = Graph([[4, 5]])
    for i in range(len(result)):
        assert result[i] == G
        if i < len(solution_ext):
            G.k_extension(*solution_ext[i], dim=2, inplace=True)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Complete(3),
        graphs.CompleteBipartite(3, 3),
        graphs.Diamond(),
        graphs.ThreePrism(),
        graphs.CubeWithDiagonal(),
        Graph.from_int(6462968),
        Graph.from_int(69380589),
        Graph.from_int(19617907),
        Graph.from_int(170993054),
        Graph.from_int(173090142),
    ],
)
def test_extension_sequence(graph):
    ext = graph.extension_sequence(return_type="both")
    assert ext is not None
    current = ext[0]
    for i in range(1, len(ext)):
        current = current.k_extension(*ext[i][1])
        assert current == ext[i][0]


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.Complete(2), 2],
        [graphs.Complete(3), 2],
        [graphs.CompleteBipartite(3, 3), 2],
        [graphs.Diamond(), 2],
        [graphs.ThreePrism(), 2],
        [graphs.CubeWithDiagonal(), 2],
        [Graph.from_int(6462968), 2],
        [Graph.from_int(69380589), 2],
        [Graph.from_int(19617907), 2],
        [Graph.from_int(170993054), 2],
        [Graph.from_int(173090142), 2],
        [graphs.Complete(2), 1],
        [Graph.from_int(75), 1],
        [Graph.from_int(77), 1],
        [Graph.from_int(86), 1],
        [graphs.Complete(1), 1],
        [graphs.Complete(4), 3],
        [graphs.CompleteMinusOne(5), 3],
        [Graph.from_int(16350), 3],
        [Graph.from_int(4095), 3],
        [graphs.DoubleBanana(), 3],
    ],
)
def test_extension_sequence_dim(graph, dim):
    ext = graph.extension_sequence(dim=dim, return_type="both")
    assert ext is not None
    current = ext[0]
    for i in range(1, len(ext)):
        current = current.k_extension(*ext[i][1], dim=dim)
        assert current == ext[i][0]


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.Complete(2), 2],
        [graphs.Complete(3), 2],
        [graphs.CompleteBipartite(3, 3), 2],
        [graphs.Diamond(), 2],
        [graphs.ThreePrism(), 2],
        [graphs.CubeWithDiagonal(), 2],
        [Graph.from_int(6462968), 2],
        [Graph.from_int(69380589), 2],
        [Graph.from_int(19617907), 2],
        [Graph.from_int(170993054), 2],
        [Graph.from_int(173090142), 2],
        [graphs.Complete(2), 1],
        [Graph.from_int(75), 1],
        [Graph.from_int(77), 1],
        [Graph.from_int(86), 1],
        [graphs.Complete(1), 1],
    ],
)
def test_extension_sequence_min_rigid(graph, dim):
    ext = graph.extension_sequence(dim=dim, return_type="graphs")
    assert ext is not None
    for current in ext:
        assert current.is_min_rigid(dim)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Path(3),
        graphs.CompleteBipartite(1, 2),
        graphs.Complete(4),
        graphs.Cycle(6),
        graphs.K33plusEdge(),
        graphs.ThreePrismPlusEdge(),
        Graph.from_int(2269176),
        Graph.from_int(19650659),
        Graph.from_vertices([0]),
        Graph.from_vertices([]),
    ],
)
def test_extension_sequence_none(graph):
    assert graph.extension_sequence() is None


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.Path(3), 2],
        [graphs.CompleteBipartite(1, 2), 2],
        [graphs.Complete(4), 2],
        [graphs.Cycle(6), 2],
        [graphs.K33plusEdge(), 2],
        [graphs.ThreePrismPlusEdge(), 2],
        [Graph.from_int(2269176), 2],
        [Graph.from_int(19650659), 2],
        [Graph.from_vertices([0]), 2],
        [Graph.from_vertices([]), 2],
        [graphs.Cycle(3), 1],
        [graphs.Complete(4), 1],
        [graphs.Complete(6), 3],
    ],
)
def test_extension_sequence_dim_none(graph, dim):
    assert graph.extension_sequence(dim) is None


def test_extension_sequence_error():
    with pytest.raises(NotSupportedValueError):
        graphs.Complete(3).extension_sequence(return_type="Test")


@pytest.mark.parametrize(
    "graph",
    [
        Graph.from_vertices([]),
        Graph.from_vertices([1, 2, 3]),
        Graph.from_vertices_and_edges([1, 2, 3], [[1, 2], [2, 3]]),
        Graph([[1, 2], [2, 3]]),
    ],
)
def test__input_check_no_loop(graph):
    assert graph._input_check_no_loop() is None


@pytest.mark.parametrize(
    "graph",
    [
        Graph([[1, 1]]),
        Graph([[1, 2], [2, 3], [3, 3]]),
    ],
)
def test__input_check_no_loop_error(graph):
    with pytest.raises(LoopError):
        graph._input_check_no_loop()


@pytest.mark.parametrize(
    "vertices, edges",
    [
        [[1], [[1, 1]]],
        [[1, 2, 3], [[1, 2], [2, 3], [3, 3]]],
    ],
)
def test__input_check_no_loop_error2(vertices, edges):
    with pytest.raises(LoopError):
        Graph.from_vertices_and_edges(vertices, edges)._input_check_no_loop()


@pytest.mark.parametrize(
    "graph, vertex",
    [
        [Graph.from_vertices([1]), 1],
        [Graph.from_vertices([1, 2, 3]), 3],
        [Graph.from_vertices_and_edges([1, 2, 3], [[1, 2], [2, 3]]), 3],
        [Graph([[1, 2], [2, 3]]), 2],
        [Graph([[1, 2], [1, 1]]), 1],
        [graphs.Complete(3), 0],
        [graphs.Diamond(), 3],
        [Graph.from_vertices([1]), [1]],
        [Graph.from_vertices([1, 2, 3]), [2, 3]],
        [Graph.from_vertices_and_edges([1, 2, 3], [[1, 2], [2, 3]]), [1, 3]],
        [Graph([[1, 2], [2, 3]]), [2, 2]],
        [graphs.Complete(3), [0, 1]],
        [graphs.Diamond(), [1, 3]],
        [Graph([["a", "b"], ["b", 3]]), "a"],
        [Graph([["a", "b"], ["b", 3]]), ["a", "b"]],
        [Graph([["a", "b"], ["b", 3]]), ["a", 3]],
        [Graph([[-1, -2], [-2, 3]]), -1],
        [Graph([[-1, -2], [-2, 3]]), [-1, -2]],
        [Graph([[-1, -2], [-2, 3]]), [-1, 3]],
    ],
)
def test__input_check_vertex_members(graph, vertex):
    assert graph._input_check_vertex_members(vertex) is None


@pytest.mark.parametrize(
    "graph, vertex",
    [
        [Graph([]), 1],
        [Graph.from_vertices([1]), 2],
        [Graph.from_vertices([1, 2, 3]), 4],
        [Graph.from_vertices_and_edges([1, 2, 3], [[1, 2], [2, 3]]), -1],
        [Graph([[1, 2], [2, 3]]), 0],
        [Graph([[1, 2], [1, 1]]), 3],
        [graphs.Complete(3), "a"],
        [graphs.Diamond(), 10],
        [Graph.from_vertices([1]), [2]],
        [Graph.from_vertices([1, 2, 3]), [3, 4]],
        [Graph.from_vertices_and_edges([1, 2, 3], [[1, 2], [2, 3]]), [5, 6]],
        [Graph([[1, 2], [2, 3]]), [2, 2, 4]],
        [graphs.Complete(3), [0, 4]],
        [graphs.Diamond(), [1, 2, 12]],
        [Graph([["a", "b"], ["b", 3]]), "c"],
        [Graph([["a", "b"], ["b", 3]]), ["a", "c"]],
        [Graph([["a", "b"], ["b", 3]]), ["a", 4]],
        [Graph([[-1, -2], [-2, 3]]), -3],
        [Graph([[-1, -2], [-2, 3]]), [-1, -2, 4]],
        [Graph([[-1, -2], [-2, 3]]), [-1, 3, -3]],
    ],
)
def test__input_check_vertex_members_error(graph, vertex):
    with pytest.raises(ValueError):
        graph._input_check_vertex_members(vertex)


@pytest.mark.parametrize(
    "graph, edge",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), (1, 2)],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [3, 2]],
        [Graph([[1, 2], [2, 3]]), [1, 2]],
        [Graph([[1, 2], [1, 1]]), [1, 1]],
        [graphs.Complete(3), [0, 1]],
        [graphs.Diamond(), [1, 2]],
        [Graph([["a", "b"], ["b", 3]]), ["a", "b"]],
        [Graph([["a", "b"], ["b", 3]]), (3, "b")],
        [Graph([["a", "b"], ["b", 3]]), ["b", "a"]],
        [Graph([[-1, -2], [-2, 3]]), [-2, -1]],
        [Graph([[-1, -2], [-2, 3]]), [-1, -2]],
        [Graph([[-1, -2], [-2, 3]]), [-2, 3]],
    ],
)
def test__input_check_edge(graph, edge):
    assert graph._input_check_edge(edge) is None
    assert graph._input_check_edge_format(edge) is None


@pytest.mark.parametrize(
    "graph, edge, vertices",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), (1, 2), [1, 2, 2]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [3, 2], [1, 2, 3]],
        [Graph([[1, 2], [2, 3]]), [1, 2], [2, 1]],
        [Graph([[1, 2], [2, 3], [3, 4]]), [1, 2], [3, 2, 1]],
        [Graph([[1, 2], [1, 1]]), [1, 1], [1, 2]],
        [Graph([[1, 2], [1, 1]]), [1, 1], [1, 1]],
        [Graph([[1, 2], [1, 1]]), [1, 1], [1]],
        [graphs.Complete(3), [0, 1], [0, 1, 2, 3, 4]],
        [graphs.Diamond(), [1, 2], [1, 2, 3]],
        [Graph([["a", "b"], ["b", 3]]), ["a", "b"], ["a", "b"]],
        [Graph([["a", "b"], ["b", 3]]), (3, "b"), ["a", "b", 3]],
        [Graph([["a", "b"], ["b", 3]]), ["b", "a"], ["a", "b", 3]],
        [Graph([[-1, -2], [-2, 3]]), [-2, -1], [-3, -2, -1, 0, 1, 2, 3]],
        [Graph([[-1, -2], [-2, 3]]), [-1, -2], [-1, -2, 3]],
        [Graph([[-1, -2], [-2, 3]]), [-2, 3], [-2, 3]],
    ],
)
def test__input_check_edge_on_vertices(graph, edge, vertices):
    assert graph._input_check_edge(edge, vertices) is None


@pytest.mark.parametrize(
    "graph, edge",
    [
        [Graph([]), (1, 3)],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), (1, 3)],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [3, 1]],
        [Graph([[1, 2], [2, 3]]), [1, 3]],
        [graphs.Complete(3), [0, 4]],
        [graphs.Diamond(), [1, -2]],
        [Graph([["a", "b"], ["b", 3]]), ["a", "c"]],
        [Graph([["a", "b"], ["b", 3]]), (3, "a")],
        [Graph([["a", "b"], ["b", 3]]), ["3", "a"]],
        [Graph([[-1, -2], [-2, 3]]), [3, -1]],
        [Graph([[-1, -2], [-2, 3]]), [-1, 0]],
        [Graph([[-1, -2], [-2, 3]]), [-2, -3]],
        [Graph([[1, 2], [1, 1]]), [2, 2]],
    ],
)
def test__input_check_edge_value_error(graph, edge):
    with pytest.raises(ValueError):
        graph._input_check_edge(edge)


@pytest.mark.parametrize(
    "graph, edge",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), (1, 1)],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [3, 3]],
        [Graph([["a", "b"], ["b", 3]]), ["a", "a"]],
        [Graph([[-1, -2], [-2, 3]]), [-2, -2]],
        [Graph([[1, 2], [1, 1]]), [1, 1]],
    ],
)
def test__input_check_edge_format_loopfree_loop_error(graph, edge):
    assert graph._input_check_edge_format(edge, loopfree=False) is None
    assert graph._input_check_edge_format(edge) is None
    with pytest.raises(LoopError):
        graph._input_check_edge_format(edge, loopfree=True)


@pytest.mark.parametrize(
    "graph, edge, vertices",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), (1, 2), [1, 3, 3]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [3, 2], [1, 3]],
        [Graph([[1, 2], [2, 3]]), [1, 2], [2, 2]],
        [Graph([[1, 2], [2, 3], [3, 4]]), [1, 2], [3, 2]],
        [Graph([[1, 2], [1, 1]]), [1, 1], [2, 2]],
        [Graph([[1, 2], [1, 1]]), [1, 1], [2, 3]],
        [Graph([[1, 2], [1, 1]]), [1, 1], [0]],
        [graphs.Complete(3), [0, 1], [1, 2, 3, 4]],
        [graphs.Diamond(), [1, 2], [1, 3]],
        [Graph([["a", "b"], ["b", 3]]), ["a", "b"], ["a", "c"]],
        [Graph([["a", "b"], ["b", 3]]), (3, "b"), ["a", "b", 2]],
        [Graph([["a", "b"], ["b", 3]]), ["b", "a"], ["a"]],
        [Graph([[-1, -2], [-2, 3]]), [-2, -1], [-3, -2, 0, 1, 2, 3]],
        [Graph([[-1, -2], [-2, 3]]), [-1, -2], [-2, 3]],
        [Graph([[-1, -2], [-2, 3]]), [-2, 3], [3]],
        [graphs.Diamond(), [[1, 2], [2, 3]], None],
        [graphs.Diamond(), [[1, 2], [2, 3]], [1, 2, 3]],
    ],
)
def test__input_check_edge_on_vertices_value_error(graph, edge, vertices):
    with pytest.raises(ValueError):
        graph._input_check_edge(edge, vertices)


@pytest.mark.parametrize(
    "graph, edge",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), (1,)],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), 1],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [1]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [1, 2, 3]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), "[3, 2]"],
        [Graph([[1, 2], [2, 3]]), "12"],
        [graphs.Complete(3), [[0, 1]]],
    ],
)
def test__input_check_edge_type_error(graph, edge):
    with pytest.raises(TypeError):
        graph._input_check_edge(edge)
    with pytest.raises(TypeError):
        graph._input_check_edge_format(edge)


@pytest.mark.parametrize(
    "graph, edge, vertices",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), (1,), [1, 2, 3]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), 1, [1, 2, 3]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [1], [1, 2, 3]],
        [Graph([(1, 2), (2, 3)]), [1, 2, 3], [1, 2, 3]],
        [
            Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]),
            "[3, 2]",
            [1, 2, 3],
        ],
        [Graph([[1, 2], [2, 3]]), "12", [1, 2, 3]],
        [graphs.Complete(3), [[0, 1]], [1, 2, 3]],
        [Graph([[1, 2], [2, 3]]), [1, 2], "1"],
        [Graph([[1, 2], [2, 3]]), [1, 2], 1],
    ],
)
def test__input_check_edge_on_vertices_type_error(graph, edge, vertices):
    with pytest.raises(TypeError):
        graph._input_check_edge(edge, vertices)


@pytest.mark.parametrize(
    "graph, edge",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [(1, 2)]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [[3, 2]]],
        [Graph([[1, 2], [2, 3]]), [[1, 2]]],
        # [Graph([[1, 2], [1, 1]]), [[1, 1]]],
        [graphs.Complete(3), [[0, 1]]],
        [graphs.Diamond(), [[1, 2]]],
        [Graph([["a", "b"], ["b", 3]]), [["a", "b"]]],
        [Graph([["a", "b"], ["b", 3]]), [(3, "b")]],
        [Graph([["a", "b"], ["b", 3]]), [["b", "a"]]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, -1]]],
        [Graph([[-1, -2], [-2, 3]]), [[-1, -2]]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, 3]]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [(1, 2), (3, 2)]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [[3, 2], [1, 2]]],
        [Graph([[1, 2], [2, 3]]), [[1, 2], (2, 3)]],
        [graphs.Complete(3), [[0, 1], [1, 2]]],
        [graphs.Diamond(), [[1, 2], [2, 3]]],
        [Graph([["a", "b"], ["b", 3]]), [["a", "b"], ["b", 3]]],
        [Graph([["a", "b"], ["b", 3]]), [(3, "b"), ("a", "b")]],
        [Graph([["a", "b"], ["b", 3]]), [["b", "a"], (3, "b")]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, -1], [-2, 3]]],
        [Graph([[-1, -2], [-2, 3]]), [[-1, -2], (-2, 3)]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, 3], [-1, -2]]],
    ],
)
def test__input_check_edge_list(graph, edge):
    assert graph._input_check_edge_list(edge) is None
    assert graph._input_check_edge_format_list(edge) is None


@pytest.mark.parametrize(
    "graph, edge",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [(1, 3)]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [[3, 1]]],
        [Graph([[1, 2], [2, 3]]), [[1, 3]]],
        [graphs.Complete(3), [[0, 4]]],
        [graphs.Diamond(), [[1, -2]]],
        [Graph([["a", "b"], ["b", 3]]), [["a", "c"]]],
        [Graph([["a", "b"], ["b", 3]]), [(3, "a")]],
        [Graph([["a", "b"], ["b", 3]]), [["3", "a"]]],
        [Graph([[-1, -2], [-2, 3]]), [[3, -1]]],
        [Graph([[-1, -2], [-2, 3]]), [[-1, 0]]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, -3]]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [(1, 2), (3, 3)]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [[3, 2], [1, 3]]],
        [Graph([[1, 2], [2, 3]]), [[1, 2], (2, 4)]],
        [graphs.Complete(3), [[0, 1], [1, -2]]],
        [graphs.Diamond(), [[1, 5], [2, 3]]],
        [Graph([["a", "b"], ["b", 3]]), [["a", "c"], ["b", 3]]],
        [Graph([["a", "b"], ["b", 3]]), [(3, "b"), ("a", "d")]],
        [Graph([["a", "b"], ["b", 3]]), [["b", "3"], (3, "b")]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, -1], [1, 3]]],
        [Graph([[-1, -2], [-2, 3]]), [[-1, 5], (-2, 3)]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, -3], [-1, -2]]],
        [graphs.Diamond(), [[[1, 2], [2, 3]]]],
    ],
)
def test__input_check_edge_list_value_error(graph, edge):
    with pytest.raises(ValueError):
        graph._input_check_edge_list(edge)


@pytest.mark.parametrize(
    "graph, edge",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), (1,)],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), 1],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [(1,)]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [1]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), "[3, 2]"],
        [Graph([[1, 2], [2, 3]]), "12"],
        [graphs.Complete(3), [0, 1]],
        [graphs.Diamond(), (1, 2)],
    ],
)
def test__input_check_edge_list_type_error(graph, edge):
    with pytest.raises(TypeError):
        graph._input_check_edge_list(edge)
    with pytest.raises(TypeError):
        graph._input_check_edge_format_list(edge)


@pytest.mark.parametrize(
    "graph, vertex_order",
    [
        [Graph([("a", 1.8), ("a", "#"), ("#", 0), (0, 1.8)]), ["a", "#", 0, 1.8]],
        [Graph([[1, 2], [2, 3]]), [1, 2, 3]],
        [Graph([[1, 2], [2, 3]]), [1, 3, 2]],
        [graphs.Complete(3), [0, 1, 2]],
    ],
)
def test__input_check_vertex_order(graph, vertex_order):
    assert graph._input_check_vertex_order(vertex_order) == vertex_order


@pytest.mark.parametrize(
    "graph, vertex_order",
    [
        [Graph([("a", 1.8), ("a", "#"), ("#", 0), (0, 1.8)]), ["a", "#", 0, "s"]],
        [Graph([[1, 2], [2, 3]]), [1, 3]],
        [Graph([[1, 2], [2, 3]]), [1, 2, 2]],
        [Graph([[1, 2], [2, 3]]), [1, 2, 2, 3]],
        [Graph([[1, 2], [2, 3]]), [1, 2, 3, 4]],
        [graphs.Complete(3), [1, 2, 3]],
    ],
)
def test__input_check_vertex_order_error(graph, vertex_order):
    with pytest.raises(ValueError):
        graph._input_check_vertex_order(vertex_order)


@pytest.mark.parametrize(
    "graph, edge_order",
    [
        [
            Graph([("a", 1.8), ("a", "#"), ("#", 0), (0, 1.8)]),
            [(0, "#"), ("a", 1.8), (0, 1.8), ("#", "a")],
        ],
        [Graph([[1, 2], [2, 3]]), [[1, 2], [2, 3]]],
        [Graph([[1, 2], [2, 3]]), [[2, 1], [3, 2]]],
        [Graph([[1, 2], [2, 3]]), [[2, 3], [1, 2]]],
        [graphs.Complete(3), [[0, 1], [1, 2], [2, 0]]],
    ],
)
def test__input_check_edge_order(graph, edge_order):
    assert graph._input_check_edge_order(edge_order) == edge_order


@pytest.mark.parametrize(
    "graph, edge_order",
    [
        [
            Graph([("a", 1.8), ("a", "#"), ("#", 0), (0, 1.8)]),
            [("#", "#"), ("a", 1.8), (0, 1.8), ("#", "a")],
        ],
        [Graph([[1, 2], [2, 3]]), [[1, 2], [2, 4]]],
        [Graph([[1, 2], [2, 3]]), [[1, 2], [2, 3], [1, 3]]],
        [Graph([[1, 2], [2, 3]]), [[1, 2], [2, 3], [1, 2]]],
        [graphs.Complete(3), [[0, 1], [1, 2], [1, 2]]],
    ],
)
def test__input_check_edge_order_error(graph, edge_order):
    with pytest.raises(ValueError):
        graph._input_check_edge_order(edge_order)


def test_from_vertices_and_edges():
    G = Graph.from_vertices_and_edges([], [])
    assert G.vertex_list() == [] and G.edge_list() == []
    G = Graph.from_vertices_and_edges([0], [])
    assert G.vertex_list() == [0] and G.edge_list() == []
    G = Graph.from_vertices_and_edges([0, 1, 2, 3, 4, 5], [[0, 1]])
    assert G.vertex_list() == [0, 1, 2, 3, 4, 5] and G.edge_list() == [[0, 1]]
    G = Graph.from_vertices_and_edges([0, 1, 2], [[0, 1], [0, 2], [1, 2]])
    assert G.vertex_list() == [0, 1, 2] and G.edge_list() == [[0, 1], [0, 2], [1, 2]]
    G = Graph.from_vertices_and_edges(["a", "b", "c", "d"], [["a", "c"], ["a", "d"]])
    assert G.vertex_list() == ["a", "b", "c", "d"] and G.edge_list() == [
        ["a", "c"],
        ["a", "d"],
    ]
    with pytest.raises(ValueError):
        Graph.from_vertices_and_edges([1, 2, 3], [[1, 2], [2, 4]])


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [graphs.Complete(4), 2, 2],
        [Graph([[0, 0], [0, 1], [1, 1]]), 2, 1],
        [graphs.K66MinusPerfectMatching(), 3, 6],
        [graphs.DoubleBanana(), 3, 6],
    ],
)
def test_is_kl_tight(graph, K, L):
    assert graph.is_kl_tight(K, L)


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [graphs.CompleteBipartite(4, 4), 3, 6],
    ],
)
def test_is_not_kl_tight(graph, K, L):
    assert not graph.is_kl_tight(K, L)


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [Graph([[0, 0], [0, 1], [1, 1]]), 2, 1],
        [graphs.K66MinusPerfectMatching(), 3, 6],
        [graphs.DoubleBanana(), 3, 6],
    ],
)
def test_is_kl_sparse(graph, K, L):
    assert graph.is_kl_sparse(K, L)


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [Graph([[0, 1], [1, 1]]), 1, 1],
        [graphs.DoubleBanana() + Graph([[0, 1]]), 3, 6],
    ],
)
def test_is_not_kl_sparse(graph, K, L):
    assert not graph.is_kl_sparse(K, L)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(4),
        graphs.CompleteBipartite(3, 4),
        graphs.CompleteBipartite(2, 4),
        graphs.ThreePrism(),
        graphs.ThreePrismPlusEdge(),
        graphs.Cycle(5),
        graphs.Path(4),
    ],
)
def test_spanning_kl_sparse_subgraph(graph):
    for K in range(1, 3):
        for L in range(0, 2 * K):
            spanning_subgraph = graph.spanning_kl_sparse_subgraph(K, L)
            assert spanning_subgraph.is_kl_sparse(K, L, algorithm="subgraph")
            assert set(graph.vertex_list()) == set(spanning_subgraph.vertex_list())


def test_plot():
    G = graphs.DoubleBanana()
    G.plot(layout="random")
    plt.close("all")


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 2],
        [graphs.CompleteBipartite(3, 3), 16],
        [graphs.Diamond(), 4],
        [graphs.ThreePrism(), 24],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_count_reflection(graph, num_of_realizations):
    assert graph.number_of_realizations(count_reflection=True) == num_of_realizations


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 1],
        [graphs.CompleteBipartite(3, 3), 8],
        [graphs.Diamond(), 2],
        [graphs.ThreePrism(), 12],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations(graph, num_of_realizations):
    assert graph.number_of_realizations() == num_of_realizations


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 1],
        [graphs.CompleteBipartite(3, 3), 8],
        [graphs.Diamond(), 2],
        [graphs.ThreePrism(), 16],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_sphere(graph, num_of_realizations):
    assert graph.number_of_realizations(spherical=True) == num_of_realizations


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 2],
        [graphs.CompleteBipartite(3, 3), 16],
        [graphs.Diamond(), 4],
        [graphs.ThreePrism(), 32],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_sphere_count_reflection(graph, num_of_realizations):
    assert (
        graph.number_of_realizations(spherical=True, count_reflection=True)
        == num_of_realizations
    )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(4),
        graphs.K33plusEdge(),
        graphs.ThreePrismPlusEdge(),
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.Path(3),
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_error(graph):
    with pytest.raises(ValueError):
        graph.number_of_realizations()


@pytest.mark.parametrize(
    "graph",
    [graphs.Cycle(n) for n in range(3, 7)]
    + [Graph.from_vertices_and_edges([0, 1, 2, 3], [[0, 1], [0, 2], [1, 2]])],
)
def test_is_Rd_circuit_d1(graph):
    assert graph.is_Rd_circuit(dim=1)


@pytest.mark.parametrize(
    "graph",
    [
        Graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]),
        Graph([(0, 1), (2, 3)]),
        graphs.Complete(2),
        graphs.Diamond(),
        graphs.K33plusEdge(),
        graphs.ThreePrism(),
        graphs.ThreePrismPlusEdge(),
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.Path(3),
        graphs.K66MinusPerfectMatching(),
    ],
)
def test_is_not_Rd_circuit_d1(graph):
    assert not graph.is_Rd_circuit(dim=1)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(4),
        graphs.ThreePrismPlusEdge(),
        graphs.K33plusEdge(),
        Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (3, 0), (3, 1), (2, 4)]),
    ]
    + [graphs.Wheel(n) for n in range(3, 7)],
)
def test_is_Rd_circuit_d2(graph):
    assert graph.is_Rd_circuit(dim=2)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Complete(5),
        graphs.Diamond(),
        graphs.ThreePrism(),
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.Path(3),
        graphs.Cycle(4),
        graphs.K66MinusPerfectMatching(),
        Graph(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 0),
                (0, 3),
                (0, 2),
                (1, 3),
                (3, 5),
            ]
        ),
        graphs.Complete(4) + Graph([(3, 4), (4, 5), (5, 6), (6, 3), (3, 5), (4, 6)]),
    ],
)
def test_is_not_Rd_circuit_d2(graph):
    assert not graph.is_Rd_circuit(dim=2, algorithm="default")
    assert not graph.is_Rd_circuit(dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [graphs.Complete(5), graphs.ThreeConnectedR3Circuit(), graphs.DoubleBanana()],
)
def test_is_Rd_circuit_d3(graph):
    assert graph.is_Rd_circuit(dim=3)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Path(5),
        graphs.Complete(4),
        graphs.Cycle(6),
        graphs.ThreePrism(),
        graphs.K33plusEdge(),
    ],
)
def test_is_not_Rd_circuit_d3(graph):
    assert not graph.is_Rd_circuit(dim=3)


@pytest.mark.parametrize(
    "graph, dim",
    [
        [Graph([(0, 1), (2, 3)]), 1],
        [Graph([(0, 1), (1, 2), (0, 2), (3, 4)]), 1],
        [graphs.Complete(4), 2],
        [graphs.Cycle(4), 2],
        [Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 4), (2, 5)]), 2],
    ],
)
def test_is_Rd_closed(graph, dim):
    if dim <= 1:
        assert graph.is_Rd_closed(dim=dim, algorithm="graphic")
        assert graph.is_Rd_closed(dim=dim, algorithm="randomized")
    else:
        assert graph.is_Rd_closed(dim=dim, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.Path(4), 1],
        [graphs.ThreePrism(), 2],
        [graphs.ThreePrismPlusEdge(), 2],
        pytest.param(graphs.K66MinusPerfectMatching(), 2, marks=pytest.mark.slow_main),
        [graphs.Octahedral(), 3],
        [graphs.DoubleBanana(), 3],
    ],
)
def test_is_not_Rd_closed(graph, dim):
    if dim <= 1:
        assert not graph.is_Rd_closed(dim=dim, algorithm="graphic")
        assert not graph.is_Rd_closed(dim=dim, algorithm="randomized")
    else:
        assert not graph.is_Rd_closed(dim=dim, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Diamond(),
        graphs.K33plusEdge(),
        graphs.ThreePrism(),
        graphs.ThreePrismPlusEdge(),
        graphs.CompleteBipartite(2, 3),
        graphs.K66MinusPerfectMatching(),
    ]
    + [graphs.Cycle(n) for n in range(3, 7)],
)
def test_is_Rd_dependent_d1(graph):
    assert graph.is_Rd_dependent(dim=1)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.CompleteBipartite(1, 3),
        graphs.Path(3),
    ],
)
def test_is_Rd_independent_d1(graph):
    assert graph.is_Rd_independent(dim=1)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(4),
        graphs.ThreePrismPlusEdge(),
        graphs.K33plusEdge(),
        graphs.Complete(5),
        graphs.CompleteBipartite(3, 4),
        graphs.K66MinusPerfectMatching(),
    ]
    + [graphs.Wheel(n) for n in range(3, 8)],
)
def test_is_Rd_dependent_d2(graph):
    assert graph.is_Rd_dependent(dim=2)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Diamond(),
        graphs.ThreePrism(),
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.CompleteBipartite(3, 3),
        graphs.Path(3),
        graphs.Cycle(4),
    ],
)
def test_is_Rd_independent_d2(graph):
    assert graph.is_Rd_independent(dim=2)


@pytest.mark.parametrize(
    "graph",
    [graphs.Complete(5), graphs.ThreeConnectedR3Circuit(), graphs.DoubleBanana()],
)
def test_is_Rd_dependent_d3(graph):
    assert graph.is_Rd_dependent(dim=3)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(4),
        graphs.Cycle(6),
        graphs.ThreePrism(),
        graphs.K33plusEdge(),
        graphs.K66MinusPerfectMatching(),
        graphs.Path(5),
    ],
)
def test_is_Rd_independent_d3(graph):
    assert graph.is_Rd_independent(dim=3)


def test_is_Rd_independent_d3_warning():
    G = graphs.K33plusEdge()
    with pytest.warns(RandomizedAlgorithmWarning):
        G.is_Rd_independent(dim=3)


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Cycle(4), 1],
        [graphs.Diamond(), 2],
        [graphs.Complete(4), math.inf],
        [Graph([(0, 1), (2, 3)]), 0],
        [graphs.Complete(5), math.inf],
        [graphs.Frustum(3), 2],
        [graphs.ThreePrism(), 2],
        [graphs.DoubleBanana(), 2],
        [graphs.CompleteMinusOne(5), 3],
        [graphs.Octahedral(), 3],
        [graphs.K66MinusPerfectMatching(), 3],
    ],
)
def test_max_rigid_dimension(graph, k):
    assert graph.max_rigid_dimension() == k


def test_max_rigid_dimension_warning():
    G = graphs.K66MinusPerfectMatching()
    with pytest.warns(RandomizedAlgorithmWarning):
        G.max_rigid_dimension()


def test_cone():
    G = graphs.Complete(5).cone()
    assert set(G.nodes) == set([0, 1, 2, 3, 4, 5]) and len(G.nodes) == 6
    G = graphs.Complete(4).cone(vertex="a")
    assert "a" in G.nodes
    G = graphs.Cycle(4).cone()
    assert G.number_of_nodes() == G.max_degree() + 1


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Cycle(4), 0],
        [graphs.Diamond(), 0],
        [graphs.Complete(4), 0],
        [Graph([(0, 1), (2, 3)]), 0],
        [graphs.Complete(5), 1],
        [graphs.Complete(6), 2],
        [graphs.Frustum(3), 0],
        [graphs.ThreePrism(), 0],
        [graphs.DoubleBanana(), 1],
        [graphs.Octahedral(), 0],
        [Graph.from_int(8191), 1],
    ]
    + [[graphs.Wheel(n).cone(), 1] for n in range(3, 8)],
)
def test_is_k_vertex_apex(graph, k):
    assert graph.is_k_vertex_apex(k)


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Complete(5), 0],
        [graphs.Complete(6), 1],
        [graphs.DoubleBanana(), 0],
    ]
    + [[graphs.Wheel(n).cone(), 0] for n in range(3, 8)],
)
def test_is_not_k_vertex_apex(graph, k):
    assert not graph.is_k_vertex_apex(k)


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Cycle(4), 0],
        [graphs.Diamond(), 0],
        [graphs.Complete(4), 0],
        [Graph([(0, 1), (2, 3)]), 0],
        [graphs.Complete(5), 1],
        [graphs.Complete(6), 3],
        [graphs.Frustum(3), 0],
        [graphs.ThreePrism(), 0],
        [graphs.DoubleBanana(), 2],
        [graphs.Octahedral(), 0],
        [Graph.from_int(16351), 1],
    ]
    + [[graphs.Wheel(n).cone(), 1] for n in range(3, 8)],
)
def test_is_k_edge_apex(graph, k):
    assert graph.is_k_edge_apex(k)


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Complete(5), 0],
        [graphs.Complete(6), 2],
        [graphs.DoubleBanana(), 1],
        [graphs.K66MinusPerfectMatching(), 0],
    ]
    + [[graphs.Wheel(n).cone(), 0] for n in range(3, 8)],
)
def test_is_not_k_edge_apex(graph, k):
    assert not graph.is_k_edge_apex(k)


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Cycle(4), 0],
        [graphs.Diamond(), 0],
        [graphs.Complete(4), 0],
        [Graph([(0, 1), (2, 3)]), 0],
        [graphs.Complete(5), 1],
        [graphs.Complete(6), 2],
        [graphs.Frustum(3), 0],
        [graphs.ThreePrism(), 0],
        [graphs.DoubleBanana(), 3],
        [graphs.Octahedral(), 0],
        [Graph.from_int(8191), 2],
    ]
    + [[graphs.Wheel(n).cone(), 1] for n in range(3, 8)],
)
def test_is_critically_k_vertex_apex(graph, k):
    assert graph.is_critically_k_vertex_apex(k)


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Complete(5), 0],
        [graphs.Complete(6), 1],
        [graphs.DoubleBanana(), 2],
        [graphs.K66MinusPerfectMatching(), 0],
        [Graph.from_int(8191), 1],
    ]
    + [[graphs.Wheel(n).cone(), 0] for n in range(3, 8)],
)
def test_is_not_critically_k_vertex_apex(graph, k):
    assert not graph.is_critically_k_vertex_apex(k)


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Cycle(4), 0],
        [graphs.Diamond(), 0],
        [graphs.Complete(4), 0],
        [Graph([(0, 1), (2, 3)]), 0],
        [graphs.Complete(5), 1],
        [graphs.Complete(6), 7],
        [graphs.Frustum(3), 0],
        [graphs.ThreePrism(), 0],
        pytest.param(graphs.DoubleBanana(), 8, marks=pytest.mark.slow_main),
        [graphs.Octahedral(), 0],
        [Graph.from_int(112468), 1],
        [Graph.from_int(481867), 2],
        pytest.param(graphs.Wheel(5).cone(), 7, marks=pytest.mark.slow_main),
    ]
    + [[graphs.Wheel(n).cone(), 1 if n == 3 else 2 * n - 3] for n in range(3, 5)],
)
def test_is_critically_k_edge_apex(graph, k):
    assert graph.is_critically_k_edge_apex(k)


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Complete(5), 0],
        [graphs.Complete(6), 6],
        [graphs.DoubleBanana(), 7],
        [Graph.from_int(481867), 1],
        [Graph.from_int(16351), 1],
    ]
    + [[graphs.Wheel(n).cone(), 0 if n == 3 else 2 * n - 4] for n in range(3, 6)],
)
def test_is_not_critically_k_edge_apex(graph, k):
    assert not graph.is_critically_k_edge_apex(k)


@pytest.mark.long_local
def test_randomized_apex_properties():  # noqa: C901
    search_space = [range(1, 8), range(10)]
    for n, _ in product(*search_space):
        for m in range(3, math.comb(n, 2) + 1):
            G = Graph(nx.gnm_random_graph(n, m))
            prop_apex = G.is_k_edge_apex(1)
            prop_2_apex = G.is_k_edge_apex(2)
            prop_3_apex = G.is_k_edge_apex(3)
            prop_vapex = G.is_k_vertex_apex(1)
            prop_2_vapex = G.is_k_vertex_apex(2)
            prop_3_vapex = G.is_k_vertex_apex(3)
            prop_crit_apex = G.is_critically_k_edge_apex(1)
            prop_crit_2_apex = G.is_critically_k_edge_apex(2)
            prop_crit_3_apex = G.is_critically_k_edge_apex(3)
            prop_crit_vapex = G.is_critically_k_vertex_apex(1)
            prop_crit_2_vapex = G.is_critically_k_vertex_apex(2)
            prop_crit_3_vapex = G.is_critically_k_vertex_apex(3)

            if prop_apex:
                assert prop_vapex
                assert prop_2_apex
                assert prop_3_apex
            if prop_2_apex:
                assert prop_2_vapex
                assert prop_3_apex
            if prop_3_apex:
                assert prop_3_vapex
            if prop_vapex:
                assert prop_2_vapex
                assert prop_3_vapex
            if prop_2_vapex:
                assert prop_3_vapex

            if prop_crit_apex:
                assert prop_apex
            if prop_crit_2_apex:
                assert prop_2_apex
            if prop_crit_3_apex:
                assert prop_3_apex
            if prop_crit_vapex:
                assert prop_vapex
            if prop_crit_2_vapex:
                assert prop_2_vapex
            if prop_crit_3_vapex:
                assert prop_3_vapex


@pytest.mark.long_local
def test_randomized_rigidity_properties():  # noqa: C901
    search_space = [range(1, 4), range(1, 7), range(10)]
    for dim, n, _ in product(*search_space):
        for m in range(1, math.comb(n, 2) + 1):
            G = Graph(nx.gnm_random_graph(n, m))
            assert G.number_of_nodes() == n
            assert G.number_of_edges() == m

            prop_rigid = G.is_rigid(dim)
            prop_min_rigid = G.is_min_rigid(dim)
            prop_glob_rigid = G.is_globally_rigid(dim)
            prop_red_rigid = G.is_redundantly_rigid(dim)
            prop_2_red_rigid = G.is_k_redundantly_rigid(2, dim)
            prop_3_red_rigid = G.is_k_redundantly_rigid(3, dim)
            prop_vred_rigid = G.is_vertex_redundantly_rigid(dim)
            prop_2_vred_rigid = G.is_k_vertex_redundantly_rigid(2, dim)
            prop_3_vred_rigid = G.is_k_vertex_redundantly_rigid(3, dim)
            prop_min_red_rigid = G.is_min_redundantly_rigid(dim)
            prop_min_2_red_rigid = G.is_min_k_redundantly_rigid(2, dim)
            prop_min_3_red_rigid = G.is_min_k_redundantly_rigid(3, dim)
            prop_min_vred_rigid = G.is_min_vertex_redundantly_rigid(dim)
            prop_min_2_vred_rigid = G.is_min_k_vertex_redundantly_rigid(2, dim)
            prop_min_3_vred_rigid = G.is_min_k_vertex_redundantly_rigid(3, dim)
            prop_sparse = G.is_kl_sparse(dim, math.comb(dim + 1, 2))
            prop_tight = G.is_kl_tight(dim, math.comb(dim + 1, 2))
            prop_seq = G.has_extension_sequence(dim)
            prop_dep = G.is_Rd_dependent(dim)
            prop_indep = G.is_Rd_independent(dim)
            prop_circ = G.is_Rd_circuit(dim)

            # randomized algorithm
            rprop_rigid = G.is_rigid(dim, algorithm="randomized")
            rprop_min_rigid = G.is_min_rigid(dim, algorithm="randomized")
            rprop_glob_rigid = G.is_globally_rigid(dim, algorithm="randomized")
            rprop_red_rigid = G.is_redundantly_rigid(dim, algorithm="randomized")
            rprop_2_red_rigid = G.is_k_redundantly_rigid(2, dim, algorithm="randomized")
            rprop_3_red_rigid = G.is_k_redundantly_rigid(3, dim, algorithm="randomized")
            rprop_vred_rigid = G.is_vertex_redundantly_rigid(
                dim, algorithm="randomized"
            )
            rprop_2_vred_rigid = G.is_k_vertex_redundantly_rigid(
                2, dim, algorithm="randomized"
            )
            rprop_3_vred_rigid = G.is_k_vertex_redundantly_rigid(
                3, dim, algorithm="randomized"
            )
            rprop_min_red_rigid = G.is_min_redundantly_rigid(
                dim, algorithm="randomized"
            )
            rprop_min_2_red_rigid = G.is_min_k_redundantly_rigid(
                2, dim, algorithm="randomized"
            )
            rprop_min_3_red_rigid = G.is_min_k_redundantly_rigid(
                3, dim, algorithm="randomized"
            )
            rprop_min_vred_rigid = G.is_min_vertex_redundantly_rigid(
                dim, algorithm="randomized"
            )
            rprop_min_2_vred_rigid = G.is_min_k_vertex_redundantly_rigid(
                2, dim, algorithm="randomized"
            )
            rprop_min_3_vred_rigid = G.is_min_k_vertex_redundantly_rigid(
                3, dim, algorithm="randomized"
            )
            rprop_dep = G.is_Rd_dependent(dim, algorithm="randomized")
            rprop_indep = G.is_Rd_independent(dim, algorithm="randomized")
            rprop_circ = G.is_Rd_circuit(dim, algorithm="randomized")

            # subgraph algorithm
            sprop_sparse = G.is_kl_sparse(
                dim, math.comb(dim + 1, 2), algorithm="subgraph"
            )
            sprop_tight = G.is_kl_tight(
                dim, math.comb(dim + 1, 2), algorithm="subgraph"
            )

            # cones
            res_cone = G.cone()
            cprop_rigid = res_cone.is_rigid(dim + 1)
            cprop_min_rigid = res_cone.is_min_rigid(dim + 1)
            cprop_glob_rigid = res_cone.is_globally_rigid(dim + 1)

            # extensions
            if n > dim:
                res_ext0 = G.all_k_extensions(0, dim)
            else:
                res_ext0 = []
            if m > 1 and n > dim + 1:
                res_ext1 = G.all_k_extensions(1, dim)
            else:
                res_ext1 = []

            # framework
            F = G.random_framework(dim)
            fprop_inf_rigid = F.is_inf_rigid()
            fprop_inf_flex = F.is_inf_flexible()
            fprop_min_inf_rigid = F.is_min_inf_rigid()
            fprop_red_rigid = F.is_redundantly_inf_rigid()
            fprop_dep = F.is_dependent()
            fprop_indep = F.is_independent()

            # (min) rigidity
            if prop_min_rigid:
                assert rprop_min_rigid
                assert cprop_min_rigid
                assert prop_rigid
                assert fprop_min_inf_rigid
                assert prop_indep
                if n > dim:
                    assert m == n * dim - math.comb(dim + 1, 2)
                    assert F.rigidity_matrix_rank() == n * dim - math.comb(dim + 1, 2)
                    assert G.min_degree() >= dim
                    assert G.min_degree() <= 2 * dim - 1
                    assert prop_sparse
                    assert prop_tight
                    assert prop_seq
                else:
                    assert m == math.comb(n, 2)
                for graph in res_ext0:
                    assert graph.is_min_rigid(dim)
                for graph in res_ext1:
                    assert graph.is_min_rigid(dim)
            if rprop_min_rigid:
                assert prop_min_rigid
            if prop_rigid:
                assert rprop_rigid
                assert cprop_rigid
                assert fprop_inf_rigid
                if n > dim:
                    assert m >= n * dim - math.comb(dim + 1, 2)
                    assert F.rigidity_matrix_rank() == n * dim - math.comb(dim + 1, 2)
                    assert G.min_degree() >= dim
                    if m > n * dim - math.comb(dim + 1, 2):
                        assert prop_dep
                    else:
                        assert prop_indep
                else:
                    assert m == math.comb(n, 2)
                    assert prop_indep
                if prop_circ:
                    assert m == n * dim - math.comb(dim + 1, 2) + 1
            if rprop_rigid:
                assert prop_rigid

            # sparsity
            if prop_sparse:
                assert sprop_sparse
            if sprop_sparse:
                assert prop_sparse
            if prop_tight:
                assert sprop_tight
                if dim == 2 or dim == 1:
                    assert prop_min_rigid
            if sprop_tight:
                assert prop_tight

            # redundancy
            if prop_red_rigid:
                assert rprop_red_rigid
                assert prop_rigid
                assert fprop_red_rigid
                assert m >= n * dim - math.comb(dim + 1, 2) + 1
                if G.number_of_nodes() >= dim + 1 + 1:
                    assert G.min_degree() >= dim + 1  # thm-vertex-red-min-deg
            if rprop_red_rigid:
                assert prop_red_rigid
            if prop_2_red_rigid:
                assert rprop_2_red_rigid
                assert prop_rigid
                assert prop_red_rigid
                assert m >= n * dim - math.comb(dim + 1, 2) + 2
                if G.number_of_nodes() >= dim + 2 + 1:
                    assert G.min_degree() >= dim + 2  # thm-vertex-red-min-deg
            if rprop_2_red_rigid:
                assert prop_2_red_rigid
            if prop_3_red_rigid:
                assert rprop_3_red_rigid
                assert prop_rigid
                assert prop_2_red_rigid
                assert prop_red_rigid
                assert m >= n * dim - math.comb(dim + 1, 2) + 2
                if G.number_of_nodes() >= dim + 3 + 1:
                    assert G.min_degree() >= dim + 3  # thm-vertex-red-min-deg
            if rprop_3_red_rigid:
                assert prop_3_red_rigid
            if prop_vred_rigid:
                assert rprop_vred_rigid
                assert prop_rigid
                if G.number_of_nodes() >= dim + 1 + 1:
                    assert prop_red_rigid  # thm-vertex-implies_edge
                    assert G.min_degree() >= dim + 1  # thm-vertex-red-min-deg
            if rprop_vred_rigid:
                assert prop_vred_rigid
            if prop_2_vred_rigid:
                assert rprop_2_vred_rigid
                assert prop_rigid
                assert prop_vred_rigid
                if G.number_of_nodes() >= dim + 2 + 1:
                    assert prop_2_red_rigid  # thm-vertex-implies_edge
                    assert G.min_degree() >= dim + 2  # thm-vertex-red-min-deg
            if rprop_2_vred_rigid:
                assert prop_2_vred_rigid
            if prop_3_vred_rigid:
                assert rprop_3_vred_rigid
                assert prop_rigid
                assert prop_2_vred_rigid
                assert prop_vred_rigid
                if G.number_of_nodes() >= dim + 3 + 1:
                    assert prop_3_red_rigid  # thm-vertex-implies_edge
                    assert G.min_degree() >= dim + 3  # thm-vertex-red-min-deg
            if rprop_3_vred_rigid:
                assert prop_3_vred_rigid

            # minimal redundancy
            if prop_min_red_rigid:
                assert rprop_min_red_rigid
                assert prop_rigid
                assert prop_red_rigid
                assert m >= n * dim - math.comb(dim + 1, 2) + 1
                if G.number_of_nodes() >= dim + 1 + 1:
                    assert G.min_degree() >= dim + 1  # thm-vertex-red-min-deg
            if rprop_min_red_rigid:
                assert prop_min_red_rigid
            if prop_min_2_red_rigid:
                assert rprop_min_2_red_rigid
                assert prop_rigid
                assert prop_red_rigid
                assert prop_2_red_rigid
                assert m >= n * dim - math.comb(dim + 1, 2) + 2
                if G.number_of_nodes() >= dim + 2 + 1:
                    assert G.min_degree() >= dim + 2  # thm-vertex-red-min-deg
            if rprop_min_2_red_rigid:
                assert prop_min_2_red_rigid
            if prop_min_3_red_rigid:
                assert rprop_min_3_red_rigid
                assert prop_rigid
                assert prop_2_red_rigid
                assert prop_red_rigid
                assert prop_3_red_rigid
                assert m >= n * dim - math.comb(dim + 1, 2) + 3
                if G.number_of_nodes() >= dim + 3 + 1:
                    assert G.min_degree() >= dim + 3  # thm-vertex-red-min-deg
            if rprop_min_3_red_rigid:
                assert prop_min_3_red_rigid
            if prop_min_vred_rigid:
                assert rprop_min_vred_rigid
                assert prop_rigid
                assert prop_vred_rigid
                if G.number_of_nodes() >= dim + 1 + 1:
                    assert prop_red_rigid  # thm-vertex-implies_edge
                    assert G.min_degree() >= dim + 1  # thm-vertex-red-min-deg
            if rprop_min_vred_rigid:
                assert prop_min_vred_rigid
            if prop_min_2_vred_rigid:
                assert rprop_min_2_vred_rigid
                assert prop_rigid
                assert prop_vred_rigid
                assert prop_2_vred_rigid
                if G.number_of_nodes() >= dim + 2 + 1:
                    assert prop_2_red_rigid  # thm-vertex-implies_edge
                    assert G.min_degree() >= dim + 2  # thm-vertex-red-min-deg
            if rprop_min_2_vred_rigid:
                assert prop_min_2_vred_rigid
            if prop_min_3_vred_rigid:
                assert rprop_min_3_vred_rigid
                assert prop_rigid
                assert prop_2_vred_rigid
                assert prop_vred_rigid
                assert prop_3_vred_rigid
                if G.number_of_nodes() >= dim + 3 + 1:
                    assert prop_3_red_rigid  # thm-vertex-implies_edge
                    assert G.min_degree() >= dim + 3  # thm-vertex-red-min-deg
            if rprop_min_3_vred_rigid:
                assert prop_min_3_vred_rigid

            # global rigidity
            if prop_glob_rigid:
                assert rprop_glob_rigid
                assert prop_rigid
                assert cprop_glob_rigid
                if n > dim + 1:
                    assert m >= n * dim - math.comb(dim + 1, 2)
                    assert prop_red_rigid
                    assert G.vertex_connectivity() >= dim + 1
                else:
                    assert m == math.comb(n, 2)
                if prop_min_rigid:
                    assert m == math.comb(n, 2)
            if rprop_glob_rigid:
                assert prop_glob_rigid

            # cones
            if cprop_min_rigid:
                assert prop_min_rigid
            if cprop_rigid:
                assert prop_rigid
            if cprop_glob_rigid:
                assert prop_glob_rigid

            if not prop_rigid:
                assert not prop_min_rigid
                assert not prop_glob_rigid
                assert not prop_red_rigid
                assert not prop_2_red_rigid
                assert not prop_3_red_rigid
                assert not prop_vred_rigid
                assert not prop_2_vred_rigid
                assert not prop_3_vred_rigid
                assert not prop_min_red_rigid
                assert not prop_min_2_red_rigid
                assert not prop_min_3_red_rigid
                assert not prop_min_vred_rigid
                assert not prop_min_2_vred_rigid
                assert not prop_min_3_vred_rigid

            # dependence
            if prop_circ:
                assert rprop_circ
                assert prop_dep
                assert not prop_indep
            if rprop_circ:
                assert prop_circ
            if prop_indep:
                assert rprop_indep
                assert not prop_circ
                assert not prop_dep
                assert fprop_indep
                if n > dim:
                    assert m <= n * dim - math.comb(dim + 1, 2)
            if rprop_indep:
                assert prop_indep
            if prop_dep:
                assert rprop_dep
                assert fprop_dep
            if rprop_dep:
                assert prop_dep

            # closure
            res_close = Graph(G.Rd_closure())
            assert res_close.is_Rd_closed()
            res_close = Graph(G.Rd_closure(), algorithm="randomized")
            assert res_close.is_Rd_closed()

            # frameworks
            if fprop_inf_rigid:
                assert prop_rigid
                assert not fprop_inf_flex
            if fprop_min_inf_rigid:
                assert prop_min_rigid
            if fprop_red_rigid:
                assert prop_red_rigid
            if fprop_indep:
                assert prop_indep
            if fprop_dep:
                assert prop_dep
            if fprop_inf_flex:
                assert not fprop_inf_rigid


@pytest.mark.long_local
def test_randomized_sparsity_properties():  # noqa: C901
    search_space = [range(1, 8), range(10)]
    for n, _ in product(*search_space):
        for m in range(1, math.comb(n, 2) + 1):
            G = Graph(nx.gnm_random_graph(n, m))
            assert G.number_of_nodes() == n
            assert G.number_of_edges() == m

            prop_sparse = {k: [G.is_kl_sparse(k,ell) for ell in range(math.comb(k+1,2)+2)] for k in range(1,5)}
            prop_tight = {k: [G.is_kl_tight(k,ell) for ell in range(math.comb(k+1,2)+2)] for k in range(1,5)}

            for k in range(1,4):
                for ell in range(math.comb(k+1,2)+1):
                    if prop_tight[k][ell]:
                        assert prop_sparse[k][ell]
                        if n>=k:
                            assert m == k*n - ell
                    if prop_sparse[k][ell]:
                        if n >= k:
                            assert m <= k*n - ell
                        for ell2 in range(ell):
                            assert prop_sparse[k][ell2]

            if prop_sparse[1][1]:
                if nx.is_connected(G):
                    assert nx.is_tree(G)

