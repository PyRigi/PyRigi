from pyrigi.graph import Graph
import pyrigi.graphDB as graphs
from pyrigi.exception import (
    LoopError,
    NotSupportedValueError,
)
import matplotlib.pyplot as plt

import pytest
from sympy import Matrix
import math
import networkx as nx
from random import randint
import pyrigi.misc as misc


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
    assert graph.is_rigid(dim=2, algorithm="combinatorial")
    assert graph.is_rigid(dim=2, algorithm="randomized")


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
    assert not graph.is_rigid(dim=2, algorithm="combinatorial")
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
    ],
)
def test_is_rigid_d1(graph):
    assert graph.is_rigid(dim=1, algorithm="combinatorial")
    assert graph.is_rigid(dim=1, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [Graph.from_vertices(range(3)), Graph([[0, 1], [2, 3]])],
)
def test_is_not_rigid_d1(graph):
    assert not graph.is_rigid(dim=1, algorithm="combinatorial")
    assert not graph.is_rigid(dim=1, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, dim",
    [[graphs.K66MinusPerfectMatching(), 3]]
    + [[graphs.Complete(n), d] for d in range(1, 5) for n in range(1, d + 2)],
)
def test_is_rigid(graph, dim):
    assert graph.is_rigid(dim, algorithm="combinatorial" if (dim < 3) else "randomized")


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
    assert graph.is_sparse(2, 3, algorithm="subgraph")
    assert graph.is_sparse(2, 3, algorithm="pebble")


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
    assert not graph.is_sparse(2, 3, algorithm="subgraph")
    assert not graph.is_sparse(2, 3, algorithm="pebble")


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
    assert graph.is_tight(2, 3, algorithm="pebble")
    assert graph.is_tight(2, 3, algorithm="subgraph")


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
    assert not graph.is_tight(2, 3, algorithm="subgraph")
    assert not graph.is_tight(2, 3, algorithm="pebble")


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
    assert graph.is_min_rigid(dim=1, algorithm="combinatorial")
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
    assert not graph.is_min_rigid(dim=1, algorithm="combinatorial")
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
    assert graph.is_min_rigid(dim=2, algorithm="combinatorial")
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
    ],
)
def test_is_not_min_rigid_d2(graph):
    assert not graph.is_min_rigid(dim=2, algorithm="combinatorial")
    assert not graph.is_min_rigid(dim=2, algorithm="extension_sequence")
    assert not graph.is_min_rigid(dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(3),
        graphs.Complete(4),
        graphs.Octahedral(),
        graphs.K66MinusPerfectMatching(),
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
        pytest.param(graphs.ThreeConnectedR3Circuit(), marks=pytest.mark.slow_main),
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
        [read_globally("D19V21"), 19],
        pytest.param(read_globally("D19V22"), 19, marks=pytest.mark.slow_main),
        pytest.param(read_globally("D19V23"), 19, marks=pytest.mark.slow_main),
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
        pytest.param(Graph.from_int(1048575), 3, marks=pytest.mark.slow_main),
    ],
)
def test_is_min_k_vertex_redundantly_rigid_d2(graph, k):
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=2)
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=2, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(507903), 1],
        pytest.param(Graph.from_int(1048575), 2, marks=pytest.mark.slow_main),
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
        graphs.Complete(7),
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
        pytest.param(graphs.Octahedral(), 2, marks=pytest.mark.slow_main),
        pytest.param(graphs.Complete(6), 2, marks=pytest.mark.slow_main),
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
        pytest.param(Graph.from_int(507851), 2, marks=pytest.mark.slow_main),
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
        pytest.param(Graph.from_int(32767), 2, marks=pytest.mark.slow_main),
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
        pytest.param(Graph.from_int(16351), 2, marks=pytest.mark.slow_main),
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


def test_rigid_components():
    G = graphs.Path(6)
    rigid_components = G.rigid_components(dim=1)
    assert rigid_components[0] == [0, 1, 2, 3, 4, 5]
    G.remove_edge(2, 3)
    rigid_components = G.rigid_components(dim=1)
    assert {frozenset(H) for H in rigid_components} == {
        frozenset([0, 1, 2]),
        frozenset([3, 4, 5]),
    }

    G = graphs.Path(5)
    rigid_components = G.rigid_components(algorithm="randomized")
    assert sorted([sorted(H) for H in rigid_components]) == [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
    ]

    G = Graph(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (0, 3),
            (1, 4),
            (2, 5),
            (0, "a"),
            (0, "b"),
            ("a", "b"),
        ]
    )
    rigid_components = G.rigid_components(algorithm="randomized")
    assert {frozenset(H) for H in rigid_components} == {
        frozenset([0, "a", "b"]),
        frozenset([0, 1, 2, 3, 4, 5]),
    }

    G = Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    rigid_components = G.rigid_components(algorithm="randomized")
    assert {frozenset(H) for H in rigid_components} == {
        frozenset([0, 1, 2]),
        frozenset([3, 4, 5]),
    }

    G = graphs.Complete(3)
    G.add_vertex(3)
    rigid_components = G.rigid_components(algorithm="randomized")
    assert {frozenset(H) for H in rigid_components} == {
        frozenset([0, 1, 2]),
        frozenset([3]),
    }

    G = graphs.ThreePrism()
    rigid_components = G.rigid_components(algorithm="randomized")
    assert len(rigid_components) == 1 and (rigid_components == [[0, 1, 2, 3, 4, 5]])

    G = graphs.ThreeConnectedR3Circuit()
    G.remove_node(0)
    rigid_components = G.rigid_components(algorithm="randomized")
    assert {frozenset(H) for H in rigid_components} == {
        frozenset([1, 2, 3, 4]),
        frozenset([1, 10, 11, 12]),
        frozenset([4, 5, 6, 7]),
        frozenset([7, 8, 9, 10]),
    }

    G = graphs.DoubleBanana()
    rigid_components = G.rigid_components(dim=3, algorithm="randomized")
    assert {frozenset(H) for H in rigid_components} == {
        frozenset([0, 1, 2, 3, 4]),
        frozenset([0, 1, 5, 6, 7]),
    }


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
def test_dimension_combinatorial_error(method, params):
    with pytest.raises(ValueError):
        G = graphs.DoubleBanana()
        func = getattr(G, method)
        func(*params, algorithm="combinatorial")


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
        [Graph.from_int(31), 0, 2, [223, 239, 254]],
        [Graph.from_int(63), 0, 3, [511]],
        [Graph.from_int(511), 0, 1, [1535, 8703]],
        [
            Graph.from_int(16350),
            2,
            3,
            [257911, 260603, 376807, 384943, 1497823, 1973983],
        ],
        [Graph.from_int(511), 2, 3, [4095, 7679, 7935, 8187]],
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
def test_has_extension_sequence_false(graph):
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
        [Graph.from_int(63), 3],
        [Graph.from_int(511), 3],
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


def test_CompleteOnVertices():
    assert Graph.CompleteOnVertices([0, 1, 2, 3, 4, 5]) == graphs.Complete(6)
    assert Graph.CompleteOnVertices(
        ["a", "b", "c", "d", "e", "f", "g", "h"]
    ).is_isomorphic(graphs.Complete(8))
    assert Graph.CompleteOnVertices(["vertex", 1, "vertex_1", 3, 4]).is_isomorphic(
        graphs.Complete(5)
    )
    assert Graph.CompleteOnVertices(["vertex", 1]).is_isomorphic(graphs.Complete(2))
    assert Graph.CompleteOnVertices(["vertex"]).is_isomorphic(graphs.Complete(1))
    assert Graph.CompleteOnVertices(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    ).is_isomorphic(graphs.Complete(20))


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
        [Graph.from_int(7), 0],
        [Graph.from_int(31), 3],
        [Graph.from_vertices([1]), [1]],
        [Graph.from_vertices([1, 2, 3]), [2, 3]],
        [Graph.from_vertices_and_edges([1, 2, 3], [[1, 2], [2, 3]]), [1, 3]],
        [Graph([[1, 2], [2, 3]]), [2, 2]],
        [Graph.from_int(7), [0, 1]],
        [Graph.from_int(31), [1, 3]],
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
        [Graph.from_int(7), "a"],
        [Graph.from_int(31), 10],
        [Graph.from_vertices([1]), [2]],
        [Graph.from_vertices([1, 2, 3]), [3, 4]],
        [Graph.from_vertices_and_edges([1, 2, 3], [[1, 2], [2, 3]]), [5, 6]],
        [Graph([[1, 2], [2, 3]]), [2, 2, 4]],
        [Graph.from_int(7), [0, 4]],
        [Graph.from_int(31), [1, 2, 12]],
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
        [Graph.from_int(7), [0, 1]],
        [Graph.from_int(31), [1, 2]],
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
        [Graph.from_int(7), [0, 1], [0, 1, 2, 3, 4]],
        [Graph.from_int(31), [1, 2], [1, 2, 3]],
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
        [Graph.from_int(7), [0, 4]],
        [Graph.from_int(31), [1, -2]],
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
        [Graph.from_int(7), [0, 1], [1, 2, 3, 4]],
        [Graph.from_int(31), [1, 2], [1, 3]],
        [Graph([["a", "b"], ["b", 3]]), ["a", "b"], ["a", "c"]],
        [Graph([["a", "b"], ["b", 3]]), (3, "b"), ["a", "b", 2]],
        [Graph([["a", "b"], ["b", 3]]), ["b", "a"], ["a"]],
        [Graph([[-1, -2], [-2, 3]]), [-2, -1], [-3, -2, 0, 1, 2, 3]],
        [Graph([[-1, -2], [-2, 3]]), [-1, -2], [-2, 3]],
        [Graph([[-1, -2], [-2, 3]]), [-2, 3], [3]],
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
        [Graph.from_int(7), [[0, 1]]],
        [Graph.from_int(31), [[1, 2], [2, 3]]],
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
        [Graph.from_int(7), [[0, 1]], [1, 2, 3]],
        [Graph.from_int(31), [[1, 2], [2, 3]], [1, 2, 3]],
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
        [Graph.from_int(7), [[0, 1]]],
        [Graph.from_int(31), [[1, 2]]],
        [Graph([["a", "b"], ["b", 3]]), [["a", "b"]]],
        [Graph([["a", "b"], ["b", 3]]), [(3, "b")]],
        [Graph([["a", "b"], ["b", 3]]), [["b", "a"]]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, -1]]],
        [Graph([[-1, -2], [-2, 3]]), [[-1, -2]]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, 3]]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [(1, 2), (3, 2)]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [[3, 2], [1, 2]]],
        [Graph([[1, 2], [2, 3]]), [[1, 2], (2, 3)]],
        [Graph.from_int(7), [[0, 1], [1, 2]]],
        [Graph.from_int(31), [[1, 2], [2, 3]]],
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
        [Graph.from_int(7), [[0, 4]]],
        [Graph.from_int(31), [[1, -2]]],
        [Graph([["a", "b"], ["b", 3]]), [["a", "c"]]],
        [Graph([["a", "b"], ["b", 3]]), [(3, "a")]],
        [Graph([["a", "b"], ["b", 3]]), [["3", "a"]]],
        [Graph([[-1, -2], [-2, 3]]), [[3, -1]]],
        [Graph([[-1, -2], [-2, 3]]), [[-1, 0]]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, -3]]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [(1, 2), (3, 3)]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [[3, 2], [1, 3]]],
        [Graph([[1, 2], [2, 3]]), [[1, 2], (2, 4)]],
        [Graph.from_int(7), [[0, 1], [1, -2]]],
        [Graph.from_int(31), [[1, 5], [2, 3]]],
        [Graph([["a", "b"], ["b", 3]]), [["a", "c"], ["b", 3]]],
        [Graph([["a", "b"], ["b", 3]]), [(3, "b"), ("a", "d")]],
        [Graph([["a", "b"], ["b", 3]]), [["b", "3"], (3, "b")]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, -1], [1, 3]]],
        [Graph([[-1, -2], [-2, 3]]), [[-1, 5], (-2, 3)]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, -3], [-1, -2]]],
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
        [Graph.from_int(7), [0, 1]],
        [Graph.from_int(31), (1, 2)],
        [Graph.from_int(31), [[[1, 2], [2, 3]]]],
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
        [Graph.from_int(7), [0, 1, 2]],
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
        [Graph.from_int(7), [1, 2, 3]],
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
        [Graph.from_int(7), [[0, 1], [1, 2], [2, 0]]],
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
        [Graph.from_int(7), [[0, 1], [1, 2], [1, 2]]],
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


def test_is_3_6_sparse():
    """The Double Banana is (3,6)-tight."""
    G = graphs.DoubleBanana()
    assert G.is_sparse(3, 6)
    G.add_edge(0, 1)
    assert not G.is_sparse(3, 6)
    G = graphs.K66MinusPerfectMatching()
    assert G.is_sparse(3, 6)


def test_is_tight():
    G = graphs.Complete(4)
    assert G.is_tight(2, 2)
    G = graphs.CompleteBipartite(4, 4)
    assert not G.is_tight(3, 6)
    G = graphs.K66MinusPerfectMatching()
    assert G.is_tight(3, 6)


def test_plot():
    G = graphs.DoubleBanana()
    G.plot(layout="random")
    plt.close("all")


@pytest.mark.parametrize(
    "graph, n",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 2],
        [graphs.CompleteBipartite(3, 3), 16],
        [graphs.Diamond(), 4],
        [graphs.ThreePrism(), 24],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_count_reflection(graph, n):
    assert graph.number_of_realizations(count_reflection=True) == n


@pytest.mark.parametrize(
    "graph, n",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 1],
        [graphs.CompleteBipartite(3, 3), 8],
        [graphs.Diamond(), 2],
        [graphs.ThreePrism(), 12],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations(graph, n):
    assert graph.number_of_realizations() == n


@pytest.mark.parametrize(
    "graph, n",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 1],
        [graphs.CompleteBipartite(3, 3), 8],
        [graphs.Diamond(), 2],
        [graphs.ThreePrism(), 16],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_sphere(graph, n):
    assert graph.number_of_realizations(spherical_realizations=True) == n


@pytest.mark.parametrize(
    "graph, n",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 2],
        [graphs.CompleteBipartite(3, 3), 16],
        [graphs.Diamond(), 4],
        [graphs.ThreePrism(), 32],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_sphere_count_reflection(graph, n):
    assert (
        graph.number_of_realizations(spherical_realizations=True, count_reflection=True)
        == n
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
    [graphs.Cycle(n) for n in range(3, 7)],
)
def test_is_Rd_circuit_d1(graph):
    assert graph.is_Rd_circuit(dim=1)


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
    ],
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
    ],
)
def test_is_not_Rd_circuit_d2(graph):
    assert not graph.is_Rd_circuit(dim=2)


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
        assert graph.is_Rd_closed(dim=dim, algorithm="combinatorial")
        assert graph.is_Rd_closed(dim=dim, algorithm="randomized")
    else:
        assert graph.is_Rd_closed(dim=dim, algorithm="randomized")


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.Path(4), 1],
        [graphs.ThreePrism(), 2],
        [graphs.ThreePrismPlusEdge(), 2],
        [graphs.K66MinusPerfectMatching(), 2],
        [graphs.Octahedral(), 3],
        [graphs.DoubleBanana(), 3],
    ],
)
def test_is_not_Rd_closed(graph, dim):
    if dim <= 1:
        assert not graph.is_Rd_closed(dim=dim, algorithm="combinatorial")
        assert not graph.is_Rd_closed(dim=dim, algorithm="randomized")
    else:
        assert not graph.is_Rd_closed(dim=dim, algorithm="randomized")


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
    ],
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
        graphs.Path(5),
        graphs.Complete(4),
        graphs.Cycle(6),
        graphs.ThreePrism(),
        graphs.K33plusEdge(),
        graphs.K66MinusPerfectMatching(),
    ],
)
def test_is_Rd_independent_d3(graph):
    assert graph.is_Rd_independent(dim=3)


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
