from pyrigi.graph import Graph
import pyrigi.graphDB as graphs
from pyrigi.exception import LoopError

import pytest
from sympy import Matrix
import math
import networkx as nx
from random import randint


def test_add():
    G = Graph([[0, 1], [1, 2], [2, 0]])
    H = Graph([[0, 1], [1, 3], [3, 0]])
    assert G + H == Graph([[0, 1], [1, 2], [2, 0], [1, 3], [3, 0]])
    G = Graph([[0, 1], [1, 2], [2, 0]])
    H = Graph([[3, 4], [4, 5], [5, 3]])
    assert G + H == Graph([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]])
    G = Graph.from_vertices_and_edges([0, 1, 2, 3], [[0, 1], [1, 2]])
    H = Graph.from_vertices_and_edges([0, 1, 2, 4], [[0, 1]])
    assert G + H == Graph.from_vertices_and_edges([0, 1, 2, 3, 4], [[0, 1], [1, 2]])


def test_KL_values_are_correct():
    assert Graph._pebble_values_are_correct(2, 3)
    assert Graph._pebble_values_are_correct(1, 1)
    assert Graph._pebble_values_are_correct(20, 20)
    assert Graph._pebble_values_are_correct(5, 1)
    assert Graph._pebble_values_are_correct(2, 0)
    assert Graph._pebble_values_are_correct(40, 79)


def test_KL_values_are_not_correct():
    assert not Graph._pebble_values_are_correct(2, 4)
    assert not Graph._pebble_values_are_correct(1, -1)
    assert not Graph._pebble_values_are_correct(0, 0)
    assert not Graph._pebble_values_are_correct(1, 5)
    assert not Graph._pebble_values_are_correct(2.0, 3)
    assert not Graph._pebble_values_are_correct(2, 3.14)
    assert not Graph._pebble_values_are_correct(2, "three")
    assert not Graph._pebble_values_are_correct(-2, -1)


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
    ],
)
def test_rigid_in_d2(graph):
    assert graph.is_rigid(dim=2, combinatorial=True)
    assert graph.is_rigid(dim=2, combinatorial=False)


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
def test_not_rigid_in_d2(graph):
    assert not graph.is_rigid(dim=2, combinatorial=True)
    assert not graph.is_rigid(dim=2, combinatorial=False)


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
def test_rigid_in_d1(graph):
    assert graph.is_rigid(dim=1, combinatorial=True)
    assert graph.is_rigid(dim=1, combinatorial=False)


@pytest.mark.parametrize(
    "graph",
    [Graph.from_vertices(range(3)), Graph([[0, 1], [2, 3]])],
)
def test_not_rigid_in_d1(graph):
    assert not graph.is_rigid(dim=1, combinatorial=True)
    assert not graph.is_rigid(dim=1, combinatorial=False)


@pytest.mark.parametrize(
    "graph, dim",
    [[graphs.Complete(n), d] for d in range(1, 5) for n in range(1, d + 2)],
)
def test_is_rigid(graph, dim):
    assert graph.is_rigid(dim, combinatorial=(dim < 3))


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
def test_2_3_sparse(graph):
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
def test_not_2_3_sparse(graph):
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
def test_2_3_tight(graph):
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
def test_not_2_3_tight(graph):
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
def test_min_rigid_in_d1(graph):
    assert graph.is_min_rigid(dim=1, combinatorial=True)
    assert graph.is_min_rigid(dim=1, combinatorial=False)


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
def test_not_min_rigid_in_d1(graph):
    assert not graph.is_min_rigid(dim=1, combinatorial=True)
    assert not graph.is_min_rigid(dim=1, combinatorial=False)


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
def test_min_rigid_in_d2(graph):
    assert graph.is_min_rigid(dim=2, combinatorial=True)
    assert graph.is_min_rigid(dim=2, combinatorial=False)


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
def test_not_min_rigid_in_d2(graph):
    assert not graph.is_min_rigid(dim=2, combinatorial=True)
    assert not graph.is_min_rigid(dim=2, combinatorial=False)


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
def test_globally_rigid_in_d2(graph):
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
        [read_globally("D19V22"), 19],
        [read_globally("D19V23"), 19],
    ],
)
def test_globally_rigid_in_d(graph, gdim):
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
def test_not_globally_rigid_in_d(graph, gdim):
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
def test_not_globally_in_d2(graph):
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
def test_vertex_redundantly_rigid_in_d2(graph):
    assert graph.is_vertex_redundantly_rigid(dim=2)
    assert graph.is_vertex_redundantly_rigid(dim=2, combinatorial=False)


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
def test_k_vertex_redundantly_rigid_in_d1(graph, k):
    assert graph.is_k_vertex_redundantly_rigid(k, dim=1)
    assert graph.is_k_vertex_redundantly_rigid(k, dim=1, combinatorial=False)


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
def test_k_vertex_redundantly_rigid_in_d2(graph, k):
    assert graph.is_k_vertex_redundantly_rigid(k, dim=2)
    assert graph.is_k_vertex_redundantly_rigid(k, dim=2, combinatorial=False)


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
def test_k_vertex_redundantly_rigid_in_d3(graph, k):
    assert graph.is_k_vertex_redundantly_rigid(k, dim=3, combinatorial=False)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.CompleteBipartite(3, 3),
        graphs.ThreePrismPlusEdge(),
        graphs.K33plusEdge(),
        Graph([["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"]]),
    ],
)
def test_not_vertex_redundantly_rigid_in_d2(graph):
    assert not graph.is_vertex_redundantly_rigid(dim=2)
    assert not graph.is_vertex_redundantly_rigid(dim=2, combinatorial=False)


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(101739), 1],
        [Graph.from_int(255567), 2],
        [Graph.from_int(515576), 3],
        [Graph([["a", "b"], ["b", "c"], ["c", "a"], ["d", "a"], ["e", "d"]]), 1],
    ],
)
def test_not_k_vertex_redundantly_rigid_in_d1(graph, k):
    assert not graph.is_k_vertex_redundantly_rigid(k, dim=1)
    assert not graph.is_k_vertex_redundantly_rigid(k, dim=1, combinatorial=False)


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.ThreePrism(), 1],
        [Graph.from_int(8191), 2],
        [Graph.from_int(1048059), 3],
        [Graph([["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"]]), 1],
    ],
)
def test_not_k_vertex_redundantly_rigid_in_d2(graph, k):
    assert not graph.is_k_vertex_redundantly_rigid(k, dim=2)
    assert not graph.is_k_vertex_redundantly_rigid(k, dim=2, combinatorial=False)


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
def test_not_k_vertex_redundantly_rigid_in_d3(graph, k):
    assert not graph.is_k_vertex_redundantly_rigid(k, dim=3, combinatorial=False)


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
def test_min_k_vertex_redundantly_rigid_in_d1(graph, k):
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=1)
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=1, combinatorial=False)


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
def test_min_k_vertex_redundantly_rigid_in_d2(graph, k):
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=2)
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=2, combinatorial=False)


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(507903), 1],
        pytest.param(Graph.from_int(1048575), 2, marks=pytest.mark.slow_main),
    ],
)
def test_min_k_vertex_redundantly_rigid_in_d3(graph, k):
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=3, combinatorial=False)


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(101739), 1],
        [Graph.from_int(223), 1],
        [graphs.CompleteMinusOne(5), 2],
        [Graph.from_int(16351), 3],
    ],
)
def test_not_min_k_vertex_redundantly_rigid_in_d1(graph, k):
    assert not graph.is_min_k_vertex_redundantly_rigid(k, dim=1)
    assert not graph.is_min_k_vertex_redundantly_rigid(k, dim=1, combinatorial=False)


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.ThreePrism(), 1],
        [Graph.from_int(8191), 1],
        [Graph.from_int(32767), 2],
        [Graph.from_int(2097151), 3],
    ],
)
def test_not_min_k_vertex_redundantly_rigid_in_d2(graph, k):
    assert not graph.is_min_k_vertex_redundantly_rigid(k, dim=2)
    assert not graph.is_min_k_vertex_redundantly_rigid(k, dim=2, combinatorial=False)


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
def test_not_min_k_vertex_redundantly_rigid_in_d3(graph, k):
    assert not graph.is_min_k_vertex_redundantly_rigid(k, dim=3, combinatorial=False)


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
def test_redundantly_rigid_in_d2(graph):
    assert graph.is_redundantly_rigid(dim=2)
    assert graph.is_redundantly_rigid(dim=2, combinatorial=False)


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
def test_k_redundantly_rigid_in_d1(graph, k):
    assert graph.is_k_redundantly_rigid(k, dim=1)
    assert graph.is_k_redundantly_rigid(k, dim=1, combinatorial=False)


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
def test_k_redundantly_rigid_in_d2(graph, k):
    assert graph.is_k_redundantly_rigid(k, dim=2)
    assert graph.is_k_redundantly_rigid(k, dim=2, combinatorial=False)


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
def test_k_redundantly_rigid_in_d3(graph, k):
    assert graph.is_k_redundantly_rigid(k, dim=3, combinatorial=False)


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
def test_not_redundantly_rigid_in_d2(graph):
    assert not graph.is_redundantly_rigid(dim=2)
    assert not graph.is_redundantly_rigid(dim=2, combinatorial=False)


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(15), 1],
        [graphs.Diamond(), 2],
        [Graph([["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"]]), 3],
    ],
)
def test_not_k_redundantly_rigid_in_d1(graph, k):
    assert not graph.is_k_redundantly_rigid(k, dim=1)
    assert not graph.is_k_redundantly_rigid(k, dim=1, combinatorial=False)


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
def test_not_k_redundantly_rigid_in_d2(graph, k):
    assert not graph.is_k_redundantly_rigid(k, dim=2)
    assert not graph.is_k_redundantly_rigid(k, dim=2, combinatorial=False)


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
def test_not_k_redundantly_rigid_in_d3(graph, k):
    assert not graph.is_k_redundantly_rigid(k, dim=3, combinatorial=False)


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
def test_min_k_redundantly_rigid_in_d1(graph, k):
    assert graph.is_min_k_redundantly_rigid(k, dim=1)
    assert graph.is_min_k_redundantly_rigid(k, dim=1, combinatorial=False)


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
def test_min_k_redundantly_rigid_in_d2(graph, k):
    assert graph.is_min_k_redundantly_rigid(k, dim=2)
    assert graph.is_min_k_redundantly_rigid(k, dim=2, combinatorial=False)


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Complete(5), 1],
        [Graph.from_int(16351), 1],
        pytest.param(Graph.from_int(32767), 2, marks=pytest.mark.slow_main),
    ],
)
def test_min_k_redundantly_rigid_in_d3(graph, k):
    assert graph.is_min_k_redundantly_rigid(k, dim=3, combinatorial=False)


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
def test_not_min_k_redundantly_rigid_in_d1(graph, k):
    assert not graph.is_min_k_redundantly_rigid(k, dim=1)
    assert not graph.is_min_k_redundantly_rigid(k, dim=1, combinatorial=False)


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.ThreePrism(), 1],
        [Graph.from_int(8191), 1],
        pytest.param(Graph.from_int(16351), 2, marks=pytest.mark.slow_main),
        # [Graph.from_int(1048063), 3],
    ],
)
def test_not_min_k_redundantly_rigid_in_d2(graph, k):
    assert not graph.is_min_k_redundantly_rigid(k, dim=2)
    assert not graph.is_min_k_redundantly_rigid(k, dim=2, combinatorial=False)


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
def test_not_min_k_redundantly_rigid_in_d3(graph, k):
    assert not graph.is_min_k_redundantly_rigid(k, dim=3, combinatorial=False)


def test_rigid_components():
    G = graphs.Path(6)
    rigid_components = G.rigid_components(dim=1)
    assert rigid_components[0] == [0, 1, 2, 3, 4, 5]
    G.remove_edge(2, 3)
    rigid_components = G.rigid_components(dim=1)
    assert [set(H) for H in rigid_components] == [
        set([0, 1, 2]),
        set([3, 4, 5]),
    ] or [set(H) for H in rigid_components] == [
        set([3, 4, 5]),
        set([0, 1, 2]),
    ]

    G = graphs.Path(5)
    rigid_components = G.rigid_components()
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
    rigid_components = G.rigid_components()
    assert [set(H) for H in rigid_components] == [
        set([0, "a", "b"]),
        set([0, 1, 2, 3, 4, 5]),
    ] or [set(H) for H in rigid_components] == [
        set([0, 1, 2, 3, 4, 5]),
        set([0, "a", "b"]),
    ]

    G = Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    rigid_components = G.rigid_components()
    assert [set(H) for H in rigid_components] == [
        set([0, 1, 2]),
        set([3, 4, 5]),
    ] or [set(H) for H in rigid_components] == [
        set([3, 4, 5]),
        set([0, 1, 2]),
    ]

    G = graphs.Complete(3)
    G.add_vertex(3)
    rigid_components = G.rigid_components()
    assert [set(H) for H in rigid_components] == [set([0, 1, 2]), set([3])] or [
        set(H) for H in rigid_components
    ] == [set([3]), set([0, 1, 2])]

    G = graphs.ThreePrism()
    rigid_components = G.rigid_components()
    assert len(rigid_components) == 1 and (rigid_components == [[0, 1, 2, 3, 4, 5]])

    G = graphs.ThreeConnectedR3Circuit()
    G.remove_node(0)
    rigid_components = G.rigid_components()
    assert sorted([sorted(H) for H in rigid_components]) == [
        [1, 2, 3, 4],
        [1, 10, 11, 12],
        [4, 5, 6, 7],
        [7, 8, 9, 10],
    ]

    G = graphs.DoubleBanana()
    rigid_components = G.rigid_components(dim=3)
    assert [set(H) for H in rigid_components] == [
        set([0, 1, 2, 3, 4]),
        set([0, 1, 5, 6, 7]),
    ] or [set(H) for H in rigid_components] == [
        set([0, 1, 5, 6, 7]),
        set([0, 1, 2, 3, 4]),
    ]


def test_str():
    G = Graph([[2, 1], [2, 3]])
    assert str(G) == "Graph with vertices [1, 2, 3] and edges [[1, 2], [2, 3]]"
    G = Graph([(chr(i + 67), i + 1) for i in range(3)] + [(i, i + 1) for i in range(3)])
    assert str(G) == (
        "Graph with vertices ['C', 1, 'D', 2, 'E', 3, 0] "
        "and edges [('C', 1), (1, 0), (1, 2), ('D', 2), (2, 3), ('E', 3)]"
    )
    G = Graph.from_vertices(["C", 1, "D", 2, "E", 3, 0])
    assert str(G) == "Graph with vertices ['C', 1, 'D', 2, 'E', 3, 0] and edges []"


def test_vertex_edge_lists():
    G = Graph([[2, 1], [2, 3]])
    assert G.vertex_list() == [1, 2, 3]
    assert G.edge_list() == [[1, 2], [2, 3]]
    G = Graph([(chr(i + 67), i + 1) for i in range(3)] + [(i, i + 1) for i in range(3)])
    assert set(G.vertex_list()) == set(["C", 1, "D", 2, "E", 3, 0])
    assert set(G.edge_list()) == set(
        [("C", 1), (1, 0), (1, 2), ("D", 2), (2, 3), ("E", 3)]
    )
    G = Graph.from_vertices(["C", 1, "D", 2, "E", 3, 0])
    assert set(G.vertex_list()) == set(["C", 2, "E", 1, "D", 3, 0])
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


def test_integer_representation_fail():
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
    ],
)
def test_loops(method, params):
    with pytest.raises(LoopError):
        G = Graph([[1, 2], [1, 1], [2, 3], [1, 3]])
        func = getattr(G, method)
        func(*params)


def test_k_extension():
    assert str(graphs.Complete(2).zero_extension([0, 1])) == str(graphs.Complete(3))
    assert str(graphs.Complete(2).zero_extension([1], dim=1)) == str(graphs.Path(3))
    assert str(graphs.Complete(4).one_extension([0, 1, 2], (0, 1))) == str(
        Graph([(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4)])
    )
    assert str(
        graphs.CompleteBipartite(3, 2).one_extension([0, 1, 2, 3, 4], (0, 3), dim=4)
    ) == str(
        Graph(
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
    )
    assert str(
        graphs.CompleteBipartite(3, 2).k_extension(
            2, [0, 1, 3], [(0, 3), (1, 3)], dim=1
        )
    ) == str(Graph([(0, 4), (0, 5), (1, 4), (1, 5), (2, 3), (2, 4), (3, 5)]))
    assert str(
        graphs.CompleteBipartite(3, 2).k_extension(2, [0, 1, 3, 4], [(0, 3), (1, 3)])
    ) == str(Graph([(0, 4), (0, 5), (1, 4), (1, 5), (2, 3), (2, 4), (3, 5), (4, 5)]))
    assert str(
        graphs.Cycle(6).k_extension(
            4, [0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4)], dim=1
        )
    ) == str(Graph([(0, 5), (0, 6), (1, 6), (2, 6), (3, 6), (4, 5), (4, 6)]))


def test_all_k_extensions():
    for extension in graphs.Complete(4).all_k_extensions(1, 1):
        assert str(extension) in {
            str(Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3]])),
            str(Graph([[0, 1], [0, 3], [0, 4], [1, 2], [1, 3], [2, 3], [2, 4]])),
            str(Graph([[0, 1], [0, 2], [0, 4], [1, 2], [1, 3], [2, 3], [3, 4]])),
            str(Graph([[0, 1], [0, 2], [0, 3], [1, 3], [1, 4], [2, 3], [2, 4]])),
            str(Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 3], [3, 4]])),
            str(Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 4], [3, 4]])),
        }
    for extension in graphs.Complete(4).all_k_extensions(
        2, 2, only_non_isomorphic=True
    ):
        assert str(extension) in {
            str(
                Graph([[0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
            ),
            str(
                Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 4], [3, 4]])
            ),
        }
    all_diamond_0_2 = list(
        graphs.Diamond().all_k_extensions(0, 2, only_non_isomorphic=True)
    )
    assert (
        len(all_diamond_0_2) == 3
        and str(all_diamond_0_2[0])
        == str(Graph([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 4], [2, 3]]))
        and str(all_diamond_0_2[1])
        == str(Graph([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [2, 4]]))
        and str(all_diamond_0_2[2])
        == str(Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 3], [3, 4]]))
    )
    all_diamond_1_2 = graphs.Diamond().all_k_extensions(1, 2, only_non_isomorphic=True)
    assert str(next(all_diamond_1_2)) == str(
        Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 4], [2, 3], [2, 4]])
    ) and str(next(all_diamond_1_2)) == str(
        Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 4], [2, 3], [3, 4]])
    )


def test_k_extension_fail():
    with pytest.raises(TypeError):
        graphs.Complete(6).k_extension(2, [0, 1, 2], [[0, 1], [0, 2]], dim=-1)
    with pytest.raises(ValueError):
        graphs.Complete(6).k_extension(2, [0, 1, 6], [[0, 1], [0, 6]], dim=1)
    with pytest.raises(ValueError):
        graphs.Complete(6).k_extension(2, [0, 1, 2], [[0, 1]], dim=1)
    with pytest.raises(ValueError):
        graphs.CompleteBipartite(2, 3).k_extension(
            2, [0, 1, 2], [[0, 1], [0, 2]], dim=1
        )
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
def test_extension_sequence(graph):
    assert graph.extension_sequence()


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
def test_extension_sequence_false(graph):
    assert not graph.extension_sequence()


def test_extension_sequence_solution():
    result = graphs.Complete(2).extension_sequence(return_solution=True)
    solution = [
        Graph([[0, 1]]),
    ]
    for i in range(len(result)):
        assert str(result[i]) == str(solution[i])

    result = graphs.Complete(3).extension_sequence(return_solution=True)
    solution = [
        Graph([[1, 2]]),
        Graph([[0, 1], [0, 2], [1, 2]]),
    ]
    for i in range(len(result)):
        assert str(result[i]) == str(solution[i])

    result = graphs.CompleteBipartite(3, 3).extension_sequence(return_solution=True)
    solution = [
        Graph([[3, 4]]),
        Graph([[2, 3], [2, 4], [3, 4]]),
        Graph([[1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]),
        Graph([[1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4]]),
        Graph(
            [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]],
        ),
    ]
    for i in range(len(result)):
        assert str(result[i]) == str(solution[i])
    solution_ext = [
        [0, [3, 4], [], 2],  # k, vertices, edges, new_vertex
        [0, [3, 4], [], 1],
        [0, [1, 2], [], 5],
        [1, [3, 4, 5], [(3, 4)], 0],
    ]
    G = Graph([[3, 4]])
    for i in range(len(result)):
        assert str(result[i]) == str(G)
        if i < len(solution_ext):
            G.k_extension(*solution_ext[i], dim=2, inplace=True)

    result = graphs.Diamond().extension_sequence(return_solution=True)
    solution = [
        Graph([[2, 3]]),
        Graph([[0, 2], [0, 3], [2, 3]]),
        Graph([[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]]),
    ]
    for i in range(len(result)):
        assert str(result[i]) == str(solution[i])

    result = graphs.ThreePrism().extension_sequence(return_solution=True)
    solution = [
        Graph([[4, 5]]),
        Graph([[3, 4], [3, 5], [4, 5]]),
        Graph([[1, 3], [1, 4], [3, 4], [3, 5], [4, 5]]),
        Graph([[1, 2], [1, 3], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]]),
        Graph(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]],
        ),
    ]
    for i in range(len(result)):
        assert str(result[i]) == str(solution[i])
    solution_ext = [
        [0, [4, 5], [], 3],  # k, vertices, edges, new_vertex
        [0, [3, 4], [], 1],
        [0, [1, 5], [], 2],
        [1, [1, 2, 3], [(1, 3)], 0],
    ]
    G = Graph([[4, 5]])
    for i in range(len(result)):
        assert str(result[i]) == str(G)
        if i < len(solution_ext):
            G.k_extension(*solution_ext[i], dim=2, inplace=True)


def test_CompleteOnVertices():
    assert str(Graph.CompleteOnVertices([0, 1, 2, 3, 4, 5])) == str(graphs.Complete(6))
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


def test_check_edge_list():
    G = Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)])
    G._check_edge((1, 2))
    G._check_edge([3, 2])
    G._check_edge_list([(1, 2), (2, 3)])
    G._check_edge_list([(1, 2)], [1, 2])
    G._check_edge_list([(2, 3)], [2, 3])
    with pytest.raises(ValueError):
        G._check_edge((1, 3))
    with pytest.raises(ValueError):
        G._check_edge((1, 4))
    with pytest.raises(ValueError):
        G._check_edge_list([(1, 2), (1, 3), (2, 3)])
    with pytest.raises(ValueError):
        G._check_edge_list([(1, 2), (2, 3)], [1, 2])
    with pytest.raises(TypeError):
        G._check_edge_list([(2,)])
    with pytest.raises(TypeError):
        G._check_edge_list([2, 3])
    with pytest.raises(TypeError):
        G._check_edge_list(["23"])


def test_check_edge_format_list():
    G = Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)])
    G._check_edge_format((1, 3))
    G._check_edge_format([3, 1])
    G._check_edge_format_list([(1, 2), (1, 3)])
    G._check_edge_format_list([(1, 2), (1, 3), (2, 3)])
    with pytest.raises(ValueError):
        G._check_edge_format((1, 4))
    with pytest.raises(TypeError):
        G._check_edge_format_list([(2,)])
    with pytest.raises(TypeError):
        G._check_edge_format_list([2, 3])
    with pytest.raises(TypeError):
        G._check_edge_format_list(["23"])
    with pytest.raises(LoopError):
        G._check_edge_format([3, 3])
    with pytest.raises(LoopError):
        G._check_edge_format_list([(1, 1), (1, 3), (2, 3)])


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


def test_is_3_6_sparse():
    """The Double Banana is (3,6)-tight."""
    G = graphs.DoubleBanana()
    assert G.is_sparse(3, 6)
    G.add_edge(0, 1)
    assert not G.is_sparse(3, 6)


def test_is_k_l_tight():
    G = graphs.Complete(4)
    assert G.is_tight(2, 2)
    G = graphs.CompleteBipartite(4, 4)
    assert not G.is_tight(3, 6)


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
def test_number_of_realizations_cf(graph, n):
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
def test_number_of_realizations_sphere_cf(graph, n):
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
def test_Rd_circuit_d1(graph):
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
    ],
)
def test_not_Rd_circuit_d1(graph):
    assert not graph.is_Rd_circuit(dim=1)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(4),
        graphs.ThreePrismPlusEdge(),
        graphs.K33plusEdge(),
    ],
)
def test_Rd_circuit_d2(graph):
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
    ],
)
def test_not_Rd_circuit_d2(graph):
    assert not graph.is_Rd_circuit(dim=2)


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
    ],
)
def test_max_rigid_dimension(graph, k):
    assert graph.max_rigid_dimension() == k
