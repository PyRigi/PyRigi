from pyrigi.graph import Graph
import pyrigi.graphDB as graphs
from pyrigi.exception import LoopError

import pytest
from sympy import Matrix
import math


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
        [Graph.from_int(1048575), 3],
    ],
)
def test_min_k_vertex_redundantly_rigid_in_d2(graph, k):
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=2)
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=2, combinatorial=False)


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(507903), 1],
        [Graph.from_int(1048575), 2],
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
        [graphs.Octahedral(), 2],
        [graphs.Complete(6), 2],
        [graphs.Complete(6), 3],
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
        [Graph.from_int(507851), 2],
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
        [Graph.from_int(32767), 2],
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
        [Graph.from_int(16351), 2],
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
        [Graph.from_int(1048063), 2],
        # [Graph.from_int(1048575), 3],
        # [Graph.from_int(134201311), 3],
    ],
)
def test_not_min_k_redundantly_rigid_in_d3(graph, k):
    assert not graph.is_min_k_redundantly_rigid(k, dim=3, combinatorial=False)


def test_min_rigid_subgraphs():
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
    assert [set(H) for H in G.min_rigid_subgraphs()] == [
        set([0, "a", "b"]),
        set([0, 1, 5, 3, 2, 4]),
    ] or [set(H) for H in G.min_rigid_subgraphs()] == [
        set([0, 1, 5, 3, 2, 4]),
        set([0, "a", "b"]),
    ]

    G = Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    assert [set(H) for H in G.max_rigid_subgraphs()] == [
        set([0, 1, 2]),
        set([3, 4, 5]),
    ] or [set(H) for H in G.max_rigid_subgraphs()] == [
        set([3, 4, 5]),
        set([0, 1, 2]),
    ]

    G = graphs.ThreePrism()
    min_subgraphs = G.min_rigid_subgraphs()
    assert len(min_subgraphs) == 2 and (
        min_subgraphs == [[0, 1, 2], [3, 4, 5]]
        or min_subgraphs == [[3, 4, 5], [0, 1, 2]]
    )


def test_max_rigid_subgraphs():
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
    assert [set(H) for H in G.max_rigid_subgraphs()] == [
        set([0, "a", "b"]),
        set([0, 1, 5, 3, 2, 4]),
    ] or [set(H) for H in G.max_rigid_subgraphs()] == [
        set([0, 1, 5, 3, 2, 4]),
        set([0, "a", "b"]),
    ]

    G = Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    assert [set(H) for H in G.max_rigid_subgraphs()] == [
        set([0, 1, 2]),
        set([3, 4, 5]),
    ] or [set(H) for H in G.max_rigid_subgraphs()] == [
        set([3, 4, 5]),
        set([0, 1, 2]),
    ]

    G = graphs.ThreePrism()
    G.delete_edge([4, 5])
    max_subgraphs = G.max_rigid_subgraphs()
    assert len(max_subgraphs) == 1 and max_subgraphs[0] == [0, 1, 2]


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
        ["max_rigid_subgraphs", []],
        ["min_rigid_subgraphs", []],
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
