from itertools import combinations, product
import math
from random import randint, sample

import networkx as nx
import pytest

import pyrigi.graphDB as graphs
import pyrigi.sparsity as sparsity
from pyrigi.graph import Graph
from test_graph import read_sparsity, TEST_WRAPPED_FUNCTIONS


is_kl_sparse_algorithms_sparsity_all_kl = ["default", "subgraph"]
is_kl_sparse_algorithms_sparsity_pebble = is_kl_sparse_algorithms_sparsity_all_kl + [
    "pebble"
]


###############################################################
# is_sparse
###############################################################
@pytest.mark.parametrize(
    "graph, K, L",
    [
        [graphs.CompleteBipartite(4, 4), 3, 6],
        [Graph.from_int(32764), 4, 10],
        [Graph([[0, 1], [0, 2], [0, 3], [0, 4]]), 4, 8],
        # tight also
        [graphs.K66MinusPerfectMatching(), 3, 6],
        [graphs.DoubleBanana(), 3, 6],
        [Graph.from_int(32766), 4, 10],
        [graphs.Complete(3), 4, 9],
        [graphs.Complete(6), 4, 9],
        [Graph.from_int(2097150), 4, 8],
    ],
)
@pytest.mark.parametrize("algorithm", is_kl_sparse_algorithms_sparsity_all_kl)
def test_is_kl_sparse(graph, K, L, algorithm):
    assert nx.number_of_selfloops(graph) == 0
    assert graph.is_kl_sparse(K, L, algorithm=algorithm)
    graph2 = graph + Graph.from_vertices_and_edges([max(graph.vertex_list()) + 1], [])
    assert graph2.is_kl_sparse(K, L, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert sparsity.is_kl_sparse(nx.Graph(graph), K, L, algorithm=algorithm)
        assert sparsity.is_kl_sparse(nx.Graph(graph2), K, L, algorithm=algorithm)


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [graphs.DoubleBanana() + Graph([[0, 1]]), 3, 6],
        [graphs.Complete(6), 4, 10],
        [graphs.Complete(6) + Graph.from_vertices_and_edges([7], []), 4, 10],
        [graphs.Complete(7), 4, 8],
        [Graph.from_int(2097136), 3, 6],
    ],
)
@pytest.mark.parametrize("algorithm", is_kl_sparse_algorithms_sparsity_all_kl)
def test_is_not_kl_sparse(graph, K, L, algorithm):
    assert nx.number_of_selfloops(graph) == 0
    assert not graph.is_kl_sparse(K, L, algorithm=algorithm)
    graph2 = graph + Graph.from_vertices_and_edges([max(graph.vertex_list()) + 1], [])
    assert not graph2.is_kl_sparse(K, L, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not sparsity.is_kl_sparse(nx.Graph(graph), K, L, algorithm=algorithm)
        assert not sparsity.is_kl_sparse(nx.Graph(graph2), K, L, algorithm=algorithm)


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
    if TEST_WRAPPED_FUNCTIONS:
        assert sparsity.is_kl_sparse(nx.Graph(graph), 2, 3, algorithm="subgraph")
        assert sparsity.is_kl_sparse(nx.Graph(graph), 2, 3, algorithm="pebble")


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
    if TEST_WRAPPED_FUNCTIONS:
        assert not sparsity.is_kl_sparse(nx.Graph(graph), 2, 3, algorithm="subgraph")
        assert not sparsity.is_kl_sparse(nx.Graph(graph), 2, 3, algorithm="pebble")


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [Graph([[0, 1], [0, 2], [0, 3], [0, 4]]), 1, 0],
        [Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]), 2, 0],
        [Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]]), 2, 1],
        [Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]]), 2, 2],
        [Graph([[0, 1], [0, 2], [0, 3], [1, 2]]), 2, 3],
        [graphs.Complete(4), 2, 2],
        [graphs.Complete(5), 2, 0],
        [graphs.CompleteMinusOne(5), 2, 1],
        [Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]]), 2, 3],
        [Graph([[0, 1], [0, 2], [0, 3], [0, 4]]), 1, 1],
        [Graph([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2]]), 1, 0],
    ],
)
@pytest.mark.parametrize("algorithm", is_kl_sparse_algorithms_sparsity_pebble)
def test_is_kl_sparse_pebble(graph, K, L, algorithm):
    assert nx.number_of_selfloops(graph) == 0
    assert graph.is_kl_sparse(K, L, algorithm=algorithm)
    graph2 = graph + Graph.from_vertices_and_edges([max(graph.vertex_list()) + 1], [])
    assert graph2.is_kl_sparse(K, L, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert sparsity.is_kl_sparse(nx.Graph(graph), K, L, algorithm=algorithm)
        assert sparsity.is_kl_sparse(nx.Graph(graph2), K, L, algorithm=algorithm)


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [Graph.from_int(32764), 2, 0],
        [Graph.from_int(32692), 2, 1],
        [graphs.CompleteMinusOne(5), 2, 2],
        [Graph([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 3]]), 2, 3],
        [graphs.Cycle(4), 1, 1],
    ],
)
@pytest.mark.parametrize("algorithm", is_kl_sparse_algorithms_sparsity_pebble)
def test_is_not_kl_sparse_pebble(graph, K, L, algorithm):
    assert nx.number_of_selfloops(graph) == 0
    assert not graph.is_kl_sparse(K, L, algorithm=algorithm)
    graph2 = graph + Graph.from_vertices_and_edges([max(graph.vertex_list()) + 1], [])
    assert not graph2.is_kl_sparse(K, L, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not sparsity.is_kl_sparse(nx.Graph(graph), K, L, algorithm=algorithm)
        assert not sparsity.is_kl_sparse(nx.Graph(graph2), K, L, algorithm=algorithm)


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [Graph([[0, 0], [0, 1], [1, 1]]), 2, 1],
        [Graph([[0, 0]]), 2, 1],  # corner case, only one vertex
        [Graph([[0, 0], [1, 1]]), 3, 0],  # Two disjoint loops
        [Graph([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2]]), 2, 1],
        [Graph([[0, 0], [0, 1], [1, 2], [2, 2], [2, 0]]), 2, 0],
        [
            graphs.Complete(6)
            + Graph([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]),
            4,
            2,
        ],
        [graphs.CompleteLooped(2), 2, 0],
        [graphs.CompleteLooped(3), 2, 0],
    ],
)
@pytest.mark.parametrize("algorithm", is_kl_sparse_algorithms_sparsity_pebble)
def test_is_kl_sparse_with_loops(graph, K, L, algorithm):
    assert nx.number_of_selfloops(graph) > 0
    assert graph.is_kl_sparse(K, L, algorithm=algorithm)
    graph2 = graph + Graph.from_vertices_and_edges([max(graph.vertex_list()) + 1], [])
    assert graph2.is_kl_sparse(K, L, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert sparsity.is_kl_sparse(nx.Graph(graph), K, L, algorithm=algorithm)
        assert sparsity.is_kl_sparse(nx.Graph(graph2), K, L, algorithm=algorithm)


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [Graph([[0, 1], [1, 1]]), 1, 1],
        [Graph([[0, 0]]), 2, 2],  # corner case, only one vertex
        [graphs.CompleteLooped(3), 1, 0],
        [graphs.CompleteLooped(3), 1, 1],
        [graphs.CompleteLooped(3), 2, 1],
        [graphs.CompleteLooped(3), 2, 2],
        [graphs.CompleteLooped(3), 2, 3],
    ],
)
@pytest.mark.parametrize("algorithm", is_kl_sparse_algorithms_sparsity_pebble)
def test_is_not_kl_sparse_with_loops(graph, K, L, algorithm):
    assert nx.number_of_selfloops(graph) > 0
    assert not graph.is_kl_sparse(K, L, algorithm=algorithm)
    graph2 = graph + Graph.from_vertices_and_edges([max(graph.vertex_list()) + 1], [])
    assert not graph2.is_kl_sparse(K, L, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not sparsity.is_kl_sparse(nx.Graph(graph), K, L, algorithm=algorithm)
        assert not sparsity.is_kl_sparse(nx.Graph(graph2), K, L, algorithm=algorithm)


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
def test_is_not_sparse_graphs_big_random(graph, K, L):
    assert not graph.is_kl_sparse(K, L, algorithm="pebble")
    if TEST_WRAPPED_FUNCTIONS:
        assert not sparsity.is_kl_sparse(nx.Graph(graph), K, L, algorithm="pebble")


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [Graph([[0, 1], [1, 2]]), 2, 4],
        [Graph([[0, 1], [1, 2]]), 3, 6],
        [Graph([[0, 1], [1, 2]]), 4, 10],
        [Graph([[0, 1], [1, 2]]), 3, 7],
        [Graph([[0, 1], [2, 0]]), 0, 0],
        [Graph([[0, 1], [2, 0]]), 1, -1],
        [Graph([[0, 1], [2, 0]]), -1, 1],
    ],
)
def test_is_kl_sparse_pebble_value_error(graph, K, L):
    assert nx.number_of_selfloops(graph) == 0
    with pytest.raises(ValueError):
        graph.is_kl_sparse(K, L, algorithm="pebble")
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.raises(ValueError):
            sparsity.is_kl_sparse(nx.Graph(graph), K, L, algorithm="pebble")


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [Graph([[0, 1], [1, 2]]), 2, 4],
        [Graph([[0, 1], [1, 2]]), 3, 7],
        [Graph([[0, 1], [2, 0]]), 0, 0],
        [Graph([[0, 1], [2, 0]]), 1, -1],
        [Graph([[0, 1], [2, 0]]), -1, 1],
    ],
)
@pytest.mark.parametrize("algorithm", is_kl_sparse_algorithms_sparsity_all_kl)
def test_is_kl_sparse_value_error(graph, K, L, algorithm):
    assert nx.number_of_selfloops(graph) == 0
    with pytest.raises(ValueError):
        graph.is_kl_sparse(K, L, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.raises(ValueError):
            sparsity.is_kl_sparse(nx.Graph(graph), K, L, algorithm=algorithm)


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [Graph([[0, 1], [1, 2], [2, 0]]), 1.2, 0],
        [Graph([[0, 1], [1, 2], [2, 0]]), 2, 0.5],
        [Graph([[0, 1], [1, 2], [2, 0]]), "1", 0],
        [Graph([[0, 1], [1, 2], [2, 0]]), 2, "0"],
    ],
)
@pytest.mark.parametrize("algorithm", is_kl_sparse_algorithms_sparsity_all_kl)
def test_is_kl_sparse_type_error(graph, K, L, algorithm):
    assert nx.number_of_selfloops(graph) == 0
    with pytest.raises(TypeError):
        graph.is_kl_sparse(K, L, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.raises(TypeError):
            sparsity.is_kl_sparse(nx.Graph(graph), K, L, algorithm=algorithm)


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [Graph([[0, 0]]), 2, 4],  # corner case, only one vertex
        [Graph([[0, 0], [1, 1]]), 3, 6],  # Two disjoint loops
        [Graph([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2]]), 2, 4],
        [Graph([[0, 0], [0, 1], [1, 2], [2, 2], [2, 0]]), 0, 0],
        [Graph([[0, 0], [0, 1], [1, 2], [2, 2], [2, 0]]), 1, -1],
        [Graph([[0, 0], [0, 1], [1, 2], [2, 2], [2, 0]]), -1, 0],
    ],
)
@pytest.mark.parametrize("algorithm", is_kl_sparse_algorithms_sparsity_pebble)
def test_is_kl_sparse_with_loops_value_error(graph, K, L, algorithm):
    assert nx.number_of_selfloops(graph) > 0
    with pytest.raises(ValueError):
        graph.is_kl_sparse(K, L, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.raises(ValueError):
            sparsity.is_kl_sparse(nx.Graph(graph), K, L, algorithm=algorithm)


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [Graph([[0, 0], [0, 1], [1, 2], [2, 2], [2, 0]]), 1.2, 0],
        [Graph([[0, 0], [0, 1], [1, 2], [2, 2], [2, 0]]), 2, 0.5],
        [Graph([[0, 0], [0, 1], [1, 2], [2, 2], [2, 0]]), "1", 0],
        [Graph([[0, 0], [0, 1], [1, 2], [2, 2], [2, 0]]), 2, "0"],
    ],
)
@pytest.mark.parametrize("algorithm", is_kl_sparse_algorithms_sparsity_all_kl)
def test_is_kl_sparse_with_loops_type_error(graph, K, L, algorithm):
    assert nx.number_of_selfloops(graph) > 0
    with pytest.raises(TypeError):
        graph.is_kl_sparse(K, L, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.raises(TypeError):
            sparsity.is_kl_sparse(nx.Graph(graph), K, L, algorithm=algorithm)


###############################################################
# is_tight
###############################################################
@pytest.mark.parametrize(
    "graph, K, L",
    [
        [graphs.K66MinusPerfectMatching(), 3, 6],
        [graphs.DoubleBanana(), 3, 6],
        [Graph.from_int(32766), 4, 10],
        [graphs.Complete(3), 4, 9],
        [graphs.Complete(6), 4, 9],
        [Graph.from_int(2097150), 4, 8],
    ],
)
@pytest.mark.parametrize("algorithm", is_kl_sparse_algorithms_sparsity_all_kl)
def test_is_kl_tight(graph, K, L, algorithm):
    assert nx.number_of_selfloops(graph) == 0
    assert graph.is_kl_tight(K, L, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert sparsity.is_kl_tight(nx.Graph(graph), K, L, algorithm=algorithm)


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [graphs.CompleteBipartite(4, 4), 3, 6],
        [Graph.from_int(32764), 4, 10],
        [Graph([[0, 1], [0, 2], [0, 3], [0, 4]]), 4, 8],
        # not sparse either
        [graphs.DoubleBanana() + Graph([[0, 1]]), 3, 6],
        [graphs.Complete(6), 4, 10],
        [graphs.Complete(6) + Graph.from_vertices_and_edges([7], []), 4, 10],
        [graphs.Complete(7), 4, 8],
        [Graph.from_int(2097136), 3, 6],
    ],
)
@pytest.mark.parametrize("algorithm", is_kl_sparse_algorithms_sparsity_all_kl)
def test_is_not_kl_tight(graph, K, L, algorithm):
    assert nx.number_of_selfloops(graph) == 0
    assert not graph.is_kl_tight(K, L, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not sparsity.is_kl_tight(nx.Graph(graph), K, L, algorithm=algorithm)


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
    if TEST_WRAPPED_FUNCTIONS:
        assert sparsity.is_kl_tight(nx.Graph(graph), 2, 3, algorithm="pebble")
        assert sparsity.is_kl_tight(nx.Graph(graph), 2, 3, algorithm="subgraph")


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
    if TEST_WRAPPED_FUNCTIONS:
        assert not sparsity.is_kl_tight(nx.Graph(graph), 2, 3, algorithm="pebble")
        assert not sparsity.is_kl_tight(nx.Graph(graph), 2, 3, algorithm="subgraph")


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [graphs.Complete(4), 2, 2],
        [graphs.Complete(5), 2, 0],
        [graphs.CompleteMinusOne(5), 2, 1],
        [Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]]), 2, 3],
        [Graph([[0, 1], [0, 2], [0, 3], [0, 4]]), 1, 1],
        [Graph([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2]]), 1, 0],
    ],
)
@pytest.mark.parametrize("algorithm", is_kl_sparse_algorithms_sparsity_pebble)
def test_is_kl_tight_pebble(graph, K, L, algorithm):
    assert nx.number_of_selfloops(graph) == 0
    assert graph.is_kl_tight(K, L, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert sparsity.is_kl_tight(nx.Graph(graph), K, L, algorithm=algorithm)


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [Graph([[0, 1], [0, 2], [0, 3], [0, 4]]), 1, 0],
        [Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]), 2, 0],
        [Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]]), 2, 1],
        [Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]]), 2, 2],
        [Graph([[0, 1], [0, 2], [0, 3], [1, 2]]), 2, 3],
        [graphs.Cycle(4), 1, 1],
        # not sparse either
        [Graph.from_int(32764), 2, 0],
        [Graph.from_int(32692), 2, 1],
        [graphs.CompleteMinusOne(5), 2, 2],
        [Graph([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 3]]), 2, 3],
        [graphs.Cycle(4), 1, 1],
    ],
)
@pytest.mark.parametrize("algorithm", is_kl_sparse_algorithms_sparsity_pebble)
def test_is_not_kl_tight_pebble(graph, K, L, algorithm):
    assert nx.number_of_selfloops(graph) == 0
    assert not graph.is_kl_tight(K, L, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not sparsity.is_kl_tight(nx.Graph(graph), K, L, algorithm=algorithm)


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [Graph([[0, 0]]), 2, 1],  # corner case, only one vertex
        [Graph([[0, 0], [1, 1]]), 1, 0],  # Two disjoint loops
        [Graph([[0, 0], [0, 1], [1, 1]]), 2, 1],
        [graphs.CompleteLooped(3), 2, 0],
        [
            graphs.Complete(6)
            + Graph([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]),
            4,
            3,
        ],
    ],
)
@pytest.mark.parametrize("algorithm", is_kl_sparse_algorithms_sparsity_pebble)
def test_is_kl_tight_with_loops(graph, K, L, algorithm):
    assert nx.number_of_selfloops(graph) > 0
    assert graph.is_kl_tight(K, L, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert sparsity.is_kl_tight(nx.Graph(graph), K, L, algorithm=algorithm)


@pytest.mark.parametrize(
    "graph, K, L",
    [
        [Graph([[0, 0], [0, 1], [1, 2], [2, 2], [2, 0]]), 2, 0],
        [
            graphs.Complete(6)
            + Graph([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]),
            4,
            2,
        ],
        [graphs.CompleteLooped(2), 2, 0],
        # not sparse either
        [Graph([[0, 1], [1, 1]]), 1, 1],
        [Graph([[0, 0]]), 2, 2],
        [graphs.CompleteLooped(3), 1, 0],
        [graphs.CompleteLooped(3), 1, 1],
        [graphs.CompleteLooped(3), 2, 1],
        [graphs.CompleteLooped(3), 2, 2],
        [graphs.CompleteLooped(3), 2, 3],
    ],
)
@pytest.mark.parametrize("algorithm", is_kl_sparse_algorithms_sparsity_pebble)
def test_is_not_kl_tight_with_loops(graph, K, L, algorithm):
    assert nx.number_of_selfloops(graph) > 0
    assert not graph.is_kl_tight(K, L, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not sparsity.is_kl_tight(nx.Graph(graph), K, L, algorithm=algorithm)


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
def test_is_tight_big_random(graph, K, L):
    assert graph.is_kl_tight(K, L, algorithm="pebble")
    if TEST_WRAPPED_FUNCTIONS:
        assert sparsity.is_kl_tight(nx.Graph(graph), K, L, algorithm="pebble")


###############################################################
# spanning_kl_sparse_subgraph
###############################################################
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
            if TEST_WRAPPED_FUNCTIONS:
                spanning_subgraph = sparsity.spanning_kl_sparse_subgraph(
                    nx.Graph(graph), K, L
                )
                assert sparsity.is_kl_sparse(
                    spanning_subgraph, K, L, algorithm="subgraph"
                )
                assert set(graph.nodes) == set(spanning_subgraph.nodes)


###############################################################
# large tests
###############################################################
@pytest.mark.long_local
def test_sparsity_properties_random_graphs_with_loops():
    search_space = [range(1, 8), range(10)]
    for n, _ in product(*search_space):
        for m in range(1, math.comb(n, 2) + 1):
            G = Graph(nx.gnm_random_graph(n, m))
            loops = [[v, v] for v in G.vertex_list() if randint(0, 1)]
            G.add_edges(loops)
            assert G.number_of_nodes() == n
            assert G.number_of_edges() == m + len(loops)

            _run_sparsity_test_on_graph(G)


@pytest.mark.long_local
def test_sparsity_properties_random_graphs_without_loops():
    search_space = [range(1, 8), range(10)]
    for n, _ in product(*search_space):
        for m in range(1, math.comb(n, 2) + 1):
            G = Graph(nx.gnm_random_graph(n, m))
            assert G.number_of_nodes() == n
            assert G.number_of_edges() == m

            _run_sparsity_test_on_graph(G)


def _run_sparsity_test_on_graph(G: Graph) -> None:
    """
    Run a set of sparsity tests on a given graph
    """
    kmax = 6
    m = G.number_of_edges()
    n = G.number_of_nodes()

    # distinguish range for L depending on loops
    def get_max_L(G: Graph, K: int):
        if nx.number_of_selfloops(G) > 0:
            return 2 * K
        else:
            return math.comb(K + 1, 2) + 1

    prop_sparse = {
        K: [G.is_kl_sparse(K, L) for L in range(get_max_L(G, K))]
        for K in range(1, kmax)
    }
    prop_tight = {
        K: [G.is_kl_tight(K, L) for L in range(get_max_L(G, K))] for K in range(1, kmax)
    }

    prop_sparse_s = {
        K: [G.is_kl_sparse(K, L, algorithm="subgraph") for L in range(get_max_L(G, K))]
        for K in range(1, kmax)
    }
    prop_tight_s = {
        K: [G.is_kl_tight(K, L, algorithm="subgraph") for L in range(get_max_L(G, K))]
        for K in range(1, kmax)
    }

    for K in range(1, kmax):
        for L in range(get_max_L(G, K)):
            # check output type
            assert isinstance(prop_sparse[K][L], bool)
            assert isinstance(prop_sparse_s[K][L], bool)
            assert isinstance(prop_tight[K][L], bool)
            assert isinstance(prop_tight_s[K][L], bool)

            # compare different algorithms
            assert prop_tight[K][L] == prop_tight_s[K][L]
            assert prop_sparse[K][L] == prop_sparse_s[K][L]

            # sanity checks on properties
            if prop_tight[K][L]:
                assert prop_sparse[K][L]
                if n >= K:
                    assert m == K * n - L
            if prop_sparse[K][L]:
                if n >= K:
                    assert m <= K * n - L
                for L2 in range(L):
                    assert prop_sparse[K][L2]

    if prop_sparse[1][1]:
        if nx.is_connected(G):
            assert nx.is_tree(G)


@pytest.mark.long_local
def test_sparsity_properties_small_graphs_without_loops():
    for n in range(1, 5):
        for i in range(math.comb(n, 2) + 1):
            for edges in combinations(combinations(range(n), 2), i):
                G = Graph.from_vertices_and_edges(range(n), edges)
                assert G.number_of_nodes() == n
                assert G.number_of_edges() == len(edges)

                _run_sparsity_test_on_graph(G)


@pytest.mark.long_local
def test_sparsity_properties_small_graphs_without_loops_different_vertex_names():
    for n in range(1, 4):
        encodings = product(
            *list(
                zip(
                    range(n),
                    sample(range(n, 3 * n), n),
                    [
                        chr(i) for i in sample(range(97, 122), n)
                    ],  # does not work for graphs with more than 26 vertices
                    [
                        tuple([chr(i) if randint(0, 1) == 0 else i for i in elem])
                        for elem in product(sample(range(97, 122), n), repeat=2)
                    ],
                )
            )
        )
        for vertices in encodings:
            for i in range(math.comb(n, 2) + 1):
                for edges in combinations(combinations(vertices, 2), i):
                    G = Graph.from_vertices_and_edges(vertices, edges)
                    assert G.number_of_nodes() == n
                    assert len(vertices) == n
                    assert G.number_of_edges() == len(edges)

                    _run_sparsity_test_on_graph(G)


@pytest.mark.long_local
def test_sparsity_properties_small_graphs_with_loops():
    for n in range(1, 5):
        for i in range(math.comb(n, 2) + 1):
            for edges in combinations(combinations(range(n), 2), i):
                G = Graph.from_vertices_and_edges(range(n), edges)
                for j in range(n + 1):
                    for loops in combinations(range(n), j):
                        G.add_edges([[jj, jj] for jj in loops])
                        assert G.number_of_nodes() == n
                        assert G.number_of_edges() == len(edges) + len(loops)

                        _run_sparsity_test_on_graph(G)
                        G.delete_edges([[jj, jj] for jj in loops])


@pytest.mark.long_local
def test_sparsity_properties_small_graphs_with_loops_different_vertex_names():
    for n in range(1, 4):
        encodings = product(
            *list(
                zip(
                    range(n),
                    sample(range(n, 3 * n), n),
                    [
                        chr(i) for i in sample(range(97, 122), n)
                    ],  # does not work for graphs with more than 26 vertices
                    [
                        tuple([chr(i) if randint(0, 1) == 0 else i for i in elem])
                        for elem in product(sample(range(97, 122), n), repeat=2)
                    ],
                )
            )
        )
        for vertices in encodings:
            for i in range(math.comb(n, 2) + 1):
                for edges in combinations(combinations(vertices, 2), i):
                    G = Graph.from_vertices_and_edges(vertices, edges)
                    for j in range(n + 1):
                        for loops in combinations(vertices, j):
                            G.add_edges([[jj, jj] for jj in loops])
                            assert G.number_of_nodes() == n
                            assert len(vertices) == n
                            assert G.number_of_edges() == len(edges) + len(loops)

                            _run_sparsity_test_on_graph(G)
                            G.delete_edges([[jj, jj] for jj in loops])
