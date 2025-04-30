import networkx as nx
import pytest
from test_graph import (
    TEST_WRAPPED_FUNCTIONS,
    is_rigid_algorithms_all_d,
    is_rigid_algorithms_d1,
    is_rigid_algorithms_d2,
)

import pyrigi.graphDB as graphs
import pyrigi.redundant_rigidity as redundant_rigidity
from pyrigi.graph import Graph


###############################################################
# is_vertex_redundantly_rigid
###############################################################
@pytest.mark.parametrize(
    "graph",
    [
        graphs.CompleteMinusOne(5),
        graphs.Complete(5),
        Graph.from_int(7679),
        Graph([["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"], ["b", "d"]]),
    ],
)
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d2)
def test_is_vertex_redundantly_rigid_d2(graph, algorithm):
    assert graph.is_vertex_redundantly_rigid(dim=2, algorithm=algorithm)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.CompleteBipartite(3, 3),
        graphs.ThreePrismPlusEdge(),
        graphs.K33plusEdge(),
        Graph([["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"]]),
    ],
)
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d2)
def test_is_not_vertex_redundantly_rigid_d2(graph, algorithm):
    assert not graph.is_vertex_redundantly_rigid(dim=2, algorithm=algorithm)


###############################################################
# is_k_vertex_redundantly_rigid
###############################################################
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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d1)
def test_is_k_vertex_redundantly_rigid_d1(graph, k, algorithm):
    assert graph.is_k_vertex_redundantly_rigid(k, dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert redundant_rigidity.is_k_vertex_redundantly_rigid(
            nx.Graph(graph), k, dim=1, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(101739), 1],
        [Graph.from_int(255567), 2],
        [Graph.from_int(515576), 3],
        [Graph([["a", "b"], ["b", "c"], ["c", "a"], ["d", "a"], ["e", "d"]]), 1],
    ],
)
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d1)
def test_is_not_k_vertex_redundantly_rigid_d1(graph, k, algorithm):
    assert not graph.is_k_vertex_redundantly_rigid(k, dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not redundant_rigidity.is_k_vertex_redundantly_rigid(
            nx.Graph(graph), k, dim=1, algorithm=algorithm
        )


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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d2)
def test_is_k_vertex_redundantly_rigid_d2(graph, k, algorithm):
    assert graph.is_k_vertex_redundantly_rigid(k, dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert redundant_rigidity.is_k_vertex_redundantly_rigid(
            nx.Graph(graph), k, dim=2, algorithm=algorithm
        )


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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d2)
def test_is_not_k_vertex_redundantly_rigid_d2(graph, k, algorithm):
    assert not graph.is_k_vertex_redundantly_rigid(k, dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not redundant_rigidity.is_k_vertex_redundantly_rigid(
            nx.Graph(graph), k, dim=2, algorithm=algorithm
        )


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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_all_d)
def test_is_k_vertex_redundantly_rigid_d3(graph, k, algorithm):
    assert graph.is_k_vertex_redundantly_rigid(k, dim=3, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert redundant_rigidity.is_k_vertex_redundantly_rigid(
            nx.Graph(graph), k, dim=3, algorithm=algorithm
        )


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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_all_d)
def test_is_not_k_vertex_redundantly_rigid_d3(graph, k, algorithm):
    assert not graph.is_k_vertex_redundantly_rigid(k, dim=3, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not redundant_rigidity.is_k_vertex_redundantly_rigid(
            nx.Graph(graph), k, dim=3, algorithm=algorithm
        )


###############################################################
# is_min_k_vertex_redundantly_rigid
###############################################################
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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d1)
def test_is_min_k_vertex_redundantly_rigid_d1(graph, k, algorithm):
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert redundant_rigidity.is_min_k_vertex_redundantly_rigid(
            nx.Graph(graph), k, dim=1, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(101739), 1],
        [Graph.from_int(223), 1],
        [graphs.CompleteMinusOne(5), 2],
        [Graph.from_int(16351), 3],
    ],
)
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d1)
def test_is_not_min_k_vertex_redundantly_rigid_d1(graph, k, algorithm):
    assert not graph.is_min_k_vertex_redundantly_rigid(k, dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not redundant_rigidity.is_min_k_vertex_redundantly_rigid(
            nx.Graph(graph), k, dim=1, algorithm=algorithm
        )


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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d2)
def test_is_min_k_vertex_redundantly_rigid_d2(graph, k, algorithm):
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert redundant_rigidity.is_min_k_vertex_redundantly_rigid(
            nx.Graph(graph), k, dim=2, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.ThreePrism(), 1],
        [Graph.from_int(8191), 1],
        [Graph.from_int(32767), 2],
        [Graph.from_int(2097151), 3],
    ],
)
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d2)
def test_is_not_min_k_vertex_redundantly_rigid_d2(graph, k, algorithm):
    assert not graph.is_min_k_vertex_redundantly_rigid(k, dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not redundant_rigidity.is_min_k_vertex_redundantly_rigid(
            nx.Graph(graph), k, dim=2, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(507903), 1],
        [Graph.from_int(1048575), 2],
    ],
)
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_all_d)
def test_is_min_k_vertex_redundantly_rigid_d3(graph, k, algorithm):
    assert graph.is_min_k_vertex_redundantly_rigid(k, dim=3, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert redundant_rigidity.is_min_k_vertex_redundantly_rigid(
            nx.Graph(graph), k, dim=3, algorithm=algorithm
        )


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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_all_d)
def test_is_not_min_k_vertex_redundantly_rigid_d3(graph, k, algorithm):
    assert not graph.is_min_k_vertex_redundantly_rigid(k, dim=3, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not redundant_rigidity.is_min_k_vertex_redundantly_rigid(
            nx.Graph(graph), k, dim=3, algorithm=algorithm
        )


###############################################################
# is_redundantly_rigid
###############################################################
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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d2)
def test_is_redundantly_rigid_d2(graph, algorithm):
    assert graph.is_redundantly_rigid(dim=2, algorithm=algorithm)


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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d2)
def test_is_not_redundantly_rigid_d2(graph, algorithm):
    assert not graph.is_redundantly_rigid(dim=2, algorithm=algorithm)


###############################################################
# is_k_redundantly_rigid
###############################################################
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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d1)
def test_is_k_redundantly_rigid_d1(graph, k, algorithm):
    assert graph.is_k_redundantly_rigid(k, dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert redundant_rigidity.is_k_redundantly_rigid(
            nx.Graph(graph), k, dim=1, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph, k",
    [
        [Graph.from_int(15), 1],
        [graphs.Diamond(), 2],
        [Graph([["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["a", "c"]]), 3],
    ],
)
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d1)
def test_is_not_k_redundantly_rigid_d1(graph, k, algorithm):
    assert not graph.is_k_redundantly_rigid(k, dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not redundant_rigidity.is_k_redundantly_rigid(
            nx.Graph(graph), k, dim=1, algorithm=algorithm
        )


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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d2)
def test_is_k_redundantly_rigid_d2(graph, k, algorithm):
    assert graph.is_k_redundantly_rigid(k, dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert redundant_rigidity.is_k_redundantly_rigid(
            nx.Graph(graph), k, dim=2, algorithm=algorithm
        )


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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d2)
def test_is_not_k_redundantly_rigid_d2(graph, k, algorithm):
    assert not graph.is_k_redundantly_rigid(k, dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not redundant_rigidity.is_k_redundantly_rigid(
            nx.Graph(graph), k, dim=2, algorithm=algorithm
        )


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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_all_d)
def test_is_k_redundantly_rigid_d3(graph, k, algorithm):
    assert graph.is_k_redundantly_rigid(k, dim=3, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert redundant_rigidity.is_k_redundantly_rigid(
            nx.Graph(graph), k, dim=3, algorithm=algorithm
        )


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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_all_d)
def test_is_not_k_redundantly_rigid_d3(graph, k, algorithm):
    assert not graph.is_k_redundantly_rigid(k, dim=3, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not redundant_rigidity.is_k_redundantly_rigid(
            nx.Graph(graph), k, dim=3, algorithm=algorithm
        )


###############################################################
# is_min_k_redundantly_rigid
###############################################################
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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d1)
def test_is_min_k_redundantly_rigid_d1(graph, k, algorithm):
    assert graph.is_min_k_redundantly_rigid(k, dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert redundant_rigidity.is_min_k_redundantly_rigid(
            nx.Graph(graph), k, dim=1, algorithm=algorithm
        )


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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d1)
def test_is_not_min_k_redundantly_rigid_d1(graph, k, algorithm):
    assert not graph.is_min_k_redundantly_rigid(k, dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not redundant_rigidity.is_min_k_redundantly_rigid(
            nx.Graph(graph), k, dim=1, algorithm=algorithm
        )


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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d2)
def test_is_min_k_redundantly_rigid_d2(graph, k, algorithm):
    assert graph.is_min_k_redundantly_rigid(k, dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert redundant_rigidity.is_min_k_redundantly_rigid(
            nx.Graph(graph), k, dim=2, algorithm=algorithm
        )


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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d2)
def test_is_not_min_k_redundantly_rigid_d2(graph, k, algorithm):
    assert not graph.is_min_k_redundantly_rigid(k, dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not redundant_rigidity.is_min_k_redundantly_rigid(
            nx.Graph(graph), k, dim=2, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Complete(5), 1],
        [Graph.from_int(16351), 1],
        [Graph.from_int(32767), 2],
    ],
)
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_all_d)
def test_is_min_k_redundantly_rigid_d3(graph, k, algorithm):
    assert graph.is_min_k_redundantly_rigid(k, dim=3, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert redundant_rigidity.is_min_k_redundantly_rigid(
            nx.Graph(graph), k, dim=3, algorithm=algorithm
        )


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
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_all_d)
def test_is_not_min_k_redundantly_rigid_d3(graph, k, algorithm):
    assert not graph.is_min_k_redundantly_rigid(k, dim=3, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not redundant_rigidity.is_min_k_redundantly_rigid(
            nx.Graph(graph), k, dim=3, algorithm=algorithm
        )
