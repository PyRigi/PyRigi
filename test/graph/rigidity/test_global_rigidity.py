from itertools import combinations

import networkx as nx
import pytest

import pyrigi.graph.rigidity.global_ as global_rigidity
import pyrigi.graphDB as graphs
from pyrigi.graph import Graph
from test import TEST_WRAPPED_FUNCTIONS
from test.graph.test_graph import read_globally, read_redundantly

###############################################################
# is_globally_rigid
###############################################################


# Examples of globally rigid graphs taken from:
# Grasegger, G. (2022). Dataset of globally rigid graphs [Data set].
# Zenodo. https://doi.org/10.5281/zenodo.7473052
@pytest.mark.parametrize(
    "graph, dim",
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
def test_is_globally_rigid(graph, dim):
    assert graph.is_globally_rigid(dim=dim)
    if TEST_WRAPPED_FUNCTIONS:
        assert global_rigidity.is_globally_rigid(nx.Graph(graph), dim=dim)


@pytest.mark.parametrize(
    "graph, dim",
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
def test_is_not_globally_rigid(graph, dim):
    assert not graph.is_globally_rigid(dim=dim)
    if TEST_WRAPPED_FUNCTIONS:
        assert not global_rigidity.is_globally_rigid(nx.Graph(graph), dim=dim)


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
    if TEST_WRAPPED_FUNCTIONS:
        assert global_rigidity.is_globally_rigid(nx.Graph(graph), dim=2)


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
    if TEST_WRAPPED_FUNCTIONS:
        assert not global_rigidity.is_globally_rigid(nx.Graph(graph), dim=2)


###############################################################
# is_weakly_globally_linked
###############################################################
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
    assert graph.is_weakly_globally_linked(u, v, dim=2)
    if TEST_WRAPPED_FUNCTIONS:
        assert global_rigidity.is_weakly_globally_linked(nx.Graph(graph), u, v, dim=2)
