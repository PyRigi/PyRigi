import math
from itertools import product

import networkx as nx
import pytest

import pyrigi.apex as apex
import pyrigi.graphDB as graphs
from pyrigi.graph import Graph
from test_graph import TEST_WRAPPED_FUNCTIONS


###############################################################
# is_k_vertex_apex
###############################################################
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
    if TEST_WRAPPED_FUNCTIONS:
        assert apex.is_k_vertex_apex(nx.Graph(graph), k)


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
    if TEST_WRAPPED_FUNCTIONS:
        assert not apex.is_k_vertex_apex(nx.Graph(graph), k)


###############################################################
# is_k_edge_apex
###############################################################
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
    if TEST_WRAPPED_FUNCTIONS:
        assert apex.is_k_edge_apex(nx.Graph(graph), k)


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
    if TEST_WRAPPED_FUNCTIONS:
        assert not apex.is_k_edge_apex(nx.Graph(graph), k)


###############################################################
# is_critically_k_vertex_apex
###############################################################
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
    if TEST_WRAPPED_FUNCTIONS:
        assert apex.is_critically_k_vertex_apex(nx.Graph(graph), k)


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
    if TEST_WRAPPED_FUNCTIONS:
        assert not apex.is_critically_k_vertex_apex(nx.Graph(graph), k)


###############################################################
# is_critically_k_edge_apex
###############################################################
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
    if TEST_WRAPPED_FUNCTIONS:
        assert apex.is_critically_k_edge_apex(nx.Graph(graph), k)


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
    if TEST_WRAPPED_FUNCTIONS:
        assert not apex.is_critically_k_edge_apex(nx.Graph(graph), k)


###############################################################
# randomized test
###############################################################
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
