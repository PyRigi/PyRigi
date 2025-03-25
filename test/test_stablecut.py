from collections import defaultdict
from itertools import product
import logging
import random
import networkx as nx
from pyrigi.data_type import Vertex
import pytest
import numpy as np

from pyrigi import Graph
from pyrigi.data_type import StableSeparatingCut
from pyrigi._cuts import _revertable_set_removal


def test_stable_set_eq():
    set1 = StableSeparatingCut({1, 2}, {3, 4}, {5})
    set2 = StableSeparatingCut({3, 4}, {1, 2}, {5})
    set3 = StableSeparatingCut({3, 4}, {1, 2}, {6})
    assert set1 == set1
    assert set1 == set2
    assert set1 != set3


def test_is_stable_set():
    graph = nx.Graph([(0, 1), (1, 2), (2, 3)])

    assert Graph.is_stable_set(graph, {0, 1, 2}) in [(False, (0, 1)), (False, (1, 2))]
    assert Graph.is_stable_set(graph, {0, 1}) == (False, (0, 1))
    assert Graph.is_stable_set(graph, {0, 2}) is (True, None)


def test__revertable_set_removal():
    graph1 = nx.Graph([(0, 1), (1, 2), (2, 3)])
    graph2 = graph1.copy()

    def noop(_: nx.Graph):
        pass

    _revertable_set_removal(graph2, set(), noop)
    assert nx.is_isomorphic(graph1, graph2)
    _revertable_set_removal(graph2, {2}, noop)
    assert nx.is_isomorphic(graph1, graph2)
    _revertable_set_removal(graph2, {0, 1, 2, 3}, noop)
    assert nx.is_isomorphic(graph1, graph2)
    _revertable_set_removal(nx.induced_subgraph(graph2, [0, 1, 2]), {1}, noop)
    assert nx.is_isomorphic(graph1, graph2)


def test_is_separating_set():
    graph = nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)])

    assert Graph.is_separating_set(graph, {2})
    assert Graph.is_separating_set(graph, {0, 2})
    assert Graph.is_separating_set(graph, {0, 1})
    assert not Graph.is_separating_set(graph, set())
    assert not Graph.is_separating_set(graph, {4, 3})

    assert Graph.is_separating_set_dividing(graph, {2}, 0, 3)
    assert Graph.is_separating_set_dividing(graph, {2}, 4, 3)
    assert not Graph.is_separating_set_dividing(graph, {2}, 0, 1)

    with pytest.raises(ValueError):
        Graph.is_separating_set_dividing(graph, {2}, 0, 2)


def test_stable_separating_cut_in_flexible_graph_edge_cases():
    # empty graph
    graph = nx.Graph()
    orig = graph.copy()

    cut = Graph.stable_separating_set_in_flexible_graph(graph)
    assert cut is None
    assert nx.is_isomorphic(graph, orig)

    # more vertices
    graph = Graph.from_vertices_and_edges([0, 1, 2], [])
    orig = graph.copy()
    cut = Graph.stable_separating_set_in_flexible_graph(graph)
    assert cut is not None
    assert Graph.is_stable_cutset(graph, cut)
    assert nx.is_isomorphic(graph, orig)

    # single vertex graph
    graph = Graph.from_vertices_and_edges([0], [])
    orig = graph.copy()
    cut = Graph.stable_separating_set_in_flexible_graph(graph)
    assert cut is None
    assert nx.is_isomorphic(graph, orig)

    # single edge graph
    graph = Graph.from_vertices_and_edges([0, 1], [(0, 1)])
    orig = graph.copy()
    cut = Graph.stable_separating_set_in_flexible_graph(graph)
    assert cut is None
    assert nx.is_isomorphic(graph, orig)

    # triangle graph
    graph = Graph.from_vertices_and_edges([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
    orig = graph.copy()
    cut = Graph.stable_separating_set_in_flexible_graph(graph)
    assert cut is None
    assert nx.is_isomorphic(graph, orig)


def test_stable_separating_cut_in_flexible_graph():
    from pyrigi import Graph as Graph

    graph = Graph.from_vertices_and_edges(
        list(range(8)),
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (3, 7),
        ],
    )
    orig = graph.copy()

    assert not graph.is_rigid()

    cut = Graph.stable_separating_set_in_flexible_graph(graph)
    assert cut is not None
    assert Graph.is_stable_cutset(graph, cut)
    assert nx.is_isomorphic(graph, orig)

    cut = Graph.stable_separating_set_in_flexible_graph(graph, 0)
    assert cut is not None
    assert Graph.is_stable_cutset(graph, cut)
    assert nx.is_isomorphic(graph, orig)

    cut = Graph.stable_separating_set_in_flexible_graph(graph, 0, 1)
    assert cut is None
    assert nx.is_isomorphic(graph, orig)

    cut = Graph.stable_separating_set_in_flexible_graph(graph, 0, 4)
    assert cut is None
    assert nx.is_isomorphic(graph, orig)

    for i in [5, 6, 7]:
        cut = Graph.stable_separating_set_in_flexible_graph(graph, 0, i)
        assert cut is not None
        assert Graph.is_stable_cutset(graph, cut)
        assert nx.is_isomorphic(graph, orig)
    for i in [0, 1, 2]:
        cut = Graph.stable_separating_set_in_flexible_graph(graph, 7, i)
        assert cut is not None
        assert Graph.is_stable_cutset(graph, cut)
        assert nx.is_isomorphic(graph, orig)


def test_stable_separating_cut_in_flexible_graph_prism():
    from pyrigi.graphDB import ThreePrism

    graph = ThreePrism()
    orig = graph.copy()

    for u, v in product(graph.nodes, graph.nodes):
        cut = Graph.stable_separating_set_in_flexible_graph(graph, u, v)
        assert cut is None
        assert nx.is_isomorphic(graph, orig)


@pytest.mark.slow_main
@pytest.mark.parametrize(
    ("n", "p"), [(4, 0.3), (8, 0.3), (13, 0.4), (16, 0.4), (16, 0.4)]
)
@pytest.mark.parametrize("graph_no", [69])
@pytest.mark.parametrize("seed", [42, None])
@pytest.mark.parametrize("connected", [True, False])
def test_fuzzy_stable_separating_cut_in_flexible_graph(
    n: int,
    p: float,
    graph_no: int,
    seed: int | None,
    connected: bool,
):
    from pyrigi import Graph

    pairs_per_graph = int(np.sqrt(n))

    rand = random.Random(seed)
    graphs = 0
    while graphs < graph_no:
        graph = Graph(nx.gnp_random_graph(n, p, seed=rand.randint(0, 2**30)))

        # Filter out unreasonable graphs
        if nx.is_connected(graph) != connected or graph.is_rigid():
            continue

        # Create mapping from vertex to rigid component ids
        # used later to assert expected result
        rigid_components = graph.rigid_components()
        vertex_to_comp_id: dict[Vertex, set[int]] = defaultdict(set)
        for i, comp in enumerate(rigid_components):
            for v in comp:
                vertex_to_comp_id[v].add(i)

        # Makes sure that the underlying code does not require pyrigi.Graph
        graph = nx.Graph(graph)

        # Take first pairs_per_graph pairs of vertices
        pairs = list(product(graph.nodes, graph.nodes))
        rand.shuffle(pairs)

        tests_negative = 0
        tests_positive = 0
        for pair in pairs:
            u, v = pair

            # makes sure both types of tests are run enough
            if (
                tests_positive > pairs_per_graph * 2 // 3
                and tests_negative > pairs_per_graph * 1 // 3
            ):
                break

            try:
                if u == v or (vertex_to_comp_id[u] & vertex_to_comp_id[v]):
                    # invalid input
                    logging.disable(logging.WARNING)
                    cut = Graph.stable_separating_set_in_flexible_graph(graph, u, v)
                    logging.disable(0)

                    assert cut is None
                    tests_negative += 1
                else:
                    # valid input
                    cut = Graph.stable_separating_set_in_flexible_graph(graph, u, v)
                    assert cut is not None
                    assert Graph.is_stable_separating_set_dividing(graph, cut, u, v)
                    tests_positive += 1

            except AssertionError as e:
                # Changes the error message to include graph
                error_message = (
                    f"Assertion failed: {e}\n"
                    + f"Graph: {nx.nx_agraph.to_agraph(graph)}\n"
                    + f"Components: {rigid_components}\n"
                    + f"Vertices: [{u} {v}]"
                )
                raise AssertionError(error_message) from None

        graphs += 1
