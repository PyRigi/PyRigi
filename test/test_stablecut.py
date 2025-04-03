from collections import defaultdict
from itertools import product
import logging
import random
import networkx as nx
from pyrigi.data_type import Vertex
import pytest
import numpy as np

from pyrigi import Graph
from pyrigi.separating_set import _revertable_set_removal


def _eq(g1: Graph, g2: nx.Graph):
    if g1 != g2:
        return False
    for v in g1.nodes:
        if g1.nodes[v] != g2.nodes[v]:
            return False
    for u, v in g1.edges:
        if g1.edges[u, v] != g2.edges[u, v]:
            return False
    return True


def _add_metadata(graph: nx.Graph):
    for v in graph.nodes:
        graph.nodes[v]["test_prop"] = v
    for u, v in graph.edges:
        graph.edges[u, v]["test_prop"] = (u, v)


def test_is_stable_set():
    graph = nx.Graph([(0, 1), (1, 2), (2, 3)])

    assert not Graph.is_stable_set(graph, {0, 1, 2}, certificate=False)
    assert Graph.is_stable_set(graph, {0, 1, 2}, certificate=True) in [
        (False, (0, 1)),
        (False, (1, 2)),
    ]
    assert Graph.is_stable_set(graph, {0, 1}, certificate=True) == (False, (0, 1))
    assert Graph.is_stable_set(graph, {0, 2}, certificate=True) == (True, None)


def test__revertable_set_removal():
    graph1 = Graph([(0, 1), (1, 2), (2, 3)])
    _add_metadata(graph1)
    graph2 = graph1.copy()

    def noop(_: nx.Graph):
        pass

    _revertable_set_removal(graph2, set(), noop)
    assert _eq(graph1, graph2)
    _revertable_set_removal(graph2, {2}, noop)
    assert _eq(graph1, graph2)
    _revertable_set_removal(graph2, {0, 1, 2, 3}, noop)
    assert _eq(graph1, graph2)
    _revertable_set_removal(nx.induced_subgraph(graph2, [0, 1, 2]), {1}, noop)
    assert _eq(graph1, graph2)


def test_is_separating_set():
    graph = Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)])

    assert graph.is_separating_set({2})
    assert graph.is_separating_set({0, 2})
    assert graph.is_separating_set({0, 1})
    assert not graph.is_separating_set(set())
    assert not graph.is_separating_set({4, 3})

    assert Graph.is_uv_separating_set(graph, {2}, 0, 3)
    assert Graph.is_uv_separating_set(graph, {2}, 4, 3)
    assert not Graph.is_uv_separating_set(graph, {2}, 0, 1)

    with pytest.raises(ValueError):
        Graph.is_uv_separating_set(graph, {2}, 0, 2)


def test_stable_separating_set_edge_cases():
    # empty graph
    graph = Graph()
    _add_metadata(graph)
    orig = graph.copy()

    with pytest.raises(ValueError):
        Graph.stable_separating_set(graph)
    assert _eq(graph, orig)

    # more vertices
    graph = Graph.from_vertices_and_edges([0, 1, 2], [])
    _add_metadata(graph)
    orig = graph.copy()
    cut = graph.stable_separating_set()
    assert graph.is_stable_separating_set(cut)
    assert _eq(graph, orig)

    # single vertex graph
    graph = Graph.from_vertices_and_edges([0], [])
    _add_metadata(graph)
    orig = graph.copy()
    with pytest.raises(ValueError):
        graph.stable_separating_set()
    assert _eq(graph, orig)

    # single edge graph
    graph = Graph.from_vertices_and_edges([0, 1], [(0, 1)])
    _add_metadata(graph)
    orig = graph.copy()
    with pytest.raises(ValueError):
        graph.stable_separating_set()
    assert _eq(graph, orig)

    # triangle graph
    graph = Graph.from_vertices_and_edges([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
    _add_metadata(graph)
    orig = graph.copy()
    with pytest.raises(ValueError):
        graph.stable_separating_set()
    assert _eq(graph, orig)


def test_stable_separating_set():
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

    cut = graph.stable_separating_set()
    assert Graph.is_stable_separating_set(graph, cut)
    assert _eq(graph, orig)

    cut = graph.stable_separating_set(0)
    assert Graph.is_stable_separating_set(graph, cut)
    assert _eq(graph, orig)

    with pytest.raises(ValueError):
        graph.stable_separating_set(0, 1)
    assert _eq(graph, orig)

    with pytest.raises(ValueError):
        graph.stable_separating_set(0, 4)
    assert _eq(graph, orig)

    for i in [5, 6, 7]:
        cut = graph.stable_separating_set(0, i)
        assert Graph.is_stable_separating_set(graph, cut)
        assert _eq(graph, orig)
    for i in [0, 1, 2]:
        cut = graph.stable_separating_set(7, i)
        assert Graph.is_stable_separating_set(graph, cut)
        assert _eq(graph, orig)


def test_stable_separating_set_prism():
    from pyrigi.graphDB import ThreePrism

    graph = ThreePrism()

    for u, v in product(graph.nodes, graph.nodes):
        with pytest.raises(ValueError):
            graph.stable_separating_set(u, v)


@pytest.mark.slow_main
@pytest.mark.parametrize(
    ("n", "p"), [(4, 0.3), (8, 0.3), (13, 0.3), (16, 0.2), (16, 0.3)]
)
@pytest.mark.parametrize("graph_no", [69])
@pytest.mark.parametrize("seed", [42, None])
@pytest.mark.parametrize("connected", [True, False])
def test_fuzzy_stable_separating_set(
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
        if nx.is_connected(graph) != connected or graph.is_rigid(dim=2):
            continue

        # Create mapping from vertex to rigid component ids
        # used later to assert expected result
        rigid_components = graph.rigid_components()
        vertex_to_comp_id: dict[Vertex, set[int]] = defaultdict(set)
        for i, comp in enumerate(rigid_components):
            for v in comp:
                vertex_to_comp_id[v].add(i)

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
                    with pytest.raises(ValueError):
                        Graph.stable_separating_set(graph, u, v)
                    logging.disable(0)

                    tests_negative += 1
                else:
                    # valid input
                    cut = Graph.stable_separating_set(graph, u, v)
                    assert Graph.is_uv_separating_set(graph, cut, u, v)
                    assert Graph.is_stable_set(graph, cut)
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
