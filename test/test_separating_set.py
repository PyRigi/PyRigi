from collections import defaultdict
from itertools import product
import logging
import random

import networkx as nx
import numpy as np
import pytest

import pyrigi.graphDB as graphs
from pyrigi import Graph
from pyrigi.data_type import Vertex
from pyrigi.separating_set import _revertable_set_removal
from test_graph import relabeled_inc


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


@pytest.mark.parametrize(
    "graph, stable_set",
    [
        [graphs.Complete(2), [0]],
        [graphs.Complete(3), []],
        [graphs.CompleteBipartite(3, 3), [0, 1]],
        [graphs.CompleteBipartite(3, 4), [0, 1, 2]],
        [graphs.CompleteBipartite(4, 4), [4, 5, 6, 7]],
        [graphs.Diamond(), [1, 3]],
        [graphs.ThreePrism(), [0, 4]],
        [Graph([(0, 1), (2, 3)]), [0, 2]],
    ],
)
def test_is_stable_set(graph, stable_set):
    assert graph.is_stable_set(stable_set)


@pytest.mark.parametrize(
    "graph, stable_set",
    [
        [graphs.Complete(2), [0, 1]],
        [graphs.Complete(3), [0, 1]],
        [graphs.Complete(4), [2, 3]],
        [graphs.CompleteBipartite(3, 3), [0, 3, 4]],
        [graphs.CompleteBipartite(3, 4), [0, 3, 4, 5]],
        [graphs.Diamond(), [0, 2]],
        [graphs.K33plusEdge(), [0, 1, 5]],
        [Graph([(0, 1), (2, 3)]), [3, 2]],
    ],
)
def test_is_not_stable_set(graph, stable_set):
    assert not graph.is_stable_set(stable_set)


def test_is_stable_set_certificate():
    graph = nx.Graph([(0, 1), (1, 2), (2, 3)])

    assert Graph.is_stable_set(graph, {0, 1, 2}, certificate=True) in [
        (False, (0, 1)),
        (False, (1, 2)),
    ]
    assert Graph.is_stable_set(graph, {0, 1}, certificate=True) == (False, (0, 1))
    assert Graph.is_stable_set(graph, {0, 2}, certificate=True) == (True, None)


def test__revertable_set_removal():
    graph1 = graphs.Complete(4)
    _add_metadata(graph1)
    graph2 = graph1.copy()

    def noop(_: nx.Graph):
        pass

    _revertable_set_removal(graph2, set(), noop, use_copy=True)
    assert _eq(graph1, graph2)
    _revertable_set_removal(graph2, set(), noop, use_copy=False)
    assert _eq(graph1, graph2)
    _revertable_set_removal(graph2, {2}, noop, use_copy=False)
    assert _eq(graph1, graph2)
    _revertable_set_removal(graph2, {2}, noop, use_copy=True)
    assert _eq(graph1, graph2)
    _revertable_set_removal(graph2, {0, 1, 2, 3}, noop, use_copy=False)
    assert _eq(graph1, graph2)
    _revertable_set_removal(
        nx.induced_subgraph(graph2, [0, 1, 2]), {1}, noop, use_copy=False
    )
    assert _eq(graph1, graph2)


@pytest.mark.parametrize(
    "graph, separating_set",
    [
        [Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)]), {2}],
        [Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)]), {0, 2}],
        [Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)]), {0, 1}],
        [graphs.CompleteBipartite(3, 4), [0, 1, 2]],
        [graphs.CompleteBipartite(4, 4), [0, 4, 5, 6, 7]],
        [graphs.Diamond(), [0, 2]],
        [graphs.ThreePrism(), [0, 4, 5]],
        [Graph([(0, 1), (2, 3)]), [0, 2]],
        [Graph([(0, 1), (2, 3)]), [0]],
        [Graph([(0, 1), (2, 3)]), []],
    ],
)
def test_is_separating_set(graph, separating_set):
    assert graph.is_separating_set(separating_set)


@pytest.mark.parametrize(
    "graph, separating_set",
    [
        [Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)]), set()],
        [Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)]), {4, 3}],
        [Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)]), {1}],
        [graphs.Complete(2), [0]],
        [graphs.Complete(3), []],
        [graphs.CompleteBipartite(3, 3), [0, 1]],
        [graphs.Diamond(), [1, 3]],
        [graphs.ThreePrism(), [0, 4]],
        [graphs.Complete(3), [0, 1]],
        [graphs.Complete(4), [2, 3]],
        [graphs.CompleteBipartite(3, 3), [0, 3, 4]],
        [graphs.CompleteBipartite(3, 4), [0, 3, 4, 5]],
        [graphs.K33plusEdge(), [0, 1, 5]],
    ],
)
def test_is_not_separating_set(graph, separating_set):
    assert not graph.is_separating_set(separating_set)


def test_is_separating_set_error():
    with pytest.raises(ValueError):
        graphs.Complete(2).is_separating_set([0, 1])


@pytest.mark.parametrize(
    "graph, separating_set, u, v",
    [
        [Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)]), {2}, 0, 3],
        [Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)]), {2}, 4, 3],
        [Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)]), {0, 2}, 1, 3],
        [Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)]), {0, 1}, 2, 4],
        [graphs.CompleteBipartite(3, 4), [0, 1, 2], 3, 4],
        [graphs.Diamond(), [0, 2], 1, 3],
        [graphs.ThreePrism(), [0, 4, 5], 1, 3],
        [Graph([(0, 1), (2, 3)]), [0, 2], 1, 3],
        [Graph([(0, 1), (2, 3)]), [0], 1, 3],
        [Graph([(0, 1), (2, 3)]), [], 1, 3],
    ],
)
def test_is_uv_separating_set(graph, separating_set, u, v):
    assert graph.is_uv_separating_set(separating_set, u, v)


@pytest.mark.parametrize(
    "graph, separating_set, u, v",
    [
        [Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)]), {2}, 0, 1],
        [Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)]), {4, 3}, 0, 1],
        [graphs.Complete(3), [], 1, 2],
        [graphs.CompleteBipartite(3, 3), [0, 1], 4, 5],
        [graphs.Diamond(), [1, 3], 0, 2],
        [graphs.ThreePrism(), [0, 4], 1, 3],
        [graphs.Complete(4), [2, 3], 0, 1],
        [graphs.CompleteBipartite(3, 4), [0, 3, 4, 5], 1, 2],
        [graphs.K33plusEdge(), [0, 1, 5], 3, 4],
    ],
)
def test_is_not_uv_separating_set(graph, separating_set, u, v):
    assert not graph.is_uv_separating_set(separating_set, u, v)


def test_is_uv_separating_set_error():
    with pytest.raises(ValueError):
        Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)]).is_uv_separating_set(
            {2}, 0, 2
        )
    with pytest.raises(ValueError):
        graphs.CompleteBipartite(4, 4).is_uv_separating_set([0, 4, 5, 6, 7], 0, 2)
    with pytest.raises(ValueError):
        graphs.Complete(2).is_uv_separating_set([0], 1, 2)


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
    graph = graphs.Complete(3)
    _add_metadata(graph)
    orig = graph.copy()
    with pytest.raises(ValueError):
        graph.stable_separating_set()
    assert _eq(graph, orig)


@pytest.mark.parametrize(
    "graph, one_chosen_vertex, two_chosen_vertices",
    [
        [graphs.Cycle(4), [0, 1], [[0, 2], [1, 3]]],
        [graphs.Cycle(5), [0, 2], [[0, 2], [0, 3]]],
        [graphs.Path(3), [0], [[0, 2]]],
        [graphs.Path(4), [0, 1], [[0, 2], [0, 3]]],
        [graphs.Grid(2, 4), [2, 7], [[0, 2], [0, 6]]],
        [graphs.Grid(3, 4), [0, 1, 4, 5], [[0, 11], [1, 10]]],
        [graphs.CompleteBipartite(2, 4), [0, 5], [[0, 1], [2, 3]]],
        [graphs.CompleteBipartite(2, 3), [0, 2], [[0, 1], [2, 3]]],
        [graphs.Complete(3) + relabeled_inc(graphs.Complete(3), 2), [0], [[0, 3]]],
        [
            graphs.Complete(3) + relabeled_inc(graphs.Complete(3), 3) + Graph([(2, 3)]),
            [0, 2],
            [[0, 3], [1, 5]],
        ],
        [
            graphs.CompleteMinusOne(5) + relabeled_inc(graphs.Complete(5), 4),
            [1, 2, 3, 8],
            [[1, 5], [3, 8]],
        ],
        [
            graphs.Complete(4) + relabeled_inc(graphs.Complete(5), 4) + Graph([(3, 4)]),
            [0, 3, 8],
            [[3, 6], [2, 4]],
        ],
    ],
)
def test_stable_separating_set(graph, one_chosen_vertex, two_chosen_vertices):
    orig = graph.copy()

    cut = graph.stable_separating_set()
    assert graph.is_stable_separating_set(cut)
    assert _eq(graph, orig)

    for vertex in one_chosen_vertex:
        cut = graph.stable_separating_set(vertex)
        assert graph.is_stable_separating_set(cut)
        assert _eq(graph, orig)

    for u, v in two_chosen_vertices:
        cut = graph.stable_separating_set(u, v)
        assert graph.is_stable_separating_set(cut)
        assert _eq(graph, orig)


@pytest.mark.parametrize(
    "graph, separating_vertex",
    [
        [graphs.Path(3), 1],
        [graphs.Complete(3) + relabeled_inc(graphs.Complete(3), 2), 2],
        [graphs.CompleteMinusOne(5) + relabeled_inc(graphs.Complete(5), 4), 4],
    ],
)
def test_stable_separating_set_error_single_vertex_separating_set(
    graph, separating_vertex
):
    """
    Test that an error is raised when a vertex that is a unique separating set
    is asked to be avoided.
    """
    with pytest.raises(ValueError):
        graph.stable_separating_set(separating_vertex)


def test_stable_separating_set_2by4_Grid():
    graph = graphs.Grid(2, 4)
    orig = graph.copy()

    with pytest.raises(ValueError):
        graph.stable_separating_set(0, 1)
    assert _eq(graph, orig)

    with pytest.raises(ValueError):
        graph.stable_separating_set(1, 2)
    assert _eq(graph, orig)

    for i in [3, 6, 7]:
        cut = graph.stable_separating_set(1, i)
        assert graph.is_stable_separating_set(cut)
        assert _eq(graph, orig)
    for i in [0, 1, 4]:
        cut = graph.stable_separating_set(6, i)
        assert graph.is_stable_separating_set(cut)
        assert _eq(graph, orig)


def test_stable_separating_set_prism():
    graph = graphs.ThreePrism()

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
