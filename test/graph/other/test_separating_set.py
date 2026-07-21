import math
import random
from collections import defaultdict
from itertools import product

import networkx as nx
import pytest

import pyrigi.graph._other.separating_set as separating_set
import pyrigi.graph._rigidity.generic as generic_rigidity
import pyrigi.graphDB as graphs
from pyrigi import Graph
from pyrigi.data_type import Vertex
from pyrigi.graph._other.separating_set import _remove_apply_restore_vertices
from test.graph.test_graph import relabeled_inc


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
    assert separating_set.is_stable_set(nx.Graph(graph), stable_set)


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
    assert not separating_set.is_stable_set(nx.Graph(graph), stable_set)


def test_is_stable_set_certificate():
    graph = nx.Graph(graphs.Path(4))

    assert separating_set.is_stable_set(graph, {0, 1, 2}, certificate=True) in [
        (False, (0, 1)),
        (False, (1, 2)),
    ]
    assert separating_set.is_stable_set(graph, {0, 1}, certificate=True) == (
        False,
        (0, 1),
    )
    assert separating_set.is_stable_set(graph, {0, 2}, certificate=True) == (True, None)


def test__remove_apply_restore_vertices():
    graph1 = graphs.Complete(4)
    _add_metadata(graph1)
    graph2 = graph1.copy()

    def noop(_: nx.Graph):  # noqa: U101
        pass

    _remove_apply_restore_vertices(graph2, set(), noop, use_copy=True)
    assert _eq(graph1, graph2)
    _remove_apply_restore_vertices(graph2, set(), noop, use_copy=False)
    assert _eq(graph1, graph2)
    _remove_apply_restore_vertices(graph2, {2}, noop, use_copy=False)
    assert _eq(graph1, graph2)
    _remove_apply_restore_vertices(graph2, {2}, noop, use_copy=True)
    assert _eq(graph1, graph2)
    _remove_apply_restore_vertices(graph2, {0, 1, 2, 3}, noop, use_copy=False)
    assert _eq(graph1, graph2)
    _remove_apply_restore_vertices(
        nx.induced_subgraph(graph2, [0, 1, 2]), {1}, noop, use_copy=False
    )
    assert _eq(graph1, graph2)


@pytest.mark.parametrize(
    "graph, sep_set",
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
def test_is_separating_set(graph, sep_set):
    assert separating_set.is_separating_set(nx.Graph(graph), sep_set)


@pytest.mark.parametrize(
    "graph, sep_set",
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
def test_is_not_separating_set(graph, sep_set):
    assert not separating_set.is_separating_set(nx.Graph(graph), sep_set)


def test_is_separating_set_error():
    with pytest.raises(ValueError):
        separating_set.is_separating_set(nx.Graph(graphs.Complete(2)), [0, 1])


@pytest.mark.parametrize(
    "graph, sep_set, u, v",
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
def test_is_uv_separating_set(graph, sep_set, u, v):
    assert separating_set.is_uv_separating_set(nx.Graph(graph), sep_set, u, v)


@pytest.mark.parametrize(
    "graph, sep_set, u, v",
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
def test_is_not_uv_separating_set(graph, sep_set, u, v):
    assert not separating_set.is_uv_separating_set(nx.Graph(graph), sep_set, u, v)


def test_is_uv_separating_set_error():
    with pytest.raises(ValueError):
        separating_set.is_uv_separating_set(
            nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)]),
            {2},
            0,
            2,
        )
    with pytest.raises(ValueError):
        separating_set.is_uv_separating_set(
            nx.Graph(graphs.CompleteBipartite(4, 4)), [0, 4, 5, 6, 7], 0, 2
        )
    with pytest.raises(ValueError):
        separating_set.is_uv_separating_set(nx.Graph(graphs.Complete(2)), [0], 1, 2)


def test_stable_separating_set_edge_cases():
    graph = nx.Graph(Graph.from_vertices_and_edges([0, 1, 2], []))
    _add_metadata(graph)
    orig = Graph(graph.copy())
    cut = separating_set.stable_separating_set(graph)
    assert separating_set.is_stable_separating_set(graph, cut)
    assert _eq(orig, graph)

    # single vertex graph
    graph = nx.Graph(Graph.from_vertices_and_edges([0], []))
    _add_metadata(graph)
    orig = Graph(graph.copy())
    with pytest.raises(ValueError):
        separating_set.stable_separating_set(graph)
    assert _eq(orig, graph)

    # single edge graph
    graph = nx.Graph(Graph.from_vertices_and_edges([0, 1], [(0, 1)]))
    _add_metadata(graph)
    orig = Graph(graph.copy())
    with pytest.raises(ValueError):
        separating_set.stable_separating_set(graph)
    assert _eq(orig, graph)

    # triangle graph
    graph = nx.Graph(graphs.Complete(3))
    _add_metadata(graph)
    orig = Graph(graph.copy())
    with pytest.raises(ValueError):
        separating_set.stable_separating_set(graph)
    assert _eq(orig, graph)

    graph = nx.Graph(
        graphs.Complete(3) + relabeled_inc(graphs.CompleteBipartite(2, 3), 2)
    )
    _add_metadata(graph)
    orig = Graph(graph.copy())
    cut = separating_set.stable_separating_set(graph, 2)
    assert separating_set.is_stable_separating_set(graph, cut)
    assert _eq(orig, graph)


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
@pytest.mark.parametrize("check_flexible", [True, False])
@pytest.mark.parametrize("check_connected", [True, False])
def test_stable_separating_set(
    graph,
    one_chosen_vertex,
    two_chosen_vertices,
    check_flexible,
    check_connected,
):
    orig = Graph(graph.copy())
    graph = nx.Graph(graph)

    cut = separating_set.stable_separating_set(
        graph,
        check_flexible=check_flexible,
        check_connected=check_connected,
    )
    assert separating_set.is_stable_separating_set(graph, cut)
    assert _eq(orig, graph)

    for vertex in one_chosen_vertex:
        cut = separating_set.stable_separating_set(
            graph,
            vertex,
            check_flexible=check_flexible,
            check_connected=check_connected,
        )
        assert separating_set.is_stable_separating_set(graph, cut)
        assert _eq(orig, graph)

    for check_distinct_rigid_components in [True, False]:
        for u, v in two_chosen_vertices:
            cut = separating_set.stable_separating_set(
                graph,
                u,
                v,
                check_flexible=check_flexible,
                check_connected=check_connected,
                check_distinct_rigid_components=check_distinct_rigid_components,
            )
            assert separating_set.is_stable_separating_set(graph, cut)
            assert _eq(orig, graph)


@pytest.mark.parametrize(
    "graph, separating_vertex",
    [
        [graphs.Path(3), 1],
        [graphs.Complete(3) + relabeled_inc(graphs.Complete(3), 2), 2],
        [graphs.CompleteMinusOne(5) + relabeled_inc(graphs.Complete(5), 4), 4],
        [graphs.Complete(3) + relabeled_inc(graphs.CompleteBipartite(3, 3), 2), 2],
    ],
)
def test_stable_separating_set_error_single_vertex_separating_set(
    graph, separating_vertex
):
    """
    Test that an error is raised when a vertex that is a (unique?) separating set
    is asked to be avoided.
    """
    with pytest.raises(ValueError):
        separating_set.stable_separating_set(nx.Graph(graph), separating_vertex)


def test_stable_separating_set_2by4_Grid():
    graph = nx.Graph(graphs.Grid(2, 4))
    orig = Graph(graph.copy())

    with pytest.raises(ValueError):
        separating_set.stable_separating_set(graph, 0, 1)
    assert _eq(orig, graph)

    with pytest.raises(ValueError):
        separating_set.stable_separating_set(graph, 1, 2)
    assert _eq(orig, graph)

    for i in [3, 6, 7]:
        cut = separating_set.stable_separating_set(graph, 1, i)
        assert separating_set.is_stable_separating_set(graph, cut)
        assert _eq(orig, graph)
    for i in [0, 1, 4]:
        cut = separating_set.stable_separating_set(graph, 6, i)
        assert separating_set.is_stable_separating_set(graph, cut)
        assert _eq(orig, graph)


def test_stable_separating_set_prism():
    graph = nx.Graph(graphs.ThreePrism())
    with pytest.raises(ValueError):
        separating_set.stable_separating_set(graph, check_flexible=False)

    for u, v in product(graph.nodes, graph.nodes):
        with pytest.raises(ValueError):
            separating_set.stable_separating_set(graph, u, v)
        with pytest.raises(ValueError):
            separating_set.stable_separating_set(graph, u, v, check_flexible=False)


@pytest.mark.parametrize("threshold", ["connectivity", "rigidity"])
@pytest.mark.parametrize("n", [13, 16, 20, 30])
@pytest.mark.parametrize(
    "graph_no",
    [
        pytest.param(1, marks=pytest.mark.slow_main),
        pytest.param(20, marks=pytest.mark.long_local),
    ],
)
@pytest.mark.parametrize("seed", [42, None])
def test_stable_separating_set_random_graphs(
    threshold: str,
    n: int,
    graph_no: int,
    seed: int | None,
):
    pairs_per_graph = n

    rand = random.Random(seed)
    num_tested = 0
    if threshold == "connectivity":
        # it is likely that graph is disconnected
        # if the probability is below this threshold
        p = math.log(n) / n
    if threshold == "rigidity":
        # it is likely that graph is 2-flexible
        # if the probability is below this threshold
        # https://doi.org/10.1112/blms.12740
        p = (math.log(n) + math.log(math.log(n))) / n
    while num_tested < graph_no:
        graph = nx.gnp_random_graph(n, p, seed=rand.randint(0, 2**30))

        # Filter out unreasonable graphs
        if generic_rigidity.is_rigid(graph, dim=2):
            continue

        # Create mapping from vertex to rigid component ids
        # used later to assert expected result
        rigid_components = generic_rigidity.rigid_components(graph, dim=2)
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
                    with pytest.raises(ValueError):
                        separating_set.stable_separating_set(graph, u, v)

                    tests_negative += 1
                else:
                    # valid input
                    cut = separating_set.stable_separating_set(graph, u, v)
                    assert separating_set.is_uv_separating_set(graph, cut, u, v)
                    assert separating_set.is_stable_set(graph, cut)
                    tests_positive += 1

            except AssertionError as e:
                # Changes the error message to include graph
                error_message = (
                    f"Assertion failed: {e}\n"
                    + f"Graph: {repr(graph)}\n"
                    + f"Components: {rigid_components}\n"
                    + f"Vertices: [{u} {v}]"
                )
                raise AssertionError(error_message) from None

        num_tested += 1
