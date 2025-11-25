import math
from itertools import combinations, product
from copy import deepcopy

import networkx as nx
import pytest

import pyrigi.graphDB as graphs
from pyrigi.graph import Graph
from pyrigi.graph._rigidity import realization_counting
import importlib.util

realization_count_plane_algorithms = ["default", "lnumber", "pyrigi"]
realization_count_sphere_algorithms = ["default", "lnumber", "pyrigi"]


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 2],
        [graphs.CompleteBipartite(3, 3), 16],
        [graphs.Diamond(), 4],
        [graphs.ThreePrism(), 24],
        [Graph.from_int(112525), 48],
        [Graph.from_int(1269995), 56],
    ],
)
@pytest.mark.realization_counting
@pytest.mark.parametrize("algorithm", realization_count_plane_algorithms)
def test_number_of_realizations_count_reflection_min_rigid(
    graph, num_of_realizations, algorithm
):
    assert (
        graph.number_of_realizations(algorithm=algorithm, count_reflection=True)
        == num_of_realizations
    )


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 2],
        [graphs.Complete(4), 2],
        [graphs.K33plusEdge(), 2],
        [graphs.ThreePrismPlusEdge(), 2],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_count_reflection_globally_rigid(
    graph, num_of_realizations
):
    assert graph.number_of_realizations(count_reflection=True) == num_of_realizations


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [Graph.from_int(7903), 4],
        [Graph.from_int(102399), 8],
        [Graph.from_int(811699455), 64],
        [Graph.from_int(812043950), 96],
        [Graph.from_int(1624383), 24],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_count_reflection_rigid(graph, num_of_realizations):
    assert graph.number_of_realizations(count_reflection=True) == num_of_realizations


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.CompleteBipartite(1, 3), 2],
        [graphs.CompleteBipartite(2, 3), 2],
        [graphs.Path(3), 2],
        [graphs.ThreePrism(), 4],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_count_reflection_flex(graph, dim):
    assert graph.number_of_realizations(dim, count_reflection=True) == math.inf


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 1],
        [graphs.CompleteBipartite(3, 3), 8],
        [graphs.Diamond(), 2],
        [graphs.ThreePrism(), 12],
        [Graph.from_int(112525), 24],
        [Graph.from_int(1269995), 28],
    ],
)
@pytest.mark.realization_counting
@pytest.mark.parametrize("algorithm", realization_count_plane_algorithms)
def test_number_of_realizations_min_rigid(graph, num_of_realizations, algorithm):
    assert graph.number_of_realizations(algorithm=algorithm) == num_of_realizations


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [Graph.from_int(7903), 2],
        [Graph.from_int(102399), 4],
        [Graph.from_int(811699455), 32],
        [Graph.from_int(812043950), 48],
        [Graph.from_int(1624383), 12],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_rigid(graph, num_of_realizations):
    assert graph.number_of_realizations() == num_of_realizations


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.Complete(2), 2],
        [graphs.Complete(3), 2],
        [graphs.Complete(4), 2],
        [graphs.K33plusEdge(), 2],
        [graphs.ThreePrismPlusEdge(), 2],
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 3],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_globally_rigid(graph, dim):
    assert graph.number_of_realizations(dim) == 1


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.CompleteBipartite(1, 3), 2],
        [graphs.CompleteBipartite(2, 3), 2],
        [graphs.Path(3), 2],
        [graphs.ThreePrism(), 4],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_flex(graph, dim):
    assert graph.number_of_realizations(dim) == math.inf


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 1],
        [graphs.CompleteBipartite(3, 3), 8],
        [graphs.Diamond(), 2],
        [graphs.ThreePrism(), 16],
        [Graph.from_int(112525), 32],
        [Graph.from_int(481867), 24],
    ],
)
@pytest.mark.realization_counting
@pytest.mark.parametrize("algorithm", realization_count_sphere_algorithms)
def test_number_of_realizations_sphere_min_rigid(graph, num_of_realizations, algorithm):
    assert (
        graph.number_of_realizations(algorithm=algorithm, spherical=True)
        == num_of_realizations
    )


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [Graph.from_int(7903), 2],
        [Graph.from_int(102399), 4],
        [Graph.from_int(811699455), 32],
        [Graph.from_int(812043950), 64],
        [Graph.from_int(1624383), 16],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_sphere_rigid(graph, num_of_realizations):
    assert graph.number_of_realizations(spherical=True) == num_of_realizations


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.Complete(2), 2],
        [graphs.Complete(3), 2],
        [graphs.Complete(4), 2],
        [graphs.K33plusEdge(), 2],
        [graphs.ThreePrismPlusEdge(), 2],
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 3],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_sphere_globally_rigid(graph, dim):
    assert graph.number_of_realizations(dim, spherical=True) == 1


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.CompleteBipartite(1, 3), 2],
        [graphs.CompleteBipartite(2, 3), 2],
        [graphs.Path(3), 2],
        [graphs.ThreePrism(), 4],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_sphere_flex(graph, dim):
    assert graph.number_of_realizations(dim, spherical=True) == math.inf


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 2],
        [graphs.CompleteBipartite(3, 3), 16],
        [graphs.Diamond(), 4],
        [graphs.ThreePrism(), 32],
        [Graph.from_int(112525), 64],
        [Graph.from_int(481867), 48],
    ],
)
@pytest.mark.realization_counting
@pytest.mark.parametrize("algorithm", realization_count_sphere_algorithms)
def test_number_of_realizations_sphere_count_reflection_min_rigid(
    graph, num_of_realizations, algorithm
):
    assert (
        graph.number_of_realizations(
            algorithm=algorithm, spherical=True, count_reflection=True
        )
        == num_of_realizations
    )


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [Graph.from_int(7903), 4],
        [Graph.from_int(102399), 8],
        [Graph.from_int(811699455), 64],
        [Graph.from_int(812043950), 128],
        [Graph.from_int(1624383), 32],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_sphere_count_reflection_rigid(
    graph, num_of_realizations
):
    assert (
        graph.number_of_realizations(spherical=True, count_reflection=True)
        == num_of_realizations
    )


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 2],
        [graphs.Complete(4), 2],
        [graphs.K33plusEdge(), 2],
        [graphs.ThreePrismPlusEdge(), 2],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_sphere_count_reflection_globally_rigid(
    graph, num_of_realizations
):
    assert (
        graph.number_of_realizations(spherical=True, count_reflection=True)
        == num_of_realizations
    )


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.CompleteBipartite(1, 3), 2],
        [graphs.CompleteBipartite(2, 3), 2],
        [graphs.Path(3), 2],
        [graphs.ThreePrism(), 4],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_sphere_count_reflection_flex(graph, dim):
    assert (
        graph.number_of_realizations(dim, spherical=True, count_reflection=True)
        == math.inf
    )


@pytest.mark.parametrize(
    "graph, dim",
    [
        [Graph.from_int(511), 3],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_dim_error(graph, dim):
    with pytest.raises(NotImplementedError):
        graph.number_of_realizations(dim)


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.Complete(3), 2.0],
        [graphs.Complete(3), 3 / 2],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_type_error(graph, dim):
    with pytest.raises(TypeError):
        graph.number_of_realizations(dim)


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.Complete(3), 1],
        [graphs.Complete(4), 2],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_method_error(graph, dim):
    with pytest.raises(ValueError):
        graph.number_of_realizations(dim, algorithm="lnumber")


@pytest.mark.parametrize(
    "biedges, select, result",
    [
        [
            [[[1, 2], [2, 3]], [[1, 2], [1, 2]], [[1, 3], [1, 2]]],
            [[[1, 2], [2, 3]]],
            [[[4, 4], [1, 2]], [[4, 3], [1, 2]]],
        ],
        [[[[1, 2], [1, 2]], [[1, 2], [1, 2]]], [[[1, 2], [1, 2]]], [[[3, 3], [1, 2]]]],
        [[[[1, 2], [1, 2]], [[2, 3], [2, 3]]], [[[1, 2], [1, 2]]], [[[4, 3], [2, 3]]]],
        [
            [[[1, 2], [1, 2]], [[2, 3], [2, 3]], [[1, 3], [1, 3]]],
            [[[1, 2], [1, 2]]],
            [[[4, 3], [2, 3]], [[4, 3], [1, 3]]],
        ],
        [
            [[[1, 2], [1, 2]], [[2, 3], [2, 3]], [[1, 3], [1, 3]]],
            [[[1, 2], [1, 2]], [[2, 3], [2, 3]]],
            [[[4, 4], [1, 3]]],
        ],
        [
            [
                [[1, 2], [1, 2]],
                [[2, 3], [1, 2]],
                [[3, 4], [1, 3]],
                [[3, 4], [3, 4]],
                [[4, 1], [2, 3]],
                [[5, 6], [3, 4]],
            ],
            [[[3, 4], [1, 3]], [[4, 1], [2, 3]]],
            [[[7, 2], [1, 2]], [[2, 7], [1, 2]], [[7, 7], [3, 4]], [[5, 6], [3, 4]]],
        ],
        [
            [
                [[9, 6], [10, 8]],
                [[9, 7], [3, 5]],
                [[9, 6], [3, 8]],
                [[4, 7], [10, 5]],
                [[4, 6], [10, 8]],
            ],
            [[[4, 7], [10, 5]], [[9, 6], [10, 8]]],
            [[[12, 11], [3, 5]], [[12, 12], [3, 8]], [[11, 12], [10, 8]]],
        ],
    ],
)
@pytest.mark.realization_counting
def test__bigraph_contract_delete(biedges, select, result):
    assert realization_counting._bigraph_contract_delete(biedges, select) == result


@pytest.mark.parametrize(
    "biedges, select, result",
    [
        [
            [[[1, 2], [2, 3]], [[1, 2], [1, 2]], [[1, 3], [1, 2]]],
            [[[1, 2], [2, 3]]],
            [[[1, 2], [1, 4]], [[1, 3], [1, 4]]],
        ],
        [[[[1, 2], [1, 2]], [[1, 2], [1, 2]]], [[[1, 2], [1, 2]]], [[[1, 2], [3, 3]]]],
        [[[[1, 2], [1, 2]], [[2, 3], [2, 3]]], [[[1, 2], [1, 2]]], [[[2, 3], [4, 3]]]],
        [
            [[[1, 2], [1, 2]], [[2, 3], [2, 3]], [[1, 3], [1, 3]]],
            [[[1, 2], [1, 2]]],
            [[[2, 3], [4, 3]], [[1, 3], [4, 3]]],
        ],
        [
            [[[1, 2], [1, 2]], [[2, 3], [2, 3]], [[1, 3], [1, 3]]],
            [[[1, 2], [1, 2]], [[2, 3], [2, 3]]],
            [[[1, 3], [4, 4]]],
        ],
    ],
)
@pytest.mark.realization_counting
def test__bigraph_delete_contract(biedges, select, result):
    assert realization_counting._bigraph_delete_contract(biedges, select) == result


def _run_realization_test_on_graph(G: Graph, dim: int) -> None:
    """
    Run a set of realization counting tests on a given graph
    """
    cp = G.number_of_realizations(dim)
    cpr = G.number_of_realizations(dim, count_reflection=True)
    cs = G.number_of_realizations(dim, spherical=True)
    csr = G.number_of_realizations(dim, spherical=True, count_reflection=True)

    assert cp <= cs
    assert cp > 0
    if G.is_rigid(dim):
        assert cs < math.inf
        if G.number_of_nodes() >= dim + 1:
            G2 = G.zero_extension(G.vertex_list()[0:dim], dim=dim)
            assert 2 * cp == G2.number_of_realizations(dim)
            assert 2 * cs == G2.number_of_realizations(dim, spherical=True)
    else:
        assert cs == math.inf

    if G.min_degree() == dim and G.number_of_nodes() > dim + 1:
        G2 = deepcopy(G)
        min_v = [v for v in G2.vertex_list() if G2.degree(v) == dim]
        G2.delete_vertex(min_v[0])
        assert cp == 2 * G2.number_of_realizations(dim)
        assert cs == 2 * G2.number_of_realizations(dim, spherical=True)

    try:
        cp_p = G.number_of_realizations(dim, algorithm="pyrigi")
        assert cp_p == cp
    except ValueError:
        assert True

    try:
        cs_p = G.number_of_realizations(dim, spherical=True, algorithm="pyrigi")
        assert cs_p == cs
    except ValueError:
        assert True

    if (
        dim == 2
        and importlib.util.find_spec("lnumber") is not None
        and G.is_min_rigid(dim)
    ):
        cp_l = G.number_of_realizations(dim, algorithm="lnumber")
        assert cp_l == cp
        cs_l = G.number_of_realizations(dim, spherical=True, algorithm="lnumber")
        assert cs_l == cs

    assert cpr == 2 * cp or (cpr == cp and cp == 1)
    assert csr == 2 * cs or (csr == cs and cs == 1)


def test_randomized_realization_counting():
    search_space = [range(1, 2), range(1, 7), range(10)]
    for dim, n, _ in product(*search_space):
        for m in range(1, math.comb(n, 2) + 1):
            G = Graph(nx.gnm_random_graph(n, m))
            assert G.number_of_nodes() == n
            assert G.number_of_edges() == m

            _run_realization_test_on_graph(G, dim)


def test_small_realizations_counting():
    for n in range(1, 5):
        for i in range(math.comb(n, 2) + 1):
            for edges in combinations(combinations(range(n), 2), i):
                G = Graph.from_vertices_and_edges(range(n), edges)
                assert G.number_of_nodes() == n
                assert G.number_of_edges() == len(edges)

                for dim in [1, 2]:
                    _run_realization_test_on_graph(G, dim)
