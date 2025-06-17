import math
from itertools import combinations, product
from random import randint
from copy import deepcopy

import networkx as nx
import pytest

import pyrigi.graphDB as graphs
from pyrigi.graph import Graph

@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 2],
        [graphs.CompleteBipartite(3, 3), 16],
        [graphs.Diamond(), 4],
        [graphs.ThreePrism(), 24],
        [graphs.Complete(4), 2],
        [graphs.K33plusEdge(), 2],
        [graphs.ThreePrismPlusEdge(), 2],
        [graphs.CompleteBipartite(1, 3), math.inf],
        [graphs.CompleteBipartite(2, 3), math.inf],
        [graphs.Path(3), math.inf],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_count_reflection(graph, num_of_realizations):
    assert graph.number_of_realizations(count_reflection=True) == num_of_realizations


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 1],
        [graphs.CompleteBipartite(3, 3), 8],
        [graphs.Diamond(), 2],
        [graphs.ThreePrism(), 12],
        [graphs.Complete(4), 1],
        [graphs.K33plusEdge(), 1],
        [graphs.ThreePrismPlusEdge(), 1],
        [graphs.CompleteBipartite(1, 3), math.inf],
        [graphs.CompleteBipartite(2, 3), math.inf],
        [graphs.Path(3), math.inf],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations(graph, num_of_realizations):
    assert graph.number_of_realizations() == num_of_realizations


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 1],
        [graphs.CompleteBipartite(3, 3), 8],
        [graphs.Diamond(), 2],
        [graphs.ThreePrism(), 16],
        [graphs.Complete(4), 1],
        [graphs.K33plusEdge(), 1],
        [graphs.ThreePrismPlusEdge(), 1],
        [graphs.CompleteBipartite(1, 3), math.inf],
        [graphs.CompleteBipartite(2, 3), math.inf],
        [graphs.Path(3), math.inf],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_sphere(graph, num_of_realizations):
    assert graph.number_of_realizations(spherical=True) == num_of_realizations


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 2],
        [graphs.CompleteBipartite(3, 3), 16],
        [graphs.Diamond(), 4],
        [graphs.ThreePrism(), 32],
        [graphs.Complete(4), 2],
        [graphs.K33plusEdge(), 2],
        [graphs.ThreePrismPlusEdge(), 2],
        [graphs.CompleteBipartite(1, 3), math.inf],
        [graphs.CompleteBipartite(2, 3), math.inf],
        [graphs.Path(3), math.inf],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_sphere_count_reflection(graph, num_of_realizations):
    assert (
        graph.number_of_realizations(spherical=True, count_reflection=True)
        == num_of_realizations
    )


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.Complete(3), 3],
        [graphs.ThreePrism(), 4],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_error(graph, dim):
    with pytest.raises(ValueError):
        graph.number_of_realizations(dim)

@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.Complete(3), 2.0],
        [graphs.Complete(3), 3/2],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_error(graph, dim):
    with pytest.raises(TypeError):
        graph.number_of_realizations(dim)



def _run_realization_test_on_graph(G: Graph, dim: int) -> None:
    """
    Run a set of sparsity tests on a given graph
    """
    cp = G.number_of_realizations(dim)
    cs = G.number_of_realizations(dim, spherical=True)

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

                for dim in [1,2]:
                    _run_realization_test_on_graph(G, dim)
