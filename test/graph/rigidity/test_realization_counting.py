import math
from itertools import product
from random import randint

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
