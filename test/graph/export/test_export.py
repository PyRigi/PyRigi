import networkx as nx
import pytest
from sympy import Matrix

import pyrigi.graphDB as graphs
from pyrigi.graph import Graph
from pyrigi.graph._export import export
from test import TEST_WRAPPED_FUNCTIONS


@pytest.mark.parametrize(
    "graph, gint",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 7],
        [graphs.Complete(4), 63],
        [graphs.CompleteBipartite(3, 4), 507840],
        [graphs.CompleteBipartite(4, 4), 31965120],
        [graphs.ThreePrism(), 29327],
    ],
)
def test_integer_representation(graph, gint):
    assert graph.to_int() == gint
    assert Graph.from_int(gint).is_isomorphic(graph)
    assert Graph.from_int(gint).to_int() == gint
    assert Graph.from_int(graph.to_int()).is_isomorphic(graph)
    if TEST_WRAPPED_FUNCTIONS:
        assert export.to_int(nx.Graph(graph)) == gint


def test_integer_representation_error():
    with pytest.raises(ValueError):
        Graph([]).to_int()
    with pytest.raises(ValueError):
        M = Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        G = Graph.from_adjacency_matrix(M)
        G.to_int()
    with pytest.raises(ValueError):
        Graph.from_int(0)
    with pytest.raises(TypeError):
        Graph.from_int(1 / 2)
    with pytest.raises(TypeError):
        Graph.from_int(1.2)
    with pytest.raises(ValueError):
        Graph.from_int(-1)
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.raises(ValueError):
            export.to_int(nx.Graph([]))
        with pytest.raises(ValueError):
            M = Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
            G = Graph.from_adjacency_matrix(M)
            export.to_int(nx.Graph(G))
