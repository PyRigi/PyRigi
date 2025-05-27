import networkx as nx
from sympy import Matrix

import pyrigi.graphDB as graphs
from pyrigi.graph import Graph
from pyrigi.graph import _general as general
from test import TEST_WRAPPED_FUNCTIONS


def test_adjacency_matrix():
    G = Graph()
    assert G.adjacency_matrix() == Matrix([])
    G = Graph([[2, 1], [2, 3]])
    assert G.adjacency_matrix() == Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert G.adjacency_matrix(vertex_order=[2, 3, 1]) == Matrix(
        [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
    )
    assert graphs.Complete(4).adjacency_matrix() == Matrix.ones(4) - Matrix.diag(
        [1, 1, 1, 1]
    )
    G = Graph.from_vertices(["C", 1, "D"])
    assert G.adjacency_matrix() == Matrix.zeros(3)
    G = Graph.from_vertices_and_edges(["C", 1, "D"], [[1, "D"], ["C", "D"]])
    assert G.adjacency_matrix(vertex_order=["C", 1, "D"]) == Matrix(
        [[0, 0, 1], [0, 0, 1], [1, 1, 0]]
    )
    M = Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert G.from_adjacency_matrix(M).adjacency_matrix() == M
    M = Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert G.from_adjacency_matrix(M).adjacency_matrix() == M
    if TEST_WRAPPED_FUNCTIONS:
        G = Graph.from_vertices_and_edges(["C", 1, "D"], [[1, "D"], ["C", "D"]])
        assert general.adjacency_matrix(
            nx.Graph(G), vertex_order=["C", 1, "D"]
        ) == Matrix([[0, 0, 1], [0, 0, 1], [1, 1, 0]])


def test_vertex_and_edge_lists():
    G = Graph([[2, 1], [2, 3]])
    assert G.vertex_list() == [1, 2, 3]
    assert G.edge_list() == [[1, 2], [2, 3]]
    G = Graph([(chr(i + 67), i + 1) for i in range(3)] + [(i, i + 1) for i in range(3)])
    assert set(G.vertex_list()) == {"C", 1, "D", 2, "E", 3, 0}
    assert set(G.edge_list()) == {("C", 1), (1, 0), (1, 2), ("D", 2), (2, 3), ("E", 3)}
    G = Graph.from_vertices(["C", 1, "D", 2, "E", 3, 0])
    assert set(G.vertex_list()) == {"C", 2, "E", 1, "D", 3, 0}
    assert G.edge_list() == []
    if TEST_WRAPPED_FUNCTIONS:
        G = nx.Graph([[2, 1], [2, 3]])
        assert general.vertex_list(G) == [1, 2, 3]
        assert general.edge_list(G) == [[1, 2], [2, 3]]
        G = nx.Graph(
            [(chr(i + 67), i + 1) for i in range(3)] + [(i, i + 1) for i in range(3)]
        )
        assert set(general.vertex_list(G)) == {"C", 1, "D", 2, "E", 3, 0}
        assert set(general.edge_list(G)) == {
            ("C", 1),
            (1, 0),
            (1, 2),
            ("D", 2),
            (2, 3),
            ("E", 3),
        }
        G = nx.Graph(Graph.from_vertices(["C", 1, "D", 2, "E", 3, 0]))
        assert set(general.vertex_list(G)) == {"C", 2, "E", 1, "D", 3, 0}
        assert general.edge_list(G) == []
