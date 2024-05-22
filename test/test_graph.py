from pyrigi.graph import Graph
import pyrigi.graphDB as graphs

import pytest
from sympy import Matrix


@pytest.mark.parametrize(
    "graph, expected_result",
    [
        (graphs.Cycle(4), False),
        (graphs.Diamond(), True),
        (graphs.Complete(2), True),
        (graphs.Complete(3), True),
        (graphs.Complete(4), True),
        (graphs.Path(2), True),
        (graphs.Path(3), False),
        (graphs.Path(4), False),
    ],
)
def test_is_rigid_d2(graph, expected_result):
    assert graph.is_rigid(dim=2, combinatorial=True) == expected_result
    assert graph.is_rigid(dim=2, combinatorial=False) == expected_result


@pytest.mark.slow
def test_min_max_rigid_subgraphs():
    G = Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5, "a", "b"])
    G.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (0, 3),
            (1, 4),
            (2, 5),
            (0, "a"),
            (0, "b"),
            ("a", "b"),
        ]
    )
    max_subgraphs = G.max_rigid_subgraphs()
    assert (
        len(max_subgraphs) == 2
        and len(max_subgraphs[0].vertex_list()) in [3, 6]
        and len(max_subgraphs[1].vertex_list()) in [3, 6]
        and len(max_subgraphs[0].edges) in [3, 9]
        and len(max_subgraphs[1].edges) in [3, 9]
    )
    min_subgraphs = G.min_rigid_subgraphs()
    print(min_subgraphs[0])
    print(min_subgraphs[1])
    assert (
        len(min_subgraphs) == 2
        and len(min_subgraphs[0].vertex_list()) in [3, 6]
        and len(min_subgraphs[1].vertex_list()) in [3, 6]
        and len(min_subgraphs[0].edges) in [3, 9]
        and len(min_subgraphs[1].edges) in [3, 9]
    )


def test_graph_sparsity():
    assert graphs.Cycle(4).is_sparse(2, 3)
    assert (
        graphs.Diamond().is_tight(2, 3)
        and graphs.Diamond().is_min_rigid(dim=2, combinatorial=True)
        and not graphs.Diamond().is_globally_rigid(dim=2)
    )
    assert not graphs.Complete(4).is_tight(2, 3) and graphs.Complete(
        4
    ).is_globally_rigid(dim=2)


def test_str():
    G = Graph([[2, 1], [2, 3]])
    assert str(G) == "Graph with vertices [1, 2, 3] and edges [[1, 2], [2, 3]]"
    G = Graph([(chr(i + 67), i + 1) for i in range(3)] + [(i, i + 1) for i in range(3)])
    assert str(G) == (
        "Graph with vertices ['C', 1, 'D', 2, 'E', 3, 0] "
        "and edges [('C', 1), (1, 0), (1, 2), ('D', 2), (2, 3), ('E', 3)]"
    )
    G = Graph.from_vertices(["C", 1, "D", 2, "E", 3, 0])
    assert str(G) == "Graph with vertices ['C', 1, 'D', 2, 'E', 3, 0] and edges []"


def test_vertex_edge_lists():
    G = Graph([[2, 1], [2, 3]])
    assert G.vertex_list() == [1, 2, 3]
    assert G.edge_list() == [[1, 2], [2, 3]]
    G = Graph([(chr(i + 67), i + 1) for i in range(3)] + [(i, i + 1) for i in range(3)])
    assert set(G.vertex_list()) == set(["C", 1, "D", 2, "E", 3, 0])
    assert set(G.edge_list()) == set(
        [("C", 1), (1, 0), (1, 2), ("D", 2), (2, 3), ("E", 3)]
    )
    G = Graph.from_vertices(["C", 1, "D", 2, "E", 3, 0])
    assert set(G.vertex_list()) == set(["C", 2, "E", 1, "D", 3, 0])
    assert G.edge_list() == []


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
