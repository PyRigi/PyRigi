import pytest

import pyrigi._graph_input_check as _graph_input_check
import pyrigi.graphDB as graphs
from pyrigi.exception import LoopError
from pyrigi.graph import Graph


@pytest.mark.parametrize(
    "graph",
    [
        Graph.from_vertices([]),
        Graph.from_vertices([1, 2, 3]),
        Graph.from_vertices_and_edges([1, 2, 3], [[1, 2], [2, 3]]),
        Graph([[1, 2], [2, 3]]),
    ],
)
def test_no_loop(graph):
    assert _graph_input_check.no_loop(graph) is None


@pytest.mark.parametrize(
    "graph",
    [
        Graph([[1, 1]]),
        Graph([[1, 2], [2, 3], [3, 3]]),
    ],
)
def test_no_loop_error(graph):
    with pytest.raises(LoopError):
        _graph_input_check.no_loop(graph)


@pytest.mark.parametrize(
    "vertices, edges",
    [
        [[1], [[1, 1]]],
        [[1, 2, 3], [[1, 2], [2, 3], [3, 3]]],
    ],
)
def test_no_loop_error2(vertices, edges):
    with pytest.raises(LoopError):
        _graph_input_check.no_loop(Graph.from_vertices_and_edges(vertices, edges))


@pytest.mark.parametrize(
    "graph, vertex",
    [
        [Graph.from_vertices([1]), 1],
        [Graph.from_vertices([1, 2, 3]), 3],
        [Graph.from_vertices_and_edges([1, 2, 3], [[1, 2], [2, 3]]), 3],
        [Graph([[1, 2], [2, 3]]), 2],
        [Graph([[1, 2], [1, 1]]), 1],
        [graphs.Complete(3), 0],
        [graphs.Diamond(), 3],
        [Graph.from_vertices([1]), [1]],
        [Graph.from_vertices([1, 2, 3]), [2, 3]],
        [Graph.from_vertices_and_edges([1, 2, 3], [[1, 2], [2, 3]]), [1, 3]],
        [Graph([[1, 2], [2, 3]]), [2, 2]],
        [graphs.Complete(3), [0, 1]],
        [graphs.Diamond(), [1, 3]],
        [Graph([["a", "b"], ["b", 3]]), "a"],
        [Graph([["a", "b"], ["b", 3]]), ["a", "b"]],
        [Graph([["a", "b"], ["b", 3]]), ["a", 3]],
        [Graph([[-1, -2], [-2, 3]]), -1],
        [Graph([[-1, -2], [-2, 3]]), [-1, -2]],
        [Graph([[-1, -2], [-2, 3]]), [-1, 3]],
    ],
)
def test_vertex_members(graph, vertex):
    assert _graph_input_check.vertex_members(graph, vertex) is None


@pytest.mark.parametrize(
    "graph, vertex",
    [
        [Graph([]), 1],
        [Graph.from_vertices([1]), 2],
        [Graph.from_vertices([1, 2, 3]), 4],
        [Graph.from_vertices_and_edges([1, 2, 3], [[1, 2], [2, 3]]), -1],
        [Graph([[1, 2], [2, 3]]), 0],
        [Graph([[1, 2], [1, 1]]), 3],
        [graphs.Complete(3), "a"],
        [graphs.Diamond(), 10],
        [Graph.from_vertices([1]), [2]],
        [Graph.from_vertices([1, 2, 3]), [3, 4]],
        [Graph.from_vertices_and_edges([1, 2, 3], [[1, 2], [2, 3]]), [5, 6]],
        [Graph([[1, 2], [2, 3]]), [2, 2, 4]],
        [graphs.Complete(3), [0, 4]],
        [graphs.Diamond(), [1, 2, 12]],
        [Graph([["a", "b"], ["b", 3]]), "c"],
        [Graph([["a", "b"], ["b", 3]]), ["a", "c"]],
        [Graph([["a", "b"], ["b", 3]]), ["a", 4]],
        [Graph([[-1, -2], [-2, 3]]), -3],
        [Graph([[-1, -2], [-2, 3]]), [-1, -2, 4]],
        [Graph([[-1, -2], [-2, 3]]), [-1, 3, -3]],
    ],
)
def test_vertex_members_error(graph, vertex):
    with pytest.raises(ValueError):
        _graph_input_check.vertex_members(graph, vertex)


@pytest.mark.parametrize(
    "graph, edge",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), (1, 2)],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [3, 2]],
        [Graph([[1, 2], [2, 3]]), [1, 2]],
        [Graph([[1, 2], [1, 1]]), [1, 1]],
        [graphs.Complete(3), [0, 1]],
        [graphs.Diamond(), [1, 2]],
        [Graph([["a", "b"], ["b", 3]]), ["a", "b"]],
        [Graph([["a", "b"], ["b", 3]]), (3, "b")],
        [Graph([["a", "b"], ["b", 3]]), ["b", "a"]],
        [Graph([[-1, -2], [-2, 3]]), [-2, -1]],
        [Graph([[-1, -2], [-2, 3]]), [-1, -2]],
        [Graph([[-1, -2], [-2, 3]]), [-2, 3]],
    ],
)
def test_is_edge(graph, edge):
    assert _graph_input_check.is_edge(graph, edge) is None
    assert _graph_input_check.edge_format(graph, edge) is None


@pytest.mark.parametrize(
    "graph, edge, vertices",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), (1, 2), [1, 2, 2]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [3, 2], [1, 2, 3]],
        [Graph([[1, 2], [2, 3]]), [1, 2], [2, 1]],
        [Graph([[1, 2], [2, 3], [3, 4]]), [1, 2], [3, 2, 1]],
        [Graph([[1, 2], [1, 1]]), [1, 1], [1, 2]],
        [Graph([[1, 2], [1, 1]]), [1, 1], [1, 1]],
        [Graph([[1, 2], [1, 1]]), [1, 1], [1]],
        [graphs.Complete(3), [0, 1], [0, 1, 2, 3, 4]],
        [graphs.Diamond(), [1, 2], [1, 2, 3]],
        [Graph([["a", "b"], ["b", 3]]), ["a", "b"], ["a", "b"]],
        [Graph([["a", "b"], ["b", 3]]), (3, "b"), ["a", "b", 3]],
        [Graph([["a", "b"], ["b", 3]]), ["b", "a"], ["a", "b", 3]],
        [Graph([[-1, -2], [-2, 3]]), [-2, -1], [-3, -2, -1, 0, 1, 2, 3]],
        [Graph([[-1, -2], [-2, 3]]), [-1, -2], [-1, -2, 3]],
        [Graph([[-1, -2], [-2, 3]]), [-2, 3], [-2, 3]],
    ],
)
def test_is_edge_on_vertices(graph, edge, vertices):
    assert _graph_input_check.is_edge(graph, edge, vertices) is None


@pytest.mark.parametrize(
    "graph, edge",
    [
        [Graph([]), (1, 3)],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), (1, 3)],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [3, 1]],
        [Graph([[1, 2], [2, 3]]), [1, 3]],
        [graphs.Complete(3), [0, 4]],
        [graphs.Diamond(), [1, -2]],
        [Graph([["a", "b"], ["b", 3]]), ["a", "c"]],
        [Graph([["a", "b"], ["b", 3]]), (3, "a")],
        [Graph([["a", "b"], ["b", 3]]), ["3", "a"]],
        [Graph([[-1, -2], [-2, 3]]), [3, -1]],
        [Graph([[-1, -2], [-2, 3]]), [-1, 0]],
        [Graph([[-1, -2], [-2, 3]]), [-2, -3]],
        [Graph([[1, 2], [1, 1]]), [2, 2]],
    ],
)
def test_is_edge_value_error(graph, edge):
    with pytest.raises(ValueError):
        _graph_input_check.is_edge(graph, edge)


@pytest.mark.parametrize(
    "graph, edge",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), (1, 1)],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [3, 3]],
        [Graph([["a", "b"], ["b", 3]]), ["a", "a"]],
        [Graph([[-1, -2], [-2, 3]]), [-2, -2]],
        [Graph([[1, 2], [1, 1]]), [1, 1]],
    ],
)
def test_edge_format_loopfree_loop_error(graph, edge):
    assert _graph_input_check.edge_format(graph, edge, loopfree=False) is None
    assert _graph_input_check.edge_format(graph, edge) is None
    with pytest.raises(LoopError):
        _graph_input_check.edge_format(graph, edge, loopfree=True)


@pytest.mark.parametrize(
    "graph, edge, vertices",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), (1, 2), [1, 3, 3]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [3, 2], [1, 3]],
        [Graph([[1, 2], [2, 3]]), [1, 2], [2, 2]],
        [Graph([[1, 2], [2, 3], [3, 4]]), [1, 2], [3, 2]],
        [Graph([[1, 2], [1, 1]]), [1, 1], [2, 2]],
        [Graph([[1, 2], [1, 1]]), [1, 1], [2, 3]],
        [Graph([[1, 2], [1, 1]]), [1, 1], [0]],
        [graphs.Complete(3), [0, 1], [1, 2, 3, 4]],
        [graphs.Diamond(), [1, 2], [1, 3]],
        [Graph([["a", "b"], ["b", 3]]), ["a", "b"], ["a", "c"]],
        [Graph([["a", "b"], ["b", 3]]), (3, "b"), ["a", "b", 2]],
        [Graph([["a", "b"], ["b", 3]]), ["b", "a"], ["a"]],
        [Graph([[-1, -2], [-2, 3]]), [-2, -1], [-3, -2, 0, 1, 2, 3]],
        [Graph([[-1, -2], [-2, 3]]), [-1, -2], [-2, 3]],
        [Graph([[-1, -2], [-2, 3]]), [-2, 3], [3]],
        [graphs.Diamond(), [[1, 2], [2, 3]], None],
        [graphs.Diamond(), [[1, 2], [2, 3]], [1, 2, 3]],
    ],
)
def test_is_edge_on_vertices_value_error(graph, edge, vertices):
    with pytest.raises(ValueError):
        _graph_input_check.is_edge(graph, edge, vertices)


@pytest.mark.parametrize(
    "graph, edge",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), (1,)],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), 1],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [1]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [1, 2, 3]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), "[3, 2]"],
        [Graph([[1, 2], [2, 3]]), "12"],
        [graphs.Complete(3), [[0, 1]]],
    ],
)
def test_is_edge_type_error(graph, edge):
    with pytest.raises(TypeError):
        _graph_input_check.is_edge(graph, edge)
    with pytest.raises(TypeError):
        _graph_input_check.edge_format(graph, edge)


@pytest.mark.parametrize(
    "graph, edge, vertices",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), (1,), [1, 2, 3]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), 1, [1, 2, 3]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [1], [1, 2, 3]],
        [Graph([(1, 2), (2, 3)]), [1, 2, 3], [1, 2, 3]],
        [
            Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]),
            "[3, 2]",
            [1, 2, 3],
        ],
        [Graph([[1, 2], [2, 3]]), "12", [1, 2, 3]],
        [graphs.Complete(3), [[0, 1]], [1, 2, 3]],
        [Graph([[1, 2], [2, 3]]), [1, 2], "1"],
        [Graph([[1, 2], [2, 3]]), [1, 2], 1],
    ],
)
def test_is_edge_on_vertices_type_error(graph, edge, vertices):
    with pytest.raises(TypeError):
        _graph_input_check.is_edge(graph, edge, vertices)


@pytest.mark.parametrize(
    "graph, edge",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [(1, 2)]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [[3, 2]]],
        [Graph([[1, 2], [2, 3]]), [[1, 2]]],
        # [Graph([[1, 2], [1, 1]]), [[1, 1]]],
        [graphs.Complete(3), [[0, 1]]],
        [graphs.Diamond(), [[1, 2]]],
        [Graph([["a", "b"], ["b", 3]]), [["a", "b"]]],
        [Graph([["a", "b"], ["b", 3]]), [(3, "b")]],
        [Graph([["a", "b"], ["b", 3]]), [["b", "a"]]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, -1]]],
        [Graph([[-1, -2], [-2, 3]]), [[-1, -2]]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, 3]]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [(1, 2), (3, 2)]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [[3, 2], [1, 2]]],
        [Graph([[1, 2], [2, 3]]), [[1, 2], (2, 3)]],
        [graphs.Complete(3), [[0, 1], [1, 2]]],
        [graphs.Diamond(), [[1, 2], [2, 3]]],
        [Graph([["a", "b"], ["b", 3]]), [["a", "b"], ["b", 3]]],
        [Graph([["a", "b"], ["b", 3]]), [(3, "b"), ("a", "b")]],
        [Graph([["a", "b"], ["b", 3]]), [["b", "a"], (3, "b")]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, -1], [-2, 3]]],
        [Graph([[-1, -2], [-2, 3]]), [[-1, -2], (-2, 3)]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, 3], [-1, -2]]],
    ],
)
def test_is_edge_list(graph, edge):
    assert _graph_input_check.is_edge_list(graph, edge) is None
    assert _graph_input_check.edge_format_list(graph, edge) is None


@pytest.mark.parametrize(
    "graph, edge",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [(1, 3)]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [[3, 1]]],
        [Graph([[1, 2], [2, 3]]), [[1, 3]]],
        [graphs.Complete(3), [[0, 4]]],
        [graphs.Diamond(), [[1, -2]]],
        [Graph([["a", "b"], ["b", 3]]), [["a", "c"]]],
        [Graph([["a", "b"], ["b", 3]]), [(3, "a")]],
        [Graph([["a", "b"], ["b", 3]]), [["3", "a"]]],
        [Graph([[-1, -2], [-2, 3]]), [[3, -1]]],
        [Graph([[-1, -2], [-2, 3]]), [[-1, 0]]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, -3]]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [(1, 2), (3, 3)]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [[3, 2], [1, 3]]],
        [Graph([[1, 2], [2, 3]]), [[1, 2], (2, 4)]],
        [graphs.Complete(3), [[0, 1], [1, -2]]],
        [graphs.Diamond(), [[1, 5], [2, 3]]],
        [Graph([["a", "b"], ["b", 3]]), [["a", "c"], ["b", 3]]],
        [Graph([["a", "b"], ["b", 3]]), [(3, "b"), ("a", "d")]],
        [Graph([["a", "b"], ["b", 3]]), [["b", "3"], (3, "b")]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, -1], [1, 3]]],
        [Graph([[-1, -2], [-2, 3]]), [[-1, 5], (-2, 3)]],
        [Graph([[-1, -2], [-2, 3]]), [[-2, -3], [-1, -2]]],
        [graphs.Diamond(), [[[1, 2], [2, 3]]]],
    ],
)
def test_is_edge_list_value_error(graph, edge):
    with pytest.raises(ValueError):
        _graph_input_check.is_edge_list(graph, edge)


@pytest.mark.parametrize(
    "graph, edge",
    [
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), (1,)],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), 1],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [(1,)]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), [1]],
        [Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)]), "[3, 2]"],
        [Graph([[1, 2], [2, 3]]), "12"],
        [graphs.Complete(3), [0, 1]],
        [graphs.Diamond(), (1, 2)],
    ],
)
def test_is_edge_list_type_error(graph, edge):
    with pytest.raises(TypeError):
        _graph_input_check.is_edge_list(graph, edge)
    with pytest.raises(TypeError):
        _graph_input_check.edge_format_list(graph, edge)


@pytest.mark.parametrize(
    "graph, vertex_order",
    [
        [Graph([("a", 1.8), ("a", "#"), ("#", 0), (0, 1.8)]), ["a", "#", 0, 1.8]],
        [Graph([[1, 2], [2, 3]]), [1, 2, 3]],
        [Graph([[1, 2], [2, 3]]), [1, 3, 2]],
        [graphs.Complete(3), [0, 1, 2]],
    ],
)
def test_is_vertex_order(graph, vertex_order):
    assert _graph_input_check.is_vertex_order(graph, vertex_order) == vertex_order


@pytest.mark.parametrize(
    "graph, vertex_order",
    [
        [Graph([("a", 1.8), ("a", "#"), ("#", 0), (0, 1.8)]), ["a", "#", 0, "s"]],
        [Graph([[1, 2], [2, 3]]), [1, 3]],
        [Graph([[1, 2], [2, 3]]), [1, 2, 2]],
        [Graph([[1, 2], [2, 3]]), [1, 2, 2, 3]],
        [Graph([[1, 2], [2, 3]]), [1, 2, 3, 4]],
        [graphs.Complete(3), [1, 2, 3]],
    ],
)
def test_is_vertex_order_error(graph, vertex_order):
    with pytest.raises(ValueError):
        _graph_input_check.is_vertex_order(graph, vertex_order)


@pytest.mark.parametrize(
    "graph, edge_order",
    [
        [
            Graph([("a", 1.8), ("a", "#"), ("#", 0), (0, 1.8)]),
            [(0, "#"), ("a", 1.8), (0, 1.8), ("#", "a")],
        ],
        [Graph([[1, 2], [2, 3]]), [[1, 2], [2, 3]]],
        [Graph([[1, 2], [2, 3]]), [[2, 1], [3, 2]]],
        [Graph([[1, 2], [2, 3]]), [[2, 3], [1, 2]]],
        [graphs.Complete(3), [[0, 1], [1, 2], [2, 0]]],
    ],
)
def test_is_edge_order(graph, edge_order):
    assert _graph_input_check.is_edge_order(graph, edge_order) == edge_order


@pytest.mark.parametrize(
    "graph, edge_order",
    [
        [
            Graph([("a", 1.8), ("a", "#"), ("#", 0), (0, 1.8)]),
            [("#", "#"), ("a", 1.8), (0, 1.8), ("#", "a")],
        ],
        [Graph([[1, 2], [2, 3]]), [[1, 2], [2, 4]]],
        [Graph([[1, 2], [2, 3]]), [[1, 2], [2, 3], [1, 3]]],
        [Graph([[1, 2], [2, 3]]), [[1, 2], [2, 3], [1, 2]]],
        [graphs.Complete(3), [[0, 1], [1, 2], [1, 2]]],
    ],
)
def test_is_edge_order_error(graph, edge_order):
    with pytest.raises(ValueError):
        _graph_input_check.is_edge_order(graph, edge_order)
