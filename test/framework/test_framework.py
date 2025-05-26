import pytest

import pyrigi.frameworkDB as fws
import pyrigi.graphDB as graphs
from pyrigi.exception import LoopError
from pyrigi.framework import Framework
from pyrigi.graph import Graph


def test__str__():
    assert (
        str(fws.Complete(2))
        == """Framework in 2-dimensional space consisting of:
Graph with vertices [0, 1] and edges [[0, 1]]
Realization {0:(0, 0), 1:(1, 0)}"""
    )


def test__repr__():
    assert (
        repr(fws.Complete(2)) == "Framework(Graph.from_vertices_and_edges"
        "([0, 1], [(0, 1)]), {0: ['0', '0'], 1: ['1', '0']})"
    )
    F1 = Framework(Graph([(0, 1)]), {0: ["1/2"], 1: ["sqrt(2)"]})
    F2 = eval(repr(F1))
    assert F1[0] == F2[0] and F1[1] == F2[1]


def test_loop_error():
    with pytest.raises(LoopError):
        G = Graph([[1, 2], [1, 1], [2, 3], [1, 3]])
        F = Framework(G, {1: (0, 0), 2: (1, 1), 3: (2, 0)})
    with pytest.raises(LoopError):
        G = Graph([[1, 2], [2, 3], [1, 3]])
        F = Framework(G, {1: (0, 0), 2: (1, 1), 3: (2, 0)})
        F.add_edge([1, 1])
    with pytest.raises(LoopError):
        G = Graph([[1, 2], [2, 3], [1, 3], [2, 2]])
        Framework.Random(G)


@pytest.mark.parametrize(
    "param",
    [
        0,
        -2,
    ],
)
def test_dimension_value_error(param):
    with pytest.raises(ValueError):
        G = Graph([[1, 2], [1, 1], [2, 3], [1, 3]])
        Framework.Random(G, param)
    with pytest.raises(ValueError):
        Framework.Empty(param)


@pytest.mark.parametrize(
    "param",
    [
        1.1,
        3 / 2,
    ],
)
def test_dimension_type_error(param):
    with pytest.raises(TypeError):
        G = Graph([[1, 2], [1, 1], [2, 3], [1, 3]])
        Framework.Random(G, param)
    with pytest.raises(TypeError):
        Framework.Empty(param)


@pytest.mark.parametrize("dim", range(1, 5))
def test_Random(dim):
    graph = graphs.Complete(dim + 1)
    framework = Framework.Random(graph, dim=dim)
    assert framework.dim == dim

    def min_coord(F):
        return min([min(F[u]) for u in graph])

    def max_coord(F):
        return max([max(F[u]) for u in graph])

    framework = Framework.Random(graph, dim=dim, rand_range=10)
    assert -10 <= min_coord(framework) and max_coord(framework) <= 10

    framework = Framework.Random(graph, dim=dim, rand_range=[10, 100])
    assert 10 <= min_coord(framework) and max_coord(framework) <= 100

    framework = Framework.Random(graph, dim=dim, numerical=True)
    assert -1 <= min_coord(framework) and max_coord(framework) <= 1

    framework = Framework.Random(graph, dim=dim, rand_range=10, numerical=True)
    assert -10 <= min_coord(framework) and max_coord(framework) <= 10

    framework = Framework.Random(graph, dim=dim, rand_range=[10, 100], numerical=True)
    assert 10 <= min_coord(framework) and max_coord(framework) <= 100
