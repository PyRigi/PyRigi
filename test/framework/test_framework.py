import os

import matplotlib.pyplot as plt
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


@pytest.mark.parametrize(
    "realization",
    [
        {0: [0, 0, 0, 0], 1: [1, 1, 1, 1]},
        {0: [0, 0, 1, 0], 1: [1, 1, 1, 1]},
        {0: [0, 0, 0, 0], 1: [0, 0, 0, 0]},
    ],
)
def test_plot_error(realization):
    F = Framework(graphs.Complete(2), realization)
    with pytest.raises(ValueError):
        F.plot()

    plt.close()


def test_plot():
    F = Framework(graphs.Complete(2), {0: [1, 0], 1: [0, 1]})
    F.plot()

    F = Framework(graphs.Complete(2), {0: [1, 0, 0], 1: [0, 1, 1]})
    F.plot()

    plt.close("all")


def test_plot2D():
    F = Framework(graphs.Complete(2), {0: [1, 0, 0, 0], 1: [0, 1, 0, 0]})
    with pytest.raises(ValueError):
        F.plot2D(projection_matrix=[[1, 0], [0, 1], [0, 0]])
    F.plot2D(projection_matrix=[[1, 0, 0, 0], [0, 1, 0, 0]])

    F = Framework(graphs.Complete(2), {0: [0, 0, 0], 1: [1, 0, 0]})
    with pytest.raises(ValueError):
        F.plot2D(projection_matrix=[[1, 0], [0, 1]])
    F.plot2D()

    F = Framework(graphs.Path(3), {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 1]})
    with pytest.raises(ValueError):
        F.plot2D(inf_flex={0: [-1, 0, 0], 1: [1, 0, 0], 2: [0, 0, 0]})
    F.plot2D(inf_flex=0)

    F = fws.Complete(4)
    F.plot2D(stress=0, dpi=200, filename="K4_Test_output")
    os.remove("K4_Test_output.png")

    F = fws.Complete(4, dim=1)
    F.plot2D(stress=0)

    plt.close("all")


def test_plot3D():
    F = Framework(graphs.Complete(2), {0: [1, 0, 0, 0], 1: [0, 1, 0, 0]})
    with pytest.raises(ValueError):
        F.plot3D(projection_matrix=[[1, 0, 0], [0, 0, 1], [0, 0, 0]])
    F.plot3D()

    F = Framework(graphs.Complete(2), {0: [0, 0, 0, 0], 1: [1, 0, 0, 0]})
    with pytest.raises(ValueError):
        F.plot3D(projection_matrix=[[1, 0, 0], [0, 0, 1]])
    F.plot3D(projection_matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    F = Framework(graphs.Path(3), {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 1]})
    F.plot3D(inf_flex=0)

    F = fws.Octahedron(realization="Bricard_plane")
    F.plot3D(inf_flex=0, stress=0)

    F = fws.Complete(4)
    F.plot3D(stress=0, dpi=200, filename="K4_Test_output")
    os.remove("K4_Test_output.png")

    F = fws.Complete(4, dim=1)
    F.plot3D(stress=0)

    plt.close("all")


def test_animate3D_rotation():
    F = fws.Complete(4, dim=3)
    F.animate3D_rotation()

    F = fws.Complete(3)
    with pytest.raises(ValueError):
        F.animate3D_rotation()

    F = fws.Complete(5, dim=4)
    with pytest.raises(ValueError):
        F.animate3D_rotation()


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
