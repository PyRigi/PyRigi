import matplotlib.pyplot as plt
import pytest
from sympy import Matrix, pi, sqrt

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


def test_translate():
    G = graphs.Complete(3)
    F = Framework(G, {0: (0, 0), 1: (2, 0), 2: (1, 1)})

    newF = F.translate((0, 0), False)
    for v, pos in newF.realization().items():
        assert pos.equals(F[v])

    translation = Matrix([[1], [1]])
    newF = F.translate(translation, False)
    assert newF[0].equals(F[0] + translation)
    assert newF[1].equals(F[1] + translation)
    assert newF[2].equals(F[2] + translation)


def test_rescale():
    G = graphs.Complete(4)
    F = Framework(G, {0: (-1, 0), 1: (2, 0), 2: (1, 1), 3: (3, -2)})

    newF = F.rescale(1, False)
    for v, pos in newF.realization().items():
        assert pos.equals(F[v])

    newF = F.rescale(2, False)
    assert newF[0].equals(Matrix([p * 2 for p in F[0]]))
    assert newF[1].equals(Matrix([p * 2 for p in F[1]]))
    assert newF[2].equals(Matrix([p * 2 for p in F[2]]))


def test_projected_realization():
    F = fws.Complete(4, dim=3)
    _r = F.projected_realization(
        proj_dim=2, projection_matrix=Matrix([[0, 1, 1], [1, 0, 1]])
    )
    assert (
        len(_r) == 2
        and all([len(val) == 2 for val in _r[0].values()])
        and _r[0][0] == (0, 0)
        and _r[0][1] == (0, 1)
        and _r[0][2] == (1, 0)
        and _r[0][3] == (1, 1)
    )

    _r = F.projected_realization(
        proj_dim=3, projection_matrix=Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    )
    assert (
        len(_r) == 2
        and all([len(val) == 3 for val in _r[0].values()])
        and F.is_congruent_realization(_r[0])
    )

    with pytest.raises(ValueError):
        F.projected_realization(proj_dim=2, projection_matrix=Matrix([[0, 1, 1]]))
        F.projected_realization(proj_dim=2, projection_matrix=Matrix([[0, 1], [1, 0]]))

    F = fws.Complete(6, dim=5)
    with pytest.raises(ValueError):
        F.projected_realization(proj_dim=4)


def test_rotate2D():
    G = graphs.Complete(3)
    F = Framework(G, {0: (0, 0), 1: (2, 0), 2: (1, 1)})

    newF = F.rotate2D(0, inplace=False)
    for v, pos in newF.realization().items():
        assert pos.equals(F[v])

    newF = F.rotate2D(pi * 4, inplace=False)
    for v, pos in newF.realization().items():
        assert pos.equals(F[v])

    newF = F.rotate2D(pi / 2, inplace=False)
    assert newF[0].equals(Matrix([[0], [0]]))
    assert newF[1].equals(Matrix([[0], [2]]))
    assert newF[2].equals(Matrix([[-1], [1]]))

    newF = F.rotate2D(pi / 4, inplace=False)
    assert newF[0].equals(Matrix([[0], [0]]))
    assert newF[1].equals(Matrix([[sqrt(2)], [(sqrt(2))]]))
    assert newF[2].equals(Matrix([[0], [sqrt(2)]]))

    F = Framework(G, {0: (0, 0), 1: (2, 0), 2: (0, 2)})
    newF = F.rotate2D(pi, rotation_center=[1, 1], inplace=False)
    assert newF[0].equals(Matrix([[2], [2]]))
    assert newF[1].equals(Matrix([[0], [2]]))
    assert newF[2].equals(Matrix([[2], [0]]))


def test_rotate3D():
    G = graphs.Complete(3)
    F = Framework(G, {0: (0, 0, 0), 1: (2, 0, 0), 2: (1, 1, 0)})

    newF = F.rotate3D(0, inplace=False)
    for v, pos in newF.realization().items():
        assert pos.equals(F[v])

    newF = F.rotate3D(pi * 4, inplace=False)
    for v, pos in newF.realization().items():
        assert pos.equals(F[v])

    newF = F.rotate3D(pi / 2, inplace=False)
    assert newF[0].equals(Matrix([[0], [0], [0]]))
    assert newF[1].equals(Matrix([[0], [2], [0]]))
    assert newF[2].equals(Matrix([[-1], [1], [0]]))

    newF = F.rotate3D(pi / 4, inplace=False)
    assert newF[0].equals(Matrix([[0], [0], [0]]))
    assert newF[1].equals(Matrix([[sqrt(2)], [(sqrt(2))], [0]]))
    assert newF[2].equals(Matrix([[0], [sqrt(2)], [0]]))

    F.rotate3D(pi / 2, axis_direction=[0, 1, 0], inplace=True)
    assert F[0].equals(Matrix([[0], [0], [0]]))
    assert F[1].equals(Matrix([[0], [0], [-2]]))
    assert F[2].equals(Matrix([[0], [1], [-1]]))

    F = Framework(G, {0: (1, 0, 0), 1: (0, 1, 0), 2: (0, 0, 1)})
    newF = F.rotate3D(2 * pi / 3, axis_direction=[1, 1, 1], inplace=False)
    assert newF[0].equals(Matrix([[0], [1], [0]]))
    assert newF[1].equals(Matrix([[0], [0], [1]]))
    assert newF[2].equals(Matrix([[1], [0], [0]]))

    F.rotate3D(4 * pi / 3, axis_direction=[1, 1, 1], inplace=True)
    assert F[0].equals(Matrix([[0], [0], [1]]))
    assert F[1].equals(Matrix([[1], [0], [0]]))
    assert F[2].equals(Matrix([[0], [1], [0]]))

    F = Framework(G, {0: (0, 0, 0), 1: (2, 0, 0), 2: (0, 2, 0)})
    F.rotate3D(pi, axis_shift=[1, 1, 0], inplace=True)
    assert F[0].equals(Matrix([[2], [2], [0]]))
    assert F[1].equals(Matrix([[0], [2], [0]]))
    assert F[2].equals(Matrix([[2], [0], [0]]))


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
    F.plot2D(stress=0)

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
    F.plot3D(stress=0)

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


@pytest.mark.meshing
def test__generate_stl_bar():
    mesh = Framework._generate_stl_bar(30, 4, 10, 5)
    assert mesh is not None


@pytest.mark.meshing
@pytest.mark.parametrize(
    "holes_dist, holes_diam, bar_w, bar_h",
    [
        # negative values are not allowed
        [30, 4, 10, -5],
        [30, 4, -10, 5],
        [30, -4, 10, 5],
        [-30, 4, 10, 5],
        # zero values are not allowed
        [30, 4, 10, 0],
        [30, 4, 0, 5],
        [30, 0, 10, 5],
        [0, 4, 10, 5],
        # width must be greater than diameter
        [30, 4, 3, 5],
        [30, 4, 4, 5],
        # holes_distance > 2 * holes_diameter
        [6, 4, 10, 5],
        [10, 5, 12, 12],
    ],
)
def test__generate_stl_bar_error(holes_dist, holes_diam, bar_w, bar_h):
    with pytest.raises(ValueError):
        Framework._generate_stl_bar(holes_dist, holes_diam, bar_w, bar_h)


@pytest.mark.meshing
def test_generate_stl_bars():
    gr = Graph([(0, 1), (1, 2), (2, 3), (0, 3)])
    fr = Framework(
        gr, {0: [0, 0], 1: [1, 0], 2: [1, "1/2 * sqrt(5)"], 3: [1 / 2, "4/3"]}
    )
    assert fr.generate_stl_bars(scale=20, filename_prefix="mybar") is None
