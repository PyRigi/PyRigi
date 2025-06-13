import os

import matplotlib.pyplot as plt
import pytest

import pyrigi.frameworkDB as fws
import pyrigi.graphDB as graphs
from pyrigi.framework import Framework
from pyrigi.framework._plot import plot as framework_plot
from test import TEST_WRAPPED_FUNCTIONS
from test.framework import _to_FrameworkBase


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
    if TEST_WRAPPED_FUNCTIONS:
        F = _to_FrameworkBase(F)
        with pytest.raises(ValueError):
            framework_plot.plot(F)

        plt.close()


def test_plot():
    F = Framework(graphs.Complete(2), {0: [1, 0], 1: [0, 1]})
    F.plot()

    F = Framework(graphs.Complete(2), {0: [1, 0, 0], 1: [0, 1, 1]})
    F.plot()

    plt.close("all")
    if TEST_WRAPPED_FUNCTIONS:
        F = Framework(graphs.Complete(2), {0: [1, 0], 1: [0, 1]})
        F = _to_FrameworkBase(F)
        framework_plot.plot(F)

        F = Framework(graphs.Complete(2), {0: [1, 0, 0], 1: [0, 1, 1]})
        F = _to_FrameworkBase(F)
        framework_plot.plot(F)

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
    F.plot2D(inf_flex=0, fixed_vertices=[0])
    F.plot2D(inf_flex=0, fixed_vertices=[0, 1])

    F = fws.Complete(4)
    F.plot2D(stress=0, dpi=200, filename="K4_Test_output")
    os.remove("K4_Test_output.png")

    F = fws.Complete(4, dim=1)
    F.plot2D(stress=0)

    plt.close("all")
    if TEST_WRAPPED_FUNCTIONS:
        F = Framework(graphs.Complete(2), {0: [1, 0, 0, 0], 1: [0, 1, 0, 0]})
        F = _to_FrameworkBase(F)
        with pytest.raises(ValueError):
            framework_plot.plot2D(F, projection_matrix=[[1, 0], [0, 1], [0, 0]])
        framework_plot.plot2D(F, projection_matrix=[[1, 0, 0, 0], [0, 1, 0, 0]])

        F = Framework(graphs.Complete(2), {0: [0, 0, 0], 1: [1, 0, 0]})
        F = _to_FrameworkBase(F)
        with pytest.raises(ValueError):
            framework_plot.plot2D(F, projection_matrix=[[1, 0], [0, 1]])
        framework_plot.plot2D(F)

        F = Framework(graphs.Path(3), {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 1]})
        F = _to_FrameworkBase(F)
        with pytest.raises(ValueError):
            framework_plot.plot2D(
                F, inf_flex={0: [-1, 0, 0], 1: [1, 0, 0], 2: [0, 0, 0]}
            )
        framework_plot.plot2D(F, inf_flex=0)
        framework_plot.plot2D(F, inf_flex=0, fixed_vertices=[0])
        framework_plot.plot2D(F, inf_flex=0, fixed_vertices=[0, 1])

        F = fws.Complete(4)
        F = _to_FrameworkBase(F)
        framework_plot.plot2D(F, stress=0, dpi=200, filename="K4_Test_output")
        os.remove("K4_Test_output.png")

        F = fws.Complete(4, dim=1)
        F = _to_FrameworkBase(F)
        framework_plot.plot2D(F, stress=0)

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
    F.plot3D(inf_flex=0, fixed_vertices=[0, 1])

    F = fws.Octahedron(realization="Bricard_plane")
    F.plot3D(inf_flex=0, stress=0)
    F.plot3D(inf_flex=0, stress=0, fixed_vertices=F._graph.edge_list()[0])

    F = fws.Complete(4)
    F.plot3D(stress=0, dpi=200, filename="K4_Test_output")
    os.remove("K4_Test_output.png")

    F = fws.Complete(4, dim=1)
    F.plot3D(stress=0)

    plt.close("all")
    if TEST_WRAPPED_FUNCTIONS:
        F = Framework(graphs.Complete(2), {0: [1, 0, 0, 0], 1: [0, 1, 0, 0]})
        F = _to_FrameworkBase(F)
        with pytest.raises(ValueError):
            framework_plot.plot3D(
                F, projection_matrix=[[1, 0, 0], [0, 0, 1], [0, 0, 0]]
            )
        framework_plot.plot3D(F)

        F = Framework(graphs.Complete(2), {0: [0, 0, 0, 0], 1: [1, 0, 0, 0]})
        F = _to_FrameworkBase(F)
        with pytest.raises(ValueError):
            framework_plot.plot3D(F, projection_matrix=[[1, 0, 0], [0, 0, 1]])
        framework_plot.plot3D(
            F, projection_matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        )

        F = Framework(graphs.Path(3), {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 1]})
        F = _to_FrameworkBase(F)
        framework_plot.plot3D(F, inf_flex=0)
        framework_plot.plot3D(F, inf_flex=0, fixed_vertices=[0, 1])

        F = fws.Octahedron(realization="Bricard_plane")
        F = _to_FrameworkBase(F)
        framework_plot.plot3D(F, inf_flex=0, stress=0)
        framework_plot.plot3D(
            F, inf_flex=0, stress=0, fixed_vertices=F._graph.edge_list()[0]
        )

        F = fws.Complete(4)
        F = _to_FrameworkBase(F)
        framework_plot.plot3D(F, stress=0, dpi=200, filename="K4_Test_output")
        os.remove("K4_Test_output.png")

        F = fws.Complete(4, dim=1)
        F = _to_FrameworkBase(F)
        framework_plot.plot3D(F, stress=0)

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
    if TEST_WRAPPED_FUNCTIONS:
        F = fws.Complete(4, dim=3)
        F = _to_FrameworkBase(F)
        framework_plot.animate3D_rotation(F)

        F = fws.Complete(3)
        F = _to_FrameworkBase(F)
        with pytest.raises(ValueError):
            framework_plot.animate3D_rotation(F)

        F = fws.Complete(5, dim=4)
        F = _to_FrameworkBase(F)
        with pytest.raises(ValueError):
            framework_plot.animate3D_rotation(F)
