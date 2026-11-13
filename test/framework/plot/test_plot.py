import os

import pytest

from itertools import product

import pyrigi.frameworkDB as fws
import pyrigi.graphDB as graphs
from pyrigi.graph import GainGraph
from pyrigi.framework import Framework, PeriodicFramework
from pyrigi.framework._plot import plot as framework_plot
from pyrigi.framework._rigidity import infinitesimal as infinitesimal_rigidity
from pyrigi.framework._rigidity import stress as stress_rigidity
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
    F = _to_FrameworkBase(F)
    with pytest.raises(ValueError):
        framework_plot.plot(F)


def test_plot():
    F = Framework(graphs.Complete(2), {0: [1, 0], 1: [0, 1]})
    F = _to_FrameworkBase(F)
    framework_plot.plot(F)

    F = Framework(graphs.Complete(2), {0: [1, 0, 0], 1: [0, 1, 1]})
    F = _to_FrameworkBase(F)
    framework_plot.plot(F)


def test_plot2D():
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
        framework_plot.plot2D(F, inf_flex={0: [-1, 0, 0], 1: [1, 0, 0], 2: [0, 0, 0]})
    framework_plot.plot2D(F, inf_flex=0)
    framework_plot.plot2D(F, inf_flex=infinitesimal_rigidity.inf_flexes(F)[0])
    framework_plot.plot2D(
        F, inf_flex=infinitesimal_rigidity.inf_flexes(F, numerical=True)[0]
    )
    framework_plot.plot2D(F, inf_flex=0, fixed_vertices=[0])
    framework_plot.plot2D(F, inf_flex=0, fixed_vertices=[0, 1])

    F = fws.Complete(4)
    F = _to_FrameworkBase(F)
    framework_plot.plot2D(F, stress=0, dpi=200, filename="K4_Test_output")
    os.remove("K4_Test_output.png")
    framework_plot.plot2D(F, stress=stress_rigidity.stresses(F)[0])
    framework_plot.plot2D(F, stress=stress_rigidity.stresses(F, numerical=True)[0])

    F = fws.Complete(4, dim=1)
    F = _to_FrameworkBase(F)
    framework_plot.plot2D(F, stress=0)
    framework_plot.plot2D(F, stress=stress_rigidity.stresses(F)[0])
    framework_plot.plot2D(F, stress=stress_rigidity.stresses(F, numerical=True)[0])


def test_plot3D():
    F = Framework(graphs.Complete(2), {0: [1, 0, 0, 0], 1: [0, 1, 0, 0]})
    F = _to_FrameworkBase(F)
    with pytest.raises(ValueError):
        framework_plot.plot3D(F, projection_matrix=[[1, 0, 0], [0, 0, 1], [0, 0, 0]])
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
    framework_plot.plot3D(F, inf_flex=infinitesimal_rigidity.inf_flexes(F)[0])
    framework_plot.plot3D(
        F, inf_flex=infinitesimal_rigidity.inf_flexes(F, numerical=True)[0]
    )
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
    framework_plot.plot3D(F, stress=stress_rigidity.stresses(F)[0])
    framework_plot.plot3D(F, stress=stress_rigidity.stresses(F, numerical=True)[0])

    F = fws.Complete(4, dim=1)
    F = _to_FrameworkBase(F)
    framework_plot.plot3D(F, stress=0)
    framework_plot.plot3D(F, stress=stress_rigidity.stresses(F)[0])
    framework_plot.plot3D(F, stress=stress_rigidity.stresses(F, numerical=True)[0])


def test_animate3D_rotation():
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


@pytest.mark.parametrize(
    "vertex_colors, edge_colors", list(product([True, False], [True, False]))
)
def test_plot2D_periodic_keywords(vertex_colors, edge_colors):
    graph = GainGraph([(0, 1, (1, 0))], dim=2)
    framework = PeriodicFramework(graph, {0: [1, 0], 1: [0, 1]}, [[2, 0], [0, 2]])

    framework.plot2D_periodic(
        adjacent_cells=1,
        custom_vertex_orbit_colors=vertex_colors,
        custom_edge_orbit_colors=edge_colors,
    )


@pytest.mark.parametrize(
    "realization, lattice",
    [
        [{0: (0.5, 0.5)}, [(1, 0), (0, 1)]],
        [{0: (0.25, 0.75)}, [(1, 1), (-1, 2**0.5)]],
        [{0: (0.5, 0.5, 0.5)}, [(1, 0, 0), (0, 1, 0)]],
    ],
)
def test_plot2D_periodic_cell_count(realization, lattice):
    import matplotlib.pyplot as plt

    graph = GainGraph.from_vertices([0], dim=2)
    framework = PeriodicFramework(graph, realization, lattice)

    for adjacent_cells, expected in [
        (0, 1),
        (1, 9),
        (2, 25),
    ]:
        framework.plot2D_periodic(
            adjacent_cells=adjacent_cells,
            vertex_labels=False,
        )

        ax = plt.gcf().axes[0]

        # One scatter collection per vertex orbit.
        assert len(ax.collections[0].get_offsets()) == expected


def test_plot2D_periodic_edge_count():
    import matplotlib.pyplot as plt

    graph = GainGraph([(0, 0, (1, 0))], dim=2)
    framework = PeriodicFramework(graph, {0: (0.5, 0.5)}, [(1, 0), (0, 1)])

    framework.plot2D_periodic(adjacent_cells=1, vertex_labels=False)

    ax = plt.gcf().axes[0]
    assert len(ax.collections[0].get_segments()) == 9
