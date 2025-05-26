import pytest

from pyrigi.framework import Framework
from pyrigi.framework._export import export as framework_export
from pyrigi.graph import Graph
from test import TEST_WRAPPED_FUNCTIONS
from test.framework import _to_FrameworkBase


@pytest.mark.meshing
def test__generate_stl_bar():
    mesh = framework_export._generate_stl_bar(30, 4, 10, 5)
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
        framework_export._generate_stl_bar(holes_dist, holes_diam, bar_w, bar_h)


@pytest.mark.meshing
def test_generate_stl_bars():
    G = Graph([(0, 1), (1, 2), (2, 3), (0, 3)])
    F = Framework(G, {0: [0, 0], 1: [1, 0], 2: [1, "1/2 * sqrt(5)"], 3: [1 / 2, "4/3"]})
    assert F.generate_stl_bars(scale=20, filename_prefix="mybar") is None
    if TEST_WRAPPED_FUNCTIONS:
        F = _to_FrameworkBase(F)
        assert (
            framework_export.generate_stl_bars(F, scale=20, filename_prefix="mybar")
            is None
        )
