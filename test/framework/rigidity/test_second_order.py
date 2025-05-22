import pytest

import pyrigi.frameworkDB as fws
from pyrigi.framework import Framework
from pyrigi.framework._rigidity import second_order as second_order_rigidity
from test import TEST_WRAPPED_FUNCTIONS
from test.framework import _to_FrameworkBase


@pytest.mark.parametrize(
    "framework",
    [
        fws.Complete(4, dim=2),
        fws.Frustum(3),
        fws.Frustum(4),
        fws.Frustum(6),
        fws.K33plusEdge(),
        fws.ThreePrism(realization="parallel"),
        fws.Octahedron(realization="regular"),
        Framework(
            fws.Cube().graph.cone(),
            fws.Cube().realization(as_points=True) | {8: ["1/2", "1/2", "1/2"]},
        ),
    ],
)
def test_is_prestress_stable(framework):
    assert framework.is_prestress_stable()
    assert framework.is_prestress_stable(numerical=True)
    if TEST_WRAPPED_FUNCTIONS:
        assert second_order_rigidity.is_prestress_stable(_to_FrameworkBase(framework))
        assert second_order_rigidity.is_prestress_stable(
            _to_FrameworkBase(framework), numerical=True
        )


@pytest.mark.parametrize(
    "framework",
    [
        fws.Square(),
        fws.ThreePrism(realization="flexible"),
        fws.Octahedron(realization="Bricard_plane"),
        fws.Octahedron(realization="Bricard_line"),
        fws.Cube(),
        Framework(
            fws.Cube().graph.cone(),
            fws.Cube().realization(as_points=True) | {8: ["1/2", "1/2", "7/6"]},
        ),
    ],
)
def test_is_not_prestress_stable(framework):
    assert not framework.is_prestress_stable()
    assert not framework.is_prestress_stable(numerical=True)
    if TEST_WRAPPED_FUNCTIONS:
        assert not second_order_rigidity.is_prestress_stable(
            _to_FrameworkBase(framework)
        )
        assert not second_order_rigidity.is_prestress_stable(
            _to_FrameworkBase(framework), numerical=True
        )


@pytest.mark.parametrize(
    "framework",
    [
        fws.CompleteBipartite(3, 3, realization="collinear"),
        fws.SecondOrderRigid(),
    ],
)
def test_is_prestress_stable_error(framework):
    with pytest.raises(ValueError):
        framework.is_prestress_stable()
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.raises(ValueError):
            second_order_rigidity.is_prestress_stable(_to_FrameworkBase(framework))


@pytest.mark.parametrize(
    "framework",
    [
        fws.Complete(4, dim=2),
        fws.Frustum(3),
        fws.Frustum(4),
        fws.Frustum(6),
        fws.K33plusEdge(),
        fws.ThreePrism(realization="parallel"),
        fws.Octahedron(realization="regular"),
        Framework(
            fws.Cube().graph.cone(),
            fws.Cube().realization(as_points=True) | {8: ["1/2", "1/2", "1/2"]},
        ),
    ],
)
def test_is_second_order_rigid(framework):
    assert framework.is_second_order_rigid()
    assert framework.is_second_order_rigid(numerical=True)
    if TEST_WRAPPED_FUNCTIONS:
        assert second_order_rigidity.is_second_order_rigid(_to_FrameworkBase(framework))
        assert second_order_rigidity.is_second_order_rigid(
            _to_FrameworkBase(framework), numerical=True
        )


@pytest.mark.parametrize(
    "framework",
    [
        fws.Square(),
        fws.ThreePrism(realization="flexible"),
        fws.Octahedron(realization="Bricard_plane"),
        fws.Octahedron(realization="Bricard_line"),
        fws.Cube(),
        Framework(
            fws.Cube().graph.cone(),
            fws.Cube().realization(as_points=True) | {8: ["1/2", "1/2", "7/6"]},
        ),
    ],
)
def test_is_not_second_order_rigid(framework):
    assert not framework.is_second_order_rigid()
    assert not framework.is_second_order_rigid(numerical=True)
    if TEST_WRAPPED_FUNCTIONS:
        assert not second_order_rigidity.is_second_order_rigid(
            _to_FrameworkBase(framework)
        )
        assert not second_order_rigidity.is_second_order_rigid(
            _to_FrameworkBase(framework), numerical=True
        )


@pytest.mark.parametrize(
    "framework",
    [
        fws.CompleteBipartite(3, 3, realization="collinear"),
        fws.SecondOrderRigid(),
    ],
)
def test_is_second_order_rigid_error(framework):
    with pytest.raises(ValueError):
        framework.is_second_order_rigid()
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.raises(ValueError):
            second_order_rigidity.is_second_order_rigid(_to_FrameworkBase(framework))
