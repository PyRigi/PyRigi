import pytest

import pyrigi.frameworkDB as fws
from pyrigi.framework._rigidity import redundant as redundant_rigidity
from test import TEST_WRAPPED_FUNCTIONS
from test.framework import _to_FrameworkBase


@pytest.mark.parametrize(
    "framework",
    [
        fws.Complete(3, dim=1),
        fws.Complete(4, dim=2),
        fws.Cycle(4, dim=1),
        fws.Cycle(5, dim=1),
        fws.CompleteBipartite(4, 3),
        fws.CompleteBipartite(4, 4),
    ],
)
def test_is_redundantly_inf_rigid(framework):
    assert framework.is_redundantly_inf_rigid()
    assert framework.is_redundantly_inf_rigid(numerical=True)
    if TEST_WRAPPED_FUNCTIONS:
        assert redundant_rigidity.is_redundantly_inf_rigid(_to_FrameworkBase(framework))
        assert redundant_rigidity.is_redundantly_inf_rigid(
            _to_FrameworkBase(framework), numerical=True
        )


@pytest.mark.parametrize(
    "framework",
    [
        fws.Complete(2, dim=1),
        fws.Path(3, dim=2),
        fws.Path(4, dim=2),
        fws.Cycle(5, dim=2),
        fws.Complete(2, dim=2),
        fws.Complete(3, dim=2),
        fws.Complete(4, dim=3),
        fws.CompleteBipartite(3, 3),
        fws.K33plusEdge(),
        fws.Diamond(),
        fws.ThreePrism(),
        fws.Complete(3, dim=3),
        fws.Octahedron(),
        fws.Cube(),
    ],
)
def test_is_not_redundantly_inf_rigid(framework):
    assert not framework.is_redundantly_inf_rigid()
    assert not framework.is_redundantly_inf_rigid(numerical=True)
    if TEST_WRAPPED_FUNCTIONS:
        assert not redundant_rigidity.is_redundantly_inf_rigid(
            _to_FrameworkBase(framework)
        )
        assert not redundant_rigidity.is_redundantly_inf_rigid(
            _to_FrameworkBase(framework), numerical=True
        )
