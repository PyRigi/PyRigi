import pytest

import pyrigi.frameworkDB as fws


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
