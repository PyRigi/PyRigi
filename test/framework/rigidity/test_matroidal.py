import pytest

import pyrigi.frameworkDB as fws
import pyrigi.graphDB as graphs
from pyrigi.framework import Framework


@pytest.mark.parametrize(
    "framework",
    [
        fws.Complete(2, dim=1),
        fws.Complete(2, dim=2),
        fws.Complete(3, dim=2),
        fws.Complete(3, dim=3),
        fws.Complete(4, dim=3),
        fws.CompleteBipartite(3, 3),
        fws.CompleteBipartite(1, 3),
        fws.CompleteBipartite(2, 3),
        fws.Diamond(),
        fws.ThreePrism(),
        Framework.from_points([[i] for i in range(4)]),
        fws.Cycle(4, dim=2),
        fws.Cycle(5, dim=2),
        fws.Path(3, dim=1),
        fws.Path(3, dim=2),
        fws.Path(4, dim=2),
        fws.Path(3, dim=3),
        fws.Path(4, dim=3),
    ]
    + [fws.Complete(2, dim=n) for n in range(1, 7)]
    + [fws.Complete(3, dim=n) for n in range(2, 7)]
    + [fws.Complete(n - 1, dim=n) for n in range(2, 7)]
    + [fws.Complete(n, dim=n) for n in range(1, 7)]
    + [fws.Complete(n + 1, dim=n) for n in range(1, 7)]
    + [fws.Cycle(n - 1, dim=n) for n in range(5, 7)]
    + [fws.Cycle(n, dim=n) for n in range(4, 7)]
    + [fws.Cycle(n + 1, dim=n) for n in range(3, 7)],
)
def test_is_independent(framework):
    assert framework.is_independent()
    assert framework.is_independent(numerical=True)


@pytest.mark.parametrize(
    "framework",
    [
        fws.K33plusEdge(),
        fws.ThreePrismPlusEdge(),
        Framework.Collinear(graphs.Complete(3), dim=2),
        fws.Complete(3, dim=1),
        fws.Complete(4, dim=1),
        fws.Complete(4, dim=2),
        fws.CompleteBipartite(3, 3, "dixonI"),
        fws.CompleteBipartite(3, 4),
        fws.CompleteBipartite(4, 4),
        fws.ThreePrism("flexible"),
        fws.ThreePrism("parallel"),
        fws.Cycle(4, dim=1),
        fws.Cycle(5, dim=1),
    ]
    + [Framework.Random(graphs.Complete(n), dim=n - 2) for n in range(3, 8)],
)
def test_is_dependent(framework):
    assert framework.is_dependent()
    assert framework.is_dependent(numerical=True)


@pytest.mark.parametrize(
    "framework",
    [
        fws.Complete(2, dim=1),
        fws.Complete(2, dim=2),
        fws.Complete(3, dim=2),
        fws.Complete(3, dim=3),
        fws.Complete(4, dim=3),
        fws.CompleteBipartite(3, 3),
        fws.Diamond(),
        fws.ThreePrism(),
        fws.Path(3, dim=1),
    ]
    + [fws.Complete(2, dim=n) for n in range(1, 7)]
    + [fws.Complete(3, dim=n) for n in range(2, 7)]
    + [fws.Complete(n - 1, dim=n) for n in range(2, 7)],
)
def test_is_isostatic(framework):
    assert framework.is_isostatic()
    assert framework.is_isostatic(numerical=True)


@pytest.mark.parametrize(
    "framework",
    [
        fws.K33plusEdge(),
        fws.ThreePrismPlusEdge(),
        Framework.Collinear(graphs.Complete(3), dim=2),
        fws.Complete(3, dim=1),
        fws.Complete(4, dim=1),
        fws.Complete(4, dim=2),
        fws.CompleteBipartite(1, 3),
        fws.CompleteBipartite(2, 3),
        fws.CompleteBipartite(3, 3, "dixonI"),
        fws.CompleteBipartite(3, 4),
        fws.CompleteBipartite(4, 4),
        Framework.from_points([[i] for i in range(4)]),
        fws.Cycle(4, dim=2),
        fws.Cycle(5, dim=2),
        fws.Path(3, dim=2),
        fws.Path(4, dim=2),
        fws.Path(3, dim=3),
        fws.Path(4, dim=3),
        fws.ThreePrism("flexible"),
        fws.ThreePrism("parallel"),
        fws.Cycle(4, dim=1),
        fws.Cycle(5, dim=1),
    ]
    + [Framework.Random(graphs.Complete(n), dim=n - 2) for n in range(3, 8)],
)
def test_is_not_isostatic(framework):
    assert not framework.is_isostatic()
