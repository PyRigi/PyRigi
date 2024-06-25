from pyrigi.graph import Graph
from pyrigi.framework import Framework
import pyrigi.graphDB as graphs
import pyrigi.frameworkDB as fws
from pyrigi.exception import LoopError

import pytest
from sympy import Matrix

import sympy as sp


@pytest.mark.parametrize(
    "framework",
    [
        fws.Complete(2, d=1),
        fws.Complete(3, d=1),
        fws.Complete(4, d=1),
        fws.Cycle(4, d=1),
        fws.Cycle(5, d=1),
        fws.Path(3, d=1),
        fws.Path(4, d=1),
        fws.Complete(2, d=2),
        fws.Complete(3, d=2),
        fws.Complete(4, d=2),
        fws.CompleteBipartite(3, 3),
        fws.CompleteBipartite(3, 4),
        fws.CompleteBipartite(4, 4),
        fws.Diamond(),
        fws.K33plusEdge(),
        fws.ThreePrism(),
        fws.ThreePrismPlusEdge(),
        fws.Complete(3, d=3),
        fws.Complete(4, d=3),
    ]
    + [fws.Complete(2, d=n) for n in range(1, 10)]
    + [fws.Complete(3, d=n) for n in range(1, 10)]
    + [fws.Complete(n - 1, d=n) for n in range(2, 10)]
    + [fws.Complete(n, d=n) for n in range(1, 10)]
    + [fws.Complete(n + 1, d=n) for n in range(1, 10)],
)
def test_inf_rigid(framework):
    assert framework.is_inf_rigid()


@pytest.mark.parametrize(
    "framework",
    [
        Framework.from_points([[i] for i in range(4)]),
        Framework.Collinear(graphs.Complete(3), d=2),
        fws.CompleteBipartite(1, 3),
        fws.CompleteBipartite(2, 3),
        fws.CompleteBipartite(3, 3, "dixonI"),
        fws.Cycle(4, d=2),
        fws.Cycle(5, d=2),
        fws.Path(3, d=2),
        fws.Path(4, d=2),
        fws.ThreePrism("flexible"),
        fws.ThreePrism("parallel"),
        fws.Cycle(4, d=3),
        fws.Path(3, d=3),
        fws.Path(4, d=3),
    ]
    + [fws.Cycle(n - 1, d=n) for n in range(5, 10)]
    + [fws.Cycle(n, d=n) for n in range(4, 10)]
    + [fws.Cycle(n + 1, d=n) for n in range(3, 10)],
)
def test_not_inf_rigid(framework):
    assert not framework.is_inf_rigid()


@pytest.mark.parametrize(
    "framework",
    [
        fws.Complete(2, d=1),
        fws.Path(3, d=1),
        fws.Path(4, d=1),
        fws.Complete(2, d=2),
        fws.Complete(3, d=2),
        fws.CompleteBipartite(3, 3),
        fws.Diamond(),
        fws.ThreePrism(),
        fws.Complete(3, d=3),
        fws.Complete(4, d=3),
    ]
    + [fws.Complete(2, d=n) for n in range(1, 7)]
    + [fws.Complete(3, d=n) for n in range(2, 7)]
    + [fws.Complete(n - 1, d=n) for n in range(2, 7)]
    + [fws.Complete(n, d=n) for n in range(1, 7)]
    + [fws.Complete(n + 1, d=n) for n in range(1, 7)],
)
def test_inf_min_rigid(framework):
    assert framework.is_min_inf_rigid()


@pytest.mark.parametrize(
    "framework",
    [
        fws.K33plusEdge(),
        fws.ThreePrismPlusEdge(),
        Framework.from_points([[i] for i in range(4)]),
        fws.Complete(3, d=1),
        fws.Complete(4, d=1),
        fws.Cycle(4, d=1),
        fws.Cycle(5, d=1),
        Framework.Collinear(graphs.Complete(3), d=2),
        fws.Complete(4, d=2),
        fws.CompleteBipartite(1, 3),
        fws.CompleteBipartite(2, 3),
        fws.CompleteBipartite(3, 3, "dixonI"),
        fws.CompleteBipartite(3, 4),
        fws.CompleteBipartite(4, 4),
        fws.Cycle(4, d=2),
        fws.Cycle(5, d=2),
        fws.Path(3, d=2),
        fws.Path(4, d=2),
        fws.ThreePrism("flexible"),
        fws.ThreePrism("parallel"),
        fws.Cycle(4, d=3),
        fws.Path(3, d=3),
        fws.Path(4, d=3),
    ]
    + [fws.Cycle(n - 1, d=n) for n in range(5, 7)]
    + [fws.Cycle(n, d=n) for n in range(4, 7)]
    + [fws.Cycle(n + 1, d=n) for n in range(3, 7)],
)
def test_not_min_inf_rigid(framework):
    assert not framework.is_min_inf_rigid()


def test_dimension():
    assert fws.Complete(2, 2).dim() == fws.Complete(2, 2).dimension()
    assert fws.Complete(2, 2).dim() == 2
    assert Framework.Empty(dim=3).dim() == 3


def test_vertex_addition():
    F = Framework.Empty()
    F.add_vertices([[1.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
    F_ = Framework.Empty()
    F_.add_vertices([[1.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
    F_.set_realization(F.realization())
    assert (
        F.realization() == F_.realization()
        and F.graph().vertex_list() == F_.graph().vertex_list()
        and F.dim() == F_.dim()
    )
    assert F.graph().vertex_list() == [0, 1, 2] and len(F.graph().edges()) == 0
    F.set_vertex_positions_from_lists([0, 2], [[3.0, 0.0], [0.0, 3.0]])
    F_.set_vertex_pos(1, [2.0, 2.0])
    array = F_.realization()
    array[0] = (3, 0)
    assert F[0] != F_[0] and F[1] != F_[1] and F[2] != F_[2]


def test_inf_flexes():
    Q1 = Matrix.hstack(*(fws.Complete(2, 2).inf_flexes(include_trivial=True)))
    Q2 = Matrix.hstack(*(fws.Complete(2, 2).trivial_inf_flexes()))
    assert Q1.rank() == Q2.rank() and Q1.rank() == Matrix.hstack(Q1, Q2).rank()
    assert len(fws.Square().inf_flexes(include_trivial=False)) == 1


def test_framework_loops():
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


def test_framework_translation():
    G = graphs.Complete(3)
    F = Framework(G, {0: (0, 0), 1: (2, 0), 2: (1, 1)})

    newF = F.translate((0, 0), False)
    for v, pos in newF._realization.items():
        assert pos.equals(F._realization[v])

    translation = Matrix([[1], [1]])
    newF = F.translate(translation, False)
    assert newF._realization[0].equals(F._realization[0] + translation)
    assert newF._realization[1].equals(F._realization[1] + translation)
    assert newF._realization[2].equals(F._realization[2] + translation)


def test_framework_rotation():
    G = graphs.Complete(3)
    F = Framework(G, {0: (0, 0), 1: (2, 0), 2: (1, 1)})

    newF = F.rotate2D(0, False)
    for v, pos in newF._realization.items():
        assert pos.equals(F._realization[v])

    newF = F.rotate2D(sp.pi * 4, False)
    for v, pos in newF._realization.items():
        assert pos.equals(F._realization[v])

    newF = F.rotate2D(sp.pi/2, False)
    assert newF._realization[0].equals(Matrix([[0], [0]]))
    assert newF._realization[1].equals(Matrix([[0], [2]]))
    assert newF._realization[2].equals(Matrix([[-1], [1]]))

    newF = F.rotate2D(sp.pi/4, False)
    assert newF._realization[0].equals(Matrix([[0], [0]]))
    assert newF._realization[1].equals(Matrix([[sp.sqrt(2)], [(sp.sqrt(2))]]))
    assert newF._realization[2].equals(Matrix([[0], [sp.sqrt(2)]]))


def test_framework_is_equivalent():
    F1 = fws.Complete(4, 2)
    assert F1.is_equivalent_realization(F1._realization)
    assert F1.is_equivalent(F1)

    F2 = fws.Complete(3, 2)
    with pytest.raises(ValueError):
        F1.is_equivalent_realization(F2._realization)

    with pytest.raises(ValueError):
        F1.is_equivalent(F2)

    G1 = graphs.ThreePrism()
    G1.delete_vertex(5)

    F3 = Framework(
        G1, {0: [0, 0], 1: [3, 0], 2: [2, 1], 3: [
            0, 4], 4: sp.sympify("[5/2, 9/7]")}
    )

    F4 = F3.translate((1, 1))

    assert F3.is_equivalent(F4)

    realization1 = F1._realization


def test_framework_is_congruent():
    F = Framework(Graph([[1, 2], [2, 4]]), {1: (1, 2), 2: (0, 0), 4: (1, 1)})
    assert F.is_congruent_realization(F._realization)
    assert F.is_congruent(F)
