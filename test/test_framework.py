from pyrigi.graph import Graph
from pyrigi.framework import Framework
import pyrigi.graphDB as graphs
import pyrigi.frameworkDB as fws
from pyrigi.exception import LoopError
from pyrigi.data_type import point_to_vector

from copy import deepcopy

import pytest
from sympy import Matrix, pi, sqrt


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


def test_is_injective():
    F1 = fws.Complete(4, 2)
    assert F1.is_injective()
    assert F1.is_injective(numerical=True)

    F2 = deepcopy(F1)
    F2.set_vertex_pos(0, F2[1])
    assert not F2.is_injective()
    assert not F2.is_injective(numerical=True)

    # test symbolical injectivity with irrational numbers
    F3 = F1.translate(["sqrt(2)", "pi"], inplace=False)
    F3.rotate2D(pi / 3, inplace=True)
    assert F3.is_injective()
    assert F3.is_injective(numerical=True)

    # test numerical injectivity
    F4 = deepcopy(F3)
    F4.set_realization(F4.realization(numerical=True))
    assert F4.is_injective(numerical=True)

    # test numerically not injective, but symbolically injective framework
    F5 = deepcopy(F3)
    F5.set_vertex_pos(0, F5[1] + point_to_vector([1e-10, 1e-10]))
    assert not F5.is_injective(numerical=True, tolerance=1e-8)
    assert not F5.is_injective(numerical=True, tolerance=1e-9)
    assert F5.is_injective()

    # test tolerance in numerical injectivity check
    F6 = deepcopy(F3)
    F6.set_vertex_pos(0, F6[1] + point_to_vector([1e-8, 1e-8]))
    assert F6.is_injective(numerical=True, tolerance=1e-9)
    assert F6.is_injective()


def test_is_quasi_injective():
    F1 = fws.Complete(4, 2)
    assert F1.is_quasi_injective()
    assert F1.is_quasi_injective(numerical=True)

    # test framework that is quasi-injective, but not injective
    F1.set_vertex_pos(1, F1[2])
    F1.delete_edge((1, 2))
    assert F1.is_quasi_injective()
    assert F1.is_quasi_injective(numerical=True)

    # test not quasi-injective framework
    F2 = deepcopy(F1)
    F2.set_vertex_pos(0, F2[1])
    assert not F2.is_quasi_injective()
    assert not F2.is_quasi_injective(numerical=True)

    # test symbolical and numerical quasi-injectivity with irrational numbers
    F3 = F1.translate(["sqrt(2)", "pi"], inplace=False)
    F3 = F3.rotate2D(pi / 2, inplace=False)
    assert F3.is_quasi_injective()
    assert F3.is_quasi_injective(numerical=True)

    # test numerical quasi-injectivity
    F4 = deepcopy(F3)
    F4.set_realization(F4.realization(numerical=True))
    assert F4.is_quasi_injective(numerical=True)

    # test numerically not quasi-injective, but symbollicaly quasi-injective framework
    F5 = deepcopy(F3)
    F5.set_vertex_pos(0, F5[1] + point_to_vector([1e-10, 1e-10]))
    assert not F5.is_quasi_injective(numerical=True, tolerance=1e-8)
    assert not F5.is_quasi_injective(numerical=True, tolerance=1e-9)
    assert F5.is_quasi_injective()

    # test tolerance in numerical quasi-injectivity check
    F6 = deepcopy(F3)
    F6.set_vertex_pos(0, F6[1] + point_to_vector([1e-8, 1e-8]))
    assert F6.is_quasi_injective(numerical=True, tolerance=1e-9)
    assert F6.is_quasi_injective()


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


def test_rotate2D():
    G = graphs.Complete(3)
    F = Framework(G, {0: (0, 0), 1: (2, 0), 2: (1, 1)})

    newF = F.rotate2D(0, False)
    for v, pos in newF.realization().items():
        assert pos.equals(F[v])

    newF = F.rotate2D(pi * 4, False)
    for v, pos in newF.realization().items():
        assert pos.equals(F[v])

    newF = F.rotate2D(pi / 2, False)
    assert newF[0].equals(Matrix([[0], [0]]))
    assert newF[1].equals(Matrix([[0], [2]]))
    assert newF[2].equals(Matrix([[-1], [1]]))

    newF = F.rotate2D(pi / 4, False)
    assert newF[0].equals(Matrix([[0], [0]]))
    assert newF[1].equals(Matrix([[sqrt(2)], [(sqrt(2))]]))
    assert newF[2].equals(Matrix([[0], [sqrt(2)]]))


def test_is_equivalent():
    F1 = fws.Complete(4, 2)
    assert F1.is_equivalent_realization(F1.realization(), numerical=False)
    assert F1.is_equivalent_realization(F1.realization(), numerical=True)
    assert F1.is_equivalent(F1)

    F2 = fws.Complete(3, 2)
    with pytest.raises(ValueError):
        F1.is_equivalent_realization(F2.realization())

    with pytest.raises(ValueError):
        F1.is_equivalent(F2)

    G1 = graphs.ThreePrism()
    G1.delete_vertex(5)

    F3 = Framework(G1, {0: [0, 0], 1: [3, 0], 2: [2, 1], 3: [0, 4], 4: ["5/2", "9/7"]})

    F4 = F3.translate((1, 1), False)
    assert F3.is_equivalent(F4, numerical=True)
    assert F3.is_equivalent(F4)

    F5 = F3.rotate2D(pi / 2, False)
    assert F5.is_equivalent(F3)
    assert F5.is_equivalent(F4)
    assert F5.is_equivalent_realization(F4.realization())

    G2 = Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [3, 4]])
    F6 = Framework(G2, {0: [0, 0], 1: [3, 0], 2: [2, 1], 3: [0, 4], 4: ["5/2", "17/7"]})
    F7 = Framework(
        G2,
        {
            0: [0, 0],
            1: [3, 0],
            2: [2, 1],
            3: ["2*sqrt(2)", "2*sqrt(2)"],
            4: [
                "-93/14 - 31*sqrt(2)/7 + (8 + 6*sqrt(2))*(-432/2359 \
                    - sqrt(-6924487 + 4971663*sqrt(2))/2359 + 1909*sqrt(2)/2359)",
                "-432/2359 - sqrt(-6924487 + 4971663*sqrt(2))/2359 + 1909*sqrt(2)/2359",
            ],
        },
    )
    F8 = Framework(
        G2,
        {
            0: [0, 0],
            1: [3, 0],
            2: [2, 1],
            3: ["2*sqrt(2)", "2*sqrt(2)"],
            4: [
                "-93/14 - 31*sqrt(2)/7 + (8 + 6*sqrt(2))*(-432/2359 + sqrt(-6924487 + \
                    4971663*sqrt(2))/2359 + 1909*sqrt(2)/2359)",
                "-432/2359 + sqrt(-6924487 + 4971663*sqrt(2))/2359 + 1909*sqrt(2)/2359",
            ],
        },
    )

    assert F6.is_equivalent(F7)
    assert F6.is_equivalent(F8)
    assert F7.is_equivalent(F8)

    F9 = F5.translate((pi, "2/3"), False)
    assert F5.is_equivalent(F9)

    with pytest.raises(ValueError):
        assert F8.is_equivalent(F2)

    # testing numerical equivalence

    R1 = {v: pos.evalf() for v, pos in F9.realization().items()}

    assert not F9.is_equivalent_realization(R1, numerical=False)
    assert F9.is_equivalent_realization(R1, numerical=True)


def test_is_congruent():
    G1 = Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [3, 4]])
    F1 = Framework(G1, {0: [0, 0], 1: [3, 0], 2: [2, 1], 3: [0, 4], 4: ["5/2", "17/7"]})
    F2 = Framework(
        G1,
        {
            0: [0, 0],
            1: [3, 0],
            2: [2, 1],
            3: ["2*sqrt(2)", "2*sqrt(2)"],
            4: [
                "-93/14 - 31*sqrt(2)/7 + (8 + 6*sqrt(2))*(-432/2359 - \
                    sqrt(-6924487 + 4971663*sqrt(2))/2359 + 1909*sqrt(2)/2359)",
                "-432/2359 - sqrt(-6924487 + 4971663*sqrt(2))/2359 + 1909*sqrt(2)/2359",
            ],
        },
    )
    F3 = Framework(
        G1,
        {
            0: [0, 0],
            1: [3, 0],
            2: [2, 1],
            3: ["2*sqrt(2)", "2*sqrt(2)"],
            4: [
                "-93/14 - 31*sqrt(2)/7 + (8 + 6*sqrt(2))*(-432/2359 + \
                    sqrt(-6924487 + 4971663*sqrt(2))/2359 + 1909*sqrt(2)/2359)",
                "-432/2359 + sqrt(-6924487 + 4971663*sqrt(2))/2359 + 1909*sqrt(2)/2359",
            ],
        },
    )

    assert F1.is_congruent_realization(F1.realization(), numerical=False)
    assert F1.is_congruent(F1, numerical=False)
    assert F1.is_congruent(F1, numerical=True)

    assert not F1.is_congruent(F2)  # equivalent, but not congruent
    assert not F1.is_congruent(F3)  # equivalent, but not congruent
    assert not F2.is_congruent(F3)  # equivalent, but not congruent

    F4 = F1.translate((pi, "2/3"), False)
    F5 = F1.rotate2D(pi / 2, False)
    assert F1.is_congruent(F4)
    assert F1.is_congruent(F5)
    assert F5.is_congruent(F4)

    F6 = fws.Complete(4, 2)
    F7 = fws.Complete(3, 2)
    with pytest.raises(ValueError):
        assert F6.is_congruent(F7)

    # testing numerical congruence
    R1 = {v: pos.evalf() for v, pos in F4.realization().items()}

    assert not F4.is_congruent_realization(R1)
    assert F4.is_congruent_realization(R1, numerical=True)
