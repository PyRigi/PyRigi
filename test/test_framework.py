from pyrigi.graph import Graph
from pyrigi.framework import Framework
import pyrigi.graphDB as graphs
import pyrigi.frameworkDB as fws
from pyrigi.exception import LoopError
from pyrigi.data_type import point_to_vector
import matplotlib.pyplot as plt

from copy import deepcopy

import pytest
from sympy import Matrix, pi, sqrt, sympify


@pytest.mark.parametrize(
    "framework",
    [
        fws.Complete(2, dim=1),
        fws.Complete(3, dim=1),
        fws.Complete(4, dim=1),
        fws.Cycle(4, dim=1),
        fws.Cycle(5, dim=1),
        fws.Path(3, dim=1),
        fws.Path(4, dim=1),
        fws.Complete(2, dim=2),
        fws.Complete(3, dim=2),
        fws.Complete(4, dim=2),
        fws.CompleteBipartite(3, 3),
        fws.CompleteBipartite(3, 4),
        fws.CompleteBipartite(4, 4),
        fws.Diamond(),
        fws.K33plusEdge(),
        fws.ThreePrism(),
        fws.ThreePrismPlusEdge(),
        fws.Complete(3, dim=3),
        fws.Complete(4, dim=3),
        fws.Octahedron(),
    ]
    + [fws.Complete(2, dim=n) for n in range(1, 10)]
    + [fws.Complete(3, dim=n) for n in range(1, 10)]
    + [fws.Complete(n - 1, dim=n) for n in range(2, 10)]
    + [fws.Complete(n, dim=n) for n in range(1, 10)]
    + [fws.Complete(n + 1, dim=n) for n in range(1, 10)],
)
def test_is_inf_rigid(framework):
    assert framework.is_inf_rigid()


def test_check_vertex_and_edge_order():
    F = Framework.Random(Graph([("a", 1.8), ("a", "#"), ("#", 0), (0, 1.8)]))
    vertex_order = ["a", "#", 0, 1.8]
    edge_order = [(0, "#"), ("a", 1.8), (0, 1.8), ("#", "a")]
    assert F._check_vertex_order(vertex_order) and F._check_edge_order(edge_order)
    vertex_order = ["a", "#", 0, "s"]
    edge_order = [("#", "#"), ("a", 1.8), (0, 1.8), ("#", "a")]
    with pytest.raises(ValueError):
        F._check_vertex_order(vertex_order)
    with pytest.raises(ValueError):
        F._check_edge_order(edge_order)


@pytest.mark.parametrize(
    "framework",
    [
        Framework.from_points([[i] for i in range(4)]),
        Framework.Collinear(graphs.Complete(3), dim=2),
        fws.CompleteBipartite(1, 3),
        fws.CompleteBipartite(2, 3),
        fws.CompleteBipartite(3, 3, "dixonI"),
        fws.Cycle(4, dim=2),
        fws.Cycle(5, dim=2),
        fws.Path(3, dim=2),
        fws.Path(4, dim=2),
        fws.ThreePrism("flexible"),
        fws.ThreePrism("parallel"),
        fws.Cycle(4, dim=3),
        fws.Path(3, dim=3),
        fws.Path(4, dim=3),
        fws.Frustum(3),
        fws.Cube(),
        fws.Octahedron(realization="Bricard_line"),
        fws.Octahedron(realization="Bricard_plane"),
    ]
    + [fws.Cycle(n - 1, dim=n) for n in range(5, 10)]
    + [fws.Cycle(n, dim=n) for n in range(4, 10)]
    + [fws.Cycle(n + 1, dim=n) for n in range(3, 10)],
)
def test_is_not_inf_rigid(framework):
    assert not framework.is_inf_rigid()


@pytest.mark.parametrize(
    "framework",
    [
        fws.Complete(2, dim=1),
        fws.Path(3, dim=1),
        fws.Path(4, dim=1),
        fws.Complete(2, dim=2),
        fws.Complete(3, dim=2),
        fws.CompleteBipartite(3, 3),
        fws.Diamond(),
        fws.ThreePrism(),
        fws.Complete(3, dim=3),
        fws.Complete(4, dim=3),
        fws.Octahedron(),
    ]
    + [fws.Complete(2, dim=n) for n in range(1, 7)]
    + [fws.Complete(3, dim=n) for n in range(2, 7)]
    + [fws.Complete(n - 1, dim=n) for n in range(2, 7)]
    + [fws.Complete(n, dim=n) for n in range(1, 7)]
    + [fws.Complete(n + 1, dim=n) for n in range(1, 7)],
)
def test_is_min_inf_rigid(framework):
    assert framework.is_min_inf_rigid()


@pytest.mark.parametrize(
    "framework",
    [
        fws.K33plusEdge(),
        fws.ThreePrismPlusEdge(),
        Framework.from_points([[i] for i in range(4)]),
        fws.Complete(3, dim=1),
        fws.Complete(4, dim=1),
        fws.Cycle(4, dim=1),
        fws.Cycle(5, dim=1),
        Framework.Collinear(graphs.Complete(3), dim=2),
        fws.Complete(4, dim=2),
        fws.CompleteBipartite(1, 3),
        fws.CompleteBipartite(2, 3),
        fws.CompleteBipartite(3, 3, "dixonI"),
        fws.CompleteBipartite(3, 4),
        fws.CompleteBipartite(4, 4),
        fws.Cycle(4, dim=2),
        fws.Cycle(5, dim=2),
        fws.Path(3, dim=2),
        fws.Path(4, dim=2),
        fws.ThreePrism("flexible"),
        fws.ThreePrism("parallel"),
        fws.Cycle(4, dim=3),
        fws.Path(3, dim=3),
        fws.Path(4, dim=3),
        fws.Cube(),
        fws.Octahedron(realization="Bricard_line"),
        fws.Octahedron(realization="Bricard_plane"),
    ]
    + [fws.Cycle(n - 1, dim=n) for n in range(5, 7)]
    + [fws.Cycle(n, dim=n) for n in range(4, 7)]
    + [fws.Cycle(n + 1, dim=n) for n in range(3, 7)],
)
def test_is_not_min_inf_rigid(framework):
    assert not framework.is_min_inf_rigid()


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
        fws.Cycle(4, dim=2),
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

    F = fws.ThreePrism(realization="flexible")
    C = Framework(graphs.Complete(6), realization=F.realization())
    explicit_flex = sympify(
        [0, 0, 0, 0, 0, 0, "-sqrt(2)*pi", 0, "-sqrt(2)*pi", 0, "-sqrt(2)*pi", 0]
    )
    assert (
        F.is_vector_inf_flex(explicit_flex)
        and F.is_vector_nontrivial_inf_flex(explicit_flex)
        and F.is_vector_nontrivial_inf_flex(explicit_flex, numerical=True)
        and F.is_nontrivial_flex(explicit_flex, numerical=True)
    )
    explicit_flex_reorder = sympify(
        ["-sqrt(2)*pi", 0, "-sqrt(2)*pi", 0, "-sqrt(2)*pi", 0, 0, 0, 0, 0, 0, 0]
    )
    assert (
        F.is_vector_inf_flex(explicit_flex_reorder, vertex_order=[5, 4, 3, 0, 2, 1])
        and F.is_vector_nontrivial_inf_flex(
            explicit_flex_reorder, vertex_order=[5, 4, 3, 0, 2, 1]
        )
        and F.is_vector_nontrivial_inf_flex(
            explicit_flex_reorder, vertex_order=[5, 4, 3, 0, 2, 1], numerical=True
        )
        and F.is_nontrivial_flex(
            explicit_flex_reorder, vertex_order=[5, 4, 3, 0, 2, 1], numerical=True
        )
    )
    QF = Matrix.hstack(*(F.nontrivial_inf_flexes()))
    QC = Matrix.hstack(*(C.nontrivial_inf_flexes()))
    assert QF.rank() == 1 and QC.rank() == 0
    assert F.trivial_inf_flexes() == C.trivial_inf_flexes()
    QF = Matrix.hstack(*(F.inf_flexes(include_trivial=True)))
    QC = Matrix.hstack(*(F.trivial_inf_flexes()))
    Q_exp = Matrix(explicit_flex)
    assert Matrix.hstack(QF, QC).rank() == 4
    assert Matrix.hstack(QF, Q_exp).rank() == 4

    F = fws.Path(4)
    for inf_flex in F.nontrivial_inf_flexes():
        dict_flex = F._transform_inf_flex_to_pointwise(inf_flex)
        assert F.is_dict_inf_flex(dict_flex) and F.is_dict_nontrivial_inf_flex(
            dict_flex
        )
    assert Matrix.hstack(*(F.nontrivial_inf_flexes())).rank() == 2

    F = fws.Frustum(4)
    explicit_flex = [1, 0, 0, -1, 0, -1, 1, 0, 1, -1, 1, -1, 0, 0, 0, 0]
    assert (
        F.is_vector_inf_flex(explicit_flex)
        and F.is_vector_nontrivial_inf_flex(explicit_flex)
        and F.is_vector_nontrivial_inf_flex(explicit_flex, numerical=True)
        and F.is_nontrivial_flex(explicit_flex, numerical=True)
    )
    QF = Matrix.hstack(*(F.inf_flexes(include_trivial=True)))
    Q_exp = Matrix(explicit_flex)
    assert QF.rank() == 5 and Matrix.hstack(QF, Q_exp).rank() == 5
    QF = Matrix.hstack(*(F.inf_flexes(include_trivial=False)))
    assert QF.rank() == 2 and Matrix.hstack(QF, Q_exp).rank() == 2

    F = fws.Complete(5)
    F_triv = F.trivial_inf_flexes()
    for inf_flex in F_triv:
        dict_flex = F._transform_inf_flex_to_pointwise(inf_flex)
        assert F.is_dict_inf_flex(dict_flex) and F.is_dict_trivial_inf_flex(dict_flex)
    F_all = F.inf_flexes(include_trivial=True)
    assert Matrix.hstack(*F_triv).rank() == Matrix.hstack(*(F_all + F_triv)).rank()

    F = Framework.Random(graphs.DoubleBanana(), dim=3)
    inf_flexes = F.nontrivial_inf_flexes()
    dict_flex = F._transform_inf_flex_to_pointwise(inf_flexes[0])
    assert F.is_dict_inf_flex(dict_flex) and F.is_dict_nontrivial_inf_flex(dict_flex)
    assert Matrix.hstack(*inf_flexes).rank() == 1


def test_is_vector_inf_flex():
    F = Framework.Complete([[0, 0], [1, 0], [0, 1]])
    assert F.is_vector_inf_flex([0, 0, 0, 1, -1, 0])
    assert not F.is_vector_inf_flex([0, 0, 0, 1, -2, 0])
    assert F.is_vector_inf_flex([0, 1, 0, 0, -1, 0], [1, 0, 2])

    F.delete_edge([1, 2])
    assert F.is_vector_inf_flex([0, 0, 0, 1, -1, 0])
    assert F.is_vector_inf_flex([0, 0, 0, -1, -2, 0])
    assert not F.is_vector_inf_flex([0, 0, 2, 1, -2, 1])

    F = fws.ThreePrism(realization="flexible")
    for inf_flex in F.inf_flexes(include_trivial=True):
        assert F.is_vector_inf_flex(inf_flex)


def test_is_dict_inf_flex():
    F = Framework.Complete([[0, 0], [1, 0], [0, 1]])
    assert F.is_dict_inf_flex({0: [0, 0], 1: [0, 1], 2: [-1, 0]})
    assert not F.is_dict_inf_flex({0: [0, 0], 1: [0, -1], 2: [-2, 0]})

    F.delete_edge([1, 2])
    assert F.is_dict_inf_flex({0: [0, 0], 1: [0, 1], 2: [-1, 0]})
    assert F.is_dict_inf_flex({0: [0, 0], 1: [0, -1], 2: [-2, 0]})
    assert not F.is_dict_inf_flex({0: [0, 0], 1: [2, 1], 2: [-2, 1]})

    F = fws.ThreePrism(realization="flexible")
    assert F.is_dict_inf_flex(
        {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [1, 0], 4: [1, 0], 5: [1, 0]}
    )


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


def test_animate_rotating_framework3D():
    F = fws.Complete(4, dim=3)
    F.animate_rotating_framework3D()

    F = fws.Complete(3)
    with pytest.raises(ValueError):
        F.animate_rotating_framework3D()

    F = fws.Complete(5, dim=4)
    with pytest.raises(ValueError):
        F.animate_rotating_framework3D()

    plt.close("all")


def test_rigidity_matrix():
    F = fws.Complete(2)
    assert F.rigidity_matrix() == Matrix([-1, 0, 1, 0]).transpose()

    F = fws.Path(3)
    assert F.rigidity_matrix() == Matrix([[-1, 0, 1, 0, 0, 0], [0, 0, 1, -1, -1, 1]])

    F = fws.Complete(3, dim=1)
    assert F.rigidity_matrix() == Matrix([[-1, 1, 0], [-2, 0, 2], [0, -1, 1]])

    F = fws.Complete(4, dim=3)
    assert F.rigidity_matrix().shape == (6, 12)

    G = Graph([(0, "a"), ("b", "a"), ("b", 1.9), (1.9, 0)])
    F = Framework(G, {0: (0, 0), "a": (1, 0), "b": (1, 1), 1.9: (0, 1)})
    vertex_order = ["a", 1.9, "b", 0]
    assert F.rigidity_matrix(vertex_order=vertex_order) == Matrix(
        [
            [1, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 1, 0, 0, 0, -1],
            [0, -1, 0, 0, 0, 1, 0, 0],
            [0, 0, -1, 0, 1, 0, 0, 0],
        ]
    )


def test_rigidity_matrix_rank():
    K4 = Framework.Complete([(0, 0), (0, 1), (1, 0), (1, 1)])
    assert K4.rigidity_matrix_rank() == 5

    # Deleting one edge does not change the rank of the rigidity matrix ...
    K4.delete_edge([0, 1])
    assert K4.rigidity_matrix_rank() == 5

    # ... whereas deleting two edges does
    K4.delete_edge([2, 3])
    assert K4.rigidity_matrix_rank() == 4

    F = fws.Frustum(3)  # has a single infinitesimal motion and stress
    assert F.rigidity_matrix_rank() == 8


def test_stress_matrix():
    F = fws.Complete(4)
    assert F.stress_matrix([1, -1, 1, 1, -1, 1]) == Matrix(
        [[1, -1, 1, -1], [-1, 1, -1, 1], [1, -1, 1, -1], [-1, 1, -1, 1]]
    )

    F = fws.Frustum(3)
    assert F.stress_matrix([2, 2, 6, 2, 6, 6, -1, -1, -1]) == Matrix(
        [
            [10, -2, -2, -6, 0, 0],
            [-2, 10, -2, 0, -6, 0],
            [-2, -2, 10, 0, 0, -6],
            [-6, 0, 0, 4, 1, 1],
            [0, -6, 0, 1, 4, 1],
            [0, 0, -6, 1, 1, 4],
        ]
    )

    G = Graph([(0, "a"), ("b", "a"), ("b", 1.9), (1.9, 0), ("b", 0), ("a", 1.9)])
    F = Framework(G, {0: (0, 0), "a": (1, 0), "b": (1, 1), 1.9: (0, 1)})
    edge_order = [("a", 0), (1.9, "b"), (1.9, 0), ("a", "b"), ("a", 1.9), (0, "b")]
    stress = F.stresses(edge_order=edge_order)[0].transpose().tolist()[0]
    assert F.stress_matrix(stress, edge_order=edge_order) == Matrix(
        [[-1, 1, -1, 1], [1, -1, 1, -1], [-1, 1, -1, 1], [1, -1, 1, -1]]
    )


def test_stresses():
    Q1 = Matrix.hstack(
        *(fws.CompleteBipartite(4, 4).rigidity_matrix().transpose().nullspace())
    )
    Q2 = Matrix.hstack(*(fws.CompleteBipartite(4, 4).stresses()))
    assert Q1.rank() == Q2.rank() and Q1.rank() == Matrix.hstack(Q1, Q2).rank()
    F = fws.Complete(5)
    assert all(
        [
            F.is_stress([entry for entry in s.transpose()], numerical=True)
            for s in F.stresses()
        ]
    )


def test_edge_lengths():
    G = Graph([(0, 1), (1, 2), (2, 3), (0, 3)])
    F = Framework(G, {0: [0, 0], 1: [1, 0], 2: [1, "1/2 * sqrt(5)"], 3: ["1/2", "4/3"]})
    l_dict = F.edge_lengths(numerical=True)

    expected_result = {
        (0, 1): 1.0,
        (0, 3): 1.4240006242195884,
        (1, 2): 1.118033988749895,
        (2, 3): 0.5443838790578374,
    }

    for edge, length in l_dict.items():
        assert abs(length - expected_result[edge]) < 1e-10

    l_dict = F.edge_lengths(numerical=False)

    expected_result = {
        (0, 1): 1,
        (0, 3): "sqrt(1/4 + 16/9)",
        (1, 2): "1/2 * sqrt(5)",
        (2, 3): "sqrt(1/4 + (1/2 * sqrt(5) - 4/3)**2)",
    }

    for edge, length in l_dict.items():
        assert (sympify(expected_result[edge]) - length).is_zero

    F = fws.Cycle(6)
    assert all([(v - 1).is_zero for v in F.edge_lengths(numerical=False).values()])


@pytest.mark.meshing
def test__generate_stl_bar():
    mesh = Framework._generate_stl_bar(30, 4, 10, 5)
    assert mesh is not None

    with pytest.raises(ValueError):
        # negative values are not allowed
        Framework._generate_stl_bar(30, 4, 10, -5)
    with pytest.raises(ValueError):
        # width must be greater than diameter
        Framework._generate_stl_bar(30, 4, 3, 5)
    with pytest.raises(ValueError):
        # holes_distance <= 2 * holes_diameter
        Framework._generate_stl_bar(6, 4, 10, 5)


@pytest.mark.meshing
def test_generate_stl_bars():
    gr = Graph([(0, 1), (1, 2), (2, 3), (0, 3)])
    fr = Framework(
        gr, {0: [0, 0], 1: [1, 0], 2: [1, "1/2 * sqrt(5)"], 3: [1 / 2, "4/3"]}
    )
    n = fr.generate_stl_bars(scale=20, filename_prefix="mybar")
    assert n is None
