from copy import deepcopy

import matplotlib.pyplot as plt
import pytest
from sympy import Matrix, pi, sqrt, sympify
import numpy as np

import pyrigi.graphDB as graphs
import pyrigi.frameworkDB as fws
from pyrigi.exception import LoopError
from pyrigi.framework import Framework
from pyrigi.graph import Graph
from pyrigi.misc import point_to_vector, sympy_expr_to_float, is_zero, is_zero_vector


def test__str__():
    assert (
        str(fws.Complete(2))
        == """Framework in 2-dimensional space consisting of:
Graph with vertices [0, 1] and edges [[0, 1]]
Realization {0:(0, 0), 1:(1, 0)}"""
    )


def test__repr__():
    assert (
        repr(fws.Complete(2)) == "Framework(Graph.from_vertices_and_edges"
        "([0, 1], [(0, 1)]), {0: ['0', '0'], 1: ['1', '0']})"
    )
    F1 = Framework(Graph([(0, 1)]), {0: ["1/2"], 1: ["sqrt(2)"]})
    F2 = eval(repr(F1))
    assert F1[0] == F2[0] and F1[1] == F2[1]


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
        fws.Wheel(4),
        fws.Wheel(5),
        fws.Wheel(6),
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
        fws.Icosahedron(),
    ]
    + [fws.Complete(2, dim=n) for n in range(1, 10)]
    + [fws.Complete(3, dim=n) for n in range(1, 10)]
    + [fws.Complete(n - 1, dim=n) for n in range(2, 10)]
    + [fws.Complete(n, dim=n) for n in range(1, 10)]
    + [fws.Complete(n + 1, dim=n) for n in range(1, 10)],
)
def test_is_inf_rigid(framework):
    assert framework.is_inf_rigid()


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
        fws.Dodecahedron(),
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
        pytest.param(fws.Icosahedron(), marks=pytest.mark.long_local),
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
        fws.Wheel(4),
        fws.Wheel(5),
        fws.Wheel(6),
        fws.ThreePrism("flexible"),
        fws.ThreePrism("parallel"),
        fws.Cycle(4, dim=3),
        fws.Path(3, dim=3),
        fws.Path(4, dim=3),
        fws.Cube(),
        fws.Octahedron(realization="Bricard_line"),
        fws.Octahedron(realization="Bricard_plane"),
        fws.Dodecahedron(),
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
        fws.Complete(4, dim=2),
        fws.Frustum(3),
        fws.Frustum(4),
        fws.Frustum(6),
        fws.K33plusEdge(),
        fws.ThreePrism(realization="parallel"),
        fws.Octahedron(realization="regular"),
        Framework(
            fws.Cube().graph().cone(),
            fws.Cube().realization(as_points=True) | {8: ["1/2", "1/2", "1/2"]},
        ),
        pytest.param(fws.Frustum(5), marks=pytest.mark.long_local),
    ],
)
def test_is_prestress_stable(framework):
    assert framework.is_prestress_stable()
    assert framework.is_prestress_stable(numerical=True)


@pytest.mark.parametrize(
    "framework",
    [
        fws.Square(),
        fws.ThreePrism(realization="flexible"),
        fws.Octahedron(realization="Bricard_plane"),
        fws.Octahedron(realization="Bricard_line"),
        fws.Cube(),
    ],
)
def test_is_not_prestress_stable(framework):
    assert not framework.is_prestress_stable()
    assert not framework.is_prestress_stable(numerical=True)


@pytest.mark.parametrize(
    "framework",
    [
        fws.CompleteBipartite(3, 3, realization="collinear"),
        fws.ConnellyExampleSecondOrderRigidity(),
    ],
)
def test_is_prestress_stable_error(framework):
    with pytest.raises(ValueError):
        framework.is_prestress_stable()


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
            fws.Cube().graph().cone(),
            fws.Cube().realization(as_points=True) | {8: ["1/2", "1/2", "1/2"]},
        ),
        pytest.param(fws.Frustum(5), marks=pytest.mark.long_local),
    ],
)
def test_is_second_order_rigid(framework):
    assert framework.is_second_order_rigid()
    assert framework.is_second_order_rigid(numerical=True)


@pytest.mark.parametrize(
    "framework",
    [
        fws.Square(),
        fws.ThreePrism(realization="flexible"),
        fws.Octahedron(realization="Bricard_plane"),
        fws.Octahedron(realization="Bricard_line"),
        fws.Cube(),
    ],
)
def test_is_not_second_order_rigid(framework):
    assert not framework.is_second_order_rigid()
    assert not framework.is_second_order_rigid(numerical=True)


@pytest.mark.parametrize(
    "framework",
    [
        fws.CompleteBipartite(3, 3, realization="collinear"),
        fws.ConnellyExampleSecondOrderRigidity(),
    ],
)
def test_is_second_order_rigid_error(framework):
    with pytest.raises(ValueError):
        framework.is_second_order_rigid()


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
def test_is_redundantly_rigid(framework):
    assert framework.is_redundantly_rigid()


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
def test_is_not_redundantly_rigid(framework):
    assert not framework.is_redundantly_rigid()


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


def test_inf_flexes_numerical():
    F = fws.ThreePrism(realization="flexible")
    C = Framework(graphs.Complete(6), realization=F.realization())
    QF = np.hstack(F.nontrivial_inf_flexes(numerical=True))
    QC = C.nontrivial_inf_flexes(numerical=True)
    assert np.linalg.matrix_rank(QF) == 1 and len(QC) == 0

    F = fws.Path(4)
    for inf_flex in F.nontrivial_inf_flexes(numerical=True):
        dict_flex = F._transform_inf_flex_to_pointwise(inf_flex)
        assert F.is_dict_inf_flex(
            dict_flex, numerical=True
        ) and F.is_dict_nontrivial_inf_flex(dict_flex, numerical=True, tolerance=1e-4)
    assert (
        np.linalg.matrix_rank(np.vstack(tuple(F.nontrivial_inf_flexes(numerical=True))))
        == 2
    )

    F = fws.Frustum(4)
    QF = np.vstack(tuple(F.inf_flexes(include_trivial=True, numerical=True)))
    assert np.linalg.matrix_rank(QF) == 5
    QF = np.vstack(tuple(F.inf_flexes(include_trivial=False, numerical=True)))
    assert np.linalg.matrix_rank(QF) == 2

    F = fws.Complete(5)
    F_all = F.inf_flexes(include_trivial=True, numerical=True)
    assert len(F_all) == 10

    F = Framework.Random(graphs.DoubleBanana(), dim=3)
    inf_flexes = F.nontrivial_inf_flexes(numerical=True)
    dict_flex = F._transform_inf_flex_to_pointwise(inf_flexes[0])
    assert F.is_dict_inf_flex(
        dict_flex, numerical=True, tolerance=1e-4
    ) and F.is_dict_nontrivial_inf_flex(dict_flex, numerical=True, tolerance=1e-4)
    assert np.linalg.matrix_rank(np.vstack(inf_flexes)) == 1


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

    # test numerically not quasi-injective, but symbolically quasi-injective framework
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


@pytest.mark.parametrize(
    "param",
    [
        0,
        -2,
    ],
)
def test_dimension_value_error(param):
    with pytest.raises(ValueError):
        G = Graph([[1, 2], [1, 1], [2, 3], [1, 3]])
        Framework.Random(G, param)
    with pytest.raises(ValueError):
        Framework.Empty(param)


@pytest.mark.parametrize(
    "param",
    [
        1.1,
        3 / 2,
    ],
)
def test_dimension_type_error(param):
    with pytest.raises(TypeError):
        G = Graph([[1, 2], [1, 1], [2, 3], [1, 3]])
        Framework.Random(G, param)
    with pytest.raises(TypeError):
        Framework.Empty(param)


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


def test_rescale():
    G = graphs.Complete(4)
    F = Framework(G, {0: (-1, 0), 1: (2, 0), 2: (1, 1), 3: (3, -2)})

    newF = F.rescale(1, False)
    for v, pos in newF.realization().items():
        assert pos.equals(F[v])

    newF = F.rescale(2, False)
    assert newF[0].equals(Matrix([p * 2 for p in F[0]]))
    assert newF[1].equals(Matrix([p * 2 for p in F[1]]))
    assert newF[2].equals(Matrix([p * 2 for p in F[2]]))


def test_projected_realization():
    F = fws.Complete(4, dim=3)
    _r = F.projected_realization(
        proj_dim=2, projection_matrix=Matrix([[0, 1, 1], [1, 0, 1]])
    )
    assert (
        len(_r) == 2
        and all([len(val) == 2 for val in _r[0].values()])
        and _r[0][0] == (0, 0)
        and _r[0][1] == (0, 1)
        and _r[0][2] == (1, 0)
        and _r[0][3] == (1, 1)
    )

    _r = F.projected_realization(
        proj_dim=3, projection_matrix=Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    )
    assert (
        len(_r) == 2
        and all([len(val) == 3 for val in _r[0].values()])
        and F.is_congruent_realization(_r[0])
    )

    with pytest.raises(ValueError):
        F.projected_realization(proj_dim=2, projection_matrix=Matrix([[0, 1, 1]]))
        F.projected_realization(proj_dim=2, projection_matrix=Matrix([[0, 1], [1, 0]]))

    F = fws.Complete(6, dim=5)
    with pytest.raises(ValueError):
        F.projected_realization(proj_dim=4)


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

    R1 = {v: sympy_expr_to_float(pos) for v, pos in F9.realization().items()}

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
    assert not F1.is_congruent(F2, numerical=True)  # equivalent, but not congruent
    assert not F1.is_congruent(F3, numerical=True)  # equivalent, but not congruent
    assert not F2.is_congruent(F3, numerical=True)  # equivalent, but not congruent

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
    R1 = {v: sympy_expr_to_float(pos) for v, pos in F4.realization().items()}

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


def test_animate3D_rotation():
    F = fws.Complete(4, dim=3)
    F.animate3D_rotation()

    F = fws.Complete(3)
    with pytest.raises(ValueError):
        F.animate3D_rotation()

    F = fws.Complete(5, dim=4)
    with pytest.raises(ValueError):
        F.animate3D_rotation()


@pytest.mark.parametrize(
    "framework, rigidity_matrix",
    [
        [fws.Complete(2), Matrix([-1, 0, 1, 0]).transpose()],
        [fws.Path(3), Matrix([[-1, 0, 1, 0, 0, 0], [0, 0, 1, -1, -1, 1]])],
        [fws.Complete(3, dim=1), Matrix([[-1, 1, 0], [-2, 0, 2], [0, -1, 1]])],
    ],
)
def test_rigidity_matrix_parametric(framework, rigidity_matrix):
    assert framework.rigidity_matrix() == rigidity_matrix


def test_rigidity_matrix():
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


@pytest.mark.parametrize(
    "framework, rank",
    [
        [Framework.Complete([(0, 0), (0, 1), (1, 0), (1, 1)]), 5],
        [
            Framework(
                graphs.Diamond(),
                {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)},
            ),
            5,
        ],
        [
            Framework(
                graphs.Cycle(4),
                {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)},
            ),
            4,
        ],
        [fws.Frustum(3), 8],
    ],
)
def test_rigidity_matrix_rank(framework, rank):
    assert framework.rigidity_matrix_rank() == rank


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


@pytest.mark.parametrize(
    "framework, num_stresses",
    [
        pytest.param(fws.CompleteBipartite(4, 4), 3, marks=pytest.mark.slow_main),
        [fws.Complete(4), 1],
        pytest.param(fws.Complete(5), 3, marks=pytest.mark.slow_main),
        [fws.Frustum(3), 1],
        [fws.Frustum(4), 1],
        pytest.param(fws.Frustum(5), 1, marks=pytest.mark.long_local),
        [fws.ThreePrism(realization="flexible"), 1],
        [fws.ThreePrism(realization="parallel"), 1],
        [fws.ConnellyExampleSecondOrderRigidity(), 2],
        [fws.CompleteBipartite(3, 3, realization="collinear"), 4],
    ],
)
def test_stresses(framework, num_stresses):
    Q1 = Matrix.hstack(*(framework.rigidity_matrix().transpose().nullspace()))
    Q2 = Matrix.hstack(*(framework.stresses()))
    assert Q1.rank() == Q2.rank() and Q1.rank() == Matrix.hstack(Q1, Q2).rank()

    stresses = framework.stresses()
    assert len(stresses) == num_stresses and all(
        [framework.is_stress(s) for s in stresses]
    )


@pytest.mark.parametrize(
    "framework, num_stresses",
    [
        [fws.CompleteBipartite(4, 4), 3],
        [fws.Complete(4), 1],
        [fws.Complete(5), 3],
        [fws.Complete(6), 6],
        [fws.ThreePrism(realization="flexible"), 1],
        [fws.ThreePrism(realization="parallel"), 1],
        [fws.ConnellyExampleSecondOrderRigidity(), 2],
        [fws.CompleteBipartite(3, 3, realization="collinear"), 4],
    ]
    + [[fws.Frustum(i), 1] for i in range(3, 8)],
)
def test_stresses_numerical(framework, num_stresses):
    stresses = framework.stresses(numerical=True)
    assert len(stresses) == num_stresses and all(
        [framework.is_stress(s, numerical=True) for s in stresses]
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
        assert is_zero(sympify(expected_result[edge]) - length)

    F = fws.Cycle(6)
    assert is_zero_vector([v - 1 for v in F.edge_lengths(numerical=False).values()])


@pytest.mark.parametrize(
    "framework1, framework2",
    [
        [
            fws.Complete(3, dim=2),
            Framework(Graph.from_int(7), {0: [0, 0], 1: [1, 0], 2: [1, 1]}),
        ],
        [
            fws.Complete(4, dim=2),
            fws.Complete(4, dim=2),
        ],
    ],
)
def test__input_check_underlying_graphs(framework1, framework2):
    assert framework1._input_check_underlying_graphs(framework2) is None
    assert framework2._input_check_underlying_graphs(framework1) is None


@pytest.mark.parametrize(
    "framework1, framework2",
    [
        [
            fws.Complete(3, dim=2),
            Framework(Graph.from_int(31), {0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [2, 2]}),
        ],
        [
            Framework(Graph([[1, 2], [2, 3]]), {1: [1, 0], 2: [1, 1], 3: [2, 2]}),
            Framework(Graph([[0, 1], [1, 2]]), {0: [0, 0], 1: [1, 0], 2: [1, 1]}),
        ],
    ],
)
def test__input_check_underlying_graphs_error(framework1, framework2):
    with pytest.raises(ValueError):
        framework1._input_check_underlying_graphs(framework2)


@pytest.mark.parametrize(
    "framework, realization, v",
    [
        [
            Framework(Graph([[1, 2], [2, 3]]), {1: [1, 0], 2: [1, 1], 3: [2, 2]}),
            None,
            1,
        ],
        [
            Framework.Random(Graph([[1, 2], [2, 3]])),
            {1: [1, 0], 2: [1, 1], 3: [2, 2]},
            2,
        ],
        [
            Framework.Random(Graph([["a", 2], [2, -3]])),
            {2: [1, 0], -3: [1, 1], "a": [2, 2]},
            2,
        ],
    ],
)
def test__input_check_vertex_key(framework, realization, v):
    assert framework._input_check_vertex_key(v, realization) is None


@pytest.mark.parametrize(
    "framework, realization, v",
    [
        [
            Framework(Graph([[1, 2], [2, 3]]), {1: [1, 0], 2: [1, 1], 3: [2, 2]}),
            None,
            4,
        ],
        [Framework.Random(Graph([[1, 2], [2, 3]])), {1: [1, 0], 2: [1, 1]}, 3],
        [
            Framework.Random(Graph([["a", 2], [2, -3]])),
            {2: [1, 0], -3: [1, 1], "a": [2, 2]},
            "b",
        ],
    ],
)
def test__input_check_vertex_key_error(framework, realization, v):
    with pytest.raises(KeyError):
        framework._input_check_vertex_key(v, realization)


@pytest.mark.parametrize(
    "framework, point",
    [
        [Framework(Graph([[1, 2], [2, 3]]), {1: [1, 0], 2: [1, 1], 3: [2, 2]}), [2, 3]],
        [Framework.Random(Graph([[1, 2], [2, 3]]), 3), [2, 3, 4]],
        [Framework.Random(Graph([["a", 2], [2, -3]]), 1), [2]],
    ],
)
def test__input_check_point_dimension(framework, point):
    assert framework._input_check_point_dimension(point) is None


@pytest.mark.parametrize(
    "framework, point",
    [
        [
            Framework(Graph([[1, 2], [2, 3]]), {1: [1, 0], 2: [1, 1], 3: [2, 2]}),
            [2, 3, 3],
        ],
        [Framework.Random(Graph([[1, 2], [2, 3]]), 3), [2, 3]],
        [Framework.Random(Graph([["a", 2], [2, -3]]), 1), []],
    ],
)
def test__input_check_point_dimension_error(framework, point):
    with pytest.raises(ValueError):
        framework._input_check_point_dimension(point)


@pytest.mark.meshing
def test__generate_stl_bar():
    mesh = Framework._generate_stl_bar(30, 4, 10, 5)
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
        Framework._generate_stl_bar(holes_dist, holes_diam, bar_w, bar_h)


@pytest.mark.meshing
def test_generate_stl_bars():
    gr = Graph([(0, 1), (1, 2), (2, 3), (0, 3)])
    fr = Framework(
        gr, {0: [0, 0], 1: [1, 0], 2: [1, "1/2 * sqrt(5)"], 3: [1 / 2, "4/3"]}
    )
    assert fr.generate_stl_bars(scale=20, filename_prefix="mybar") is None
