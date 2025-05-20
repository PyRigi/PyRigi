import numpy as np
import pytest
from sympy import Matrix, sympify

import pyrigi.frameworkDB as fws
import pyrigi.graphDB as graphs
from pyrigi.framework import Framework
from pyrigi.framework.rigidity import infinitesimal as infinitesimal_rigidity
from pyrigi.graph import Graph


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
    assert framework.is_inf_rigid(numerical=True)


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
def test_is_inf_flexible(framework):
    assert framework.is_inf_flexible()
    assert framework.is_inf_flexible(numerical=True)


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
    assert framework.is_min_inf_rigid(numerical=True)


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
        dict_flex = infinitesimal_rigidity._transform_inf_flex_to_pointwise(F, inf_flex)
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
        dict_flex = infinitesimal_rigidity._transform_inf_flex_to_pointwise(F, inf_flex)
        assert F.is_dict_inf_flex(dict_flex) and F.is_dict_trivial_inf_flex(dict_flex)
    F_all = F.inf_flexes(include_trivial=True)
    assert Matrix.hstack(*F_triv).rank() == Matrix.hstack(*(F_all + F_triv)).rank()

    F = Framework.Random(graphs.DoubleBanana(), dim=3)
    inf_flexes = F.nontrivial_inf_flexes()
    dict_flex = infinitesimal_rigidity._transform_inf_flex_to_pointwise(
        F, inf_flexes[0]
    )
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
        dict_flex = infinitesimal_rigidity._transform_inf_flex_to_pointwise(F, inf_flex)
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
    dict_flex = infinitesimal_rigidity._transform_inf_flex_to_pointwise(
        F, inf_flexes[0]
    )
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
    assert framework.rigidity_matrix_rank(numerical=True) == rank
