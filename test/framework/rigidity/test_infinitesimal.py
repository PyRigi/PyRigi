import numpy as np
import pytest
from sympy import Matrix, sympify

import pyrigi.frameworkDB as fws
import pyrigi.graphDB as graphs
from pyrigi._utils._zero_check import is_zero_vector
from pyrigi.framework import Framework
from pyrigi.framework._rigidity import infinitesimal as infinitesimal_rigidity
from pyrigi.graph import Graph
from test.framework import _to_FrameworkBase


@pytest.mark.parametrize(
    "framework",
    [
        Framework.Collinear(graphs.Complete(2), dim=2),
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
    assert infinitesimal_rigidity.is_inf_rigid(_to_FrameworkBase(framework))
    assert infinitesimal_rigidity.is_inf_rigid(
        _to_FrameworkBase(framework), numerical=True
    )


@pytest.mark.parametrize(
    "framework",
    [
        Framework.from_points([[i] for i in range(4)]),
        Framework.Collinear(graphs.Complete(3), dim=2),
        Framework.Collinear(graphs.Complete(4), dim=2),
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
    assert infinitesimal_rigidity.is_inf_flexible(_to_FrameworkBase(framework))
    assert infinitesimal_rigidity.is_inf_flexible(
        _to_FrameworkBase(framework), numerical=True
    )


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
    assert infinitesimal_rigidity.is_min_inf_rigid(_to_FrameworkBase(framework))
    assert infinitesimal_rigidity.is_min_inf_rigid(
        _to_FrameworkBase(framework), numerical=True
    )


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
    assert not infinitesimal_rigidity.is_min_inf_rigid(_to_FrameworkBase(framework))
    assert not infinitesimal_rigidity.is_min_inf_rigid(
        _to_FrameworkBase(framework), numerical=True
    )


def test_inf_flexes():
    F1 = _to_FrameworkBase(fws.Complete(2, 2))
    Q1 = Matrix.hstack(*(infinitesimal_rigidity.inf_flexes(F1, include_trivial=True)))
    Q2 = Matrix.hstack(*(infinitesimal_rigidity.trivial_inf_flexes(F1)))
    assert Q1.rank() == Q2.rank() and Q1.rank() == Matrix.hstack(Q1, Q2).rank()
    F2 = _to_FrameworkBase(fws.Square())
    assert len(infinitesimal_rigidity.inf_flexes(F2, include_trivial=False)) == 1

    F = fws.ThreePrism(realization="flexible")
    F = _to_FrameworkBase(F)
    C = Framework(graphs.Complete(6), realization=F.realization())
    C = _to_FrameworkBase(C)
    explicit_flex = sympify(
        [0, 0, 0, 0, 0, 0, "-sqrt(2)*pi", 0, "-sqrt(2)*pi", 0, "-sqrt(2)*pi", 0]
    )
    assert (
        infinitesimal_rigidity.is_vector_inf_flex(F, explicit_flex)
        and infinitesimal_rigidity.is_vector_nontrivial_inf_flex(F, explicit_flex)
        and infinitesimal_rigidity.is_vector_nontrivial_inf_flex(
            F, explicit_flex, numerical=True
        )
        and infinitesimal_rigidity.is_nontrivial_flex(F, explicit_flex, numerical=True)
    )
    explicit_flex_reorder = sympify(
        ["-sqrt(2)*pi", 0, "-sqrt(2)*pi", 0, "-sqrt(2)*pi", 0, 0, 0, 0, 0, 0, 0]
    )
    assert (
        infinitesimal_rigidity.is_vector_inf_flex(
            F, explicit_flex_reorder, vertex_order=[5, 4, 3, 0, 2, 1]
        )
        and infinitesimal_rigidity.is_vector_nontrivial_inf_flex(
            F, explicit_flex_reorder, vertex_order=[5, 4, 3, 0, 2, 1]
        )
        and infinitesimal_rigidity.is_vector_nontrivial_inf_flex(
            F,
            explicit_flex_reorder,
            vertex_order=[5, 4, 3, 0, 2, 1],
            numerical=True,
        )
        and infinitesimal_rigidity.is_nontrivial_flex(
            F,
            explicit_flex_reorder,
            vertex_order=[5, 4, 3, 0, 2, 1],
            numerical=True,
        )
    )
    QF = Matrix.hstack(*(infinitesimal_rigidity.nontrivial_inf_flexes(F)))
    QC = Matrix.hstack(*(infinitesimal_rigidity.nontrivial_inf_flexes(C)))
    assert QF.rank() == 1 and QC.rank() == 0
    assert infinitesimal_rigidity.trivial_inf_flexes(
        F
    ) == infinitesimal_rigidity.trivial_inf_flexes(C)
    QF = Matrix.hstack(*(infinitesimal_rigidity.inf_flexes(F, include_trivial=True)))
    QC = Matrix.hstack(*(infinitesimal_rigidity.trivial_inf_flexes(F)))
    Q_exp = Matrix(explicit_flex)
    assert Matrix.hstack(QF, QC).rank() == 4
    assert Matrix.hstack(QF, Q_exp).rank() == 4

    F = fws.Path(4)
    F = _to_FrameworkBase(F)
    for inf_flex in infinitesimal_rigidity.nontrivial_inf_flexes(F):
        dict_flex = infinitesimal_rigidity._transform_inf_flex_to_pointwise(F, inf_flex)
        assert infinitesimal_rigidity.is_dict_inf_flex(
            F, dict_flex
        ) and infinitesimal_rigidity.is_dict_nontrivial_inf_flex(F, dict_flex)
    assert Matrix.hstack(*(infinitesimal_rigidity.nontrivial_inf_flexes(F))).rank() == 2

    F = fws.Frustum(4)
    F = _to_FrameworkBase(F)
    explicit_flex = [1, 0, 0, -1, 0, -1, 1, 0, 1, -1, 1, -1, 0, 0, 0, 0]
    assert (
        infinitesimal_rigidity.is_vector_inf_flex(F, explicit_flex)
        and infinitesimal_rigidity.is_vector_nontrivial_inf_flex(F, explicit_flex)
        and infinitesimal_rigidity.is_vector_nontrivial_inf_flex(
            F, explicit_flex, numerical=True
        )
        and infinitesimal_rigidity.is_nontrivial_flex(F, explicit_flex, numerical=True)
    )
    QF = Matrix.hstack(*(infinitesimal_rigidity.inf_flexes(F, include_trivial=True)))
    Q_exp = Matrix(explicit_flex)
    assert QF.rank() == 5 and Matrix.hstack(QF, Q_exp).rank() == 5
    QF = Matrix.hstack(*(infinitesimal_rigidity.inf_flexes(F, include_trivial=False)))
    assert QF.rank() == 2 and Matrix.hstack(QF, Q_exp).rank() == 2

    F = fws.Complete(5)
    F = _to_FrameworkBase(F)
    F_triv = infinitesimal_rigidity.trivial_inf_flexes(F)
    for inf_flex in F_triv:
        dict_flex = infinitesimal_rigidity._transform_inf_flex_to_pointwise(F, inf_flex)
        assert infinitesimal_rigidity.is_dict_inf_flex(
            F, dict_flex
        ) and infinitesimal_rigidity.is_dict_trivial_inf_flex(F, dict_flex)
    F_all = infinitesimal_rigidity.inf_flexes(F, include_trivial=True)
    assert Matrix.hstack(*F_triv).rank() == Matrix.hstack(*(F_all + F_triv)).rank()

    F = Framework.Random(graphs.DoubleBanana(), dim=3)
    F = _to_FrameworkBase(F)
    inf_flexes = infinitesimal_rigidity.nontrivial_inf_flexes(F)
    dict_flex = infinitesimal_rigidity._transform_inf_flex_to_pointwise(
        F, inf_flexes[0]
    )
    assert infinitesimal_rigidity.is_dict_inf_flex(
        F, dict_flex
    ) and infinitesimal_rigidity.is_dict_nontrivial_inf_flex(F, dict_flex)
    assert Matrix.hstack(*inf_flexes).rank() == 1

    G = graphs.Complete(3)
    F = Framework(G, {0: [0, 0], 1: [1, 0], 2: [2, 0]})
    F = _to_FrameworkBase(F)
    inf_flexes = infinitesimal_rigidity.nontrivial_inf_flexes(F)
    assert len(inf_flexes) == 1
    dict_flex = infinitesimal_rigidity._transform_inf_flex_to_pointwise(
        F, inf_flexes[0]
    )
    assert infinitesimal_rigidity.is_dict_inf_flex(
        F, dict_flex
    ) and infinitesimal_rigidity.is_dict_nontrivial_inf_flex(F, dict_flex)
    assert Matrix.hstack(*inf_flexes).rank() == 1

    G = graphs.Complete(4)
    F = Framework(G, {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 0], 3: [0, 1, 0]})
    F = _to_FrameworkBase(F)
    inf_flexes = infinitesimal_rigidity.nontrivial_inf_flexes(F)
    assert len(inf_flexes) == 1
    dict_flex = infinitesimal_rigidity._transform_inf_flex_to_pointwise(
        F, inf_flexes[0]
    )
    assert infinitesimal_rigidity.is_dict_inf_flex(
        F, dict_flex
    ) and infinitesimal_rigidity.is_dict_nontrivial_inf_flex(F, dict_flex)
    assert Matrix.hstack(*inf_flexes).rank() == 1


def test_inf_flexes_numerical():
    F = fws.ThreePrism(realization="flexible")
    F = _to_FrameworkBase(F)
    C = Framework(graphs.Complete(6), realization=F.realization())
    C = _to_FrameworkBase(C)
    QF = np.hstack(infinitesimal_rigidity.nontrivial_inf_flexes(F, numerical=True))
    QC = infinitesimal_rigidity.nontrivial_inf_flexes(C, numerical=True)
    assert np.linalg.matrix_rank(QF) == 1 and len(QC) == 0

    F = fws.Path(4)
    F = _to_FrameworkBase(F)
    for inf_flex in infinitesimal_rigidity.nontrivial_inf_flexes(F, numerical=True):
        dict_flex = infinitesimal_rigidity._transform_inf_flex_to_pointwise(F, inf_flex)
        assert infinitesimal_rigidity.is_dict_inf_flex(
            F, dict_flex, numerical=True
        ) and infinitesimal_rigidity.is_dict_nontrivial_inf_flex(
            F, dict_flex, numerical=True, tolerance=1e-4
        )
    assert (
        np.linalg.matrix_rank(
            np.vstack(
                tuple(infinitesimal_rigidity.nontrivial_inf_flexes(F, numerical=True))
            )
        )
        == 2
    )

    F = fws.Frustum(4)
    F = _to_FrameworkBase(F)
    QF = np.vstack(
        tuple(
            infinitesimal_rigidity.inf_flexes(F, include_trivial=True, numerical=True)
        )
    )
    assert np.linalg.matrix_rank(QF) == 5
    QF = np.vstack(
        tuple(
            infinitesimal_rigidity.inf_flexes(F, include_trivial=False, numerical=True)
        )
    )
    assert np.linalg.matrix_rank(QF) == 2

    F = fws.Complete(5, dim=4)
    F = _to_FrameworkBase(F)
    assert (
        len(infinitesimal_rigidity.inf_flexes(F, include_trivial=True, numerical=True))
        == 10
    )

    @pytest.mark.parametrize(
        "framework",
        [
            Framework.Random(graphs.DoubleBanana(), dim=3),
            Framework(graphs.Complete(3), {0: [0, 0], 1: [1, 0], 2: [2, 0]}),
            Framework(
                graphs.Complete(4),
                {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 0], 3: [0, 1, 0]},
            ),
        ],
    )
    def test_nontrivial_inf_flexes_numerical(framework):
        F = _to_FrameworkBase(framework)
        inf_flexes = infinitesimal_rigidity.nontrivial_inf_flexes(F, numerical=True)
        assert len(inf_flexes) == 1
        dict_flex = infinitesimal_rigidity._transform_inf_flex_to_pointwise(
            F, inf_flexes[0]
        )
        assert infinitesimal_rigidity.is_dict_inf_flex(
            F, dict_flex, numerical=True, tolerance=1e-4
        ) and infinitesimal_rigidity.is_dict_nontrivial_inf_flex(
            F, dict_flex, numerical=True, tolerance=1e-4
        )
        assert np.linalg.matrix_rank(np.vstack(inf_flexes)) == 1


@pytest.mark.parametrize(
    "include_trivial, numerical",
    [[True, True], [True, False], [False, True], [False, False]],
)
@pytest.mark.parametrize(
    "framework",
    [
        fws.Path(4),
        fws.Cycle(5),
        fws.Frustum(3),
        fws.Frustum(4),
        fws.Frustum(6),
        fws.K33plusEdge(),
        fws.ThreePrism(realization="parallel"),
        fws.ThreePrism(realization="flexible"),
        fws.Octahedron(realization="regular"),
        Framework(
            fws.Cube().graph.cone(),
            fws.Cube().realization(as_points=True) | {8: ["1/2", "1/2", "1/2"]},
        ),
    ],
)
def test_vertex_fixing(framework, include_trivial, numerical):
    fixed_vertices = framework._graph.edge_list()[0]
    flexes = infinitesimal_rigidity.inf_flexes(
        _to_FrameworkBase(framework),
        include_trivial=include_trivial,
        numerical=numerical,
        fixed_vertices=fixed_vertices,
    )
    pointwise_zero_flexes = [
        infinitesimal_rigidity._transform_inf_flex_to_pointwise(
            _to_FrameworkBase(framework), flex
        )
        for flex in flexes
    ]
    assert all(
        is_zero_vector(point[v], numerical=numerical)
        for v in fixed_vertices
        for point in pointwise_zero_flexes
    )
    assert all(
        any(
            not is_zero_vector(point[v], numerical=numerical)
            for v in framework._graph.nodes
        )
        for point in pointwise_zero_flexes
    )


def test_is_vector_inf_flex():
    F = Framework.Complete([[0, 0], [1, 0], [0, 1]])
    F = _to_FrameworkBase(F)
    assert infinitesimal_rigidity.is_vector_inf_flex(F, [0, 0, 0, 1, -1, 0])
    assert not infinitesimal_rigidity.is_vector_inf_flex(F, [0, 0, 0, 1, -2, 0])
    assert infinitesimal_rigidity.is_vector_inf_flex(F, [0, 1, 0, 0, -1, 0], [1, 0, 2])

    F.delete_edge([1, 2])
    assert infinitesimal_rigidity.is_vector_inf_flex(F, [0, 0, 0, 1, -1, 0])
    assert infinitesimal_rigidity.is_vector_inf_flex(F, [0, 0, 0, -1, -2, 0])
    assert not infinitesimal_rigidity.is_vector_inf_flex(F, [0, 0, 2, 1, -2, 1])

    F = fws.ThreePrism(realization="flexible")
    F = _to_FrameworkBase(F)
    for inf_flex in infinitesimal_rigidity.inf_flexes(F, include_trivial=True):
        assert infinitesimal_rigidity.is_vector_inf_flex(F, inf_flex)


def test_is_dict_inf_flex():
    F = Framework.Complete([[0, 0], [1, 0], [0, 1]])
    F = _to_FrameworkBase(F)
    assert infinitesimal_rigidity.is_dict_inf_flex(
        F, {0: [0, 0], 1: [0, 1], 2: [-1, 0]}
    )
    assert not infinitesimal_rigidity.is_dict_inf_flex(
        F, {0: [0, 0], 1: [0, -1], 2: [-2, 0]}
    )

    F.delete_edge([1, 2])
    assert infinitesimal_rigidity.is_dict_inf_flex(
        F, {0: [0, 0], 1: [0, 1], 2: [-1, 0]}
    )
    assert infinitesimal_rigidity.is_dict_inf_flex(
        F, {0: [0, 0], 1: [0, -1], 2: [-2, 0]}
    )
    assert not infinitesimal_rigidity.is_dict_inf_flex(
        F, {0: [0, 0], 1: [2, 1], 2: [-2, 1]}
    )

    F = fws.ThreePrism(realization="flexible")
    F = _to_FrameworkBase(F)
    assert infinitesimal_rigidity.is_dict_inf_flex(
        F, {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [1, 0], 4: [1, 0], 5: [1, 0]}
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
    assert (
        infinitesimal_rigidity.rigidity_matrix(_to_FrameworkBase(framework))
        == rigidity_matrix
    )


def test_rigidity_matrix():
    F = fws.Complete(4, dim=3)
    F = _to_FrameworkBase(F)
    assert infinitesimal_rigidity.rigidity_matrix(F).shape == (6, 12)

    G = Graph([(0, "a"), ("b", "a"), ("b", 1.9), (1.9, 0)])
    F = Framework(G, {0: (0, 0), "a": (1, 0), "b": (1, 1), 1.9: (0, 1)})
    F = _to_FrameworkBase(F)
    vertex_order = ["a", 1.9, "b", 0]
    assert infinitesimal_rigidity.rigidity_matrix(
        F, vertex_order=vertex_order
    ) == Matrix(
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
    assert (
        infinitesimal_rigidity.rigidity_matrix_rank(_to_FrameworkBase(framework))
        == rank
    )
    assert (
        infinitesimal_rigidity.rigidity_matrix_rank(
            _to_FrameworkBase(framework), numerical=True
        )
        == rank
    )
