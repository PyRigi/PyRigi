import pytest
from sympy import Matrix

import pyrigi.frameworkDB as fws
from pyrigi.framework import Framework
from pyrigi.framework._rigidity import stress as stress_rigidity
from pyrigi.graph import Graph
from test import TEST_WRAPPED_FUNCTIONS
from test.framework import _to_FrameworkBase


def test_stress_matrix():
    F = fws.Complete(4)
    M = Matrix([[1, -1, 1, -1], [-1, 1, -1, 1], [1, -1, 1, -1], [-1, 1, -1, 1]])
    stress = [1, -1, 1, 1, -1, 1]
    assert F.stress_matrix(stress) == M
    if TEST_WRAPPED_FUNCTIONS:
        assert stress_rigidity.stress_matrix(_to_FrameworkBase(F), stress) == M

    F = fws.Frustum(3)
    M = Matrix(
        [
            [10, -2, -2, -6, 0, 0],
            [-2, 10, -2, 0, -6, 0],
            [-2, -2, 10, 0, 0, -6],
            [-6, 0, 0, 4, 1, 1],
            [0, -6, 0, 1, 4, 1],
            [0, 0, -6, 1, 1, 4],
        ]
    )
    stress = [2, 2, 6, 2, 6, 6, -1, -1, -1]
    assert F.stress_matrix(stress) == M
    if TEST_WRAPPED_FUNCTIONS:
        assert stress_rigidity.stress_matrix(_to_FrameworkBase(F), stress) == M

    G = Graph([(0, "a"), ("b", "a"), ("b", 1.9), (1.9, 0), ("b", 0), ("a", 1.9)])
    F = Framework(G, {0: (0, 0), "a": (1, 0), "b": (1, 1), 1.9: (0, 1)})
    edge_order = [("a", 0), (1.9, "b"), (1.9, 0), ("a", "b"), ("a", 1.9), (0, "b")]
    M = Matrix([[-1, 1, -1, 1], [1, -1, 1, -1], [-1, 1, -1, 1], [1, -1, 1, -1]])
    stress = F.stresses(edge_order=edge_order)[0].transpose().tolist()[0]
    assert F.stress_matrix(stress, edge_order=edge_order) == M
    # if TEST_WRAPPED_FUNCTIONS:
    #     F = _to_FrameworkBase(F)
    #     stress = (
    #         stress_rigidity.stresses(F, edge_order=edge_order)[0]
    #         .transpose()
    #         .tolist()[0]
    #     )
    #     assert stress_rigidity.stress_matrix(F, stress) == M


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
        [fws.SecondOrderRigid(), 2],
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
        [fws.SecondOrderRigid(), 2],
        [fws.CompleteBipartite(3, 3, realization="collinear"), 4],
    ]
    + [[fws.Frustum(i), 1] for i in range(3, 8)],
)
def test_stresses_numerical(framework, num_stresses):
    stresses = framework.stresses(numerical=True)
    assert len(stresses) == num_stresses and all(
        [framework.is_stress(s, numerical=True) for s in stresses]
    )
    if TEST_WRAPPED_FUNCTIONS:
        F = _to_FrameworkBase(framework)
        stresses = stress_rigidity.stresses(F, numerical=True)
        assert len(stresses) == num_stresses and all(
            [stress_rigidity.is_stress(F, s, numerical=True) for s in stresses]
        )
