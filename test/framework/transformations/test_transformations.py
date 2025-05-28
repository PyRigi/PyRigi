import pytest
from sympy import Matrix, pi, sqrt

import pyrigi.frameworkDB as fws
import pyrigi.graphDB as graphs
from pyrigi.framework import Framework
from pyrigi.framework._general import is_congruent_realization
from pyrigi.framework._transformations import (
    transformations as framework_transformations,
)
from test import TEST_WRAPPED_FUNCTIONS
from test.framework import _to_FrameworkBase


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

    if TEST_WRAPPED_FUNCTIONS:
        F = Framework(G, {0: (0, 0), 1: (2, 0), 2: (1, 1)})
        F = _to_FrameworkBase(F)
        newF = framework_transformations.translate(F, (0, 0), False)
        for v, pos in newF.realization().items():
            assert pos.equals(F[v])

        translation = Matrix([[1], [1]])
        newF = framework_transformations.translate(F, translation, False)
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

    if TEST_WRAPPED_FUNCTIONS:
        F = Framework(G, {0: (-1, 0), 1: (2, 0), 2: (1, 1), 3: (3, -2)})
        F = _to_FrameworkBase(F)
        newF = framework_transformations.rescale(F, 1, False)
        for v, pos in newF.realization().items():
            assert pos.equals(F[v])

        newF = framework_transformations.rescale(F, 2, False)
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

    if TEST_WRAPPED_FUNCTIONS:
        F = fws.Complete(4, dim=3)
        F = _to_FrameworkBase(F)
        _r = framework_transformations.projected_realization(
            F, proj_dim=2, projection_matrix=Matrix([[0, 1, 1], [1, 0, 1]])
        )
        assert (
            len(_r) == 2
            and all([len(val) == 2 for val in _r[0].values()])
            and _r[0][0] == (0, 0)
            and _r[0][1] == (0, 1)
            and _r[0][2] == (1, 0)
            and _r[0][3] == (1, 1)
        )

        _r = framework_transformations.projected_realization(
            F, proj_dim=3, projection_matrix=Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        )
        assert (
            len(_r) == 2
            and all([len(val) == 3 for val in _r[0].values()])
            and is_congruent_realization(F, _r[0])
        )

        with pytest.raises(ValueError):
            framework_transformations.projected_realization(
                F, proj_dim=2, projection_matrix=Matrix([[0, 1, 1]])
            )
            framework_transformations.projected_realization(
                F, proj_dim=2, projection_matrix=Matrix([[0, 1], [1, 0]])
            )

        F = fws.Complete(6, dim=5)
        F = _to_FrameworkBase(F)
        with pytest.raises(ValueError):
            framework_transformations.projected_realization(F, proj_dim=4)


def test_rotate2D():
    G = graphs.Complete(3)
    F = Framework(G, {0: (0, 0), 1: (2, 0), 2: (1, 1)})

    newF = F.rotate2D(0, inplace=False)
    for v, pos in newF.realization().items():
        assert pos.equals(F[v])

    newF = F.rotate2D(pi * 4, inplace=False)
    for v, pos in newF.realization().items():
        assert pos.equals(F[v])

    newF = F.rotate2D(pi / 2, inplace=False)
    assert newF[0].equals(Matrix([[0], [0]]))
    assert newF[1].equals(Matrix([[0], [2]]))
    assert newF[2].equals(Matrix([[-1], [1]]))

    newF = F.rotate2D(pi / 4, inplace=False)
    assert newF[0].equals(Matrix([[0], [0]]))
    assert newF[1].equals(Matrix([[sqrt(2)], [(sqrt(2))]]))
    assert newF[2].equals(Matrix([[0], [sqrt(2)]]))

    F = Framework(G, {0: (0, 0), 1: (2, 0), 2: (0, 2)})
    newF = F.rotate2D(pi, rotation_center=[1, 1], inplace=False)
    assert newF[0].equals(Matrix([[2], [2]]))
    assert newF[1].equals(Matrix([[0], [2]]))
    assert newF[2].equals(Matrix([[2], [0]]))

    if TEST_WRAPPED_FUNCTIONS:
        F = Framework(G, {0: (0, 0), 1: (2, 0), 2: (1, 1)})
        F = _to_FrameworkBase(F)
        newF = framework_transformations.rotate2D(F, 0, inplace=False)
        for v, pos in newF.realization().items():
            assert pos.equals(F[v])

        newF = framework_transformations.rotate2D(F, pi * 4, inplace=False)
        for v, pos in newF.realization().items():
            assert pos.equals(F[v])

        newF = framework_transformations.rotate2D(F, pi / 2, inplace=False)
        assert newF[0].equals(Matrix([[0], [0]]))
        assert newF[1].equals(Matrix([[0], [2]]))
        assert newF[2].equals(Matrix([[-1], [1]]))

        newF = framework_transformations.rotate2D(F, pi / 4, inplace=False)
        assert newF[0].equals(Matrix([[0], [0]]))
        assert newF[1].equals(Matrix([[sqrt(2)], [(sqrt(2))]]))
        assert newF[2].equals(Matrix([[0], [sqrt(2)]]))

        F = Framework(G, {0: (0, 0), 1: (2, 0), 2: (0, 2)})
        F = _to_FrameworkBase(F)
        newF = framework_transformations.rotate2D(
            F, pi, rotation_center=[1, 1], inplace=False
        )
        assert newF[0].equals(Matrix([[2], [2]]))
        assert newF[1].equals(Matrix([[0], [2]]))
        assert newF[2].equals(Matrix([[2], [0]]))


def test_rotate3D():
    G = graphs.Complete(3)
    F = Framework(G, {0: (0, 0, 0), 1: (2, 0, 0), 2: (1, 1, 0)})

    newF = F.rotate3D(0, inplace=False)
    for v, pos in newF.realization().items():
        assert pos.equals(F[v])

    newF = F.rotate3D(pi * 4, inplace=False)
    for v, pos in newF.realization().items():
        assert pos.equals(F[v])

    newF = F.rotate3D(pi / 2, inplace=False)
    assert newF[0].equals(Matrix([[0], [0], [0]]))
    assert newF[1].equals(Matrix([[0], [2], [0]]))
    assert newF[2].equals(Matrix([[-1], [1], [0]]))

    newF = F.rotate3D(pi / 4, inplace=False)
    assert newF[0].equals(Matrix([[0], [0], [0]]))
    assert newF[1].equals(Matrix([[sqrt(2)], [(sqrt(2))], [0]]))
    assert newF[2].equals(Matrix([[0], [sqrt(2)], [0]]))

    F.rotate3D(pi / 2, axis_direction=[0, 1, 0], inplace=True)
    assert F[0].equals(Matrix([[0], [0], [0]]))
    assert F[1].equals(Matrix([[0], [0], [-2]]))
    assert F[2].equals(Matrix([[0], [1], [-1]]))

    F = Framework(G, {0: (1, 0, 0), 1: (0, 1, 0), 2: (0, 0, 1)})
    newF = F.rotate3D(2 * pi / 3, axis_direction=[1, 1, 1], inplace=False)
    assert newF[0].equals(Matrix([[0], [1], [0]]))
    assert newF[1].equals(Matrix([[0], [0], [1]]))
    assert newF[2].equals(Matrix([[1], [0], [0]]))

    F.rotate3D(4 * pi / 3, axis_direction=[1, 1, 1], inplace=True)
    assert F[0].equals(Matrix([[0], [0], [1]]))
    assert F[1].equals(Matrix([[1], [0], [0]]))
    assert F[2].equals(Matrix([[0], [1], [0]]))

    F = Framework(G, {0: (0, 0, 0), 1: (2, 0, 0), 2: (0, 2, 0)})
    F.rotate3D(pi, axis_shift=[1, 1, 0], inplace=True)
    assert F[0].equals(Matrix([[2], [2], [0]]))
    assert F[1].equals(Matrix([[0], [2], [0]]))
    assert F[2].equals(Matrix([[2], [0], [0]]))

    if TEST_WRAPPED_FUNCTIONS:
        F = Framework(G, {0: (0, 0, 0), 1: (2, 0, 0), 2: (1, 1, 0)})
        F = _to_FrameworkBase(F)

        newF = framework_transformations.rotate3D(F, 0, inplace=False)
        for v, pos in newF.realization().items():
            assert pos.equals(F[v])

        newF = framework_transformations.rotate3D(F, pi * 4, inplace=False)
        for v, pos in newF.realization().items():
            assert pos.equals(F[v])

        newF = framework_transformations.rotate3D(F, pi / 2, inplace=False)
        assert newF[0].equals(Matrix([[0], [0], [0]]))
        assert newF[1].equals(Matrix([[0], [2], [0]]))
        assert newF[2].equals(Matrix([[-1], [1], [0]]))

        newF = framework_transformations.rotate3D(F, pi / 4, inplace=False)
        assert newF[0].equals(Matrix([[0], [0], [0]]))
        assert newF[1].equals(Matrix([[sqrt(2)], [(sqrt(2))], [0]]))
        assert newF[2].equals(Matrix([[0], [sqrt(2)], [0]]))

        framework_transformations.rotate3D(
            F, pi / 2, axis_direction=[0, 1, 0], inplace=True
        )
        assert F[0].equals(Matrix([[0], [0], [0]]))
        assert F[1].equals(Matrix([[0], [0], [-2]]))
        assert F[2].equals(Matrix([[0], [1], [-1]]))

        F = Framework(G, {0: (1, 0, 0), 1: (0, 1, 0), 2: (0, 0, 1)})
        newF = F.rotate3D(2 * pi / 3, axis_direction=[1, 1, 1], inplace=False)
        assert newF[0].equals(Matrix([[0], [1], [0]]))
        assert newF[1].equals(Matrix([[0], [0], [1]]))
        assert newF[2].equals(Matrix([[1], [0], [0]]))

        F.rotate3D(4 * pi / 3, axis_direction=[1, 1, 1], inplace=True)
        assert F[0].equals(Matrix([[0], [0], [1]]))
        assert F[1].equals(Matrix([[1], [0], [0]]))
        assert F[2].equals(Matrix([[0], [1], [0]]))

        F = Framework(G, {0: (0, 0, 0), 1: (2, 0, 0), 2: (0, 2, 0)})
        F.rotate3D(pi, axis_shift=[1, 1, 0], inplace=True)
        assert F[0].equals(Matrix([[2], [2], [0]]))
        assert F[1].equals(Matrix([[0], [2], [0]]))
        assert F[2].equals(Matrix([[2], [0], [0]]))
