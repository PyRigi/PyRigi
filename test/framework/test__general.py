from copy import deepcopy

import pytest
from sympy import pi

import pyrigi.frameworkDB as fws
import pyrigi.graphDB as graphs
from pyrigi._utils._conversion import point_to_vector, sympy_expr_to_float
from pyrigi.framework import (
    Framework,
)
from pyrigi.framework import _general as framework_general
from pyrigi.framework._transformations import (
    transformations as framework_transformations,
)
from pyrigi.graph import Graph
from test.framework import _to_FrameworkBase


def test_is_injective():
    F1 = fws.Complete(4, 2)
    F1 = _to_FrameworkBase(F1)

    F2 = deepcopy(F1)
    F2 = _to_FrameworkBase(F2)
    F2.set_vertex_pos(0, F2[1])

    # test symbolical injectivity with irrational numbers
    F3 = framework_transformations.translate(F1, ["sqrt(2)", "pi"], inplace=False)
    framework_transformations.rotate2D(F3, pi / 3, inplace=True)

    # test numerical injectivity
    F4 = deepcopy(F3)
    F4.set_realization(F4.realization(numerical=True))

    # test numerically not injective, but symbolically injective framework
    F5 = deepcopy(F3)
    F5.set_vertex_pos(0, F5[1] + point_to_vector([1e-10, 1e-10]))

    # test tolerance in numerical injectivity check
    F6 = deepcopy(F3)
    F6.set_vertex_pos(0, F6[1] + point_to_vector([1e-8, 1e-8]))

    assert framework_general.is_injective(F1)
    assert framework_general.is_injective(F1, numerical=True)

    assert not framework_general.is_injective(F2)
    assert not framework_general.is_injective(F2, numerical=True)

    # test symbolical injectivity with irrational numbers
    assert framework_general.is_injective(F3)
    assert framework_general.is_injective(F3, numerical=True)

    # test numerical injectivity
    assert framework_general.is_injective(F4, numerical=True)

    # test numerically not injective, but symbolically injective framework
    assert not framework_general.is_injective(F5, numerical=True, tolerance=1e-8)
    assert not framework_general.is_injective(F5, numerical=True, tolerance=1e-9)
    assert framework_general.is_injective(F5)

    # test tolerance in numerical injectivity check
    F6 = _to_FrameworkBase(F6)
    assert framework_general.is_injective(F6, numerical=True, tolerance=1e-9)
    assert framework_general.is_injective(F6)


def test_is_quasi_injective():
    F1 = fws.Complete(4, 2)
    F1 = _to_FrameworkBase(F1)

    # test framework that is quasi-injective, but not injective
    F1.set_vertex_pos(1, F1[2])
    F1.delete_edge((1, 2))

    # test not quasi-injective framework
    F2 = deepcopy(F1)
    F2.set_vertex_pos(0, F2[1])

    # test symbolical and numerical quasi-injectivity with irrational numbers
    F3 = framework_transformations.translate(F1, ["sqrt(2)", "pi"], inplace=False)
    F3 = framework_transformations.rotate2D(F3, pi / 2, inplace=False)

    # test numerical quasi-injectivity
    F4 = deepcopy(F3)
    F4.set_realization(F4.realization(numerical=True))

    # test numerically not quasi-injective, but symbolically quasi-injective framework
    F5 = deepcopy(F3)
    F5.set_vertex_pos(0, F5[1] + point_to_vector([1e-10, 1e-10]))

    # test tolerance in numerical quasi-injectivity check
    F6 = deepcopy(F3)
    F6.set_vertex_pos(0, F6[1] + point_to_vector([1e-8, 1e-8]))

    assert framework_general.is_quasi_injective(F1)
    assert framework_general.is_quasi_injective(F1, numerical=True)

    # test framework that is quasi-injective, but not injective
    F1 = fws.Complete(4, 2)
    F1 = _to_FrameworkBase(F1)
    F1.set_vertex_pos(1, F1[2])
    F1.delete_edge((1, 2))
    assert framework_general.is_quasi_injective(F1)
    assert framework_general.is_quasi_injective(F1, numerical=True)

    # test not quasi-injective framework
    assert not framework_general.is_quasi_injective(F2)
    assert not framework_general.is_quasi_injective(F2, numerical=True)

    # test symbolical and numerical quasi-injectivity with irrational numbers
    assert framework_general.is_quasi_injective(F3)
    assert framework_general.is_quasi_injective(F3, numerical=True)

    # test numerical quasi-injectivity
    assert framework_general.is_quasi_injective(F4, numerical=True)

    # test numerically not quasi-injective, but symbolically quasi-injective framework
    assert not framework_general.is_quasi_injective(F5, numerical=True, tolerance=1e-8)
    assert not framework_general.is_quasi_injective(F5, numerical=True, tolerance=1e-9)
    assert framework_general.is_quasi_injective(F5)

    # test tolerance in numerical quasi-injectivity check
    assert framework_general.is_quasi_injective(F6, numerical=True, tolerance=1e-9)
    assert framework_general.is_quasi_injective(F6)


def test_is_equivalent():
    F1 = fws.Complete(4, 2)
    F1 = _to_FrameworkBase(F1)

    F2 = fws.Complete(3, 2)
    F2 = _to_FrameworkBase(F2)

    G1 = graphs.ThreePrism()
    G1.delete_vertex(5)

    F3 = Framework(G1, {0: [0, 0], 1: [3, 0], 2: [2, 1], 3: [0, 4], 4: ["5/2", "9/7"]})
    F3 = _to_FrameworkBase(F3)

    F4 = framework_transformations.translate(F3, (1, 1), inplace=False)

    F5 = framework_transformations.rotate2D(F3, pi / 2, inplace=False)

    G2 = Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [3, 4]])
    F6 = Framework(G2, {0: [0, 0], 1: [3, 0], 2: [2, 1], 3: [0, 4], 4: ["5/2", "17/7"]})
    F6 = _to_FrameworkBase(F6)

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
    F7 = _to_FrameworkBase(F7)

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
    F8 = _to_FrameworkBase(F8)

    F9 = framework_transformations.translate(F5, (pi, "2/3"), False)

    # testing numerical equivalence

    R1 = {v: sympy_expr_to_float(pos) for v, pos in F9.realization().items()}

    assert framework_general.is_equivalent_realization(
        F1, F1.realization(), numerical=False
    )
    assert framework_general.is_equivalent_realization(
        F1, F1.realization(), numerical=True
    )
    assert framework_general.is_equivalent(F1, F1)

    with pytest.raises(ValueError):
        framework_general.is_equivalent_realization(F1, F2.realization())

    with pytest.raises(ValueError):
        framework_general.is_equivalent(F1, F2)

    assert framework_general.is_equivalent(F3, F4, numerical=True)
    assert framework_general.is_equivalent(F3, F4)

    assert framework_general.is_equivalent(F5, F3)
    assert framework_general.is_equivalent(F5, F4)
    assert framework_general.is_equivalent_realization(F5, F4.realization())

    assert framework_general.is_equivalent(F6, F7)
    assert framework_general.is_equivalent(F6, F8)
    assert framework_general.is_equivalent(F7, F8)

    assert framework_general.is_equivalent(F5, F9)

    with pytest.raises(ValueError):
        assert framework_general.is_equivalent(F8, F2)

    # testing numerical equivalence
    assert not framework_general.is_equivalent_realization(F9, R1, numerical=False)
    assert framework_general.is_equivalent_realization(F9, R1, numerical=True)


def test_is_congruent():
    G1 = Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [3, 4]])
    F1 = Framework(G1, {0: [0, 0], 1: [3, 0], 2: [2, 1], 3: [0, 4], 4: ["5/2", "17/7"]})
    F1 = _to_FrameworkBase(F1)

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
    F2 = _to_FrameworkBase(F2)

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
    F3 = _to_FrameworkBase(F3)

    F4 = framework_transformations.translate(F1, (pi, "2/3"), False)
    F5 = framework_transformations.rotate2D(F1, pi / 2, inplace=False)

    F6 = fws.Complete(4, 2)
    F6 = _to_FrameworkBase(F6)
    F7 = fws.Complete(3, 2)
    F7 = _to_FrameworkBase(F7)

    # testing numerical congruence
    R1 = {v: sympy_expr_to_float(pos) for v, pos in F4.realization().items()}

    assert framework_general.is_congruent_realization(
        F1, F1.realization(), numerical=False
    )
    assert framework_general.is_congruent(F1, F1, numerical=False)
    assert framework_general.is_congruent(F1, F1, numerical=True)

    assert not framework_general.is_congruent(F1, F2)  # equivalent, but not congruent
    assert not framework_general.is_congruent(F1, F3)  # equivalent, but not congruent
    assert not framework_general.is_congruent(F2, F3)  # equivalent, but not congruent
    assert not framework_general.is_congruent(
        F1, F2, numerical=True
    )  # equivalent, but not congruent
    assert not framework_general.is_congruent(
        F1, F3, numerical=True
    )  # equivalent, but not congruent
    assert not framework_general.is_congruent(
        F2, F3, numerical=True
    )  # equivalent, but not congruent

    assert framework_general.is_congruent(F1, F4)
    assert framework_general.is_congruent(F1, F5)
    assert framework_general.is_congruent(F5, F4)

    with pytest.raises(ValueError):
        assert framework_general.is_congruent(F6, F7)

    # testing numerical congruence
    assert not framework_general.is_congruent_realization(F4, R1)
    assert framework_general.is_congruent_realization(F4, R1, numerical=True)
