from copy import deepcopy

import pytest
from sympy import pi

import pyrigi.frameworkDB as fws
import pyrigi.graphDB as graphs
from pyrigi.framework import (
    Framework,
)
from pyrigi.framework import _general as framework_general
from pyrigi.graph import Graph
from pyrigi.misc.misc import point_to_vector, sympy_expr_to_float
from test import TEST_WRAPPED_FUNCTIONS
from test.framework import _to_FrameworkBase


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

    if TEST_WRAPPED_FUNCTIONS:
        F1 = _to_FrameworkBase(F1)
        assert framework_general.is_injective(F1)
        assert framework_general.is_injective(F1, numerical=True)

        F2 = _to_FrameworkBase(F2)
        assert not framework_general.is_injective(F2)
        assert not framework_general.is_injective(F2, numerical=True)

        # test symbolical injectivity with irrational numbers
        F3 = _to_FrameworkBase(F3)
        assert framework_general.is_injective(F3)
        assert framework_general.is_injective(F3, numerical=True)

        # test numerical injectivity
        F4 = _to_FrameworkBase(F4)
        assert framework_general.is_injective(F4, numerical=True)

        # test numerically not injective, but symbolically injective framework
        F5 = _to_FrameworkBase(F5)
        assert not framework_general.is_injective(F5, numerical=True, tolerance=1e-8)
        assert not framework_general.is_injective(F5, numerical=True, tolerance=1e-9)
        assert framework_general.is_injective(F5)

        # test tolerance in numerical injectivity check
        F6 = _to_FrameworkBase(F6)
        assert framework_general.is_injective(F6, numerical=True, tolerance=1e-9)
        assert framework_general.is_injective(F6)


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

    if TEST_WRAPPED_FUNCTIONS:
        F1 = _to_FrameworkBase(F1)
        assert framework_general.is_quasi_injective(F1)
        assert framework_general.is_quasi_injective(F1, numerical=True)

        # test framework that is quasi-injective, but not injective
        F1 = fws.Complete(4, 2)
        F1.set_vertex_pos(1, F1[2])
        F1.delete_edge((1, 2))
        F1 = _to_FrameworkBase(F1)
        assert framework_general.is_quasi_injective(F1)
        assert framework_general.is_quasi_injective(F1, numerical=True)

        # test not quasi-injective framework
        F2 = _to_FrameworkBase(F2)
        assert not framework_general.is_quasi_injective(F2)
        assert not framework_general.is_quasi_injective(F2, numerical=True)

        # test symbolical and numerical quasi-injectivity with irrational numbers
        F3 = _to_FrameworkBase(F3)
        assert framework_general.is_quasi_injective(F3)
        assert framework_general.is_quasi_injective(F3, numerical=True)

        # test numerical quasi-injectivity
        F4 = _to_FrameworkBase(F4)
        assert framework_general.is_quasi_injective(F4, numerical=True)

        # test numerically not quasi-injective, but symbolically quasi-injective framework
        F5 = _to_FrameworkBase(F5)
        assert not framework_general.is_quasi_injective(
            F5, numerical=True, tolerance=1e-8
        )
        assert not framework_general.is_quasi_injective(
            F5, numerical=True, tolerance=1e-9
        )
        assert framework_general.is_quasi_injective(F5)

        # test tolerance in numerical quasi-injectivity check
        F6 = _to_FrameworkBase(F6)
        assert framework_general.is_quasi_injective(F6, numerical=True, tolerance=1e-9)
        assert framework_general.is_quasi_injective(F6)


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

    F4 = F3.translate((1, 1), inplace=False)
    assert F3.is_equivalent(F4, numerical=True)
    assert F3.is_equivalent(F4)

    F5 = F3.rotate2D(pi / 2, inplace=False)
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

    if TEST_WRAPPED_FUNCTIONS:
        F1 = _to_FrameworkBase(F1)
        assert framework_general.is_equivalent_realization(
            F1, F1.realization(), numerical=False
        )
        assert framework_general.is_equivalent_realization(
            F1, F1.realization(), numerical=True
        )
        assert framework_general.is_equivalent(F1, F1)

        F2 = _to_FrameworkBase(F2)
        with pytest.raises(ValueError):
            framework_general.is_equivalent_realization(F1, F2.realization())

        with pytest.raises(ValueError):
            framework_general.is_equivalent(F1, F2)

        F3 = _to_FrameworkBase(F3)
        F4 = _to_FrameworkBase(F4)
        assert framework_general.is_equivalent(F3, F4, numerical=True)
        assert framework_general.is_equivalent(F3, F4)

        F5 = _to_FrameworkBase(F5)
        assert framework_general.is_equivalent(F5, F3)
        assert framework_general.is_equivalent(F5, F4)
        assert framework_general.is_equivalent_realization(F5, F4.realization())

        F6 = _to_FrameworkBase(F6)
        F7 = _to_FrameworkBase(F7)
        F8 = _to_FrameworkBase(F8)

        assert framework_general.is_equivalent(F6, F7)
        assert framework_general.is_equivalent(F6, F8)
        assert framework_general.is_equivalent(F7, F8)

        F9 = _to_FrameworkBase(F9)
        assert framework_general.is_equivalent(F5, F9)

        with pytest.raises(ValueError):
            assert framework_general.is_equivalent(F8, F2)

        # testing numerical equivalence
        assert not framework_general.is_equivalent_realization(F9, R1, numerical=False)
        assert framework_general.is_equivalent_realization(F9, R1, numerical=True)


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
    F5 = F1.rotate2D(pi / 2, inplace=False)
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

    if TEST_WRAPPED_FUNCTIONS:
        F1 = _to_FrameworkBase(F1)
        F2 = _to_FrameworkBase(F2)
        F3 = _to_FrameworkBase(F3)
        F4 = _to_FrameworkBase(F4)
        F5 = _to_FrameworkBase(F5)
        F6 = _to_FrameworkBase(F6)
        F7 = _to_FrameworkBase(F7)
        assert framework_general.is_congruent_realization(
            F1, F1.realization(), numerical=False
        )
        assert framework_general.is_congruent(F1, F1, numerical=False)
        assert framework_general.is_congruent(F1, F1, numerical=True)

        assert not framework_general.is_congruent(
            F1, F2
        )  # equivalent, but not congruent
        assert not framework_general.is_congruent(
            F1, F3
        )  # equivalent, but not congruent
        assert not framework_general.is_congruent(
            F2, F3
        )  # equivalent, but not congruent
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
