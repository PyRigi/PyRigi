import math
from math import isclose, pi
from random import randint

import pytest
import numpy as np
import sympy as sp

from pyrigi.graph import Graph
from pyrigi.misc import (
    is_zero_vector,
    _generate_two_orthonormal_vectors,
    sympy_expr_to_float,
    is_isomorphic_graph_list,
    _normalize_flex,
    _vector_distance_pointwise,
    point_to_vector,
    is_zero,
)


def test_is_zero_vector():
    V1 = point_to_vector([0, 0])
    assert is_zero_vector(V1)
    assert is_zero_vector(V1, numerical=True)

    V2 = point_to_vector([1, 0])
    assert not is_zero_vector(V2)
    assert not is_zero_vector(V2, numerical=True)

    V3 = point_to_vector([0, 1])
    assert not is_zero_vector(V3)
    assert not is_zero_vector(V3, numerical=True)

    # test symbolic check
    V4 = ["(2/3)^2 - 8/18", "sqrt(2)^2 - 2"]
    assert is_zero_vector(V4)
    assert is_zero_vector(V4, numerical=True)

    # test tolerance
    V5 = point_to_vector([1e-10, 1e-10])
    assert not is_zero_vector(V5)
    assert is_zero_vector(V5, numerical=True, tolerance=1e-9)

    V6 = point_to_vector([1e-8, 1e-8])
    assert not is_zero_vector(V6)
    assert not is_zero_vector(V6, numerical=True, tolerance=1e-9)


def test__generate_two_orthonormal_vectors():
    for _ in range(15):
        m = _generate_two_orthonormal_vectors(randint(2, 10))
        assert abs(np.dot(m[:, 0], m[:, 1])) < 1e-9
        assert abs(np.linalg.norm(m[:, 0])) - 1 < 1e-9
        assert abs(np.linalg.norm(m[:, 1])) - 1 < 1e-9


def test_sympy_expr_to_float():
    with pytest.raises(ValueError):
        sympy_expr_to_float("12mkcd")
        sympy_expr_to_float("sin(pi)^")
        sympy_expr_to_float(["sin(pi)^"])
    assert sympy_expr_to_float(["cos(0)", "sin(pi)"]) == [1, 0]
    assert sympy_expr_to_float(
        ["sqrt(2)^2", "0.123123123123123123123123123123123"]
    ) == [
        2,
        0.12312312312312312,
    ]
    assert sympy_expr_to_float(["1/4", -1]) == [0.25, -1]
    assert sympy_expr_to_float("1/4") == 0.25


def test__normalize_flex():
    flex = _normalize_flex([1, 0, 1])
    assert sum(p**2 for p in flex) == 1
    flex = _normalize_flex([1, 0, 1, -2, 3], numerical=True)
    assert isclose(np.linalg.norm(flex), 1.0)
    flex = _normalize_flex(
        {0: [1.0, 0.0], 1: [1.0, -2.5], 2: [pi, np.sqrt(15)]}, numerical=True
    )
    assert isclose(np.linalg.norm(sum([list(val) for val in flex.values()], [])), 1.0)
    flex = _normalize_flex(
        {0: (1, 0), 1: (sp.cos(1), sp.sin(2)), 2: (sp.sqrt(5), sp.Rational(1 / 2))}
    )
    assert sp.simplify(sum(sum(p**2 for p in pt) for pt in flex.values())) == 1

    with pytest.raises(ValueError):
        _normalize_flex([0])
        _normalize_flex([0], numerical=True)


def test_vector_distance_pointwise():
    assert is_zero(_vector_distance_pointwise({0: [1, 1]}, {0: [1, 1]}))
    assert is_zero(
        _vector_distance_pointwise({0: [1, 1], 1: [1, -1]}, {0: [1, -1], 1: [1, -1]})
        - 2
    )
    assert isclose(
        _vector_distance_pointwise({0: [1, 1]}, {0: [1, -1]}, numerical=True), 2
    )
    assert isclose(
        _vector_distance_pointwise(
            {0: [1, 1], 1: [1, -1]}, {0: [1, -1], 1: [1, 1]}, numerical=True
        ),
        math.sqrt(8),
    )

    with pytest.raises(ValueError):
        _vector_distance_pointwise({0: [1, 1]}, {1: [1, 1]})


@pytest.mark.parametrize(
    "list1, list2",
    [
        [[239, 254, 254], [254, 254, 239]],
        [[239, 254], [254, 239]],
        [[31], [31]],
        [[254], [947]],
    ],
)
def test_is_isomorphic_graph_list(list1, list2):
    assert is_isomorphic_graph_list(
        [Graph.from_int(g) for g in list1], [Graph.from_int(g) for g in list2]
    )


@pytest.mark.parametrize(
    "list1, list2",
    [
        [[239, 254, 254], [254, 31, 239]],
        [[239, 254, 254], [254, 239, 239]],
        [[239, 254, 254], [254, 239]],
        [[239, 254], [31, 239]],
    ],
)
def test_is_not_isomorphic_graph_list(list1, list2):
    assert not is_isomorphic_graph_list(
        [Graph.from_int(g) for g in list1], [Graph.from_int(g) for g in list2]
    )
