import math
from math import isclose, pi
from random import randint

import numpy as np
import pytest
import sympy as sp

from pyrigi._utils._zero_check import is_zero
from pyrigi._utils.linear_algebra import (
    _generate_two_orthonormal_vectors,
    _normalize_flex,
    _vector_distance_pointwise,
)


def test__generate_two_orthonormal_vectors():
    for _ in range(15):
        m = _generate_two_orthonormal_vectors(randint(2, 10))
        assert abs(np.dot(m[:, 0], m[:, 1])) < 1e-9
        assert abs(np.linalg.norm(m[:, 0])) - 1 < 1e-9
        assert abs(np.linalg.norm(m[:, 1])) - 1 < 1e-9


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
