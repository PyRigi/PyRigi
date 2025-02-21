from pyrigi.misc import point_to_vector

import pytest
from sympy import Matrix, sqrt, Rational
import numpy as np


@pytest.mark.parametrize(
    "input_vect, output",
    [
        [[1, 2, 3], Matrix([1, 2, 3])],
        [(1, 2, 3), Matrix([1, 2, 3])],
        [[1.0], Matrix([1.0])],
        [(2.3,), Matrix([2.3])],
        [("sqrt(2)+1", 2, 3), Matrix([sqrt(2) + 1, 2, 3])],
        [("sin(pi/4)", "2/3"), Matrix([sqrt(2) / 2, Rational(2, 3)])],
        [np.array([[1.5234, 0.123]]), Matrix([1.5234, 0.123])],
    ],
)
def test_point_to_vector(input_vect, output):
    assert point_to_vector(input_vect) == output


def test_point_to_vector_error():
    with pytest.raises(TypeError):
        point_to_vector("12mkcd")
    with pytest.raises(TypeError):
        point_to_vector(12)
    with pytest.raises(ValueError):
        point_to_vector(["12-"])
    with pytest.raises(ValueError):
        point_to_vector(["sin(pi)^"])
