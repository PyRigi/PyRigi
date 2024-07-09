from pyrigi.data_type import point_to_vector

import pytest
from sympy import Matrix, sqrt, Rational


@pytest.mark.parametrize(
    "input, output",
    [
        [[1, 2, 3], Matrix([1, 2, 3])],
        [(1, 2, 3), Matrix([1, 2, 3])],
        [[1.0], Matrix([1.0])],
        [(2.3,), Matrix([2.3])],
        [("sqrt(2)+1", 2, 3), Matrix([sqrt(2) + 1, 2, 3])],
        [("sin(pi/4)", "2/3"), Matrix([sqrt(2) / 2, Rational(2, 3)])],
    ],
)
def test_point_to_vector(input, output):
    assert point_to_vector(input) == output


def test_point_to_vector_errors():
    with pytest.raises(TypeError):
        point_to_vector("12mkcd")
    with pytest.raises(TypeError):
        point_to_vector(12)
    with pytest.raises(ValueError):
        point_to_vector(["12-"])
    with pytest.raises(ValueError):
        point_to_vector(["sin(pi)^"])
