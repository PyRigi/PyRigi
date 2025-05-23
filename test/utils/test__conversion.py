import pytest

from pyrigi._utils._conversion import sympy_expr_to_float


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
