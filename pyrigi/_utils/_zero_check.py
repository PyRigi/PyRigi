from math import isclose

import sympy as sp
from sympy import Matrix

from pyrigi.data_type import Number, Sequence

from ._conversion import point_to_vector, sympy_expr_to_float


def is_zero(expr: Number, numerical: bool = False, tolerance: float = 1e-9) -> bool:
    """
    Return if the given expression is zero.

    Parameters
    ----------
    expr:
        Expression that is checked.
    numerical:
        If ``True``, then the check is done only numerically with the given tolerance.
        If ``False`` (default), the check is done symbolically,
        ``sympy`` method ``equals`` is used.
    tolerance:
        The tolerance that is used in the numerical check coordinate-wise.
    """
    if not numerical:
        zero_bool = sp.cancel(sp.sympify(expr)).equals(0)
        if zero_bool is None:
            raise RuntimeError(
                "It could not be determined by sympy "
                + "whether the given sympy expression is zero."
                + "Please report this as an issue on Github "
                + "(https://github.com/PyRigi/PyRigi/issues)."
            )
        return zero_bool
    else:
        return isclose(
            sympy_expr_to_float(expr, tolerance=tolerance),
            0,
            abs_tol=tolerance,
        )


def is_zero_vector(
    vector: Sequence[Number], numerical: bool = False, tolerance: float = 1e-9
) -> bool:
    """
    Return if the given vector is zero.

    Parameters
    ----------
    vector:
        Vector that is checked.
    numerical:
        If ``True``, then the check is done only numerically with the given tolerance.
        If ``False`` (default), the check is done symbolically,
        ``sympy`` attribute ``is_zero`` is used.
    tolerance:
        The tolerance that is used in the numerical check coordinate-wise.
    """
    if not isinstance(vector, Matrix):
        vector = point_to_vector(vector)
    return all(
        [is_zero(coord, numerical=numerical, tolerance=tolerance) for coord in vector]
    )
