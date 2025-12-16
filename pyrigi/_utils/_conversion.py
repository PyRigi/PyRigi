from math import log10

import numpy as np
import sympy as sp
from sympy import Matrix, MatrixBase

from pyrigi.data_type import Number, Point, Sequence


def sympy_expr_to_float(
    expression: Sequence[Number] | Matrix | Number, tolerance: float = 1e-9
) -> list[float] | float:
    """
    Convert a sympy expression to (numerical) floats.

    If the given ``expression`` is a ``Sequence`` of ``Numbers`` or a ``Matrix``,
    then each individual element is evaluated and a list of ``float`` is returned.
    If the input is just a single sympy expression, it is evaluated and
    returned as a ``float``.

    Parameters
    ----------
    expression:
        The sympy expression.
    tolerance:
        Intended level of numerical accuracy.

    Notes
    -----
    The method :func:`.data_type.point_to_vector` is used to ensure that
    the input is consistent with the sympy format.
    """
    try:
        if isinstance(expression, list | tuple | Matrix):
            return [
                float(
                    sp.sympify(coord).evalf(
                        int(round(2.5 * log10(tolerance ** (-1) + 1)))
                    )
                )
                for coord in point_to_vector(expression)
            ]
        return float(
            sp.sympify(expression).evalf(int(round(2.5 * log10(tolerance ** (-1) + 1))))
        )
    except sp.SympifyError:
        raise ValueError(f"The expression `{expression}` could not be parsed by sympy.")


def point_to_vector(point: Point) -> Matrix:
    """
    Return point as single column sympy Matrix.
    """
    if isinstance(point, MatrixBase) or isinstance(point, np.ndarray):
        if (
            len(point.shape) > 1 and point.shape[0] != 1 and point.shape[1] != 1
        ) or len(point.shape) > 2:
            raise ValueError("Point could not be interpreted as column vector.")
        if isinstance(point, np.ndarray):
            point = np.array([point]) if len(point.shape) == 1 else point
            point = Matrix(
                [
                    [float(point[i, j]) for i in range(point.shape[0])]
                    for j in range(point.shape[1])
                ]
            )
        return point if (point.shape[1] == 1) else point.transpose()

    if not isinstance(point, Sequence) or isinstance(point, str):
        raise TypeError("The point must be a Sequence of Numbers.")

    try:
        res = Matrix(point)
    except Exception as e:
        raise ValueError("A coordinate could not be interpreted by sympify:\n" + str(e))

    if res.shape[0] != 1 and res.shape[1] != 1:
        raise ValueError("Point could not be interpreted as column vector.")
    return res if (res.shape[1] == 1) else res.transpose()


def _rgb_to_hex_array(color_array):
    """
    Map an array of colors to their hexadecimal representation or string
    representation so that `svg`-based methods can read the colors.
    """
    rgb_factor = 1
    for col in color_array:
        if not (isinstance(col, list) or isinstance(col, tuple) or isinstance(col, str)) or (isinstance(col, list) or isinstance(col, tuple)) and not len(col) == 3:
            raise ValueError("The `color_array` does not only consist of RGB and ")
        if (isinstance(col, list) or isinstance(col, tuple)) and any(isinstance(c, float) and c<1 for c in col):
            # Distinguish between rgb and RGB
            rgb_factor = 255
    return [
        "#" + "".join(f"{int(round(c * rgb_factor)):02x}" for c in col)
        if isinstance(col, list) or isinstance(col, tuple)
        else col
        for col in color_array
    ]