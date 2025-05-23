"""
This module provides various following miscellaneous functions.
"""

from math import isclose, log10

import networkx as nx
import numpy as np
import sympy as sp
from sympy import Matrix, MatrixBase

from pyrigi.data_type import Number, Point, Sequence

try:
    from IPython.core.magic import register_cell_magic

    @register_cell_magic
    def skip_execution(line, cell):  # noqa: U100
        print(
            "This cell was marked to be skipped (probably due to long execution time)."
        )
        print("Remove the cell magic `%%skip_execution` to run it.")
        return

except NameError:

    def skip_execution():
        pass


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


def is_isomorphic_graph_list(list1: list[nx.Graph], list2: list[nx.Graph]) -> bool:
    """
    Return whether two lists of graphs are the same up to graph isomorphism.
    """
    if len(list1) != len(list2):
        return False
    for graph1 in list1:
        count_copies = 0
        for grapht in list1:
            if nx.is_isomorphic(graph1, grapht):
                count_copies += 1
        count_found = 0
        for graph2 in list2:
            if nx.is_isomorphic(graph1, graph2):
                count_found += 1
                if count_found == count_copies:
                    break
        else:
            return False
    return True


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
