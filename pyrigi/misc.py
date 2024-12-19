"""
Module for miscellaneous functions.
"""

import math
from pyrigi.data_type import Coordinate, point_to_vector
from typing import List, Sequence
from sympy import Matrix
import numpy as np
from math import isclose, log10


def doc_category(category):
    def decorator_doc_category(func):
        setattr(func, "_doc_category", category)
        return func

    return decorator_doc_category


def generate_category_tables(cls, tabs, cat_order=[], include_all=False) -> str:
    categories = {}
    for func in dir(cls):
        if callable(getattr(cls, func)) and func[:2] != "__":
            f = getattr(cls, func)
            if hasattr(f, "_doc_category"):
                category = f._doc_category
                if category in categories:
                    categories[category].append(func)
                else:
                    categories[category] = [func]
            elif include_all:
                if "Not categorized" in categories:
                    categories["Not categorized"].append(func)
                else:
                    categories["Not categorized"] = [func]

    for category in categories:
        if category not in cat_order:
            cat_order.append(category)

    res = "Methods\n-------\n"
    for category, functions in sorted(
        categories.items(), key=lambda t: cat_order.index(t[0])
    ):
        res += f"**{category}**"
        res += "\n\n.. autosummary::\n\n    "
        res += "\n    ".join(functions)
        res += "\n\n"
    indent = "".join(["    " for _ in range(tabs)])
    return ("\n" + indent).join(res.splitlines())


def generate_two_orthonormal_vectors(dim: int, random_seed: int = None) -> Matrix:
    """
    Generate two random numeric orthonormal vectors in the given dimension.

    The vectors are in the columns of the returned matrix.

    Parameters
    ----------
    dim:
        The dimension in which the vectors are generated.
    random_seed:
        Seed for generating random vectors.
        When the same value is provided, the same vectors are generated.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    matrix = np.random.randn(dim, 2)

    # for numerical stability regenerate some elements
    tmp = np.random.randint(0, dim - 1)
    while abs(matrix[tmp, 1]) < 1e-6:
        matrix[tmp, 1] = np.random.randn(1, 1)

    while abs(matrix[-1, 0]) < 1e-6:
        matrix[-1, 0] = np.random.randn(1, 1)

    tmp = np.dot(matrix[:-1, 0], matrix[:-1, 1]) * -1
    matrix[-1, 1] = tmp / matrix[-1, 0]

    # normalize
    matrix[:, 0] = matrix[:, 0] / np.linalg.norm(matrix[:, 0])
    matrix[:, 1] = matrix[:, 1] / np.linalg.norm(matrix[:, 1])
    return matrix


def check_integrality_and_range(
    n: int, name: str = "number n", min_n: int = 0, max_n: int = math.inf
) -> None:
    if not isinstance(n, int):
        raise TypeError("The " + name + f" has to be an integer, not {type(n)}.")
    if n < min_n or n > max_n:
        raise ValueError(
            "The " + name + f" has to be an integer in [{min_n},{max_n}], not {n}."
        )


def is_zero_vector(
    vector: Sequence[Coordinate], numerical: bool = False, tolerance: float = 1e-9
) -> bool:
    """
    Check if the given vector is zero.

    Parameters
    ----------
    vector:
        Vector that is checked.
    numerical:
        If True, then the check is done only numerically with the given tolerance.
        If False (default), the check is done symbolically, sympy is_zero is used.
    tolerance:
        The tolerance that is used in the numerical check coordinate-wise.
    """
    if not isinstance(vector, Matrix):
        vector = point_to_vector(vector)

    if not numerical:
        return all([coord.is_zero for coord in vector])
    else:
        return all(
            [
                isclose(
                    coord,
                    0,
                    abs_tol=tolerance,
                )
                for coord in eval_sympy_vector(vector, tolerance=tolerance)
            ]
        )


def eval_sympy_vector(
    vector: Sequence[Coordinate], tolerance: float = 1e-9
) -> List[float]:
    """
    Converts a sympy vector to a (numerical) list of floats.

    Parameters
    ----------
    vector:
        The sympy vector.
    tolerance:
        Intended level of numerical accuracy.

    Notes
    -----
    The method :func:`.data_type.point_to_vector` is used to ensure that
    the input is consistent with the sympy format.
    """
    return [
        float(coord.evalf(int(round(2.5 * log10(tolerance ** (-1) + 1)))))
        for coord in point_to_vector(vector)
    ]
