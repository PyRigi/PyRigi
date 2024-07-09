"""
Module for miscellaneous functions.
"""

import math
from pyrigi.data_type import Point, point_to_vector
from sympy import Matrix, simplify, Abs


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
    vector: Point, numerical: bool = False, tolerance: float = 1e-9
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
        for coord in vector:
            if not simplify(coord).is_zero:
                break
        else:
            return True
    else:
        for coord in vector:
            if Abs(coord) > tolerance:
                break
        else:
            return True
    return False
