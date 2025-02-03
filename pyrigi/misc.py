"""
Module for miscellaneous functions.
"""

from pyrigi.data_type import Sequence, Number, point_to_vector, InfFlex, Vertex
from sympy import Matrix
import sympy as sp
import numpy as np
from math import isclose, log10


try:
    from IPython.core.magic import register_cell_magic

    @register_cell_magic
    def skip_execution(line, cell):
        print(
            "This cell was marked to be skipped (probably due to its long execution time."
        )
        print("Remove the cell magic `%%skip_execution` to run it.")
        return

except NameError:
    pass


def doc_category(category):
    def decorator_doc_category(func):
        setattr(func, "_doc_category", category)
        return func

    return decorator_doc_category


def generate_category_tables(cls, tabs, cat_order=None, include_all=False) -> str:
    if cat_order is None:
        cat_order = []
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


def generate_three_orthonormal_vectors(dim: int, random_seed: int = None) -> Matrix:
    """
    Generate three random numeric orthonormal vectors in the given dimension.

    Notes
    -----
    The vectors are in the columns of the returned matrix. To ensure that the
    vectors are uniformly distributed over the Stiefel manifold, we need to
    ensure that the triangular matrix `R` has positive diagonal elements.

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

    matrix = np.random.randn(dim, 3)
    Q, R = np.linalg.qr(matrix)
    return Q @ np.diag(np.sign(np.diag(R)))


def is_zero_vector(
    vector: Sequence[Number], numerical: bool = False, tolerance: float = 1e-9
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
    vector: Sequence[Number] | Matrix, tolerance: float = 1e-9
) -> list[float]:
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


def normalize_flex(inf_flex: InfFlex, numerical: bool = False) -> InfFlex:
    """
    Divides a vector by its Euclidean norm.
    """
    if isinstance(inf_flex, dict):
        if numerical:
            _inf_flex = {
                v: [float(sp.sympify(q).evalf(15)) for q in flex]
                for v, flex in inf_flex.items()
            }
            flex_norm = np.linalg.norm(sum(_inf_flex.values(), []))
            return {
                v: tuple([pt / flex_norm for pt in q]) for v, q in _inf_flex.items()
            }
        flex_norm = sp.sqrt(sum([q**2 for val in inf_flex.values() for q in val]))
        return {v: tuple([pt / flex_norm for pt in q]) for v, q in inf_flex.items()}
    elif isinstance(inf_flex, Sequence):
        if numerical:
            _inf_flex = [float(sp.sympify(q).evalf(15)) for q in inf_flex]
            flex_norm = np.linalg.norm(_inf_flex)
            return [q / flex_norm for q in _inf_flex]
        flex_norm = sp.sqrt(sum([q**2 for q in inf_flex]))
        return [q / flex_norm for q in inf_flex]
    else:
        raise TypeError("`inf_flex` does not have the correct type.")


def vector_distance_pointwise(
    dict1: dict[Vertex, Sequence[Number]],
    dict2: dict[Vertex, Sequence[Number]],
    numerical: bool = False,
) -> float:
    """
    Computes the Euclidean distance between two realizations or pointwise vectors.

    This method computes the Euclidean distance from the realization `dict_1`
    to `dict2`. These dicts need to be based on the same vertex set.
    """
    if not set(dict1.keys()) == set(dict2.keys()) or not len(dict1) == len(dict2):
        raise ValueError("`dict1` and `dict2` are not based on the same vertex set.")
    if numerical:
        return float(
            np.linalg.norm(
                [
                    p1 - p2
                    for v in dict1.keys()
                    for p1, p2 in zip(
                        dict1[v],
                        dict2[v],
                    )
                ]
            )
        )
    return sp.sqrt(
        sum(
            [
                (p1 - p2) ** 2
                for v in dict1.keys()
                for p1, p2 in zip(
                    dict1[v],
                    dict2[v],
                )
            ]
        )
    )


def is_isomorphic_graph_list(list1, list2):
    """
    Check whether two lists of graphs are the same up to graph isomorphism.
    """
    if len(list1) != len(list2):
        return False
    for graph1 in list1:
        count_copies = 0
        for grapht in list1:
            if graph1.is_isomorphic(grapht):
                count_copies += 1
        count_found = 0
        for graph2 in list2:
            if graph1.is_isomorphic(graph2):
                count_found += 1
                if count_found == count_copies:
                    break
        else:
            return False
    return True
