"""
This module provides various following miscellaneous functions.
"""

from math import isclose, log10

import numpy as np
import sympy as sp
from sympy import Matrix, MatrixBase

from pyrigi.data_type import Sequence, Number, InfFlex, Vertex, Point


try:
    from IPython.core.magic import register_cell_magic

    @register_cell_magic
    def skip_execution(line, cell):
        print(
            "This cell was marked to be skipped (probably due to long execution time)."
        )
        print("Remove the cell magic `%%skip_execution` to run it.")
        return

except NameError:
    pass


def _doc_category(category):
    """
    Decorator for doc categories.
    """

    def decorator_doc_category(func):
        setattr(func, "_doc_category", category)
        return func

    return decorator_doc_category


def _generate_category_tables(
    cls, tabs, cat_order=None, include_all=False, add_attributes=True
) -> str:
    """
    Generate a formatted string that categorizes methods of a given class.

    Parameters
    ----------
    cls:
        A class.
    tabs:
        The number of indentation levels that are applied to the output.
    cat_order:
        Optional list specifying the order in which categories appear
        in the output.
    include_all:
        Optional boolean determining whether methods without a specific category
        should be included.
    add_attributes:
        Optional boolean determining whether the public attributes should
        be listed among attribute getters.
    """
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
        elif isinstance(getattr(cls, func), property) and add_attributes:
            category = "Attribute getters"
            if category in categories:
                categories[category].append(func)
            else:
                categories[category] = [func]

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


def _generate_two_orthonormal_vectors(dim: int, random_seed: int = None) -> Matrix:
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


def _generate_three_orthonormal_vectors(dim: int, random_seed: int = None) -> Matrix:
    """
    Generate three random numeric orthonormal vectors in the given dimension.

    Parameters
    ----------
    dim:
        The dimension in which the vectors are generated.
    random_seed:
        Seed for generating random vectors.
        When the same value is provided, the same vectors are generated.

    Notes
    -----
    The vectors are in the columns of the returned matrix. To ensure that the
    vectors are uniformly distributed over the Stiefel manifold, we need to
    ensure that the triangular matrix `R` has positive diagonal elements.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    matrix = np.random.randn(dim, 3)
    Q, R = np.linalg.qr(matrix)
    return Q @ np.diag(np.sign(np.diag(R)))


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


def _normalize_flex(
    inf_flex: InfFlex, numerical: bool = False, tolerance: float = 1e-9
) -> InfFlex:
    """
    Divide a vector by its Euclidean norm.

    Parameters
    ----------
    inf_flex:
        The infinitesimal flex that is supposed to be normalized.
    numerical:
        Boolean determining whether a numerical or symbolic normalization is performed.
    tolerance:
        Intended level of numerical accuracy.
    """
    if isinstance(inf_flex, dict):
        if numerical:
            _inf_flex = {
                v: sympy_expr_to_float(flex, tolerance=tolerance)
                for v, flex in inf_flex.items()
            }
            flex_norm = np.linalg.norm(sum(_inf_flex.values(), []))
            if isclose(flex_norm, 0, abs_tol=tolerance):
                raise ValueError("The norm of this flex is almost zero.")
            return {
                v: tuple([q / flex_norm for q in flex]) for v, flex in _inf_flex.items()
            }
        flex_norm = sp.sqrt(sum([q**2 for flex in inf_flex.values() for q in flex]))
        if is_zero(flex_norm, numerical=numerical, tolerance=tolerance):
            raise ValueError("The norm of this flex is zero.")
        return {v: tuple([q / flex_norm for q in flex]) for v, flex in inf_flex.items()}
    elif isinstance(inf_flex, Sequence):
        if numerical:
            _inf_flex = [
                sympy_expr_to_float(flex, tolerance=tolerance) for flex in inf_flex
            ]
            flex_norm = np.linalg.norm(_inf_flex)
            if isclose(flex_norm, 0, abs_tol=tolerance):
                raise ValueError("The norm of this flex is almost zero.")
            return [flex / flex_norm for flex in _inf_flex]
        flex_norm = sp.sqrt(sum([flex**2 for flex in inf_flex]))
        if is_zero(flex_norm, numerical=numerical, tolerance=tolerance):
            raise ValueError("The norm of this flex is zero.")
        return [flex / flex_norm for flex in inf_flex]
    else:
        raise TypeError("`inf_flex` does not have the correct type.")


def _vector_distance_pointwise(
    dict1: dict[Vertex, Sequence[Number]],
    dict2: dict[Vertex, Sequence[Number]],
    numerical: bool = False,
) -> float:
    """
    Compute the Euclidean distance between two realizations or pointwise vectors.

    This method computes the Euclidean distance from the realization ``dict_1``
    to ``dict2`` considering them as vectors.
    The keys of ``dict1`` and ``dict2`` must be the same.

    Parameters
    ----------
    dict1, dict2:
        The dictionaries that are used for the distance computation.
    numerical:
        Boolean determining whether a numerical or symbolic normalization is performed.
    """
    if not set(dict1.keys()) == set(dict2.keys()) or not len(dict1) == len(dict2):
        raise ValueError("`dict1` and `dict2` are not based on the same vertex set.")
    if numerical:
        return float(
            np.linalg.norm(
                [
                    x - y
                    for v in dict1.keys()
                    for x, y in zip(
                        dict1[v],
                        dict2[v],
                    )
                ]
            )
        )
    return sp.sqrt(
        sum(
            [
                (x - y) ** 2
                for v in dict1.keys()
                for x, y in zip(
                    dict1[v],
                    dict2[v],
                )
            ]
        )
    )


def is_isomorphic_graph_list(list1, list2) -> bool:
    """
    Return whether two lists of graphs are the same up to graph isomorphism.
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


def _null_space(A: np.array, tolerance: float = 1e-8) -> np.array:
    """
    Compute the kernel of a numpy matrix.

    Parameters
    ----------
    tolerance:
        Used tolerance for the selection of the vectors
        in the kernel of the numerical matrix.
    """
    _, s, vh = np.linalg.svd(A, full_matrices=True)
    tol = np.amax(s) * tolerance
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:, :].T.conj()
    return Q
