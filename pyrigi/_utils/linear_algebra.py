"""
This modules provides various linear algebra functions.
"""

from math import isclose

import numpy as np
import sympy as sp
from sympy import Matrix

from pyrigi.data_type import InfFlex, Number, Sequence, Vertex
from ._input_check import integrality_and_range
from ._conversion import sympy_expr_to_float
from ._zero_check import is_zero


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


def _null_space(A: np.array, tolerance: float = 1e-9) -> np.array:
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


def _reduced_null_space(
    A: np.ndarray | Matrix,
    free_columns: Sequence[int] = [],
    numerical=False,
    tolerance=1e-9,
):
    """
    Compute the reduced kernel of `A`.

    Compute the linear subspace `reduced_kernel` of `A`'s kernel in which the `i`-th
    element of each vector `v` in `reduced_kernel` is constrained to be 0. In other
    words, this subspace can be obtained by deleting all but the columns in `free_columns`
    from `A` to obtain a matrix `A_reduced`, computing the kernel of `A_reduced`
    and padding the vectors in this kernel with zeros in the indices from
    `[0, ..., #columns(A) - 1]` that are not contained in `free_columns`.

    Parameters
    ----------
    free_columns:
        The columns of `A` that are not deleted.
    numerical:
        Determines whether the output is symbolic (default) or numerical.
    tolerance:
        Used tolerance when computing the infinitesimal flex numerically.
    """
    all(
        integrality_and_range(col, "free_columns", min_val=0, max_val=A.shape[1])
        for col in free_columns
    )
    if not list(set(free_columns)).sort() == free_columns.sort():
        raise ValueError("The `free_columns` can only be composed of unique integers.")

    A_reduced = A[:, free_columns]
    if numerical:
        reduced_kernel = _null_space(A_reduced, tolerance=tolerance)
        kernel = np.zeros((A.shape[1], reduced_kernel.shape[1]))
        kernel[free_columns, :] = reduced_kernel
    else:
        reduced_kernel = A_reduced.nullspace()
        kernel = sp.zeros(A.shape[1], len(reduced_kernel))
        for i in range(len(reduced_kernel)):
            for j, col in enumerate(free_columns):
                kernel[col, i] = reduced_kernel[i][j]
        kernel = Matrix(kernel)
    return kernel
