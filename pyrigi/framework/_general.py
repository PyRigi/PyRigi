"""
This module provides some general functionality for frameworks.
"""

from itertools import combinations

import numpy as np
import sympy as sp
from sympy import Matrix

import pyrigi.graph._utils._input_check as _graph_input_check
from pyrigi._utils._conversion import point_to_vector
from pyrigi._utils._zero_check import is_zero, is_zero_vector
from pyrigi.data_type import Edge, Number, Point, Vertex
from pyrigi.framework.base import FrameworkBase


def is_quasi_injective(
    framework: FrameworkBase, numerical: bool = False, tolerance: float = 1e-9
) -> bool:
    """
    Return whether the realization is quasi-injective.

    Definitions
    -----------
    :prf:ref:`Quasi-injectivity <def-realization>`

    Parameters
    ----------
    numerical:
        Whether the check is symbolic (default) or numerical.
    tolerance:
        Used tolerance when checking numerically.

    Notes
    -----
    For comparing whether two vectors are the same,
    :func:`.misc.is_zero_vector` is used.
    See its documentation for the description of the parameters.
    """

    for u, v in framework._graph.edges:
        edge_vector = framework[u] - framework[v]
        if is_zero_vector(edge_vector, numerical, tolerance):
            return False
    return True


def is_injective(
    framework: FrameworkBase, numerical: bool = False, tolerance: float = 1e-9
) -> bool:
    """
    Return whether the realization is injective.

    Parameters
    ----------
    numerical:
        Whether the check is symbolic (default) or numerical.
    tolerance:
        Used tolerance when checking numerically.

    Notes
    -----
    For comparing whether two vectors are the same,
    :func:`.misc.is_zero_vector` is used.
    See its documentation for the description of the parameters.
    """

    for u, v in combinations(framework._graph.nodes, 2):
        edge_vector = framework[u] - framework[v]
        if is_zero_vector(edge_vector, numerical, tolerance):
            return False
    return True


def is_congruent_realization(
    framework: FrameworkBase,
    other_realization: dict[Vertex, Point] | dict[Vertex, Matrix],
    numerical: bool = False,
    tolerance: float = 1e-9,
) -> bool:
    """
    Return whether the given realization is congruent to the framework.

    Definitions
    -----------
    :prf:ref:`Congruent frameworks <def-equivalent-framework>`

    Parameters
    ----------
    other_realization
        The realization for checking the congruence.
    numerical
        Whether the check is symbolic (default) or numerical.
    tolerance
        Used tolerance when checking numerically.
    """
    _graph_input_check.is_vertex_order(
        framework._graph, list(other_realization.keys()), "other_realization"
    )

    for u, v in combinations(framework._graph.nodes, 2):
        edge_vec = framework[u] - framework[v]
        dist_squared = (edge_vec.T * edge_vec)[0, 0]

        other_edge_vec = point_to_vector(other_realization[u]) - point_to_vector(
            other_realization[v]
        )
        otherdist_squared = (other_edge_vec.T * other_edge_vec)[0, 0]

        difference = sp.simplify(dist_squared - otherdist_squared)
        if not is_zero(difference, numerical=numerical, tolerance=tolerance):
            return False
    return True


def is_congruent(
    framework: FrameworkBase,
    other_framework: FrameworkBase,
    numerical: bool = False,
    tolerance: float = 1e-9,
) -> bool:
    """
    Return whether the given framework is congruent to the framework.

    Definitions
    -----------
    :prf:ref:`Congruent frameworks <def-equivalent-framework>`

    Parameters
    ----------
    other_framework
        The framework for checking the congruence.
    numerical
        Whether the check is symbolic (default) or numerical.
    tolerance
        Used tolerance when checking numerically.
    """

    framework._input_check_underlying_graphs(other_framework)

    return is_congruent_realization(
        framework, other_framework._realization, numerical, tolerance
    )


def is_equivalent_realization(
    framework: FrameworkBase,
    other_realization: dict[Vertex, Point] | dict[Vertex, Matrix],
    numerical: bool = False,
    tolerance: float = 1e-9,
) -> bool:
    """
    Return whether the given realization is equivalent to the framework.

    Definitions
    -----------
    :prf:ref:`Equivalent frameworks <def-equivalent-framework>`

    Parameters
    ----------
    other_realization
        The realization for checking the equivalence.
    numerical
        Whether the check is symbolic (default) or numerical.
    tolerance
        Used tolerance when checking numerically.
    """
    _graph_input_check.is_vertex_order(
        framework._graph, list(other_realization.keys()), "other_realization"
    )

    for u, v in framework._graph.edges:
        edge_vec = framework[u] - framework[v]
        dist_squared = (edge_vec.T * edge_vec)[0, 0]

        other_edge_vec = point_to_vector(other_realization[u]) - point_to_vector(
            other_realization[v]
        )
        otherdist_squared = (other_edge_vec.T * other_edge_vec)[0, 0]

        difference = sp.simplify(otherdist_squared - dist_squared)
        if not is_zero(difference, numerical=numerical, tolerance=tolerance):
            return False
    return True


def is_equivalent(
    framework: FrameworkBase,
    other_framework: FrameworkBase,
    numerical: bool = False,
    tolerance: float = 1e-9,
) -> bool:
    """
    Return whether the given framework is equivalent to the framework.

    Definitions
    -----------
    :prf:ref:`Equivalent frameworks <def-equivalent-framework>`

    Parameters
    ----------
    other_framework
        The framework for checking the equivalence.
    numerical
        Whether the check is symbolic (default) or numerical.
    tolerance
        Used tolerance when checking numerically.
    """

    framework._input_check_underlying_graphs(other_framework)

    return is_equivalent_realization(
        framework, other_framework._realization, numerical, tolerance
    )


def edge_lengths(
    framework: FrameworkBase, numerical: bool = False
) -> dict[Edge, Number]:
    """
    Return the dictionary of the edge lengths.

    Parameters
    -------
    numerical:
        If ``True``, numerical positions are used for the computation of the edge lengths.

    Examples
    --------
    >>> G = Graph([(0,1), (1,2), (2,3), (0,3)])
    >>> F = Framework(G, {0:[0,0], 1:[1,0], 2:[1,'1/2 * sqrt(5)'], 3:['1/2','4/3']})
    >>> F.edge_lengths(numerical=False)
    {(0, 1): 1, (0, 3): sqrt(73)/6, (1, 2): sqrt(5)/2, (2, 3): sqrt((-4/3 + sqrt(5)/2)**2 + 1/4)}
    >>> F.edge_lengths(numerical=True)
    {(0, 1): 1.0, (0, 3): 1.4240006242195884, (1, 2): 1.118033988749895, (2, 3): 0.5443838790578374}
    """  # noqa: E501
    if numerical:
        points = framework.realization(as_points=True, numerical=True)
        return {
            tuple(e): float(
                np.linalg.norm(np.array(points[e[0]]) - np.array(points[e[1]]))
            )
            for e in framework._graph.edges
        }
    else:
        points = framework.realization(as_points=True)
        return {
            tuple(e): sp.sqrt(
                sum([(x - y) ** 2 for x, y in zip(points[e[0]], points[e[1]])])
            )
            for e in framework._graph.edges
        }
