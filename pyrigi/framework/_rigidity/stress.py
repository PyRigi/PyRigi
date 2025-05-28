"""
This module provides functionality related to stresses of frameworks.
"""

import numpy as np
import sympy as sp
from sympy import Matrix

import pyrigi.graph._utils._input_check as _graph_input_check
from pyrigi._utils._zero_check import is_zero_vector
from pyrigi._utils.linear_algebra import _null_space
from pyrigi.data_type import (
    Edge,
    Number,
    Sequence,
    Stress,
    Vertex,
)
from pyrigi.framework._rigidity import infinitesimal as infinitesimal_rigidity
from pyrigi.framework.base import FrameworkBase
from pyrigi.graph import _general as graph_general


def is_dict_stress(
    framework: FrameworkBase, dict_stress: dict[Edge, Number], **kwargs
) -> bool:
    """
    Return whether a dictionary specifies an equilibrium stress of the framework.

    Definitions
    -----------
    :prf:ref:`Equilibrium Stress <def-equilibrium-stress>`

    Parameters
    ----------
    dict_stress:
        Dictionary that maps the edges to stress values.

    Examples
    --------
    >>> F = Framework.Complete([[0,0], [1,0], ['1/2',0]])
    >>> F.is_dict_stress({(0,1):'-1/2', (0,2):1, (1,2):1})
    True
    >>> F.is_dict_stress({(0,1):1, (1,2):'-1/2', (0,2):1})
    False

    Notes
    -----
    See :meth:`.is_vector_stress`.
    """
    stress_edge_list = [tuple(e) for e in list(dict_stress.keys())]
    _graph_input_check.is_edge_order(framework._graph, stress_edge_list, "dict_stress")
    graph_edge_list = [tuple(e) for e in graph_general.edge_list(framework._graph)]
    dict_to_list = []

    for e in graph_edge_list:
        dict_to_list += [
            (
                dict_stress[e]
                if e in stress_edge_list
                else dict_stress[tuple([e[1], e[0]])]
            )
        ]

    return is_vector_stress(
        framework,
        dict_to_list,
        edge_order=graph_general.edge_list(framework._graph),
        **kwargs,
    )


def is_vector_stress(
    framework: FrameworkBase,
    stress: Sequence[Number],
    edge_order: Sequence[Edge] = None,
    numerical: bool = False,
    tolerance=1e-9,
) -> bool:
    r"""
    Return whether a vector is an equilibrium stress.

    Definitions
    -----------
    :prf:ref:`Equilibrium stress <def-equilibrium-stress>`

    Parameters
    ----------
    stress:
        A vector to be checked whether it is a stress of the framework.
    edge_order:
        A list of edges, providing the ordering for the entries of the ``stress``.
        If none is provided, the list from :meth:`.Graph.edge_list` is taken.
    numerical:
        A Boolean determining whether the evaluation of the product of the ``stress``
        and the rigidity matrix is symbolic or numerical.
    tolerance:
        Absolute tolerance that is the threshold for acceptable equilibrium
        stresses. This parameter is used to determine the number of digits,
        to which accuracy the symbolic expressions are evaluated.

    Examples
    --------
    >>> G = Graph([[0,1],[0,2],[0,3],[1,2],[2,3],[3,1]])
    >>> pos = {0: (0, 0), 1: (0,1), 2: (-1,-1), 3: (1,-1)}
    >>> F = Framework(G, pos)
    >>> omega1 = [-8, -4, -4, 2, 2, 1]
    >>> F.is_stress(omega1)
    True
    >>> omega1[0] = 0
    >>> F.is_stress(omega1)
    False
    >>> from pyrigi import frameworkDB
    >>> F = frameworkDB.Complete(5, dim=2)
    >>> stresses=F.stresses()
    >>> F.is_stress(stresses[0])
    True
    """
    edge_order = _graph_input_check.is_edge_order(
        framework._graph, edge_order=edge_order
    )
    return is_zero_vector(
        Matrix(stress).transpose()
        * infinitesimal_rigidity.rigidity_matrix(framework, edge_order=edge_order),
        numerical=numerical,
        tolerance=tolerance,
    )


def is_stress(framework: FrameworkBase, stress: Stress, **kwargs) -> bool:
    """
    Alias for :meth:`.is_vector_stress` and
    :meth:`.is_dict_stress`.

    One of the alias methods is called depending on the type of the input.

    Parameters
    ----------
    stress
    """
    if isinstance(stress, list | Matrix):
        return is_vector_stress(framework, stress, **kwargs)
    elif isinstance(stress, dict):
        return is_dict_stress(framework, stress, **kwargs)
    else:
        raise TypeError(
            "The `stress` must be specified either by a list/Matrix or a dictionary!"
        )


def stress_matrix(
    framework: FrameworkBase,
    stress: Stress,
    edge_order: Sequence[Edge] = None,
    vertex_order: Sequence[Vertex] = None,
) -> Matrix:
    r"""
    Construct the stress matrix of a stress.

    Definitions
    -----
    * :prf:ref:`Stress Matrix <def-stress-matrix>`

    Parameters
    ----------
    stress:
        A stress of the framework given as a vector.
    edge_order:
        A list of edges, providing the ordering of edges in ``stress``.
        If ``None``, :meth:`.Graph.edge_list` is assumed.
    vertex_order:
        Specification of row/column order of the stress matrix.
        If ``None``, :meth:`.Graph.vertex_list` is assumed.

    Examples
    --------
    >>> G = Graph([[0,1],[0,2],[0,3],[1,2],[2,3],[3,1]])
    >>> pos = {0: (0, 0), 1: (0,1), 2: (-1,-1), 3: (1,-1)}
    >>> F = Framework(G, pos)
    >>> omega = [-8, -4, -4, 2, 2, 1]
    >>> F.stress_matrix(omega)
    Matrix([
    [-16,  8,  4,  4],
    [  8, -4, -2, -2],
    [  4, -2, -1, -1],
    [  4, -2, -1, -1]])
    """
    vertex_order = _graph_input_check.is_vertex_order(framework._graph, vertex_order)
    edge_order = _graph_input_check.is_edge_order(framework._graph, edge_order)
    if not is_stress(framework, stress, edge_order=edge_order, numerical=True):
        raise ValueError(
            "The provided stress does not lie in the cokernel of the rigidity matrix!"
        )
    # creation of a zero |V|x|V| matrix
    stress_matr = sp.zeros(len(framework._graph))
    v_to_i = {v: i for i, v in enumerate(vertex_order)}

    for edge, edge_stress in zip(edge_order, stress):
        for v in edge:
            stress_matr[v_to_i[v], v_to_i[v]] += edge_stress

    for e, stressval in zip(edge_order, stress):
        i, j = v_to_i[e[0]], v_to_i[e[1]]
        stress_matr[i, j] = -stressval
        stress_matr[j, i] = -stressval

    return stress_matr


def stresses(
    framework: FrameworkBase,
    edge_order: Sequence[Edge] = None,
    numerical: bool = False,
    tolerance: float = 1e-9,
) -> list[Matrix] | list[list[float]]:
    r"""
    Return a basis of the space of equilibrium stresses.

    Definitions
    -----------
    :prf:ref:`Equilibrium stress <def-equilibrium-stress>`

    Parameters
    ----------
    edge_order:
        A list of edges, providing the ordering for the entries of the stresses.
        If none is provided, the list from :meth:`.Graph.edge_list` is taken.
    numerical:
        Determines whether the output is symbolic (default) or numerical.
    tolerance:
        Used tolerance when computing the stresses numerically.

    Examples
    --------
    >>> G = Graph([[0,1],[0,2],[0,3],[1,2],[2,3],[3,1]])
    >>> pos = {0: (0, 0), 1: (0,1), 2: (-1,-1), 3: (1,-1)}
    >>> F = Framework(G, pos)
    >>> F.stresses()
    [Matrix([
    [-8],
    [-4],
    [-4],
    [ 2],
    [ 2],
    [ 1]])]
    """
    if not numerical:
        return (
            infinitesimal_rigidity.rigidity_matrix(framework, edge_order=edge_order)
            .transpose()
            .nullspace()
        )
    F = FrameworkBase(
        framework._graph, framework.realization(as_points=True, numerical=True)
    )
    stresses = _null_space(
        np.array(
            infinitesimal_rigidity.rigidity_matrix(F, edge_order=edge_order).transpose()
        ).astype(np.float64),
        tolerance=tolerance,
    )
    return [list(stresses[:, i]) for i in range(stresses.shape[1])]


def _transform_stress_to_edgewise(
    framework: FrameworkBase, stress: Matrix, edge_order: Sequence[Edge] = None
) -> dict[Edge, Number]:
    r"""
    Transform the natural data type of a stress (``Matrix``) to a
    dictionary that maps an edge to a coordinate.

    Parameters
    ----------
    stress:
        An equilibrium stress in the form of a ``Matrix``.
    edge_order:
        If ``None``, the :meth:`.Graph.edge_list`
        is taken as the edge order.

    Examples
    ----
    >>> F = Framework.Complete([(0,0),(1,0),(1,1),(0,1)])
    >>> stress = F.stresses()[0]
    >>> from pyrigi.framework._rigidity.stress import _transform_stress_to_edgewise
    >>> _transform_stress_to_edgewise(F, stress)
    {(0, 1): 1, (0, 2): -1, (0, 3): 1, (1, 2): 1, (1, 3): -1, (2, 3): 1}

    Notes
    ----
    For example, this method can be used for generating an
    equilibrium stresss for plotting purposes.
    """
    edge_order = _graph_input_check.is_edge_order(framework._graph, edge_order)
    return {tuple(edge_order[i]): stress[i] for i in range(len(edge_order))}
