"""
This module provides algorithms related to infinitesimal rigidity of frameworks.
"""

from copy import deepcopy

import numpy as np
import sympy as sp
from sympy import Matrix, binomial, flatten

import pyrigi.graph._utils._input_check as _graph_input_check
from pyrigi._utils._conversion import sympy_expr_to_float
from pyrigi._utils._zero_check import is_zero_vector
from pyrigi._utils.linear_algebra import _null_space
from pyrigi.data_type import (
    Edge,
    InfFlex,
    Number,
    Sequence,
    Vertex,
)
from pyrigi.framework.base import FrameworkBase
from pyrigi.graph import _general as graph_general
from pyrigi.graphDB import Complete as CompleteGraph


def rigidity_matrix(
    framework: FrameworkBase,
    vertex_order: Sequence[Vertex] = None,
    edge_order: Sequence[Edge] = None,
) -> Matrix:
    r"""
    Construct the rigidity matrix of the framework.

    Definitions
    -----------
    * :prf:ref:`Rigidity matrix <def-rigidity-matrix>`

    Parameters
    ----------
    vertex_order:
        A list of vertices, providing the ordering for the columns
        of the rigidity matrix.
        If none is provided, the list from :meth:`.Graph.vertex_list` is taken.
    edge_order:
        A list of edges, providing the ordering for the rows
        of the rigidity matrix.
        If none is provided, the list from :meth:`.Graph.edge_list` is taken.

    Examples
    --------
    >>> F = Framework.Complete([(0,0),(2,0),(1,3)])
    >>> F.rigidity_matrix()
    Matrix([
    [-2,  0, 2,  0,  0, 0],
    [-1, -3, 0,  0,  1, 3],
    [ 0,  0, 1, -3, -1, 3]])
    """
    vertex_order = _graph_input_check.is_vertex_order(framework._graph, vertex_order)
    edge_order = _graph_input_check.is_edge_order(framework._graph, edge_order)

    # ``delta`` is responsible for distinguishing the edges (i,j) and (j,i)
    def delta(e, w):
        # the parameter e represents an edge
        # the parameter w represents a vertex
        if w == e[0]:
            return 1
        if w == e[1]:
            return -1
        return 0

    return Matrix(
        [
            flatten(
                [
                    delta(e, w) * (framework[e[0]] - framework[e[1]])
                    for w in vertex_order
                ]
            )
            for e in edge_order
        ]
    )


def rigidity_matrix_rank(
    framework: FrameworkBase, numerical: bool = False, tolerance: float = 1e-9
) -> int:
    """
    Return the rank of the rigidity matrix.

    Definitions
    -----------
    :prf:ref:`Rigidity matrix <def-rigidity-matrix>`

    Parameters
    ----------
    numerical:
        If ``True``, the rank of the rigidity matrix with entries as floats
        is computed.

        *Warning:* For ``numerical=True`` the numerical rank computation
        may produce different results than the computation over exact
        coordinates.
    tolerance:
        Numerical tolerance used for computing the rigidity matrix rank.

    Examples
    --------
    >>> K4 = Framework.Complete([[0,0], [1,0], [1,1], [0,1]])
    >>> K4.rigidity_matrix_rank()   # the complete graph is a circuit
    5
    >>> K4.delete_edge([0,1])
    >>> K4.rigidity_matrix_rank()   # deleting a bar gives full rank
    5
    >>> K4.delete_edge([2,3])
    >>> K4.rigidity_matrix_rank()   #so now deleting an edge lowers the rank
    4
    """
    if numerical:
        F = FrameworkBase(
            framework._graph, framework.realization(as_points=True, numerical=True)
        )
        return np.linalg.matrix_rank(
            np.array(rigidity_matrix(F)).astype(np.float64), tol=tolerance
        )
    return rigidity_matrix(framework).rank()


def trivial_inf_flexes(
    framework: FrameworkBase, vertex_order: Sequence[Vertex] = None
) -> list[Matrix]:
    r"""
    Return a basis of the vector subspace of trivial infinitesimal flexes.

    Definitions
    -----------
    :prf:ref:`Trivial infinitesimal flexes <def-trivial-inf-flex>`

    Parameters
    ----------
    vertex_order:
        A list of vertices, providing the ordering for the entries
        of the infinitesimal flexes.

    Examples
    --------
    >>> F = Framework.Complete([(0,0), (2,0), (0,2)])
    >>> F.trivial_inf_flexes()
    [Matrix([
    [1],
    [0],
    [1],
    [0],
    [1],
    [0]]), Matrix([
    [0],
    [1],
    [0],
    [1],
    [0],
    [1]]), Matrix([
    [ 0],
    [ 0],
    [ 0],
    [ 2],
    [-2],
    [ 0]])]
    """
    vertex_order = _graph_input_check.is_vertex_order(framework._graph, vertex_order)
    dim = framework.dim
    translations = [
        Matrix.vstack(*[A for _ in vertex_order]) for A in Matrix.eye(dim).columnspace()
    ]
    basis_skew_symmetric = []
    for i in range(1, dim):
        for j in range(i):
            A = Matrix.zeros(dim)
            A[i, j] = 1
            A[j, i] = -1
            basis_skew_symmetric += [A]
    inf_rot = [
        Matrix.vstack(*[A * framework[v] for v in vertex_order])
        for A in basis_skew_symmetric
    ]
    matrix_inf_flexes = Matrix.hstack(*(translations + inf_rot))
    return matrix_inf_flexes.transpose().echelon_form().transpose().columnspace()


def nontrivial_inf_flexes(framework: FrameworkBase, **kwargs) -> list[Matrix]:
    """
    Return non-trivial infinitesimal flexes.

    See :meth:`~Framework.inf_flexes` for possible keywords.

    Definitions
    -----------
    :prf:ref:`Infinitesimal flex <def-inf-rigid-framework>`

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> F = Framework.Circular(graphs.CompleteBipartite(3, 3))
    >>> F.nontrivial_inf_flexes()
    [Matrix([
    [       3/2],
    [-sqrt(3)/2],
    [         1],
    [         0],
    [         0],
    [         0],
    [       3/2],
    [-sqrt(3)/2],
    [         1],
    [         0],
    [         0],
    [         0]])]
    """
    return inf_flexes(framework, include_trivial=False, **kwargs)


def inf_flexes(
    framework: FrameworkBase,
    include_trivial: bool = False,
    vertex_order: Sequence[Vertex] = None,
    numerical: bool = False,
    tolerance: float = 1e-9,
) -> list[Matrix] | list[list[float]]:
    r"""
    Return a basis of the space of infinitesimal flexes.

    Return a lift of a basis of the quotient of
    the vector space of infinitesimal flexes
    modulo trivial infinitesimal flexes, if ``include_trivial=False``.
    Return a basis of the vector space of infinitesimal flexes
    if ``include_trivial=True``.

    Definitions
    -----------
    :prf:ref:`Infinitesimal flex <def-inf-flex>`

    Parameters
    ----------
    include_trivial:
        Boolean that decides, whether the trivial flexes should
        be included.
    vertex_order:
        A list of vertices, providing the ordering for the entries
        of the infinitesimal flexes.
        If none is provided, the list from :meth:`.Graph.vertex_list` is taken.
    numerical:
        Determines whether the output is symbolic (default) or numerical.
    tolerance
        Used tolerance when computing the infinitesimal flex numerically.

    Examples
    --------
    >>> F = Framework.Complete([[0,0], [1,0], [1,1], [0,1]])
    >>> F.delete_edges([(0,2), (1,3)])
    >>> F.inf_flexes(include_trivial=False)
    [Matrix([
    [1],
    [0],
    [1],
    [0],
    [0],
    [0],
    [0],
    [0]])]
    >>> F = Framework(
    ...     Graph([[0, 1], [0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]]),
    ...     {0: [0, 0], 1: [0, 1], 2: [0, 2], 3: [1, 2], 4: [-1, 2]},
    ... )
    >>> F.inf_flexes()
    [Matrix([
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
    [0],
    [0],
    [0],
    [0]])]
    """
    vertex_order = _graph_input_check.is_vertex_order(framework._graph, vertex_order)
    if include_trivial:
        if not numerical:
            return rigidity_matrix(framework, vertex_order=vertex_order).nullspace()
        else:
            F = FrameworkBase(
                framework._graph, framework.realization(as_points=True, numerical=True)
            )
            return _null_space(
                np.array(rigidity_matrix(F, vertex_order=vertex_order)).astype(
                    np.float64
                )
            )

    if not numerical:
        rig_matrix = rigidity_matrix(framework, vertex_order=vertex_order)

        all_inf_flexes = rig_matrix.nullspace()
        triv_inf_flexes = trivial_inf_flexes(framework, vertex_order=vertex_order)
        s = len(triv_inf_flexes)
        extend_basis_matrix = Matrix.hstack(*triv_inf_flexes)
        for inf_flex in all_inf_flexes:
            tmp_matrix = Matrix.hstack(extend_basis_matrix, inf_flex)
            if not tmp_matrix.rank() == extend_basis_matrix.rank():
                extend_basis_matrix = Matrix.hstack(extend_basis_matrix, inf_flex)
        basis = extend_basis_matrix.columnspace()
        return basis[s:]
    else:
        F = FrameworkBase(
            framework._graph, framework.realization(as_points=True, numerical=True)
        )
        flexes = _null_space(
            np.array(rigidity_matrix(F, vertex_order=vertex_order)).astype(np.float64),
            tolerance=tolerance,
        )
        flexes = [flexes[:, i] for i in range(flexes.shape[1])]
        Kn = FrameworkBase(
            CompleteGraph(len(framework._graph)),
            framework.realization(as_points=True, numerical=True),
        )
        inf_flexes_trivial = _null_space(
            np.array(rigidity_matrix(Kn, vertex_order=vertex_order)).astype(np.float64),
            tolerance=tolerance,
        )
        s = inf_flexes_trivial.shape[1]
        extend_basis_matrix = inf_flexes_trivial
        for inf_flex in flexes:
            inf_flex = np.reshape(inf_flex, (-1, 1))
            tmp_matrix = np.hstack((inf_flexes_trivial, inf_flex))
            if not np.linalg.matrix_rank(
                tmp_matrix, tol=tolerance
            ) == np.linalg.matrix_rank(inf_flexes_trivial, tol=tolerance):
                extend_basis_matrix = np.hstack((extend_basis_matrix, inf_flex))
        Q, R = np.linalg.qr(extend_basis_matrix)
        Q = Q[:, s : np.linalg.matrix_rank(R, tol=tolerance)]
        return [list(Q[:, i]) for i in range(Q.shape[1])]


def is_inf_rigid(
    framework: FrameworkBase, numerical: bool = False, tolerance: float = 1e-9
) -> bool:
    """
    Return whether the framework is infinitesimally rigid.

    Definitions
    -----------
    :prf:ref:`Infinitesimal rigidity <def-inf-rigid-framework>`

    Parameters
    ----------
    numerical:
        If ``True``, the rigidity matrix rank computation for determining
        rigidity is numerical.

        *Warning:* For ``numerical=True`` the numerical rank computation
        may produce different results than the computation over symbolic
        coordinates.
    tolerance:
        Numerical tolerance used for computing the rigidity matrix rank.

    Examples
    --------
    >>> from pyrigi import frameworkDB
    >>> F1 = frameworkDB.CompleteBipartite(4,4)
    >>> F1.is_inf_rigid()
    True
    >>> F2 = frameworkDB.Cycle(4,dim=2)
    >>> F2.is_inf_rigid()
    False
    """

    if framework._graph.number_of_nodes() <= framework.dim + 1:
        return rigidity_matrix_rank(
            framework, numerical=numerical, tolerance=tolerance
        ) == binomial(framework._graph.number_of_nodes(), 2)
    else:
        return rigidity_matrix_rank(
            framework, numerical=numerical, tolerance=tolerance
        ) == framework.dim * framework._graph.number_of_nodes() - binomial(
            framework.dim + 1, 2
        )


def is_inf_flexible(framework: FrameworkBase, **kwargs) -> bool:
    """
    Return whether the framework is infinitesimally flexible.

    For implementation details and possible parameters, see
    :meth:`~Framework.is_inf_rigid`.

    Definitions
    -----------
    :prf:ref:`Infinitesimal rigidity <def-inf-rigid-framework>`
    """
    return not is_inf_rigid(framework, **kwargs)


def is_min_inf_rigid(framework: FrameworkBase, use_copy: bool = True, **kwargs) -> bool:
    """
    Return whether the framework is minimally infinitesimally rigid.

    For implementation details and possible parameters, see
    :meth:`~Framework.is_inf_rigid`.

    Definitions
    -----
    :prf:ref:`Minimal infinitesimal rigidity <def-min-rigid-framework>`

    Parameters
    ----------
    use_copy:
        If ``False``, the framework's edges are deleted and added back
        during runtime.
        Otherwise, a new modified framework is created,
        while the original framework remains unchanged (default).

    Examples
    --------
    >>> F = Framework.Complete([[0,0], [1,0], [1,1], [0,1]])
    >>> F.is_min_inf_rigid()
    False
    >>> F.delete_edge((0,2))
    >>> F.is_min_inf_rigid()
    True
    """
    if not is_inf_rigid(framework, **kwargs):
        return False

    F = framework
    if use_copy:
        F = deepcopy(framework)
    for edge in graph_general.edge_list(F._graph):
        F.delete_edge(edge)
        if is_inf_rigid(F, **kwargs):
            F.add_edge(edge)
            return False
        F.add_edge(edge)
    return True


def _transform_inf_flex_to_pointwise(
    framework: FrameworkBase,
    inf_flex: Matrix | Sequence,
    vertex_order: Sequence[Vertex] = None,
) -> dict[Vertex, list[Number]]:
    r"""
    Transform the natural data type of a flex (``Matrix``) to a
    dictionary that maps a vertex to a ``Sequence`` of coordinates
    (i.e. a vector).

    Parameters
    ----------
    inf_flex:
        An infinitesimal flex in the form of a ``Matrix``.
    vertex_order:
        If ``None``, the :meth:`.Graph.vertex_list`
        is taken as the vertex order.

    Examples
    ----
    >>> F = Framework.from_points([(0,0), (1,0), (0,1)])
    >>> F.add_edges([(0,1),(0,2)])
    >>> flex = F.nontrivial_inf_flexes()[0]
    >>> from pyrigi.framework._rigidity.infinitesimal import _transform_inf_flex_to_pointwise
    >>> _transform_inf_flex_to_pointwise(F, flex)
    {0: [1, 0], 1: [1, 0], 2: [0, 0]}

    Notes
    ----
    For example, this method can be used for generating an
    infinitesimal flex for plotting purposes.
    """  # noqa: E501
    vertex_order = _graph_input_check.is_vertex_order(framework._graph, vertex_order)
    if (
        isinstance(inf_flex, Matrix)
        and (
            inf_flex.shape[1] != 1
            or inf_flex.shape[0] != framework.dim * len(vertex_order)
        )
    ) or (
        isinstance(inf_flex, Sequence)
        and len(inf_flex) != framework.dim * len(vertex_order)
    ):
        raise ValueError("The provided `inf_flex` does not have the correct format.")

    return {
        vertex_order[i]: [inf_flex[i * framework.dim + j] for j in range(framework.dim)]
        for i in range(len(vertex_order))
    }


def is_vector_inf_flex(
    framework: FrameworkBase,
    inf_flex: Sequence[Number],
    vertex_order: Sequence[Vertex] = None,
    numerical: bool = False,
    tolerance: float = 1e-9,
) -> bool:
    r"""
    Return whether a vector is an infinitesimal flex of the framework.

    Definitions
    -----------
    * :prf:ref:`Infinitesimal flex <def-inf-flex>`
    * :prf:ref:`Rigidity Matrix <def-rigidity-matrix>`

    Parameters
    ----------
    inf_flex:
        An infinitesimal flex of the framework specified by a vector.
    vertex_order:
        A list of vertices specifying the order in which ``inf_flex`` is given.
        If none is provided, the list from :meth:`~.Graph.vertex_list` is taken.
    numerical:
        A Boolean determining whether the evaluation of the product of
        the ``inf_flex`` and the rigidity matrix is symbolic or numerical.
    tolerance:
        Absolute tolerance that is the threshold for acceptable numerical flexes.
        This parameter is used to determine the number of digits, to which
        accuracy the symbolic expressions are evaluated.

    Examples
    --------
    >>> from pyrigi import frameworkDB as fws
    >>> F = fws.Square()
    >>> q = [0,0,0,0,-2,0,-2,0]
    >>> F.is_vector_inf_flex(q)
    True
    >>> q[0] = 1
    >>> F.is_vector_inf_flex(q)
    False
    >>> F = Framework.Complete([[0,0], [1,1]])
    >>> F.is_vector_inf_flex(["sqrt(2)","-sqrt(2)",0,0], vertex_order=[1,0])
    True
    """
    vertex_order = _graph_input_check.is_vertex_order(framework._graph, vertex_order)
    return is_zero_vector(
        rigidity_matrix(framework, vertex_order=vertex_order) * Matrix(inf_flex),
        numerical=numerical,
        tolerance=tolerance,
    )


def is_dict_inf_flex(
    framework: FrameworkBase, vert_to_flex: dict[Vertex, Sequence[Number]], **kwargs
) -> bool:
    """
    Return whether a dictionary specifies an infinitesimal flex of the framework.

    Definitions
    -----------
    :prf:ref:`Infinitesimal flex <def-inf-flex>`

    Parameters
    ----------
    vert_to_flex:
        Dictionary that maps the vertex labels to
        vectors of the same dimension as the framework is.

    Examples
    --------
    >>> F = Framework.Complete([[0,0], [1,1]])
    >>> F.is_dict_inf_flex({0:[0,0], 1:[-1,1]})
    True
    >>> F.is_dict_inf_flex({0:[0,0], 1:["sqrt(2)","-sqrt(2)"]})
    True

    Notes
    -----
    See :meth:`.is_vector_inf_flex`.
    """
    _graph_input_check.is_vertex_order(
        framework._graph, list(vert_to_flex.keys()), "vert_to_flex"
    )

    dict_to_list = []
    for v in graph_general.vertex_list(framework._graph):
        dict_to_list += list(vert_to_flex[v])

    return is_vector_inf_flex(
        framework,
        dict_to_list,
        vertex_order=graph_general.vertex_list(framework._graph),
        **kwargs,
    )


def is_vector_nontrivial_inf_flex(
    framework: FrameworkBase,
    inf_flex: Sequence[Number],
    vertex_order: Sequence[Vertex] = None,
    numerical: bool = False,
    tolerance: float = 1e-9,
) -> bool:
    r"""
    Return whether an infinitesimal flex is nontrivial.

    Definitions
    -----------
    :prf:ref:`Nontrivial infinitesimal flex <def-trivial-inf-flex>`

    Parameters
    ----------
    inf_flex:
        An infinitesimal flex of the framework.
    vertex_order:
        A list of vertices specifying the order in which ``inf_flex`` is given.
        If none is provided, the list from :meth:`.Graph.vertex_list` is taken.
    numerical:
        A Boolean determining whether the evaluation of the product of the `inf_flex`
        and the rigidity matrix is symbolic or numerical.
    tolerance:
        Absolute tolerance that is the threshold for acceptable numerical flexes.
        This parameter is used to determine the number of digits, to which
        accuracy the symbolic expressions are evaluated.

    Examples
    --------
    >>> from pyrigi import frameworkDB as fws
    >>> F = fws.Square()
    >>> q = [0,0,0,0,-2,0,-2,0]
    >>> F.is_vector_nontrivial_inf_flex(q)
    True
    >>> q = [1,-1,1,1,-1,1,-1,-1]
    >>> F.is_vector_inf_flex(q)
    True
    >>> F.is_vector_nontrivial_inf_flex(q)
    False

    Notes
    -----
    This is done by solving a linear system composed of a matrix $A$ whose columns
    are given by a basis of the trivial flexes and the vector $b$ given by the
    input flex. $b$ is trivial if and only if there is a linear combination of
    the columns in $A$ producing $b$. In other words, when there is a solution to
    $Ax=b$, then $b$ is a trivial infinitesimal motion. Otherwise, $b$ is
    nontrivial.

    In the ``numerical=True`` case we compute a least squares solution $x$ of the
    overdetermined linear system and compare the values in $Ax$ to the values
    in $b$.
    """
    vertex_order = _graph_input_check.is_vertex_order(framework._graph, vertex_order)
    if not is_vector_inf_flex(
        framework,
        inf_flex,
        vertex_order=vertex_order,
        numerical=numerical,
        tolerance=tolerance,
    ):
        return False

    if not numerical:
        Q_trivial = Matrix.hstack(
            *(trivial_inf_flexes(framework, vertex_order=vertex_order))
        )
        system = Q_trivial, Matrix(inf_flex)
        return sp.linsolve(system) == sp.EmptySet
    else:
        Q_trivial = np.array(
            [
                sympy_expr_to_float(flex, tolerance=tolerance)
                for flex in trivial_inf_flexes(framework, vertex_order=vertex_order)
            ]
        ).transpose()
        b = np.array(sympy_expr_to_float(inf_flex, tolerance=tolerance)).transpose()
        x = np.linalg.lstsq(Q_trivial, b, rcond=None)[0]
        return not is_zero_vector(
            np.dot(Q_trivial, x) - b, numerical=True, tolerance=tolerance
        )


def is_dict_nontrivial_inf_flex(
    framework: FrameworkBase, vert_to_flex: dict[Vertex, Sequence[Number]], **kwargs
) -> bool:
    r"""
    Return whether a dictionary specifies an infinitesimal flex which is nontrivial.

    See :meth:`.is_vector_nontrivial_inf_flex` for details,
    particularly concerning the possible parameters.

    Definitions
    -----------
    :prf:ref:`Nontrivial infinitesimal flex <def-trivial-inf-flex>`

    Parameters
    ----------
    vert_to_flex:
        An infinitesimal flex of the framework in the form of a dictionary.

    Examples
    --------
    >>> from pyrigi import frameworkDB as fws
    >>> F = fws.Square()
    >>> q = {0:[0,0], 1: [0,0], 2:[-2,0], 3:[-2,0]}
    >>> F.is_dict_nontrivial_inf_flex(q)
    True
    >>> q = {0:[1,-1], 1: [1,1], 2:[-1,1], 3:[-1,-1]}
    >>> F.is_dict_nontrivial_inf_flex(q)
    False
    """
    _graph_input_check.is_vertex_order(
        framework._graph, list(vert_to_flex.keys()), "vert_to_flex"
    )

    dict_to_list = []
    for v in graph_general.vertex_list(framework._graph):
        dict_to_list += list(vert_to_flex[v])

    return is_vector_nontrivial_inf_flex(
        framework,
        dict_to_list,
        vertex_order=graph_general.vertex_list(framework._graph),
        **kwargs,
    )


def is_nontrivial_flex(
    framework: FrameworkBase,
    inf_flex: InfFlex,
    **kwargs,
) -> bool:
    """
    Alias for :meth:`.is_vector_nontrivial_inf_flex` and
    :meth:`.is_dict_nontrivial_inf_flex`.

    It is distinguished between instances of ``list`` and instances of ``dict`` to
    call one of the alias methods.

    Definitions
    -----------
    :prf:ref:`Nontrivial infinitesimal flex <def-trivial-inf-flex>`

    Parameters
    ----------
    inf_flex
    """
    if isinstance(inf_flex, list | tuple | Matrix):
        return is_vector_nontrivial_inf_flex(framework, inf_flex, **kwargs)
    elif isinstance(inf_flex, dict):
        return is_dict_nontrivial_inf_flex(framework, inf_flex, **kwargs)
    else:
        raise TypeError(
            "The `inf_flex` must be specified either by a vector or a dictionary!"
        )


def is_vector_trivial_inf_flex(
    framework: FrameworkBase, inf_flex: Sequence[Number], **kwargs
) -> bool:
    r"""
    Return whether an infinitesimal flex is trivial.

    See also :meth:`.is_nontrivial_vector_inf_flex` for details,
    particularly concerning the possible parameters.

    Definitions
    -----------
    :prf:ref:`Trivial infinitesimal flex <def-trivial-inf-flex>`

    Parameters
    ----------
    inf_flex:
        An infinitesimal flex of the framework.

    Examples
    --------
    >>> from pyrigi import frameworkDB as fws
    >>> F = fws.Square()
    >>> q = [0,0,0,0,-2,0,-2,0]
    >>> F.is_vector_trivial_inf_flex(q)
    False
    >>> q = [1,-1,1,1,-1,1,-1,-1]
    >>> F.is_vector_trivial_inf_flex(q)
    True
    """
    if not is_vector_inf_flex(framework, inf_flex, **kwargs):
        return False
    return not is_vector_nontrivial_inf_flex(framework, inf_flex, **kwargs)


def is_dict_trivial_inf_flex(
    framework: FrameworkBase, inf_flex: dict[Vertex, Sequence[Number]], **kwargs
) -> bool:
    r"""
    Return whether an infinitesimal flex specified by a dictionary is trivial.

    See :meth:`.is_vector_trivial_inf_flex` for details,
    particularly concerning the possible parameters.

    Definitions
    -----------
    :prf:ref:`Trivial infinitesimal flex <def-trivial-inf-flex>`

    Parameters
    ----------
    inf_flex:
        An infinitesimal flex of the framework in the form of a dictionary.

    Examples
    --------
    >>> from pyrigi import frameworkDB as fws
    >>> F = fws.Square()
    >>> q = {0:[0,0], 1: [0,0], 2:[-2,0], 3:[-2,0]}
    >>> F.is_dict_trivial_inf_flex(q)
    False
    >>> q = {0:[1,-1], 1: [1,1], 2:[-1,1], 3:[-1,-1]}
    >>> F.is_dict_trivial_inf_flex(q)
    True
    """
    _graph_input_check.is_vertex_order(
        framework._graph, list(inf_flex.keys()), "vert_to_flex"
    )

    dict_to_list = []
    for v in graph_general.vertex_list(framework._graph):
        dict_to_list += list(inf_flex[v])

    return is_vector_trivial_inf_flex(
        framework,
        dict_to_list,
        vertex_order=graph_general.vertex_list(framework._graph),
        **kwargs,
    )


def is_trivial_flex(
    framework: FrameworkBase,
    inf_flex: InfFlex,
    **kwargs,
) -> bool:
    """
    Alias for :meth:`.is_vector_trivial_inf_flex` and
    :meth:`.is_dict_trivial_inf_flex`.

    Ii is distinguished between instances of ``list`` and instances of ``dict`` to
    call one of the alias methods.

    Definitions
    -----------
    :prf:ref:`Trivial infinitesimal flex <def-trivial-inf-flex>`

    Parameters
    ----------
    inf_flex
    """
    if isinstance(inf_flex, list | tuple | Matrix):
        return is_vector_trivial_inf_flex(framework, inf_flex, **kwargs)
    elif isinstance(inf_flex, dict):
        return is_dict_trivial_inf_flex(framework, inf_flex, **kwargs)
    else:
        raise TypeError(
            "The `inf_flex` must be specified either by a vector or a dictionary!"
        )
