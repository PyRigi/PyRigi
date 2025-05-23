"""
This module provides algorithms related to generic rigidity.
"""

import math
from itertools import combinations
from typing import TypeVar

import networkx as nx
from sympy import oo

import pyrigi._utils._input_check as _input_check
import pyrigi.graph._constructions.extensions as graph_extension
import pyrigi.graph._sparsity.sparsity as sparsity
import pyrigi.graph._utils._input_check as _graph_input_check
from pyrigi.data_type import Inf, Vertex
from pyrigi.exception import NotSupportedValueError
from pyrigi.warning import _warn_randomized_alg as warn_randomized_alg

T = TypeVar("T")


def is_rigid(
    graph: nx.Graph,
    dim: int = 2,
    algorithm: str = "default",
    use_precomputed_pebble_digraph: bool = False,
    prob: float = 0.0001,
) -> bool:
    """
    Return whether the graph is ``dim``-rigid.

    Definitions
    -----------
    :prf:ref:`Generic dim-rigidity <def-gen-rigid>`

    Parameters
    ----------
    dim:
        Dimension.
    algorithm:
        If ``"graphic"`` (only if ``dim=1``), then the graphic matroid
        is used, namely, it is checked whether the graph is connected.

        If ``"sparsity"`` (only if ``dim=2``),
        then the existence of a spanning
        :prf:ref:`(2,3)-tight <def-kl-sparse-tight>` subgraph and
        :prf:ref:`thm-2-gen-rigidity` are used
        with the pebble game algorithm (:prf:ref:`alg-pebble-game`).

        If ``"randomized"``, a probabilistic check is performed.
        It may give false negatives (with probability at most ``prob``),
        but no false positives. See :prf:ref:`thm-probabilistic-rigidity-check`.

        If ``"numerical"``, a numerical check on the rigidity matrix rank
        is performed. See :meth:`.Framework.is_inf_rigid` for further details.

        If ``"default"``, then ``"graphic"`` is used for ``dim=1``
        and ``"sparsity"`` for ``dim=2`` and ``"randomized"`` for ``dim>=3``.
    use_precomputed_pebble_digraph:
        Only relevant if ``algorithm="sparsity"``.
        If ``True``, the :prf:ref:`pebble digraph <def-pebble-digraph>`
        present in the cache is used.
        If ``False``, recompute the pebble digraph.
        Use ``True`` only if you are certain that the pebble game digraph
        is consistent with the graph.
    prob:
        Only relevant if ``algorithm="randomized"``.
        It determines the bound on the probability of
        the randomized algorithm to yield false negatives.

    Examples
    --------
    >>> from pyrigi import Graph
    >>> G = Graph([(0,1), (1,2), (2,3), (3,0)])
    >>> G.is_rigid()
    False
    >>> G.add_edge(0,2)
    >>> G.is_rigid()
    True
    """
    _input_check.dimension(dim)
    _graph_input_check.no_loop(graph)

    n = graph.number_of_nodes()
    # edge count, compare :prf:ref:`thm-gen-rigidity-tight`
    if graph.number_of_edges() < dim * n - math.comb(dim + 1, 2):
        return False
    # small graphs are rigid iff complete :prf:ref:`thm-gen-rigidity-small-complete`
    elif n <= dim + 1:
        return graph.number_of_edges() == math.comb(n, 2)

    if algorithm == "default":
        if dim == 1:
            algorithm = "graphic"
        elif dim == 2:
            algorithm = "sparsity"
        else:
            algorithm = "randomized"
            warn_randomized_alg(graph, is_rigid, "algorithm='randomized'")

    if algorithm == "graphic":
        _input_check.dimension_for_algorithm(dim, [1], "the graphic algorithm")
        return nx.is_connected(graph)

    if algorithm == "sparsity":
        _input_check.dimension_for_algorithm(dim, [2], "the sparsity algorithm")
        pebble_digraph = sparsity._get_pebble_digraph(
            graph, 2, 3, use_precomputed_pebble_digraph=use_precomputed_pebble_digraph
        )
        return pebble_digraph.number_of_edges() == 2 * n - 3

    if algorithm == "randomized":
        N = int((n * dim - math.comb(dim + 1, 2)) / prob)
        if N < 1:
            raise ValueError("The parameter prob is too large!")
        from pyrigi.framework import Framework

        F = Framework.Random(graph, dim, rand_range=[1, N])
        return F.is_inf_rigid()

    if algorithm == "numerical":
        from pyrigi.framework import Framework

        F = Framework.Random(
            graph,
            dim,
            rand_range=[-1, 1],
            numerical=True,
        )
        return F.is_inf_rigid(numerical=True)

    raise NotSupportedValueError(algorithm, "algorithm", is_rigid)


def is_min_rigid(
    graph: nx.Graph,
    dim: int = 2,
    algorithm: str = "default",
    use_precomputed_pebble_digraph: bool = False,
    prob: float = 0.0001,
) -> bool:
    """
    Return whether the graph is minimally ``dim``-rigid.

    Definitions
    -----------
    :prf:ref:`Minimal dim-rigidity <def-min-rigid-graph>`

    Parameters
    ----------
    dim:
        Dimension.
    algorithm:
        If ``"graphic"`` (only if ``dim=1``), then the graphic matroid
        is used, namely, it is checked whether the graph is a tree.

        If ``"sparsity"`` (only if ``dim=2``),
        then :prf:ref:`(2,3)-tightness <def-kl-sparse-tight>` and
        :prf:ref:`thm-2-gen-rigidity` are used.

        If ``"randomized"``, a probabilistic check is performed.
        It may give false negatives (with probability at most ``prob``),
        but no false positives. See :prf:ref:`thm-probabilistic-rigidity-check`.

        If ``"extension_sequence"`` (only if ``dim=2``),
        then the existence of a sequence
        of rigidity preserving extensions is checked,
        see :meth:`.has_extension_sequence`.

        If ``"default"``, then ``"graphic"`` is used for ``dim=1``
        and ``"sparsity"`` for ``dim=2`` and ``"randomized"`` for ``dim>=3``.
    use_precomputed_pebble_digraph:
        Only relevant if ``algorithm="sparsity"``.
        If ``True``, the :prf:ref:`pebble digraph <def-pebble-digraph>`
        present in the cache is used.
        If ``False``, recompute the pebble digraph.
        Use ``True`` only if you are certain that the pebble game digraph
        is consistent with the graph.
    prob:
        Only relevant if ``algorithm="randomized"``.
        It determines the bound on the probability of
        the randomized algorithm to yield false negatives.

    Examples
    --------
    >>> G = Graph([(0,1), (1,2), (2,3), (3,0), (1,3)])
    >>> G.is_min_rigid()
    True
    >>> G.add_edge(0,2)
    >>> G.is_min_rigid()
    False

    Suggested Improvements
    ----------------------
    Implement ``algorithm="numerical"``.
    """
    _input_check.dimension(dim)
    _graph_input_check.no_loop(graph)

    n = graph.number_of_nodes()
    # small graphs are minimally rigid iff complete
    # :pref:ref:`thm-gen-rigidity-small-complete`
    if n <= dim + 1:
        return graph.number_of_edges() == math.comb(n, 2)
    # edge count, compare :prf:ref:`thm-gen-rigidity-tight`
    if graph.number_of_edges() != dim * n - math.comb(dim + 1, 2):
        return False

    if algorithm == "default":
        if dim == 1:
            algorithm = "graphic"
        elif dim == 2:
            algorithm = "sparsity"
        else:
            algorithm = "randomized"
            warn_randomized_alg(graph, is_min_rigid, "algorithm='randomized'")

    if algorithm == "graphic":
        _input_check.dimension_for_algorithm(dim, [1], "the graphic algorithm")
        return nx.is_tree(graph)

    if algorithm == "sparsity":
        _input_check.dimension_for_algorithm(
            dim, [2], "the (2,3)-sparsity/tightness algorithm"
        )
        return sparsity.is_kl_tight(
            graph,
            2,
            3,
            algorithm="pebble",
            use_precomputed_pebble_digraph=use_precomputed_pebble_digraph,
        )

    if algorithm == "extension_sequence":
        _input_check.dimension_for_algorithm(
            dim, [1, 2], "the algorithm using extension sequences"
        )
        return graph_extension.has_extension_sequence(graph, dim=dim)

    if algorithm == "randomized":
        N = int((n * dim - math.comb(dim + 1, 2)) / prob)
        if N < 1:
            raise ValueError("The parameter prob is too large!")
        from pyrigi.framework import Framework

        F = Framework.Random(graph, dim, rand_range=[1, N])
        return F.is_min_inf_rigid()

    raise NotSupportedValueError(algorithm, "algorithm", is_min_rigid)


def is_linked(graph: nx.Graph, u: Vertex, v: Vertex, dim: int = 2) -> bool:
    """
    Return whether a pair of vertices is ``dim``-linked.

    :prf:ref:`lem-linked-pair-rigid-component` is used for the check.

    Definitions
    -----------
    :prf:ref:`dim-linked pair <def-linked-pair>`

    Parameters
    ----------
    u,v:
    dim:
        Currently, only the dimension ``dim=2`` is supported.

    Examples
    --------
    >>> H = Graph([[0, 1], [0, 2], [1, 3], [1, 5], [2, 3], [2, 6], [3, 5], [3, 7], [5, 7], [6, 7], [3, 6]])
    >>> H.is_linked(1,7)
    True
    >>> H = Graph([[0, 1], [0, 2], [1, 3], [2, 3]])
    >>> H.is_linked(0,3)
    False
    >>> H.is_linked(1,3)
    True

    Suggested Improvements
    ----------------------
    Implement also for other dimensions.
    """  # noqa: E501
    _input_check.dimension_for_algorithm(dim, [2], "the algorithm to check linkedness")
    _graph_input_check.vertex_members(graph, [u, v])
    return any(
        [(u in C and v in C) for C in rigid_components(graph, algorithm="default")]
    )


def max_rigid_dimension(
    graph: nx.Graph, algorithm: str = "randomized", prob: float = 0.0001
) -> int | Inf:
    """
    Compute the maximum dimension in which the graph is generically rigid.

    For checking rigidity, the method uses a randomized algorithm,
    see :meth:`~.is_rigid` for details.

    Definitions
    -----------
    :prf:ref:`Generical rigidity <def-gen-rigid>`

    Parameters
    ----------
    algorithm:
        If ``"randomized"``, the rigidity of the graph is checked
        in each dimension using :meth:`.is_rigid` with
        ``algorithm="randomized"``.
        Since this is a randomized algorithm, false negatives are possible.
        However, the actual maximum rigid dimension is never lower than
        the output of this method.

        If ``"numerical"``, the rigidity of the graph is checked
        in each dimension using :meth:`.is_rigid` with
        ``algorithm="numerical"``.
        With this choice of algorithm, we do not have the guarantee that
        is mentioned above on the maximum rigid dimension.
    prob:
        A bound on the probability for false negatives of the rigidity testing.

        *Warning:* this is not the probability of wrong results in this method,
        but is just passed on to rigidity testing.

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.Complete(3)
    >>> rigid_dim = G.max_rigid_dimension(); rigid_dim
    oo
    >>> rigid_dim.is_infinite
    True

    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.Complete(4)
    >>> G.add_edges([(0,4),(1,4),(2,4)])
    >>> G.max_rigid_dimension()
    3

    Notes
    -----
    This is done by taking the dimension predicted by the Maxwell count
    as a starting point and iteratively reducing the dimension until
    generic rigidity is found.
    This method returns ``sympy.oo`` (infinity) if and only if the graph
    is complete. It has the data type ``Inf``.
    """
    _graph_input_check.no_loop(graph)

    if not nx.is_connected(graph):
        return 0

    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    # Only the complete graph is rigid in all dimensions
    if m == n * (n - 1) / 2:
        return oo
    # Find the largest d such that d*(d+1)/2 - d*n + m = 0
    max_dim = int(math.floor(0.5 * (2 * n + math.sqrt((1 - 2 * n) ** 2 - 8 * m) - 1)))
    warn_randomized_alg(graph, max_rigid_dimension)
    for dim in range(max_dim, 0, -1):
        if is_rigid(graph, dim, algorithm=algorithm, prob=prob):
            return dim


def rigid_components(  # noqa: 901
    graph: nx.Graph, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
) -> list[list[Vertex]]:
    """
    Return the list of the vertex sets of ``dim``-rigid components.

    Definitions
    -----
    :prf:ref:`Rigid components <def-rigid-components>`

    Parameters
    ---------
    dim:
        The dimension that is used for the rigidity check.
    algorithm:
        If ``"graphic"`` (only if ``dim=1``),
        then the connected components are returned.

        If ``"subgraphs-pebble"`` (only if ``dim=2``),
        then all subgraphs are checked
        using :meth:`.is_rigid` with ``algorithm="pebble"``.

        If ``"pebble"`` (only if ``dim=2``),
        then :meth:`.Rd_closure` with ``algorithm="pebble"``
        is used.

        If ``"randomized"``, all subgraphs are checked
        using :meth:`.is_rigid` with ``algorithm="randomized"``.

        If ``"numerical"``, all subgraphs are checked
        using :meth:`.is_rigid` with ``algorithm="numerical"``.

        If ``"default"``, then ``"graphic"`` is used for ``dim=1``,
        ``"pebble"`` for ``dim=2``, and ``"randomized"`` for ``dim>=3``.
    prob:
        A bound on the probability for false negatives of the rigidity testing
        when ``algorithm="randomized"``.

        *Warning:* this is not the probability of wrong results in this method,
        but is just passed on to rigidity testing.

    Examples
    --------
    >>> G = Graph([(0,1), (1,2), (2,3), (3,0)])
    >>> G.rigid_components(algorithm="randomized")
    [[0, 1], [0, 3], [1, 2], [2, 3]]

    >>> G = Graph([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0), (0,2), (5,3)])
    >>> G.is_rigid()
    False
    >>> G.rigid_components(algorithm="randomized")
    [[0, 5], [2, 3], [0, 1, 2], [3, 4, 5]]

    Notes
    -----
    If the graph itself is rigid, it is clearly maximal and is returned.
    Every edge is part of a rigid component. Isolated vertices form
    additional rigid components.

    For the pebble game algorithm we use the fact that the ``R2_closure``
    consists of edge disjoint cliques, so we only have to determine them.
    """
    _input_check.dimension(dim)
    _graph_input_check.no_loop(graph)

    if algorithm == "default":
        if dim == 1:
            algorithm = "graphic"
        elif dim == 2:
            algorithm = "pebble"
        else:
            algorithm = "randomized"
            warn_randomized_alg(graph, rigid_components, "algorithm='randomized'")

    if algorithm == "graphic":
        _input_check.dimension_for_algorithm(dim, [1], "the graphic algorithm")
        return [list(comp) for comp in nx.connected_components(graph)]

    if algorithm == "pebble":
        _input_check.dimension_for_algorithm(
            dim, [2], "the rigid component algorithm based on pebble games"
        )
        components = []

        import pyrigi.graph._rigidity.matroidal as matroidal_rigidity

        closure = nx.Graph(
            matroidal_rigidity.Rd_closure(graph, dim=2, algorithm="pebble")
        )
        for u, v in closure.edges:
            closure.edges[u, v]["used"] = False
        for u, v in closure.edges:
            if not closure.edges[u, v]["used"]:
                common_neighs = nx.common_neighbors(closure, u, v)
                comp = [u, v] + list(common_neighs)
                components.append(comp)
                for w1, w2 in combinations(comp, 2):
                    closure.edges[w1, w2]["used"] = True

        return components + [[v] for v in graph.nodes if nx.is_isolate(graph, v)]

    if algorithm in ["randomized", "numerical", "subgraphs-pebble"]:
        if not nx.is_connected(graph):
            res = []
            for comp in nx.connected_components(graph):
                res += rigid_components(graph.subgraph(comp), dim, algorithm=algorithm)
            return res

        if algorithm == "subgraphs-pebble":
            _input_check.dimension_for_algorithm(
                dim, [2], "the subgraph algorithm using pebble games"
            )
            alg_is_rigid = "sparsity"
        else:
            alg_is_rigid = algorithm

        if is_rigid(graph, dim, algorithm=alg_is_rigid, prob=prob):
            return [list(graph.nodes)]

        rigid_subgraphs = {
            tuple(vertex_subset): True
            for n in range(2, graph.number_of_nodes() - 1)
            for vertex_subset in combinations(graph.nodes, n)
            if is_rigid(
                graph.subgraph(vertex_subset), dim, algorithm=alg_is_rigid, prob=prob
            )
        }

        sorted_rigid_subgraphs = sorted(
            rigid_subgraphs.keys(), key=lambda t: len(t), reverse=True
        )
        for i, H1 in enumerate(sorted_rigid_subgraphs):
            if rigid_subgraphs[H1] and i + 1 < len(sorted_rigid_subgraphs):
                for H2 in sorted_rigid_subgraphs[i + 1 :]:
                    if set(H2).issubset(set(H1)):
                        rigid_subgraphs[H2] = False
        return [list(H) for H, is_max in rigid_subgraphs.items() if is_max]

    raise NotSupportedValueError(algorithm, "algorithm", rigid_components)
