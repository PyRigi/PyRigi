"""
This module provides algorithms related to the generic rigidity matroid.
"""

from copy import deepcopy
from itertools import combinations

import networkx as nx

import pyrigi._utils._input_check as _input_check
import pyrigi.graph._rigidity.generic as generic_rigidity
import pyrigi.graph._sparsity.sparsity as sparsity
import pyrigi.graph._utils._input_check as _graph_input_check
from pyrigi.data_type import Edge, Vertex
from pyrigi.exception import NotSupportedValueError
from pyrigi.warning import _warn_randomized_alg as warn_randomized_alg


def is_Rd_independent(
    graph: nx.Graph,
    dim: int = 2,
    algorithm: str = "default",
    use_precomputed_pebble_digraph: bool = False,
) -> bool:
    """
    Return whether the edge set is independent in the generic ``dim``-rigidity matroid.

    Definitions
    ---------
    * :prf:ref:`Independence <def-matroid>`
    * :prf:ref:`Generic rigidity matroid <def-gen-rigidity-matroid>`

    Parameters
    ---------
    dim:
        Dimension of the rigidity matroid.
    algorithm:
        If ``"graphic"`` (only if ``dim=1``),
        then the (non-)presence of cycles is checked.

        If ``"sparsity"`` (only if ``dim=2``),
        then :prf:ref:`(2,3)-sparsity <def-kl-sparse-tight>` is checked
        using the :prf:ref:`pebble game algorithm <alg-pebble-game>`.

        If ``"randomized"``, the following check is performed on a random framework:
        a set of edges forms an independent set in the rigidity matroid
        if and only if it has no self-stress, i.e.,
        there are no linear relations between the rows of the rigidity matrix.

        If ``"default"``, then ``"graphic"`` is used for ``dim=1``,
        ``"sparsity"`` for ``dim=2``, and ``"randomized"`` for ``dim>=3``.
    use_precomputed_pebble_digraph:
        Only relevant if ``algorithm="sparsity"``.
        If ``True``, the :prf:ref:`pebble digraph <def-pebble-digraph>`
        present in the cache is used.
        If ``False``, recompute the pebble digraph.
        Use ``True`` only if you are certain that the pebble game digraph
        is consistent with the graph.

    Examples
    --------
    >>> G = Graph([(0,1), (1,2), (2,3), (3,0)])
    >>> G.is_Rd_independent()
    True

    Suggested Improvements
    ----------------------
    ``prob`` parameter for the randomized algorithm.
    """  # noqa: E501
    _input_check.dimension(dim)
    _graph_input_check.no_loop(graph)

    if algorithm == "default":
        if dim == 1:
            algorithm = "graphic"
        elif dim == 2:
            algorithm = "sparsity"
        else:
            algorithm = "randomized"
            warn_randomized_alg(
                graph, is_Rd_independent, explicit_call="algorithm='randomized"
            )

    if algorithm == "graphic":
        _input_check.dimension_for_algorithm(
            dim,
            [
                1,
            ],
            "the graphic algorithm",
        )
        return len(nx.cycle_basis(graph)) == 0

    if algorithm == "sparsity":
        _input_check.dimension_for_algorithm(dim, [2], "the sparsity algorithm")
        return sparsity.is_kl_sparse(
            graph, 2, 3, use_precomputed_pebble_digraph=use_precomputed_pebble_digraph
        )

    elif algorithm == "randomized":
        from pyrigi.framework import Framework

        F = Framework.Random(graph, dim=dim)
        return len(F.stresses()) == 0

    raise NotSupportedValueError(algorithm, "algorithm", is_Rd_independent)


def is_Rd_dependent(
    graph: nx.Graph,
    dim: int = 2,
    algorithm: str = "default",
    use_precomputed_pebble_digraph: bool = False,
) -> bool:
    """
    Return whether the edge set is dependent in the generic ``dim``-rigidity matroid.

    See :meth:`.is_Rd_independent` for the possible parameters.

    Definitions
    -----------
    * :prf:ref:`Dependence <def-matroid>`
    * :prf:ref:`Generic rigidity matroid <def-gen-rigidity-matroid>`

    Examples
    --------
    >>> from pyrigi import graphDB
    >>> G = graphDB.K33plusEdge()
    >>> G.is_Rd_dependent()
    True

    Notes
    -----
    See :meth:`.is_independent` for details.
    """
    return not is_Rd_independent(
        graph,
        dim,
        algorithm=algorithm,
        use_precomputed_pebble_digraph=use_precomputed_pebble_digraph,
    )


def is_Rd_circuit(  # noqa: C901
    graph: nx.Graph,
    dim: int = 2,
    algorithm: str = "default",
    use_precomputed_pebble_digraph: bool = False,
) -> bool:
    """
    Return whether the edge set is a circuit in the generic ``dim``-rigidity matroid.

    Definitions
    ---------
    * :prf:ref:`Circuit <def-matroid>`
    * :prf:ref:`Generic rigidity matroid <def-gen-rigidity-matroid>`

    Parameters
    ---------
    dim:
        Dimension of the rigidity matroid.
    algorithm:
        If ``"graphic"`` (only if ``dim=1``),
        it is checked whether the graph is a union of cycles.

        If ``"sparsity"`` (only if ``dim=2``),
        a :prf:ref:`(2,3)-sparse <def-kl-sparse-tight>` spanning subgraph
        is computed (using :prf:ref:`pebble games <alg-pebble-game>`)
        and checked whether it misses only a single edge
        whose fundamental circuit is the whole graph.

        If ``"randomized"``, it is checked using randomized
        :meth:`.is_Rd_independent` whether removing
        every single edge from the graph results in an Rd-independent graph.

        If ``"default"``, then ``"graphic"`` is used for ``dim=1``,
        ``"sparsity"`` for ``dim=2``, and ``"randomized"`` for ``dim>=3``.
    use_precomputed_pebble_digraph:
        Only relevant if ``algorithm="sparsity"``.
        If ``True``, the :prf:ref:`pebble digraph <def-pebble-digraph>`
        present in the cache is used.
        If ``False``, recompute the pebble digraph.
        Use ``True`` only if you are certain that the pebble game digraph
        is consistent with the graph.

    Examples
    --------
    >>> from pyrigi import graphDB
    >>> G = graphDB.K33plusEdge()
    >>> G.is_Rd_circuit()
    True
    >>> G.add_edge(1,2)
    >>> G.is_Rd_circuit()
    False

    Suggested Improvements
    ----------------------
    ``prob`` parameter for the randomized algorithm
    """
    _input_check.dimension(dim)
    _graph_input_check.no_loop(graph)

    if algorithm == "default":
        if dim == 1:
            algorithm = "graphic"
        elif dim == 2:
            algorithm = "sparsity"
        else:
            algorithm = "randomized"
            warn_randomized_alg(
                graph, is_Rd_circuit, explicit_call="algorithm='randomized'"
            )

    if algorithm == "graphic":
        _input_check.dimension_for_algorithm(dim, [1], "the graphic algorithm")
        # Check if every vertex has degree 2 or 0
        V = []
        for vertex in graph.nodes:
            if graph.degree(vertex) != 2 and graph.degree(vertex) != 0:
                return False
            if graph.degree(vertex) == 2:
                V.append(vertex)
        H = graph.subgraph(V)
        if not nx.is_connected(H):
            return False
        return True

    if algorithm == "sparsity":
        _input_check.dimension_for_algorithm(dim, [2], "the sparsity algorithm")
        # get max sparse sugraph and check the fundamental circuit of
        # the one last edge
        if graph.number_of_edges() != 2 * graph.number_of_nodes() - 2:
            return False
        max_sparse_subgraph = sparsity.spanning_kl_sparse_subgraph(
            graph,
            K=2,
            L=3,
            use_precomputed_pebble_digraph=use_precomputed_pebble_digraph,
        )
        if max_sparse_subgraph.number_of_edges() != 2 * graph.number_of_nodes() - 3:
            return False

        remaining_edges = [
            e for e in graph.edges if not max_sparse_subgraph.has_edge(*e)
        ]
        if len(remaining_edges) != 1:
            # this should not happen
            raise RuntimeError

        pebble_digraph = sparsity._get_pebble_digraph(
            graph, K=2, L=3, use_precomputed_pebble_digraph=True
        )
        return (
            len(
                pebble_digraph.fundamental_circuit(
                    u=remaining_edges[0][0],
                    v=remaining_edges[0][1],
                )
            )
            == graph.number_of_nodes()
        )
    elif algorithm == "randomized":
        if is_Rd_independent(graph, dim=dim, algorithm="randomized"):
            return False
        G = deepcopy(graph)
        for e in G.edges:
            G.remove_edge(*e)
            if is_Rd_dependent(G, dim=dim, algorithm="randomized"):
                return False
            G.add_edge(*e)
        return True

    raise NotSupportedValueError(algorithm, "algorithm", is_Rd_circuit)


def is_Rd_closed(graph: nx.Graph, dim: int = 2, algorithm: str = "default") -> bool:
    """
    Return whether the edge set is closed in the generic ``dim``-rigidity matroid.

    Definitions
    -----------
    * :prf:ref:`Rd-closed <def-rank-function-closure>`
    * :prf:ref:`Generic rigidity matroid <def-gen-rigidity-matroid>`

    Parameters
    ---------
    dim:
        Dimension of the rigidity matroid.
    algorithm:
        See :meth:`.Rd_closure` for the options.

    Examples
    --------
    >>> G = Graph([(0,1),(1,2),(0,2),(3,4)])
    >>> G.is_Rd_closed(dim=1)
    True
    """
    return len(Rd_closure(graph, dim, algorithm)) == graph.number_of_edges()


def Rd_closure(graph: nx.Graph, dim: int = 2, algorithm: str = "default") -> list[Edge]:
    """
    Return the set of edges given by closure in the generic ``dim``-rigidity matroid.

    Definitions
    -----------
    * :prf:ref:`Rd-closure <def-rank-function-closure>`
    * :prf:ref:`Generic rigidity matroid <def-gen-rigidity-matroid>`

    Parameters
    ---------
    dim:
        Dimension of the rigidity matroid.
    algorithm:
        If ``"graphic"`` (only if ``dim=1``),
        then the closure is computed using connected components.

        If ``"pebble"`` (only if ``dim=2``),
        then pebble games are used
        (see notes below and :prf:ref:`alg-pebble-game`).

        If ``"randomized"``, then adding
        non-edges is tested one by one on a random framework.

        If ``"default"``, then ``"graphic"`` is used
        for ``dim=1``, ``"pebble"`` for ``dim=2``
        and ``"randomized"`` for ``dim>=3``.

    Examples
    --------
    >>> G = Graph([(0,1),(0,2),(3,4)])
    >>> G.Rd_closure(dim=1)
    [[0, 1], [0, 2], [1, 2], [3, 4]]

    Notes
    -----
    The pebble game algorithm proceeds as follows:
    Iterate through the vertex pairs of each connected component
    and check if there exists a rigid component containing both.
    This can be done by trying to add a new edge between the vertices.
    If there is such a rigid component, we can add every vertex pair from there:
    they are certainly within a rigid component.

    Suggested Improvements
    ----------------------
    ``prob`` parameter for the randomized algorithm
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
            warn_randomized_alg(graph, Rd_closure, "algorithm='randomized'")

    if algorithm == "graphic":
        _input_check.dimension_for_algorithm(dim, [1], "the graphic algorithm ")
        return [
            [u, v]
            for comp in nx.connected_components(graph)
            for u, v in combinations(comp, 2)
        ]

    if algorithm == "pebble":
        _input_check.dimension_for_algorithm(
            dim, [2], "the algorithm based on pebble games"
        )

        pebble_digraph = sparsity._get_pebble_digraph(graph, 2, 3)
        if pebble_digraph.number_of_edges() == 2 * graph.number_of_nodes() - 3:
            return list(combinations(graph.nodes, 2))
        else:
            closure = deepcopy(graph)
            for connected_comp in nx.connected_components(graph):
                for u, v in combinations(connected_comp, 2):
                    if not closure.has_edge(u, v):
                        circuit = pebble_digraph.fundamental_circuit(u, v)
                        if circuit is not None:
                            for e in combinations(circuit, 2):
                                closure.add_edge(*e)
            return list(closure.edges)

    if algorithm == "randomized":
        from pyrigi.framework import Framework

        F_rank = Framework.Random(graph, dim=dim).rigidity_matrix_rank()
        G = deepcopy(graph)
        result = list(G.edges)
        for e in combinations(graph.nodes, 2):
            if G.has_edge(*e):
                continue
            G.add_edge(*e)
            F1 = Framework.Random(G, dim=dim)
            if F_rank == F1.rigidity_matrix_rank():
                result.append(e)
            G.remove_edge(*e)
        return result

    raise NotSupportedValueError(algorithm, "algorithm", Rd_closure)


def _Rd_fundamental_circuit(
    graph: nx.Graph, u: Vertex, v: Vertex, dim: int = 2
) -> list[Edge]:
    """
    Return the fundamental circuit of ``uv`` in the generic ``dim``-rigidity matroid.

    Definitions
    -----------
    * :prf:ref:`Fundamental circuit <def-fundamental-circuit>`
    * :prf:ref:`Generic rigidity matroid <def-gen-rigidity-matroid>`

    Parameters
    ----------
    u, v:
    dim:
        Currently, only the dimension ``dim=2`` is supported.

    Examples
    --------
    >>> H = Graph([[0, 1], [0, 2], [1, 3], [1, 5], [2, 3], [2, 6], [3, 5], [3, 7], [5, 7], [6, 7], [3, 6]])
    >>> sorted(_Rd_fundamental_circuit(H, 1, 7))
    [(1, 3), (1, 5), (3, 5), (3, 7), (5, 7)]
    >>> sorted(_Rd_fundamental_circuit(H, 2, 5))
    [(2, 3), (2, 6), (3, 5), (3, 6), (3, 7), (5, 7), (6, 7)]

    The following example is the Figure 5 of the article :cite:p:`JordanVillanyi2024`

    >>> from pyrigi.graph._rigidity.global_ import _block_3
    >>> G = Graph([[0, 1], [0, 5], [0, 7], [1, 2], [1, 3], [1, 7], [2, 3], [2, 4], [3, 4], [4, 5], [4, 8], [4, 11], [5, 6], [5, 8], [5, 14], [6, 10], [6, 11], [6, 12], [7, 8], [7, 13], [8, 12], [9, 10], [9, 13], [10, 14], [11, 12], [13, 14]])
    >>> H = _block_3(G, 0,11)
    >>> sorted([tuple(sorted(list(e))) for e in _Rd_fundamental_circuit(H, 0, 11)])
    [(0, 1), (0, 5), (0, 7), (1, 4), (1, 7), (4, 5), (4, 8), (4, 11), (5, 6), (5, 8), (6, 11), (6, 12), (7, 8), (8, 12), (11, 12)]

    Suggested Improvements
    ----------------------
    Implement also other dimensions.
    """  # noqa: E501

    _input_check.dimension_for_algorithm(
        dim, [2], "the algorithm that computes a circuit"
    )
    _graph_input_check.no_loop(graph)
    _graph_input_check.vertex_members(graph, [u, v])
    # check (u, v) are non-adjacent linked pair
    if graph.has_edge(u, v):
        raise ValueError("The vertices must not be connected by an edge.")
    elif not generic_rigidity.is_linked(graph, u, v, dim=dim):
        raise ValueError("The vertices must be a linked pair.")

    pebble_digraph = sparsity._get_pebble_digraph(graph, K=2, L=3)
    set_nodes = pebble_digraph.fundamental_circuit(u, v)
    F = nx.Graph(pebble_digraph.to_undirected())
    return list(nx.subgraph(F, set_nodes).edges)
