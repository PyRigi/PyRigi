"""
This module provides algorithms related to graph sparsity.
"""

import math
from itertools import combinations

import networkx as nx

import pyrigi._utils._input_check as _input_check
from pyrigi.exception import NotSupportedValueError
from pyrigi.graph._sparsity._pebble_digraph import PebbleDiGraph


def _build_pebble_digraph(graph: nx.Graph, K: int, L: int) -> None:
    r"""
    Build and save the pebble digraph from scratch.

    Edges are added one-by-one, as long as they can.
    Discard edges that are not :prf:ref:`(K, L)-independent <def-kl-sparse-tight>`
    from the rest of the graph.

    Parameters
    ----------
    K:
    L:
    """
    _input_check.pebble_values(K, L)

    dir_graph = PebbleDiGraph(K, L)
    dir_graph.add_nodes_from(graph.nodes)
    for edge in graph.edges:
        u, v = edge[0], edge[1]
        dir_graph.add_edge_maintaining_digraph(u, v)
    graph._pebble_digraph = dir_graph


def _get_pebble_digraph(
    graph: nx.Graph, K: int, L: int, use_precomputed_pebble_digraph: bool = False
) -> PebbleDiGraph:
    """
    Return the pebble digraph for the graph.

    Parameters
    ----------
    use_precomputed_pebble_digraph:
        If ``use_precomputed_pebble_digraph`` is ``True``,
        then the cached one is used. Otherwise,
        :func:`_build_pebble_digraph` is called first.
        Use ``True`` only if you are certain that the pebble game digraph
        is consistent with the graph.
    """
    if (
        not use_precomputed_pebble_digraph
        or not hasattr(graph, "_pebble_digraph")
        or K != graph._pebble_digraph.K
        or L != graph._pebble_digraph.L
    ):
        _build_pebble_digraph(graph, K, L)
    return graph._pebble_digraph


def spanning_kl_sparse_subgraph(
    graph: nx.Graph, K: int, L: int, use_precomputed_pebble_digraph: bool = False
) -> nx.Graph:
    r"""
    Return a maximal (``K``, ``L``)-sparse subgraph.

    Based on the directed graph calculated by the :prf:ref:`pebble game algorithm <alg-pebble-game>`, return
    a maximal :prf:ref:`(K, L)-sparse <def-kl-sparse-tight>` of the graph.
    There are multiple possible maximal (``K``, ``L``)-sparse subgraphs, all of which have
    the same number of edges.

    Definitions
    -----------
    :prf:ref:`(K, L)-sparsity <def-kl-sparse-tight>`

    Parameters
    ----------
    K:
    L:
    use_precomputed_pebble_digraph:
        If ``True``, the :prf:ref:`pebble digraph <def-pebble-digraph>`
        present in the cache is used.
        If ``False``, recompute the pebble digraph.
        Use ``True`` only if you are certain that the pebble game digraph
        is consistent with the graph.

    Examples
    --------
    >>> from pyrigi import graphDB
    >>> G = graphDB.Complete(4)
    >>> H = G.spanning_kl_sparse_subgraph(2,3)
    >>> print(H)
    Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]]
    """  # noqa: E501
    pebble_digraph = _get_pebble_digraph(graph, K, L, use_precomputed_pebble_digraph)

    return graph.__class__(pebble_digraph.to_undirected())


def _is_pebble_digraph_sparse(
    graph: nx.Graph, K: int, L: int, use_precomputed_pebble_digraph: bool = False
) -> bool:
    """
    Return whether the pebble digraph has the same number of edges as the graph.

    Definitions
    -----------
    :prf:ref:`pebble digraph <def-pebble-digraph>`

    Parameters
    ----------
    K:
    L:
    use_precomputed_pebble_digraph:
        If ``True``, the pebble digraph present in the cache is used.
        If ``False``, recompute the pebble digraph.
        Use ``True`` only if you are certain that the pebble game digraph
        is consistent with the graph.
    """
    pebble_digraph = _get_pebble_digraph(graph, K, L, use_precomputed_pebble_digraph)

    # all edges are in fact inside the pebble digraph
    return graph.number_of_edges() == pebble_digraph.number_of_edges()


def is_kl_sparse(
    graph: nx.Graph,
    K: int,
    L: int,
    algorithm: str = "default",
    use_precomputed_pebble_digraph: bool = False,
) -> bool:
    r"""
    Return whether the graph is (``K``, ``L``)-sparse.

    Definitions
    -----------
    :prf:ref:`(K, L)-sparsity <def-kl-sparse-tight>`

    Parameters
    ----------
    K:
    L:
    algorithm:
        If ``"pebble"``, the function uses the pebble game algorithm to check
        for sparseness (see :prf:ref:`alg-pebble-game`).
        If ``"subgraph"``, it checks each subgraph following the definition.
        It defaults to ``"pebble"`` whenever ``K>0`` and ``0<=L<2K``,
        otherwise to ``"subgraph"``.
    use_precomputed_pebble_digraph:
        If ``True``, the :prf:ref:`pebble digraph <def-pebble-digraph>`
        present in the cache is used.
        If ``False``, recompute the pebble digraph.
        Use ``True`` only if you are certain that the pebble game digraph
        is consistent with the graph.

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.DoubleBanana()
    >>> G.is_kl_sparse(3,6)
    True
    >>> G.add_edge(0,1)
    >>> G.is_kl_sparse(3,6)
    False
    """
    _input_check.integrality_and_range(K, "K", min_val=1)
    _input_check.integrality_and_range(L, "L", min_val=0)

    if algorithm == "default":
        try:
            _input_check.pebble_values(K, L)
            algorithm = "pebble"
        except ValueError:
            algorithm = "subgraph"

    if algorithm == "pebble":
        _input_check.pebble_values(K, L)
        return _is_pebble_digraph_sparse(
            graph, K, L, use_precomputed_pebble_digraph=use_precomputed_pebble_digraph
        )

    if algorithm == "subgraph":
        if nx.number_of_selfloops(graph) > 0:
            _input_check.pebble_values(K, L)
            for i in range(1, graph.number_of_nodes() + 1):
                for vertex_set in combinations(graph.nodes, i):
                    G = graph.subgraph(vertex_set)
                    m = G.number_of_edges()
                    if m >= 1 and m > K * G.number_of_nodes() - L:
                        return False
            return True
        else:
            _input_check.integrality_and_range(
                L, "L", min_val=0, max_val=math.comb(K + 1, 2)
            )
            for i in range(K, graph.number_of_nodes() + 1):
                for vertex_set in combinations(graph.nodes, i):
                    G = graph.subgraph(vertex_set)
                    if G.number_of_edges() > K * G.number_of_nodes() - L:
                        return False
            return True

    # reaching this position means that the algorithm is unknown
    raise NotSupportedValueError(algorithm, "algorithm", is_kl_sparse)


def is_kl_tight(
    graph: nx.Graph,
    K: int,
    L: int,
    algorithm: str = "default",
    use_precomputed_pebble_digraph: bool = False,
) -> bool:
    r"""
    Return whether the graph is (``K``, ``L``)-tight.

    Definitions
    -----------
    :prf:ref:`(K, L)-tightness <def-kl-sparse-tight>`

    Parameters
    ----------
    K:
    L:
    algorithm:
        See :meth:`.is_kl_sparse`.
    use_precomputed_pebble_digraph:
        If ``True``, the :prf:ref:`pebble digraph <def-pebble-digraph>`
        present in the cache is used.
        If ``False``, recompute the pebble digraph.
        Use ``True`` only if you are certain that the pebble game digraph
        is consistent with the graph.

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.Complete(4)
    >>> G.is_kl_tight(2,2)
    True
    >>> G1 = graphs.CompleteBipartite(4,4)
    >>> G1.is_kl_tight(3,6)
    False
    """
    return graph.number_of_edges() == K * graph.number_of_nodes() - L and is_kl_sparse(
        graph,
        K,
        L,
        algorithm,
        use_precomputed_pebble_digraph=use_precomputed_pebble_digraph,
    )
