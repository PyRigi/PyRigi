"""
This module provides algorithms related to global rigidity.
"""

import math
from copy import deepcopy
from random import randint

import networkx as nx
from sympy import zeros

import pyrigi._utils._input_check as _input_check
import pyrigi.graph._rigidity.generic as generic_rigidity
import pyrigi.graph._rigidity.redundant as redundant_rigidity
import pyrigi.graph._sparsity.sparsity as sparsity
import pyrigi.graph._utils._input_check as _graph_input_check
from pyrigi.data_type import Vertex
from pyrigi.exception import NotSupportedValueError
from pyrigi.warning import _warn_randomized_alg as warn_randomized_alg


def is_globally_rigid(
    graph: nx.Graph, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
) -> bool:
    """
    Return whether the graph is globally ``dim``-rigid.

    Definitions
    -----------
    :prf:ref:`Global dim-rigidity <def-globally-rigid-graph>`

    Parameters
    ----------
    dim:
        Dimension.
    algorithm:
        If ``"graphic"`` (only if ``dim=1``), then 2-connectivity is checked.

        If ``"redundancy"`` (only if ``dim=2``),
        then :prf:ref:`thm-globally-redundant-3connected` is used.

        If ``"randomized"``, a probabilistic check is performed.
        It may give false negatives (with probability at most ``prob``),
        but no false positives. See :prf:ref:`thm-globally-randomize-algorithm`.

        If ``"default"``, then ``"graphic"`` is used for ``dim=1``,
        ``"redundancy"`` for ``dim=2``, and ``"randomized"`` for ``dim>=3``.
    prob:
        Only relevant if ``algorithm="randomized"``.
        It determines the bound on the probability of
        the randomized algorithm to yield false negatives.

    Examples
    --------
    >>> G = Graph([(0,1), (1,2), (2,0)])
    >>> G.is_globally_rigid()
    True
    >>> import pyrigi.graphDB as graphs
    >>> J = graphs.ThreePrism()
    >>> J.is_globally_rigid(dim=3)
    False
    >>> J.is_globally_rigid()
    False
    >>> K = graphs.Complete(6)
    >>> K.is_globally_rigid()
    True
    >>> K.is_globally_rigid(dim=3)
    True
    >>> C = graphs.CompleteMinusOne(5)
    >>> C.is_globally_rigid()
    True
    >>> C.is_globally_rigid(dim=3)
    False
    """
    _input_check.dimension(dim)
    _graph_input_check.no_loop(graph)

    # small graphs are globally rigid iff complete
    # :pref:ref:`thm-gen-rigidity-small-complete`
    n = graph.number_of_nodes()
    if n <= dim + 1:
        return graph.number_of_edges() == math.comb(n, 2)

    if algorithm == "default":
        if dim == 1:
            algorithm = "graphic"
        elif dim == 2:
            algorithm = "redundancy"
        else:
            algorithm = "randomized"
            warn_randomized_alg(graph, is_globally_rigid, "algorithm='randomized'")

    if algorithm == "graphic":
        _input_check.dimension_for_algorithm(dim, [1], "the graphic algorithm")
        return nx.node_connectivity(graph) >= 2

    if algorithm == "redundancy":
        _input_check.dimension_for_algorithm(dim, [2], "the algorithm using redundancy")
        return (
            redundant_rigidity.is_k_redundantly_rigid(graph, k=1)
            and nx.node_connectivity(graph) >= 3
        )

    if algorithm == "randomized":
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        t = n * dim - math.comb(dim + 1, 2)  # rank of the rigidity matrix
        N = int(1 / prob) * n * math.comb(n, 2) + 2

        if m < t:
            return False
        # take a random framework with integer coordinates
        from pyrigi.framework import Framework

        F = Framework.Random(graph, dim=dim, rand_range=[1, N])
        stresses = F.stresses()
        if m == t:
            omega = zeros(F.rigidity_matrix().rows, 1)
            return F.stress_matrix(omega).rank() == n - dim - 1
        elif stresses:
            omega = sum([randint(1, N) * stress for stress in stresses], stresses[0])
            return F.stress_matrix(omega).rank() == n - dim - 1
        else:
            raise RuntimeError("There must be at least one stress but none was found!")
    raise NotSupportedValueError(algorithm, "algorithm", is_globally_rigid)


def _neighbors_of_set(
    graph: nx.Graph, vertices: list[Vertex] | set[Vertex]
) -> set[Vertex]:
    """
    Return the set of neighbors of a set of vertices.

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.Complete(5)
    >>> _neighbors_of_set(G, [1,2])
    {0, 3, 4}
    >>> G = Graph([[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
    >>> _neighbors_of_set(G, [1,2])
    {3, 4}
    >>> _neighbors_of_set(G, [3,4])
    {0, 1, 2}

    """  # noqa: E501

    _graph_input_check.vertex_members(graph, vertices)

    res = set()
    for v in vertices:
        res.update(graph.neighbors(v))
    return res.difference(vertices)


def _make_outside_neighbors_clique(
    graph: nx.Graph, vertices: list[Vertex] | set[Vertex]
) -> nx.Graph:
    """
    Create a graph by selecting the subgraph of the given graph induced by ``vertices``,
    contracting each connected component of the graph minus ``vertices``
    to a single vertex, and making their neighbors in ``vertices`` into a clique.
    See :prf:ref:`thm-weakly-globally-linked`.

    Definitions
    -----------
    :prf:ref:`clique <def-clique>`

    Examples
    --------
    >>> G = Graph([[0, 1], [0, 3], [0, 4], [1, 2], [1, 5], [2, 3], [2, 4], [3, 5]])
    >>> H = _make_outside_neighbors_clique(G, [0,1,2,3])
    >>> print(H)
    Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    >>> G = Graph([[0, 1], [0, 5], [0, 7], [1, 4], [1, 7], [4, 5], [4, 8], [4, 11], [5, 6], [5, 8], [5, 14], [6, 10], [6, 11], [6, 12], [7, 8], [7, 13], [8, 12], [10, 13], [10, 14], [11, 12], [13, 14]])
    >>> H = _make_outside_neighbors_clique(G, [0,1,4,5,6,7,8,11,12])
    >>> print(H)
    Graph with vertices [0, 1, 4, 5, 6, 7, 8, 11, 12] and edges [[0, 1], [0, 5], [0, 7], [1, 4], [1, 7], [4, 5], [4, 8], [4, 11], [5, 6], [5, 7], [5, 8], [6, 7], [6, 11], [6, 12], [7, 8], [8, 12], [11, 12]]
    """  # noqa: E501

    _graph_input_check.vertex_members(graph, vertices)

    H = deepcopy(graph)
    H.remove_nodes_from(vertices)
    conn_comps = nx.connected_components(H)
    H = deepcopy(graph)
    import pyrigi.graphDB as graphs

    for conn_comp in conn_comps:
        H.remove_nodes_from(conn_comp)
        K = graphs.Complete(vertices=_neighbors_of_set(graph, conn_comp))
        H = K + H
    return H


def _block_3(graph: nx.Graph, u: Vertex, v: Vertex) -> nx.Graph:
    """
    Return the 3-block of (``u``, ``v``) via cleaving operations.

    Definitions
    -----------
    :prf:ref:`3-block <def-block-3>`
    :prf:ref:`3-block lemma <lem-3-block>`

    Examples
    --------
    >>> G = Graph([[0, 1], [0, 5], [0, 7], [1, 2], [1, 3], [1, 7], [2, 3], [2, 4], [3, 4], [4, 5], [4, 8], [4, 11], [5, 6], [5, 8], [5, 14], [6, 10], [6, 11], [6, 12], [7, 8], [7, 13], [8, 12], [9, 10], [9, 13], [10, 14], [11, 12], [13, 14]])
    >>> print(_block_3(G, 0,11))
    Graph with vertices [0, 1, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14] and edges [[0, 1], [0, 5], [0, 7], [1, 4], [1, 7], [4, 5], [4, 8], [4, 11], [5, 6], [5, 8], [5, 14], [6, 10], [6, 11], [6, 12], [7, 8], [7, 13], [8, 12], [10, 13], [10, 14], [11, 12], [13, 14]]
    """  # noqa: E501
    try:
        cut = next(nx.all_node_cuts(graph))
        if len(cut) >= 3:
            return graph
        H = deepcopy(graph)
        H.remove_nodes_from(cut)
        for conn_comp in nx.connected_components(H):
            conn_comp.update(cut)
            if u in conn_comp and v in conn_comp:
                break
        B = nx.subgraph(graph, conn_comp).copy()
        B.add_edge(*cut)
        return _block_3(B, u, v)
    except StopIteration:
        return graph


def is_weakly_globally_linked(
    graph: nx.Graph, u: Vertex, v: Vertex, dim: int = 2
) -> bool:
    """
    Return whether the vertices ``u`` and ``v`` are weakly globally ``dim``-linked.

    :prf:ref:`thm-weakly-globally-linked` is used for the check.

    Definitions
    -----------
    :prf:ref:`Weakly globally linked pair <def-globally-linked>`

    Parameters
    ----------
    u, v:
    dim:
        Currently, only the dimension ``dim=2`` is supported.

    Examples
    --------
    >>> G = Graph([[0,4],[0,6],[0,7],[1,3],[1,6],[1,7],[2,6],[2,7],[3,5],[4,5],[4,7],[5,6],[5,7],[6,7]])
    >>> G.is_weakly_globally_linked(0,1)
    True
    >>> G.is_weakly_globally_linked(1,5)
    True
    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.Complete(10)
    >>> G.is_weakly_globally_linked(0,1)
    True

    The following example is Figure 1 of the article :cite:p:`JordanVillanyi2024`

    >>> G = Graph([[0,1],[0,2],[0,4],[1,2],[1,4],[2,3],[3,4]])
    >>> G.is_weakly_globally_linked(2,4)
    True
    """  # noqa: E501

    _input_check.dimension_for_algorithm(dim, [2], "the weakly globally linked method")
    _graph_input_check.vertex_members(graph, [u, v])
    # we focus on the 2-connected components of the graph
    # and check if the two given vertices are in the same 2-connected component
    if not nx.is_biconnected(graph):
        for bicon_comp in nx.biconnected_components(graph):
            if u in bicon_comp and v in bicon_comp:
                F = nx.subgraph(graph, bicon_comp)
                return F.is_weakly_globally_linked(u, v)
        return False
    # check (u,v) are non adjacent
    if graph.has_edge(u, v):
        return True  # they are actually globally linked, not just weakly
    # check (u,v) are linked pair
    if not generic_rigidity.is_linked(graph, u, v, dim=dim):
        return False

    # check (u,v) are such that kappa_graph(u,v) > 2
    if nx.algorithms.connectivity.local_node_connectivity(graph, u, v) <= 2:
        return False

    # if (u,v) separating pair in graph
    H = deepcopy(graph)
    H.remove_nodes_from([u, v])
    if not nx.is_connected(H):
        return True
    # OR
    # elif Clique(B,V_0) is globally rigid
    B = _block_3(graph, u, v)
    pebble_digraph = sparsity._get_pebble_digraph(B, K=2, L=3)
    V_0 = pebble_digraph.fundamental_circuit(u, v)
    return is_globally_rigid(_make_outside_neighbors_clique(B, V_0))
