"""
This module provides algorithms related to redundant rigidity.
"""

import math
from copy import deepcopy
from itertools import combinations

import networkx as nx

import pyrigi._utils._input_check as _input_check
import pyrigi.graph._general as graph_general
import pyrigi.graph._rigidity.generic as generic_rigidity
import pyrigi.graph._utils._input_check as _graph_input_check


def is_k_vertex_redundantly_rigid(
    graph: nx.Graph,
    k: int,
    dim: int = 2,
    algorithm: str = "default",
    prob: float = 0.0001,
) -> bool:
    """
    Return whether the graph is ``k``-vertex redundantly ``dim``-rigid.

    Preliminary checks from
    :prf:ref:`thm-k-vertex-redundant-edge-bound-general`,
    :prf:ref:`thm-k-vertex-redundant-edge-bound-general2`,
    :prf:ref:`thm-1-vertex-redundant-edge-bound-dim2`,
    :prf:ref:`thm-2-vertex-redundant-edge-bound-dim2`,
    :prf:ref:`thm-k-vertex-redundant-edge-bound-dim2`,
    :prf:ref:`thm-3-vertex-redundant-edge-bound-dim3` and
    :prf:ref:`thm-k-vertex-redundant-edge-bound-dim3`
    are used.

    Definitions
    -----------
    :prf:ref:`k-vertex redundant dim-rigidity <def-redundantly-rigid-graph>`

    Parameters
    ----------
    k:
        level of redundancy
    dim:
        dimension
    algorithm:
        See :meth:`.is_rigid` for the possible algorithms used
        for checking rigidity in this method.
    prob:
        bound on the probability for false negatives of the rigidity testing
        when ``algorithm="randomized"``.

        *Warning:* this is not the probability of wrong results in this method
        but is just passed on to rigidity testing.

    Examples
    --------
    >>> G = Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3],
    ...            [1, 4], [2, 3], [2, 4], [3, 4]])
    >>> G.is_k_vertex_redundantly_rigid(1, 2)
    True
    >>> G.is_k_vertex_redundantly_rigid(2, 2)
    False
    >>> G = Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 4], [3, 4]])
    >>> G.is_k_vertex_redundantly_rigid(1, 2)
    False
    """
    _input_check.dimension(dim)
    _input_check.integrality_and_range(k, "k", min_val=0)
    _graph_input_check.no_loop(graph)

    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    # :prf:ref:`from thm-vertex-red-min-deg`
    if n >= dim + k + 1 and graph_general.min_degree(graph) < dim + k:
        return False
    if dim == 1:
        return nx.node_connectivity(graph) >= k + 1
    if (
        dim == 2
        and (
            # edge bound from :prf:ref:`thm-1-vertex-redundant-edge-bound-dim2`
            (k == 1 and n >= 5 and m < 2 * n - 1)
            or
            # edge bound from :prf:ref:`thm-2-vertex-redundant-edge-bound-dim2`
            (k == 2 and n >= 6 and m < 2 * n + 2)
            or
            # edge bound from :prf:ref:`thm-k-vertex-redundant-edge-bound-dim2`
            (k >= 3 and n >= 6 * (k + 1) + 23 and m < ((k + 2) * n + 1) // 2)
        )
    ) or (
        dim == 3
        and (
            # edge bound from :prf:ref:`thm-3-vertex-redundant-edge-bound-dim3`
            (k == 3 and n >= 15 and m < 3 * n + 5)
            or
            # edge bound from :prf:ref:`thm-k-vertex-redundant-edge-bound-dim3`
            (
                k >= 4
                and n >= 12 * (k + 1) + 10
                and n % 2 == 0
                and m < ((k + 3) * n + 1) // 2
            )
        )
    ):
        return False
    # edge bound from :prf:ref:`thm-k-vertex-redundant-edge-bound-general`
    if (
        #
        n >= dim * dim + dim + k + 1
        and m < dim * n - math.comb(dim + 1, 2) + k * dim + max(0, k - (dim + 1) // 2)
    ):
        return False
    # edge bound from :prf:ref:`thm-vertex-redundant-edge-bound-general2`
    if k >= dim + 1 and n >= dim + k + 1 and m < ((dim + k) * n + 1) // 2:
        return False

    # in all other cases check by definition
    # and :prf:ref:`thm-redundant-vertex-subset`
    if graph.number_of_nodes() < k + 2:
        if not generic_rigidity.is_rigid(
            graph, dim=dim, algorithm=algorithm, prob=prob
        ):
            return False
        for cur_k in range(1, k):
            if not is_k_vertex_redundantly_rigid(
                graph, cur_k, dim=dim, algorithm=algorithm, prob=prob
            ):
                return False
    G = deepcopy(graph)
    for vertex_set in combinations(graph.nodes, k):
        adj = [[v, list(G.neighbors(v))] for v in vertex_set]
        G.remove_nodes_from(vertex_set)
        if not generic_rigidity.is_rigid(G, dim=dim, algorithm=algorithm, prob=prob):
            return False
        # add vertices and edges back
        G.add_nodes_from(vertex_set)
        for v, neighbors in adj:
            for neighbor in neighbors:
                G.add_edge(v, neighbor)
    return True


def is_vertex_redundantly_rigid(
    graph: nx.Graph, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
) -> bool:
    """
    Return whether the graph is vertex redundantly ``dim``-rigid.

    See :meth:`.is_k_vertex_redundantly_rigid` (using ``k=1``) for details.

    Definitions
    -----------
    :prf:ref:`vertex redundantly dim-rigid <def-redundantly-rigid-graph>`
    """
    return is_k_vertex_redundantly_rigid(
        graph, 1, dim=dim, algorithm=algorithm, prob=prob
    )


def is_min_k_vertex_redundantly_rigid(
    graph: nx.Graph,
    k: int,
    dim: int = 2,
    algorithm: str = "default",
    prob: float = 0.0001,
) -> bool:
    """
    Return whether the graph is minimally ``k``-vertex redundantly ``dim``-rigid.

    Preliminary checks from
    :prf:ref:`thm-minimal-k-vertex-redundant-upper-edge-bound`,
    :prf:ref:`thm-minimal-k-vertex-redundant-upper-edge-bound-dim1`
    are used.

    Definitions
    -----------
    :prf:ref:`Minimal k-vertex redundant dim-rigidity <def-redundantly-rigid-graph>`

    Parameters
    ----------
    k:
        Level of redundancy.
    dim:
        Dimension.
    algorithm:
        See :meth:`.is_rigid` for the possible algorithms used
        for checking rigidity in this method.
    prob:
        A bound on the probability for false negatives of the rigidity testing
        when ``algorithm="randomized"``.

        *Warning:* this is not the probability of wrong results in this method,
        but is just passed on to rigidity testing.

    Examples
    --------
    >>> G = Graph([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5],
    ...            [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]])
    >>> G.is_min_k_vertex_redundantly_rigid(1, 2)
    True
    >>> G.is_min_k_vertex_redundantly_rigid(2, 2)
    False
    >>> G = Graph([[0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3],
    ...            [1, 4], [1, 5], [2, 4], [2, 5], [3, 4], [3, 5]])
    >>> G.is_k_vertex_redundantly_rigid(1, 2)
    True
    >>> G.is_min_k_vertex_redundantly_rigid(1, 2)
    False
    """

    _input_check.dimension(dim)
    _input_check.integrality_and_range(k, "k", min_val=0)
    _graph_input_check.no_loop(graph)

    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    # edge bound from :prf:ref:`thm-minimal-k-vertex-redundant-upper-edge-bound`
    if m > (dim + k) * n - math.comb(dim + k + 1, 2):
        return False
    # edge bound from :prf:ref:`thm-minimal-k-vertex-redundant-upper-edge-bound-dim1`
    if dim == 1:
        if n >= 3 * (k + 1) - 1 and m > (k + 1) * n - (k + 1) * (k + 1):
            return False

    if not is_k_vertex_redundantly_rigid(
        graph, k, dim=dim, algorithm=algorithm, prob=prob
    ):
        return False

    # for the following we need to know that the graph is k-vertex-redundantly rigid
    if (
        dim == 2
        and (
            # edge bound from :prf:ref:`thm-1-vertex-redundant-edge-bound-dim2`
            (k == 1 and n >= 5 and m == 2 * n - 1)
            or
            # edge bound from :prf:ref:`thm-2-vertex-redundant-edge-bound-dim2`
            (k == 2 and n >= 6 and m == 2 * n + 2)
            or
            # edge bound from :prf:ref:`thm-k-vertex-redundant-edge-bound-dim2`
            (k >= 3 and n >= 6 * (k + 1) + 23 and m == ((k + 2) * n + 1) // 2)
        )
    ) or (
        dim == 3
        and (
            # edge bound from :prf:ref:`thm-3-vertex-redundant-edge-bound-dim3`
            (k == 3 and n >= 15 and m == 3 * n + 5)
            or
            # edge bound from :prf:ref:`thm-k-vertex-redundant-edge-bound-dim3`
            (
                k >= 4
                and n >= 12 * (k + 1) + 10
                and n % 2 == 0
                and m == ((k + 3) * n + 1) // 2
            )
        )
    ):
        return True
    # edge bound from :prf:ref:`thm-k-vertex-redundant-edge-bound-general`
    if (
        #
        n >= dim * dim + dim + k + 1
        and m == dim * n - math.comb(dim + 1, 2) + k * dim + max(0, k - (dim + 1) // 2)
    ):
        return True
    # edge bound from :prf:ref:`thm-vertex-redundant-edge-bound-general2`
    if k >= dim + 1 and n >= dim + k + 1 and m == ((dim + k) * n + 1) // 2:
        return True

    # in all other cases check by definition
    G = deepcopy(graph)
    for e in graph.edges:
        G.remove_edge(*e)
        if is_k_vertex_redundantly_rigid(G, k, dim=dim, algorithm=algorithm, prob=prob):
            return False
        G.add_edge(*e)
    return True


def is_min_vertex_redundantly_rigid(
    graph: nx.Graph, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
) -> bool:
    """
    Return whether the graph is minimally vertex redundantly ``dim``-rigid.

    See :meth:`.is_min_k_vertex_redundantly_rigid` (using ``k=1``) for details.

    Definitions
    -----------
    :prf:ref:`Minimal vertex redundant dim-rigidity <def-min-redundantly-rigid-graph>`
    """
    return is_min_k_vertex_redundantly_rigid(
        graph, 1, dim=dim, algorithm=algorithm, prob=prob
    )


def is_k_redundantly_rigid(
    graph: nx.Graph,
    k: int,
    dim: int = 2,
    algorithm: str = "default",
    prob: float = 0.0001,
) -> bool:
    """
    Return whether the graph is ``k``-redundantly ``dim``-rigid.

    Preliminary checks from
    :prf:ref:`thm-globally-mindeg6-dim2`,
    :prf:ref:`thm-globally-redundant-3connected`,
    :prf:ref:`thm-k-edge-redundant-edge-bound-dim2`,
    :prf:ref:`thm-2-edge-redundant-edge-bound-dim2`,
    :prf:ref:`thm-2-edge-redundant-edge-bound-dim3` and
    :prf:ref:`thm-k-edge-redundant-edge-bound-dim3`
    are used.

    Definitions
    -----------
    :prf:ref:`k-redundant dim-rigidity <def-redundantly-rigid-graph>`

    Parameters
    ----------
    k:
        Level of redundancy.
    dim:
        Dimension.
    algorithm:
        See :meth:`.is_rigid` for the possible algorithms used
        for checking rigidity in this method.
    prob:
        A bound on the probability for false negatives of the rigidity testing
        when ``algorithm="randomized"``.

        *Warning:* this is not the probability of wrong results in this method,
        but is just passed on to rigidity testing.

    Examples
    --------
    >>> G = Graph([[0, 1], [0, 2], [0, 3], [0, 5], [1, 2],
    ...            [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]])
    >>> G.is_k_redundantly_rigid(1, 2)
    True
    >>> G = Graph([[0, 3], [0, 4], [1, 2], [1, 3], [1, 4],
    ...            [2, 3], [2, 4], [3, 4]])
    >>> G.is_k_redundantly_rigid(1, 2)
    False
    >>> G = Graph([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2],
    ...            [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
    >>> G.is_k_redundantly_rigid(2, 2)
    True

    Suggested Improvements
    ----------------------
    Improve with pebble games.
    """
    _input_check.dimension(dim)
    _input_check.integrality_and_range(k, "k", min_val=0)
    _graph_input_check.no_loop(graph)

    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    if m < dim * n - math.comb(dim + 1, 2) + k:
        return False
    if graph_general.min_degree(graph) < dim + k:
        return False
    if dim == 1:
        return nx.edge_connectivity(graph) >= k + 1
    # edge bounds
    if (
        dim == 2
        and (
            # basic edge bound
            (k == 1 and m < 2 * n - 2)
            or
            # edge bound from :prf:ref:`thm-2-edge-redundant-edge-bound-dim2`
            (k == 2 and n >= 5 and m < 2 * n)
            or
            # edge bound from :prf:ref:`thm-k-edge-redundant-edge-bound-dim2`
            (k >= 3 and n >= 6 * (k + 1) + 23 and m < ((k + 2) * n + 1) // 2)
        )
    ) or (
        dim == 3
        and (
            # edge bound from :prf:ref:`thm-2-edge-redundant-edge-bound-dim3`
            (k == 2 and n >= 14 and m < 3 * n - 4)
            or
            # edge bound from :prf:ref:`thm-k-edge-redundant-edge-bound-dim3`
            (
                k >= 4
                and n >= 12 * (k + 1) + 10
                and n % 2 == 0
                and m < ((k + 3) * n + 1) // 2
            )
        )
    ):
        return False
    # use global rigidity property of :prf:ref:`thm-globally-redundant-3connected`
    # and :prf:ref:`thm-globally-mindeg6-dim2`
    if dim == 2 and k == 1 and nx.node_connectivity(graph) >= 6:
        return True

    # in all other cases check by definition
    # and :prf:ref:`thm-redundant-edge-subset`
    G = deepcopy(graph)
    for edge_set in combinations(graph.edges, k):
        G.remove_edges_from(edge_set)
        if not generic_rigidity.is_rigid(G, dim=dim, algorithm=algorithm, prob=prob):
            return False
        G.add_edges_from(edge_set)
    return True


def is_redundantly_rigid(
    graph: nx.Graph, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
) -> bool:
    """
    Return whether the graph is redundantly ``dim``-rigid.

    See :meth:`.is_k_redundantly_rigid` (using ``k=1``) for details.

    Definitions
    -----------
    :prf:ref:`Redundant dim-rigidity<def-redundantly-rigid-graph>`
    """
    return is_k_redundantly_rigid(graph, 1, dim=dim, algorithm=algorithm, prob=prob)


def is_min_k_redundantly_rigid(
    graph: nx.Graph,
    k: int,
    dim: int = 2,
    algorithm: str = "default",
    prob: float = 0.0001,
) -> bool:
    """
    Return whether the graph is minimally ``k``-redundantly ``dim``-rigid.

    Preliminary checks from
    :prf:ref:`thm-minimal-1-edge-redundant-upper-edge-bound-dim2`
    are used.

    Definitions
    -----------
    :prf:ref:`Minimal k-redundant dim-rigidity <def-redundantly-rigid-graph>`

    Parameters
    ----------
    k:
        Level of redundancy.
    dim:
        Dimension.
    algorithm:
        See :meth:`.is_rigid` for the possible algorithms used
        for checking rigidity in this method.
    prob:
        A bound on the probability for false negatives of the rigidity testing
        when ``algorithm="randomized"``.

        *Warning:* this is not the probability of wrong results in this method,
        but is just passed on to rigidity testing.

    Examples
    --------
    >>> G = Graph([[0, 2], [0, 3], [0, 4], [1, 2],
    ...            [1, 3], [1, 4], [2, 4], [3, 4]])
    >>> G.is_min_k_redundantly_rigid(1, 2)
    True
    >>> G.is_min_k_redundantly_rigid(2, 2)
    False
    >>> G = Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3],
    ...            [1, 4], [2, 3], [2, 4], [3, 4]])
    >>> G.is_k_redundantly_rigid(1, 2)
    True
    >>> G.is_min_k_redundantly_rigid(1, 2)
    False
    """

    _input_check.dimension(dim)
    _input_check.integrality_and_range(k, "k", min_val=0)
    _graph_input_check.no_loop(graph)

    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    # use bound from thm-minimal-1-edge-redundant-upper-edge-bound-dim2
    if dim == 2:
        if k == 1:
            if n >= 7 and m > 3 * n - 9:
                return False

    if not is_k_redundantly_rigid(graph, k, dim=dim, algorithm=algorithm, prob=prob):
        return False

    # for the following we need to know that the graph is k-redundantly rigid
    if (
        dim == 2
        and (
            # basic edge bound
            (k == 1 and m == 2 * n - 2)
            or
            # edge bound from :prf:ref:`thm-2-edge-redundant-edge-bound-dim2`
            (k == 2 and n >= 5 and m == 2 * n)
            or
            # edge bound from :prf:ref:`thm-k-edge-redundant-edge-bound-dim2`
            (k >= 3 and n >= 6 * (k + 1) + 23 and m == ((k + 2) * n + 1) // 2)
        )
    ) or (
        dim == 3
        and (
            # edge bound from :prf:ref:`thm-2-edge-redundant-edge-bound-dim3`
            (k == 2 and n >= 14 and m == 3 * n - 4)
            or
            # edge bound from :prf:ref:`thm-k-edge-redundant-edge-bound-dim3`
            (
                k >= 4
                and n >= 12 * (k + 1) + 10
                and n % 2 == 0
                and m == ((k + 3) * n + 1) // 2
            )
        )
    ):
        return True

    # in all other cases check by definition
    G = deepcopy(graph)
    for e in graph.edges:
        G.remove_edge(*e)
        if is_k_redundantly_rigid(G, k, dim=dim, algorithm=algorithm, prob=prob):
            return False
        G.add_edge(*e)
    return True


def is_min_redundantly_rigid(
    graph: nx.Graph, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
) -> bool:
    """
    Return whether the graph is minimally redundantly ``dim``-rigid.

    See :meth:`.is_min_k_redundantly_rigid` (using ``k=1``) for details.

    Definitions
    -----------
    :prf:ref:`Minimal redundant dim-rigidity <def-min-redundantly-rigid-graph>`
    """
    return is_min_k_redundantly_rigid(graph, 1, dim=dim, algorithm=algorithm, prob=prob)
