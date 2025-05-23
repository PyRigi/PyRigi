"""
This module provides functionality related to apex graphs.
"""

from copy import deepcopy
from itertools import combinations

import networkx as nx

import pyrigi._utils._input_check as _input_check


def is_vertex_apex(graph: nx.Graph) -> bool:
    """
    Return whether the graph is vertex apex.

    Alias for :meth:`~.Graph.is_k_vertex_apex` with ``k=1``.

    Definitions
    -----------
    :prf:ref:`Vertex apex graph <def-apex-graph>`
    """
    return is_k_vertex_apex(graph, 1)


def is_k_vertex_apex(graph: nx.Graph, k: int) -> bool:
    """
    Return whether the graph is ``k``-vertex apex.

    Definitions
    -----------
    :prf:ref:`k-vertex apex graph <def-apex-graph>`

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.Complete(5)
    >>> G.is_k_vertex_apex(1)
    True
    """
    _input_check.integrality_and_range(
        k, "k", min_val=0, max_val=graph.number_of_nodes()
    )
    _G = deepcopy(graph)
    for vertex_list in combinations(graph.nodes, k):
        incident_edges = list(_G.edges(vertex_list))
        _G.remove_nodes_from(vertex_list)
        if nx.is_planar(_G):
            return True
        _G.add_edges_from(incident_edges)
    return False


def is_edge_apex(graph: nx.Graph) -> bool:
    """
    Return whether the graph is edge apex.

    Alias for :meth:`~.Graph.is_k_edge_apex` with ``k=1``

    Definitions
    -----------
    :prf:ref:`Edge apex graph <def-apex-graph>`
    """
    return is_k_edge_apex(graph, 1)


def is_k_edge_apex(graph: nx.Graph, k: int) -> bool:
    """
    Return whether the graph is ``k``-edge apex.

    Definitions
    -----------
    :prf:ref:`k-edge apex graph <def-apex-graph>`

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.Complete(5)
    >>> G.is_k_edge_apex(1)
    True
    """
    _input_check.integrality_and_range(
        k, "k", min_val=0, max_val=graph.number_of_edges()
    )
    _G = deepcopy(graph)
    for edge_list in combinations(graph.edges, k):
        _G.remove_edges_from(edge_list)
        if nx.is_planar(_G):
            return True
        _G.add_edges_from(edge_list)
    return False


def is_critically_vertex_apex(graph: nx.Graph) -> bool:
    """
    Return whether the graph is critically vertex apex.

    Alias for :meth:`~.Graph.is_critically_k_vertex_apex` with ``k=1``.

    Definitions
    -----------
    :prf:ref:`Critically vertex apex graph <def-apex-graph>`
    """
    return is_critically_k_vertex_apex(graph, 1)


def is_critically_k_vertex_apex(graph: nx.Graph, k: int) -> bool:
    """
    Return whether the graph is critically ``k``-vertex apex.

    Definitions
    -----------
    :prf:ref:`Critically k-vertex apex graph <def-apex-graph>`

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.Complete(5)
    >>> G.is_critically_k_vertex_apex(1)
    True
    """
    _input_check.integrality_and_range(
        k, "k", min_val=0, max_val=graph.number_of_nodes()
    )
    _G = deepcopy(graph)
    for vertex_list in combinations(graph.nodes, k):
        incident_edges = list(_G.edges(vertex_list))
        _G.remove_nodes_from(vertex_list)
        if not nx.is_planar(_G):
            return False
        _G.add_edges_from(incident_edges)
    return True


def is_critically_edge_apex(graph: nx.Graph) -> bool:
    """
    Return whether the graph is critically edge apex.

    Alias for :meth:`~.Graph.is_critically_k_edge_apex` with ``k=1``.

    Definitions
    -----------
    :prf:ref:`Critically edge apex graph <def-apex-graph>`
    """
    return is_critically_k_edge_apex(graph, 1)


def is_critically_k_edge_apex(graph: nx.Graph, k: int) -> bool:
    """
    Return whether the graph is critically ``k``-edge apex.

    Definitions
    -----------
    :prf:ref:`Critically k-edge apex graph <def-apex-graph>`

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.Complete(5)
    >>> G.is_critically_k_edge_apex(1)
    True
    """
    _input_check.integrality_and_range(
        k, "k", min_val=0, max_val=graph.number_of_edges()
    )
    _G = deepcopy(graph)
    for edge_list in combinations(graph.edges, k):
        _G.remove_edges_from(edge_list)
        if not nx.is_planar(_G):
            return False
        _G.add_edges_from(edge_list)
    return True
