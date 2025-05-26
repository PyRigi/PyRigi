"""
This module provides constructions of graphs.
"""

from copy import deepcopy

import networkx as nx

import pyrigi.graph._utils._input_check as _graph_input_check
from pyrigi.data_type import Edge, Vertex


def cone(graph: nx.Graph, inplace: bool = False, vertex: Vertex = None) -> nx.Graph:
    """
    Return a coned version of the graph.

    Definitions
    -----------
    :prf:ref:`Cone graph <def-cone-graph>`

    Parameters
    ----------
    inplace:
        If ``True``, the graph is modified,
        otherwise a new modified graph is created,
        while the original graph remains unchanged (default).
    vertex:
        It is possible to give the added cone vertex a name using
        the keyword ``vertex``.

    Examples
    --------
    >>> G = Graph([(0,1)]).cone()
    >>> G.is_isomorphic(Graph([(0,1),(1,2),(0,2)]))
    True
    """
    if vertex in graph.nodes:
        raise KeyError(f"Vertex {vertex} is already a vertex of the graph!")
    if vertex is None:
        vertex = graph.number_of_nodes()
        while vertex in graph.nodes:
            vertex += 1

    if inplace:
        graph.add_edges_from([(u, vertex) for u in graph.nodes])
        return graph
    else:
        G = deepcopy(graph)
        G.add_edges_from([(u, vertex) for u in G.nodes])
        return G


def sum_t(graph: nx.Graph, other_graph: nx.Graph, edge: Edge, t: int = 2) -> nx.Graph:
    """
    Return the ``t``-sum with ``other_graph`` along the given edge.

    Definitions
    -----------
    :prf:ref:`t-sum <def-t-sum>`

    Examples
    --------
    >>> G1 = Graph([[1,2],[2,3],[3,1],[3,4]])
    >>> G2 = Graph([[0,1],[1,2],[2,3],[3,1]])
    >>> H = G2.sum_t(G1, [1, 2], 3)
    >>> print(H)
    Graph with vertices [0, 1, 2, 3, 4] and edges [[0, 1], [1, 3], [2, 3], [3, 4]]
    """
    if edge not in graph.edges or edge not in other_graph.edges:
        raise ValueError(f"The edge {edge} is not in the intersection of the graphs!")
    # check if the intersection is a t-complete graph
    if not nx.is_isomorphic(intersection(graph, other_graph), nx.complete_graph(t)):
        raise ValueError(
            f"The intersection of the graphs must be a {t}-complete graph!"
        )
    G = graph + other_graph
    G.remove_edge(edge[0], edge[1])
    return G


def intersection(graph: nx.Graph, other_graph: nx.Graph) -> nx.Graph:
    """
    Return the intersection with ``other_graph``.

    Parameters
    ----------
    other_graph: Graph

    Examples
    --------
    >>> H = Graph([[1,2],[2,3],[3,1],[3,4]])
    >>> G = Graph([[0,1],[1,2],[2,3],[3,1]])
    >>> graph = G.intersection(H)
    >>> print(graph)
    Graph with vertices [1, 2, 3] and edges [[1, 2], [1, 3], [2, 3]]
    >>> G = Graph([[0,1],[0,2],[1,2]])
    >>> G.add_vertex(3)
    >>> H = Graph([[0,1],[1,2],[2,4],[4,0]])
    >>> H.add_vertex(3)
    >>> graph = G.intersection(H)
    >>> print(graph)
    Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [1, 2]]
    """
    G = graph.__class__()
    vertices = [v for v in graph.nodes if v in other_graph.nodes]
    edges = [e for e in graph.edges if e in other_graph.edges]
    G.add_nodes_from(vertices)
    _graph_input_check.edge_format_list(G, edges)
    G.add_edges_from(edges)
    return G
