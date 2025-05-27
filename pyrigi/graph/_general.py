"""
This module provides some general graph functionality.
"""

from typing import Sequence

import networkx as nx
from sympy import Matrix

from pyrigi.data_type import Edge, Vertex

from ._utils import _input_check as _graph_input_check


def min_degree(graph: nx.Graph) -> int:
    """
    Return the minimum of the vertex degrees.

    Examples
    --------
    >>> G = Graph([(0,1), (1,2)])
    >>> G.min_degree()
    1
    """
    return min([int(graph.degree(v)) for v in graph.nodes])


def max_degree(graph: nx.Graph) -> int:
    """
    Return the maximum of the vertex degrees.

    Examples
    --------
    >>> G = Graph([(0,1), (1,2)])
    >>> G.max_degree()
    2
    """
    return max([int(graph.degree(v)) for v in graph.nodes])


def degree_sequence(
    graph: nx.Graph, vertex_order: Sequence[Vertex] = None
) -> list[int]:
    """
    Return a list of degrees of the vertices of the graph.

    Parameters
    ----------
    vertex_order:
        By listing vertices in the preferred order, the degree_sequence
        can be computed in a way the user expects. If no vertex order is
        provided, :meth:`~.Graph.vertex_list()` is used.

    Examples
    --------
    >>> G = Graph([(0,1), (1,2)])
    >>> G.degree_sequence()
    [1, 2, 1]
    """
    vertex_order = _graph_input_check.is_vertex_order(graph, vertex_order)
    return [int(graph.degree(v)) for v in vertex_order]


def adjacency_matrix(graph: nx.Graph, vertex_order: Sequence[Vertex] = None) -> Matrix:
    """
    Return the adjacency matrix of the graph.

    Parameters
    ----------
    vertex_order:
        By listing vertices in the preferred order, the adjacency matrix
        can be computed in a way the user expects. If no vertex order is
        provided, :meth:`~.Graph.vertex_list()` is used.

    Examples
    --------
    >>> G = Graph([(0,1), (1,2), (1,3)])
    >>> G.adjacency_matrix()
    Matrix([
    [0, 1, 0, 0],
    [1, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 0]])

    Notes
    -----
    :func:`networkx.linalg.graphmatrix.adjacency_matrix`
    requires ``scipy``. To avoid unnecessary imports, the method is implemented here.
    """
    vertex_order = _graph_input_check.is_vertex_order(graph, vertex_order)

    row_list = [
        [+((v1, v2) in graph.edges) for v2 in vertex_order] for v1 in vertex_order
    ]

    return Matrix(row_list)


def edge_list(graph: nx.Graph, as_tuples: bool = False) -> list[Edge]:
    """
    Return the list of edges.

    The output is sorted if possible,
    otherwise, the internal order is used instead.

    Parameters
    ----------
    as_tuples:
        If ``True``, all edges are returned as tuples instead of lists.

    Examples
    --------
    >>> G = Graph([[0, 3], [3, 1], [0, 1], [2, 0]])
    >>> G.edge_list()
    [[0, 1], [0, 2], [0, 3], [1, 3]]

    >>> G = Graph.from_vertices(['a', 'c', 'b'])
    >>> G.edge_list()
    []

    >>> G = Graph([['c', 'b'], ['b', 'a']])
    >>> G.edge_list()
    [['a', 'b'], ['b', 'c']]

    >>> G = Graph([['c', 1], [2, 'a']]) # incomparable vertices
    >>> G.edge_list()
    [('c', 1), (2, 'a')]
    """
    try:
        if as_tuples:
            return sorted([tuple(sorted(e)) for e in graph.edges])
        else:
            return sorted([sorted(e) for e in graph.edges])
    except BaseException:
        if as_tuples:
            return [tuple(e) for e in graph.edges]
        else:
            return list(graph.edges)


def vertex_list(graph: nx.Graph) -> list[Vertex]:
    """
    Return the list of vertices.

    The output is sorted if possible,
    otherwise, the internal order is used instead.

    Examples
    --------
    >>> G = Graph.from_vertices_and_edges([2, 0, 3, 1], [[0, 1], [0, 2], [0, 3]])
    >>> G.vertex_list()
    [0, 1, 2, 3]

    >>> G = Graph.from_vertices(['c', 'a', 'b'])
    >>> G.vertex_list()
    ['a', 'b', 'c']

    >>> G = Graph.from_vertices(['b', 1, 'a']) # incomparable vertices
    >>> G.vertex_list()
    ['b', 1, 'a']
    """
    try:
        return sorted(graph.nodes)
    except BaseException:
        return list(graph.nodes)
