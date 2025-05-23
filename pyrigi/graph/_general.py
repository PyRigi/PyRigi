"""
This module provides some general graph functionality.
"""

import networkx as nx


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
