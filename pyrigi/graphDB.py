"""
This is a module for providing common types of graphs.
"""

import networkx as nx
from pyrigi.graph import Graph


def Cycle(n):
    """Return the cycle graph on n vertices."""
    return Graph(nx.cycle_graph(n))


def Complete(n):
    """Return the complete graph on n vertices."""
    return Graph(nx.complete_graph(n))


def Path(n):
    """Return the path graph with n vertices."""
    return Graph(nx.path_graph(n))


def Diamond():
    """Return the complete graph on 4 vertices minus an edge."""
    return Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])


def ThreePrism():
    """Return the 3-prism graph."""
    return Graph(
        [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (0, 3), (1, 4), (2, 5)]
    )
