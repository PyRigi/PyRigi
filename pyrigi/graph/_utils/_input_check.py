"""
The following functions can be used for input checks of :class:`.Graph`.
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx

from pyrigi.data_type import Edge, Sequence, Vertex
from pyrigi.exception import LoopError
from pyrigi.graph import _general as graph_general


def no_loop(graph: nx.Graph) -> None:
    """
    Check whether the graph has loops and raise an error in case.
    """
    if nx.number_of_selfloops(graph) > 0:
        raise LoopError()


def vertex_members(
    graph: nx.Graph, to_check: Iterable[Vertex] | Vertex, name: str = ""
) -> None:
    """
    Check whether the elements of a list are indeed vertices of the graph and
    raise error otherwise.

    Parameters
    ----------
    to_check:
        A vertex or ``Iterable`` of vertices for which the containment in the graph
        is checked.
    name:
        A name of the ``Iterable`` ``to_check`` can be picked.
    """
    if not isinstance(to_check, Iterable):
        if not graph.has_node(to_check):
            raise ValueError(f"The element {to_check} is not a vertex of the graph!")
    else:
        for vertex in to_check:
            if not graph.has_node(vertex):
                raise ValueError(
                    f"The element {vertex} from "
                    + name
                    + f" {to_check} is not a vertex of the graph!"
                )


def edge_format(graph: nx.Graph, edge: Edge, loopfree: bool = False) -> None:
    """
    Check if an ``edge`` is a pair of (distinct) vertices of the graph and
    raise an error otherwise.

    Parameters
    ----------
    edge:
        Edge for which the containment in the given graph is checked.
    loopfree:
        If ``True``, an error is raised if ``edge`` is a loop.
    """
    if not isinstance(edge, list | tuple) or not len(edge) == 2:
        raise TypeError(f"The input {edge} must be a tuple or list of length 2!")
    vertex_members(graph, edge, "the input pair")
    if loopfree and edge[0] == edge[1]:
        raise LoopError(f"The input {edge} must be two distinct vertices.")


def is_edge(graph: nx.Graph, edge: Edge, vertices: Sequence[Vertex] = None) -> None:
    """
    Check if the given input is an edge of the graph with endvertices in vertices and
    raise an error otherwise.

    Parameters
    ----------
    edge:
        an edge to be checked
    vertices:
        Check if the endvertices of the edge are contained in the list ``vertices``.
        If ``None``, the function considers all vertices of the graph.
    """
    edge_format(graph, edge)
    if vertices and (edge[0] not in vertices or edge[1] not in vertices):
        raise ValueError(
            f"The elements of the edge {edge} are not among vertices {vertices}!"
        )
    if not graph.has_edge(edge[0], edge[1]):
        raise ValueError(f"Edge {edge} is not contained in the graph!")


def is_edge_list(
    graph: nx.Graph, edges: Sequence[Edge], vertices: Sequence[Vertex] = None
) -> None:
    """
    Apply :func:`~.is_edge` to all edges in a list.

    Parameters
    ----------
    edges:
        A list of edges to be checked.
    vertices:
        Check if the endvertices of the edges are contained in the list ``vertices``.
        If ``None`` (default), the function considers all vertices of the graph.
    """
    for edge in edges:
        is_edge(graph, edge, vertices)


def edge_format_list(graph: nx.Graph, edges: Sequence[Edge]) -> None:
    """
    Apply :func:`~.edge_format` to all edges in a list.

    Parameters
    ----------
    edges:
        A list of pairs to be checked.
    """
    for edge in edges:
        edge_format(graph, edge)


def is_vertex_order(
    graph: nx.Graph, vertex_order: Sequence[Vertex], name: str = ""
) -> list[Vertex]:
    """
    Check whether the provided ``vertex_order`` contains the same elements
    as the graph vertex set and raise an error otherwise.

    The ``vertex_order`` is also returned.

    Parameters
    ----------
    vertex_order:
        List of vertices in the preferred order.
        If ``None``, then all vertices are returned
        using :meth:`~Graph.vertex_list`.
    """
    if vertex_order is None:
        return graph_general.vertex_list(graph)
    else:
        if not graph.number_of_nodes() == len(vertex_order) or not set(
            graph_general.vertex_list(graph)
        ) == set(vertex_order):
            raise ValueError(
                "The vertices in `"
                + name
                + "` must be exactly "
                + "the same vertices as in the graph!"
            )
        return list(vertex_order)


def is_edge_order(
    graph: nx.Graph, edge_order: Sequence[Edge], name: str = ""
) -> list[Edge]:
    """
    Check whether the provided ``edge_order`` contains the same elements
    as the graph edge set and raise an error otherwise.

    The ``edge_order`` is also returned.

    Parameters
    ----------
    edge_order:
        List of edges in the preferred order.
        If ``None``, then all edges are returned
        using :meth:`~Graph.edge_list`.
    """
    if edge_order is None:
        return graph_general.edge_list(graph)
    else:
        if not graph.number_of_edges() == len(edge_order) or not all(
            [
                set(e) in [set(e) for e in edge_order]
                for e in graph_general.edge_list(graph)
            ]
        ):
            raise ValueError(
                "The edges in `" + name + "` must be exactly "
                "the same edges as in the graph!"
            )
        return list(edge_order)
