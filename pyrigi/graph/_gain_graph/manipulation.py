from collections.abc import Iterable, Sequence

import networkx as nx

from pyrigi.data_type import Vertex, Gain, GainEdge
from pyrigi.graph._gain_graph.general import _negate_gain, _validate_gain, edge_list


def add_vertex(graph: nx.MultiGraph, vertex: Vertex) -> None:
    """
    Add a single vertex to the gain graph.

    Alias for :meth:`nx.MultiGraph.add_node()`.

    Parameters
    ----------
    graph:
    vertex:
        The vertex to be added.
    """
    graph.add_node(vertex)


def add_vertices(graph: nx.MultiGraph, vertices: Sequence[Vertex]) -> None:
    """
    Add multiple vertices to the gain graph.

    Alias for :meth:`nx.MultiGraph.add_nodes_from()`.

    Parameters
    ----------
    graph:
    vertices:
        A sequence of vertices to be added.
    """
    graph.add_nodes_from(vertices)


def delete_vertex(graph: nx.MultiGraph, vertex: Vertex) -> None:
    """
    Delete a single vertex from the gain graph.

    Alias for :meth:`nx.MultiGraph.remove_node()`.

    Parameters
    ----------
    vertex:
        The vertex to be removed.
    """
    graph.remove_node(vertex)


def delete_vertices(graph: nx.MultiGraph, vertices: Sequence[Vertex]) -> None:
    """
    Delete multiple vertices from the gain graph.

    Alias for :meth:`nx.MultiGraph.remove_nodes_from()`.

    Parameters
    ----------
    graph:
    vertices:
        A sequence of vertices to be removed.
    """
    graph.remove_nodes_from(vertices)


def delete_edge(graph: nx.MultiGraph, edge: tuple) -> None:
    """
    Delete an edge.

    Parameter
    ---------
    graph:
    edge:
        This can be either an endpoint pair ``(u, v)`` or a gain
        edge ``(u, v, gain)``. For a pair, one such edge is deleted.
        For a triple, the first matching gain edge is removed unless
        a key is specified.
    """
    if not isinstance(edge, (tuple, list)):
        raise TypeError("An edge must be a pair or a gain-edge triple.")

    if len(edge) == 2:
        graph.remove_edge(edge[0], edge[1])
        return

    if len(edge) != 3:
        raise TypeError("An edge must be a pair or a gain-edge triple.")

    u, v, gain = edge
    target_gain = gain
    edge_data = graph.get_edge_data(u, v, default={})

    for edge_key, data in edge_data.items():
        if data["tail"] == u and data["head"] == v:
            candidate_gain = data["gain"]
        else:
            candidate_gain = _negate_gain(data["gain"])

        if u == v:
            matches = min(candidate_gain, _negate_gain(candidate_gain)) == min(
                target_gain, _negate_gain(target_gain)
            )
        else:
            matches = candidate_gain == target_gain

        if matches:
            graph.remove_edge(u, v, key=edge_key)
            return

    raise ValueError(f"Gain edge {edge!r} is not contained in the graph.")


def delete_edges(graph: nx.MultiGraph, edges: Iterable[tuple]) -> None:
    """
    Delete multiple gain edges.

    Calls :meth:`delete_edge` for each edge in ``edges``.

    Parameters
    ----------
    graph:
    edges:
        An list of gain edges to be removed.
    """
    for edge in edges:
        delete_edge(graph, edge)


def delete_loops(graph: nx.MultiGraph) -> None:
    """
    Remove all loops.

    Calls :meth:`nx.MultiGraph.remove_edge` for each loop in the graph.

    Parameters
    ----------
    graph:
    """
    for u, v in list(graph.edges()):
        if u == v:
            graph.remove_edge(u, v)


def add_edge(
    graph: nx.MultiGraph, u_for_edge: Vertex, v_for_edge: Vertex, gain: Gain = None
):
    r"""
    Add an edge with a specified gain.

    If the edge is already contained in the GainGraph, nothing happens.

    Parameters
    ----------
    graph:
    u_for_edge, v_for_edge:
        The endpoints, in the chosen orientation.
    gain:
        The gain vector in :math:`\mathbb{Z}^d`.
    """
    gain = _validate_gain(graph, gain)
    edge_data = graph.get_edge_data(u_for_edge, v_for_edge, default={})

    if (u_for_edge, v_for_edge, gain) in edge_list(graph) or (
        v_for_edge,
        u_for_edge,
        _negate_gain(gain),
    ) in edge_list(graph):
        return

    edge_data = dict()
    edge_data.update(key=tuple(gain), gain=gain, tail=u_for_edge, head=v_for_edge)

    nx.MultiGraph.add_edge(graph, u_for_edge, v_for_edge, **edge_data)


def add_edges(graph: nx.MultiGraph, edges: Iterable[GainEdge]) -> None:
    """
    Add multiple edges.

    Accepts edges triples ``(u, v, gain)``.

    Parameters
    ----------
    graph:
    edges:
        An iterable of gain edges.
    """
    for edge in edges:
        if not isinstance(edge, (tuple, list)):
            raise nx.NetworkXError(f"Edge {edge!r} must be a tuple or list.")

        if len(edge) == 3:
            u, v, gain = edge
        else:
            raise nx.NetworkXError(f"Edge {edge!r} must have 3 elements.")

        add_edge(graph, u, v, gain)
