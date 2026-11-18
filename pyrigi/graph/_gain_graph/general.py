from collections.abc import Iterable

import networkx as nx

from pyrigi.data_type import Vertex, GainEdge, Gain


def _is_Zd_gain_graph_data(
    incoming_graph_data: Iterable[GainEdge] | nx.Graph | None,
) -> bool:
    """
    Return whether the input has the structure of gain-graph data.

    Parameters
    ----------
    incoming_graph_data:
        The input data to be checked. This can either be a list of
        gain edges, a networkx graph, or ``None``.
    """
    if incoming_graph_data is None:
        return True

    if isinstance(incoming_graph_data, nx.Graph):
        edges = list(incoming_graph_data.edges(data=True))
        return (
            all("gain" in data for _, _, data in edges)
            if edges
            else hasattr(incoming_graph_data, "dim")
        )

    return all(
        isinstance(edge, (tuple, list))
        and ((len(edge) == 3 and isinstance(edge[2], (tuple, list))) or len(edge) == 2)
        for edge in incoming_graph_data
    )


def _negate_gain(gain: Gain) -> Gain:
    """
    Return the inverse of a gain.

    Parameters
    ----------
    gain:
        The gain to be negated. This currently must be a tuple of integers.
    """
    if isinstance(gain, tuple):
        return tuple(-x for x in gain)
    else:
        raise TypeError("A gain must be a tuple of integers.")


def vertex_list(graph: nx.MultiGraph) -> list[Vertex]:
    """
    Return the list of vertices in the gain graph.

    Parameters
    ----------
    graph:
    """
    try:
        return sorted(graph.nodes)
    except BaseException:
        return list(graph.nodes)


def edge_list(graph: nx.MultiGraph, as_tuples: bool = True) -> list[tuple]:
    """
    Return the list of gain edges.

    Each edge is returned as ``(u, v, gain)``.

    Parameters
    ----------
    graph:
    as_tuples:
        If ``True``, all edges are returned as tuples instead of lists.
    """
    edges = []

    for _, _, data in graph.edges(data=True):
        edge = (data["tail"], data["head"], data["gain"])
        edges.append(edge)

    try:
        if as_tuples:
            return sorted([tuple(e) for e in edges])
        else:
            return sorted([list(e) for e in edges])
    except BaseException:
        if as_tuples:
            return [tuple(e) for e in edges]
        else:
            return [list(e) for e in edges]


def gains_between(graph: nx.MultiGraph, u: Vertex, v: Vertex) -> list[Gain]:
    """
    Return all gains of edges oriented from ``u`` to ``v``.

    If the orientation in the GainGraph is reversed, return the
    negated gain instead.

    Parameters
    ----------
    graph:
    u, v:
        The endpoints of the edges.
    """
    edge_data = graph.get_edge_data(u, v, default={})
    result = []

    for data in edge_data.values():
        if data["tail"] == u and data["head"] == v:
            result.append(data["gain"])
        else:
            result.append(_negate_gain(data["gain"]))

    return result


def _is_gain_edge(graph: nx.MultiGraph, gain: Gain) -> bool:
    """
    Return whether a gain is a gain edge of the correct format.

    Parameters
    ----------
    graph:
    gain:
        The gain to be checked.

    Notes
    -----
    This currently only checks whether the gain is a tuple of integers and
    whether the length of the gain matches the dimension of the graph or, alternatively,
    whether it can be interpreted as a higher-dimensional gain by padding it with
    trailing zeros.
    """
    return (
        isinstance(gain, tuple | list)
        and all(isinstance(x, int) for x in list(gain))
        and all(
            isinstance(edge[2], tuple | list) and len(edge[2]) >= len(gain)
            for edge in graph.edge_list(graph)
        )
    )


def _validate_gain(graph: nx.MultiGraph, gain: Gain) -> Gain:
    """
    Validate a gain.
    """
    if not _is_gain_edge(graph, gain):
        raise TypeError(f"Gain {gain!r} is not of type {Gain!r}.")
    return gain
