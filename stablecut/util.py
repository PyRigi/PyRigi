from typing import Iterable, Optional
import networkx as nx

from pyrigi.data_type import Vertex

from stablecut.types import SeparatingCut


def _to_vertices[T](vertices: Iterable[T] | SeparatingCut[T]) -> set[T]:
    """
    Converts multiple input formats into the graph separator.
    """
    if isinstance(vertices, set):
        return vertices
    if isinstance(vertices, SeparatingCut):
        return vertices.cut
    return set(vertices)


def stable_set_violation[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | SeparatingCut[T],
) -> Optional[tuple[T, T]]:
    """
    Checks if the given set of vertices is stable in the given graph.

    Parameters
    ----------
    graph:
        The graph to check
    vertices:
        The vertices to check
    """

    vertices = _to_vertices(vertices)
    for v in vertices:
        for n in graph.neighbors(v):
            if n in vertices:
                return v, n
    return None


def is_stable_set[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | SeparatingCut[T],
) -> bool:
    """
    Checks if the given set of vertices is stable in the given graph.

    Parameters
    ----------
    graph:
        The graph to check
    vertices:
        The vertices to check
    """
    return stable_set_violation(graph, vertices) is None


def is_separator[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | SeparatingCut[T],
    copy: bool = True,
) -> bool:
    """
    Checks if the given set of vertices is a separator in the given graph.

    Parameters
    ----------
    graph:
        The graph to check
    vertices:
        The vertices to check
    """

    vertices = _to_vertices(vertices)

    if copy:
        graph = nx.Graph(graph)

    graph.remove_nodes_from(vertices)
    return not nx.is_connected(graph)


def is_separator_dividing[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | SeparatingCut[T],
    u: T,
    v: T,
    copy: bool = True,
) -> bool:
    """
    Checks if the given cut separates vertices u and v.

    Parameters
    ----------
    graph:
        The graph to check
    vertices:
        The vertices to check
    u:
        The first vertex
    v:
        The second vertex

    Raises
    ------
    If either of the vertices is contained in the set, exception is thrown
    """
    vertices = _to_vertices(vertices)

    if u in vertices:
        raise ValueError(f"u={u} is in the cut set")
    if v in vertices:
        raise ValueError(f"v={v} is in the cut set")

    if copy:
        graph = nx.Graph(graph)

    graph.remove_nodes_from(vertices)
    components = nx.connected_components(graph)
    for c in components:
        if u in c and v in c:
            return False
    return True


def is_stable_cutset[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | SeparatingCut[T],
    copy: bool = True,
) -> bool:
    """
    Checks if the given set of vertices is a stable cut in the given graph.

    Parameters
    ----------
    graph:
        The graph to check
    vertices:
        The vertices to check
    """
    return is_stable_set(graph, vertices) and is_separator(graph, vertices, copy=copy)


def is_stable_cutset_dividing[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | SeparatingCut[T],
    u: T,
    v: T,
    copy: bool = True,
) -> bool:
    """
    Checks if the given set of vertices is a stable cut in the given graph.

    Parameters
    ----------
    graph:
        The graph to check
    vertices:
        The vertices to check
    """
    return is_stable_set(graph, vertices) and is_separator_dividing(
        graph, vertices, u, v, copy=copy
    )
