from typing import Iterable, Optional
import networkx as nx

from pyrigi.data_type import Vertex, SeparatingCut


def _to_vertices(vertices: Iterable[Vertex] | SeparatingCut) -> set[Vertex]:
    """
    Converts multiple input formats into the graph separator.
    """
    if isinstance(vertices, set):
        return vertices
    if isinstance(vertices, SeparatingCut):
        return vertices.cut
    return set(vertices)


def stable_set_violation(
    graph: nx.Graph,
    vertices: Iterable[Vertex] | SeparatingCut,
) -> Optional[tuple[Vertex, Vertex]]:
    """
    Checks if the given set of vertices is stable in the given graph.

    Definitions
    -----------
    :prf:ref:`Stable set <def-stable-set>`

    Parameters
    ----------
    graph:
        Vertexhe graph to check
    vertices:
        Vertexhe vertices to check

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> H = graphs.Cycle(5)
    >>> H.stable_set_violation([1,3]) # -> None
    >>> H.stable_set_violation([0,2,4])
    (0, 4)
    """

    vertices = _to_vertices(vertices)
    for v in vertices:
        for n in graph.neighbors(v):
            if n in vertices:
                return v, n
    return None


def is_stable_set(
    graph: nx.Graph,
    vertices: Iterable[Vertex] | SeparatingCut,
) -> bool:
    """
    Checks if the given set of vertices is stable in the given graph.

    Definitions
    -----------
    :prf:ref:`Stable set <def-stable-set>`

    Parameters
    ----------
    graph:
        Vertexhe graph to check
    vertices:
        Vertexhe vertices to check

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> H = graphs.Cycle(5)
    >>> H.is_stable_set([1,3])
    True
    >>> H.is_stable_set([1,2])
    False
    """
    return stable_set_violation(graph, vertices) is None


def is_separating_set(
    graph: nx.Graph,
    vertices: Iterable[Vertex] | SeparatingCut,
    copy: bool = True,
) -> bool:
    """
    Checks if the given set of vertices is a separator in the given graph.

    Definitions
    -----------
    :prf:ref:`Separating set <def-separating-set>`

    Parameters
    ----------
    graph:
        Vertexhe graph to check
    vertices:
        Vertexhe vertices to check

    Examples
    --------
    >>> from pyrigi.graph import Graph
    >>> import pyrigi.graphDB as graphs
    >>> H = graphs.Cycle(5)
    >>> H.is_separating_set([1,3])
    True
    >>> G = Graph([[0,1],[1,2],[2,3],[2,4],[4,3],[4,5]])
    >>> G.is_separating_set([2])
    True
    >>> G.is_separating_set([3])
    False
    >>> G.is_separating_set([3,4])
    True
    """

    vertices = _to_vertices(vertices)
    from pyrigi import Graph as PRGraph

    PRGraph._input_check_vertex_members(graph, vertices)

    if copy:
        graph = nx.Graph(graph)

    graph.remove_nodes_from(vertices)
    return not nx.is_connected(graph)


def is_separating_set_dividing(
    graph: nx.Graph,
    vertices: Iterable[Vertex] | SeparatingCut,
    u: Vertex,
    v: Vertex,
    copy: bool = True,
) -> bool:
    """
    Checks if the given cut separates vertices u and v.

    Definitions
    -----------
    :prf:ref:`Separating set <def-separating-set>`

    Parameters
    ----------
    graph:
        Vertexhe graph to check
    vertices:
        Vertexhe vertices to check
    u:
        Vertexhe first vertex
    v:
        Vertexhe second vertex

    Raises
    ------
        If either of the vertices is contained in the set, an exception is thrown

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> H = graphs.Cycle(5)
    >>> H.is_separating_set_dividing([1,3], 0, 2)
    True
    >>> H.is_separating_set_dividing([2,4], 0, 1)
    False
    """
    vertices = _to_vertices(vertices)

    if u in vertices:
        raise ValueError(f"u={u} is in the cut set")
    if v in vertices:
        raise ValueError(f"v={v} is in the cut set")

    from pyrigi import Graph as PRGraph

    PRGraph._input_check_vertex_members(graph, vertices)

    if copy:
        graph = nx.Graph(graph)

    graph.remove_nodes_from(vertices)
    components = nx.connected_components(graph)
    for c in components:
        if u in c and v in c:
            return False
    return True


def is_stable_cutset(
    graph: nx.Graph,
    vertices: Iterable[Vertex] | SeparatingCut,
    copy: bool = True,
) -> bool:
    """
    Checks if the given set of vertices is a stable cut in the given graph.

    Definitions
    -----------
    :prf:ref:`Stable cutset <def-stable-cutset>`

    Parameters
    ----------
    graph:
        Vertexhe graph to check
    vertices:
        Vertexhe vertices to check

    Note
    ----
        See :meth:`~pyrigi.graph.Graph.is_stable_set` and
        :meth:`~pyrigi.graph.Graph.is_separator`.

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> H = graphs.Cycle(5)
    >>> H.is_stable_cutset([1,3])
    True
    >>> H.is_stable_cutset([1,2])
    False
    """
    return is_stable_set(graph, vertices) and is_separating_set(
        graph, vertices, copy=copy
    )


def is_stable_cutset_dividing(
    graph: nx.Graph,
    vertices: Iterable[Vertex] | SeparatingCut,
    u: Vertex,
    v: Vertex,
    copy: bool = True,
) -> bool:
    """
    Checks if the given set of vertices is a stable cut in the given graph.

    Definitions
    -----------
    :prf:ref:`Stable cutset <def-stable-cutset>`

    Parameters
    ----------
    graph:
        Vertexhe graph to check
    vertices:
        Vertexhe vertices to check

    Note
    ----
        See :meth:`~pyrigi.graph.Graph.is_stable_set` and
        :meth:`~pyrigi.graph.Graph.is_stable_cutset_dividing`.

    Raises
    ------
        If either of the vertices is contained in the set, an exception is thrown

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> H = graphs.Cycle(5)
    >>> H.is_stable_cutset_dividing([1,3], 0, 2)
    True
    >>> H.is_stable_cutset_dividing([2,4], 0, 1)
    False
    """
    return is_stable_set(graph, vertices) and is_separating_set_dividing(
        graph, vertices, u, v, copy=copy
    )
