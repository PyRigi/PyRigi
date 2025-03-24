from typing import Callable, Iterable, Optional, TypeVar
import networkx as nx

from pyrigi.data_type import Vertex, SeparatingCut

T = TypeVar("T")


def _to_vertices(vertices: Iterable[Vertex] | SeparatingCut) -> set[Vertex]:
    """
    Convert multiple input formats into a graph separator.
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
    Check if the given set of vertices is stable in the given graph
    and if not, find a pair of vertices in the set that are neighboring.

    Definitions
    -----------
    :prf:ref:`Stable set <def-stable-set>`

    Parameters
    ----------
    graph:
        the graph to check
    vertices:
        the set of vertices

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
        for u in graph.neighbors(v):
            if u in vertices:
                return v, u
    return None


def is_stable_set(
    graph: nx.Graph,
    vertices: Iterable[Vertex] | SeparatingCut,
) -> bool:
    """
    Check if the given set of vertices is stable in the given graph.

    Definitions
    -----------
    :prf:ref:`Stable set <def-stable-set>`

    Parameters
    ----------
    graph:
        the graph to check
    vertices:
        the vertices to check

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


def _revertable_set_removal(
    graph: nx.Graph,
    vertices: set[Vertex],
    opt: Callable[[nx.Graph], T],
) -> T:
    """
    Remove given vertices from the graph, perform operation,
    return vertices along with edges back.

    Parameters
    ----------
    graph:
        The graph from which vertices will be removed
    vertices:
        Vertex set to remove
    opt:
        Operation to perform on a graph with vertices removed

    Note
    ----
        Edge and vertex data are not preserved, make a copy yourself.
    """
    copy = nx.is_frozen(graph)

    if copy:
        graph = nx.Graph(graph)
        neighbors = []
    else:
        neighbors = [(u, v) for u in vertices for v in graph.neighbors(u)]

    graph.remove_nodes_from(vertices)

    res = opt(graph)

    if not copy:
        graph.add_edges_from(neighbors)

    return res


def is_separating_set(
    graph: nx.Graph,
    vertices: Iterable[Vertex] | SeparatingCut,
) -> bool:
    """
    Check if the given set of vertices is a separator in the given graph.

    Definitions
    -----------
    :prf:ref:`Separating set <def-separating-set>`

    Parameters
    ----------
    graph:
        the graph to check
    vertices:
        the vertices to check

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

    return _revertable_set_removal(graph, vertices, lambda g: not nx.is_connected(g))


def is_separating_set_dividing(
    graph: nx.Graph,
    vertices: Iterable[Vertex] | SeparatingCut,
    u: Vertex,
    v: Vertex,
) -> bool:
    """
    Check if the given cut separates vertices u and v.

    Definitions
    -----------
    :prf:ref:`Separating set <def-separating-set>`

    Parameters
    ----------
    graph:
        the graph to check
    vertices:
        the vertices to check
    u:
        the first vertex
    v:
        the second vertex

    Raises
    ------
    ValueError:
        If either of the vertices is contained in the cutset

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

    def check_graph(g: nx.Graph) -> bool:
        components = nx.connected_components(g)
        for c in components:
            if u in c and v in c:
                return False
        return True

    return _revertable_set_removal(graph, vertices, check_graph)


def is_stable_cutset(
    graph: nx.Graph,
    vertices: Iterable[Vertex] | SeparatingCut,
) -> bool:
    """
    Check if the given set of vertices is a stable cut in the given graph.

    Definitions
    -----------
    :prf:ref:`Stable cutset <def-stable-cutset>`

    Parameters
    ----------
    graph:
        the graph to check
    vertices:
        the cutset of vertices

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> H = graphs.Cycle(5)
    >>> H.is_stable_cutset([1,3])
    True
    >>> H.is_stable_cutset([1,2])
    False

    Note
    ----
        See :meth:`~pyrigi.graph.Graph.is_stable_set` and
        :meth:`~pyrigi.graph.Graph.is_separator`.
    """
    return is_stable_set(graph, vertices) and is_separating_set(graph, vertices)


def is_stable_cutset_dividing(
    graph: nx.Graph,
    vertices: Iterable[Vertex] | SeparatingCut,
    u: Vertex,
    v: Vertex,
) -> bool:
    """
    Checks if the given set of vertices is a stable cut in the given graph
    separating ``u`` and ``v``.

    Definitions
    -----------
    :prf:ref:`Stable cutset <def-stable-cutset>`

    Parameters
    ----------
    graph:
        the graph to check
    vertices:
        the vertices to check

    Raises
    ------
    ValueError:
        If either of the vertices is contained in the cutset

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> H = graphs.Cycle(5)
    >>> H.is_stable_cutset_dividing([1,3], 0, 2)
    True
    >>> H.is_stable_cutset_dividing([2,4], 0, 1)
    False

    Note
    ----
        See :meth:`~pyrigi.graph.Graph.is_stable_set`
        and :meth:`~pyrigi.graph.Graph.is_stable_cutset_dividing`.
    """
    return is_stable_set(graph, vertices) and is_separating_set_dividing(
        graph, vertices, u, v
    )
