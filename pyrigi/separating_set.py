"""
This module provides algorithms related to separating sets.

It includes an algorithm for a stable separating set search in a 2-flexible graph
according to Algorithm 1 in :cite:p:`ClinchGaramvölgyiEtAl2024`.
"""

import logging

import networkx as nx
import numpy as np
from typing import (
    Any,
    Callable,
    Collection,
    FrozenSet,
    Optional,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
)

from pyrigi.data_type import Edge, Vertex
import pyrigi._graph_input_check

if TYPE_CHECKING:
    from pyrigi import Graph as PRGraph

T = TypeVar("T")


def is_stable_set(
    graph: nx.Graph,
    vertices: Collection[Vertex],
    certificate: bool = False,
) -> bool | tuple[bool, Optional[Edge]]:
    """
    Return if given ``vertices`` form a stable set.

    If the set is not stable, a pair of adjacent vertices
    is also returned depending on ``certificate``.

    Definitions
    -----------
    :prf:ref:`Stable set <def-stable-set>`

    Parameters
    ----------
    vertices:
        A set of vertices to be checked.
    certificate:
        If ``False``, just a boolean whether the set is stable or not is returned.
        If ``True``, a tuple is returned where the first boolean states
        whether the set is stable and the second item gives a pair of vertices
        contradicting the stable property if applicable (otherwise ``None``).

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> H = graphs.Cycle(5)
    >>> H.is_stable_set([1,3])
    True
    >>> H.is_stable_set([1,3], certificate=False)
    True
    >>> H.is_stable_set([1,3], certificate=True)
    (True, None)
    >>> H.is_stable_set([1,2], certificate=True)
    (False, (1, 2))
    >>> H.is_stable_set([0,2,4], certificate=True)
    (False, (0, 4))
    """
    for v in vertices:
        for u in graph.neighbors(v):
            if u in vertices:
                if certificate:
                    return False, (v, u)
                else:
                    return False
    if certificate:
        return True, None
    else:
        return True


def _revertable_set_removal(
    graph: nx.Graph,
    vertices: Collection[Vertex],
    opt: Callable[[nx.Graph], T],
    copy: bool,
) -> T:
    """
    Remove given ``vertices`` from the graph, return the result of ``opt(graph)``,
    and restore the original graph.

    Parameters
    ----------
    vertices:
        Vertex set to remove.
    opt:
        A function whose result is returned for the graph with vertices removed.
    copy:
        Create a copy of the graph before the vertices are removed
        and ``property`` is checked.
        Otherwise, the graph is modified in-place.
        In that case, some metadata may be lost.
    """
    copy = copy or nx.is_frozen(graph)
    vertex_data: dict[Vertex, dict[str, Any]]
    edge_data: dict[Edge, dict[str, Any]]
    neighbors: list[Tuple[Vertex, Vertex]]

    if copy:
        graph = nx.Graph(graph)
    else:
        neighbors = [(u, v) for u in vertices for v in graph.neighbors(u)]
        vertex_data = {u: graph.nodes[u] for u in vertices}
        edge_data = {(u, v): graph.edges[u, v] for u, v in neighbors}

    graph.remove_nodes_from(vertices)

    try:
        return opt(graph)
    finally:
        if not copy:
            for v in vertices:
                graph.add_node(v, **vertex_data[v])
            for u, v in neighbors:
                graph.add_edge(u, v, **edge_data[(u, v)])


def is_separating_set(
    graph: nx.Graph,
    vertices: Collection[Vertex],
    copy: bool = True,
) -> bool:
    """
    Return if ``vertices`` are a separating set.

    Definitions
    -----------
    :prf:ref:`Separating set <def-separating-set>`

    Parameters
    ----------
    vertices:
        The vertices to check.
    copy:
        Create a copy of the graph before the vertices are removed
        and connectivity is checked. Otherwise, the graph is modified in-place.
        In that case, some metadata may be lost.

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

    pyrigi._graph_input_check.vertex_members(graph, vertices)

    return _revertable_set_removal(
        graph, vertices, lambda g: not nx.is_connected(g), copy=copy
    )


def is_uv_separating_set(
    graph: nx.Graph,
    vertices: Collection[Vertex],
    u: Vertex,
    v: Vertex,
    copy: bool = True,
) -> bool:
    """
    Return if ``vertices`` separate the vertices ``u`` and ``v``.

    Definitions
    -----------
    :prf:ref:`Separating set <def-separating-set>`

    Parameters
    ----------
    vertices:
        The set of vertices to be checked to separate ``u`` and ``v``.
        If ``u`` or ``v`` is contained in ``vertices``,
        ``ValueError`` is raised.
    u, v:
    copy:
        Create a copy of the graph before the vertices are removed
        and connectivity is checked. Otherwise, the graph is modified in-place.
        In that case, some metadata may be lost.

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> H = graphs.Cycle(5)
    >>> H.is_uv_separating_set([1,3], 0, 2)
    True
    >>> H.is_uv_separating_set([2,4], 0, 1)
    False
    """
    if u in vertices:
        raise ValueError(f"u={u} is in the separating set")
    if v in vertices:
        raise ValueError(f"v={v} is in the separating set")

    pyrigi._graph_input_check.vertex_members(graph, vertices)

    def check_graph(g: nx.Graph) -> bool:
        components = nx.connected_components(g)
        for c in components:
            if u in c and v in c:
                return False
        return True

    return _revertable_set_removal(graph, vertices, check_graph, copy=copy)


def is_stable_separating_set(
    graph: nx.Graph,
    vertices: Collection[Vertex],
    copy: bool = True,
) -> bool:
    """
    Return if ``vertices`` are a stable separating set.

    See :meth:`~pyrigi.graph.Graph.is_stable_set` and
    :meth:`~pyrigi.graph.Graph.is_separating_set` for
    the description of the parameters.

    Definitions
    -----------
    * :prf:ref:`Stable set <def-stable-set>`
    * :prf:ref:`Separating set <def-separating-set>`

    Parameters
    ----------
    vertices:
    copy:

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> H = graphs.Cycle(5)
    >>> H.is_stable_separating_set([1,3])
    True
    >>> H.is_stable_separating_set([1,2])
    False
    """
    return is_stable_set(graph, vertices) and is_separating_set(
        graph, vertices, copy=copy
    )


################################################################################
def stable_separating_set(
    graph: "PRGraph",
    u: Optional[Vertex] = None,
    v: Optional[Vertex] = None,
    check_flexible: bool = True,
    check_connected: bool = True,
    check_distinct_rigid_components: bool = True,
) -> set[Vertex]:
    """
    Find a stable separating set if the graph is 2-flexible.

    Algorithm 1 in :cite:p:`ClinchGaramvölgyiEtAl2024` is used.
    The algorithm allows to specify two vertices ``u`` and ``v``
    that are separated by the returned stable separating set,
    provided that they are not in the same
    :prf:ref:`2-rigid component <def-rigid-components>`.
    Alternatively, a single vertex ``u`` can be specified that is
    avoided in the returned stable separating set,
    provided it does not separate the graph.

    Definitions
    -----------
    * :prf:ref:`Stable set <def-stable-set>`
    * :prf:ref:`Separating set <def-separating-set>`
    * :prf:ref:`Flexible graph <def-gen-rigid>`

    Parameters
    ----------
    u:
        See the description above,
        an arbitrary vertex is chosen if none is specified.
    v:
        See the description above,
        a suitable vertex is chosen if none is specified.
        It cannot be in the same 2-rigid component as ``u``.
    check_flexible:
        If ``True``, it is checked that the graph is
        2-flexible as the algorithm only works for those.
        If ``False`` and the graph is 2-rigid,
        the output is undefined.
    check_connected:
        If ``True``, checks for graph connectivity are run and
        the result may alter based on them.
        If ``False`` and the graph is disconnected,
        the output is undefined.
    check_distinct_rigid_components:
        If ``True``, it is checked that ``u`` and ``v``
        are in different 2-rigid components.
        If ``False``, both ``u`` and ``v`` must be specified.
    """
    from pyrigi import Graph as PRGraph

    if check_flexible and graph.is_rigid(dim=2):
        raise ValueError("The graph must be 2-flexible!")

    # find the smallest connected component and put it first
    if not check_connected:
        is_connected = True
    else:
        connected_components = list(nx.connected_components(graph))
        smallest_component_index = np.argmin(map(len, connected_components))
        connected_components[0], connected_components[smallest_component_index] = (
            connected_components[smallest_component_index],
            connected_components[0],
        )
        is_connected = len(connected_components) == 1

    # if v is set, u must be also set
    if u is None and v is not None:
        u, v = v, u
    assert check_distinct_rigid_components or v is not None

    # choose a vertex at random
    if u is None:
        if is_connected:
            u = next(iter(graph.nodes))
        else:
            u = next(iter(connected_components[0]))
    else:
        assert u in graph

    # make sure node is valid node
    if v is not None:
        assert v in graph

    # if the graph is not connected, we can possibly reduce the work needed
    # by finding a connected component that contains u
    # and find a separating set in it or just calling it the day
    # if v is not specified or lies in another component

    # separate a connected component that contains u
    if is_connected:
        subgraph = graph
    else:
        u_component = next(filter(lambda c: u in c, connected_components))
        subgraph = PRGraph(nx.induced_subgraph(graph, u_component))

        # if v is not specified, we just choose a different component
        # and return the empty separating set
        if v is None or v not in u_component:
            return set()

    # Makes sure v is in different rigid component
    if v is None or check_distinct_rigid_components:
        match _validate_uv_different_rigid_comps(subgraph, u, v):
            case None:
                raise ValueError(
                    "Chosen vertices must not be in the same rigid component"
                )
            case new_v:
                v = new_v

    return _find_stable_uv_separating_set(graph, u, v)


def _validate_uv_different_rigid_comps(
    graph: "PRGraph",
    u: Vertex,
    v: Optional[Vertex],
) -> Optional[Vertex]:
    """
    Make sure ``u`` and ``v`` are in different rigid components and
    find such ``v`` if not provided.

    If the graph is rigid or if ``u`` and ``v`` are in the same rigid component,
    then ``None`` is returned.
    Otherwise, a valid ``v`` is returned.

    Parameters
    ----------
    graph:
        The graph to check.
    u:
        The first vertex.
    v:
        The second vertex, will be chosen arbitrary if not provided.
    """
    rigid_components = graph.rigid_components()

    if len(rigid_components) < 2:
        logging.warning("Provided graph is not flexible")
        return None

    # vertices that share a component with u
    disallowed = set(v for c in rigid_components for v in c if u in c)

    # Check that input is valid
    if v is not None:
        if v in disallowed:
            logging.warning(f"Both vertices {u} and {v} share the same rigid component")
            return None
    else:
        # choose a vertex at random
        v = next(x for x in graph.nodes if x not in disallowed)
    return v


def _find_stable_uv_separating_set(
    graph: "PRGraph",
    u: Vertex,
    v: Vertex,
) -> set[Vertex]:
    """
    Find a stable separating set in a flexible graph.

    This is the main body of Algorithm 1 in :cite:p:`ClinchGaramvölgyiEtAl2024`.

    Parameters
    ----------
    graph:
        A mutable graph to find a stable separating set in.
    u:
        The vertex around which we look for a stable separating set.
    v:
        The vertex marking another rigid component.
    """

    # Checks neighborhood of u
    # if it is a stable set, we are done
    neighborhood = set(graph.neighbors(u))
    violation = is_stable_set(graph, neighborhood, certificate=True)[1]

    # found a stable set around u
    if violation is None:
        return neighborhood

    # used to preserve graph's metadata
    vertex_data: dict[Vertex, dict[str, Any]] = {}
    edge_data: dict[FrozenSet[Vertex], dict[str, Any]] = {}

    def contract(
        graph: nx.Graph, u: Vertex, x: Vertex
    ) -> tuple[set[Vertex], set[Vertex]]:
        """
        Contract the vertices u and x
        and return their original neighbors for easy restoration.
        """
        u_neigh = set(graph.neighbors(u))
        x_neigh = set(graph.neighbors(x))

        # Store graphs metadata
        for n in graph.neighbors(x):
            edge_data[frozenset((n, x))] = graph.edges[x, n]
        vertex_data[x] = graph.nodes[x]

        graph.remove_node(x)
        for n in x_neigh - {u}:
            graph.add_edge(u, n)
        return u_neigh, x_neigh

    def restore(
        graph: nx.Graph,
        u: Vertex,
        x: Vertex,
        u_neigh: set[Vertex],
        x_neigh: set[Vertex],
    ):
        """
        Restore the contracted graph to its original form.
        Inverse operation for contract.
        """
        for n in x_neigh - u_neigh - {u}:
            graph.remove_edge(u, n)
        graph.add_node(x, **vertex_data[x])
        for n in x_neigh:
            graph.add_edge(x, n, **edge_data[frozenset((n, x))])

    # Try the both vertices forming a triangle with u
    # Pass has to succeed with at least one of them,
    # otherwise the rigid components are not maximal or the graph is rigid.
    for x in violation:
        u_neigh, x_neigh = contract(graph, u, x)

        rigid_components = graph.rigid_components()

        # The contracted vertex is in the same rigid component as v
        problem_found = False
        for c2 in filter(lambda c: v in c, rigid_components):
            if u in c2:
                restore(graph, u, x, u_neigh, x_neigh)
                problem_found = True
                break
        if problem_found:
            continue

        # ensure that graph gets always restored to the original form properly
        try:
            return _find_stable_uv_separating_set(graph, u, v)
        finally:
            restore(graph, u, x, u_neigh, x_neigh)

    raise RuntimeError("Rigid components are not maximal")
