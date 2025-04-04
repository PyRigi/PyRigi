"""
algorithms related to separating sets.
Also includes an algorithm for a stable separating set search in a flexible graph
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
) -> bool | tuple[bool, Optional[tuple[Vertex, Vertex]]]:
    """
    Check if the given set of vertices is stable in the given graph.
    and if not, find a pair of vertices in the set that are neighboring.

    Definitions
    -----------
    :prf:ref:`Stable set <def-stable-set>`

    Parameters
    ----------
    vertices:
        the vertices to check
    certificate:
        if True, return also a pair of vertices that are in the set
        and are neighboring. See returns.

    Returns
    -------
        If certificate is ``False``,
        returns a boolean whether the set is stable or not.
        If certificate is ``True``,
        a tuple where first boolean states whenever the set is stable
        and second item gives a pair of vertices contradicting the stable
        property if applicable.

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
    Remove given vertices from the graph, perform operation,
    return vertices along with edges back.

    Parameters
    ----------
    vertices:
        Vertex set to remove
    opt:
        Operation to perform on a graph with vertices removed
    copy:
        Create a copy of the graph before the vertices are removed
        and connectivity is checked. Otherwise, the graph is modified in-place.
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
    Check if the given set of vertices is a separator in the given graph.

    Definitions
    -----------
    :prf:ref:`Separating set <def-separating-set>`

    Parameters
    ----------
    vertices:
        the vertices to check
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
    Check if the given set separates vertices u and v.

    Definitions
    -----------
    :prf:ref:`Separating set <def-separating-set>`

    Parameters
    ----------
    vertices:
        the vertices to check
    u:
        the first vertex
    v:
        the second vertex
    copy:
        Create a copy of the graph before the vertices are removed
        and connectivity is checked. Otherwise, the graph is modified in-place.
        In that case, some metadata may be lost.

    Raises
    ------
    ValueError:
        If either of the vertices is contained in the separating set

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
    Check if the given set of vertices is a stable separating in the given graph.

    Definitions
    -----------
    :prf:ref:`Stable separating set <def-stable-separating-set>`

    Parameters
    ----------
    vertices:
        the separating set of vertices
    copy:
        Create a copy of the graph before the vertices are removed
        and connectivity is checked. Otherwise, the graph is modified in-place.
        In that case, some metadata may be lost.

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> H = graphs.Cycle(5)
    >>> H.is_stable_separating_set([1,3])
    True
    >>> H.is_stable_separating_set([1,2])
    False

    Note
    ----
        See :meth:`~pyrigi.graph.Graph.is_stable_set` and
        :meth:`~pyrigi.graph.Graph.is_separating_set`.
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
    Find a stable separating set in a flexible graph
    according to Algorithm 1 in :cite:p:`ClinchGaramvölgyiEtAl2024`.

    Definitions
    -----------
    :prf:ref:`Stable separating set <def-stable-separating-set>`
    :prf:ref:`Contiguous rigidity <def-cont-rigid-framework>`

    Parameters
    ----------
    u:
        The first vertex indicating the rigid component used,
        arbitrary vertex is chosen otherwise
    v:
        The second vertex indicating the other rigid component used
        arbitrary vertex is chosen otherwise.
        Cannot share a same rigid component as ``u``.
    check_flexible:
        If ``True``, ensure that graph is flexible as
        the algorithm only works for flexible graphs.
    check_connected:
        If ``True``, checks for graph connectivity are run and result
        may altered based on them.
        If disconnected graph is passed as argument
        when the check is disabled, algorithms output is undefined.
    check_distinct_rigid_components:
        Whether to check that ``u`` and ``v``
        are in different rigid components.
        If set to ``False`` Both ``u`` and ``v`` must be specified.

    Returns
    -------
        For a valid input a stable separating set in the graph is returned.
        For rigid graphs and cases when ``u`` and ``v`` are not in the same
        rigid component, ``None`` is returned.
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
    # if v is not specified of lays in another component

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

    Parameters
    ----------
    graph:
        The graph to check
    u:
        The first vertex
    v:
        The second vertex, will be chosen arbitrary if not provided

    Returns
    -------
        None if the graph is rigid or
        if ``u`` and ``v`` are in the same rigid component.
    otherwise, returns valid ``v``.
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
        mutable graph to find a stable separating set in
    u:
        the vertex around which we look for a stable separating set
    v:
        the vertex marking another rigid component
    """

    # Checks neighborhood of u
    # if it is a stable set, we are done
    neiborhood = set(graph.neighbors(u))
    violation = is_stable_set(graph, neiborhood, certificate=True)[1]

    # found a stable set around u
    if violation is None:
        return neiborhood

    # used to preserve graph's metadata
    vertex_data: dict[Vertex, dict[str, Any]] = {}
    edge_data: dict[FrozenSet[Vertex], dict[str, Any]] = {}

    def contract(
        graph: nx.Graph, u: Vertex, x: Vertex
    ) -> tuple[set[Vertex], set[Vertex]]:
        """
        Contracts the vertices u and x
        and returns their original neighbors for easy restoration
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
        Restores contracted graph to it's original form.
        Inverse operation for contract.
        """
        for n in x_neigh - u_neigh - {u}:
            graph.remove_edge(u, n)
        graph.add_node(x, **vertex_data[x])
        for n in x_neigh:
            graph.add_edge(x, n, **edge_data[frozenset((n, x))])

    # Tries both the vertices forming a triable with u
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

        # ensures that graph gets always restored to the original form properly
        try:
            return _find_stable_uv_separating_set(graph, u, v)
        finally:
            restore(graph, u, x, u_neigh, x_neigh)

    raise RuntimeError("Rigid components are not maximal")
