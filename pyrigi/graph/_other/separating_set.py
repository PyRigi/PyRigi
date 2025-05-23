"""
This module provides algorithms related to separating sets.

It includes an algorithm for a stable separating set search in a 2-flexible graph
according to Algorithm 1 in :cite:p:`ClinchGaramvölgyiEtAl2024`.
"""

from typing import (
    Any,
    Callable,
    Collection,
    FrozenSet,
    Optional,
    Tuple,
    TypeVar,
)

import networkx as nx
import numpy as np

import pyrigi.graph._rigidity.generic as generic_rigidity
import pyrigi.graph._utils._input_check as _graph_input_check
from pyrigi.data_type import Edge, Vertex

T = TypeVar("T")


def is_stable_set(
    graph: nx.Graph,
    vertices: Collection[Vertex],
    certificate: bool = False,
) -> bool | tuple[bool, Optional[Edge]]:
    """
    Return whether given ``vertices`` form a stable set.

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
        If ``False``, a boolean is returned whether the set is stable or not.
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


def _remove_apply_restore_vertices(
    graph: nx.Graph,
    vertices: Collection[Vertex],
    func: Callable[[nx.Graph], T],
    use_copy: bool,
) -> T:
    """
    Remove given ``vertices`` from the graph,
    return the result of ``func`` applied to this subgraph,
    and restore the original graph.

    Parameters
    ----------
    vertices:
        Vertex set to remove.
    func:
        A function whose result is returned for the graph with vertices removed.
        Warning: if ``func`` modifies its input graph, then it is not guaranteed
        that the ``graph`` is restored correctly by ``_revertable_set_removal``.
    use_copy:
        If ``True`` (default),
        create a copy of the graph before the vertices are removed
        and ``property`` is checked.
        Otherwise, the graph is modified in-place.
        In that case, some metadata may be lost.
    """
    use_copy = use_copy or nx.is_frozen(graph)
    vertex_data: dict[Vertex, dict[str, Any]]
    edge_data: dict[Edge, dict[str, Any]]
    neighbors: list[Tuple[Vertex, Vertex]]

    if use_copy:
        graph = nx.Graph(graph)
    else:
        neighbors = [(u, v) for u in vertices for v in graph.neighbors(u)]
        vertex_data = {u: graph.nodes[u] for u in vertices}
        edge_data = {(u, v): graph.edges[u, v] for u, v in neighbors}

    graph.remove_nodes_from(vertices)

    try:
        return func(graph)
    finally:
        if not use_copy:
            for v in vertices:
                graph.add_node(v, **vertex_data[v])
            for u, v in neighbors:
                graph.add_edge(u, v, **edge_data[(u, v)])


def is_separating_set(
    graph: nx.Graph,
    vertices: Collection[Vertex],
    use_copy: bool = True,
) -> bool:
    """
    Return whether ``vertices`` are a separating set.

    Definitions
    -----------
    :prf:ref:`Separating set <def-separating-set>`

    Parameters
    ----------
    vertices:
        The vertices to check.
    use_copy:
        If ``True`` (default),
        create a copy of the graph before the vertices are removed
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

    _graph_input_check.vertex_members(graph, vertices)

    def check_graph(g: nx.Graph) -> bool:
        if g.number_of_nodes() == 0:
            raise ValueError(
                "The parameter `vertices` must be a proper subset of graph vertex set."
            )

        return not nx.is_connected(g)

    return _remove_apply_restore_vertices(
        graph, vertices, check_graph, use_copy=use_copy
    )


def is_uv_separating_set(
    graph: nx.Graph,
    vertices: Collection[Vertex],
    u: Vertex,
    v: Vertex,
    use_copy: bool = True,
) -> bool:
    """
    Return whether ``vertices`` separate the vertices ``u`` and ``v``.

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
    use_copy:
        If ``True`` (default),
        create a copy of the graph before the vertices are removed
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
        raise ValueError(
            f"The vertex u={u} should not be among the ones in the parameter `vertices`."
        )
    if v in vertices:
        raise ValueError(
            f"The vertex v={v} should not be among the ones in the parameter `vertices`."
        )

    _graph_input_check.vertex_members(graph, vertices)
    _graph_input_check.vertex_members(graph, (u, v))

    def check_graph(g: nx.Graph) -> bool:
        if g.number_of_nodes() == 0:
            raise ValueError(
                "The parameter `vertices` must be a proper subset of graph vertex set."
            )

        components = nx.connected_components(g)
        for comp in components:
            if u in comp and v in comp:
                return False
        return True

    return _remove_apply_restore_vertices(
        graph, vertices, check_graph, use_copy=use_copy
    )


def is_stable_separating_set(
    graph: nx.Graph,
    vertices: Collection[Vertex],
    use_copy: bool = True,
) -> bool:
    """
    Return whether ``vertices`` are a stable separating set.

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
    use_copy:

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
        graph, vertices, use_copy=use_copy
    )


def stable_separating_set(
    graph: nx.Graph,
    u: Optional[Vertex] = None,
    v: Optional[Vertex] = None,
    check_flexible: bool = True,
    check_connected: bool = True,
    check_distinct_rigid_components: bool = True,
) -> set[Vertex]:
    """
    Find a stable separating set if the graph is 2-flexible.

    If two vertices ``u`` and ``v`` that are not in the same
    :prf:ref:`2-rigid component <def-rigid-components>` are provided,
    then the returned stable separating set separates them.
    If a single vertex ``u`` is specified,
    the returned stable separating set avoids ``u``,
    provided ``u`` is not contained in all 2-rigid components.
    Algorithm 1 in :cite:p:`ClinchGaramvölgyiEtAl2024` is used.
    An empty set can be returned when the graph is disconnected.

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
        the behaviour might be arbitrary.
    check_connected:
        If ``True``, checks for graph connectivity are run and
        the result may alter based on them.
        If ``False`` and the graph is disconnected,
        the behaviour might be arbitrary.
    check_distinct_rigid_components:
        If ``True``, it is checked that ``u`` and ``v``
        are in different 2-rigid components.
        If ``False``, both ``u`` and ``v`` must be specified.
    """

    ############################################################################
    # Preconditions
    ############################################################################

    # if v is set, u must be also set
    if u is None and v is not None:
        raise ValueError("If `v` is specified, `u` is required as well.")

    if not check_distinct_rigid_components and v is None:
        raise ValueError(
            "Both `u` and `v` must be specified"
            + "when `check_distinct_rigid_components=False`."
        )

    # make sure inputs are valid vertices
    if u is not None:
        _graph_input_check.vertex_members(graph, u)
    if v is not None:
        _graph_input_check.vertex_members(graph, v)

    ############################################################################
    # Connectivity
    ############################################################################

    # if the graph is not connected, we can possibly reduce the work needed
    # by finding a connected component that contains u
    # and find a separating set in it or return an empty set
    # if v is not specified or lies in another component

    if check_connected:
        connected_components = list(nx.connected_components(graph))

        # prefer the smallest component
        smallest_component_index = np.argmin(map(len, connected_components))
        connected_components[0], connected_components[smallest_component_index] = (
            connected_components[smallest_component_index],
            connected_components[0],
        )

        if len(connected_components) > 1:
            if v is not None:
                u_component = next(filter(lambda c: u in c, connected_components))
                v_component = next(filter(lambda c: v in c, connected_components))

                # if u and v are in different components
                if u_component is not v_component:
                    return set()

                # u and v are set and are in the same component
                graph = graph.__class__(nx.induced_subgraph(graph, u_component))
            else:
                # v can be assumed to be in another component
                # so the empty set is a stable separating set
                return set()

    # Make sure that the graph or the connected component with u & v is not rigid
    if check_flexible and generic_rigidity.is_rigid(graph, dim=2):
        raise ValueError("The graph/component must be 2-flexible!")

    ############################################################################
    # Vertices validation and initialization
    ############################################################################

    # At this point the graph is connected.
    # We do not know anything about u and v

    # u needs to be chosen
    if u is None:  # v is also None
        # u cannot be an articulation point (i.e. a separating vertex)
        # as if the graph is not 2-connected
        # {u} may be the only separating set of the graph
        articulation_points = list(nx.articulation_points(graph))

        if len(articulation_points) > 0:
            return set(articulation_points[:1])

        # now it is safe to continue with the rest of the algorithm
        u = next(iter(graph.nodes))

    # Makes sure v is in a different rigid component
    if v is None or check_distinct_rigid_components:
        v = _validate_uv_different_rigid_comps(graph, u, v)

    return _find_stable_uv_separating_set(graph, u, v)


def _validate_uv_different_rigid_comps(
    graph: nx.Graph,
    u: Vertex,
    v: Optional[Vertex],
) -> Vertex:
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
    rigid_components = generic_rigidity.rigid_components(graph, dim=2)

    if len(rigid_components) < 2:
        raise ValueError("The given graph is not 2-flexible.")

    # vertices that share a component with u
    disallowed = set(v for comp in rigid_components for v in comp if u in comp)

    # Check that input is valid
    if v is not None:
        if v in disallowed:
            raise ValueError(
                f"Both vertices {u} and {v} are in the same 2-rigid component."
            )
    else:
        # choose a vertex at random
        v = next((x for x in graph.nodes if x not in disallowed), None)
        if v is None:
            raise ValueError(
                f"The vertex `u={u}` is contained in all 2-rigid components."
            )
    return v


def _find_stable_uv_separating_set(
    graph: nx.Graph,
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
        Contract the vertices ``u`` and ``x``
        and return their original neighbors for easy restoration.
        """
        u_neigh = set(graph.neighbors(u))
        x_neigh = set(graph.neighbors(x))

        # Store graphs metadata
        for neighbor in graph.neighbors(x):
            edge_data[frozenset((neighbor, x))] = graph.edges[x, neighbor]
        vertex_data[x] = graph.nodes[x]

        graph.remove_node(x)
        for neighbor in x_neigh - {u}:
            graph.add_edge(u, neighbor)
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
        Inverse operation for ``contract``.
        """
        for neighbor in x_neigh - u_neigh - {u}:
            graph.remove_edge(u, neighbor)
        graph.add_node(x, **vertex_data[x])
        for neighbor in x_neigh:
            graph.add_edge(x, neighbor, **edge_data[frozenset((neighbor, x))])

    # Try both the vertices in `violation` forming a triangle with u
    # It is guaranteed by the proof of Algorithm 1 :cite:p:`ClinchGaramvölgyiEtAl2024`
    # that at least one of them yields a result.
    for x in violation:
        u_neigh, x_neigh = contract(graph, u, x)

        rigid_components = generic_rigidity.rigid_components(graph, dim=2)

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

    raise RuntimeError("Something went wrong!")
