"""
Algorithm for a stable cut search in a flexible graph
according to Algorithm 1 in :cite:p:`ClinchGaramvölgyiEtAl2024`.
"""

import logging
from typing import TYPE_CHECKING, Optional


from pyrigi._cuts import is_stable_set
from pyrigi.data_type import StableSeparatingCut
import networkx as nx
import numpy as np

from pyrigi.data_type import Vertex

if TYPE_CHECKING:
    from pyrigi import Graph as PRGraph


def stable_separating_set_in_flexible_graph(
    graph: nx.Graph,
    u: Optional[Vertex] = None,
    v: Optional[Vertex] = None,
) -> Optional[StableSeparatingCut]:
    """
    Find a stable cut in a flexible graph
    according to Algorithm 1 in :cite:p:`ClinchGaramvölgyiEtAl2024`.

    Definitions
    -----------
    :prf:ref:`Stable separating set <def-stable-separating-set>`
    :prf:ref:`Contiguous rigidity <def-cont-rigid-framework>`

    Parameters
    ----------
    graph:
        The flexible graph to search
    u:
        The first vertex indicating the rigid component used,
        arbitrary vertex is chosen otherwise
    v:
        The second vertex indicating the other rigid component used
        arbitrary vertex is chosen otherwise.
        Cannot share a same rigid component as ``u``.

    Returns
    -------
        For a valid input a :class:`pyrigi.data_type.StableCut` in the graph is returned.
        For rigid graphs and cases when ``u`` and ``v`` are not in the same
        rigid component, ``None`` is returned.

    Note
    ----
        Vertex and edge metadata may be dropped, if you use them,
        make copy of your graph first.
    """
    from pyrigi import Graph as PRGraph

    if graph.number_of_nodes() <= 1:
        return None

    # find the smallest connected component and put it first
    connected_components = list(nx.connected_components(graph))
    smallest_component_index = np.argmin(map(len, connected_components))
    connected_components[0], connected_components[smallest_component_index] = (
        connected_components[smallest_component_index],
        connected_components[0],
    )

    # if v is set, u must be also set
    if u is None and v is not None:
        u, v = v, u

    # choose a vertex at random
    if u is None:
        u = next(iter(connected_components[0]))
    else:
        assert u in graph

    # make sure node is valid node
    if v is not None:
        assert v in graph

    # if there is only a single component, fallback to a faster algorithm
    if len(connected_components) == 1:
        return stable_separating_set_in_flexible_graph_fast(graph, u, v)

    # if the graph is not connected, we can possibly reduce the work needed
    # by finding a connected component that contains u
    # and find a cut in it or just calling it the day
    # if v is not specified of lays in another component

    # separate a connected component that contains u
    # partition from more_itertools could have been used
    u_component = next(filter(lambda c: u in c, connected_components))
    other = filter(lambda c: u not in c, connected_components)
    subgraph = PRGraph(nx.induced_subgraph(graph, u_component))

    # if v is not specified, we just choose a different component and return the empty cut
    if v is None or v not in u_component:
        return StableSeparatingCut(
            set(u_component), set(x for c in other for x in c), set()
        )

    # Makes sure v is in different rigid component
    match _find_and_validate_u_and_v(subgraph, u, v):
        case None:
            return None
        case _:
            v = v

    if not isinstance(graph, PRGraph):
        graph = PRGraph(subgraph)

    cut = _process(graph, u, v)
    StableSeparatingCut(
        a=cut.a, b=cut.b | set(x for c in other for x in c), cut=cut.cut
    )
    return cut


def stable_separating_set_in_flexible_graph_fast(
    graph: nx.Graph,
    u: Optional[Vertex] = None,
    v: Optional[Vertex] = None,
    ensure_rigid_components: bool = True,
) -> Optional[StableSeparatingCut]:
    """
    Find a stable cut in a flexible graph
    according to Algorithm 1 in :cite:p:`ClinchGaramvölgyiEtAl2024`.

    Similar to :meth:`~pyrigi.graph.Graph.stable_separating_set_in_flexible_graph`,
    but faster as connectivity are removed.
    The algorithm's output is undefined in these cases.

    Definitions
    -----------
    :prf:ref:`Stable separating set <def-stable-separating-set>`
    :prf:ref:`Contiguous rigidity <def-cont-rigid-framework>`

    Parameters
    ----------
    graph:
        The flexible graph to search
    u:
        The first vertex indicating the rigid component used,
        arbitrary vertex is chosen otherwise
    v:
        The second vertex indicating the other rigid component used
        arbitrary vertex is chosen otherwise.
        Cannot share a same rigid component as ``u``.
    ensure_rigid_components:
        Whether to ensure that ``u`` and ``v``
        are not in the same rigid component.
        Both ``u`` and ``v`` must be specified.

    Returns
    -------
        For a valid input a :class:`pyrigi.data_type.StableSeparatingCut` in the graph is returned.
        For rigid graphs and cases when ``u`` and ``v`` are not in the same
        rigid component, ``None`` is returned.

    Note
    ----
        Vertex and edge metadata may be dropped, if you use them,
        make copy of your graph first.
    """
    from pyrigi import Graph as PRGraph

    if graph.number_of_nodes() <= 1:
        return None

    # if v is set, u must be also set
    if u is None and v is not None:
        u, v = v, u

    # choose a vertex at random
    if u is None:
        u = next(graph.nodes())

    # check is disabled => v must be set
    assert ensure_rigid_components or v is not None

    # graph will be modified in place
    if not isinstance(graph, PRGraph):
        graph = PRGraph(graph)

    if ensure_rigid_components:
        match _find_and_validate_u_and_v(graph, u, v):
            case None:
                return None
            case _:
                v = v

    return _process(graph, u, v)


def _find_and_validate_u_and_v(
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


def _process(
    graph: "PRGraph",
    u: Vertex,
    v: Vertex,
) -> StableSeparatingCut:
    """
    Find a stable cut in a flexible graph.

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
    violation = is_stable_set(graph, neiborhood)[1]

    # found a stable set around u
    if violation is None:
        return StableSeparatingCut(
            neiborhood | {u},
            set(graph.nodes) - {u},
            neiborhood,
        )

    def contract(
        graph: nx.Graph, u: Vertex, x: Vertex
    ) -> tuple[set[Vertex], set[Vertex]]:
        """
        Contracts the vertices u and x
        and returns their original neighbors for easy restoration
        """
        u_neigh = set(graph.neighbors(u))
        x_neigh = set(graph.neighbors(x))
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
        graph.add_node(x)
        for n in x_neigh:
            graph.add_edge(x, n)

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

        res = _process(graph, u, v)
        restore(graph, u, x, u_neigh, x_neigh)
        return res

    raise RuntimeError("Rigid components are not maximal")
