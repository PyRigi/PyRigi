"""
This modules is responsible for checking if a graph can have a NAC coloring.
"""

import networkx as nx

from pyrigi.graph._flexibility.nac.core import NACColoring
from pyrigi.graph._flexibility.nac.mono_classes import MonoClassType, find_mono_classes
from pyrigi.graph._rigidity.generic import is_min_rigid


def check_NAC_constrains(self: nx.Graph) -> bool:
    """
    Check basic NAC-coloring existence constraints.

    Checks whether graph has self loops (prohibited),
    is non-empty, directed and has at least 2 edges.
    Serves as basic input sanitation.


    Throws:
        ValueError: if the graph is empty or has loops.

    Returns:
        True if a NAC coloring may exist, False if none exists for sure.
    """
    if nx.number_of_selfloops(self) > 0:
        raise LookupError()

    if self.nodes() == 0:
        raise ValueError("Undefined for an empty graph")

    if nx.is_directed(self):
        raise ValueError("Cannot process a directed graph")

    # NAC is a surjective edge coloring, you passed a graph with less then 2 edges
    if len(nx.edges(self)) < 2:
        return False

    return True


def _can_have_flexible_labeling(
    graph: nx.Graph,
) -> bool:
    """
    Assure basic conditions for NAC-coloring existence based
    on the number of vertices and edges.

    Use equivalence that graph (more below) has NAC coloring if and only if
    it has a flexible labeling. But for flexible labeling we know the upper bound
    for the number of edges in the graph.
    :cite:p:`GraseggerLegerskySchicho2019{Thm 3.1,Thm 4.7}`

    Parameters
    ----------
    graph:
        A connected graph with at leas one edge.
    ----------

    Return
        True if the graph can have NAC coloring,
        False if we are sure there is none.
    """
    if graph.number_of_edges() <= 1:
        return False
    assert nx.node_connectivity(graph) > 0
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    return m <= n * (n - 1) // 2 - (n - 2)


def _check_for_simple_stable_cut(
    graph: nx.Graph,
) -> NACColoring | None:
    """
    If a stable cut exits in a graph, a NAC-coloring also exists. Search for simple ones.

    If there is a single vertex outside of any triangle component,
    we can trivially find NAC coloring for the graph.
    Also handles nodes with degree <= 2.
    :cite:p:`GraseggerLegerskySchicho2019{Thm 4.4,Cor 4.5}`

    Parameters
    ----------
    graph:
        The graph to work with, basic NAC coloring constrains
        should be already checked.
    ----------

    Returns
        If no NAC coloring can be found using this method, None is
        returned. If some NAC coloring can be found, return a certificate.
    """

    # remove isolated vertices
    vertices_outside_triangle_components: set[int] = set(
        u for u, d in graph.degree() if d > 0
    )

    # remove vertices that are part of a triangle connected component
    _, component_to_edge = find_mono_classes(graph, MonoClassType.TRI_CONNECTED)
    for component_edges in component_to_edge:
        if len(component_edges) == 1:
            continue

        vertices_outside_triangle_components.difference_update(
            v for edge in component_edges for v in edge
        )

    if len(vertices_outside_triangle_components) == 0:
        return None

    # create corresponding NAC coloring certificate
    for v in vertices_outside_triangle_components:
        red = set((v, u) for u in graph.neighbors(v))
        if len(red) == graph.number_of_edges():
            # we found a wrong vertex
            # this may happen if the red part is the whole graph
            continue
        blue = set(graph.edges)

        # remove shared edges
        blue.difference_update(red)
        blue.difference_update((u, v) for v, u in red)

        assert len(red) > 0
        assert len(blue) > 0

        return (red, blue)

    raise ValueError("NAC coloring was not found even though it should exist.")


def _check_is_min_rigid_and_NAC_coloring_exists(
    graph: nx.Graph,
) -> bool | None:
    """
    Check for minimally rigid graphs special properties.

    For minimally rigid graphs it holds that
    there exists NAC coloring iff graph is not triangle connected.
    :cite:p:`ClinchGaramvölgyiEtAl2024{Thm 3.4}`

    Return
        True if the graph has a NAC-coloring,
        False if we are sure there is none.
        None if we cannot decide (the graph is not min_rigid)
    """

    min_rigid = is_min_rigid(graph, dim=2)
    if not min_rigid:
        return None

    _, components_to_edges = find_mono_classes(graph, MonoClassType.TRI_CONNECTED)
    return len(components_to_edges) != 1


def has_NAC_coloring_checks(graph: nx.Graph) -> bool | None:
    """
    Implementation for has_NAC_coloring, but without fallback to single_NAC_coloring.

    May be used before an exhaustive search that would not find anything anyway.
    """
    if not check_NAC_constrains(graph):
        return False

    if _check_for_simple_stable_cut(graph) is not None:
        return True

    if nx.algorithms.connectivity.node_connectivity(graph) < 2:
        return True

    # Needs to be run after connectivity checks
    if not _can_have_flexible_labeling(graph):
        return False

    res = _check_is_min_rigid_and_NAC_coloring_exists(graph)
    if res is not None:
        return res

    return None
