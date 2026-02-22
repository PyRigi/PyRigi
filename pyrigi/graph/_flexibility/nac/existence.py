"""
Modules checks whether a graph can have a :prf:ref:`NAC-colorings <def-nac>`.
"""

import networkx as nx

from pyrigi.graph._flexibility.nac.core import NACColoring
from pyrigi.graph._flexibility.nac.mono_classes import MonoClassType, find_mono_classes
from pyrigi.graph._rigidity.generic import is_min_rigid


def check_NAC_constrains(self: nx.Graph) -> bool:
    """
    Check basic NAC-coloring existence constraints.

    Check whether graph has no self loops,
    is non-empty, directed and has at least 2 edges.
    Serves as basic input sanitation.

    Raises
    ------
    ValueError if the graph is empty or has loops.

    Return
    ------
    True if a :prf:ref:`NAC-coloring <def-nac>` may exist for the given graph,
    False otherwise.
    """
    if nx.number_of_selfloops(self) > 0:
        raise LookupError()

    if self.nodes() == 0:
        raise ValueError("Undefined for an empty graph")

    if nx.is_directed(self):
        raise ValueError("Cannot process a directed graph")

    if len(nx.edges(self)) < 2:
        return False

    return True


def _can_have_flexible_realization(
    graph: nx.Graph,
) -> bool:
    """
    Check flexible realization existence conditions.

    Use equivalence from :cite:p:`GraseggerLegerskySchicho2019{Thm 3.1}`
    that a graph has a :prf:ref:`NAC-colorings <def-nac>`
    if and only if it has a flexible realization.
    For a flexible realization, the upper bound
    on the number of edges in a graph is known
    :cite:p:`GraseggerLegerskySchicho2019{Thm 4.7}`.

    Definitions
    -----------
    * :prf:ref:`NAC-coloring <def-nac>`
    * :prf:ref:`Flexibility <def-cont-rigid-framework>`
    * :prf:ref:`Realization <def-realization>`

    Parameters
    ----------
    graph:
        A connected graph with at leas one edge.

    Return
    ------
    True if the graph may have :prf:ref:`NAC-colorings <def-nac>`, False otherwise.
    """
    if graph.number_of_edges() <= 1:
        return False
    assert nx.node_connectivity(graph) > 0
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    return m <= n * (n - 1) // 2 - (n - 2)


def _check_for_simple_stable_separating_cut(
    graph: nx.Graph,
) -> NACColoring | None:
    """
    Search for a simple stable separating set.

    If there is a single vertex outside any triangle-connected component,
    we can trivially find a :prf:ref:`NAC-coloring <def-nac>` for the graph
    according to :cite:p:`GraseggerLegerskySchicho2019{Thm 4.4,Cor 4.5}`.

    Definitions
    -----------
    * :prf:ref:`NAC-coloring <def-nac>`
    * :prf:ref:`Triangle-connected components <def-triangle-connected-comp>`
    * :prf:ref:`Stable set <def-stable-set>`
    * :prf:ref:`Separating set <def-separating-set>`


    Parameters
    ----------
    graph:
        The graph to work with, basic a :prf:ref:`NAC-coloring <def-nac>`
        constraints should be already checked.

    Return
    ------
    If no a :prf:ref:`NAC-coloring <def-nac>` can be found using this method,
    None is returned. Otherwise, return a certificate.
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

    # create corresponding NAC-coloring certificate
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

    raise ValueError("NAC-coloring was not found even though it should exist.")


def _check_is_min_rigid_and_NAC_coloring_exists(
    graph: nx.Graph,
) -> bool | None:
    """
    Check for minimally 2-rigid special graph properties.

    For minimally 2-rigid graphs it holds that
    there exists a :prf:ref:`NAC-coloring <def-nac>`
    iff graph is not triangle connected
    according to :cite:p:`ClinchGaramvölgyiEtAl2024{Thm 3.4}`.

    Definitions
    -----------
    * :prf:ref:`Minimal dim-rigidity <def-min-rigid-graph>`
    * :prf:ref:`NAC-coloring <def-nac>`
    * :prf:ref:`2-rigidity <def-gen-rigidity>`

    Parameters
    ----------
    graph:
        The graph to check.

    Return
    ------
    True if the graph has a NAC-coloring,
    False if we are sure there is none and
    None if we cannot decide (the graph is not minimally 2-rigid)
    """

    min_rigid = is_min_rigid(graph, dim=2)
    if not min_rigid:
        return None

    _, components_to_edges = find_mono_classes(graph, MonoClassType.TRI_CONNECTED)
    return len(components_to_edges) != 1


def has_NAC_coloring_checks(graph: nx.Graph) -> bool | None:
    """
    Check whether the graph has a :prf:ref:`NAC-coloring <def-nac>`.

    Implementation for has_NAC_coloring, but without fallback to single_NAC_coloring.
    May be used before an exhaustive search that would not find anything anyway.

    Definitions
    -----------
    * :prf:ref:`NAC-coloring <def-nac>`

    Parameters
    ----------
    graph:
        The graph to check.

    Return
    ------
    True if the graph has a :prf:ref:`NAC-coloring <def-nac>`, False otherwise.
    """
    if not check_NAC_constrains(graph):
        return False

    if _check_for_simple_stable_separating_cut(graph) is not None:
        return True

    if nx.algorithms.connectivity.node_connectivity(graph) < 2:
        return True

    # Needs to be run after connectivity checks
    if not _can_have_flexible_realization(graph):
        return False

    res = _check_is_min_rigid_and_NAC_coloring_exists(graph)
    if res is not None:
        return res

    return None
