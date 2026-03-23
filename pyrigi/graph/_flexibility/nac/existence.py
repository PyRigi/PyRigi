"""
This module checks whether a graph can have a :prf:ref:`NAC-colorings <def-nac>`.
"""

import networkx as nx

from pyrigi.graph._flexibility.nac.core import NACColoring, can_have_NAC_coloring
from pyrigi.graph._flexibility.nac.mono_classes import MonoClassType, find_mono_classes
from pyrigi.graph._rigidity.generic import is_min_rigid


def _check_for_vertex_out_of_3_cycle(
    graph: nx.Graph,
) -> NACColoring | None:
    """
    Search for a vertex with stable neighborhood.

    A certificate NAC-coloring is returned
    if it can be constructed from the stable neighborhood of a vertex,
    ``None`` otherwise.

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

        if len(red) == 0 or len(blue) == 0:
            raise RuntimeError(
                "Construction of a NAC-coloring from a stable separating set failed."
            )

        return (red, blue)

    raise RuntimeError("NAC-coloring was not found even though it should exist.")


def _check_is_min_rigid_and_NAC_coloring_exists(
    graph: nx.Graph,
) -> bool | None:
    """
    Check if a NAC-coloring exists for minimally 2-rigid graphs.

    If the graph is not minimally 2-rigid, ``None`` is returned.
    Otherwise, the existence is decided using the following:
    for minimally 2-rigid graphs it holds that
    there exists a :prf:ref:`NAC-coloring <def-nac>`
    iff the graph is not triangle connected
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
    """

    min_rigid = is_min_rigid(graph, dim=2)
    if not min_rigid:
        return None

    _, components_to_edges = find_mono_classes(graph, MonoClassType.TRI_CONNECTED)
    return len(components_to_edges) != 1


def has_NAC_coloring_checks(graph: nx.Graph) -> bool | None:
    """
    Return whether the graph has a NAC-coloring.

    Implementation of has_NAC_coloring, but without falling back to single_NAC_coloring.
    May be used before an exhaustive search that would not find anything anyway.

    Definitions
    -----------
    * :prf:ref:`NAC-coloring <def-nac>`

    Parameters
    ----------
    graph:
        The graph to check.
    """
    if graph.number_of_edges() <= 1:
        return False

    if _check_for_vertex_out_of_3_cycle(graph) is not None:
        return True

    if nx.algorithms.connectivity.node_connectivity(graph) < 2:
        return True

    # Needs to be run after connectivity checks
    if not can_have_NAC_coloring(graph):
        return False

    res = _check_is_min_rigid_and_NAC_coloring_exists(graph)
    if res is not None:
        return res

    return None
