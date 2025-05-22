"""
Implement various approaches for :prf:ref:`NAC-coloring <def-nac>` search.

Made in a generic way to also support
:prf:ref:`Cartesian NAC-coloring <def-cartesian-nac>` in the future.
"""

from typing import Callable, Iterable
import networkx as nx

from pyrigi.data_type import Edge
from pyrigi.graph._flexibility.nac.core import (
    NACColoring,
    coloring_from_mask,
    create_bitmask_for_component_graph_cycle,
    mask_matches_templates,
)
from pyrigi.graph._flexibility.nac.cycle_detection import find_cycles


def NAC_colorings_naive(
    graph: nx.Graph,
    class_ids: list[int],
    class_to_edges: list[list[Edge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
) -> Iterable[NACColoring]:
    """
    Naive implementation of the basic :prf:ref:`NAC-coloring <def-nac>`
    search algorithm.

    Parameters
    ----------
    graph:
        The graph to search on.
    class_ids:
        List of classes IDs.
    class_to_edges:
        Mapping from component ID to its edges.
    is_NAC_coloring_routine:
        Used to check if the coloring is :prf:ref:`NAC-coloring <def-nac>`
        or :prf:ref:`Cartesian NAC-coloring <def-cartesian-nac>`.

    Yields
    ------
    All colorings satisfying `is_NAC_coloring_routine`.
    """

    # iterate all the coloring variants
    # division by 2 is used as the problem is symmetrical
    for mask in range(1, 2 ** len(class_ids) // 2):
        coloring = coloring_from_mask(
            class_ids,
            class_to_edges,
            mask,
        )

        if not is_NAC_coloring_routine(graph, coloring):
            continue

        yield (coloring[0], coloring[1])
        yield (coloring[1], coloring[0])


def NAC_colorings_cycles(
    graph: nx.Graph,
    components_ids: list[int],
    component_to_edges: list[list[Edge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
) -> Iterable[NACColoring]:
    """
    Implementation of the naive algorithm improved by using cycles.

    Parameters
    ----------
    graph:
        The graph to search on.
    components_ids:
        List of components IDs.
    component_to_edges:
        Mapping from component ID to its edges.
    is_NAC_coloring_routine:
        Used to check if the coloring is :prf:ref:`NAC-coloring <def-nac>`
        or :prf:ref:`Cartesian NAC-coloring <def-cartesian-nac>`.
    """
    # so we start with 0
    components_ids.sort()

    # find some small cycles for state filtering
    cycles = find_cycles(
        graph,
        set(components_ids),
        component_to_edges,
    )
    # the idea is that smaller cycles reduce the state space more
    cycles = sorted(cycles, key=lambda c: len(c))

    # bit mask templates representing cycles
    templates = [
        create_bitmask_for_component_graph_cycle(
            graph, component_to_edges.__getitem__, c
        )
        for c in cycles
    ]
    templates = [t for t in templates if t[1] > 0]

    # this is used for mask inversion, because how ~ works on python
    # numbers, if we used some kind of bit arrays,
    # this would not be needed.
    subgraph_mask = 0  # 2 ** len(components_ids) - 1
    for v in components_ids:
        subgraph_mask |= 1 << v

    # iterate all the coloring variants
    # division by 2 is used as the problem is symmetrical
    for mask in range(1, 2 ** len(components_ids) // 2):
        if mask_matches_templates(templates, mask, subgraph_mask):
            continue

        coloring = coloring_from_mask(
            components_ids,
            component_to_edges,
            mask,
        )

        if not is_NAC_coloring_routine(graph, coloring):
            continue

        yield (coloring[0], coloring[1])
        yield (coloring[1], coloring[0])
