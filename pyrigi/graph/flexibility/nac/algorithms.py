"""
Implement various approaches for :prf:ref:`NAC-coloring <def-nac>` search.

Made in a generic way to also support
:prf:ref:`Cartesian NAC-coloring <def-cartesian-nac>` in the future.
"""

from typing import *

import networkx as nx

from pyrigi.data_type import Edge
from pyrigi.graph.flexibility.nac.core import (
    NACColoring,
    coloring_from_mask,
)


def NAC_colorings_naive(
    graph: nx.Graph,
    class_ids: List[int],
    class_to_edges: List[List[Edge]],
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
