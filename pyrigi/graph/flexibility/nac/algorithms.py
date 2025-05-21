from typing import *

import networkx as nx

from pyrigi.graph.flexibility.nac.core import (
    NACColoring,
    coloring_from_mask,
)

from pyrigi.data_type import Edge


def NAC_colorings_naive(
    graph: nx.Graph,
    component_ids: List[int],
    component_to_edges: List[List[Edge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
) -> Iterable[NACColoring]:
    """
    Naive implementation of the basic search algorithm
    """

    # iterate all the coloring variants
    # division by 2 is used as the problem is symmetrical
    for mask in range(1, 2 ** len(component_ids) // 2):
        coloring = coloring_from_mask(
            component_ids,
            component_to_edges,
            mask,
        )

        if not is_NAC_coloring_routine(graph, coloring):
            continue

        yield (coloring[0], coloring[1])
        yield (coloring[1], coloring[0])
