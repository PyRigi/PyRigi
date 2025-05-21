from typing import *

from pyrigi.data_type import Edge

NACColoring: TypeAlias = Tuple[Collection[Edge], Collection[Edge]]


def coloring_from_mask(
    ordered_comp_ids: List[int],
    component_to_edges: List[List[Edge]],
    mask: int,
    allow_mask: int | None = None,
) -> NACColoring:
    """
    Converts a mask representing a red-blue edge coloring.

    Parameters
    ----------
    ordered_comp_ids:
        list of component IDs, mask points into it
    component_to_edges:
        mapping from component ID to its edges
    mask:
        bit mask pointing into ordered_comp_ids,
        1 means red and 0 blue (or otherwise)
    allow_mask:
        mask allowing only some components.
        Used when generating coloring for subgraph.
    """

    if allow_mask is None:
        allow_mask = 2 ** len(ordered_comp_ids) - 1

    red, blue = [], []  # set(), set()
    for i, e in enumerate(ordered_comp_ids):
        address = 1 << i

        if address & allow_mask == 0:
            continue

        edges = component_to_edges[e]
        # (red if mask & address else blue).update(edges)
        (red if mask & address else blue).extend(edges)
    return (red, blue)
