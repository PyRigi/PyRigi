"""
The module contains functions related to converting from and to
:prf:ref:`NAC-mono <def-nac-mono>` classes represented by bit masks
to sets of edges representing a :prf:ref:`NAC-coloring <def-nac>`.
"""

from typing import *

from pyrigi.data_type import Edge

"""
Represents a :prf:ref:`NAC-coloring <def-nac>`.
Meant for internal use only as :meth:`~frozenset`
should be exposed to the user instead.
Tuple is faster for internal use.
"""
NACColoring: TypeAlias = Tuple[Collection[Edge], Collection[Edge]]


def coloring_from_mask(
    ordered_comp_ids: List[int],
    component_to_edges: List[List[Edge]],
    mask: int,
    allow_mask: int | None = None,
) -> NACColoring:
    """
    Convert a mask representing a red-blue edge coloring.

    Parameters
    ----------
    ordered_comp_ids:
        List of component IDs, mask's bits point into it.
    component_to_edges:
        Mapping from component ID to its edges.
    mask:
        Bit mask pointing into `ordered_comp_ids`,
        1 means red and 0 blue (or otherwise).
    allow_mask:
        Mask allowing only some components.
        Used for subgraphs.
    """

    if allow_mask is None:
        allow_mask = 2 ** len(ordered_comp_ids) - 1

    red, blue = [], []  # set(), set()
    for i, e in enumerate(ordered_comp_ids):
        address = 1 << i

        if address & allow_mask == 0:
            continue

        edges = component_to_edges[e]
        (red if mask & address else blue).extend(edges)
    return (red, blue)
