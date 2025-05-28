from typing import Iterable

import networkx as nx

from pyrigi.data_type import Edge
from pyrigi.graph._flexibility.nac.core import NACColoring
from pyrigi.graph._flexibility.nac.mono_classes import MonoClassType
from pyrigi.graph._flexibility.nac.search import NAC_colorings_impl


def NAC_colorings(
    graph: nx.Graph,
    algorithm: str = "default",
    use_cycles_optimization: bool = True,
    use_blocks_decomposition: bool = True,
    mono_class_type: str = "triangle-extended",
    seed: int | None = 42,
) -> Iterable[tuple[list[Edge], list[Edge]]]:
    """
    Find all NAC-colorings of the graph.

    Definitions
    -----------
    :prf:ref:`NAC-coloring <def-nac>`

    Parameters
    ----------
    algorithm:
        The algorithm to use.
        The options are ``"naive"`` for the naive approach and
        ``"subgraphs"`` for the subgraphs decomposition approach,
        see :ref:`nac-computation`.
        Strategies can be specified for the subgraphs algorithm
        as follows: ``"subgraphs-<split_strategy>-<merging_strategy>-<subgraphs_size>"``.
        Split strategies are ``none``, ``neighbors``, and ``neighbors_degree``,
        merging strategies are ``linear`` and ``shared_vertices``.
        The default strategy is ``"subgraphs-neighbors-linear-5"``.
    use_cycles_optimization:
        Use cycles optimization for the given algorithm.
        This is always enabled for subgraphs strategies.
    use_blocks_decomposition:
        If enabled, the graph is first decomposed into blocks,
        and :prf:ref:`NAC-colorings <def-nac>` are found for each
        block (:prf:ref:`2-vertex connected component <def-k-connected>`)
        separately and then combined.
    mono_class_type:
        The type of :prf:ref:`NAC-mono classes <def-nac-mono>` to use.
        The options are ``"edges"`` (each edge is a NAC-mono class),
        ``"triangle"``
        for :prf:ref:`triangle-connected components <def-triangle-connected-comp>`,
        or ``"triangle-extended"`` (default) for
        :prf:ref:`triangle-extended classes <def-triangle-extended-class>`.
    seed:
        The seed to use for randomization.
    """

    def coloring_map(coloring: NACColoring) -> tuple[list[Edge], list[Edge]]:
        return list(coloring[0]), list(coloring[1])

    if algorithm == "default":
        algorithm = "subgraphs-neighbors-linear-5"

    yield from map(
        coloring_map,
        NAC_colorings_impl(
            graph=graph,
            algorithm=algorithm,
            use_cycles_optimization=use_cycles_optimization,
            use_blocks_decomposition=use_blocks_decomposition,
            mono_class_type=MonoClassType.from_string(mono_class_type),
            seed=seed,
        ),
    )
