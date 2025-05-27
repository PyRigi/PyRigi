from typing import Container, Iterable, Literal

import networkx as nx

from pyrigi.data_type import Edge
from pyrigi.graph._flexibility.nac.core import NACColoring
from pyrigi.graph._flexibility.nac.mono_classes import MonochromaticClassType
from pyrigi.graph._flexibility.nac.search import NAC_colorings_impl


def NAC_colorings(
    graph: nx.Graph,
    algorithm: str | Literal["naive", "subgraphs"] = "subgraphs",
    use_cycles_optimization: bool = True,
    use_blocks_decomposition: bool = True,
    mono_class_type: MonoClassType = MonoClassType.TRI_EXTENDED,
    seed: int | None = 42,
) -> Iterable[tuple[Container[Edge], Container[Edge]]]:
    """
    Find all :prf:ref:`NAC-colorings <def-nac>` of the given graph.

    Parameters
    ----------
    self:
        The graph to search on.
    algorithm:
        The algorithm to use.
        The options are `naive` for the naive approach and
        `subgraphs` for the subgraphs decomposition approach.
        Strategies can be specified for the subgraphs algorithm
        as follows: `subgraphs-{split_strategy}-{merging_stragey}-{subgraphs_size}`.
        Split strategies are `none`, `neighbors`, and `neighbors-degree`,
        merging strategies are `linear` and `shared_vertices`.
        See docs for further details.
    use_cycles_optimization:
        Use cycles optimization for the given algorithm.
        This is always enabled for subgraphs strategies.
    use_decompositions:
        If enabled, graph is first decomposed into blocks,
        and :prf:ref:`NAC-colorings <def-nac>` are found for each
        block (:prf:ref:`2-vertex connected component <def-k-connected>`)
        separately and then combined.
    mono_class_type:
        The type of :prf:ref:`NAC-mono classes <def-nac-mono>` to use.
    seed:
        The seed to use for randomization.

    Yield
    -----
    All :prf:ref:`NAC-colorings <def-nac>` for a given graph.
    """

    def coloring_map(coloring: NACColoring) -> tuple[Container[Edge], Container[Edge]]:
        return coloring

    yield from map(
        coloring_map,
        NAC_colorings_impl(
            graph=graph,
            algorithm=algorithm,
            use_cycles_optimization=use_cycles_optimization,
            use_blocks_decomposition=use_blocks_decomposition,
            mono_class_type=mono_class_type,
            seed=seed,
        ),
    )
