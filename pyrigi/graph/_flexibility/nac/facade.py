from typing import Iterable

import networkx as nx

import pyrigi.graph._utils._input_check as _graph_input_check
from pyrigi.data_type import Edge
from pyrigi.graph._flexibility.nac.core import NACColoring
from pyrigi.graph._flexibility.nac.mono_classes import MonoClassType
from pyrigi.graph._flexibility.nac.search import NAC_colorings_impl
from pyrigi.graph._flexibility.nac.single import (
    has_NAC_coloring_impl,
    single_NAC_coloring_impl,
)

_DEFAULT_ALGORITHM = "subgraphs-neighbors-linear-5"


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
    * :prf:ref:`NAC-coloring <def-nac>`

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
    _check_input_graph_for_NAC_coloring(graph)

    if algorithm == "default":
        algorithm = _DEFAULT_ALGORITHM

    yield from map(
        _coloring_map,
        NAC_colorings_impl(
            graph=graph,
            algorithm=algorithm,
            use_cycles_optimization=use_cycles_optimization,
            use_blocks_decomposition=use_blocks_decomposition,
            mono_class_type=MonoClassType.from_string(mono_class_type),
            seed=seed,
        ),
    )


def has_NAC_coloring(
    graph: nx.Graph,
    algorithm: str = "default",
    use_cycles_optimization: bool = True,
    mono_class_type: str = "triangle-extended",
    seed: int | None = 42,
) -> bool:
    """
    Return if the graph has a NAC-coloring.

    Same as :func:`~.single_NAC_coloring`,
    but no certificate of existence is provided.

    See :func:`~.NAC_colorings` for parameter descriptions.

    Definitions
    -----------
    * :prf:ref:`NAC-coloring <def-nac>`
    """
    _check_input_graph_for_NAC_coloring(graph)

    if algorithm == "default":
        algorithm = _DEFAULT_ALGORITHM

    return has_NAC_coloring_impl(
        graph,
        algorithm=algorithm,
        use_cycles_optimization=use_cycles_optimization,
        mono_class_type=MonoClassType.from_string(mono_class_type),
        seed=seed,
    )


def single_NAC_coloring(
    graph: nx.Graph,
    algorithm: str = "default",
    use_cycles_optimization: bool = True,
    mono_class_type: str = "triangle-extended",
    seed: int | None = 42,
) -> tuple[list[Edge], list[Edge]] | None:
    """
    Return a single NAC-coloring.

    If no NAC-coloring exists, ``None`` is returned.
    Some polynomial time checks are run.
    If they fail, an exhaustive search is run.

    See :func:`~.NAC_colorings` for parameter descriptions.

    Definitions
    -----------
    * :prf:ref:`NAC-coloring <def-nac>`
    """
    _check_input_graph_for_NAC_coloring(graph)

    if algorithm == "default":
        algorithm = _DEFAULT_ALGORITHM

    res = single_NAC_coloring_impl(
        graph,
        algorithm=algorithm,
        use_cycles_optimization=use_cycles_optimization,
        mono_class_type=MonoClassType.from_string(mono_class_type),
        seed=seed,
    )
    if res is not None:
        res = _coloring_map(res)
    return res


def _coloring_map(coloring: NACColoring) -> tuple[list[Edge], list[Edge]]:
    return list(coloring[0]), list(coloring[1])


def _check_input_graph_for_NAC_coloring(graph: nx.Graph):
    _graph_input_check.no_loop(graph)
