"""
This module is the starting point for the NAC-colorings search.
The main entry point is
the function :func:`pyrigi.graph._flexibility.nac.search.NAC_colorings_impl`.
The given graph is relabeled if the vertices are not integers from 0 to N-1.

Then algorithm argument is parsed and either
Naive algorithm (_NAC_colorings_naive) or
subgraph decomposition algorithm (_NAC_colorings_subgraphs) is chosen.
The according strategies are chosen depending on the given parameters.

The main pair of the search uses only a bit mask representing
the coloring and converts it back to NAC-coloring once a check is need.
"""

import random
from functools import reduce
from typing import Callable, Iterable, Literal, Sequence, cast

import networkx as nx

from pyrigi._utils.repetable_iterator import RepeatableIterator
from pyrigi.graph._flexibility.nac.algorithms import (
    NAC_colorings_cycles,
    NAC_colorings_naive,
    NAC_colorings_subgraphs,
)
from pyrigi.graph._flexibility.nac.check import _is_NAC_coloring_impl
from pyrigi.graph._flexibility.nac.core import (
    IntEdge,
    NAC_colorings_with_non_surjective,
    NACColoring,
    can_have_NAC_coloring,
)
from pyrigi.graph._flexibility.nac.mono_classes import MonoClassType, find_mono_classes


def _NAC_coloring_product(
    first: Iterable[NACColoring], second: Iterable[NACColoring]
) -> Iterable[NACColoring]:
    """
    This makes a Cartesian cross product of two NAC coloring iterators.
    Unlike the Python :func:`~itertools.crossproduct`,
    this adds the colorings inside the tuples.
    """
    cache = RepeatableIterator(first)
    for s in second:
        for f in cache:
            # yield s[0].extend(f[0]), s[1].extend(f[1])
            yield s[0] + f[0], s[1] + f[1]


################################################################################
def _NAC_colorings_from_articulation_points(
    processor: Callable[[nx.Graph], Iterable[NACColoring]],
    graph: nx.Graph,
) -> Iterable[NACColoring]:
    colorings: list[Iterable[NACColoring]] = []
    for component in nx.components.biconnected_components(graph):
        subgraph = nx.induced_subgraph(graph, component)
        iterable = processor(subgraph)
        iterable = NAC_colorings_with_non_surjective(subgraph, iterable)
        colorings.append(iterable)

    iterator = reduce(_NAC_coloring_product, colorings)

    # Skip initial invalid coloring that are not surjective
    iterator = filter(lambda x: len(x[0]) and len(x[1]) != 0, iterator)

    return iterator


################################################################################
def _renamed_coloring(
    ordered_vertices: Sequence[int],
    colorings: Iterable[NACColoring],
) -> Iterable[NACColoring]:
    """
    Rename relabeled vertices in :prf:ref:`NAC-colorings <def-nac>`
    to appropriate names in the original graph.

    Expects graph vertices to be named from 0 to N-1.
    """
    for coloring in colorings:
        yield tuple(
            [(ordered_vertices[u], ordered_vertices[v]) for u, v in group]
            for group in coloring
        )


def _relabel_graph_for_NAC_coloring(
    processor: Callable[[nx.Graph], Iterable[NACColoring]],
    graph: nx.Graph,
) -> Iterable[NACColoring]:
    """
    Relabel vertices of the graph to be named from 0 to N-1.

    Returns
    -------
    processor:
        Lists all :prf:ref:`NAC-colorings <def-nac>` for a given graph.
    graph:
        The graph to relabels and search on.
    seed:
    copy:
        If the graph should be copied before making any destructive changes.
    """
    vertices = list(graph.nodes)

    # no relabeling is needed
    if set(vertices) == set(range(graph.number_of_nodes())):
        return processor(graph)

    mapping = {v: k for k, v in enumerate(vertices)}
    graph = nx.relabel_nodes(graph, mapping, copy=True)

    # rename found NAC-colorings to the original labels
    return _renamed_coloring(vertices, processor(graph))


################################################################################
def _run_algorithm(
    graph: nx.Graph,
    algorithm: str | Literal["naive", "subgraphs"],
    use_cycles_optimization: bool,
    mono_class_type: MonoClassType,
    seed: int | None,
) -> Iterable[NACColoring]:
    """
    Executes specified algorithm
    """
    graph = nx.Graph(graph)
    rand = random.Random(seed)

    # in case graph has no edges because of some previous optimizations,
    # there are no NAC colorings
    if graph.number_of_edges() == 0:
        return []

    # (edge_to_class, class_to_edges)
    _, class_to_edges = cast(
        tuple[dict[IntEdge, int], list[list[IntEdge]]],
        find_mono_classes(
            graph,
            mono_class_type,
        ),
    )
    class_ids = list(range(len(class_to_edges)))

    # this will be replaced by Cartesian NAC-coloring check in the future
    is_NAC_coloring = _is_NAC_coloring_impl

    if algorithm == "naive":
        if not use_cycles_optimization:
            return NAC_colorings_naive(
                graph,
                class_ids,
                class_to_edges,
                is_NAC_coloring,
            )
        else:
            return NAC_colorings_cycles(
                graph,
                class_ids,
                class_to_edges,
                is_NAC_coloring,
            )

    algorithm_parts = list(algorithm.split("-"))
    if algorithm_parts[0] != "subgraphs":
        raise ValueError(f"Unknown algorighm type: {algorithm}")

    if not use_cycles_optimization:
        raise ValueError("Cycles optimization is required for subgraphs algorithm")

    if algorithm == "subgraphs":
        return NAC_colorings_subgraphs(
            graph,
            class_ids,
            class_to_edges,
            is_NAC_coloring,
            seed=rand.randint(0, 2**30),
        )

    return NAC_colorings_subgraphs(
        graph,
        class_ids,
        class_to_edges,
        is_NAC_coloring,
        seed=rand.randint(0, 2**30 - 1),
        split_strategy=algorithm_parts[1],
        merge_strategy=algorithm_parts[2],
        preferred_chunk_size=int(algorithm_parts[3]),
    )


def NAC_colorings_impl(
    graph: nx.Graph,
    algorithm: str | Literal["naive", "subgraphs"],
    use_cycles_optimization: bool,
    use_blocks_decomposition: bool,
    mono_class_type: MonoClassType,
    seed: int | None,
) -> Iterable[NACColoring]:
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

    Suggested improvements
    ----------------------
    Allow running polynomial checks for NAC-coloring existence on startup.
    Reference subgraphs algorithm.
    Evaluate if the copy is needed at the beginning of `run` function.
    """
    if not can_have_NAC_coloring(graph):
        return []

    def apply_processor(
        processor: Callable[[nx.Graph], Iterable[NACColoring]],
        func: Callable[
            [Callable[[nx.Graph], Iterable[NACColoring]], nx.Graph],
            Iterable[NACColoring],
        ],
    ) -> Callable[[nx.Graph], Iterable[NACColoring]]:
        """
        Wraps the core callable with some modifying call
        """
        return lambda g: func(processor, g)

    def run(g: nx.Graph) -> Iterable[NACColoring]:
        return _run_algorithm(
            graph=g,
            algorithm=algorithm,
            use_cycles_optimization=use_cycles_optimization,
            mono_class_type=mono_class_type,
            seed=seed,
        )

    processor: Callable[[nx.Graph], Iterable[NACColoring]] = run

    processor = apply_processor(
        processor,
        lambda p, g: _relabel_graph_for_NAC_coloring(p, g),
    )

    # this has to be run before relabeling, so each block
    # is relabeled 0..N-1 where N is size of the block
    if use_blocks_decomposition:
        processor = apply_processor(
            processor,
            lambda p, g: _NAC_colorings_from_articulation_points(p, g),
        )

    # 1. find blocks
    # 2. relabel graph
    # 3. find NAC-colorings
    return processor(graph)
