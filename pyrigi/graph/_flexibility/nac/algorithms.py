"""
Implement various approaches for :prf:ref:`NAC-coloring <def-nac>` search.

Made in a generic way to also support
:prf:ref:`Cartesian NAC-coloring <def-cartesian-nac>` in the future.
"""

import itertools
import math
import random
from typing import Callable, Iterable, Iterator, Literal

import networkx as nx

from pyrigi._utils.repetable_iterator import RepeatableIterator
from pyrigi.graph._flexibility.nac.core import (
    IntEdge,
    NACColoring,
    SubgraphColorings,
    coloring_from_mask,
    create_bitmask_for_class_graph_cycle,
    mask_matches_templates,
    mask_to_vertices,
)
from pyrigi.graph._flexibility.nac.cycle_detection import find_cycles
from pyrigi.graph._flexibility.nac.strategies import (
    linear,
    shared_vertices,
    subgraphs_strategy_neighbors,
)


def NAC_colorings_naive(
    graph: nx.Graph,
    class_ids: list[int],
    class_to_edges: list[list[IntEdge]],
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
        Mapping from class ID to its edges.
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
    classes_ids: list[int],
    class_to_edges: list[list[IntEdge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
) -> Iterable[NACColoring]:
    """
    Implementation of the naive algorithm improved by using cycles.

    Parameters
    ----------
    graph:
        The graph to search on.
    classes_ids:
        List of classes IDs.
    class_to_edges:
        Mapping from class ID to its edges.
    is_NAC_coloring_routine:
        Used to check if the coloring is :prf:ref:`NAC-coloring <def-nac>`
        or :prf:ref:`Cartesian NAC-coloring <def-cartesian-nac>`.
    """
    # so we start with 0
    classes_ids.sort()

    # find some small cycles for state filtering
    cycles = find_cycles(
        graph,
        set(classes_ids),
        class_to_edges,
    )
    # the idea is that smaller cycles reduce the state space more
    cycles = sorted(cycles, key=lambda c: len(c))

    # bit mask templates representing cycles
    templates = [
        create_bitmask_for_class_graph_cycle(graph, class_to_edges.__getitem__, c)
        for c in cycles
    ]
    templates = [t for t in templates if t[1] > 0]

    # this is used for mask inversion, because how ~ works on python
    # numbers, if we used some kind of bit arrays,
    # this would not be needed.
    subgraph_mask = 0  # 2 ** len(class_ids) - 1
    for v in classes_ids:
        subgraph_mask |= 1 << v

    # iterate all the coloring variants
    # division by 2 is used as the problem is symmetrical
    for mask in range(1, 2 ** len(classes_ids) // 2):
        if mask_matches_templates(templates, mask, subgraph_mask):
            continue

        coloring = coloring_from_mask(
            classes_ids,
            class_to_edges,
            mask,
        )

        if not is_NAC_coloring_routine(graph, coloring):
            continue

        yield (coloring[0], coloring[1])
        yield (coloring[1], coloring[0])


################################################################################
def _subgraph_colorings_generator(
    graph: nx.Graph,
    class_to_edges: list[list[IntEdge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
    ordered_class_ids: list[int],
    chunk_size: int,
    offset: int,
) -> Iterable[int]:
    """
    Find all :prf:ref:`NAC-colorings <def-nac>` for the given subgraph
    using naive approaches with cycles optimizations.

    Parameters
    ----------
    graph:
        The graph to search on.
    class_to_edges:
        A list of edges for each class in the class graph.
    is_NAC_coloring_routine:
        A function that checks if the given coloring is a NAC coloring.
    ordered_class_ids:
        List of classes IDs.
    chunk_size:
        Size of the subgraph to search on in :prf:ref:`NAC-mono classes <def-nac-mono>`
        in `ordered_class_ids`.
    offset:
        Offset in `ordered_class_ids` of the subgraph to search on.

    Yields
    ------
    All :prf:ref:`NAC-colorings <def-nac>` of a subgraph.
    """
    # The last chunk can be smaller
    local_ordered_class_ids: list[int] = ordered_class_ids[offset : offset + chunk_size]

    local_cycles = find_cycles(
        graph,
        local_ordered_class_ids,
        class_to_edges,
    )

    # local -> first chunk_size vertices
    mapping = {x: i for i, x in enumerate(local_ordered_class_ids)}

    def mapped_edge_classes(ind: int) -> list[IntEdge]:
        return class_to_edges[local_ordered_class_ids[ind]]

    local_cycles = (tuple(mapping[c] for c in cycle) for cycle in local_cycles)
    templates = [
        create_bitmask_for_class_graph_cycle(graph, mapped_edge_classes, cycle)
        for cycle in local_cycles
    ]
    templates = [t for t in templates if t[1] > 0]

    subgraph_mask = 2 ** len(local_ordered_class_ids) - 1
    for mask in range(0, 2**chunk_size // 2):
        if mask_matches_templates(templates, mask, subgraph_mask):
            continue

        coloring = coloring_from_mask(
            local_ordered_class_ids,
            class_to_edges,
            mask,
        )

        if not is_NAC_coloring_routine(graph, coloring):
            continue

        yield mask << offset


def _subgraphs_join_epochs(
    graph: nx.Graph,
    class_to_edges: list[list[IntEdge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
    ordered_class_ids: list[int],
    epoch1: Iterable[int],
    subgraph_mask_1: int,
    epoch2: RepeatableIterator,
    subgraph_mask_2: int,
) -> Iterable[int]:
    """
    Join :prf:ref:`NAC-colorings <def-nac>` of two edge disjoined subgraphs.

    This function works by taking pairs of subgraph NAC-colorings,
    joining them into red-blue-colorings on the merged subgraph,
    and checking they are :prf:ref:`NAC-coloring <def-nac>`.
    Monochromatic colorings are also included in the output.

    Parameters
    ----------
    graph:
        The graph to search on.
    class_to_edges:
        A list of edges for each class in the class graph.
    is_NAC_coloring_routine:
        A function that checks if the given coloring is a NAC coloring.
    ordered_class_ids:
        List of classes IDs.
    epoch1:
        Iterator of :prf:ref:`NAC-colorings <def-nac>` of the first subgraph.
    subgraph_mask_1:
        Mask of the first subgraph.
    epoch2:
        Iterator of :prf:ref:`NAC-colorings <def-nac>` of the second subgraph.
    subgraph_mask_2:
        Mask of the second subgraph.

    Yields
    ------
    All :prf:ref:`NAC-colorings <def-nac>` of the merged subgraph.

    Suggested Improvements
    ---------------------
    Benchmark again lazy iterator. Current implementation gets
    the first NAC-colorings and then all NAC-colorings
    of one subgraph are iterated and tried with the other NAC-coloring.
    Alternative lazy approach gets a NAC-coloring from one subgraph
    and tries to join it with all already known NAC-colorings of the other subgraph.
    In the next round, NAC-coloring is taken from the other subgraph.
    This performed worse in our tests for minimally-rigid graphs,
    but was not test different graph classes.
    It may be worth exploring this again in the future.
    """

    if subgraph_mask_1 & subgraph_mask_2:
        raise ValueError("Cannot join two subgraphs with common nodes")

    subgraph_mask = subgraph_mask_1 | subgraph_mask_2

    local_ordered_class_ids: list[int] = []

    # maps local vertex → global index
    mapping: dict[int, int] = {}

    for i, v in enumerate(ordered_class_ids):
        if (1 << i) & subgraph_mask:
            mapping[v] = i
            local_ordered_class_ids.append(v)

    local_cycles = find_cycles(
        graph,
        local_ordered_class_ids,
        class_to_edges,
    )

    def mapped_edges_classes(ind: int) -> list[IntEdge]:
        return class_to_edges[ordered_class_ids[ind]]

    # cycles with indices of the class IDs in the global order
    local_cycles = [tuple(mapping[c] for c in cycle) for cycle in local_cycles]
    templates = [
        create_bitmask_for_class_graph_cycle(graph, mapped_edges_classes, cycle)
        for cycle in local_cycles
    ]
    templates = [t for t in templates if t[1] > 0]

    for mask1 in epoch1:
        for mask2 in epoch2:
            mask = mask1 | mask2

            if mask_matches_templates(templates, mask, subgraph_mask):
                continue

            coloring = coloring_from_mask(
                ordered_class_ids,
                class_to_edges,
                mask,
                subgraph_mask,
            )

            if not is_NAC_coloring_routine(graph, coloring):
                continue

            yield mask


def _apply_split_strategy_to_order_vertices(
    graph: nx.Graph,
    class_ids: list[int],
    class_to_edges: list[list[IntEdge]],
    chunk_sizes: list[int],
    order_strategy: str,
    seed: int,
) -> tuple[list[int], list[int]]:
    match order_strategy:
        case "none":
            ordered_class_ids = class_ids
        case "neighbors" | "neighbors_degree":
            ordered_class_ids = subgraphs_strategy_neighbors(
                graph=graph,
                class_ids=class_ids,
                class_to_edges=class_to_edges,
                chunk_sizes=chunk_sizes,
                use_degree=order_strategy == "neighbors_degree",
                seed=seed,
            )
        case _:
            raise ValueError(f"Unknown split strategy: {order_strategy}")
    return ordered_class_ids, chunk_sizes


################################################################################
def _colorings_merge(
    graph: nx.Graph,
    class_to_edges: list[list[IntEdge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
    ordered_class_ids: list[int],
    colorings_1: SubgraphColorings,
    colorings_2: SubgraphColorings,
) -> SubgraphColorings:
    """
    Merge :prf:ref:`NAC-colorings <def-nac>` of two given subgraphs.

    Parameters
    ----------
    graph:
        The graph to search on.
    class_to_edges:
        Maps a :prf:ref:`NAC-mono class<def-nac-mono>` to its edges.
    is_NAC_coloring_routine:
        The NAC-coloring routine to use.
    ordered_class_ids:
        List of :prf:ref:`NAC-mono classes<def-nac-mono>`
        corresponding to bit masks.
    colorings_1:
        :prf:ref:`NAC-colorings<def-nac>` of
        the first subgraph's coloring and subgraph mask.
    colorings_2:
        :prf:ref:`NAC-colorings<def-nac>` of
        the second subgraph's coloring and subgraph mask.

    Returns
    -------
    All :prf:ref:`NAC-colorings <def-nac>` of the merged subgraph
    and its subgraph mask.

    Suggested improvements
    ---------------------
    Check if `epoch2_switched` is not causing major performance regression
    when all NAC-colorings of the second subgraph are always iterated.
    """
    (epoch1, subgraph_mask_1) = colorings_1
    (epoch2, subgraph_mask_2) = colorings_2
    epoch1 = RepeatableIterator(epoch1)
    epoch2 = RepeatableIterator(epoch2)
    epoch2_switched = RepeatableIterator(
        # this has to be list, so the iterator is exhausted and not iterated concurrently
        [coloring ^ subgraph_mask_2 for coloring in epoch2]
    )

    vertices_1 = mask_to_vertices(ordered_class_ids, class_to_edges, colorings_1[1])
    vertices_2 = mask_to_vertices(ordered_class_ids, class_to_edges, colorings_2[1])

    if len(vertices_1.intersection(vertices_2)) <= 1:

        def generator() -> Iterator[int]:
            for c1 in epoch1:
                for c2, c2s in zip(epoch2, epoch2_switched):
                    yield c1 | c2
                    yield c1 | c2s

        return SubgraphColorings(
            generator(),
            subgraph_mask_1 | subgraph_mask_2,
        )

    # if at least two vertices are shared, we need to do the full check
    return SubgraphColorings(
        itertools.chain(
            _subgraphs_join_epochs(
                graph,
                class_to_edges,
                is_NAC_coloring_routine,
                ordered_class_ids,
                epoch1,
                subgraph_mask_1,
                epoch2,
                subgraph_mask_2,
            ),
            _subgraphs_join_epochs(
                graph,
                class_to_edges,
                is_NAC_coloring_routine,
                ordered_class_ids,
                epoch1,
                subgraph_mask_1,
                epoch2_switched,
                subgraph_mask_2,
            ),
        ),
        subgraph_mask_1 | subgraph_mask_2,
    )


def _apply_merge_strategy(
    graph: nx.Graph,
    class_to_edges: list[list[IntEdge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
    ordered_class_ids: list[int],
    merge_strategy: str,
    all_epochs: list[SubgraphColorings],
) -> list[SubgraphColorings]:
    def colorings_merge_wrapper(
        colorings_1: SubgraphColorings,
        colorings_2: SubgraphColorings,
    ) -> SubgraphColorings:
        return _colorings_merge(
            graph=graph,
            class_to_edges=class_to_edges,
            is_NAC_coloring_routine=is_NAC_coloring_routine,
            ordered_class_ids=ordered_class_ids,
            colorings_1=colorings_1,
            colorings_2=colorings_2,
        )

    match merge_strategy:
        case "linear":
            return linear(
                colorings_merge_wrapper=colorings_merge_wrapper,
                all_epochs=all_epochs,
            )
        case "shared_vertices":
            return shared_vertices(
                class_to_edges=class_to_edges,
                ordered_class_ids=ordered_class_ids,
                colorings_merge_wrapper=colorings_merge_wrapper,
                all_epochs=all_epochs,
            )
        case _:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")


################################################################################
def NAC_colorings_subgraphs(
    graph: nx.Graph,
    class_ids: list[int],
    class_to_edges: list[list[IntEdge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
    seed: int,
    split_strategy: Literal["none", "neighbors", "neighbors_degree"] = "neighbors",
    merge_strategy: Literal["linear", "shared_vertices"] = "linear",
    preferred_chunk_size: int = 5,
) -> Iterable[NACColoring]:
    """
    This version of the algorithm splits the graphs into subgraphs,
    find :prf:ref:`NAC-colorings <def-nac>` for each of them.
    The subgraphs are then merged,
    and new colorings are reevaluated till we reach the original graph again.

    Parameters
    ----------
    graph:
        The original graph.
    class_ids:
        List of classes IDs.
    class_to_edges:
        List of edges for each class.
    is_NAC_coloring_routine:
        The function to check if a coloring is a :prf:ref:`NAC-coloring <def-nac>`.
    seed:
        Random seed used by some strategies.
    split_strategy:
        The strategy to split the graph into subgraphs.
    merge_strategy:
        The strategy to merge the subgraphs.
    preferred_chunk_size:
        The preferred size of the chunks.
    """
    rand = random.Random(seed)
    classes_no = len(class_ids)

    preferred_chunk_size = min(preferred_chunk_size, classes_no)
    assert preferred_chunk_size >= 1

    def create_chunk_sizes() -> list[int]:
        """
        Makes sure all the chunks are the same size of 1 bigger
        """
        chunk_no = classes_no // preferred_chunk_size
        chunk_sizes = []
        remaining_len = classes_no
        for _ in range(chunk_no):
            # ceiling floats, scary
            chunk_sizes.append(
                min(
                    math.ceil(remaining_len / (chunk_no - len(chunk_sizes))),
                    remaining_len,
                )
            )
            remaining_len -= chunk_sizes[-1]
        return chunk_sizes

    chunk_sizes = create_chunk_sizes()

    ordered_class_ids, chunk_sizes = _apply_split_strategy_to_order_vertices(
        graph=graph,
        class_ids=class_ids,
        class_to_edges=class_to_edges,
        chunk_sizes=chunk_sizes,
        order_strategy=split_strategy,
        seed=rand.randint(0, 2**30),
    )

    assert classes_no == len(ordered_class_ids)

    # Holds all the NAC colorings for a subgraph represented by the second bit mask
    all_epochs: list[SubgraphColorings] = []
    # Number of classes already processed in previous chunks
    offset = 0
    for chunk_size in chunk_sizes:
        subgraph_mask = 2**chunk_size - 1
        all_epochs.append(
            SubgraphColorings(
                _subgraph_colorings_generator(
                    graph,
                    class_to_edges,
                    is_NAC_coloring_routine,
                    ordered_class_ids,
                    chunk_size,
                    offset,
                ),
                subgraph_mask << offset,
            )
        )
        offset += chunk_size

    all_epochs = _apply_merge_strategy(
        graph=graph,
        class_to_edges=class_to_edges,
        is_NAC_coloring_routine=is_NAC_coloring_routine,
        ordered_class_ids=ordered_class_ids,
        merge_strategy=merge_strategy,
        all_epochs=all_epochs,
    )

    assert len(all_epochs) == 1
    expected_subgraph_mask = 2**classes_no - 1
    assert expected_subgraph_mask == all_epochs[0][1]

    for mask in all_epochs[0][0]:
        if mask == 0 or mask.bit_count() == len(ordered_class_ids):
            continue

        coloring = coloring_from_mask(
            ordered_class_ids,
            class_to_edges,
            mask,
        )

        yield (coloring[0], coloring[1])
        yield (coloring[1], coloring[0])
