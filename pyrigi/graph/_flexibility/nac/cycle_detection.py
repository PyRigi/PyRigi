"""
This module contains code related to finding cycles
that should correspond to :prf:ref:`NAC-mono classes <def-nac-mono>`.
"""

from collections import defaultdict
from typing import Collection

import networkx as nx


def find_cycles(
    graph: nx.Graph,
    subgraph_class_IDs: Collection[int],
    class_to_edges: Collection[Collection[tuple[int, int]]],
    per_class_limit: int = 2,
) -> set[tuple[int, ...]]:
    """
    For each edge, find all the cycles among
    :prf:ref:`NAC-mono classes <def-nac-mono>`
    of length at most five (length is the number of classes it spans).

    Not all the returned cycles are guaranteed to be actual cycles as
    this may create cycles that enter and exit a class at the same vertex.

    Parameters
    ----------
    graph:
        Input graph.
    subgraph_class_IDs:
        Classes of the subgraph.
    class_to_edges:
        A list of edges for each class.
    per_class_limit:
        The maximum number of a cycle to be returned per class.
    """
    cycles = _find_useful_cycles_for_class(
        graph=graph,
        subgraph_class_IDs=subgraph_class_IDs,
        class_to_edges=class_to_edges,
        per_class_limit=per_class_limit,
    )
    return {c for class_cycles in cycles.values() for c in class_cycles}


def _find_useful_cycles_for_class(
    graph: nx.Graph,
    subgraph_class_IDs: Collection[int],
    class_to_edges: Collection[Collection[tuple[int, int]]],
    per_class_limit: int = 2,
) -> dict[int, set[tuple[int, ...]]]:
    """
    Same as :class:`~pyrigi.graph._framework.nac.cycle_detection.find_cycles`
    except the results are grouped for each class.
    """
    classes_no = len(class_to_edges)

    # creates mapping from vertex to set of monochromatic classes if is in
    vertex_to_classes = [set() for _ in range(max(graph.nodes) + 1)]
    for class_id, class_edges in enumerate(class_to_edges):
        if class_id not in subgraph_class_IDs:
            continue
        for u, v in class_edges:
            vertex_to_classes[u].add(class_id)
            vertex_to_classes[v].add(class_id)
    neighboring_classes = [set() for _ in range(classes_no)]

    found_cycles: dict[int, set[tuple[int, ...]]] = defaultdict(set)

    # create a graph where vertices are monochromatic classes as there's
    # an edge if the monochromatic classes share a vertex
    for v in graph.nodes:
        for i in vertex_to_classes[v]:
            for j in vertex_to_classes[v]:
                if i != j:
                    neighboring_classes[i].add(j)

    _find_useful_cycles_for_edges(
        graph,
        vertex_to_classes,
        neighboring_classes,
        found_cycles,
    )

    limited = {}
    for key, value in found_cycles.items():
        limited[key] = set(list(sorted(value, key=len))[:per_class_limit])

    return limited


def _find_useful_cycles_for_edges(
    graph: nx.Graph,
    vertex_to_classes: list[set[int]],
    neighboring_classes: list[set[int]],
    found_cycles: dict[int, set[tuple[int, ...]]],
):
    """
    Find three, four and five cycles in class graph
    for each edge in the original graph.
    """
    for u, v in graph.edges:
        u_classes = vertex_to_classes[u]
        v_classes = vertex_to_classes[v]

        # remove shared classes
        intersection = u_classes.intersection(v_classes)
        u_classes = u_classes - intersection
        v_classes = v_classes - intersection

        # this check only makes sense for proper NAC-mono classes,
        # triangle-connected classes fail on it
        # assert len(intersection) <= 1

        if len(intersection) == 0:
            continue

        for u_class in u_classes:
            # triangles
            for n in neighboring_classes[u_class].intersection(v_classes):
                for i in intersection:
                    _insert_cycle(found_cycles, i, (i, u_class, n))

            for v_class in v_classes:
                # squares
                u_class_neigh = neighboring_classes[u_class]
                v_class_neigh = neighboring_classes[v_class]
                res = u_class_neigh.intersection(v_class_neigh) - intersection
                for i in intersection:
                    for r in res:
                        _insert_cycle(found_cycles, i, (i, u_class, r, v_class))

                # pentagons
                for r in u_class_neigh - set([u_class]):
                    for t in neighboring_classes[r].intersection(v_class_neigh):
                        for i in intersection:
                            _insert_cycle(found_cycles, i, (i, u_class, r, t, v_class))


def _insert_cycle(
    found_cycles: dict[int, set[tuple[int, ...]]],
    class_id: int,
    cycle: tuple[int, ...],
):
    """
    Insert cycles in a canonical form.
    to prevent having the same cycle included multiple times.

    The canonical a form such that it is that the first ID
    is the smallest class ID in the cycle
    and the second one is the lower ID of the neighbor in the cycle.
    """
    # in case one class was used multiple times
    if len(set(cycle)) != len(cycle):
        return

    # find the smallest element index
    smallest = 0
    for i, e in enumerate(cycle):
        if e < cycle[smallest]:
            smallest = i

    # makes sure that the element following the smallest one
    # is greater than the one preceding it
    if cycle[smallest - 1] < cycle[(smallest + 1) % len(cycle)]:
        cycle = list(reversed(cycle))
        smallest = len(cycle) - smallest - 1

    # rotates the list such that the element with the smallest ID is the first
    cycle = cycle[smallest:] + cycle[:smallest]

    found_cycles[class_id].add(tuple(cycle))
