"""
This module contains code related to finding cycles
that should correspond to :prf:ref:`NAC-mono classes <def-nac-mono>`.
"""

from collections import defaultdict
from typing import Collection

import networkx as nx


def find_cycles(
    graph: nx.Graph,
    subgraph_component_IDs: Collection[int],
    class_to_edges: Collection[Collection[tuple[int, int]]],
    per_class_limit: int = 2,
) -> set[tuple[int, ...]]:
    """
    For each edge, find all the cycles among
    :prf:ref:`NAC-mono classes <def-nac-mono>`
    of length at most five (length is the number of classes it spans).

    Not all the returned cycles are guaranteed to be actual cycles as
    this may create cycles that enter and exit a component at the same vertex.

    Parameters
    ----------
    graph:
        Input graph.
    subgraph_component_IDs:
        Components of the subgraph.
    class_to_edges:
        A list of edges for each component.
    per_class_limit:
        The maximum number of a cycle to be returned per class.
    """
    cycles = _find_useful_cycles_for_components(
        graph=graph,
        subgraph_component_IDs=subgraph_component_IDs,
        class_to_edges=class_to_edges,
        per_class_limit=per_class_limit,
    )
    return {c for comp_cycles in cycles.values() for c in comp_cycles}


def _find_useful_cycles_for_components(
    graph: nx.Graph,
    subgraph_component_IDs: Collection[int],
    class_to_edges: Collection[Collection[tuple[int, int]]],
    per_class_limit: int = 2,
) -> dict[int, set[tuple[int, ...]]]:
    """
    Same as :class:`~pyrigi.graph._framework.nac.cycle_detection.find_cycles`
    except the results are grouped for each component.
    """
    classes_no = len(class_to_edges)

    # creates mapping from vertex to set of monochromatic classes if is in
    vertex_to_components = [set() for _ in range(max(graph.nodes) + 1)]
    for class_id, class_edges in enumerate(class_to_edges):
        if class_id not in subgraph_component_IDs:
            continue
        for u, v in class_edges:
            vertex_to_components[u].add(class_id)
            vertex_to_components[v].add(class_id)
    neighboring_components = [set() for _ in range(classes_no)]

    found_cycles: dict[int, set[tuple[int, ...]]] = defaultdict(set)

    # create a graph where vertices are monochromatic classes as there's
    # an edge if the monochromatic classes share a vertex
    for v in graph.nodes:
        for i in vertex_to_components[v]:
            for j in vertex_to_components[v]:
                if i != j:
                    neighboring_components[i].add(j)

    _find_useful_cycles_for_edges(
        graph,
        vertex_to_components,
        neighboring_components,
        found_cycles,
    )

    limited = {}
    for key, value in found_cycles.items():
        limited[key] = set(list(sorted(value, key=len))[:per_class_limit])

    return limited


def _find_useful_cycles_for_edges(
    graph: nx.Graph,
    vertex_to_components: list[set[int]],
    neighboring_components: list[set[int]],
    found_cycles: dict[int, set[tuple[int, ...]]],
):
    """
    Find three, four and five cycles in component graph
    for each edge in the original graph.
    """
    for u, v in graph.edges:
        u_comps = vertex_to_components[u]
        v_comps = vertex_to_components[v]

        # remove shared components
        intersection = u_comps.intersection(v_comps)
        u_comps = u_comps - intersection
        v_comps = v_comps - intersection

        # this check only makes sense for proper NAC-mono classes,
        # triangle-connected components fail on it
        # assert len(intersection) <= 1

        if len(intersection) == 0:
            continue

        for u_comp in u_comps:
            # triangles
            for n in neighboring_components[u_comp].intersection(v_comps):
                for i in intersection:
                    _insert_cycle(found_cycles, i, (i, u_comp, n))

            for v_comp in v_comps:
                # squares
                u_comp_neigh = neighboring_components[u_comp]
                v_comp_neigh = neighboring_components[v_comp]
                res = u_comp_neigh.intersection(v_comp_neigh) - intersection
                for i in intersection:
                    for r in res:
                        _insert_cycle(found_cycles, i, (i, u_comp, r, v_comp))

                # pentagons
                for r in u_comp_neigh - set([u_comp]):
                    for t in neighboring_components[r].intersection(v_comp_neigh):
                        for i in intersection:
                            _insert_cycle(found_cycles, i, (i, u_comp, r, t, v_comp))


def _insert_cycle(
    found_cycles: dict[int, set[tuple[int, ...]]],
    comp_id: int,
    cycle: tuple[int, ...],
):
    """
    Insert cycles in a canonical form.
    to prevent having the same cycle included multiple times.

    The canonical a form such that it is that the first ID
    is the smallest component ID in the cycle
    and the second one is the lower ID of the neighbor in the cycle.
    """
    # in case one component was used multiple times
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

    found_cycles[comp_id].add(tuple(cycle))
