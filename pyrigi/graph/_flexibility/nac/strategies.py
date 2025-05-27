import random
from typing import Callable, Sequence

import networkx as nx

from pyrigi.graph._flexibility.nac.core import (
    IntEdge,
    SubgraphColorings,
    mask_to_vertices,
)


################################################################################
# Splitting strategies
################################################################################
def subgraphs_strategy_neighbors(
    graph: nx.Graph,
    class_ids: list[int],
    class_to_edges: list[list[IntEdge]],
    chunk_sizes: Sequence[int],
    use_degree: bool,
    seed: int | None,
) -> list[int]:
    """
    Split graph into subgraphs by neighbors.
    For exhaustive algorithm description, see documentation.

    Parameters
    ----------
    graph:
        The original graph.
    class_ids:
        List of available classes IDs.
    class_to_edges:
        List of edges for each class.
    chunk_sizes:
        List of chunk sizes.
    seed:
        Random seed used to split equal decisions.

    """
    rand = random.Random(seed)

    class_ids = list(class_ids)
    ordered_class_ids_groups: list[list[int]] = [[] for _ in chunk_sizes]

    # if False, chunk does need to assign random component
    is_random_class_required: list[bool] = [True for _ in chunk_sizes]

    edge_to_class: dict[IntEdge, int] = {
        e: class_id
        for class_id, edges_class in enumerate(class_to_edges)
        for e in edges_class
    }

    while len(class_ids) > 0:
        rand_comp = class_ids[rand.randint(0, len(class_ids) - 1)]

        # could be avoided by having a proper subgraph
        # and by class_to_edges being also appropriately updated
        local_ordered_class_ids: set[int] = {
            v
            for class_id, edges_class in enumerate(class_to_edges)
            for e in edges_class
            for v in e
            if class_id in class_ids
        }

        # represents index of the chosen target subgraph
        # classes will be added to this subgraph now
        chunk_index = min(
            range(len(ordered_class_ids_groups)),
            key=lambda x: len(ordered_class_ids_groups[x]) / chunk_sizes[x],
        )
        # list to add class to
        target = ordered_class_ids_groups[chunk_index]

        # components already added to the subgraph
        added_classes: set[int]

        # if subgraph is still empty, we add a random class to it
        if is_random_class_required[chunk_index]:
            added_classes: set[int] = set([rand_comp])
            target.append(rand_comp)
            is_random_class_required[chunk_index] = False
        else:
            added_classes = set()

        # vertices of the already chosen classes
        used_vertices = {
            v
            for targets_class in target
            for e in class_to_edges[targets_class]
            for v in e
        }

        # queue of vertices to search trough
        opened: set[int] = set()

        # add all the neighbors of the vertices of the already added classes
        for v in used_vertices:
            for u in graph.neighbors(v):
                if u in used_vertices:
                    continue
                if u not in local_ordered_class_ids:
                    continue
                opened.add(u)

        # go through the opened vertices and searches for cycles
        # fill the subgraph till we run out or vertices or fill the chunk
        iteration_no = 0
        while opened and len(target) < chunk_sizes[chunk_index]:
            class_added = False

            # compute score or each vertex
            if not use_degree:
                values = [
                    (u, len(used_vertices.intersection(graph.neighbors(u))))
                    for u in opened
                ]
            else:
                values = [
                    (
                        u,
                        (
                            len(used_vertices.intersection(graph.neighbors(u))),
                            # degree
                            -len(
                                local_ordered_class_ids.intersection(graph.neighbors(u))
                            ),
                        ),
                    )
                    for u in opened
                ]

            # shuffling seams to decrease the performance
            # rand.shuffle(values)

            # chooses a vertex with the highest score
            best_vertex = max(values, key=lambda x: x[1])[0]

            # we take the common neighborhood of
            # already used vertices and the chosen vertex
            for neighbor in used_vertices.intersection(graph.neighbors(best_vertex)):
                # vertex is not part of the current subgraph
                if neighbor not in local_ordered_class_ids:
                    continue

                # component of the edge incident to the best vertex
                # and the chosen vertex
                class_id: int = edge_to_class.get(
                    (best_vertex, neighbor),
                    edge_to_class.get((neighbor, best_vertex), None),
                )
                # the edge is part of another component
                if class_id not in class_ids:
                    continue
                # the component of the edge was already added
                if class_id in added_classes:
                    continue

                # we can add the component of the edge
                added_classes.add(class_id)
                target.append(class_id)
                class_added = True

                # Checks if the cycles can continue
                # subgraph is full
                if len(target) >= chunk_sizes[chunk_index]:
                    break

                # add new vertices to the used vertices so they can be used
                # in a next iteration
                new_vertices: set[int] = {
                    v for e in class_to_edges[class_id] for v in e
                }
                used_vertices |= new_vertices
                opened -= new_vertices

                # open neighbors of the newly added vertices
                for v in new_vertices:
                    for u in graph.neighbors(v):
                        if u in used_vertices:
                            continue
                        if u not in local_ordered_class_ids:
                            continue
                        opened.add(u)

            if class_added:
                iteration_no += 1
            else:
                opened.remove(best_vertex)

        # Nothing happened, we need to find some component randomly
        if iteration_no == 0:
            is_random_class_required[chunk_index] = True

        # Remove used classes, so they cannot be considered anymore
        class_ids = [
            class_id for class_id in class_ids if class_id not in added_classes
        ]

    return [v for group in ordered_class_ids_groups for v in group]


################################################################################
# Merging strategies
################################################################################
def linear(
    colorings_merge_wrapper: Callable[
        [SubgraphColorings, SubgraphColorings], SubgraphColorings
    ],
    all_epochs: list[SubgraphColorings],
) -> list[SubgraphColorings]:
    """
    Merge all subgraphs linearly.
    """
    res: SubgraphColorings = all_epochs[0]
    for g in all_epochs[1:]:
        res = colorings_merge_wrapper(res, g)
    return [res]


def shared_vertices(
    class_to_edges: list[list[IntEdge]],
    ordered_class_ids: list[int],
    colorings_merge_wrapper: Callable[
        [SubgraphColorings, SubgraphColorings], SubgraphColorings
    ],
    all_epochs: list[SubgraphColorings],
) -> list[SubgraphColorings]:
    """
    Merge subgraphs by preferring pairs where with more shared vertices
    """
    while len(all_epochs) > 1:
        best = (0, 0, 1)
        subgraph_vertices: list[set[int]] = [
            mask_to_vertices(ordered_class_ids, class_to_edges, mask)
            for _, mask in all_epochs
        ]
        for i in range(0, len(subgraph_vertices)):
            for j in range(i + 1, len(subgraph_vertices)):
                vert1 = subgraph_vertices[i]
                vert2 = subgraph_vertices[j]
                vertex_no = len(vert1.intersection(vert2))
                if vertex_no > best[0]:
                    best = (vertex_no, i, j)
        res = colorings_merge_wrapper(all_epochs[best[1]], all_epochs[best[2]])

        all_epochs[best[1]] = res
        all_epochs.pop(best[2])
    return all_epochs
