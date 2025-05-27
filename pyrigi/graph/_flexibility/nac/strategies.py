import random
from typing import Callable, Sequence

import networkx as nx

from pyrigi.graph._flexibility.nac.core import (
    IntEdge,
    SubgraphColorings,
    mask_to_vertices,
    vertices_of_classes,
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
    For algorithm description, see documentation.

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

    remaining_classes = list(class_ids)
    ordered_class_ids_groups: list[list[int]] = [[] for _ in chunk_sizes]

    # if False, chunk does need to assign random class
    is_random_class_required: list[bool] = [True for _ in chunk_sizes]

    edge_to_class: dict[IntEdge, int] = {
        e: class_id
        for class_id, edges_class in enumerate(class_to_edges)
        for e in edges_class
    }

    while len(remaining_classes) > 0:
        # could be avoided by having a proper subgraph
        # and by class_to_edges being also appropriately updated
        non_covered_vertices: set[int] = vertices_of_classes(
            remaining_classes, class_to_edges
        )

        # represents index of the chosen target subgraph
        # classes will be added to this subgraph now
        chunk_index = min(
            range(len(ordered_class_ids_groups)),
            key=lambda x: len(ordered_class_ids_groups[x]) / chunk_sizes[x],
        )
        # list to add classes to
        target_chunk = ordered_class_ids_groups[chunk_index]

        # classes already added to the subgraph
        added_classes: set[int] = set()

        # if subgraph is still empty, we add a random class to it
        if is_random_class_required[chunk_index]:
            rand_class = remaining_classes[rand.randint(0, len(remaining_classes) - 1)]
            added_classes.add(rand_class)
            target_chunk.append(rand_class)
            is_random_class_required[chunk_index] = False

        # vertices of the already chosen classes
        used_vertices = vertices_of_classes(target_chunk, class_to_edges)

        # vertices to search trough
        opened: set[int] = set()

        # add all the neighbors of the vertices of the already added classes
        for v in used_vertices:
            for u in graph.neighbors(v):
                if u in used_vertices:
                    continue
                opened.add(u)
        _subgraphs_strategy_neighbors_open_neighbors(
            graph=graph,
            used_vertices=used_vertices,
            opened=opened,
            to_open=non_covered_vertices,
        )

        # go through the opened vertices and searches for cycles
        # fill the subgraph till we run out or vertices or fill the chunk
        iteration_no = 0
        while opened and len(target_chunk) < chunk_sizes[chunk_index]:
            class_added_for_best_vertex = False

            best_vertex = _subgraphs_strategy_neighbors_best_vertex(
                use_degree=use_degree,
                graph=graph,
                used_vertices=used_vertices,
                non_covered_vertices=non_covered_vertices,
                opened=opened,
            )

            # we take the common neighborhood of
            # already used vertices and the chosen vertex
            for neighbor in used_vertices.intersection(graph.neighbors(best_vertex)):
                # vertex is not part of the current subgraph
                if neighbor not in non_covered_vertices:
                    continue

                # class of the edge incident to the best vertex
                # and the chosen vertex
                class_id: int = edge_to_class.get(
                    (best_vertex, neighbor),
                    edge_to_class.get((neighbor, best_vertex), None),
                )
                # the edge is part of another class
                if class_id not in remaining_classes:
                    continue
                # the class of the edge was already added
                if class_id in added_classes:
                    continue

                # we can add the class of the edge
                added_classes.add(class_id)
                target_chunk.append(class_id)
                class_added_for_best_vertex = True

                # Checks if the cycles can continue
                # subgraph is full
                if len(target_chunk) >= chunk_sizes[chunk_index]:
                    break

                # add new vertices to the used vertices so they can be used
                # in a next iteration
                new_vertices: set[int] = {
                    v for e in class_to_edges[class_id] for v in e
                }
                used_vertices |= new_vertices
                opened -= new_vertices

                # open neighbors of the newly added vertices
                _subgraphs_strategy_neighbors_open_neighbors(
                    graph=graph,
                    used_vertices=used_vertices,
                    opened=opened,
                    to_open=new_vertices,
                )

            if class_added_for_best_vertex:
                iteration_no += 1
            else:
                opened.remove(best_vertex)

        # Nothing happened, we need to find some class randomly
        if iteration_no == 0:
            is_random_class_required[chunk_index] = True

        # Remove used classes, so they cannot be considered anymore
        remaining_classes = [
            class_id for class_id in remaining_classes if class_id not in added_classes
        ]

    return [v for group in ordered_class_ids_groups for v in group]


def _subgraphs_strategy_neighbors_best_vertex(
    use_degree: bool,
    graph: nx.Graph,
    used_vertices: set[int],
    non_covered_vertices: set[int],
    opened: set[int],
) -> int:
    """
    Compute score and return the vertex with the highest value
    """

    # compute score or each vertex
    if not use_degree:
        values = [
            (u, len(used_vertices.intersection(graph.neighbors(u)))) for u in opened
        ]
    else:
        values = [
            (
                u,
                (
                    len(used_vertices.intersection(graph.neighbors(u))),
                    # degree
                    -len(non_covered_vertices.intersection(graph.neighbors(u))),
                ),
            )
            for u in opened
        ]

    # shuffling seams to decrease the performance
    # rand.shuffle(values)

    # chooses a vertex with the highest score
    best_vertex = max(values, key=lambda x: x[1])[0]
    return best_vertex


def _subgraphs_strategy_neighbors_open_neighbors(
    graph: nx.Graph,
    used_vertices: set[int],
    opened: set[int],
    to_open: set[int],
):
    for v in to_open:
        for u in graph.neighbors(v):
            if u in used_vertices:
                continue
            opened.add(u)


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
