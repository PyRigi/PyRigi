"""
The module contains functions related to converting from and to
:prf:ref:`NAC-mono <def-nac-mono>` classes represented by bit masks
to sets of edges representing a :prf:ref:`NAC-coloring <def-nac>`.
"""

from typing import Callable, Collection, Iterable, NamedTuple, TypeAlias

import networkx as nx

from pyrigi.data_type import Edge

"""
:class:`~pyrigi.data_type.Edge` is not used as it is varying.
Here, we use only the interpretation where edge is a tuple.
Also, as the graphs are relabeled, integers are used.
"""
IntEdge: TypeAlias = tuple[int, int]

"""
Represents a :prf:ref:`NAC-coloring <def-nac>`.
Meant for internal use only as :meth:`~frozenset`
should be exposed to the user instead.
Tuple is faster for internal use.
"""
NACColoring: TypeAlias = tuple[Collection[IntEdge], Collection[IntEdge]]


class SubgraphColorings(NamedTuple):
    """
    Represents a subgraph and all its
    :prf:ref:`NAC-coloring <def-nac>` colorings.
    """

    colorings: Iterable[int]
    subgraph_mask: int


def can_have_NAC_coloring(graph: nx.Graph) -> bool:
    """
    Check whether the given graph can have a :prf:ref:`NAC-coloring <def-nac>`
    and if the graph is valid for the :prf:ref:`NAC-coloring <def-nac>` search.

    Graph with less than two edges cannot have a :prf:ref:`NAC-coloring <def-nac>`.
    Graph with more than `n(n-2)/2 - (n-2)` edges
    cannot have a :prf:ref:`NAC-coloring <def-nac>`
    as show in :prf:ref:`thm-flexible-edge-bound`.

    Return
    ------
    `True` if a NAC coloring may exist, `False` if there can be none
    by doing constant complexity checks.

    Throw
    -----
    :class:`~ValueError` if the graph is empty, contains self loops or is directed.
    """
    if graph.nodes() == 0:
        raise ValueError("NAC-coloring search is undefined for the empty graph")

    if nx.number_of_selfloops(graph) > 0:
        raise LookupError()

    if nx.is_directed(graph):
        raise ValueError("NAC-coloring search is undefined for directed graphs")

    if len(nx.edges(graph)) < 2:
        return False

    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    if m > n * (n - 1) // 2 - (n - 2):
        return False

    return True


################################################################################
def coloring_from_mask(
    ordered_class_ids: list[int],
    class_to_edges: list[list[Edge]],
    mask: int,
    allow_mask: int | None = None,
) -> NACColoring:
    """
    Convert a mask representing a red-blue edge coloring.

    Parameters
    ----------
    ordered_class_ids:
        List of class IDs, mask's bits point into it.
    class_to_edges:
        Mapping from class ID to its edges.
    mask:
        Bit mask pointing into `ordered_class_ids`,
        1 means the first and 0 the second set of edges.
    allow_mask:
        Mask allowing only some class.
        Used for subgraphs.
    """

    if allow_mask is None:
        allow_mask = 2 ** len(ordered_class_ids) - 1

    red, blue = [], []
    for i, e in enumerate(ordered_class_ids):
        address = 1 << i

        if address & allow_mask == 0:
            continue

        edges = class_to_edges[e]
        (red if mask & address else blue).extend(edges)
    return (red, blue)


################################################################################
def mask_to_vertices(
    ordered_class_ids: list[int],
    class_to_edges: list[list[IntEdge]],
    subgraph_mask: int,
) -> set[int]:
    """
    Return vertices in the original graph
    that are incident to edges in classes of the subgraph.
    """
    graph_vertices: set[int] = set()

    for i, v in enumerate(ordered_class_ids):
        if (1 << i) & subgraph_mask == 0:
            continue

        edges = class_to_edges[v]
        for u, w in edges:
            graph_vertices.add(u)
            graph_vertices.add(w)
    return graph_vertices


################################################################################
def create_bitmask_for_class_graph_cycle(
    graph: nx.Graph,
    class_to_edges: Callable[[int], list[IntEdge]],
    cycle: tuple[int, ...],
    local_ordered_class_ids: set[int] | None = None,
) -> tuple[int, int]:
    """
    Create a bit mask (template) matching classes in the cycle
    and a mask matching classes of the cycle such that
    if they are the only class with the other color,
    an almost cycle exists.

    Parameters
    ----------
    graph:
        Input graph.
    class_to_edges:
        Mapping from class to its edges.
        Method :meth:`~list.__getitem__` can be also passed.
    cycle:
        A cycle in the class graph.
    local_ordered_class_ids:
        can be used if the graph given is subgraph of the original graph
        and class_to_edges also represent the original graph.

    Returns
    -------
    template:
        Bit mask representing the cycle.
    valid:
        Bit mask representing classes of the cycle such that
        if they are the only class with the other color,
        an almost cycle exists.
    """

    template = 0
    valid = 0

    for v in cycle:
        template |= 1 << v

    def check_for_connecting_edge(prev: int, curr: int, next: int) -> bool:
        """
        Checks if for the class given there exists a path through
        the class using single edge only - in that case,
        if the class is colored by the other color then the other classes,
        an almost cycle exists.
        """
        vertices_curr = {v for e in class_to_edges(curr) for v in e}

        # You may think that if the class is a single edge,
        # it must connect the circle. Because we are using a class graph,
        # the edge can share a vertex with both the neighboring classes.
        # An example for this is a star with 3+ edges.

        vertices_prev = {v for e in class_to_edges(prev) for v in e}
        vertices_next = {v for e in class_to_edges(next) for v in e}
        intersections_prev = vertices_prev.intersection(vertices_curr)
        intersections_next = vertices_next.intersection(vertices_curr)

        if local_ordered_class_ids is not None:
            intersections_prev = intersections_prev.intersection(
                local_ordered_class_ids
            )
            intersections_next = intersections_next.intersection(
                local_ordered_class_ids
            )

        for p in intersections_prev:
            neighbors = set(graph.neighbors(p))
            for n in intersections_next:
                if n in neighbors:
                    return True
        return False

    for prev, curr, next in zip(cycle[-1:] + cycle[:-1], cycle, cycle[1:] + cycle[:1]):
        if check_for_connecting_edge(prev, curr, next):
            valid |= 1 << curr

    return template, valid


################################################################################
def mask_matches_templates(
    templates: list[tuple[int, int]],
    mask: int,
    subgraph_mask: int,
) -> bool:
    """
    Check the given coloring represented by mask for any almost cycle
    that can be found by the given cycle templates.

    Parameters
    ----------
    templates:
        List of cycle templates and allow masks generated by
        :func:`create_bitmask_for_class_graph_cycle`.
    mask:
        Bit mask representing the coloring.
    subgraph_mask:
        Bit mask representing the subgraph.

    Returns
    -------
    Return `True` if an almost cycle was found.
    """

    for template, validity in templates:
        stamp1, stamp2 = mask & template, (mask ^ subgraph_mask) & template
        cnt1, cnt2 = stamp1.bit_count(), stamp2.bit_count()
        stamp, cnt = (stamp1, cnt1) if cnt1 == 1 else (stamp2, cnt2)

        if cnt != 1:
            continue

        # now we know there is one node that has a wrong color
        # we check if the node is a triangle class
        # if so, we need to know it if also ruins the coloring
        if stamp & validity:
            return True
    return False


################################################################################
def NAC_colorings_with_non_surjective(
    graph: nx.Graph,
    colorings: Iterable[NACColoring],
) -> Iterable[NACColoring]:
    """
    Add monochromatic colorings to iterator of NAC-colorings.
    """
    r, b = [], list(graph.edges())
    yield r, b
    yield b, r
    yield from colorings


################################################################################
def vertices_of_classes(
    class_ids: list[int],
    class_to_edges: list[list[IntEdge]],
) -> set[int]:
    """
    Obtain vertices corresponding to edges of
    the given :prf:ref:`NAC-mono classes <def-nac-mono>`.
    """
    return {v for class_id in class_ids for e in class_to_edges[class_id] for v in e}
