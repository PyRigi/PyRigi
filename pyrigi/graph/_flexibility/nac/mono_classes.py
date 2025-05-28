"""
This module is responsible for finding
triangle-connected components and
:prf:ref:`NAC-mono classes <def-nac-mono>`.
"""

from collections import defaultdict
from enum import Enum

import networkx as nx

from pyrigi._utils.union_find import UnionFind
from pyrigi.data_type import Edge


class MonoClassType(Enum):
    """
    Represents approaches for finding :prf:ref:`NAC-mono classes <def-nac-mono>`
    of different types - single edges, triangle-connected components,
    and triangle-extended classes.

    Suggested Improvement
    ---------------------
    Make from_string more fancy.
    """

    """
    Each edge is its own :prf:ref:`NAC-mono class <def-nac-mono>`.
    """
    EDGES = "edges"
    """
    Corresponds to
    :prf:ref:`\\triangle-connected components<def-triangle-connected-comp>`.
    """
    TRI_CONNECTED = "triangle"
    """
    Corresponds to
    :prf:ref:`\\hat\\triangle-extended classes<def-triangle-extended-class>`.
    """
    TRI_EXTENDED = "triangle-extended"

    @staticmethod
    def from_string(class_name: str) -> "MonoClassType":
        match class_name:
            case "edges":
                return MonoClassType.EDGES
            case "triangle":
                return MonoClassType.TRI_CONNECTED
            case "triangle-extended":
                return MonoClassType.TRI_EXTENDED
            case _:
                raise ValueError(f"Unknown NAC-mono class type: {class_name}")


def _trivial_mono_classes(
    graph: nx.Graph,
) -> tuple[dict[Edge, int], list[list[Edge]]]:
    """
    Makes each edge its own NAC-mono class
    """
    edge_to_class: dict[Edge, int] = {}
    class_to_edges: list[list[Edge]] = []
    for i, e in enumerate(graph.edges):
        edge_to_class[e] = i
        class_to_edges.append([e])
    return edge_to_class, class_to_edges


def find_mono_classes(
    graph: nx.Graph,
    class_type: MonoClassType = MonoClassType.TRI_EXTENDED,
) -> tuple[dict[Edge, int], list[list[Edge]]]:
    """
    Find :prf:ref:`NAC-mono classes <def-nac-mono>` based on the type given.

    First, all the classes of triangle equivalence are found.
    Then these are optionally extended to larger NAC-mono classes
    as described in :cite:p:`LastovickaLegersky2024`.

    Parameters
    ----------
    graph:
        Input graph
    class_type:
        Type of :prf:ref:`NAC-mono classes <def-nac-mono>`

    Returns
    -------
    An ID of a :prf:ref:`NAC-mono class <def-nac-mono>`
    corresponds to its index in a list of all NAC-mono classes.
    Return a mapping from edges to their class ID
    and a list of NAC-mono classes where
    the index corresponds to the class ID.
    """
    if class_type == MonoClassType.EDGES:
        return _trivial_mono_classes(graph)

    classes = UnionFind()

    # Finds triangles
    for edge in graph.edges:
        v, u = edge

        # We cannot sort vertices and we cannot expect
        # any regularity or order of the vertices
        classes.join((u, v), (v, u))

        v_neighbours = set(graph.neighbors(v))
        u_neighbours = set(graph.neighbors(u))
        intersection = v_neighbours.intersection(u_neighbours)
        for w in intersection:
            classes.join((u, v), (w, v))
            classes.join((u, v), (w, u))

    # Checks for edges & triangles over class
    # This MUST be run before search for squares for cartesian NAC-coloring
    # of other search that may produce disconnected classes,
    # as cycles may not exist!
    # There routines are highly inefficient, but the time is still
    # negligible compared to the main algorithm running time.
    if class_type == MonoClassType.TRI_EXTENDED:
        # we try again until we find no other class to merge
        # new opinions may appear later
        # could be most probably implemented smarter
        done = False
        while not done:
            done = True

            vertex_to_classes: list[set[Edge]] = [
                set() for _ in range(max(graph.nodes) + 1)
            ]

            # prepare updated vertex to class mapping
            for e in graph.edges:
                class_id = classes.find(e)
                vertex_to_classes[e[0]].add(class_id)
                vertex_to_classes[e[1]].add(class_id)

            # v is the top of the triangle over class
            for v in graph.nodes:
                # maps class to set of vertices containing it
                class_to_vertices: dict[Edge, set[int]] = defaultdict(set)
                for u in graph.neighbors(v):
                    # find all the classes v neighbors with
                    for class_of_vertex in vertex_to_classes[u]:
                        class_to_vertices[class_of_vertex].add(u)

                # if we found more edges to the same class,
                # we also found a triangle and we merge it's arms
                for class_of_vertex, vertices in class_to_vertices.items():
                    if class_of_vertex == len(vertices) <= 1:
                        continue

                    vertices = iter(vertices)
                    w = next(vertices)
                    for u in vertices:
                        # if something changed, we may have another
                        # change for improvement the next round
                        done &= not classes.join((v, w), (v, u))

    edge_to_class: dict[Edge, int] = {}
    class_to_edge: list[list[Edge]] = []

    for edge in graph.edges:
        root = classes.find(edge)

        if root not in edge_to_class:
            edge_to_class[root] = id = len(class_to_edge)
            class_to_edge.append([])
        else:
            id = edge_to_class[root]

        edge_to_class[edge] = id
        class_to_edge[id].append(edge)

    return edge_to_class, class_to_edge
