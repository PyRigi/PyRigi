"""
This module is responsible for finding
triangle-connected components and
:prf:ref:`NAC-mono classes <def-nac-mono>`.
"""

from collections import defaultdict
from enum import Enum
import networkx as nx

from pyrigi._util.union_find import UnionFind
from pyrigi.data_type import Edge


class MonoClassType(Enum):
    """
    Represents approaches for finding :prf:ref:`NAC-mono classes <def-nac-mono>`
    of different types - single edges, triangle-connected components,
    and triangle-extended classes.
    """

    """
    Each edge is its own :prf:ref:`NAC-mono class <def-nac-mono>`.
    """
    EDGES = "EDGES"
    """
    Corresponds to :prf:ref:`\\triangle-connected components<def-triangle-connected-comp>`.
    """
    TRI_CONNECTED = "TRIANGLE_CONNECTED_COMPONENTS"
    """
    Corresponds to :prf:ref:`\\hat\\triangle-extended classes<def-triangle-extended-class>`.
    """
    TRI_EXTENDED = "TRIANGLE_EXTENDED_CLASSES"


def _trivial_mono_classes(
    graph: nx.Graph,
) -> tuple[dict[Edge, int], list[list[Edge]]]:
    """
    Makes each edge its own NAC-mono class
    """
    edge_to_component: dict[Edge, int] = {}
    component_to_edges: list[list[Edge]] = []
    for i, e in enumerate(graph.edges):
        edge_to_component[e] = i
        component_to_edges.append([e])
    return edge_to_component, component_to_edges


def find_mono_classes(
    graph: nx.Graph,
    class_type: MonoClassType = MonoClassType.TRI_EXTENDED,
) -> tuple[dict[Edge, int], list[list[Edge]]]:
    """
    Find :prf:ref:`NAC-mono classes <def-nac-mono>` based on the type given.

    First, all the components of triangle equivalence are found.
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
    Return a mapping from edges to their component ID
    and a list of NAC-mono classes where
    the index corresponds to the component ID.
    """
    if class_type == MonoClassType.EDGES:
        return _trivial_mono_classes(graph)

    components = UnionFind()

    # Finds triangles
    for edge in graph.edges:
        v, u = edge

        # We cannot sort vertices and we cannot expect
        # any regularity or order of the vertices
        components.join((u, v), (v, u))

        v_neighbours = set(graph.neighbors(v))
        u_neighbours = set(graph.neighbors(u))
        intersection = v_neighbours.intersection(u_neighbours)
        for w in intersection:
            components.join((u, v), (w, v))
            components.join((u, v), (w, u))

    # Checks for edges & triangles over component
    # This MUST be run before search for squares for cartesian NAC-coloring
    # of other search that may produce disconnected components,
    # as cycles may not exist!
    # There routines are highly inefficient, but the time is still
    # negligible compared to the main algorithm running time.
    if class_type == MonoClassType.TRI_EXTENDED:
        # we try again until we find no other component to merge
        # new opinions may appear later
        # could be most probably implemented smarter
        done = False
        while not done:
            done = True

            vertex_to_components: list[set[Edge]] = [
                set() for _ in range(max(graph.nodes) + 1)
            ]

            # prepare updated vertex to component mapping
            for e in graph.edges:
                comp_id = components.find(e)
                vertex_to_components[e[0]].add(comp_id)
                vertex_to_components[e[1]].add(comp_id)

            # v is the top of the triangle over component
            for v in graph.nodes:
                # maps component to set of vertices containing it
                comp_to_vertices: dict[Edge, set[int]] = defaultdict(set)
                for u in graph.neighbors(v):
                    # find all the components v neighbors with
                    for comp in vertex_to_components[u]:
                        comp_to_vertices[comp].add(u)

                # if we found more edges to the same component,
                # we also found a triangle and we merge it's arms
                for comp, vertices in comp_to_vertices.items():
                    if comp == len(vertices) <= 1:
                        continue

                    vertices = iter(vertices)
                    w = next(vertices)
                    for u in vertices:
                        # if something changed, we may have another
                        # change for improvement the next round
                        done &= not components.join((v, w), (v, u))

    edge_to_component: dict[Edge, int] = {}
    component_to_edge: list[list[Edge]] = []

    for edge in graph.edges:
        root = components.find(edge)

        if root not in edge_to_component:
            edge_to_component[root] = id = len(component_to_edge)
            component_to_edge.append([])
        else:
            id = edge_to_component[root]

        edge_to_component[edge] = id
        component_to_edge[id].append(edge)

    return edge_to_component, component_to_edge
