"""
This module is responsible for finding
triangle-connected components and
:prf:ref:`NAC-mono classes <def-nac-mono>`.
"""

from collections import defaultdict
from enum import Enum
import networkx as nx

from pyrigi.util.union_find import UnionFind
from pyrigi.data_type import Edge


# TODO rename when all the code is refactored
class MonochromaticClassType(Enum):
    """
    Represents approaches for finding :prf:ref:`NAC-valid classes <def-nac-mono>`
    of different types - single edges, triangle-connected components,
    and NAC-mono classes.
    """

    """
    Each edge is its own :prf:ref:`NAC-mono class <def-nac-mono>`.
    """
    EDGES = "EDGES"
    """
    Each triangle-connected component is its own :prf:ref:`NAC-mono class <def-nac-mono>`.
    """
    TRIANGLES = "TRIANGLE_CONNECTED_COMPONENTS"
    """
    Each NAC-mono class is its own :prf:ref:`NAC-mono class <def-nac-mono>`.
    Classes are made based on the approach from :cite:p:`LastovickaLegersky2024`.
    """
    MONOCHROMATIC = "MONOCHROMATIC_CLASSES"


# TODO rename when all the code is refactored
def _trivial_monochromatic_classes(
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


# TODO rename when all the code is refactored
def find_monochromatic_classes(
    graph: nx.Graph,
    class_type: MonochromaticClassType = MonochromaticClassType.MONOCHROMATIC,
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
    if class_type == MonochromaticClassType.EDGES:
        return _trivial_monochromatic_classes(graph)

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
    if class_type == MonochromaticClassType.MONOCHROMATIC:
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


def create_component_graph_from_components(
    graph: nx.Graph,
    edges_to_components: dict[Edge, int],
) -> nx.Graph:
    """
    Create a component graph from the :prf:ref:`NAC-mono class <def-nac-mono>` given.
    Classes are vertices and there is an edge between them
    if they share a vertex in the original graph.
    """

    # graph used to find NAC coloring easily
    comp_graph = nx.Graph()

    def get_edge_component(e: Edge) -> int:
        u, v = e
        res = edges_to_components.get((u, v))
        if res is None:
            res = edges_to_components[(v, u)]
        return res

    for v in graph.nodes:
        edges = list(graph.edges(v))
        for i in range(0, len(edges)):
            c1 = get_edge_component(edges[i])
            # this must be explicitly added in case
            # the edge has no neighboring edges
            # in that case this is a single node
            comp_graph.add_node(c1)

            for j in range(i + 1, len(edges)):
                c2 = get_edge_component(edges[j])
                if c1 == c2:
                    continue
                elif c1 < c2:
                    comp_graph.add_edge(c1, c2)
                else:
                    comp_graph.add_edge(c2, c1)

    return comp_graph
