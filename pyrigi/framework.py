"""

Module for the functionality concerning frameworks.

.. currentmodule:: pyrigi.framework

Classes:

.. autosummary::

    Framework
    Realization




"""

from copy import deepcopy
from pyrigi.graph import Graph
from pyrigi.matrix import Matrix


class Framework(object):
    """
    This class provides the functionality for frameworks. By definition, it is a tuple of a graph and a realization.

    ATTRIBUTES
    ----------
    graph : Graph
    realization : dict
    dimension : int

    METHODS:

    .. autosummary::
        dim
        dimension
        add_vertex
        add_vertices
        add_edge
        add_edges
    """

    def __init__(self, graph, realization):
        # TODO: check that graph and realization is not empty
        assert isinstance(graph, Graph)
        dimension = len(list(realization.values())[0]
        for v in graph.vertices():
            assert v in realization
            assert len(realization[v])==dimension
        self.realization = {v:realization[v] for v in graph.vertices()}
        self.graph = deepcopy(graph)
        self.graph._part_of_framework = True
        self.dimension = dimension

    def dim(self):
        return self.dimension()

    def dimension(self):
        return self.dimension

    def add_vertex(self, point, vertex=None):
        # TODO: complain if the vertex is already contained in the graph
        if vertex == None:
            candidate = len(self.graph.vertices())
            while candidate in self.graph.vertices():
                candidate += 1
            vertex = candidate
        self.realization[vertex] = point
        self.graph.add_node(vertex)

    def add_vertices(self, points, vertices=[]):
        assert(len(points)==len(vertices) or not vertices)
        if not vertices:
            for point in points:
                self.add_vertex(point)
        else:
            for p, v in zip(points, vertices):
                self.add_vertex(p, v)

    def add_edge(self, edge):
        assert (len(edge))==2
        assert (edge[0] in self.graph.nodes and edge[1] in self.graph.nodes)
        self.graph.add_edge(*(edge))

    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge)

    @classmethod
    def from_points(cls, points):
        raise NotImplementedError()

    @classmethod
    def from_graph(cls, graph):
        raise NotImplementedError()

    @classmethod
    def empty(cls, dimension):
        raise NotImplementedError()

    def delete_vertex(self, vertex):
        raise NotImplementedError()

    def delete_vertices(self, vertices):
        raise NotImplementedError()

    def delete_edge(self, edge):
        raise NotImplementedError()

    def delete_edges(self, edges):
        raise NotImplementedError()
