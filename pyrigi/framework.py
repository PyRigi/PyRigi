"""

Module for the functionality concerning frameworks.

.. currentmodule:: pyrigi.framework

Classes:

.. autosummary::

    Framework

"""

from copy import deepcopy
from pyrigi.graph import Graph


class Framework(object):
    """
    This class provides the functionality for frameworks. By definition, it is a tuple of a graph and a realization.

    ATTRIBUTES
    ----------
    graph : Graph
    realization : dict
    dim : int

    """
    #TODO override decorator for empty constructor?
    def __init__(self, graph, realization):
        # TODO: check that graph and realization is not empty
        assert isinstance(graph, Graph)
        dim = len(list(realization.values())[0])
        for v in graph.vertices():
            assert v in realization
            assert len(realization[v])==dim
        self.realization = {v:realization[v] for v in graph.vertices()}
        self._graph = deepcopy(graph)
        self._graph._part_of_framework = True
        self.dim = dim

    def dim(self):
        return self.dim

    def dimension(self):
        return self.dim()

    def graph(self):
        """Return an immutable copy of the graph object"""
        return deepcopy(self._graph)
    
    def add_vertex(self, point, vertex=None):
        if vertex == None:
            candidate = len(self._graph.vertices())
            while candidate in self._graph.vertices():
                candidate += 1
            vertex = candidate
        assert vertex not in self._graph.vertices()
        self.realization[vertex] = point
        self._graph.add_node(vertex)

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
        assert (edge[0] in self._graph.nodes and edge[1] in self._graph.nodes)
        self._graph.add_edge(*(edge))

    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge)

    def underlying_graph(self):
        return deepcopy(self._graph)

    def print(self):
        print('Graph:\t\t', self._graph)
        print('Realization:\t', self.realization)
        
    @classmethod
    def from_points(cls, points):
        raise NotImplementedError()

    @classmethod
    def from_graph(cls, graph):
        raise NotImplementedError()

    @classmethod
    def empty(cls, dim):
        raise NotImplementedError()

    def delete_vertex(self, vertex):
        raise NotImplementedError()

    def delete_vertices(self, vertices):
        raise NotImplementedError()

    def delete_edge(self, edge):
        raise NotImplementedError()

    def delete_edges(self, edges):
        raise NotImplementedError()

    def set_vertex_position(self, vertex, point):
        raise NotImplementedError()

    def set_realization(self, realization):
        raise NotImplementedError()

    def rigidity_matrix(self):
        r""" Construct the rigidity matrix of the framework
        """
        raise NotImplementedError()

    def stress_matrix(self, data):
        r""" Construct the stress matrix from a stress of from its support
        """
        raise NotImplementedError()

    def infinitesimal_flexes(self, trivial=False):
        r""" Returns a basis of the space of infinitesimal flexes
        """
        raise NotImplementedError()

    def stresses(self):
        r""" Returns a basis of the space of stresses
        """
        raise NotImplementedError()

    def rigidity_matrix_rank(self):
        raise NotImplementedError()

    def is_infinitesimally_rigid(self):
        raise NotImplementedError()

    def is_infinitesimally_spanning(self):
        raise NotImplementedError()

    def is_minimally_infinitesimally_rigid(self):
        raise NotImplementedError()

    def is_infinitesimally_flexible(self):
        raise NotImplementedError()

    def is_independent(self):
        raise NotImplementedError()

    def is_prestress_stable(self):
        raise NotImplementedError()

    def is_congruent(self, framework_):
        raise NotImplementedError()

    def is_equivalent(self, framework_):
        raise NotImplementedError()

    def pin(self, vertices):
        raise NotImplementedError()

    def trivial_infinitesimal_flexes(self):
        raise NotImplementedError()

    def redundantly_rigid(self):
        raise NotImplementedError()
