"""

Module for the functionality concerning frameworks.

.. currentmodule:: pyrigi.framework

Classes:

.. autosummary::

    Framework
    Realization




"""

from pyrigi.graph import Graph



class Framework(object):
    """
    This class provides the functionality for frameworks. By definition, it is a tuple of a graph and a realization.

    METHODS:

    .. autosummary::
        dim
        dimension
        add_vertex
        add_vertices
        add_edge
        add_edges
    """
    graph = None
    realization = None

    class Realization(dict):
        r"""
        This class represents a realization.

        A realization is a map from the set of vertices to $\RR^d$.
        The labeling is implicit and given by the placement's order.

        METHODS:

        .. autosummary::
            add_vertex
            add_vertices
        """

        def __init__(self, vertices=[], points=[], dim=2):
            self.add_vertices(vertices, points)
            self.dimension = dim

        def add_vertex(self, vertex, point):
            assert len(point)==self.dimension
            self[vertex] = point

        def add_vertices(self, vertices, points):
            assert len(vertices)==len(points)
            for vector in points:
                assert(len(vector))==self.dimension
            self.update(zip(vertices, points))

    def __init__(self, p=[], d=2):
        self.realization = Realization(p, d)
        self.graph = Graph()

    def dim(self):
        return self.dimension()

    def dimension(self):
        return self.realization.dimension

    def add_vertex(self, point, label=None):
        if label == None:
            maxNode = max(self.graph.nodes) if len(self.graph.nodes)>0 else 0
            label = maxNode + 1
        self.realization.add_vertex(point)
        self.graph.add_node(label)

    def add_vertices(self, points, labels=[]):
        assert(len(points)==len(labels) or len(labels)==0)
        if len(labels)==0:
            for point in points:
                self.add_vertex(point)
        else:
            for i in range(len(points)):
                self.add_vertex(points[i], labels[i])

    def add_edge(self, edge):
        assert (len(edge))==2
        assert (edge[0] in self.graph.nodes and edge[1] in self.graph.nodes)
        self.graph.add_edge(*(edge))

    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge)
