"""

Module for the functionality concerning frameworks.

.. currentmodule:: pyrigi.framework

Classes:

.. autosummary::

    Framework
    Realization
    
    


"""


import networkx as nx


class Realization(object):
    r"""
    This class represents a realization.
    
    A realization is a map from the set of vertices to $\RR^d$. 
    The labeling is implicit and given by the placement's order.

    METHODS:
    
    .. autosummary::
        add_vertex
        add_vertex_list
        
    
    """
    placement = None
    dimension = None

    def __init__(self, points=[], dim=2):
        for vector in points:
            assert(len(vector))==dim
        self.placement = points
        self.dimension = dim
    
    def add_vertex(self, point):
        assert len(point)==self.dimension
        self.placement.append(point)

    def add_vertex_list(self, points):
        for vector in points:
            assert(len(vector))==self.dimension
        self.placement.append(points)

    
class Framework(object):
    """
    This class provides the functionality for frameworks. By definition, it is a tuple of a graph and a realization.
    
    METHODS:
    
    .. autosummary::
        add_vertex
        add_vertex_list
        add_edge
        add_edge_list
    """
    graph = None
    realization = None

    def __init__(self, p=[], d=2):
        self.realization = Realization(p, d)
        self.graph = nx.Graph()
    
    def add_vertex(self, point, label=None):
        if label == None:
            maxNode = max(self.graph.nodes) if len(self.graph.nodes)>0 else 0
            label = maxNode + 1
        self.realization.add_vertex(point)
        self.graph.add_node(label)
        
    def add_vertex_list(self, points, labels=[]):
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

    def add_edge_list(self, edges):
        for edge in edges:
            self.add_edge(edge)