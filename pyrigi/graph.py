""""
Module for rigidity related graph properties.
"""


import networkx as nx

class Graph(nx.Graph):
    '''
    Class representing a graph.
    '''


    def __init__(self, data):
        '''
        Constructor for Graph.
        It accepts a networkx graph as data, or a list of edges, or a pair of vertices and edges.
        '''
        self._part_of_framework = False

    def delete_vertex(self, vertex):
        if self._part_of_framework:
            raise AttributeError("This graph is part of a framework. Please use the corresponding method of the framework.")
        raise NotImplementedError()

    def delete_vertices(self, vertices):
        if self._part_of_framework:
            raise AttributeError("This graph is part of a framework. Please use the corresponding method of the framework.")
        raise NotImplementedError()

    def delete_edge(self, edge):
        raise NotImplementedError()

    def delete_edges(self, edges):
        raise NotImplementedError()
