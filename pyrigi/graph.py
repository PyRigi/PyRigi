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

    def delete_vertex(self, vertex):
        raise NotImplementedError()

    def delete_vertices(self, vertices):
        raise NotImplementedError()

    def delete_edge(self, edge):
        raise NotImplementedError()

    def delete_edges(self, edges):
        raise NotImplementedError()
