""""
Module for rigidity related graph properties.
"""


import networkx as nx

class Graph(nx.Graph):
    '''
    Class representing a graph.
    '''

    @classmethod
    def from_vertices_and_edges(self, vertices, edges):
        raise NotImplementedError()

    @classmethod
    def from_vertices(self, vertices):
        return from_vertices_and_edges(self, vertices, [])
        raise NotImplementedError()

    def delete_vertex(self, vertex):
        raise NotImplementedError()

    def delete_vertices(self, vertices):
        raise NotImplementedError()

    def delete_edge(self, edge):
        raise NotImplementedError()

    def delete_edges(self, edges):
        raise NotImplementedError()

    def is_sparse(self, K, L):
        raise NotImplementedError()

    def is_tight(self, K, L):
        raise NotImplementedError()

    
    def k0_extension(self, dim=2):
        raise NotImplementedError()
    
    def k1_extension(self, dim=2):
        raise NotImplementedError()

    def k_extesions(self, dim=2):
        raise NotImplementedError()

    def is_vertex_redundantly_rigid(self, dim=2):
        is_k_vertex_redundantly_rigid(self, 1, dim):
        raise NotImplementedError()

    def is_k_vertex_redundantly_rigid(self, k, dim=2):
        raise NotImplementedError()

    def is_redundantly_rigid(self, dim=2):
        return is_k_redundantly_rigid(self, 1, dim):
        raise NotImplementedError()

    def is_k_redundantly_rigid(self, k, dim=2):
        raise NotImplementedError()

    def is_rigid(self, dim=2):
        raise NotImplementedError()

    def is_minimally_rigid(self, dim=2):
        raise NotImplementedError()

    def is_globally_rigid(self, dim=2):
        raise NotImplementedError()

    def is_Rd_dependent(self, dim=2):
        raise NotImplementedError()

    def is_Rd_independent(self, dim=2):
        raise NotImplementedError()

    def is_Rd_circuit(self, dim=2):
        raise NotImplementedError()