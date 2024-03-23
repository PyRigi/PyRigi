""""
Module for rigidity related graph properties.
"""

import networkx as nx


class Graph(nx.Graph):
    '''
    Class representing a graph.
    '''

    @classmethod
    def from_vertices_and_edges(cls, vertices: list[any], edges: list[tuple[int,int]]) -> None:
        raise NotImplementedError()

    @classmethod
    def from_vertices(cls, vertices: list[any]) -> None:
        Graph.from_vertices_and_edges(vertices, [])

    def vertices(self) -> list[any]:
        return self.nodes

    def delete_vertex(self, vertex: any) -> None:
        raise NotImplementedError()

    def delete_vertices(self, vertices: list[any]) -> None:
        raise NotImplementedError()

    def delete_edge(self, edge: tuple[int,int]) -> None:
        raise NotImplementedError()

    def delete_edges(self, edges: list[tuple[int,int]]) -> None:
        raise NotImplementedError()

    def is_sparse(self, K: int, L: int) -> bool:
        """
        Notes
        -----
        Combinatorial Property
        """
        raise NotImplementedError()

    def is_tight(self, K: int, L: int) -> bool:
        """
        Notes
        -----
        Combinatorial Property
        """
        raise NotImplementedError()

    def zero_extension(self, vertices: list[any], dim: int = 2) -> None:
        """
        Parameters
        ----------
        Modifies self?
        """
        raise NotImplementedError()

    def one_extension(self, vertices: list[any], edge: tuple[int,int], dim: int = 2) -> None:
        """
        Parameters
        ----------
        Modifies self?
        """
        raise NotImplementedError()

    def k_extension(self, k: int, vertices: list[any], edges: tuple[int,int], dim: int = 2) -> None:
        """
        Parameters
        ----------
        Modifies self?
        """
        raise NotImplementedError()

    def all_k_extensions(self, k: int, dim: int = 2) -> None:
        """
        Parameters
        ----------
        Modifies self?
        """
        raise NotImplementedError()

    def is_vertex_redundantly_rigid(self, dim: int = 2) -> bool:
        """ Remove every vertex and call `is_rigid()`"""
        return self.is_k_vertex_redundantly_rigid(1, dim)

    def is_k_vertex_redundantly_rigid(self, k: int, dim: int = 2) -> bool:
        """ Remove every k-subset of vertices and call `is_rigid()`"""
        raise NotImplementedError()

    def is_redundantly_rigid(self, dim: int = 2) -> bool:
        """ Remove every edge and call `is_rigid()`"""
        return self.is_k_redundantly_rigid(1, dim)

    def is_k_redundantly_rigid(self, k: int, dim: int = 2) -> bool:
        """ Remove every k-subset of edges and call `is_rigid()`"""
        raise NotImplementedError()

    def is_rigid(self, dim: int = 2) -> bool:
        """
        Notes
        -----
        dim=1: Connectivity
        dim=2: Pebble-game/(2,3)-count
        dim>=1: Probabilistic Rigidity Matrix (maybe symbolic?)
        """
        raise NotImplementedError()

    def is_minimally_rigid(self, dim: int = 2) -> bool:
        """
        Notes
        -----
        dim=1: Tree
        dim=2: Pebble-game/(2,3)-tight
        dim>=1: Probabilistic Rigidity Matrix (maybe symbolic?)
        """
        raise NotImplementedError()

    def extension_sequence(self, dim: int = 2) -> any:
        raise NotImplementedError()

    # def pebble_game(self, dim=2):    raise NotImplementedError()

    # def two_spanning_trees(self):    raise NotImplementedError()

    # def three_trees(self):           raise NotImplementedError()

    def is_globally_rigid(self, dim: int = 2) -> bool:
        """
        Notes
        -----
        dim=1: 2-connectivity
        dim=2: redundantly rigid+3-connected
        dim>=1: Randomized Rigidity Matrix => Stress (symbolic maybe?)
        """
        raise NotImplementedError()

    def is_Rd_dependent(self, dim: int = 2) -> bool:
        """
        Notes
        -----
        dim=1: Graphic Matroid
        dim=2: not (2,3)-sparse
        dim>=1: Compute the rank of the rigidity matrix and compare with edge count
        """
        raise NotImplementedError()

    def is_Rd_independent(self, dim: int = 2) -> bool:
        """
        Notes
        -----
        dim=1: Graphic Matroid
        dim=2: (2,3)-sparse
        dim>=1: Compute the rank of the rigidity matrix and compare with edge count
        """
        raise NotImplementedError()

    def is_Rd_circuit(self, dim: int = 2) -> bool:
        """
        Notes
        -----
        dim=1: Graphic Matroid
        dim=2: Remove any edge and it becomes sparse (sparsity for every subgraph except whole graph?)
        dim>=1: Dependent + Remove every edge and compute the rigidity matrix' rank
        """
        raise NotImplementedError()

    def is_Rd_closed(self, dim: int = 2) -> bool:
        """
        Notes
        -----
        dim=1: Graphic Matroid
        dim=2: Ask Bill
        dim>=1: Adding any edge does not increase the rigidity matrix rank
        """
        raise NotImplementedError()

    def maximal_rigid_subgraphs(self, dim: int = 2) -> any:
        """List subgraph-maximal rigid subgraphs."""
        raise NotImplementedError()

    def minimal_rigid_subgraphs(self, dim: int = 2) -> any:
        """List subgraph-minimal non-trivial (?) rigid subgraphs."""
        raise NotImplementedError()

    def is_isomorphic(self, graph: nx.Graph) -> bool:
        return nx.is_isomorphic(self, graph)

    def graph_to_int(self) -> int:
        raise NotImplementedError()

    @classmethod
    def from_int(cls) -> nx.Graph:
        raise NotImplementedError()
