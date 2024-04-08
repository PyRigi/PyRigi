""""
Module for rigidity related graph properties.
"""

from copy import deepcopy
from itertools import combinations
import networkx as nx
from typing import TypeVar, List, Tuple, Any, Hashable
from sympy import Matrix

GraphType = TypeVar("Graph")
Vertex = Hashable
Edge = Tuple[Vertex, Vertex] | List[Vertex]
    
class Graph(nx.Graph):
    '''
    Class representing a graph.
    '''

    @classmethod
    def from_vertices_and_edges(
            cls,
            vertices: List[Vertex],
            edges: List[Edge]) -> GraphType:
        raise NotImplementedError()

    @classmethod
    def from_vertices(cls, vertices: List[Vertex]) -> GraphType:
        return Graph.from_vertices_and_edges(vertices, [])

    def vertices(self) -> List[Vertex]:
        return self.nodes

    def delete_vertex(self, vertex: Vertex) -> None:
        self.remove_node(vertex)

    def delete_vertices(self, vertices: List[Vertex]) -> None:
        self.remove_nodes_from(vertices)

    def delete_edge(self, edge: Edge) -> None:
        self.remove_edge(*edge)

    def delete_edges(self, edges: List[Edge]) -> None:
        self.remove_edges_from(edges)

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

    def zero_extension(self, vertices: List[Vertex], dim: int = 2) -> None:
        """
        Parameters
        ----------
        Modifies self?
        """
        raise NotImplementedError()

    def one_extension(
            self,
            vertices: List[Vertex],
            edge: Edge,
            dim: int = 2) -> None:
        """
        Parameters
        ----------
        Modifies self?
        """
        raise NotImplementedError()

    def k_extension(
            self,
            k: int,
            vertices: List[Vertex],
            edges: Edge,
            dim: int = 2) -> None:
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
        vertex_subsets = list(combinations(self.vertices(), k))
        for set in vertex_subsets:
            G = deepcopy(self)
            G.delete_vertices(set)
            if not G.is_rigid(dim):
                return False
        return True

    def is_redundantly_rigid(self, dim: int = 2) -> bool:
        """ Remove every edge and call `is_rigid()`"""
        return self.is_k_redundantly_rigid(1, dim)
    
    def is_k_redundantly_rigid(self, k: int, dim: int = 2) -> bool:
        """ Remove every k-subset of edges and call `is_rigid()`"""
        edge_subsets = list(combinations(self.edges, k))
        for set in edge_subsets:
            G = deepcopy(self)
            G.delete_edges(set)
            if not G.is_rigid(dim):
                return False
        return True


    def is_rigid(self, dim: int = 2) -> bool:
        """
        Notes
        -----
        dim=1: Connectivity
        dim=2: Pebble-game/(2,3)-count
        dim>=1: Probabilistic Rigidity Matrix (maybe symbolic?)
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError("The dimension needs to be a positive integer!")
        elif dim==1:
            return self.is_connected()
        elif dim==2:
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        

    def is_minimally_rigid(self, dim: int = 2) -> bool:
        """
        Notes
        -----
        dim=1: Tree
        dim=2: Pebble-game/(2,3)-tight
        dim>=1: Probabilistic Rigidity Matrix (maybe symbolic?)
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError("The dimension needs to be a positive integer!")
        elif dim==1:
            return self.is_tree()
        elif dim==2:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def extension_sequence(self, dim: int = 2) -> Any:
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
        if not isinstance(dim, int) or dim < 1:
            raise TypeError("The dimension needs to be a positive integer!")
        elif dim==1:
            return self.node_connectivity() >= 2
        elif dim==2:
            return self.is_k_vertex_redundantly_rigid and self.node_connectivity() >= 3
        else:
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

    def maximal_rigid_subgraphs(self, dim: int = 2) -> List[GraphType]:
        """List vertex-maximal rigid subgraphs. We consider a subgraph 
        to be maximal, if it has the maximal possible amount of vertices."""
        minimal_subgraphs = []
        for k in range(len(self.vertices()), -1, 0):
            vertex_subsets = list(combinations(self.vertices(), k))
            for vertex_subset in vertex_subsets:
                G = self.subgraph(vertex_subset)
                if G.is_rigid():
                    minimal_subgraphs.push(G)
            if len(minimal_subgraphs) > 0:
                break
        return minimal_subgraphs

    def minimal_rigid_subgraphs(self, dim: int = 2) -> List[GraphType]:
        """List vertex-minimal non-trivial rigid subgraphs. We consider a subgraph 
        to be minimal, if it has the maximal possible amount of vertices."""
        minimal_subgraphs = []       
        for k in range(3, len(self.vertices())+1):
            vertex_subsets = list(combinations(self.vertices(), k))
            for vertex_subset in vertex_subsets:
                G = self.subgraph(vertex_subset)
                if G.is_rigid():
                    minimal_subgraphs.push(G)
            if len(minimal_subgraphs) > 0:
                break
        return minimal_subgraphs

    def is_isomorphic(self, graph: GraphType) -> bool:
        return nx.is_isomorphic(self, graph)

    def graph_to_int(self) -> int:
        r"""
        Return the integer representation of the graph.

        The graph integer representation is the integer
        whose binary expansion is given by the sequence
        obtained by concatenation of the rows
        of the upper triangle of the adjacency matrix,
        excluding the diagonal.

        TODO
        ----
        Implement taking canonical before computing the integer representation.
        Tests.
        Specify order of vertices.
        """
        M = nx.adjacency_matrix(self, weight=None).todense()
        upper_diag = [str(b)
                      for i, row in enumerate(M.tolist())
                      for b in row[i + 1:]]
        return int(''.join(upper_diag), 2)

    @classmethod
    def from_int(cls) -> GraphType:
        raise NotImplementedError()

    def adjacency_matrix(
            self,
            vertex_order: List[Vertex] | None = None) -> Matrix:
        """

        """
        try:
            if vertex_order is None:
                vertex_order = sorted(self.vertices())
            else:
                assert set(self.vertices()) == set(vertex_order)
        except TypeError as error:
            vertex_order = self.vertices()
        return nx.adjacency_matrix(
            self, nodelist=vertex_order, weight=None).todense()
