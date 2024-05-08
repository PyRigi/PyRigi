"""
Module for rigidity related graph properties.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from random import randrange

import networkx as nx
from sympy import Matrix, shape

from pyrigi.data_type import Vertex, Edge, GraphType, List, Any


class Graph(nx.Graph):
    '''
    Class representing a graph.

    Parameters
    ----------
    vertices:
        The graph's vertices can be labelled by any `Hashable`.
    edges:
        Edges are tuples of vertices. They can either be a tuple `(i,j)` or
        a list `[i,j]` with two entries.



    Examples
    --------
    >>> G = Graph([(0,1), (1,2), (2,3), (0,3)])
    >>> print(G)
    Vertices: [0, 1, 2, 3], Edges: [(0, 1), (0, 3), (1, 2), (2, 3)]
    >>> G_ = Graph()
    >>> G_.add_vertices([0,2,5,7,'a'])
    >>> G_.add_edges([(0,7), (2,5)])
    >>> print(G)
    Vertices: [0, 2, 5, 7, 'a'],    Edges: [(0, 7), (2, 5)]


    Notes
    -----
    This class inherits the class :class:`networkx.Graph`.
    Some of the inherited methods are for instance:

    .. autosummary::

        networkx.Graph.add_edge

    Many of the :doc:`NetworkX <networkx:index>` algorithms are implemented as functions,
    namely, a :class:`Graph` instance has to be passed as the first parameter.
    See for instance:

    .. autosummary::

        ~networkx.classes.function.degree
        ~networkx.classes.function.neighbors
        ~networkx.classes.function.non_neighbors
        ~networkx.classes.function.subgraph
        ~networkx.classes.function.edge_subgraph
        ~networkx.classes.function.edges
        ~networkx.algorithms.connectivity.edge_augmentation.is_k_edge_connected
        ~networkx.algorithms.components.is_connected
        ~networkx.algorithms.tree.recognition.is_tree

    The following links give more information on :class:`networkx.Graph` functionality:

    - :doc:`Graph display <networkx:reference/drawing>`
    - :doc:`Directed Graphs <networkx:reference/classes/digraph>`
    - :doc:`Linear Algebra on Graphs <networkx:reference/linalg>`
    - :doc:`A Database of some Graphs <networkx:reference/generators>`
    - :doc:`Reading and Writing Graphs <networkx:reference/readwrite/index>`
    - :doc:`Converting to and from other Data Formats <networkx:reference/convert>`
    '''

    def __str__(self) -> str:
        """
        Return the string representation of a graph.

        Notes
        -----
        We try to sort the vertices and edges in the graph. If this fails,
        the internal order is used instead.
        """
        try:
            vertices_str = str(sorted(self.nodes))
            edges_str = "["
            for edge in self.edges:
                if edge[0] < edge[1]:
                    edges_str += str(edge)
                else:
                    edges_str += str((edge[1], edge[0]))

                if not edge == list(self.edges)[len(self.edges) - 1]:
                    edges_str += ", "
            edges_str += "]"
        except BaseException:
            vertices_str = str(self.vertex_list())
            edges_str = str(self.edges)

        return 'Vertices: ' + vertices_str + ',\t'\
            + 'Edges: ' + edges_str

    @classmethod
    def from_vertices_and_edges(
            cls,
            vertices: List[Vertex],
            edges: List[Edge]) -> GraphType:
        """This method creates a graph from a list of vertices and edges."""
        G = Graph()
        G.add_nodes_from(vertices)
        for edge in edges:
            if len(edge) != 2 or \
                    not edge[0] in G.nodes or \
                    not edge[1] in G.nodes:
                raise TypeError(
                    "Edge {edge} does not have the correct format or has adjacent vertices the graph does not contain")
            G.add_edge(*edge)
        return G

    @classmethod
    def from_vertices(cls, vertices: List[Vertex]) -> GraphType:
        """Create a graph with no edges from a list of vertices."""
        return Graph.from_vertices_and_edges(vertices, [])

    @classmethod
    def complete_graph(cls, n: int) -> GraphType:
        """Generate a complete graph on $n$ vertices. The vertices are labeled via the list $0,...,n-1$."""
        if not isinstance(n, int) or n < 1:
            raise TypeError("n needs to be a positive integer")
        vertices = range(n)
        edges = combinations(vertices, 2)
        return Graph.from_vertices_and_edges(vertices, edges)

    @classmethod
    def complete_graph_on_vertices(cls, vertices: List[Vertex]) -> GraphType:
        """
        Generate a complete graph on `vertices`.
        """
        edges = combinations(vertices, 2)
        return Graph.from_vertices_and_edges(vertices, edges)

    def vertex_list(self) -> List[Vertex]:
        """Return the list of vertices."""
        return list(self.nodes)
    
    def edge_list(self) -> List[Edge]:
        """Return the list of edges"""
        return list(self.edges)

    def delete_vertex(self, vertex: Vertex) -> None:
        """Alias for :meth:`networkx.Graph.remove_node`."""
        self.remove_node(vertex)

    def delete_vertices(self, vertices: List[Vertex]) -> None:
        """Alias for :meth:`networkx.Graph.remove_nodes_from`."""
        self.remove_nodes_from(vertices)

    def delete_edge(self, edge: Edge) -> None:
        """Alias for :meth:`networkx.Graph.remove_edge`"""
        self.remove_edge(*edge)

    def delete_edges(self, edges: List[Edge]) -> None:
        """Alias for :meth:`networkx.Graph.remove_edges_from`."""
        self.remove_edges_from(edges)

    def add_vertex(self, vertex: Vertex) -> None:
        """Alias for :meth:`networkx.Graph.add_node`."""
        self.add_node(vertex)

    def add_vertices(self, vertices: List[Vertex]) -> None:
        """Alias for :meth:`networkx.Graph.add_nodes_from`."""
        self.add_nodes_from(vertices)

    def add_edges(self, edges: List[Edge]) -> None:
        """Alias for :meth:`networkx.Graph.add_edges_from`."""
        self.add_edges_from(edges)

    def vertex_connectivity(self) -> int:
        """Alias for :func:`networkx.algorithms.connectivity.connectivity.node_connectivity`."""
        return nx.node_connectivity(self)

    def is_sparse(self, K: int, L: int) -> bool:
        r"""
        Check whether the graph is :prf:ref:`(K, L)-sparse <def-kl-sparse-tight>`.
        """
        if not (isinstance(K, int) and isinstance(L, int)):
            raise TypeError("K and L need to be integers!")

        for j in range(K, self.order() + 1):
            for vertex_set in combinations(self.nodes, j):
                G = self.subgraph(vertex_set)
                if len(G.edges) > K * len(G.nodes) - L:
                    return False
        return True

    def is_tight(self, K: int, L: int) -> bool:
        r"""
        Check whether the graph is :prf:ref:`(K, L)-tight <def-kl-sparse-tight>`.
        """
        return self.is_sparse(K, L) and \
            len(self.edges) <= K * len(self.nodes) - L

    def zero_extension(self, vertices: List[Vertex], dim: int = 2) -> None:
        """
        Notes
        -----
        Modifies self only when explicitly required.
        """
        raise NotImplementedError()

    def one_extension(
            self,
            vertices: List[Vertex],
            edge: Edge,
            dim: int = 2) -> None:
        """
        Notes
        -----
        Modifies self only when explicitly required.
        """
        raise NotImplementedError()

    def k_extension(
            self,
            k: int,
            vertices: List[Vertex],
            edges: Edge,
            dim: int = 2) -> None:
        """
        Notes
        -----
        Modifies self only when explicitly required.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")
        raise NotImplementedError()

    def all_k_extensions(self, k: int, dim: int = 2) -> None:
        """
        Return list of all possible k-extensions of the graph.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")
        raise NotImplementedError()

    def extension_sequence(self, dim: int = 2) -> Any:
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")
        raise NotImplementedError()

    def is_vertex_redundantly_rigid(self, dim: int = 2) -> bool:
        """
        Check whether the graph is :prf:ref:`vertex redundantly (generically) dim-rigid <def-minimally-redundantly-rigid-graph>`.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")
        return self.is_k_vertex_redundantly_rigid(1, dim)

    def is_k_vertex_redundantly_rigid(self, k: int, dim: int = 2) -> bool:
        """
        Check whether the graph is :prf:ref:`k-vertex redundantly (generically) dim-rigid <def-minimally-redundantly-rigid-graph>`.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")
        if not isinstance(k, int):
            raise TypeError(f"k needs to be a nonnegative integer, but is {k}!")
        for vertex_set in combinations(self.nodes, k):
            G = deepcopy(self)
            G.delete_vertices(vertex_set)
            if not G.is_rigid(dim):
                return False
        return True

    def is_redundantly_rigid(self, dim: int = 2) -> bool:
        """
        Check whether the graph is :prf:ref:`redundantly (generically) dim-rigid <def-minimally-redundantly-rigid-graph>`.
        """
        return self.is_k_redundantly_rigid(1, dim)

    def is_k_redundantly_rigid(self, k: int, dim: int = 2) -> bool:
        """
        Check whether the graph is :prf:ref:`k-redundantly (generically) dim-rigid <def-minimally-redundantly-rigid-graph>`.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")
        if not isinstance(k, int):
            raise TypeError(f"k needs to be a nonnegative integer, but is {k}!")
        for edge_set in combinations(self.edges, k):
            G = deepcopy(self)
            G.delete_edges(edge_set)
            if not G.is_rigid(dim):
                return False
        return True

    def is_rigid(self, dim: int = 2, combinatorial: bool = True) -> bool:
        """
        Check whether the graph is :prf:ref:`(generically) dim-rigid <def-gen-rigid>`.

        Notes
        -----
         * dim=1: Connectivity
         * dim=2: Pebble-game/(2,3)-rigidity
         * dim>=1: Rigidity Matrix if `combinatorial==False`
        By default, the graph is in dimension two and a combinatorial check is employed.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (3,0)])
        >>> G.is_rigid()
        False
        >>> G.add_edge(0,2)
        >>> G.is_rigid()
        True
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")
        if not isinstance(combinatorial, bool):
            raise TypeError(
                f"combinatorial determines the method of rigidity-computation. It needs to be a Boolean.")

        elif dim == 1:
            return self.is_connected()
        elif dim == 2 and combinatorial:
            deficiency = -(2 * self.order() - 3) + len(self.edges)
            if deficiency < 0:
                return False
            else:
                for edge_subset in combinations(self.edges, deficiency):
                    H = self.edge_subgraph(
                        [edge for edge in self.edges if edge not in edge_subset])
                    if H.is_tight(2, 3):
                        return True
                return False
        elif not combinatorial:
            from pyrigi.framework import Framework
            N = 10 * self.order()**2 * dim
            realization = {
                vertex: [
                    randrange(
                        1,
                        N) for _ in range(
                        0,
                        dim)] for vertex in self.nodes}
            F = Framework(self, realization, dim)
            return F.is_infinitesimally_rigid()
        else:
            raise ValueError(
                f"The Dimension for combinatorial computation must be either 1 or 2, but is {dim}")

    def is_minimally_rigid(
            self,
            dim: int = 2,
            combinatorial: bool = True) -> bool:
        """
        Check whether the graph is :prf:ref:`minimally (generically) dim-rigid <def-minimally-redundantly-rigid-graph>`.

        Notes
        -----
         * dim=1: Tree
         * dim=2: Pebble-game/(2,3)-tight
         * dim>=1: Probabilistic Rigidity Matrix (maybe symbolic?)
        By default, the graph is in dimension 2 and a combinatorial algorithm is applied.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (3,0), (1,3)])
        >>> G.is_minimally_rigid()
        True
        >>> G.add_edge(0,2)
        >>> G.is_minimally_rigid()
        False
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")
        if not isinstance(combinatorial, bool):
            raise TypeError(
                f"combinatorial determines the method of rigidity-computation. It needs to be a Boolean.")

        elif dim == 1:
            return self.is_tree()
        elif dim == 2 and combinatorial:
            return self.is_tight(2, 3)
        elif not combinatorial:
            from pyrigi.framework import Framework
            N = 10 * self.order()**2 * dim
            realization = {
                vertex: [
                    randrange(
                        1,
                        N) for _ in range(
                        0,
                        dim)] for vertex in self.nodes}
            F = Framework(self, realization, dim)
            return F.is_minimally_infinitesimally_rigid()
        else:
            raise ValueError(
                f"The dimension for combinatorial computation must be either 1 or 2, but is {dim}")

    def is_globally_rigid(self, dim: int = 2) -> bool:
        """
        Check whether the graph is :prf:ref:`globally dim-rigid <def-globally-rigid-graph>`

        Notes
        -----
         * dim=1: 2-connectivity
         * dim=2: redundantly rigid+3-connected
         * dim>=3: Randomized Rigidity Matrix => Stress (symbolic maybe?)
        By default, the graph is in dimension 2.
        A complete graph is automatically globally rigid

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,0)])
        >>> G.is_globally_rigid()
        True
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")

        elif dim == 1:
            if (len(self.nodes) == 2 and len(self.edges) == 1) or \
                    (len(self.nodes) == 1 or len(self.nodes) == 0):
                return True
            return self.vertex_connectivity() >= 2
        elif dim == 2:
            if (len(self.nodes) == 3 and len(self.edges) == 3) or \
                (len(self.nodes) == 2 and len(self.edges) == 1) or \
                    (len(self.nodes) == 1 or len(self.nodes) == 0):
                return True
            return self.is_redundantly_rigid() and self.vertex_connectivity() >= 3
        else:
            from pyrigi.framework import Framework
            # Random sampling from [1,N] for N depending quadratically on number
            # of vertices.
            raise NotImplementedError()

    def is_Rd_dependent(self, dim: int = 2) -> bool:
        """
        Notes
        -----
         * dim=1: Graphic Matroid
         * dim=2: not (2,3)-sparse
         * dim>=1: Compute the rank of the rigidity matrix and compare with edge count
        """
        raise NotImplementedError()

    def is_Rd_independent(self, dim: int = 2) -> bool:
        """
        Notes
        -----
         * dim=1: Graphic Matroid
         * dim=2: (2,3)-sparse
         * dim>=1: Compute the rank of the rigidity matrix and compare with edge count
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")
        raise NotImplementedError()

    def is_Rd_circuit(self, dim: int = 2) -> bool:
        """
        Notes
        -----
         * dim=1: Graphic Matroid
         * dim=2: Remove any edge and it becomes sparse (sparsity for every subgraph except whole graph?)
         * dim>=1: Dependent + Remove every edge and compute the rigidity matrix' rank
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")
        raise NotImplementedError()

    def is_Rd_closed(self, dim: int = 2) -> bool:
        """
        Notes
        -----
         * dim=1: Graphic Matroid
         * dim=2: ??
         * dim>=1: Adding any edge does not increase the rigidity matrix rank
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")
        raise NotImplementedError()

    def maximal_rigid_subgraphs(self, dim: int = 2) -> List[GraphType]:
        """
        List vertex-maximal rigid subgraphs of the graph.

        Definitions
        -----
        :prf:ref:`Maximal rigid subgraph <def-maximal-rigid-subgraph>`

        Notes
        -----
        We only return nontrivial subgraphs, meaning that there need to be at
        least `dim+1` vertices present. If the graph itself is rigid, it is clearly
        maximal and is returned.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (3,0)])
        >>> G.maximal_rigid_subgraphs()
        []
        >>> G_ = Graph([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0), (0,2), (5,3)])
        >>> G_.is_rigid()
        False
        >>> [print(entry) for entry in G.maximal_rigid_subgraphs()]
        Vertices: [0, 1, 2],    Edges: [(0, 1), (0, 2), (1, 2)]
        Vertices: [3, 4, 5],    Edges: [(3, 4), (3, 5), (4, 5)]
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")

        if self.order() <= dim:
            return []
        if self.is_rigid():
            return [self]
        maximal_subgraphs = []
        for vertex_subset in combinations(
            self.nodes, self.order() - 1):
            G = self.subgraph(vertex_subset)
            maximal_subgraphs = [
                j for i in [
                    maximal_subgraphs,
                    G.maximal_rigid_subgraphs(dim)] for j in i]

        # We now remove the graphs that were found at least twice.
        clean_list = []
        for i in range(0, len(maximal_subgraphs)):
            iso_bool = False
            for j in range(i + 1, len(maximal_subgraphs)):
                if set(
                        maximal_subgraphs[i].nodes) == set(
                        maximal_subgraphs[j].nodes) and maximal_subgraphs[i].is_isomorphic(
                        maximal_subgraphs[j]):
                    iso_bool = True
                    break
            if not iso_bool:
                clean_list.append(maximal_subgraphs[i])
        return clean_list

    def minimal_rigid_subgraphs(self, dim: int = 2) -> List[GraphType]:
        """
        List vertex-minimal non-trivial rigid subgraphs of the graph.

        Definitions
        -----
        :prf:ref:`Minimal rigid subgraph <def-minimal-rigid-subgraph>`

        Notes
        -----
        We only return nontrivial subgraphs, meaning that there need to be at
        least `dim+1` vertices present.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0), (0,3), (4,1), (5,2)])
        >>> G.is_rigid()
        True
        >>> [print(entry) for entry in G.minimal_rigid_subgraphs()]
        Vertices: [0, 1, 2, 3, 4, 5],   Edges: [(0, 1), (0, 5), (0, 3), (1, 2), (1, 4), (2, 3), (2, 5), (3, 4), (4, 5)]
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")

        minimal_subgraphs = []
        if self.order() <= 2:
            return []
        elif self.order() == dim + 1 and self.is_rigid():
            return [self]
        elif self.order() == dim + 1:
            return []
        for vertex_subset in combinations(
            self.nodes, self.order() - 1):
            G = self.subgraph(vertex_subset)
            subgraphs = G.minimal_rigid_subgraphs(dim)
            if len(subgraphs) == 0 and G.is_rigid():
                minimal_subgraphs.append(G)
            else:
                minimal_subgraphs = [
                    j for i in [
                        minimal_subgraphs,
                        G.minimal_rigid_subgraphs(dim)] for j in i]

        # We now remove the graphs that were found at least twice.
        clean_list = []
        for i in range(0, len(minimal_subgraphs)):
            iso_bool = False
            for j in range(i + 1, len(minimal_subgraphs)):
                if set(
                        minimal_subgraphs[i].nodes) == set(
                        minimal_subgraphs[j].nodes) and minimal_subgraphs[i].is_isomorphic(
                        minimal_subgraphs[j]):
                    iso_bool = True
                    break
            if not iso_bool:
                clean_list.append(minimal_subgraphs[i])
        # If no smaller graph is found and the graph is rigid, it is returned.
        if not clean_list and self.is_rigid():
            clean_list = [self]
        return clean_list

    def is_isomorphic(self, graph: GraphType) -> bool:
        """
        Check whether two graphs are isomorphic.

        Notes
        -----
        For further details, see :func:`networkx.algorithms.isomorphism.is_isomorphic`.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2)])
        >>> G_ = Graph([('b','c'), ('c','a')])
        >>> G.is_isomorphic(G_)
        True
        """
        return nx.is_isomorphic(self, graph)

    def graph_to_int(self) -> int:
        r"""
        Return the integer representation of the graph.

        Notes
        -----
        The graph integer representation is the integer whose binary
        expansion is given by the sequence obtained by concatenation
        of the rows of the upper triangle of the adjacency matrix,
        excluding the diagonal.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2)])
        >>> G.adjacency_matrix()
        Matrix([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]])
        >>> G.graph_to_int()
        5

        TODO
        ----
        Implement taking canonical before computing the integer representation.
        Tests.
        Specify order of vertices.
        """
        M = self.adjacency_matrix()
        upper_diag = [str(b)
                      for i, row in enumerate(M.tolist())
                      for b in row[i + 1:]]
        return int(''.join(upper_diag), 2)

    @classmethod
    def from_int(cls, n: int) -> GraphType:
        """
        Return a graph given its integer representation.

        Notes
        -----
        See :meth:`graph_to_int`.

        TODO
        -----
        binary_representation = int(bin(n)[2:])
        Graph.from_adjacency_matrix(...)
        """
        raise NotImplementedError()

    @classmethod
    def from_adjacency_matrix(cls, M: Matrix) -> GraphType:
        """
        Create a graph from a given adjacency matrix.

        Examples
        --------
        >>> M = Matrix([[0,1],[1,0]])
        >>> G = Graph.from_adjacency_matrix(M)
        >>> print(G)
        Vertices: [0, 1],       Edges: []
        """
        if not shape(M)[0] == shape(M)[1]:
            raise TypeError("Adjacency matrix does not have the right format!")
        for i, j in zip(range(shape(M)[0]), range(shape(M)[1])):
            if not (M[i, j] == 0 or M[i, j] == 1):
                raise TypeError(
                    "The provided adjancency matrix contains entries other than 0 and 1")
        vertices = range(shape(M)[0])
        edges = []
        for vertex, vertex_ in zip(range(len(vertices)), range(len(vertices))):
            if M[vertex, vertex_] == 1:
                edges += [(vertex, vertex_)]
        return Graph.from_vertices_and_edges(vertices, edges)

    def adjacency_matrix(
            self,
            vertex_order: List[Vertex] = None) -> Matrix:
        """
        Return the adjacency matrix of the graph.

        Parameters
        ----------
        vertex_order:
            By listing vertices in the preferred order, the adjacency matrix
            can be computed in a way the user expects. If no vertex order is
            provided, the internal order is assumed.

        Notes
        -----
        :func:`networkx.linalg.graphmatrix.adjacency_matrix`
        requires `scipy`. To avoid unnecessary imports, the method is implemented here.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (1,3)])
        >>> G.adjacency_matrix()
        Matrix([
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 0]])
        """
        try:
            if vertex_order is None:
                vertex_order = sorted(self.nodes)
            else:
                if (not set(self.nodes) == set(vertex_order)
                    or not self.order() == len(vertex_order)):
                    raise IndexError(
                        "The vertex_order needs to contain the same vertices as the graph!")
        except TypeError as error:
            vertex_order = self.vertex_list()

        row_list = []
        for vertex in vertex_order:
            row = []
            edge_indicator = False
            for vertex_ in vertex_order:
                for edge in self.edges:
                    if (edge[0] == vertex and edge[1] == vertex_) or \
                            (edge[1] == vertex and edge[0] == vertex_):
                        row += [1]
                        edge_indicator = True
                        break
                if not edge_indicator:
                    row += [0]
            row_list += [row]
        return Matrix(row_list)
