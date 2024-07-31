"""
Module for rigidity related graph properties.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import List, Any, Union

import networkx as nx
import matplotlib.pyplot as plt
from sympy import Matrix
import math

from pyrigi.data_type import Vertex, Edge, GraphType, FrameworkType, Point
from pyrigi.misc import doc_category, generate_category_tables
from pyrigi.exception import LoopError


class Graph(nx.Graph):
    """
    Class representing a graph.

    One option for *incoming_graph_data* is a list of edges.
    See :class:`networkx.Graph` for the other input formats
    or use class methods :meth:`~Graph.from_vertices_and_edges`
    or :meth:`~Graph.from_vertices` when specifying the vertex set is needed.

    Examples
    --------
    >>> from pyrigi import Graph
    >>> G = Graph([(0,1), (1,2), (2,3), (0,3)])
    >>> print(G)
    Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 3], [1, 2], [2, 3]]

    >>> G = Graph()
    >>> G.add_vertices([0,2,5,7,'a'])
    >>> G.add_edges([(0,7), (2,5)])
    >>> print(G)
    Graph with vertices [0, 2, 5, 7, 'a'] and edges [[0, 7], [2, 5]]

    TODO
    ----
    Implement an alias for plotting.
    Graphical output in Jupyter.
    Graph names.

    METHODS

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
    """

    def __str__(self) -> str:
        """
        Return the string representation.
        """
        return (
            self.__class__.__name__
            + f" with vertices {self.vertex_list()} and edges {self.edge_list()}"
        )

    def __repr__(self) -> str:
        """
        Return a representation.
        """
        return self.__str__()

    @classmethod
    @doc_category("Class methods")
    def from_vertices_and_edges(
        cls, vertices: List[Vertex], edges: List[Edge]
    ) -> GraphType:
        """
        Create a graph from a list of vertices and edges.

        Parameters
        ----------
        vertices
        edges:
            Edges are tuples of vertices. They can either be a tuple ``(i,j)`` or
            a list ``[i,j]`` with two entries.

        TODO
        ----
        examples, tests
        """
        G = Graph()
        G.add_nodes_from(vertices)
        for edge in edges:
            if len(edge) != 2 or not edge[0] in G.nodes or not edge[1] in G.nodes:
                raise TypeError(
                    f"Edge {edge} does not have the correct format "
                    "or has adjacent vertices the graph does not contain"
                )
            G.add_edge(*edge)
        return G

    @classmethod
    @doc_category("Class methods")
    def from_vertices(cls, vertices: List[Vertex]) -> GraphType:
        """
        Create a graph with no edges from a list of vertices.

        Examples
        --------
        >>> from pyrigi import Graph
        >>> G = Graph.from_vertices([3, 1, 7, 2, 12, 3, 0])
        >>> G
        Graph with vertices [0, 1, 2, 3, 7, 12] and edges []
        """
        return Graph.from_vertices_and_edges(vertices, [])

    @classmethod
    @doc_category("Class methods")
    def CompleteOnVertices(cls, vertices: List[Vertex]) -> GraphType:
        """
        Generate a complete graph on ``vertices``.

        TODO
        ----
        examples, tests
        """
        edges = combinations(vertices, 2)
        return Graph.from_vertices_and_edges(vertices, edges)

    @doc_category("Attribute getters")
    def vertex_list(self) -> List[Vertex]:
        """
        Return the list of vertices.

        The output is sorted if possible,
        otherwise, the internal order is used instead.

        TODO
        ----
        examples
        """
        try:
            return sorted(self.nodes)
        except BaseException:
            return list(self.nodes)

    @doc_category("Attribute getters")
    def edge_list(self) -> List[Edge]:
        """
        Return the list of edges.

        The output is sorted if possible,
        otherwise, the internal order is used instead.

        TODO
        ----
        examples
        """
        try:
            return sorted([sorted(e) for e in self.edges])
        except BaseException:
            return list(self.edges)

    @doc_category("Graph manipulation")
    def delete_vertex(self, vertex: Vertex) -> None:
        """Alias for :meth:`networkx.Graph.remove_node`."""
        self.remove_node(vertex)

    @doc_category("Graph manipulation")
    def delete_vertices(self, vertices: List[Vertex]) -> None:
        """Alias for :meth:`networkx.Graph.remove_nodes_from`."""
        self.remove_nodes_from(vertices)

    @doc_category("Graph manipulation")
    def delete_edge(self, edge: Edge) -> None:
        """Alias for :meth:`networkx.Graph.remove_edge`"""
        self.remove_edge(*edge)

    @doc_category("Graph manipulation")
    def delete_edges(self, edges: List[Edge]) -> None:
        """Alias for :meth:`networkx.Graph.remove_edges_from`."""
        self.remove_edges_from(edges)

    @doc_category("Graph manipulation")
    def add_vertex(self, vertex: Vertex) -> None:
        """Alias for :meth:`networkx.Graph.add_node`."""
        self.add_node(vertex)

    @doc_category("Graph manipulation")
    def add_vertices(self, vertices: List[Vertex]) -> None:
        """Alias for :meth:`networkx.Graph.add_nodes_from`."""
        self.add_nodes_from(vertices)

    @doc_category("Graph manipulation")
    def add_edges(self, edges: List[Edge]) -> None:
        """Alias for :meth:`networkx.Graph.add_edges_from`."""
        self.add_edges_from(edges)

    @doc_category("Graph manipulation")
    def delete_loops(self) -> None:
        """Removes all the loops from the edges to get a loop free graph."""
        self.delete_edges(nx.selfloop_edges(self))

    @doc_category("General graph theoretical properties")
    def vertex_connectivity(self) -> int:
        """Alias for :func:`networkx.algorithms.connectivity.connectivity.node_connectivity`."""  # noqa: E501
        return nx.node_connectivity(self)

    @doc_category("General graph theoretical properties")
    def degree_sequence(self, vertex_order: List[Vertex] = None) -> list[int]:
        """
        Return a list of degrees of the vertices of the graph.

        Parameters
        ----------
        vertex_order:
            By listing vertices in the preferred order, the degree_sequence
            can be computed in a way the user expects. If no vertex order is
            provided, :meth:`~.Graph.vertex_list()` is used.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2)])
        >>> G.degree_sequence()
        [1, 2, 1]
        """
        if vertex_order is None:
            vertex_order = self.vertex_list()
        else:
            if not set(self.nodes) == set(
                vertex_order
            ) or not self.number_of_nodes() == len(vertex_order):
                raise IndexError(
                    "The vertex_order must contain the same vertices as the graph!"
                )
        return [self.degree(v) for v in vertex_order]

    @doc_category("General graph theoretical properties")
    def min_degree(self) -> int:
        """
        Return the minimum of the vertex degrees.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2)])
        >>> G.min_degree()
        1
        """
        return min([self.degree(v) for v in self.nodes])

    @doc_category("General graph theoretical properties")
    def max_degree(self) -> int:
        """
        Return the maximum of the vertex degrees.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2)])
        >>> G.max_degree()
        2
        """
        return max([self.degree(v) for v in self.nodes])

    @doc_category("Sparseness")
    def is_sparse(self, K: int, L: int) -> bool:
        r"""
        Check whether the graph is :prf:ref:`(K, L)-sparse <def-kl-sparse-tight>`.

        TODO
        ----
        pebble game algorithm, examples, tests for other cases than (2,3)
        """
        if not (isinstance(K, int) and isinstance(L, int)):
            raise TypeError("K and L need to be integers!")

        for j in range(K, self.number_of_nodes() + 1):
            for vertex_set in combinations(self.nodes, j):
                G = self.subgraph(vertex_set)
                if G.number_of_edges() > K * G.number_of_nodes() - L:
                    return False
        return True

    @doc_category("Sparseness")
    def is_tight(self, K: int, L: int) -> bool:
        r"""
        Check whether the graph is :prf:ref:`(K, L)-tight <def-kl-sparse-tight>`.

        TODO
        ----
        examples, tests for other cases than (2,3)
        """
        return (
            self.is_sparse(K, L)
            and self.number_of_edges() == K * self.number_of_nodes() - L
        )

    @doc_category("Waiting for implementation")
    def zero_extension(self, vertices: List[Vertex], dim: int = 2) -> None:
        """
        Notes
        -----
        Modifies self only when explicitly required.
        """
        raise NotImplementedError()

    @doc_category("Waiting for implementation")
    def one_extension(self, vertices: List[Vertex], edge: Edge, dim: int = 2) -> None:
        """
        Notes
        -----
        Modifies self only when explicitly required.
        """
        raise NotImplementedError()

    @doc_category("Waiting for implementation")
    def k_extension(
        self, k: int, vertices: List[Vertex], edges: Edge, dim: int = 2
    ) -> None:
        """
        Notes
        -----
        Modifies self only when explicitly required.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        raise NotImplementedError()

    @doc_category("Waiting for implementation")
    def all_k_extensions(self, k: int, dim: int = 2) -> None:
        """
        Return list of all possible k-extensions of the graph.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        raise NotImplementedError()

    @doc_category("Waiting for implementation")
    def extension_sequence(self, dim: int = 2) -> Any:
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        raise NotImplementedError()

    @doc_category("Generic rigidity")
    def is_vertex_redundantly_rigid(self, dim: int = 2) -> bool:
        """
        Check whether the graph is :prf:ref:`vertex redundantly (generically) dim-rigid
        <def-redundantly-rigid-graph>`.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        return self.is_k_vertex_redundantly_rigid(1, dim)

    @doc_category("Generic rigidity")
    def is_k_vertex_redundantly_rigid(self, k: int, dim: int = 2) -> bool:
        """
        Check whether the graph is :prf:ref:`k-vertex redundantly (generically) dim-rigid
        <def-redundantly-rigid-graph>`.

        TODO
        ----
        Avoid creating deepcopies by remembering the edges.
        Tests, examples.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if not isinstance(k, int):
            raise TypeError(f"k needs to be a nonnegative integer, but is {k}!")
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()
        for vertex_set in combinations(self.nodes, k):
            G = deepcopy(self)
            G.delete_vertices(vertex_set)
            if not G.is_rigid(dim):
                return False
        return True

    @doc_category("Generic rigidity")
    def is_redundantly_rigid(self, dim: int = 2) -> bool:
        """
        Check whether the graph is :prf:ref:`redundantly (generically) dim-rigid
        <def-redundantly-rigid-graph>`.
        """
        return self.is_k_redundantly_rigid(1, dim)

    @doc_category("Generic rigidity")
    def is_k_redundantly_rigid(self, k: int, dim: int = 2) -> bool:
        """
        Check whether the graph is :prf:ref:`k-redundantly (generically) dim-rigid
        <def-redundantly-rigid-graph>`.

        TODO
        ----
        Tests, examples.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if not isinstance(k, int):
            raise TypeError(f"k needs to be a nonnegative integer, but is {k}!")
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()
        for edge_set in combinations(self.edge_list(), k):
            self.delete_edges(edge_set)
            if not self.is_rigid(dim):
                self.add_edges(edge_set)
                return False
            self.add_edges(edge_set)
        return True

    @doc_category("Generic rigidity")
    def is_rigid(self, dim: int = 2, combinatorial: bool = True) -> bool:
        """
        Check whether the graph is :prf:ref:`(generically) dim-rigid <def-gen-rigid>`.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (3,0)])
        >>> G.is_rigid()
        False
        >>> G.add_edge(0,2)
        >>> G.is_rigid()
        True

        TODO
        ----
        Pebble game algorithm for d=2.

        Notes
        -----
         * dim=1: Connectivity
         * dim=2: Pebble-game/(2,3)-rigidity
         * dim>=1: Rigidity Matrix if ``combinatorial==False``
        By default, the graph is in dimension two and a combinatorial check is employed.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if not isinstance(combinatorial, bool):
            raise TypeError(
                "combinatorial determines the method of rigidity-computation. "
                "It needs to be a Boolean."
            )
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()

        elif dim == 1:
            return self.is_connected()
        elif dim == 2 and combinatorial:
            deficiency = -(2 * self.number_of_nodes() - 3) + self.number_of_edges()
            if deficiency < 0:
                return False
            else:
                for edge_subset in combinations(self.edges, deficiency):
                    H = self.edge_subgraph(
                        [edge for edge in self.edges if edge not in edge_subset]
                    )
                    if H.is_tight(2, 3):
                        return True
                return False
        elif not combinatorial:
            from pyrigi.framework import Framework

            F = Framework.Random(self, dim)
            return F.is_inf_rigid()
        else:
            raise ValueError(
                f"The Dimension for combinatorial computation must be either 1 or 2, "
                f"but is {dim}"
            )

    @doc_category("Generic rigidity")
    def is_min_rigid(self, dim: int = 2, combinatorial: bool = True) -> bool:
        """
        Check whether the graph is :prf:ref:`minimally (generically) dim-rigid
        <def-min-rigid-graph>`.

        By default, the graph is in dimension 2 and a combinatorial algorithm is applied.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (3,0), (1,3)])
        >>> G.is_min_rigid()
        True
        >>> G.add_edge(0,2)
        >>> G.is_min_rigid()
        False

        Notes
        -----
         * dim=1: Tree
         * dim=2: Pebble-game/(2,3)-tight
         * dim>=1: Probabilistic Rigidity Matrix (maybe symbolic?)
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if not isinstance(combinatorial, bool):
            raise TypeError(
                "combinatorial determines the method of rigidity-computation. "
                "It needs to be a Boolean."
            )
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()

        elif dim == 1:
            return self.is_tree()
        elif dim == 2 and combinatorial:
            return self.is_tight(2, 3)
        elif not combinatorial:
            from pyrigi.framework import Framework

            F = Framework.Random(self, dim)
            return F.is_min_inf_rigid()
        else:
            raise ValueError(
                f"The dimension for combinatorial computation must be either 1 or 2, "
                f"but is {dim}"
            )

    @doc_category("Generic rigidity")
    def is_globally_rigid(self, dim: int = 2) -> bool:
        """
        Check whether the graph is :prf:ref:`globally dim-rigid
        <def-globally-rigid-graph>`.

        TODO
        ----
        missing definition, implementation for dim>=3

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,0)])
        >>> G.is_globally_rigid()
        True

        Notes
        -----
         * dim=1: 2-connectivity
         * dim=2: redundantly rigid+3-connected
         * dim>=3: Randomized Rigidity Matrix => Stress (symbolic maybe?)
        By default, the graph is in dimension 2.
        A complete graph is automatically globally rigid
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()

        elif dim == 1:
            if (self.number_of_nodes() == 2 and self.number_of_edges() == 1) or (
                self.number_of_nodes() == 1 or self.number_of_nodes() == 0
            ):
                return True
            return self.vertex_connectivity() >= 2
        elif dim == 2:
            if (
                (self.number_of_nodes() == 3 and self.number_of_edges() == 3)
                or (self.number_of_nodes() == 2 and self.number_of_edges() == 1)
                or (self.number_of_nodes() == 1 or self.number_of_nodes() == 0)
            ):
                return True
            return self.is_redundantly_rigid() and self.vertex_connectivity() >= 3
        else:

            # Random sampling from [1,N] for N depending quadratically on number
            # of vertices.
            raise NotImplementedError()

    @doc_category("Waiting for implementation")
    def is_Rd_dependent(self, dim: int = 2) -> bool:
        """
        Notes
        -----
         * dim=1: Graphic Matroid
         * dim=2: not (2,3)-sparse
         * dim>=1: Compute the rank of the rigidity matrix and compare with edge count
        """
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()
        raise NotImplementedError()

    @doc_category("Waiting for implementation")
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
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()
        raise NotImplementedError()

    @doc_category("Waiting for implementation")
    def is_Rd_circuit(self, dim: int = 2) -> bool:
        """
        Notes
        -----
         * dim=1: Graphic Matroid
         * dim=2: Remove any edge and it becomes sparse
           (sparsity for every subgraph except whole graph?)
         * dim>=1: Dependent + Remove every edge and compute the rigidity matrix' rank
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()
        raise NotImplementedError()

    @doc_category("Waiting for implementation")
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
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()
        raise NotImplementedError()

    @doc_category("Generic rigidity")
    def max_rigid_subgraphs(self, dim: int = 2) -> List[GraphType]:
        """
        List vertex-maximal rigid subgraphs of the graph.

        Definitions
        -----
        :prf:ref:`Maximal rigid subgraph <def-maximal-rigid-subgraph>`

        TODO
        ----
        missing definition

        Notes
        -----
        We only return nontrivial subgraphs, meaning that there need to be at
        least ``dim+1`` vertices present. If the graph itself is rigid, it is clearly
        maximal and is returned.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (3,0)])
        >>> G.max_rigid_subgraphs()
        []

        >>> G = Graph([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0), (0,2), (5,3)])
        >>> G.is_rigid()
        False
        >>> G.max_rigid_subgraphs()
        [Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]], Graph with vertices [3, 4, 5] and edges [[3, 4], [3, 5], [4, 5]]]
        """  # noqa: E501
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()

        if self.number_of_nodes() <= dim:
            return []
        if self.is_rigid():
            return [self]
        max_subgraphs = []
        for vertex_subset in combinations(self.nodes, self.number_of_nodes() - 1):
            G = self.subgraph(vertex_subset)
            max_subgraphs = [
                j for i in [max_subgraphs, G.max_rigid_subgraphs(dim)] for j in i
            ]

        # We now remove the graphs that were found at least twice.
        clean_list = []
        for i in range(len(max_subgraphs)):
            iso_bool = False
            for j in range(i + 1, len(max_subgraphs)):
                if set(max_subgraphs[i].nodes) == set(
                    max_subgraphs[j].nodes
                ) and max_subgraphs[i].is_isomorphic(max_subgraphs[j]):
                    iso_bool = True
                    break
            if not iso_bool:
                clean_list.append(max_subgraphs[i])
        return clean_list

    @doc_category("Generic rigidity")
    def min_rigid_subgraphs(self, dim: int = 2) -> List[GraphType]:
        """
        List vertex-minimal non-trivial rigid subgraphs of the graph.

        Definitions
        -----
        :prf:ref:`Minimal rigid subgraph <def-minimal-rigid-subgraph>`

        TODO
        ----
        missing definition

        Notes
        -----
        We only return nontrivial subgraphs, meaning that there need to be at
        least ``dim+1`` vertices present.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0), (0,3), (4,1), (5,2)])
        >>> G.is_rigid()
        True
        >>> G.min_rigid_subgraphs()
        [Graph with vertices [0, 1, 2, 3, 4, 5] and edges [[0, 1], [0, 3], [0, 5], [1, 2], [1, 4], [2, 3], [2, 5], [3, 4], [4, 5]]]
        """  # noqa: E501
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()

        min_subgraphs = []
        if self.number_of_nodes() <= 2:
            return []
        elif self.number_of_nodes() == dim + 1 and self.is_rigid():
            return [self]
        elif self.number_of_nodes() == dim + 1:
            return []
        for vertex_subset in combinations(self.nodes, self.number_of_nodes() - 1):
            G = self.subgraph(vertex_subset)
            subgraphs = G.min_rigid_subgraphs(dim)
            if len(subgraphs) == 0 and G.is_rigid():
                min_subgraphs.append(G)
            else:
                min_subgraphs = [
                    j for i in [min_subgraphs, G.min_rigid_subgraphs(dim)] for j in i
                ]

        # We now remove the graphs that were found at least twice.
        clean_list = []
        for i in range(len(min_subgraphs)):
            iso_bool = False
            for j in range(i + 1, len(min_subgraphs)):
                if set(min_subgraphs[i].nodes) == set(
                    min_subgraphs[j].nodes
                ) and min_subgraphs[i].is_isomorphic(min_subgraphs[j]):
                    iso_bool = True
                    break
            if not iso_bool:
                clean_list.append(min_subgraphs[i])
        # If no smaller graph is found and the graph is rigid, it is returned.
        if not clean_list and self.is_rigid():
            clean_list = [self]
        return clean_list

    @doc_category("General graph theoretical properties")
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

    @doc_category("Other")
    def to_int(self, vertex_order: List[Vertex] = None) -> int:
        r"""
        Return the integer representation of the graph.

        The graph integer representation is the integer whose binary
        expansion is given by the sequence obtained by concatenation
        of the rows of the upper triangle of the adjacency matrix,
        excluding the diagonal.

        Parameters
        ----------
        vertex_order:
            By listing vertices in the preferred order, the adjacency matrix
            is computed with the given order. If no vertex order is
            provided, :meth:`~.Graph.vertex_list()` is used.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2)])
        >>> G.adjacency_matrix()
        Matrix([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]])
        >>> G.to_int()
        5

        TODO
        ----
        Implement taking canonical before computing the integer representation.
        Tests.
        """
        if self.number_of_edges() == 0:
            raise ValueError(
                "The integer representation is only defined "
                "for graphs with at least one edge."
            )
        if self.min_degree() == 0:
            raise ValueError(
                "The integer representation only works "
                "for graphs without isolated vertices."
            )
        if nx.number_of_selfloops(self) == 0:
            M = self.adjacency_matrix(vertex_order)
            upper_diag = [
                str(b) for i, row in enumerate(M.tolist()) for b in row[i + 1 :]
            ]
            return int("".join(upper_diag), 2)
        else:
            raise LoopError()

    @classmethod
    @doc_category("Class methods")
    def from_int(cls, N: int) -> GraphType:
        """
        Return a graph given its integer representation.

        See :meth:`to_int` for the description
        of the integer representation.
        """
        if not isinstance(N, int):
            raise TypeError(f"The parameter n has to be an integer, not {type(N)}.")
        if N <= 0:
            raise ValueError(f"The parameter n has to be positive, not {N}.")
        L = bin(N)[2:]
        n = math.ceil((1 + math.sqrt(1 + 8 * len(L))) / 2)
        rows = []
        s = 0
        L = "".join(["0" for _ in range(int(n * (n - 1) / 2) - len(L))]) + L
        for i in range(n):
            rows.append(
                [0 for _ in range(i + 1)] + [int(k) for k in L[s : s + (n - i - 1)]]
            )
            s += n - i - 1
        adjMatrix = Matrix(rows)
        return Graph.from_adjacency_matrix(adjMatrix + adjMatrix.transpose())

    @classmethod
    @doc_category("Class methods")
    def from_adjacency_matrix(cls, M: Matrix) -> GraphType:
        """
        Create a graph from a given adjacency matrix.

        Examples
        --------
        >>> M = Matrix([[0,1],[1,0]])
        >>> G = Graph.from_adjacency_matrix(M)
        >>> print(G)
        Graph with vertices [0, 1] and edges [[0, 1]]
        """
        if not M.is_square:
            raise TypeError("The matrix is not square!")
        if not M.is_symmetric():
            raise TypeError("The matrix is not symmetric.")

        vertices = range(M.cols)
        edges = []
        for i, j in combinations(vertices, 2):
            if not (M[i, j] == 0 or M[i, j] == 1):
                raise TypeError(
                    "The provided adjacency matrix contains entries other than 0 and 1"
                )
            if M[i, j] == 1:
                edges += [(i, j)]
        return Graph.from_vertices_and_edges(vertices, edges)

    @doc_category("General graph theoretical properties")
    def adjacency_matrix(self, vertex_order: List[Vertex] = None) -> Matrix:
        """
        Return the adjacency matrix of the graph.

        Parameters
        ----------
        vertex_order:
            By listing vertices in the preferred order, the adjacency matrix
            can be computed in a way the user expects. If no vertex order is
            provided, :meth:`~.Graph.vertex_list()` is used.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (1,3)])
        >>> G.adjacency_matrix()
        Matrix([
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 0]])

        Notes
        -----
        :func:`networkx.linalg.graphmatrix.adjacency_matrix`
        requires `scipy`. To avoid unnecessary imports, the method is implemented here.
        """
        if vertex_order is None:
            vertex_order = self.vertex_list()
        else:
            if not set(self.nodes) == set(
                vertex_order
            ) or not self.number_of_nodes() == len(vertex_order):
                raise IndexError(
                    "The vertex_order must contain the same vertices as the graph!"
                )

        row_list = [
            [+((v1, v2) in self.edges) for v2 in vertex_order] for v1 in vertex_order
        ]

        return Matrix(row_list)

    @doc_category("Other")
    def random_framework(
        self, dim: int = 2, rand_range: Union(int, List[int]) = None
    ) -> FrameworkType:
        """
        Return framework with random realization.

        This method calls :meth:`.Framework.Random`.
        """
        from pyrigi.framework import Framework

        return Framework.Random(self, dim, rand_range)

    def _resolve_edge_colors(
        self, edge_color: Union(str, list[list[Edge]], dict[str : list[Edge]])
    ) -> tuple[list, list]:
        """
        Return the lists of colors and edges in the format for plotting.
        """
        edge_list = self.edge_list()
        edge_list_ref = []
        edge_color_array = []
        colors = [
            "red",
            "green",
            "blue",
            "yellow",
            "orange",
            "purple",
            "pink",
            "lime",
            "cyan",
            "magenta",
            "brown",
            "darkblue",
            "gold",
            "lightgreen",
            "violet",
            "lightblue",
            "orangered",
            "olive",
            "dodgerblue",
        ]
        color = ""
        if isinstance(edge_color, str):
            return [edge_color for _ in edge_list], edge_list

        if isinstance(edge_color, list):
            edges_partition = edge_color
            for i, part in enumerate(edges_partition):
                if i >= len(colors):
                    color = "black"
                else:
                    color = colors[i]
                for e in part:
                    if not self.has_edge(e[0], e[1]):
                        raise ValueError(
                            "The input includes a pair that is not an edge."
                        )
                    edge_color_array.append(color)
                    edge_list_ref.append(tuple(e))
        elif isinstance(edge_color, dict):
            color_edges_dict = edge_color
            for color, edges in color_edges_dict.items():
                for e in edges:
                    if not self.has_edge(e[0], e[1]):
                        raise ValueError(
                            "Input includes edge that is not part of the framework"
                        )
                    edge_color_array.append(color)
                    edge_list_ref.append(tuple(e))
        else:
            raise ValueError("The input color_edge has none of the supported formats.")
        for e in edge_list:
            if (e[0], e[1]) not in edge_list_ref and (e[1], e[0]) not in edge_list_ref:
                edge_color_array.append("black")
                edge_list_ref.append(e)
        if len(edge_list_ref) > self.number_of_edges():
            raise ValueError(
                "There is an edge whose color was specified multiple times."
            )
        return edge_color_array, edge_list_ref

    @doc_category("Other")
    def plot(
        self,
        placement: dict[Vertex, Point] = None,
        vertex_size: int = 300,
        vertex_color: str = "#4584B6",
        vertex_shape: str = "o",
        vertex_labels: bool = True,
        edge_width: float = 1.0,
        edge_color: Union(str, list[list[Edge]], dict[str : list[Edge]]) = "black",
        edge_style: str = "solid",
        canvas_width: int = 6.4,
        canvas_height: int = 4.8,
        aspect_ratio: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Plot the graph.

        Parameters
        ----------
        placement:
            The placement of vertices in the plane.
        vertex_size:
            The size of the vertex. By default 300.
        vertex_color:
            The color of the vertex. Color can be string or rgb (or rgba)
            tuple of floats from 0-1.
        vertex_shape:
            The shape of the vertex. Specification is as matplotlib.scatter
            marker, one of 'so^>v<dph8'. By default 'o'.
        vertex_labels:
            If True, vertex labels are displayed. By default True.
        edge_width:
            The width of the edge. By default 1.0.
        edge_color:
            The color of the edge. Color can be string or rgb (or rgba) tuple
            of floats from 0-1. By default 'k'.
        edge_style:
            Edge line style e.g.: '-', '–', '-', ':' or words like 'solid' or
            'dashed'. By default '-'.
        font_size:
            The size of the font used for the labels. By default 12.
        font_color:
            The color of the font used for the labels. By default 'k'.
        canvas_width:
            The width of the canvas in inches. By default 6.4.
        canvas_height:
            The height of the canvas in inches. By default 4.8.
        aspect_ratio:
            The ratio of y-unit to x-unit. By default 1.0.

        Notes
        -----
        Use a networkx internal routine to plot the graph."""

        fig, ax = plt.subplots()
        ax.set_adjustable("datalim")
        fig.set_figwidth(canvas_width)
        fig.set_figheight(canvas_height)
        ax.set_aspect(aspect_ratio)
        edge_color_array, edge_list_ref = self._resolve_edge_colors(edge_color)

        if placement is None:
            placement = nx.drawing.layout.spring_layout(self)

        nx.draw(
            self,
            pos=placement,
            ax=ax,
            node_size=vertex_size,
            node_color=vertex_color,
            node_shape=vertex_shape,
            with_labels=vertex_labels,
            width=edge_width,
            edge_color=edge_color_array,
            edgelist=edge_list_ref,
            style=edge_style,
            **kwargs,
        )
        plt.show()


Graph.__doc__ = Graph.__doc__.replace(
    "METHODS",
    generate_category_tables(
        Graph,
        1,
        [
            "Attribute getters",
            "Class methods",
            "Graph manipulation",
            "General graph theoretical properties",
            "Generic rigidity",
            "Sparseness",
            "Other",
            "Waiting for implementation",
        ],
        include_all=False,
    ),
)
