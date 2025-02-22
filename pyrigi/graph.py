"""
Module for rigidity related graph properties.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from copy import deepcopy
from itertools import combinations
from random import randint
from typing import Iterable

import networkx as nx
from sympy import Matrix, oo, zeros

import pyrigi._input_check as _input_check
import pyrigi._pebble_digraph
from pyrigi.data_type import Vertex, Edge, Point, Inf, Sequence
from pyrigi.exception import LoopError, NotSupportedValueError
from pyrigi.misc import doc_category, generate_category_tables
from pyrigi.plot_style import PlotStyle
from pyrigi.warning import RandomizedAlgorithmWarning


__doctest_requires__ = {("Graph.number_of_realizations",): ["lnumber"]}


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

    METHODS

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

    Suggested Improvements
    ----------------------
    Graphical output in Jupyter.
    Graph names.
    __repr__ output for larger graphs?
    """

    silence_rand_alg_warns = False

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

    def __eq__(self, other: Graph):
        """
        Return whether the other graph has the same vertices and edges.

        Examples
        --------
        >>> from pyrigi import Graph
        >>> G = Graph([[1,2]])
        >>> H = Graph([[2,1]])
        >>> G == H
        True

        Notes
        -----
        :func:`~networkx.utils.misc.graphs_equal`
        behaves strangely, hence it is not used.
        """
        if (
            self.number_of_edges() != other.number_of_edges()
            or self.number_of_nodes() != other.number_of_nodes()
        ):
            return False
        for v in self.nodes:
            if not other.has_node(v):
                return False
        for e in self.edges:
            if not other.has_edge(*e):
                return False
        return True

    def __add__(self, other: Graph):
        r"""
        Return the union of ``self`` and ``other``.

        Definitions
        -----------
        :prf:ref:`Union of two graphs <def-union-graph>`

        Examples
        --------
        >>> G = Graph([[0,1],[1,2],[2,0]])
        >>> H = Graph([[2,3],[3,4],[4,2]])
        >>> G + H
        Graph with vertices [0, 1, 2, 3, 4] and edges [[0, 1], [0, 2], [1, 2], [2, 3], [2, 4], [3, 4]]
        """  # noqa: E501
        return Graph(nx.compose(self, other))

    @classmethod
    @doc_category("Class methods")
    def from_vertices_and_edges(
        cls, vertices: Sequence[Vertex], edges: Sequence[Edge]
    ) -> Graph:
        """
        Create a graph from a list of vertices and edges.

        Parameters
        ----------
        vertices
        edges

        Examples
        --------
        >>> Graph.from_vertices_and_edges([0, 1, 2, 3], [])
        Graph with vertices [0, 1, 2, 3] and edges []
        >>> Graph.from_vertices_and_edges([0, 1, 2, 3], [[0, 1], [0, 2], [1, 3]])
        Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 2], [1, 3]]
        >>> Graph.from_vertices_and_edges(['a', 'b', 'c', 'd'], [['a','c'], ['a', 'd']])
        Graph with vertices ['a', 'b', 'c', 'd'] and edges [['a', 'c'], ['a', 'd']]
        """
        G = Graph()
        G.add_nodes_from(vertices)
        G._input_check_edge_format_list(edges)
        G.add_edges(edges)
        return G

    @classmethod
    @doc_category("Class methods")
    def from_vertices(cls, vertices: Sequence[Vertex]) -> Graph:
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
    def CompleteOnVertices(cls, vertices: Sequence[Vertex]) -> Graph:
        """
        Generate a complete graph on ``vertices``.

        Examples
        --------
        >>> Graph.CompleteOnVertices([0, 1, 2, 3, 4])
        Graph with vertices [0, 1, 2, 3, 4] and edges [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
        >>> Graph.CompleteOnVertices(['a', 'b', 'c', 'd'])
        Graph with vertices ['a', 'b', 'c', 'd'] and edges [['a', 'b'], ['a', 'c'], ['a', 'd'], ['b', 'c'], ['b', 'd'], ['c', 'd']]
        """  # noqa: E501
        edges = list(combinations(vertices, 2))
        return Graph.from_vertices_and_edges(vertices, edges)

    def _input_check_no_loop(self) -> None:
        """
        Check whether a graph has loops and raise an error in case.
        """
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()

    def _input_check_vertex_members(
        self, to_check: Iterable[Vertex] | Vertex, name: str = ""
    ) -> None:
        """
        Check whether the elements of a list are indeed vertices and
        raise error otherwise.

        Parameters
        ----------
        to_check:
            A vertex or ``Iterable`` of vertices for which the containment in ``self``
            is checked.
        name:
            A name of the ``Iterable`` ``to_check`` can be picked.
        """
        if not isinstance(to_check, Iterable):
            if not self.has_node(to_check):
                raise ValueError(
                    f"The element {to_check} is not a vertex of the graph!"
                )
        else:
            for vertex in to_check:
                if not self.has_node(vertex):
                    raise ValueError(
                        f"The element {vertex} from "
                        + name
                        + f" {to_check} is not a vertex of the graph!"
                    )

    def _input_check_edge_format(self, edge: Edge, loopfree: bool = False) -> None:
        """
        Check if an ``edge`` is a pair of (distinct) vertices of the graph and
        raise an error otherwise.

        Parameters
        ----------
        edge:
            Edge for which the containment in the graph ``self`` is checked.
        """
        if not isinstance(edge, list | tuple) or not len(edge) == 2:
            raise TypeError(f"The input {edge} must be a tuple or list of length 2!")
        self._input_check_vertex_members(edge, "the input pair")
        if loopfree and edge[0] == edge[1]:
            raise LoopError(f"The input {edge} must be two distinct vertices.")

    def _input_check_edge(self, edge: Edge, vertices: Sequence[Vertex] = None) -> None:
        """
        Check if the given input is an edge of the graph with endvertices in vertices and
        raise an error otherwise.

        Parameters
        ----------
        edge:
            an edge to be checked
        vertices:
            Check if the endvertices of the edge are contained in the list ``vertices``.
            If None, the function considers all vertices of the graph.
        """
        self._input_check_edge_format(edge)
        if vertices and (not edge[0] in vertices or not edge[1] in vertices):
            raise ValueError(
                f"The elements of the edge {edge} are not among vertices {vertices}!"
            )
        if not self.has_edge(edge[0], edge[1]):
            raise ValueError(f"Edge {edge} is not contained in the graph!")

    def _input_check_edge_list(
        self, edges: Sequence[Edge], vertices: Sequence[Vertex] = None
    ) -> None:
        """
        Apply :meth:`~Graph._input_check_edge` to all edges in a list.

        Parameters
        ----------
        edges:
            a list of edges to be checked
        vertices:
            Check if the endvertices of the edges are contained in the list ``vertices``.
            If None (default), the function considers all vertices of the graph.
        """
        for edge in edges:
            self._input_check_edge(edge, vertices)

    def _input_check_edge_format_list(self, edges: Sequence[Edge]) -> None:
        """
        Apply :meth:`~Graph._input_check_edge_format` to all edges in a list.

        Parameters
        ----------
        edges:
            a list of pairs to be checked
        """
        for edge in edges:
            self._input_check_edge_format(edge)

    def _input_check_vertex_order(
        self, vertex_order: Sequence[Vertex], name: str = ""
    ) -> list[Vertex]:
        """
        Check whether the provided `vertex_order` contains the same elements
        as the graph's vertex set.

        Parameters
        ----------
        vertex_order:
            List of vertices in the preferred order

        Notes
        -----
        Throws an error if the vertices in `vertex_order` do not agree with the
        underlying graph's vertices.
        """
        if vertex_order is None:
            return self.vertex_list()
        else:
            if not self.number_of_nodes() == len(vertex_order) or not set(
                self.vertex_list()
            ) == set(vertex_order):
                raise ValueError(
                    "The vertices in `"
                    + name
                    + "` must be exactly "
                    + "the same vertices as in the graph!"
                )
            return list(vertex_order)

    def _input_check_edge_order(
        self, edge_order: Sequence[Edge], name: str = ""
    ) -> list[Edge]:
        """
        Check whether the provided `edge_order` contains the same elements
        as the graph's edge set.

        Parameters
        ----------
        edge_order:
            List of edges in the preferred order

        Notes
        -----
        Throws an error if the edges in `edge_order` do not agree with the
        underlying graph's edges.
        """
        if edge_order is None:
            return self.edge_list()
        else:
            if not self.number_of_edges() == len(edge_order) or not all(
                [set(e) in [set(e) for e in edge_order] for e in self.edge_list()]
            ):
                raise ValueError(
                    "The edges in `" + name + "` must be exactly "
                    "the same edges as in the graph!"
                )
            return list(edge_order)

    @classmethod
    def _warn_randomized_alg(cls, method: Callable, explicit_call: str = None):
        """
        Raise a warning if a randomized algorithm is silently called.

        Parameters
        ----------
        method:
            Reference to the method that is called.
        explicit_call:
            ``str`` quantifying the referenced parameter.
        """
        if not cls.silence_rand_alg_warns:
            warnings.warn(
                RandomizedAlgorithmWarning(
                    method, explicit_call=explicit_call, class_off=cls
                )
            )

    @doc_category("Attribute getters")
    def vertex_list(self) -> list[Vertex]:
        """
        Return the list of vertices.

        Notes
        -----
        The output is sorted if possible,
        otherwise, the internal order is used instead.

        Examples
        --------
        >>> G = Graph.from_vertices_and_edges([2, 0, 3, 1], [[0, 1], [0, 2], [0, 3]])
        >>> G.vertex_list()
        [0, 1, 2, 3]

        >>> G = Graph.from_vertices(['c', 'a', 'b'])
        >>> G.vertex_list()
        ['a', 'b', 'c']

        >>> G = Graph.from_vertices(['b', 1, 'a']) # incomparable vertices
        >>> G.vertex_list()
        ['b', 1, 'a']
        """
        try:
            return sorted(self.nodes)
        except BaseException:
            return list(self.nodes)

    @doc_category("Attribute getters")
    def edge_list(self, as_tuples: bool = False) -> list[Edge]:
        """
        Return the list of edges.

        Parameters
        ----------
        as_tuples:
            If ``True``, all edges are returned as tuples.

        Notes
        -----
        The output is sorted if possible,
        otherwise, the internal order is used instead.

        Examples
        --------
        >>> G = Graph([[0, 3], [3, 1], [0, 1], [2, 0]])
        >>> G.edge_list()
        [[0, 1], [0, 2], [0, 3], [1, 3]]

        >>> G = Graph.from_vertices(['a', 'c', 'b'])
        >>> G.edge_list()
        []

        >>> G = Graph([['c', 'b'], ['b', 'a']])
        >>> G.edge_list()
        [['a', 'b'], ['b', 'c']]

        >>> G = Graph([['c', 1], [2, 'a']]) # incomparable vertices
        >>> G.edge_list()
        [('c', 1), (2, 'a')]
        """
        try:
            if as_tuples:
                return sorted([tuple(sorted(e)) for e in self.edges])
            else:
                return sorted([sorted(e) for e in self.edges])
        except BaseException:
            if as_tuples:
                return [tuple(e) for e in self.edges]
            else:
                return list(self.edges)

    @doc_category("Graph manipulation")
    def delete_vertex(self, vertex: Vertex) -> None:
        """Alias for :meth:`networkx.Graph.remove_node`."""
        self.remove_node(vertex)

    @doc_category("Graph manipulation")
    def delete_vertices(self, vertices: Sequence[Vertex]) -> None:
        """Alias for :meth:`networkx.Graph.remove_nodes_from`."""
        self.remove_nodes_from(vertices)

    @doc_category("Graph manipulation")
    def delete_edge(self, edge: Edge) -> None:
        """Alias for :meth:`networkx.Graph.remove_edge`"""
        self.remove_edge(*edge)

    @doc_category("Graph manipulation")
    def delete_edges(self, edges: Sequence[Edge]) -> None:
        """Alias for :meth:`networkx.Graph.remove_edges_from`."""
        self.remove_edges_from(edges)

    @doc_category("Graph manipulation")
    def add_vertex(self, vertex: Vertex) -> None:
        """Alias for :meth:`networkx.Graph.add_node`."""
        self.add_node(vertex)

    @doc_category("Graph manipulation")
    def add_vertices(self, vertices: Sequence[Vertex]) -> None:
        """Alias for :meth:`networkx.Graph.add_nodes_from`."""
        self.add_nodes_from(vertices)

    @doc_category("Graph manipulation")
    def add_edges(self, edges: Sequence[Edge]) -> None:
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
    def degree_sequence(self, vertex_order: Sequence[Vertex] = None) -> list[int]:
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
        vertex_order = self._input_check_vertex_order(vertex_order)

        return [int(self.degree(v)) for v in vertex_order]

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
        return min([int(self.degree(v)) for v in self.nodes])

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
        return max([int(self.degree(v)) for v in self.nodes])

    def _build_pebble_digraph(self, K: int, L: int) -> None:
        r"""
        Build and save the pebble digraph from scratch.

        Adds edges one-by-one, as long as it can.
        Discard edges that are not :prf:ref:`(K, L)-independent <def-kl-sparse-tight>`
        from the rest of the graph.

        Parameters
        ----------
        K:
        L:
        """
        _input_check.pebble_values(K, L)

        dir_graph = pyrigi._pebble_digraph.PebbleDiGraph(K, L)
        dir_graph.add_nodes_from(self.nodes)
        for edge in self.edges:
            u, v = edge[0], edge[1]
            dir_graph.add_edge_maintaining_digraph(u, v)
        self._pebble_digraph = dir_graph

    @doc_category("Sparseness")
    def spanning_kl_sparse_subgraph(
        self, K: int, L: int, use_precomputed_pebble_digraph: bool = False
    ) -> Graph:
        r"""
        Return a maximal :prf:ref:`(K, L)-sparse <def-kl-sparse-tight>` subgraph.

        Based on the directed graph calculated by the pebble game algorithm, return
        a maximal :prf:ref:`(K, L)-sparse <def-kl-sparse-tight>` of the graph.
        There are multiple possible maximal (K, L)-sparse subgraphs, all of which have
        the same number of edges.

        Parameters
        ----------
        K:
        L:
        use_precomputed_pebble_digraph:
            If ``True``, the pebble digraph present in the cache is used.
            If ``False``, recompute the pebble digraph.
            Use ``True`` only if you are certain that the pebble game digraph
            is consistent with the graph.

        Examples
        --------
        >>> G = Graph.CompleteOnVertices([0,1,2,3])
        >>> G.spanning_kl_sparse_subgraph(2,3)
        Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]]
        """  # noqa: E501
        if (
            not use_precomputed_pebble_digraph
            or K != self._pebble_digraph.K
            or L != self._pebble_digraph.L
        ):
            self._build_pebble_digraph(K, L)

        return Graph(self._pebble_digraph.to_undirected())

    def _is_pebble_digraph_sparse(
        self, K: int, L: int, use_precomputed_pebble_digraph: bool = False
    ) -> bool:
        """
        Check whether the pebble digraph has the same number of edges as the graph.

        Parameters
        ----------
        K:
        L:
        use_precomputed_pebble_digraph:
            If ``True``, the pebble digraph present in the cache is used.
            If ``False``, recompute the pebble digraph.
            Use ``True`` only if you are certain that the pebble game digraph
            is consistent with the graph.
        """
        if (
            not use_precomputed_pebble_digraph
            or K != self._pebble_digraph.K
            or L != self._pebble_digraph.L
        ):
            self._build_pebble_digraph(K, L)

        # all edges are in fact inside the pebble digraph
        return self.number_of_edges() == self._pebble_digraph.number_of_edges()

    @doc_category("Sparseness")
    def is_kl_sparse(
        self,
        K: int,
        L: int,
        algorithm: str = "default",
        use_precomputed_pebble_digraph: bool = False,
    ) -> bool:
        r"""
        Check whether the graph is :prf:ref:`(K, L)-sparse <def-kl-sparse-tight>`.

        Parameters
        ----------
        K:
        L:
        algorithm:
            If ``"pebble"``, the function uses the pebble game algorithm to check
            for sparseness. If "subgraph", it uses the subgraph method.
            If defaults to ``"pebble"`` whenever ``K>0`` and ``0<=L<2K``,
            otherwise "subgraph".
        use_precomputed_pebble_digraph:
            If ``True``, the pebble digraph present in the cache is used.
            If ``False``, recompute the pebble digraph.
            Use ``True`` only if you are certain that the pebble game digraph
            is consistent with the graph.

        Examples
        ----
        >>> import pyrigi.graphDB as graphs
        >>> G = graphs.DoubleBanana()
        >>> G.is_kl_sparse(3,6)
        True
        >>> G.add_edge(0,1)
        >>> G.is_kl_sparse(3,6)
        False
        """
        _input_check.integrality_and_range(K, "K", min_val=1)
        _input_check.integrality_and_range(L, "L", min_val=0)

        if algorithm == "default":
            try:
                _input_check.pebble_values(K, L)
                algorithm = "pebble"
            except ValueError:
                algorithm = "subgraph"

        if algorithm == "pebble":
            _input_check.pebble_values(K, L)
            return self._is_pebble_digraph_sparse(
                K, L, use_precomputed_pebble_digraph=use_precomputed_pebble_digraph
            )

        if algorithm == "subgraph":
            for i in range(K, self.number_of_nodes() + 1):
                for vertex_set in combinations(self.nodes, i):
                    G = self.subgraph(vertex_set)
                    if G.number_of_edges() > K * G.number_of_nodes() - L:
                        return False
            return True

        # reaching this position means that the algorithm is unknown
        raise NotSupportedValueError(algorithm, "algorithm", self.is_kl_sparse)

    @doc_category("Sparseness")
    def is_sparse(self) -> bool:
        r"""
        Return whether the graph is (2,3)-sparse.

        For general $(k,\ell)$-sparsity, see :meth:`.is_kl_sparse`.

        Definitions
        -----------
        :prf:ref:`(2,3)-sparse <def-kl-sparse-tight>`

        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> graphs.Path(3).is_sparse()
        True
        >>> graphs.Complete(4).is_sparse()
        False
        >>> graphs.ThreePrism().is_sparse()
        True
        """
        return self.is_kl_sparse(2, 3, algorithm="pebble")

    @doc_category("Sparseness")
    def is_kl_tight(
        self,
        K: int,
        L: int,
        algorithm: str = "default",
        use_precomputed_pebble_digraph: bool = False,
    ) -> bool:
        r"""
        Check whether the graph is (k,l)-tight.

        Definitions
        -----------
        :prf:ref:`(K, L)-tight <def-kl-sparse-tight>`

        Parameters
        ----------
        K:
        L:
        algorithm:
            See :meth:`.is_kl_sparse`.
        use_precomputed_pebble_digraph:
            If ``True``, the pebble digraph present in the cache is used.
            If ``False``, recompute the pebble digraph.
            Use ``True`` only if you are certain that the pebble game digraph
            is consistent with the graph.

        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> G = graphs.Complete(4)
        >>> G.is_kl_tight(2,2)
        True
        >>> G1 = graphs.CompleteBipartite(4,4)
        >>> G1.is_kl_tight(3,6)
        False
        """
        return (
            self.is_kl_sparse(
                K,
                L,
                algorithm,
                use_precomputed_pebble_digraph=use_precomputed_pebble_digraph,
            )
            and self.number_of_edges() == K * self.number_of_nodes() - L
        )

    @doc_category("Sparseness")
    def is_tight(self) -> bool:
        r"""
        Return whether the graph is (2,3)-tight.

        For general $(k,\ell)$-tightness, see :meth:`.is_kl_tight`.

        Definitions
        -----------
        :prf:ref:`(2,3)-tight <def-kl-sparse-tight>`

        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> graphs.Path(4).is_tight()
        False
        >>> graphs.ThreePrism().is_tight()
        True
        """
        return self.is_kl_tight(2, 3, algorithm="pebble")

    @doc_category("Graph manipulation")
    def zero_extension(
        self,
        vertices: Sequence[Vertex],
        new_vertex: Vertex = None,
        dim: int = 2,
        inplace: bool = False,
    ) -> Graph:
        """
        Return a ``dim``-dimensional 0-extension.

        Definitions
        -----------
        :prf:ref:`0-extension <def-k-extension>`

        Parameters
        ----------
        vertices:
            A new vertex will be connected to these vertices.
            All the vertices must be contained in the graph
            and there must be ``dim`` of them.
        new_vertex:
            Newly added vertex will be named according to this parameter.
            If None, the name will be set as the lowest possible integer value
            greater or equal than the number of nodes.
        dim:
            The dimension in which the k-extension is created.
        inplace:
            If True, the graph will be modified,
            otherwise a new modified graph will be created,
            while the original graph remains unchanged (default).

        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> G = graphs.Complete(3)
        >>> G
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        >>> G.zero_extension([0, 2])
        Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]]
        >>> G.zero_extension([0, 2], 5)
        Graph with vertices [0, 1, 2, 5] and edges [[0, 1], [0, 2], [0, 5], [1, 2], [2, 5]]
        >>> G
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        >>> G.zero_extension([0, 1, 2], 5, dim=3, inplace=True)
        Graph with vertices [0, 1, 2, 5] and edges [[0, 1], [0, 2], [0, 5], [1, 2], [1, 5], [2, 5]]
        >>> G
        Graph with vertices [0, 1, 2, 5] and edges [[0, 1], [0, 2], [0, 5], [1, 2], [1, 5], [2, 5]]
        """  # noqa: E501
        return self.k_extension(0, vertices, [], new_vertex, dim, inplace)

    @doc_category("Graph manipulation")
    def one_extension(
        self,
        vertices: Sequence[Vertex],
        edge: Edge,
        new_vertex: Vertex = None,
        dim: int = 2,
        inplace: bool = False,
    ) -> Graph:
        """
        Return a ``dim``-dimensional 1-extension.

        Definitions
        -----------
        :prf:ref:`1-extension <def-k-extension>`

        Parameters
        ----------
        vertices:
            A new vertex will be connected to these vertices.
            All the vertices must be contained in the graph
            and there must be ``dim + 1`` of them.
        edge:
            An edge with endvertices from the list ``vertices`` that will be deleted.
            The edge must be contained in the graph.
        new_vertex:
            Newly added vertex will be named according to this parameter.
            If None, the name will be set as the lowest possible integer value
            greater or equal than the number of nodes.
        dim:
            The dimension in which the k-extension is created.
        inplace:
            If True, the graph will be modified,
            otherwise a new modified graph will be created,
            while the original graph remains unchanged (default).

        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> G = graphs.Complete(3)
        >>> G
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        >>> G.one_extension([0, 1, 2], [0, 1])
        Graph with vertices [0, 1, 2, 3] and edges [[0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        >>> G
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        >>> G = graphs.ThreePrism()
        >>> G
        Graph with vertices [0, 1, 2, 3, 4, 5] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]]
        >>> G.one_extension([0, 1], [0, 1], dim=1)
        Graph with vertices [0, 1, 2, 3, 4, 5, 6] and edges [[0, 2], [0, 3], [0, 6], [1, 2], [1, 4], [1, 6], [2, 5], [3, 4], [3, 5], [4, 5]]
        >>> G = graphs.CompleteBipartite(3, 2)
        >>> G
        Graph with vertices [0, 1, 2, 3, 4] and edges [[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]]
        >>> G.one_extension([0, 1, 2, 3, 4], [0, 3], dim=4, inplace = True)
        Graph with vertices [0, 1, 2, 3, 4, 5] and edges [[0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 5], [4, 5]]
        >>> G
        Graph with vertices [0, 1, 2, 3, 4, 5] and edges [[0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 5], [4, 5]]
        """  # noqa: E501
        return self.k_extension(1, vertices, [edge], new_vertex, dim, inplace)

    @doc_category("Graph manipulation")
    def k_extension(
        self,
        k: int,
        vertices: Sequence[Vertex],
        edges: Sequence[Edge],
        new_vertex: Vertex = None,
        dim: int = 2,
        inplace: bool = False,
    ) -> Graph:
        """
        Return a ``dim``-dimensional ``k``-extension.

        Definitions
        -----------
        :prf:ref:`k-extension <def-k-extension>`

        Parameters
        ----------
        k
        vertices:
            A new vertex will be connected to these vertices.
            All the vertices must be contained in the graph
            and there must be ``dim + k`` of them.
        edges:
            A list of edges that will be deleted.
            The endvertices of all the edges must be contained
            in the list ``vertices``.
            The edges must be contained in the graph and there must be k of them.
        new_vertex:
            Newly added vertex will be named according to this parameter.
            If None, the name will be set as the lowest possible integer value
            greater or equal than the number of nodes.
        dim:
            The dimension in which the k-extension is created.
        inplace:
            If True, the graph will be modified,
            otherwise a new modified graph will be created,
            while the original graph remains unchanged (default).

        Notes
        -----
        See also :meth:`~Graph.zero_extension` and :meth:`~Graph.one_extension`.

        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> G = graphs.Complete(5)
        >>> G
        Graph with vertices [0, 1, 2, 3, 4] and edges [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
        >>> G.k_extension(2, [0, 1, 2, 3], [[0, 1], [0,2]])
        Graph with vertices [0, 1, 2, 3, 4, 5] and edges [[0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5]]
        >>> G
        Graph with vertices [0, 1, 2, 3, 4] and edges [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
        >>> G = graphs.Complete(5)
        >>> G
        Graph with vertices [0, 1, 2, 3, 4] and edges [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
        >>> G.k_extension(2, [0, 1, 2, 3, 4], [[0, 1], [0,2]], dim = 3)
        Graph with vertices [0, 1, 2, 3, 4, 5] and edges [[0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]]
        >>> G = graphs.Path(6)
        >>> G
        Graph with vertices [0, 1, 2, 3, 4, 5] and edges [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
        >>> G.k_extension(2, [0, 1, 2], [[0, 1], [1,2]], dim = 1, inplace = True)
        Graph with vertices [0, 1, 2, 3, 4, 5, 6] and edges [[0, 6], [1, 6], [2, 3], [2, 6], [3, 4], [4, 5]]
        >>> G
        Graph with vertices [0, 1, 2, 3, 4, 5, 6] and edges [[0, 6], [1, 6], [2, 3], [2, 6], [3, 4], [4, 5]]
        """  # noqa: E501
        _input_check.dimension(dim)
        _input_check.integrality_and_range(k, "k", min_val=0)
        self._input_check_no_loop()
        self._input_check_vertex_members(vertices, "'the vertices'")
        if len(set(vertices)) != dim + k:
            raise ValueError(
                f"List of vertices must contain {dim + k} distinct vertices!"
            )
        self._input_check_edge_list(edges, vertices)
        if len(edges) != k:
            raise ValueError(f"List of edges must contain {k} distinct edges!")
        for edge in edges:
            count = edges.count(list(edge)) + edges.count(list(edge)[::-1])
            count += edges.count(tuple(edge)) + edges.count(tuple(edge)[::-1])
            if count > 1:
                raise ValueError(
                    "List of edges must contain distinct edges, "
                    f"but {edge} appears {count} times!"
                )
        if new_vertex is None:
            candidate = self.number_of_nodes()
            while self.has_node(candidate):
                candidate += 1
            new_vertex = candidate
        if self.has_node(new_vertex):
            raise ValueError(f"Vertex {new_vertex} is already a vertex of the graph!")
        G = self
        if not inplace:
            G = deepcopy(self)
        G.remove_edges_from(edges)
        for vertex in vertices:
            G.add_edge(vertex, new_vertex)
        return G

    @doc_category("Graph manipulation")
    def all_k_extensions(
        self,
        k: int,
        dim: int = 2,
        only_non_isomorphic: bool = False,
    ) -> Iterable[Graph]:
        """
        Return an iterator over all possible
        ``dim``-dimensional ``k``-extensions
        for a given ``k``.

        Definitions
        -----------
        :prf:ref:`k-extensions <def-k-extension>`

        Parameters
        ----------
        k
        dim
        only_non_isomorphic:
            If True, only one graph per isomorphism class is included.

        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> G = graphs.Complete(3)
        >>> type(G.all_k_extensions(0))
        <class 'generator'>
        >>> len(list(G.all_k_extensions(0)))
        3
        >>> len(list(G.all_k_extensions(0, only_non_isomorphic=True)))
        1

        >>> len(list(graphs.Diamond().all_k_extensions(1, 2, only_non_isomorphic=True)))
        2

        Notes
        -----
        It turns out that possible errors on bad input paramters are only raised,
        when the output iterator is actually used,
        not when it is created.
        """
        _input_check.dimension(dim)
        self._input_check_no_loop()
        _input_check.integrality_and_range(k, "k", min_val=0)
        _input_check.greater_equal(
            self.number_of_nodes(),
            dim + k,
            "number of vertices in the graph",
            "dim + k",
        )
        _input_check.greater_equal(
            self.number_of_edges(), k, "number of edges in the graph", "k"
        )

        solutions = []
        for edges in combinations(self.edges, k):
            s = set(self.nodes)
            w = set()
            for edge in edges:
                s.discard(edge[0])
                s.discard(edge[1])
                w.add(edge[0])
                w.add(edge[1])
            if len(w) > (dim + k):
                break
            w = list(w)
            for vertices in combinations(s, dim + k - len(w)):
                current = self.k_extension(k, list(vertices) + w, edges, dim=dim)
                if only_non_isomorphic:
                    for other in solutions:
                        if current.is_isomorphic(other):
                            break
                    else:
                        solutions.append(current)
                        yield current
                else:
                    yield current

    @doc_category("Graph manipulation")
    def all_extensions(
        self,
        dim: int = 2,
        only_non_isomorphic: bool = False,
        k_min: int = 0,
        k_max: int | None = None,
    ) -> Iterable[Graph]:
        """
        Return an iterator over all possible
        ``dim``-dimensional ``k``-extensions
        for all possible ``0 <= k <= dim - 1``.

        Definitions
        -----------
        :prf:ref:`k-extensions <def-k-extension>`

        Parameters
        ----------
        dim
        only_non_isomorphic:
            If True, only one graph per isomorphism class is included.
        k_min
            Minimal value of ``k`` for the k-extensions (default 0).
        k_max
            Maximal value of ``k`` for the k-extensions (default dim - 1).

        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> G = graphs.Complete(3)
        >>> type(G.all_extensions())
        <class 'generator'>
        >>> len(list(G.all_extensions()))
        6
        >>> len(list(G.all_extensions(only_non_isomorphic=True)))
        1

        >>> list(graphs.Diamond().all_extensions(2, only_non_isomorphic=True, k_min=1, k_max=1)) == list(graphs.Diamond().all_k_extensions(1, 2, only_non_isomorphic=True))
        True

        Notes
        -----
        It turns out that possible errors on bad input paramters are only raised,
        when the output iterator is actually used,
        not when it is created.
        """  # noqa: E501
        _input_check.dimension(dim)
        self._input_check_no_loop()
        _input_check.integrality_and_range(k_min, "k_min", min_val=0)
        if k_max is None:
            k_max = dim - 1
        _input_check.integrality_and_range(k_max, "k_max", min_val=0)
        _input_check.greater_equal(k_max, k_min, "k_max", "k_min")

        extensions = []
        for k in range(k_min, k_max + 1):
            if self.number_of_nodes() >= dim + k and self.number_of_edges() >= k:
                extensions.extend(self.all_k_extensions(k, dim, only_non_isomorphic))

        solutions = []
        for current in extensions:
            if only_non_isomorphic:
                for other in solutions:
                    if current.is_isomorphic(other):
                        break
                else:
                    solutions.append(current)
                    yield current
            else:
                yield current

    @doc_category("Generic rigidity")
    def extension_sequence(  # noqa: C901
        self, dim: int = 2, return_type: str = "extensions"
    ) -> list[Graph] | list | None:
        """
        Compute a sequence of
        ``dim``-dimensional ``k``-extensions
        if it exists, where k goes from 0 to ``2 * dim - 1``.
        The sequence then starts from a complete graph on ``dim`` vertices.

        The method returns either a sequence of graphs,
        data on the extension, or both.

        Note that for dimesions larger than two, the
        extensions are not always preserving rigidity.

        Definitions
        -----------
        :prf:ref:`k-extensions <def-k-extension>`

        Parameters
        ----------
        dim:
            The dimension in which the extensions are created.
        return_type:
            Can have values "graphs", "extensions" or "both".

            "graphs": the sequence of graphs obtained from the extensions.

            "extensions": an initial graph and a sequence of extensions
            of the form [k, vertices, edges, new_vertex] as needed for the input of `k_extension`

            "both": an initial graph and a sequence of pairs [graph, extension],
            where the latter has the form from above


        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> G = graphs.Complete(3)
        >>> G
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        >>> G.extension_sequence(return_type="graphs")
        [Graph with vertices [1, 2] and edges [[1, 2]], Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]]
        >>> G = graphs.Diamond()
        >>> G
        Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]]
        >>> G.extension_sequence(return_type="graphs")
        [Graph with vertices [2, 3] and edges [[2, 3]], Graph with vertices [0, 2, 3] and edges [[0, 2], [0, 3], [2, 3]], Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]]]
        >>> G.extension_sequence(return_type="extensions")
        [Graph with vertices [2, 3] and edges [[2, 3]], [0, [3, 2], [], 0], [0, [0, 2], [], 1]]
        """  # noqa: E501
        _input_check.dimension(dim)
        self._input_check_no_loop()

        if not self.number_of_edges() == dim * self.number_of_nodes() - math.comb(
            dim + 1, 2
        ):
            return None
        if self.number_of_nodes() == dim:
            return [self]
        degrees = sorted(self.degree, key=lambda node: node[1])
        degrees = [deg for deg in degrees if deg[1] >= dim and deg[1] <= 2 * dim - 1]
        if len(degrees) == 0:
            return None

        for deg in degrees:
            if deg[1] == dim:
                G = deepcopy(self)
                neighbors = list(self.neighbors(deg[0]))
                G.remove_node(deg[0])
                branch = G.extension_sequence(dim, return_type)
                extension = [0, neighbors, [], deg[0]]
                if branch is not None:
                    if return_type == "extensions":
                        return branch + [extension]
                    elif return_type == "graphs":
                        return branch + [self]
                    elif return_type == "both":
                        return branch + [[self, extension]]
                    else:
                        raise NotSupportedValueError(
                            return_type, "return_type", self.extension_sequence
                        )
                return branch
            else:
                neighbors = list(self.neighbors(deg[0]))
                G = deepcopy(self)
                G.remove_node(deg[0])
                for k_possible_edges in combinations(
                    combinations(neighbors, 2), deg[1] - dim
                ):

                    if all([not G.has_edge(*edge) for edge in k_possible_edges]):
                        for edge in k_possible_edges:
                            G.add_edge(*edge)
                        branch = G.extension_sequence(dim, return_type)
                        if branch is not None:
                            extension = [
                                deg[1] - dim,
                                neighbors,
                                k_possible_edges,
                                deg[0],
                            ]
                            if return_type == "extensions":
                                return branch + [extension]
                            elif return_type == "graphs":
                                return branch + [self]
                            elif return_type == "both":
                                return branch + [[self, extension]]
                            else:
                                raise NotSupportedValueError(
                                    return_type, "return_type", self.extension_sequence
                                )
                        for edge in k_possible_edges:
                            G.remove_edge(*edge)
        return None

    @doc_category("Generic rigidity")
    def has_extension_sequence(
        self,
        dim: int = 2,
    ) -> bool:
        """
        Check the existence of a sequence of
        ``dim``-dimensional 0- and 1-extensions.

        The method returns whether the graph can be constructed
        by a sequence of 0- and 1-extensions starting from an edge.

        Definitions
        -----------
        :prf:ref:`0- and 1-extensions <def-k-extension>`.

        Parameters
        ----------
        dim:
            The dimension in which the extensions are created.
            Currently implemented only for ``dim==2``.

        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> G = graphs.ThreePrism()
        >>> G
        Graph with vertices [0, 1, 2, 3, 4, 5] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]]
        >>> G.has_extension_sequence()
        True
        >>> G = graphs.CompleteBipartite(1, 2)
        >>> G
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2]]
        >>> G.has_extension_sequence()
        False
        """  # noqa: E501
        return self.extension_sequence(dim) is not None

    @doc_category("Generic rigidity")
    def number_of_realizations(
        self,
        spherical: bool = False,
        check_min_rigid: bool = True,
        count_reflection: bool = False,
    ) -> int:
        """
        Count the number of complex planar or spherical realizations
        of a minimally 2-rigid graph.

        Algorithms of :cite:p:`CapcoGalletEtAl2018` and
        :cite:p:`GalletGraseggerSchicho2020` are used.
        Note, however, that here the result from these algorithms
        is by default divided by two.
        This behaviour accounts better for global rigidity,
        but it can be changed using the parameter ``count_reflection``.

        Note that by default,
        the method checks if the input graph is indeed minimally 2-rigid.

        Caution: Currently the method only works if the python package ``lnumber``
        is installed :cite:p:`Capco2024`.
        See :ref:`installation-guide` for details on installing.

        Definitions
        -----------
        * :prf:ref:`Number of complex realizations<def-number-of-realizations>`
        * :prf:ref:`Number of complex spherical realizations<def-number-of-spherical-realizations>`

        Parameters
        ----------
        check_min_rigid:
            If ``True``, a ``ValueError`` is raised if the graph is not minimally 2-rigid
            If ``False``, it is assumed that the user is inputting a minimally rigid graph.

        spherical:
            If ``True``, the number of spherical realizations of the graph is returned.
            If ``False`` (default), the number of planar realizations is returned.

        count_reflection:
            If ``True``, the number of realizations is computed modulo direct isometries.
            But reflection is counted to be non-congruent as used in
            :cite:p:`CapcoGalletEtAl2018` and
            :cite:p:`GalletGraseggerSchicho2020`.
            If ``False`` (default), reflection is not counted.

        Examples
        --------
        >>> from pyrigi import Graph
        >>> import pyrigi.graphDB as graphs
        >>> G = Graph([(0,1),(1,2),(2,0)])
        >>> G.number_of_realizations() # number of planar realizations
        1
        >>> G.number_of_realizations(spherical=True)
        1
        >>> G = graphs.ThreePrism()
        >>> G.number_of_realizations() # number of planar realizations
        12

        """  # noqa: E501
        try:
            import lnumber

            if check_min_rigid and not self.is_min_rigid():
                raise ValueError("The graph must be minimally 2-rigid!")

            if self.number_of_nodes() == 1:
                return 1

            if self.number_of_nodes() == 2 and self.number_of_edges() == 1:
                return 1

            graph_int = self.to_int()
            if count_reflection:
                fac = 1
            else:
                fac = 2
            if spherical:
                return lnumber.lnumbers(graph_int) // fac
            else:
                return lnumber.lnumber(graph_int) // fac
        except ImportError:
            raise ImportError(
                "For counting the number of realizations, "
                "the optional package 'lnumber' is used, "
                "run `pip install pyrigi[realization-counting]`!"
            )

    @doc_category("Generic rigidity")
    def is_vertex_redundantly_rigid(
        self, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
    ) -> bool:
        """
        Check whether the graph is vertex redundantly (generically) ``dim``-rigid.

        See :meth:`.is_k_vertex_redundantly_rigid` (using k = 1) for details.

        Definitions
        -----------
        :prf:ref:`vertex redundantly dim-rigid
        <def-redundantly-rigid-graph>`
        """
        _input_check.dimension(dim)
        return self.is_k_vertex_redundantly_rigid(1, dim, algorithm, prob)

    @doc_category("Generic rigidity")
    def is_k_vertex_redundantly_rigid(
        self,
        k: int,
        dim: int = 2,
        algorithm: str = "default",
        prob: float = 0.0001,
    ) -> bool:
        """
        Check whether the graph is :prf:ref:`k-vertex redundantly (generically) dim-rigid
        <def-redundantly-rigid-graph>`.

        Preliminary checks from
        :prf:ref:`thm-k-vertex-redundant-edge-bound-general`,
        :prf:ref:`thm-k-vertex-redundant-edge-bound-general2`,
        :prf:ref:`thm-1-vertex-redundant-edge-bound-dim2`,
        :prf:ref:`thm-2-vertex-redundant-edge-bound-dim2`
        :prf:ref:`thm-k-vertex-redundant-edge-bound-dim2`,
        :prf:ref:`thm-3-vertex-redundant-edge-bound-dim3`,
        :prf:ref:`thm-k-vertex-redundant-edge-bound-dim3`
        ... are used

        Parameters
        ----------
        k:
            level of redundancy
        dim:
            dimension
        algorithm:
            See :meth:`.is_rigid` for the possible algorithms used
            for checking rigidity in this method.
        prob:
            bound on the probability for false negatives of the rigidity testing
            when ``algorithm="randomized"``.
            Warning: this is not the probability of wrong results in this method
            but is just passed on to rigidity testing

        Examples
        --------
        >>> G = Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
        >>> G.is_k_vertex_redundantly_rigid(1, 2)
        True
        >>> G.is_k_vertex_redundantly_rigid(2, 2)
        False
        >>> G = Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 4], [3, 4]])
        >>> G.is_k_vertex_redundantly_rigid(1, 2)
        False

        """  # noqa: E501
        _input_check.dimension(dim)
        _input_check.integrality_and_range(k, "k", min_val=0)
        self._input_check_no_loop()

        n = self.number_of_nodes()
        m = self.number_of_edges()
        if n >= dim + k + 1 and self.min_degree() < dim + k:
            return False
        if dim == 1:
            return self.vertex_connectivity() >= k + 1
        if (
            dim == 2
            and (
                # edge bound from :prf:ref:`thm-1-vertex-redundant-edge-bound-dim2`
                (k == 1 and n >= 5 and m < 2 * n - 1)
                or
                # edge bound from :prf:ref:`thm-2-vertex-redundant-edge-bound-dim2`
                (k == 2 and n >= 6 and m < 2 * n + 2)
                or
                # edge bound from :prf:ref:`thm-k-vertex-redundant-edge-bound-dim2`
                (k >= 3 and n >= 6 * (k + 1) + 23 and m < ((k + 2) * n + 1) // 2)
            )
        ) or (
            dim == 3
            and (
                # edge bound from :prf:ref:`thm-3-vertex-redundant-edge-bound-dim3`
                (k == 3 and n >= 15 and m < 3 * n + 5)
                or
                # edge bound from :prf:ref:`thm-k-vertex-redundant-edge-bound-dim3`
                (
                    k >= 4
                    and n >= 12 * (k + 1) + 10
                    and n % 2 == 0
                    and m < ((k + 3) * n + 1) // 2
                )
            )
        ):
            return False
        # edge bound from :prf:ref:`thm-k-vertex-redundant-edge-bound-general`
        if (
            #
            n >= dim * dim + dim + k + 1
            and m
            < dim * n - math.comb(dim + 1, 2) + k * dim + max(0, k - (dim + 1) // 2)
        ):
            return False
        # edge bound from :prf:ref:`thm-vertex-redundant-edge-bound-general2`
        if k >= dim + 1 and n >= dim + k + 1 and m < ((dim + k) * n + 1) // 2:
            return False

        # in all other cases check by definition
        G = deepcopy(self)
        for vertex_set in combinations(self.nodes, k):
            adj = [[v, list(G.neighbors(v))] for v in vertex_set]
            G.delete_vertices(vertex_set)
            if not G.is_rigid(dim, algorithm, prob):
                return False
            # add vertices and edges back
            G.add_vertices(vertex_set)
            for v, neighbors in adj:
                for neighbor in neighbors:
                    G.add_edge(v, neighbor)
        return True

    @doc_category("Generic rigidity")
    def is_min_vertex_redundantly_rigid(
        self, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
    ) -> bool:
        """
        Check whether the graph is
        minimally vertex redundantly (generically) ``dim``-rigid.

        See :meth:`.is_min_k_vertex_redundantly_rigid` (using ``k=1``) for details.

        Definitions
        -----------
        :prf:ref:`minimally vertex redundantly (generically) dim-rigid
        <def-min-redundantly-rigid-graph>`
        """
        _input_check.dimension(dim)
        return self.is_min_k_vertex_redundantly_rigid(1, dim, algorithm, prob)

    @doc_category("Generic rigidity")
    def is_min_k_vertex_redundantly_rigid(
        self,
        k: int,
        dim: int = 2,
        algorithm: str = "default",
        prob: float = 0.0001,
    ) -> bool:
        """
        Check whether the graph is :prf:ref:`minimally k-vertex redundantly (generically) dim-rigid
        <def-redundantly-rigid-graph>`.

        Preliminary checks from
        :prf:ref:`thm-minimal-k-vertex-redundant-upper-edge-bound`,
        :prf:ref:`thm-minimal-k-vertex-redundant-upper-edge-bound-dim1`
        are used.

        Parameters
        ----------
        k:
            level of redundancy
        dim:
            dimension
        algorithm:
            See :meth:`.is_rigid` for the possible algorithms used
            for checking rigidity in this method.
        prob:
            bound on the probability for false negatives of the rigidity testing
            when ``algorithm="randomized"``.
            Warning: this is not the probability of wrong results in this method,
            but is just passed on to rigidity testing

        Examples
        --------
        >>> G = Graph([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]])
        >>> G.is_min_k_vertex_redundantly_rigid(1, 2)
        True
        >>> G.is_min_k_vertex_redundantly_rigid(2, 2)
        False
        >>> G = Graph([[0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4], [1, 5], [2, 4], [2, 5], [3, 4], [3, 5]])
        >>> G.is_k_vertex_redundantly_rigid(1, 2)
        True
        >>> G.is_min_k_vertex_redundantly_rigid(1, 2)
        False

        """  # noqa: E501

        _input_check.dimension(dim)
        _input_check.integrality_and_range(k, "k", min_val=0)
        self._input_check_no_loop()

        n = self.number_of_nodes()
        m = self.number_of_edges()
        # edge bound from :prf:ref:`thm-minimal-k-vertex-redundant-upper-edge-bound`
        if m > (dim + k) * n - math.comb(dim + k + 1, 2):
            return False
        # edge bound from :prf:ref:`thm-minimal-k-vertex-redundant-upper-edge-bound-dim1`
        if dim == 1:
            if n >= 3 * (k + 1) - 1 and m > (k + 1) * n - (k + 1) * (k + 1):
                return False

        if not self.is_k_vertex_redundantly_rigid(k, dim, algorithm, prob):
            return False

        # for the following we need to know that the graph is k-vertex-redundantly rigid
        if (
            dim == 2
            and (
                # edge bound from :prf:ref:`thm-1-vertex-redundant-edge-bound-dim2`
                (k == 1 and n >= 5 and m == 2 * n - 1)
                or
                # edge bound from :prf:ref:`thm-2-vertex-redundant-edge-bound-dim2`
                (k == 2 and n >= 6 and m == 2 * n + 2)
                or
                # edge bound from :prf:ref:`thm-k-vertex-redundant-edge-bound-dim2`
                (k >= 3 and n >= 6 * (k + 1) + 23 and m == ((k + 2) * n + 1) // 2)
            )
        ) or (
            dim == 3
            and (
                # edge bound from :prf:ref:`thm-3-vertex-redundant-edge-bound-dim3`
                (k == 3 and n >= 15 and m == 3 * n + 5)
                or
                # edge bound from :prf:ref:`thm-k-vertex-redundant-edge-bound-dim3`
                (
                    k >= 4
                    and n >= 12 * (k + 1) + 10
                    and n % 2 == 0
                    and m == ((k + 3) * n + 1) // 2
                )
            )
        ):
            return True
        # edge bound from :prf:ref:`thm-k-vertex-redundant-edge-bound-general`
        if (
            #
            n >= dim * dim + dim + k + 1
            and m
            == dim * n - math.comb(dim + 1, 2) + k * dim + max(0, k - (dim + 1) // 2)
        ):
            return True
        # edge bound from :prf:ref:`thm-vertex-redundant-edge-bound-general2`
        if k >= dim + 1 and n >= dim + k + 1 and m == ((dim + k) * n + 1) // 2:
            return True

        # in all other cases check by definition
        G = deepcopy(self)
        for e in self.edges:
            G.delete_edge(e)
            if G.is_k_vertex_redundantly_rigid(k, dim, algorithm, prob):
                return False
            G.add_edge(*e)
        return True

    @doc_category("Generic rigidity")
    def is_redundantly_rigid(
        self, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
    ) -> bool:
        """
        Check whether the graph is redundantly (generically) ``dim``-rigid.

        See :meth:`.is_k_redundantly_rigid` (using k = 1) for details.

        Definitions
        -----------
        :prf:ref:`redundantly (generically) dim-rigid<def-redundantly-rigid-graph>`
        """
        return self.is_k_redundantly_rigid(1, dim, algorithm, prob)

    @doc_category("Generic rigidity")
    def is_k_redundantly_rigid(
        self,
        k: int,
        dim: int = 2,
        algorithm: str = "default",
        prob: float = 0.0001,
    ) -> bool:
        """
        Check whether the graph is ``k``-redundantly (generically) ``dim``-rigid.

        Preliminary checks from
        :prf:ref:`thm-k-edge-redundant-edge-bound-dim2`,
        :prf:ref:`thm-2-edge-redundant-edge-bound-dim2`,
        :prf:ref:`thm-2-edge-redundant-edge-bound-dim3`,
        :prf:ref:`thm-k-edge-redundant-edge-bound-dim3`,
        :prf:ref:`thm-globally-redundant-3connected` and
        :prf:ref:`thm-globally-mindeg6-dim2`.
        are used

        Definitions
        -----------
        :prf:ref:`k-redundantly (generically) dim-rigid<def-redundantly-rigid-graph>`

        Parameters
        ----------
        k:
            level of redundancy
        dim:
            dimension
        algorithm:
            See :meth:`.is_rigid` for the possible algorithms used
            for checking rigidity in this method.
        prob:
            bound on the probability for false negatives of the rigidity testing
            when ``algorithm="randomized"``.
            Warning: this is not the probability of wrong results in this method,
            but is just passed on to rigidity testing.

        Examples
        --------
        >>> G = Graph([[0, 1], [0, 2], [0, 3], [0, 5], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]])
        >>> G.is_k_redundantly_rigid(1, 2)
        True
        >>> G = Graph([[0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
        >>> G.is_k_redundantly_rigid(1, 2)
        False
        >>> G = Graph([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
        >>> G.is_k_redundantly_rigid(2, 2)
        True

        Suggested Improvements
        ----------------------
        Improve with pebble games.
        """  # noqa: E501
        _input_check.dimension(dim)
        _input_check.integrality_and_range(k, "k", min_val=0)
        self._input_check_no_loop()

        n = self.number_of_nodes()
        m = self.number_of_edges()

        if m < dim * n - math.comb(dim + 1, 2) + k:
            return False
        if self.min_degree() < dim + k:
            return False
        if dim == 1:
            return nx.edge_connectivity(self) >= k + 1
        # edge bounds
        if (
            dim == 2
            and (
                # basic edge bound
                (k == 1 and m < 2 * n - 2)
                or
                # edge bound from :prf:ref:`thm-2-edge-redundant-edge-bound-dim2`
                (k == 2 and n >= 5 and m < 2 * n)
                or
                # edge bound from :prf:ref:`thm-k-edge-redundant-edge-bound-dim2`
                (k >= 3 and n >= 6 * (k + 1) + 23 and m < ((k + 2) * n + 1) // 2)
            )
        ) or (
            dim == 3
            and (
                # edge bound from :prf:ref:`thm-2-edge-redundant-edge-bound-dim3`
                (k == 2 and n >= 14 and m < 3 * n - 4)
                or
                # edge bound from :prf:ref:`thm-k-edge-redundant-edge-bound-dim3`
                (
                    k >= 4
                    and n >= 12 * (k + 1) + 10
                    and n % 2 == 0
                    and m < ((k + 3) * n + 1) // 2
                )
            )
        ):
            return False
        # use global rigidity property of :prf:ref:`thm-globally-redundant-3connected`
        # and :prf:ref:`thm-globally-mindeg6-dim2`
        if dim == 2 and k == 1 and self.vertex_connectivity() >= 6:
            return True

        # in all other cases check by definition
        G = deepcopy(self)
        for edge_set in combinations(self.edge_list(), k):
            G.delete_edges(edge_set)
            if not G.is_rigid(dim, algorithm, prob):
                return False
            G.add_edges(edge_set)
        return True

    @doc_category("Generic rigidity")
    def is_min_redundantly_rigid(
        self, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
    ) -> bool:
        """
        Check whether the graph is minimally redundantly (generically) ``dim``-rigid.

        See :meth:`.is_min_k_redundantly_rigid` (using k = 1) for details.

        Definitions
        -----------
        :prf:ref:`minimally redundantly (generically) dim-rigid
        <def-min-redundantly-rigid-graph>`
        """
        _input_check.dimension(dim)
        return self.is_min_k_redundantly_rigid(1, dim, algorithm, prob)

    @doc_category("Generic rigidity")
    def is_min_k_redundantly_rigid(
        self,
        k: int,
        dim: int = 2,
        algorithm: str = "default",
        prob: float = 0.0001,
    ) -> bool:
        """
        Check whether the graph is minimally ``k``-redundantly (generically) ``dim``-rigid.

        Preliminary checks from
        :prf:ref:`thm-minimal-1-edge-redundant-upper-edge-bound-dim2`
        are used.

        Definitions
        -----------
        :prf:ref:`minimally k-redundantly (generically) dim-rigid
        <def-redundantly-rigid-graph>`

        Parameters
        ----------
        k:
            level of redundancy
        dim:
            dimension
        algorithm:
            See :meth:`.is_rigid` for the possible algorithms used
            for checking rigidity in this method.
        prob:
            bound on the probability for false negatives of the rigidity testing
            when ``algorithm="randomized"``.
            Warning: this is not the probability of wrong results in this method,
            but is just passed on to rigidity testing.

        Examples
        --------
         >>> G = Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 4], [3, 4]])
        >>> G.is_min_k_redundantly_rigid(1, 2)
        True
        >>> G.is_min_k_redundantly_rigid(2, 2)
        False
        >>> G = Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
        >>> G.is_k_redundantly_rigid(1, 2)
        True
        >>> G.is_min_k_redundantly_rigid(1, 2)
        False

        """  # noqa: E501

        _input_check.dimension(dim)
        _input_check.integrality_and_range(k, "k", min_val=0)
        self._input_check_no_loop()

        n = self.number_of_nodes()
        m = self.number_of_edges()
        # use bound from thm-minimal-1-edge-redundant-upper-edge-bound-dim2
        if dim == 2:
            if k == 1:
                if n >= 7 and m > 3 * n - 9:
                    return False

        if not self.is_k_redundantly_rigid(k, dim, algorithm, prob):
            return False

        # for the following we need to know that the graph is k-redundantly rigid
        if (
            dim == 2
            and (
                # basic edge bound
                (k == 1 and m == 2 * n - 2)
                or
                # edge bound from :prf:ref:`thm-2-edge-redundant-edge-bound-dim2`
                (k == 2 and n >= 5 and m == 2 * n)
                or
                # edge bound from :prf:ref:`thm-k-edge-redundant-edge-bound-dim2`
                (k >= 3 and n >= 6 * (k + 1) + 23 and m == ((k + 2) * n + 1) // 2)
            )
        ) or (
            dim == 3
            and (
                # edge bound from :prf:ref:`thm-2-edge-redundant-edge-bound-dim3`
                (k == 2 and n >= 14 and m == 3 * n - 4)
                or
                # edge bound from :prf:ref:`thm-k-edge-redundant-edge-bound-dim3`
                (
                    k >= 4
                    and n >= 12 * (k + 1) + 10
                    and n % 2 == 0
                    and m == ((k + 3) * n + 1) // 2
                )
            )
        ):
            return True

        # in all other cases check by definition
        G = deepcopy(self)
        for e in self.edge_list():
            G.delete_edge(e)
            if G.is_k_redundantly_rigid(k, dim, algorithm, prob):
                return False
            G.add_edge(*e)
        return True

    @doc_category("Generic rigidity")
    def is_rigid(
        self, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
    ) -> bool:
        """
        Check whether the graph is (generically) ``dim``-rigid.

        Definitions
        -----------
        :prf:ref:`(generically) dim-rigid <def-gen-rigid>`

        Parameters
        ----------
        dim:
            dimension
        algorithm:
            If ``"graphic"`` (only if ``dim=1``), then the graphic matroid
            is used, namely, it is checked whether the graph is connected.

            If ``"sparsity"`` (only if ``dim=2``),
            then the existence of a spanning
            :prf:ref:`(2,3)-tight <def-kl-sparse-tight>` subgraph and
            :prf:ref:`thm-2-gen-rigidity` are used.

            If ``"randomized"``, a probabilistic check is performed.
            It may give false negatives (with probability at most ``prob``),
            but no false positives. See :prf:ref:`thm-probabilistic-rigidity-check`.

            If ``"default"``, then ``"graphic"`` is used for ``dim=1``
            and ``"sparsity"`` for ``dim=2`` and ``"randomized"`` for ``dim>=3``.
        prob:
            Only relevant if ``algorithm="randomized"``,
            It determines the bound on the probability of
            the randomized algorithm to yield false negatives.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (3,0)])
        >>> G.is_rigid()
        False
        >>> G.add_edge(0,2)
        >>> G.is_rigid()
        True
        """
        _input_check.dimension(dim)
        self._input_check_no_loop()

        n = self.number_of_nodes()
        # edge count, compare :prf:ref:`thm-gen-rigidity-tight`
        if self.number_of_edges() < dim * n - math.comb(dim + 1, 2):
            return False
        # small graphs are rigid iff complete :pref:ref:`thm-gen-rigidity-small-complete`
        elif n <= dim + 1:
            return self.number_of_edges() == math.comb(n, 2)

        if algorithm == "default":
            if dim == 1:
                algorithm = "graphic"
            elif dim == 2:
                algorithm = "sparsity"
            else:
                algorithm = "randomized"
                self._warn_randomized_alg(self.is_rigid, "algorithm='randomized'")

        if algorithm == "graphic":
            _input_check.dimension_for_algorithm(dim, [1], "the graphic algorithm")
            return nx.is_connected(self)

        if algorithm == "sparsity":
            _input_check.dimension_for_algorithm(dim, [2], "the sparsity algorithm")
            self._build_pebble_digraph(2, 3)
            return self._pebble_digraph.number_of_edges() == 2 * n - 3

        if algorithm == "randomized":
            N = int((n * dim - math.comb(dim + 1, 2)) / prob)
            if N < 1:
                raise ValueError("The parameter prob is too large!")
            from pyrigi.framework import Framework

            F = Framework.Random(self, dim, rand_range=[1, N])
            return F.is_inf_rigid()

        raise NotSupportedValueError(algorithm, "algorithm", self.is_rigid)

    @doc_category("Generic rigidity")
    def is_min_rigid(
        self,
        dim: int = 2,
        algorithm: str = "default",
        use_precomputed_pebble_digraph: bool = False,
        prob: float = 0.0001,
    ) -> bool:
        """
        Check whether the graph is minimally (generically) ``dim``-rigid.

        Definitions
        -----------
        :prf:ref:`minimally (generically) dim-rigid
        <def-min-rigid-graph>`

        Parameters
        ----------
        dim:
            dimension
        algorithm:
            If ``"graphic"`` (only if ``dim=1``), then the graphic matroid
            is used, namely, it is checked whether the graph is a tree.

            If ``"sparsity"`` (only if ``dim=2``),
            then :prf:ref:`(2,3)-tightness <def-kl-sparse-tight>` and
            :prf:ref:`thm-2-gen-rigidity` are used.

            If ``"randomized"``, a probabilistic check is performed.
            It may give false negatives (with probability at most ``prob``),
            but no false positives. See :prf:ref:`thm-probabilistic-rigidity-check`.

            If ``"extension_sequence"`` (only if ``dim=2``),
            then the existence of a sequence
            of rigidity preserving extensions is checked,
            see :meth:`.has_extension_sequence`.

            If ``"default"``, then ``"graphic"`` is used for ``dim=1``
            and ``"sparsity"`` for ``dim=2`` and ``"randomized"`` for ``dim>=3``.
        use_precomputed_pebble_digraph:
            Only relevant if ``algorithm="sparsity"``.
            If ``True``, the pebble digraph present in the cache is used.
            If ``False``, recompute the pebble digraph.
            Use ``True`` only if you are certain that the pebble game digraph
            is consistent with the graph.
        prob:
            Only relevant if ``algorithm="randomized"``,
            It determines the bound on the probability of
            the randomized algorithm to yield false negatives.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (3,0), (1,3)])
        >>> G.is_min_rigid()
        True
        >>> G.add_edge(0,2)
        >>> G.is_min_rigid()
        False
        """
        _input_check.dimension(dim)
        self._input_check_no_loop()

        n = self.number_of_nodes()
        # edge count, compare :prf:ref:`thm-gen-rigidity-tight`
        if self.number_of_edges() != dim * n - math.comb(dim + 1, 2):
            return False
        # small graphs are minimally rigid iff complete
        # :pref:ref:`thm-gen-rigidity-small-complete`
        elif n <= dim + 1:
            return self.number_of_edges() == math.comb(n, 2)

        if algorithm == "default":
            if dim == 1:
                algorithm = "graphic"
            elif dim == 2:
                algorithm = "sparsity"
            else:
                algorithm = "randomized"
                self._warn_randomized_alg(self.is_min_rigid, "algorithm='randomized'")

        if algorithm == "graphic":
            _input_check.dimension_for_algorithm(dim, [1], "the graphic algorithm")
            return nx.is_tree(self)

        if algorithm == "sparsity":
            _input_check.dimension_for_algorithm(
                dim, [2], "the (2,3)-sparsity/tightness algorithm"
            )
            return self.is_kl_tight(
                2,
                3,
                algorithm="pebble",
                use_precomputed_pebble_digraph=use_precomputed_pebble_digraph,
            )

        if algorithm == "extension_sequence":
            _input_check.dimension_for_algorithm(
                dim, [1, 2], "the algorithm using extension sequences"
            )
            return self.has_extension_sequence(dim=dim)

        if algorithm == "randomized":
            N = int((n * dim - math.comb(dim + 1, 2)) / prob)
            if N < 1:
                raise ValueError("The parameter prob is too large!")
            from pyrigi.framework import Framework

            F = Framework.Random(self, dim, rand_range=[1, N])
            return F.is_min_inf_rigid()

        raise NotSupportedValueError(algorithm, "algorithm", self.is_min_rigid)

    @doc_category("Generic rigidity")
    def is_globally_rigid(
        self, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
    ) -> bool:
        """
        Check whether the graph is globally ``dim``-rigid.

        Definitions
        -----------
        :prf:ref:`globally dim-rigid
        <def-globally-rigid-graph>`

        Parameters
        ----------
        dim:
            dimension d for which we test whether the graph is globally $d$-rigid
        algorithm:
            If ``"graphic"`` (only if ``dim=1``), then 2-connectivity is checked.

            If ``"redundancy"`` (only if ``dim=2``),
            then :prf:ref:`thm-globally-redundant-3connected` is used.

            If ``"randomized"``, a probabilistic check is performed.
            It may give false negatives (with probability at most ``prob``),
            but no false positives. See :prf:ref:`thm-globally-randomize-algorithm`.

            If ``"default"``, then ``"graphic"`` is used for ``dim=1``,
            ``"redundancy"`` for ``dim=2``, and ``"randomized"`` for ``dim>=3``.
        prob:
            Only relevant if ``algorithm="randomized"``,
            It determines the bound on the probability of
            the randomized algorithm to yield false negatives.

        Definitions
        -----------
        :prf:ref:`Globally d-rigid graph <def-globally-rigid-graph>`

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,0)])
        >>> G.is_globally_rigid()
        True
        >>> import pyrigi.graphDB as graphs
        >>> J = graphs.ThreePrism()
        >>> J.is_globally_rigid(dim=3)
        False
        >>> J.is_globally_rigid()
        False
        >>> K = graphs.Complete(6)
        >>> K.is_globally_rigid()
        True
        >>> K.is_globally_rigid(dim=3)
        True
        >>> C = graphs.CompleteMinusOne(5)
        >>> C.is_globally_rigid()
        True
        >>> C.is_globally_rigid(dim=3)
        False
        """
        _input_check.dimension(dim)
        self._input_check_no_loop()

        # small graphs are globally rigid iff complete
        # :pref:ref:`thm-gen-rigidity-small-complete`
        n = self.number_of_nodes()
        if n <= dim + 1:
            return self.number_of_edges() == math.comb(n, 2)

        if algorithm == "default":
            if dim == 1:
                algorithm = "graphic"
            elif dim == 2:
                algorithm = "redundancy"
            else:
                algorithm = "randomized"
                self._warn_randomized_alg(
                    self.is_globally_rigid, "algorithm='randomized'"
                )

        if algorithm == "graphic":
            _input_check.dimension_for_algorithm(dim, [1], "the graphic algorithm")
            return self.vertex_connectivity() >= 2

        if algorithm == "redundancy":
            _input_check.dimension_for_algorithm(
                dim, [2], "the algorithm using redundancy"
            )
            return self.is_redundantly_rigid() and self.vertex_connectivity() >= 3

        if algorithm == "randomized":
            n = self.number_of_nodes()
            m = self.number_of_edges()
            t = n * dim - math.comb(dim + 1, 2)  # rank of the rigidity matrix
            N = int(1 / prob) * n * math.comb(n, 2) + 2

            if m < t:
                return False
            # take a random framework with integer coordinates
            from pyrigi.framework import Framework

            F = Framework.Random(self, dim=dim, rand_range=[1, N])
            stresses = F.stresses()
            if m == t:
                omega = zeros(F.rigidity_matrix().rows, 1)
                return F.stress_matrix(omega).rank() == n - dim - 1
            elif stresses:
                omega = sum(
                    [randint(1, N) * stress for stress in stresses], stresses[0]
                )
                return F.stress_matrix(omega).rank() == n - dim - 1
            else:
                raise RuntimeError(
                    "There must be at least one stress but none was found!"
                )
        raise NotSupportedValueError(algorithm, "algorithm", self.is_globally_rigid)

    @doc_category("Rigidity Matroid")
    def is_Rd_dependent(
        self,
        dim: int = 2,
        algorithm: str = "default",
        use_precomputed_pebble_digraph: bool = False,
    ) -> bool:
        """
        Return whether the edge set is dependent in the generic dim-rigidity matroid.

        See :meth:`.is_independent` for details.

        Definitions
        -----------
        * :prf:ref:`Dependence <def-matroid>`
        * :prf:ref:`Generic Rigidity Matroid <def-gen-rigidity-matroid>`

        Examples
        --------
        >>> from pyrigi import graphDB
        >>> G = graphDB.K33plusEdge()
        >>> G.is_Rd_dependent()
        True
        """
        return not self.is_Rd_independent(
            dim,
            algorithm=algorithm,
            use_precomputed_pebble_digraph=use_precomputed_pebble_digraph,
        )

    @doc_category("Rigidity Matroid")
    def is_Rd_independent(
        self,
        dim: int = 2,
        algorithm: str = "default",
        use_precomputed_pebble_digraph: bool = False,
    ) -> bool:
        """
        Return whether the edge set is independent in the generic dim-rigidity matroid.

        Definitions
        ---------
        * :prf:ref:`Independence <def-matroid>`
        * :prf:ref:`Generic Rigidity Matroid <def-gen-rigidity-matroid>`

        Parameters
        ---------
        dim:
            Dimension of the rigidity matroid
        algorithm:
            If ``"graphic"`` (only if ``dim=1``),
            then the (non-)presence of cycles is checked.

            If ``"sparsity"`` (only if ``dim=2``),
            then :prf:ref:`(2,3)-sparsity <def-kl-sparse-tight>` is checked.

            If ``"randomized"``, the following check is performed on a random framework:
            a set of edges forms an independent set in the rigidity matroid
            if and only if it has no self-stress, i.e.,
            there are no linear relations between the rows of the rigidity matrix.

            If ``"default"``, then ``"graphic"`` is used for ``dim=1``,
            ``"sparsity"`` for ``dim=2``, and ``"randomized"`` for ``dim>=3``.
        use_precomputed_pebble_digraph:
            Only relevant if ``algorithm="sparsity"``.
            If ``True``, the pebble digraph present in the cache is used.
            If ``False``, recompute the pebble digraph.
            Use ``True`` only if you are certain that the pebble game digraph
            is consistent with the graph.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (3,0)])
        >>> G.is_Rd_independent()
        True

        Suggested Improvements
        ----------------------
        ``prob`` parameter for the randomized algorithm.
        """
        _input_check.dimension(dim)
        self._input_check_no_loop()

        if algorithm == "default":
            if dim == 1:
                algorithm = "graphic"
            elif dim == 2:
                algorithm = "sparsity"
            else:
                algorithm = "randomized"
                self._warn_randomized_alg(
                    self.is_Rd_independent, explicit_call="algorithm='randomized"
                )

        if algorithm == "graphic":
            _input_check.dimension_for_algorithm(
                dim,
                [
                    1,
                ],
                "the graphic algorithm",
            )
            return len(nx.cycle_basis(self)) == 0

        if algorithm == "sparsity":
            _input_check.dimension_for_algorithm(dim, [2], "the sparsity algorithm")
            return self.is_kl_sparse(
                2, 3, use_precomputed_pebble_digraph=use_precomputed_pebble_digraph
            )

        elif algorithm == "randomized":
            F = self.random_framework(dim=dim)
            return len(F.stresses()) == 0

        raise NotSupportedValueError(algorithm, "algorithm", self.is_Rd_independent)

    @doc_category("Rigidity Matroid")
    def is_Rd_circuit(
        self,
        dim: int = 2,
        algorithm: str = "default",
        use_precomputed_pebble_digraph: bool = False,
    ) -> bool:
        """
        Return whether the edge set is a circuit in the generic dim-rigidity matroid.

        Definitions
        ---------
        * :prf:ref:`Circuit <def-matroid>`
        * :prf:ref:`Generic Rigidity Matroid <def-gen-rigidity-matroid>`

        Parameters
        ---------
        dim:
            Dimension of the rigidity matroid
        algorithm:
            If ``"graphic"`` (only if ``dim=1``),
            it is checked whether the graph is a union of cycles.

            If ``"sparsity"`` (only if ``dim=2``),
            a :prf:ref:`(2,3)-sparse <def-kl-sparse-tight>` spanning subgraph
            is computed and it is checked (using pebble games)
            whether it misses only a single edge
            whose fundamental circuit is the whole graph.

            If ``"randomized"``, it is checked using randomized
            :meth:`.is_Rd_independent` whether removing
            every single edge from the graph results in an Rd-independent graph.

            If ``"default"``, then ``"graphic"`` is used for ``dim=1``,
            ``"sparsity"`` for ``dim=2``, and ``"randomized"`` for ``dim>=3``.
        use_precomputed_pebble_digraph:
            Only relevant if ``algorithm="sparsity"``.
            If ``True``, the pebble digraph present in the cache is used.
            If ``False``, recompute the pebble digraph.
            Use ``True`` only if you are certain that the pebble game digraph
            is consistent with the graph.

        Examples
        --------
        >>> from pyrigi import graphDB
        >>> G = graphDB.K33plusEdge()
        >>> G.is_Rd_circuit()
        True
        >>> G.add_edge(1,2)
        >>> G.is_Rd_circuit()
        False

        Suggested Improvements
        ----------------------
        ``prob`` parameter for the randomized algorithm
        """
        _input_check.dimension(dim)
        self._input_check_no_loop()

        if algorithm == "default":
            if dim == 1:
                algorithm = "graphic"
            elif dim == 2:
                algorithm = "sparsity"
            else:
                algorithm = "randomized"
                self._warn_randomized_alg(
                    self.is_Rd_circuit, explicit_call="algorithm='randomized'"
                )

        if algorithm == "graphic":
            _input_check.dimension_for_algorithm(dim, [1], "the graphic algorithm")
            if not nx.is_connected(self):
                return False

            # Check if every vertex has degree 2
            for vertex in self.nodes:
                if self.degree(vertex) != 2:
                    return False
            return True

        if algorithm == "sparsity":
            _input_check.dimension_for_algorithm(dim, [2], "the sparsity algorithm")
            # get max sparse sugraph and check the fundamental circuit of
            # the one last edge
            if self.number_of_edges() != 2 * self.number_of_nodes() - 2:
                return False
            max_sparse_subgraph = self.spanning_kl_sparse_subgraph(
                K=2, L=3, use_precomputed_pebble_digraph=use_precomputed_pebble_digraph
            )
            if max_sparse_subgraph.number_of_edges() != 2 * self.number_of_nodes() - 3:
                return False

            remaining_edges = [
                e for e in self.edges() if not max_sparse_subgraph.has_edge(*e)
            ]
            if len(remaining_edges) != 1:
                # this should not happen
                raise RuntimeError

            return (
                len(
                    self._pebble_digraph.fundamental_circuit(
                        u=remaining_edges[0][0],
                        v=remaining_edges[0][1],
                    )
                )
                == self.number_of_nodes()
            )
        elif algorithm == "randomized":
            if self.is_Rd_independent(dim=dim, algorithm="randomized"):
                return False
            G = deepcopy(self)
            for e in G.edges:
                G.delete_edge(e)
                if G.is_Rd_dependent(dim=dim, algorithm="randomized"):
                    return False
                G.add_edge(*e)
            return True

        raise NotSupportedValueError(algorithm, "algorithm", self.is_Rd_circuit)

    @doc_category("Rigidity Matroid")
    def is_Rd_closed(self, dim: int = 2, algorithm: str = "default") -> bool:
        """
        Return whether the edge set is closed in the generic dim-rigidity matroid.

        Definitions
        -----------
        * :prf:ref:`Rd-closed <def-rank-function-closure>`
        * :prf:ref:`Generic Rigidity Matroid <def-gen-rigidity-matroid>`

        Parameters
        ---------
        dim:
            Dimension of the rigidity matroid
        algorithm:
            See :meth:`.Rd_closure` for the options.

        Examples
        --------
        >>> G = Graph([(0,1),(1,2),(0,2),(3,4)])
        >>> G.is_Rd_closed(dim=1)
        True
        """
        return Graph(self.Rd_closure(dim, algorithm)) == self

    @doc_category("Rigidity Matroid")
    def Rd_closure(self, dim: int = 2, algorithm: str = "default") -> list[Edge]:
        """
        Return the set of edges given by closure in the generic dim-rigidity matroid.

        Definitions
        -----------
        * :prf:ref:`Rd-closure <def-rank-function-closure>`
        * :prf:ref:`Generic Rigidity Matroid <def-gen-rigidity-matroid>`

        Parameters
        ---------
        dim:
            Dimension of the rigidity matroid
        algorithm:
            If ``"components"``, then the closure is
            computed using ``dim``-rigid components.

            If ``"randomized"``, then adding
            non-edges is tested one by one on a random framework.

            If ``"default"``, then ``"components"`` is used
            for ``dim=1`` and ``"randomized"`` for ``dim>=2``.

        Examples
        --------
        >>> G = Graph([(0,1),(0,2),(3,4)])
        >>> G.Rd_closure(dim=1)
        [[0, 1], [0, 2], [1, 2], [3, 4]]

        Suggested Improvements
        ----------------------
        ``prob`` parameter for the randomized algorithm
        """
        _input_check.dimension(dim)
        self._input_check_no_loop()

        if algorithm == "default":
            if dim == 1:
                algorithm = "components"
            else:
                algorithm = "randomized"
                self._warn_randomized_alg(self.Rd_closure, "algorithm='randomized'")

        if algorithm == "components":
            _input_check.dimension_for_algorithm(
                dim, [1], "the algorithm using rigid components"
            )
            return [
                [u, v]
                for comp in self.rigid_components(dim=dim, algorithm="graphic")
                for u, v in combinations(comp, 2)
            ]

        if algorithm == "randomized":
            F_rank = self.random_framework(dim=dim).rigidity_matrix_rank()
            G = deepcopy(self)
            result = G.edge_list()
            for e in combinations(self.vertex_list(), 2):
                if G.has_edge(*e):
                    continue
                G.add_edge(*e)
                F1 = G.random_framework(dim=dim)
                if F_rank == F1.rigidity_matrix_rank():
                    result.append(e)
                G.remove_edge(*e)
            return result

        raise NotSupportedValueError(algorithm, "algorithm", self.Rd_closure)

    def _create_rigid_comp_matrix(self, index_map: map[Vertex, int]) -> list[list[int]]:
        """
        Return a matrix that for every pairs of vertices gives if there is a rigid
        component spanning the two vertices.

        Notes
        -----
        The matrix is a symmetric matrix of size ``n`` times ``n``, where ``n`` is the
        number of vertices.
        Implemented with an index map. Running time is O(n^3 log(n)).

        Implementation details
        -----
        Iterate through the vertex pairs of the component and check if there exists
        a rigid component containing both. This can be done by trying to add a new edge
        between the vertices.
        If there is such a rigid component, we can add every vertex pair from there:
        they are certainly within a rigid component.
        The matrix we'll get is an adjacency matrix of a graph that consists of
        edge disjoint cliques.

        Proof: If w_1 is in a rigid component C_1 with u and v
        and w_2 is also in a rigid component C_2 with u and v,
        then there is a rigid component C_3 containing both w_1 and w_2,
        since C_1 and C_2 intersect in at least two vertices.

        Parameters
        ---------
        index_map:
            mapping from vertices from the connected component to indices
            - for better running time

        Examples
        --------
        >>> index_map = {0: 0, 1: 1, 2: 2, 3: 3} # identity
        >>> G = Graph([(0, 1), (0, 2), (1, 2), (1, 3)])
        >>> G._build_pebble_digraph(2, 3)
        >>> G._create_rigid_comp_matrix(index_map)
        [[True, True, True, False],
         [True, True, True, True],
         [True, True, True, False],
         [False, True, False, True]]
        """

        # create a nodes by nodes matrix
        comp_matrix = [
            [False for _ in range(len(index_map))] for _ in range(len(index_map))
        ]

        for u, i_u in index_map.items():
            for v, i_v in index_map.items():
                if u == v:
                    comp_matrix[i_u][i_v] = True
                elif i_u > i_v:
                    # calculate only once
                    comp_matrix[i_u][i_v] = comp_matrix[i_v][i_u]
                else:
                    circuit = self._pebble_digraph.fundamental_circuit(u, v)
                    if circuit is not None:
                        for x in circuit:
                            for y in circuit:
                                comp_matrix[index_map[x]][index_map[y]] = True

        return comp_matrix

    def _calculate_maximal_cliques(
        self, index_map: map[Vertex, int], adj_matrix: list[list[int]]
    ) -> list[list[Vertex]]:
        """
        Find all maximal cliques in a graph given by an adjacency matrix.

        Implementation details
        -----
        Iterate through all pairs of vertices, checks if they are in the same component,
        and then try to extend the pair into a maximal clique by adding every vertex
        that are connected to both vertices in the pair.
        If a clique is found, it is added to the list of maximal cliques.

        Notes
        -----
        Running time is O(n^3 log(n)).

        Parameters
        ----------
        index_map:
            mapping from vertices to indices
        adj_matrix:
            adjacency matrix of the graph

        Returns
        -------
        list of maximal cliques in adj_matrix
        """

        maximal_cliques = []

        for u, i_u in index_map.items():
            for v, i_v in index_map.items():
                # note that this will not return isolated vertices
                if i_u < i_v:
                    # new clique starts here
                    if adj_matrix[i_u][i_v] is True and not any(
                        u in c and v in c for c in maximal_cliques
                    ):
                        clique = {u, v}
                        for w, i_w in index_map.items():
                            # enough to check for two vertices
                            if (
                                (w not in clique)
                                and adj_matrix[i_w][i_u] is True
                                and adj_matrix[i_w][i_v] is True
                            ):
                                clique.add(w)
                        maximal_cliques.append(clique)

        return [list(c) for c in maximal_cliques]

    def _get_2D_rigid_components_using_pebble_digraph(self) -> list[list[Vertex]]:
        """
        Compute the 2D rigid components of the graph using the pebble game algorithm.

        Implementation details
        -----
        First, the graph is split into connected components, as only
        connected components can be rigid components. Only the components
        with more than one vertex are investigated further. For each of these
        components, the pebble digraph is constructed and checked if it is rigid.
        If it is, the component is added to the list of rigid components.
        If not, create a (vertex x vertex) matrix in which we store
        if there exists a rigid component spanning the two vertices.
        Then using this matrix as an adjacency matrix of a graph we compute
        the maximal cliques resulting in the maximal rigid components of the
        connected component, which is then added to the list of maximal rigid components.

        Notes
        -----
        Running time is O(n^3 log(n)).
        We could also use 2-connected components, but there are small issues
        with isolated vertices.

        Returns
        -------
        list of maximal 2D rigid components
        """
        maximal_rigid_components = []
        # build the pebble digraph
        self._build_pebble_digraph(2, 3)
        # if rigid, we are done. Easy check
        if self._pebble_digraph.number_of_edges() == 2 * self.number_of_nodes() - 3:
            return [list(self.nodes())]
        else:
            # check only the components that can even be part of one
            for connected_comp in nx.connected_components(self):
                if len(connected_comp) == 1:
                    maximal_rigid_components += [list(connected_comp)]
                else:
                    index_map = {vertex: i for i, vertex in enumerate(connected_comp)}
                    comp_matrix = self._create_rigid_comp_matrix(index_map=index_map)
                    # find cliques
                    maximal_rigid_components += self._calculate_maximal_cliques(
                        index_map=index_map, adj_matrix=comp_matrix
                    )
        return maximal_rigid_components

    @doc_category("Generic rigidity")
    def rigid_components(
        self, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
    ) -> list[list[Vertex]]:
        """
        List the vertex sets inducing vertex-maximal rigid subgraphs.

        Definitions
        -----
        :prf:ref:`Rigid components <def-rigid-components>`

        Parameters
        ---------
        dim:
            The dimension that is used for the rigidity check.
        algorithm:
            If ``"graphic"`` (only if ``dim=1``),
            then the connected components are returned.

            If ``subgraphs-pebble"`` (only if ``dim=2``),
            then all subgraphs are checked
            using :meth:`.is_rigid` with ``algorithm="pebble"``.

            If ``pebble"`` (only if ``dim=2``),
            then using the pebble game algorithm to find rigid components
            O(n^3 log(n)) algorithm.
            See :meth:`._get_2D_rigid_components_using_pebble_digraph`.

            If ``"randomized"``, all subgraphs are checked
            using randomized :meth:`.is_rigid`.

            If ``"default"``, then ``"graphic"`` is used for ``dim=1``,
            ``"pebble"`` for ``dim=2``, and ``"randomized"`` for ``dim>=3``.
        prob:
            bound on the probability for false negatives of the rigidity testing
            when ``algorithm="randomized"``.
            Warning: this is not the probability of wrong results in this method,
            but is just passed on to rigidity testing.

        Notes
        -----
        If the graph itself is rigid, it is clearly maximal and is returned.
        Every edge is part of a rigid component. Isolated vertices form
        additional rigid components.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (3,0)])
        >>> G.rigid_components(algorithm="randomized")
        [[0, 1], [0, 3], [1, 2], [2, 3]]

        >>> G = Graph([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0), (0,2), (5,3)])
        >>> G.is_rigid()
        False
        >>> G.rigid_components(algorithm="randomized")
        [[0, 5], [2, 3], [0, 1, 2], [3, 4, 5]]

        TODO
        ----
        Implement directly using pebble games for dim=2
        to replace ``subgraphs-pebble`` algorithm.
        """
        _input_check.dimension(dim)
        self._input_check_no_loop()

        if algorithm == "default":
            if dim == 1:
                algorithm = "graphic"
            else:
                if dim == 2:
                    algorithm = "pebble"
                else:
                    algorithm = "randomized"
                    self._warn_randomized_alg(
                        self.rigid_components, "algorithm='randomized'"
                    )

        if algorithm == "graphic":
            _input_check.dimension_for_algorithm(dim, [1], "the graphic algorithm")
            return [list(comp) for comp in nx.connected_components(self)]

        if algorithm == "pebble":
            _input_check.dimension_for_algorithm(
                dim, [2], "the rigid component algorithm based on pebble games"
            )
            return self._get_2D_rigid_components_using_pebble_digraph()

        if algorithm in ["randomized", "subgraphs-pebble"]:
            if not nx.is_connected(self):
                res = []
                for comp in nx.connected_components(self):
                    res += self.subgraph(comp).rigid_components(
                        dim, algorithm=algorithm
                    )
                return res

            if algorithm == "subgraphs-pebble":
                _input_check.dimension_for_algorithm(
                    dim, [2], "the subgraph algorithm using pebble games"
                )
                alg_is_rigid = "pebble"
            else:
                alg_is_rigid = "randomized"

            if self.is_rigid(dim, algorithm=alg_is_rigid, prob=prob):
                return [list(self)]

            rigid_subgraphs = {
                tuple(vertex_subset): True
                for n in range(2, self.number_of_nodes() - 1)
                for vertex_subset in combinations(self.nodes, n)
                if self.subgraph(vertex_subset).is_rigid(
                    dim, algorithm=alg_is_rigid, prob=prob
                )
            }

            sorted_rigid_subgraphs = sorted(
                rigid_subgraphs.keys(), key=lambda t: len(t), reverse=True
            )
            for i, H1 in enumerate(sorted_rigid_subgraphs):
                if rigid_subgraphs[H1] and i + 1 < len(sorted_rigid_subgraphs):
                    for H2 in sorted_rigid_subgraphs[i + 1 :]:
                        if set(H2).issubset(set(H1)):
                            rigid_subgraphs[H2] = False
            return [list(H) for H, is_max in rigid_subgraphs.items() if is_max]

        raise NotSupportedValueError(algorithm, "algorithm", self.rigid_components)

    @doc_category("Generic rigidity")
    def max_rigid_dimension(self, prob: float = 0.0001) -> int | Inf:
        """
        Compute the maximum dimension, in which a graph is
        generically rigid.

        For checking rigidity, the method uses a randomized algorithm,
        see :meth:`~.is_rigid` for details.

        Definitions
        -----------
        :prf:ref:`generically rigid <def-gen-rigid>`

        Parameters
        ----------
        prob:
            bound on the probability for false negatives of the rigidity testing
            Warning: this is not the probability of wrong results in this method,
            but is just passed on to rigidity testing.

        Notes
        -----
        This is done by taking the dimension predicted by the Maxwell count
        as a starting point and iteratively reducing the dimension until
        generic rigidity is found.
        This method returns `sympy.oo` (infinity) if and only if the graph
        is complete. It has the data type `Inf`.

        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> G = graphs.Complete(3)
        >>> rigid_dim = G.max_rigid_dimension(); rigid_dim
        oo
        >>> rigid_dim.is_infinite
        True

        >>> import pyrigi.graphDB as graphs
        >>> G = graphs.Complete(4)
        >>> G.add_edges([(0,4),(1,4),(2,4)])
        >>> G.max_rigid_dimension()
        3
        """
        self._input_check_no_loop()

        if not nx.is_connected(self):
            return 0

        n = self.number_of_nodes()
        m = self.number_of_edges()
        # Only the complete graph is rigid in all dimensions
        if m == n * (n - 1) / 2:
            return oo
        # Find the largest d such that d*(d+1)/2 - d*n + m = 0
        max_dim = int(
            math.floor(0.5 * (2 * n + math.sqrt((1 - 2 * n) ** 2 - 8 * m) - 1))
        )
        self._warn_randomized_alg(self.max_rigid_dimension)
        for dim in range(max_dim, 0, -1):
            if self.is_rigid(dim, algorithm="randomized", prob=prob):
                return dim

    @doc_category("General graph theoretical properties")
    def is_isomorphic(self, graph: Graph) -> bool:
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
    def to_int(self, vertex_order: Sequence[Vertex] = None) -> int:
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

        Suggested Improvements
        ----------------------
        Implement taking canonical before computing the integer representation.
        """
        _input_check.greater_equal(self.number_of_edges(), 1, "number of edges")
        if self.min_degree() == 0:
            raise ValueError(
                "The integer representation only works "
                "for graphs without isolated vertices!"
            )
        self._input_check_no_loop()

        adj_matrix = self.adjacency_matrix(vertex_order)
        upper_diag = [
            str(b) for i, row in enumerate(adj_matrix.tolist()) for b in row[i + 1 :]
        ]
        return int("".join(upper_diag), 2)

    @classmethod
    @doc_category("Class methods")
    def from_int(cls, N: int) -> Graph:
        """
        Return a graph given its integer representation.

        See :meth:`~Graph.to_int` for the description
        of the integer representation.
        """
        _input_check.integrality_and_range(N, "parameter n", min_val=1)

        L = bin(N)[2:]
        c = math.ceil((1 + math.sqrt(1 + 8 * len(L))) / 2)
        rows = []
        s = 0
        L = "".join(["0" for _ in range(int(c * (c - 1) / 2) - len(L))]) + L
        for i in range(c):
            rows.append(
                [0 for _ in range(i + 1)] + [int(j) for j in L[s : s + (c - i - 1)]]
            )
            s += c - i - 1
        adj_matrix = Matrix(rows)
        return Graph.from_adjacency_matrix(adj_matrix + adj_matrix.transpose())

    @classmethod
    @doc_category("Class methods")
    def from_adjacency_matrix(cls, adj_matrix: Matrix) -> Graph:
        """
        Create a graph from a given adjacency matrix.

        Examples
        --------
        >>> M = Matrix([[0,1],[1,0]])
        >>> G = Graph.from_adjacency_matrix(M)
        >>> print(G)
        Graph with vertices [0, 1] and edges [[0, 1]]
        """
        if not adj_matrix.is_square:
            raise ValueError("The matrix is not square!")
        if not adj_matrix.is_symmetric():
            raise ValueError("The matrix is not symmetric!")

        vertices = range(adj_matrix.cols)
        edges = []
        for i, j in combinations(vertices, 2):
            if not (adj_matrix[i, j] == 0 or adj_matrix[i, j] == 1):
                raise ValueError(
                    "The provided adjacency matrix contains entries other than 0 and 1!"
                )
            if adj_matrix[i, j] == 1:
                edges += [(i, j)]
        for i in vertices:
            if adj_matrix[i, i] == 1:
                edges += [(i, i)]
        return Graph.from_vertices_and_edges(vertices, edges)

    @doc_category("General graph theoretical properties")
    def adjacency_matrix(self, vertex_order: Sequence[Vertex] = None) -> Matrix:
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
        vertex_order = self._input_check_vertex_order(vertex_order)

        row_list = [
            [+((v1, v2) in self.edges) for v2 in vertex_order] for v1 in vertex_order
        ]

        return Matrix(row_list)

    @doc_category("Other")
    def random_framework(self, dim: int = 2, rand_range: int | Sequence[int] = None):
        # the return type is intentionally omitted to avoid circular import
        """
        Return framework with random realization.

        This method calls :meth:`.Framework.Random`.
        """
        from pyrigi.framework import Framework

        return Framework.Random(self, dim, rand_range)

    @doc_category("Other")
    def to_tikz(
        self,
        layout_type: str = "spring",
        placement: dict[Vertex, Point] = None,
        vertex_style: str | list[str : Sequence[Vertex]] = "gvertex",
        edge_style: str | dict[str : Sequence[Edge]] = "edge",
        label_style: str = "labelsty",
        figure_opts: str = "",
        vertex_in_labels: bool = False,
        vertex_out_labels: bool = False,
        default_styles: bool = True,
    ) -> str:
        r"""
        Create a TikZ code for the graph.

        For using it in ``LaTeX`` you need to use the ``tikz`` package.

        Parameters
        ----------
        placement:
            If ``placement`` is not specified,
            then it is generated depending on parameter ``layout``.
        layout_type:
            The possibilities are ``spring`` (default), ``circular``,
            ``random`` or ``planar``, see also :meth:`~Graph.layout`.
        vertex_style:
            If a single style is given as a string,
            then all vertices get this style.
            If a dictionary from styles to a list of vertices is given,
            vertices are put in style accordingly.
            The vertices missing in the dictionary do not get a style.
        edge_style:
            If a single style is given as a string,
            then all edges get this style.
            If a dictionary from styles to a list of edges is given,
            edges are put in style accordingly.
            The edges missing in the dictionary do not get a style.
        label_style:
            The style for labels that are placed next to vertices.
        figure_opts:
            Options for the tikzpicture environment.
        vertex_in_labels
            A bool on whether vertex names should be put as labels on the vertices.
        vertex_out_labels
            A bool on whether vertex names should be put next to vertices.
        default_styles
            A bool on whether default style definitions should be put to the options.

        Examples
        ----------
        >>> G = Graph([(0,1), (1,2), (2,3), (0,3)])
        >>> print(G.to_tikz()) # doctest: +SKIP
        \begin{tikzpicture}[gvertex/.style={fill=black,draw=white,circle,inner sep=0pt,minimum size=4pt},edge/.style={line width=1.5pt,black!60!white}]
            \node[gvertex] (0) at (-0.98794, -0.61705) {};
            \node[gvertex] (1) at (0.62772, -1.0) {};
            \node[gvertex] (2) at (0.98514, 0.62151) {};
            \node[gvertex] (3) at (-0.62492, 0.99554) {};
            \draw[edge] (0) to (1) (0) to (3) (1) to (2) (2) to (3);
        \end{tikzpicture}

        >>> print(G.to_tikz(layout_type = "circular")) # doctest: +NORMALIZE_WHITESPACE
        \begin{tikzpicture}[gvertex/.style={fill=black,draw=white,circle,inner sep=0pt,minimum size=4pt},edge/.style={line width=1.5pt,black!60!white}]
            \node[gvertex] (0) at (1.0, 0.0) {};
            \node[gvertex] (1) at (-0.0, 1.0) {};
            \node[gvertex] (2) at (-1.0, -0.0) {};
            \node[gvertex] (3) at (0.0, -1.0) {};
            \draw[edge] (0) to (1) (0) to (3) (1) to (2) (2) to (3);
        \end{tikzpicture}

        >>> print(G.to_tikz(placement = {0:[0, 0], 1:[1, 1], 2:[2, 2], 3:[3, 3]})) # doctest: +NORMALIZE_WHITESPACE
        \begin{tikzpicture}[gvertex/.style={fill=black,draw=white,circle,inner sep=0pt,minimum size=4pt},edge/.style={line width=1.5pt,black!60!white}]
            \node[gvertex] (0) at (0, 0) {};
            \node[gvertex] (1) at (1, 1) {};
            \node[gvertex] (2) at (2, 2) {};
            \node[gvertex] (3) at (3, 3) {};
            \draw[edge] (0) to (1) (0) to (3) (1) to (2) (2) to (3);
        \end{tikzpicture}

        >>> print(G.to_tikz(layout_type = "circular", vertex_out_labels = True)) # doctest: +NORMALIZE_WHITESPACE
        \begin{tikzpicture}[gvertex/.style={fill=black,draw=white,circle,inner sep=0pt,minimum size=4pt},edge/.style={line width=1.5pt,black!60!white},labelsty/.style={font=\scriptsize,black!70!white}]
            \node[gvertex,label={[labelsty]right:$0$}] (0) at (1.0, 0.0) {};
            \node[gvertex,label={[labelsty]right:$1$}] (1) at (-0.0, 1.0) {};
            \node[gvertex,label={[labelsty]right:$2$}] (2) at (-1.0, -0.0) {};
            \node[gvertex,label={[labelsty]right:$3$}] (3) at (0.0, -1.0) {};
            \draw[edge] (0) to (1) (0) to (3) (1) to (2) (2) to (3);
        \end{tikzpicture}

        >>> print(G.to_tikz(layout_type = "circular", vertex_in_labels = True)) # doctest: +NORMALIZE_WHITESPACE
        \begin{tikzpicture}[gvertex/.style={white,fill=black,draw=black,circle,inner sep=1pt,font=\scriptsize},edge/.style={line width=1.5pt,black!60!white}]
            \node[gvertex] (0) at (1.0, 0.0) {$0$};
            \node[gvertex] (1) at (-0.0, 1.0) {$1$};
            \node[gvertex] (2) at (-1.0, -0.0) {$2$};
            \node[gvertex] (3) at (0.0, -1.0) {$3$};
            \draw[edge] (0) to (1) (0) to (3) (1) to (2) (2) to (3);
        \end{tikzpicture}

        >>> print(G.to_tikz(layout_type = "circular", vertex_style = "myvertex", edge_style = "myedge")) # doctest: +NORMALIZE_WHITESPACE
        \begin{tikzpicture}[]
            \node[myvertex] (0) at (1.0, 0.0) {};
            \node[myvertex] (1) at (-0.0, 1.0) {};
            \node[myvertex] (2) at (-1.0, -0.0) {};
            \node[myvertex] (3) at (0.0, -1.0) {};
            \draw[myedge] (0) to (1) (0) to (3) (1) to (2) (2) to (3);
        \end{tikzpicture}

        >>> print(G.to_tikz(layout_type = "circular", edge_style = {"red edge": [[1, 2]], "green edge": [[2, 3], [0, 1]]}, vertex_style = {"red vertex": [0], "blue vertex": [2, 3]})) # doctest: +NORMALIZE_WHITESPACE
        \begin{tikzpicture}[]
            \node[red vertex] (0) at (1.0, 0.0) {};
            \node[blue vertex] (2) at (-1.0, -0.0) {};
            \node[blue vertex] (3) at (0.0, -1.0) {};
            \node[] (1) at (-0.0, 1.0) {};
            \draw[red edge] (1) to (2);
            \draw[green edge] (2) to (3) (0) to (1);
            \draw[] (3) to (0);
        \end{tikzpicture}
        """  # noqa: E501

        # strings for tikz styles
        if vertex_out_labels and default_styles:
            label_style_str = r"labelsty/.style={font=\scriptsize,black!70!white}"
        else:
            label_style_str = ""

        if vertex_style == "gvertex" and default_styles:
            if vertex_in_labels:
                vertex_style_str = (
                    "gvertex/.style={white,fill=black,draw=black,circle,"
                    r"inner sep=1pt,font=\scriptsize}"
                )
            else:
                vertex_style_str = (
                    "gvertex/.style={fill=black,draw=white,circle,inner sep=0pt,"
                    "minimum size=4pt}"
                )
        else:
            vertex_style_str = ""
        if edge_style == "edge" and default_styles:
            edge_style_str = "edge/.style={line width=1.5pt,black!60!white}"
        else:
            edge_style_str = ""

        figure_str = [figure_opts, vertex_style_str, edge_style_str, label_style_str]
        figure_str = [fs for fs in figure_str if fs != ""]
        figure_str = ",".join(figure_str)

        # tikz for edges
        edge_style_dict = {}
        if type(edge_style) is str:
            edge_style_dict[edge_style] = self.edge_list()
        else:
            dict_edges = []
            for estyle, elist in edge_style.items():
                cdict_edges = [ee for ee in elist if self.has_edge(*ee)]
                edge_style_dict[estyle] = cdict_edges
                dict_edges += cdict_edges
            remaining_edges = [
                ee
                for ee in self.edge_list()
                if not ((ee in dict_edges) or (ee.reverse() in dict_edges))
            ]
            edge_style_dict[""] = remaining_edges

        edges_str = ""
        for estyle, elist in edge_style_dict.items():
            edges_str += (
                f"\t\\draw[{estyle}] "
                + " ".join([" to ".join([f"({v})" for v in e]) for e in elist])
                + ";\n"
            )

        # tikz for vertices
        if placement is None:
            placement = self.layout(layout_type)

        vertex_style_dict = {}
        if type(vertex_style) is str:
            vertex_style_dict[vertex_style] = self.vertex_list()
        else:
            dict_vertices = []
            for style, vertex_list in vertex_style.items():
                cdict_vertices = [v for v in vertex_list if (v in self.vertex_list())]
                vertex_style_dict[style] = cdict_vertices
                dict_vertices += cdict_vertices
            remaining_vertices = [
                v for v in self.vertex_list() if not (v in dict_vertices)
            ]
            vertex_style_dict[""] = remaining_vertices

        vertices_str = ""
        for style, vertex_list in vertex_style_dict.items():
            vertices_str += "".join(
                [
                    "\t\\node["
                    + style
                    + (
                        ("," if vertex_style != "" else "")
                        + f"label={{[{label_style}]right:${v}$}}"
                        if vertex_out_labels
                        else ""
                    )
                    + f"] ({v}) at "
                    + f"({round(placement[v][0], 5)}, {round(placement[v][1], 5)}) {{"
                    + (f"${v}$" if vertex_in_labels else "")
                    + "};\n"
                    for v in vertex_list
                ]
            )
        return (
            "\\begin{tikzpicture}["
            + figure_str
            + "]\n"
            + vertices_str
            + edges_str
            + "\\end{tikzpicture}"
        )

    @doc_category("Graph manipulation")
    def sum_t(self, other_graph: Graph, edge: Edge, t: int = 2):
        """
        Return the t-sum of ``self`` and ``other_graph`` along the given edge.

        Definitions
        -----------
        :prf:ref:`t-sum <def-t-sum>`

        Parameters
        ----------
        other_graph: Graph
        edge: Edge
        t: integer, default value 2

        Examples
        --------
        >>> H = Graph([[1,2],[2,3],[3,1],[3,4]])
        >>> G = Graph([[0,1],[1,2],[2,3],[3,1]])
        >>> H.sum_t(G, [1, 2], 3)
        Graph with vertices [0, 1, 2, 3, 4] and edges [[0, 1], [1, 3], [2, 3], [3, 4]]
        """
        if edge not in self.edges or edge not in other_graph.edges:
            raise ValueError(
                f"The edge {edge} is not in the intersection of the graphs!"
            )
        # check if the intersection is a t-complete graph
        if not self.intersection(other_graph).is_isomorphic(nx.complete_graph(t)):
            raise ValueError(
                f"The intersection of the graphs must be a {t}-complete graph!"
            )
        G = self + other_graph
        G.remove_edge(edge[0], edge[1])
        return G

    @doc_category("Graph manipulation")
    def intersection(self, other_graph: Graph):
        """
        Return the intersection of ``self`` and ``other_graph``.

        Parameters
        ----------
        other_graph: Graph

        Examples
        --------
        >>> H = Graph([[1,2],[2,3],[3,1],[3,4]])
        >>> G = Graph([[0,1],[1,2],[2,3],[3,1]])
        >>> G.intersection(H)
        Graph with vertices [1, 2, 3] and edges [[1, 2], [1, 3], [2, 3]]
        >>> G = Graph([[0,1],[0,2],[1,2]])
        >>> G.add_vertex(3)
        >>> H = Graph([[0,1],[1,2],[2,4],[4,0]])
        >>> H.add_vertex(3)
        >>> G.intersection(H)
        Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [1, 2]]
        """
        return Graph.from_vertices_and_edges(
            [v for v in self.nodes if v in other_graph.nodes],
            [e for e in self.edges if e in other_graph.edges],
        )

    @doc_category("Other")
    def layout(self, layout_type: str = "spring") -> dict[Vertex, Point]:
        """
        Generate a placement of the vertices.

        This method is a wrapper for the functions
        :func:`~networkx.drawing.layout.spring_layout`,
        :func:`~networkx.drawing.layout.random_layout`,
        :func:`~networkx.drawing.layout.circular_layout`
        and :func:`~networkx.drawing.layout.planar_layout`

        Parameters
        ----------
        layout_type:
            The supported layouts are ``circular``, ``planar``,
            ``random`` and ``spring`` (default).
        """
        if layout_type == "circular":
            return nx.drawing.layout.circular_layout(self)
        elif layout_type == "planar":
            return nx.drawing.layout.planar_layout(self)
        elif layout_type == "random":
            return nx.drawing.layout.random_layout(self)
        elif layout_type == "spring":
            return nx.drawing.layout.spring_layout(self)
        else:
            raise NotSupportedValueError(layout_type, "layout_type", self.layout)

    @doc_category("Other")
    def plot(
        self,
        plot_style: PlotStyle = None,
        placement: dict[Vertex, Point] = None,
        layout: str = "spring",
        **kwargs,
    ) -> None:
        """
        Plot the graph.

        See also :class:`.PlotStyle`,
        :meth:`~.Framework.plot`, :meth:`~.Framework.plot2D` and
        :meth:`~.Framework.plot3D` for possible parameters for formatting.
        To distinguish :meth:`.Framework.plot` from this method,
        the ``vertex_color`` has a different default value.

        Parameters
        ----------
        plot_style:
            An instance of the :class:`.PlotStyle` class
            that defines the visual style for plotting.
            See :class:`.PlotStyle` for more information.
        placement:
            If ``placement`` is not specified,
            then it is generated depending on parameter ``layout``.
        layout:
            The possibilities are ``spring`` (default), ``circular``,
            ``random`` or ``planar``, see also :meth:`~Graph.layout`.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (0,3)])
        >>> G.plot()

        Using keyword arguments for customizing the plot style,
        see :class:`.PlotStyle` and :class:`.PlotStyle2D` for all possible options.

        >>> G.plot(vertex_color="#FF0000", edge_color="black", vertex_size=50)

        Specifying a custom plot style

        >>> from pyrigi import PlotStyle
        >>> plot_style = PlotStyle(vertex_color="#FF0000")
        >>> G.plot(plot_style)

        Using different layout

        >>> G.plot(layout="circular")

        Using custom placement for vertices

        >>> placement = {0: (1,2), 1: (2,3), 2: (3,4), 3: (4,5)}
        >>> G.plot(placement=placement)

        Combining different customizations

        >>> G.plot(plot_style, layout="random", placement=placement)

        The following is just to close all figures after running the example:

        >>> import matplotlib.pyplot
        >>> matplotlib.pyplot.close("all")
        """
        if plot_style is None:
            plot_style = PlotStyle(vertex_color="#4169E1")

        if placement is None:
            placement = self.layout(layout)
        if (
            set(placement.keys()) != set(self.nodes)
            or len(placement.keys()) != len(self.nodes)
            or any(
                [
                    len(pos) != len(placement[list(placement.keys())[0]])
                    for pos in placement.values()
                ]
            )
        ):
            raise TypeError("The placement does not have the correct format!")
        from pyrigi import Framework

        Framework(self, placement).plot(plot_style=plot_style, **kwargs)


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
            "Rigidity Matroid",
            "Other",
            "Waiting for implementation",
        ],
        include_all=False,
    ),
)
