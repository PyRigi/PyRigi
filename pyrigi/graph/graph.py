"""
Module for rigidity related graph properties.
"""

from __future__ import annotations

import math
from copy import deepcopy
from itertools import combinations
from typing import Collection, Iterable, Optional

import networkx as nx
from sympy import Matrix

import pyrigi.misc._input_check as _input_check
from pyrigi.data_type import Edge, Inf, Point, Sequence, Vertex
from pyrigi.exception import NotSupportedValueError
from pyrigi.misc._wrap import copy_doc
from pyrigi.misc.misc import _doc_category as doc_category
from pyrigi.misc.misc import _generate_category_tables
from pyrigi.plot_style import PlotStyle

from . import _input_check as _graph_input_check
from . import apex, general, separating_set
from .constructions import extensions
from .rigidity import generic as generic_rigidity
from .rigidity import global_ as global_rigidity
from .rigidity import matroidal as matroidal_rigidity
from .rigidity import redundant as redundant_rigidity
from .sparsity import sparsity

__doctest_requires__ = {("Graph.number_of_realizations",): ["lnumber"]}


class Graph(nx.Graph):
    """
    Class representing a graph.

    One option for ``incoming_graph_data`` is a list of edges.
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
        Return a representation of a graph.
        """
        o_str = f"Graph.from_vertices_and_edges({self.vertex_list()}, "
        o_str += f"{self.edge_list(as_tuples=True)})"
        return o_str

    def __eq__(self, other: Graph) -> bool:
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

    def __add__(self, other: Graph) -> Graph:
        r"""
        Return the union of the given graph and ``other``.

        Definitions
        -----------
        :prf:ref:`Union of two graphs <def-union-graph>`

        Examples
        --------
        >>> G = Graph([[0,1],[1,2],[2,0]])
        >>> H = Graph([[2,3],[3,4],[4,2]])
        >>> graph = G + H
        >>> print(graph)
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
        vertices:
            The vertex set.
        edges:
            The edge set.

        Examples
        --------
        >>> G = Graph.from_vertices_and_edges([0, 1, 2, 3], [])
        >>> print(G)
        Graph with vertices [0, 1, 2, 3] and edges []
        >>> G = Graph.from_vertices_and_edges([0, 1, 2, 3], [[0, 1], [0, 2], [1, 3]])
        >>> print(G)
        Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 2], [1, 3]]
        >>> G = Graph.from_vertices_and_edges(['a', 'b', 'c', 'd'], [['a','c'], ['a', 'd']])
        >>> print(G)
        Graph with vertices ['a', 'b', 'c', 'd'] and edges [['a', 'c'], ['a', 'd']]
        """  # noqa: E501
        G = Graph()
        G.add_nodes_from(vertices)
        _graph_input_check.edge_format_list(G, edges)
        G.add_edges(edges)
        return G

    @classmethod
    @doc_category("Class methods")
    def from_vertices(cls, vertices: Sequence[Vertex]) -> Graph:
        """
        Create a graph with no edges from a list of vertices.

        Parameters
        ----------
        vertices

        Examples
        --------
        >>> from pyrigi import Graph
        >>> G = Graph.from_vertices([3, 1, 7, 2, 12, 3, 0])
        >>> print(G)
        Graph with vertices [0, 1, 2, 3, 7, 12] and edges []
        """
        return Graph.from_vertices_and_edges(vertices, [])

    @doc_category("Attribute getters")
    def vertex_list(self) -> list[Vertex]:
        """
        Return the list of vertices.

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

        The output is sorted if possible,
        otherwise, the internal order is used instead.

        Parameters
        ----------
        as_tuples:
            If ``True``, all edges are returned as tuples instead of lists.

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
        """
        Alias for :meth:`networkx.Graph.remove_node`.

        Parameters
        ----------
        vertex
        """
        self.remove_node(vertex)

    @doc_category("Graph manipulation")
    def delete_vertices(self, vertices: Sequence[Vertex]) -> None:
        """
        Alias for :meth:`networkx.Graph.remove_nodes_from`.

        Parameters
        ----------
        vertices
        """
        self.remove_nodes_from(vertices)

    @doc_category("Graph manipulation")
    def delete_edge(self, edge: Edge) -> None:
        """
        Alias for :meth:`networkx.Graph.remove_edge`

        Parameters
        ----------
        edge
        """
        self.remove_edge(*edge)

    @doc_category("Graph manipulation")
    def delete_edges(self, edges: Sequence[Edge]) -> None:
        """
        Alias for :meth:`networkx.Graph.remove_edges_from`.

        Parameters
        ----------
        edges
        """
        self.remove_edges_from(edges)

    @doc_category("Graph manipulation")
    def add_vertex(self, vertex: Vertex) -> None:
        """
        Alias for :meth:`networkx.Graph.add_node`.

        Parameters
        ----------
        vertex
        """
        self.add_node(vertex)

    @doc_category("Graph manipulation")
    def add_vertices(self, vertices: Sequence[Vertex]) -> None:
        """
        Alias for :meth:`networkx.Graph.add_nodes_from`.

        Parameters
        ----------
        vertices
        """
        self.add_nodes_from(vertices)

    @doc_category("Graph manipulation")
    def add_edges(self, edges: Sequence[Edge]) -> None:
        """
        Alias for :meth:`networkx.Graph.add_edges_from`.

        Parameters
        ----------
        edges
        """
        self.add_edges_from(edges)

    @doc_category("Graph manipulation")
    def delete_loops(self) -> None:
        """Remove all the loops from the edges to get a loop free graph."""
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
        vertex_order = _graph_input_check.is_vertex_order(self, vertex_order)
        return [int(self.degree(v)) for v in vertex_order]

    @doc_category("General graph theoretical properties")
    @copy_doc(general.min_degree)
    def min_degree(self) -> int:
        return general.min_degree(self)

    @doc_category("General graph theoretical properties")
    @copy_doc(general.max_degree)
    def max_degree(self) -> int:
        return general.max_degree(self)

    @doc_category("Sparseness")
    @copy_doc(sparsity.spanning_kl_sparse_subgraph)
    def spanning_kl_sparse_subgraph(
        self, K: int, L: int, use_precomputed_pebble_digraph: bool = False
    ) -> Graph:
        return sparsity.spanning_kl_sparse_subgraph(
            self, K, L, use_precomputed_pebble_digraph=use_precomputed_pebble_digraph
        )

    @doc_category("Sparseness")
    @copy_doc(sparsity.is_kl_sparse)
    def is_kl_sparse(
        self,
        K: int,
        L: int,
        algorithm: str = "default",
        use_precomputed_pebble_digraph: bool = False,
    ) -> bool:
        return sparsity.is_kl_sparse(
            self,
            K,
            L,
            algorithm=algorithm,
            use_precomputed_pebble_digraph=use_precomputed_pebble_digraph,
        )

    @doc_category("Sparseness")
    def is_sparse(self) -> bool:
        r"""
        Return whether the graph is (2,3)-sparse.

        For general $(k,\ell)$-sparsity, see :meth:`.is_kl_sparse`.

        Definitions
        -----------
        :prf:ref:`(2,3)-sparsity <def-kl-sparse-tight>`

        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> graphs.Path(3).is_sparse()
        True
        >>> graphs.Complete(4).is_sparse()
        False
        >>> graphs.ThreePrism().is_sparse()
        True

        Notes
        -----
        The pebble game algorithm is used (see :prf:ref:`alg-pebble-game`).
        """
        return self.is_kl_sparse(2, 3, algorithm="pebble")

    @doc_category("Sparseness")
    @copy_doc(sparsity.is_kl_tight)
    def is_kl_tight(
        self,
        K: int,
        L: int,
        algorithm: str = "default",
        use_precomputed_pebble_digraph: bool = False,
    ) -> bool:
        return sparsity.is_kl_tight(
            self,
            K,
            L,
            algorithm=algorithm,
            use_precomputed_pebble_digraph=use_precomputed_pebble_digraph,
        )

    @doc_category("Sparseness")
    def is_tight(self) -> bool:
        r"""
        Return whether the graph is (2,3)-tight.

        For general $(k,\ell)$-tightness, see :meth:`.is_kl_tight`.

        Definitions
        -----------
        :prf:ref:`(2,3)-tightness <def-kl-sparse-tight>`

        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> graphs.Path(4).is_tight()
        False
        >>> graphs.ThreePrism().is_tight()
        True

        Notes
        -----
        The pebble game algorithm is used (see :prf:ref:`alg-pebble-game`).
        """
        return self.is_kl_tight(2, 3, algorithm="pebble")

    @doc_category("Graph manipulation")
    @copy_doc(extensions.zero_extension)
    def zero_extension(
        self,
        vertices: Sequence[Vertex],
        new_vertex: Vertex = None,
        dim: int = 2,
        inplace: bool = False,
    ) -> Graph:
        return extensions.zero_extension(
            self, vertices=vertices, new_vertex=new_vertex, dim=dim, inplace=inplace
        )

    @doc_category("Graph manipulation")
    @copy_doc(extensions.one_extension)
    def one_extension(
        self,
        vertices: Sequence[Vertex],
        edge: Edge,
        new_vertex: Vertex = None,
        dim: int = 2,
        inplace: bool = False,
    ) -> Graph:
        return extensions.one_extension(
            self,
            vertices=vertices,
            edge=edge,
            new_vertex=new_vertex,
            dim=dim,
            inplace=inplace,
        )

    @doc_category("Graph manipulation")
    @copy_doc(extensions.k_extension)
    def k_extension(
        self,
        k: int,
        vertices: Sequence[Vertex],
        edges: Sequence[Edge],
        new_vertex: Vertex = None,
        dim: int = 2,
        inplace: bool = False,
    ) -> Graph:
        return extensions.k_extension(
            self,
            k=k,
            vertices=vertices,
            edges=edges,
            new_vertex=new_vertex,
            dim=dim,
            inplace=inplace,
        )

    @doc_category("Graph manipulation")
    @copy_doc(extensions.all_k_extensions)
    def all_k_extensions(
        self,
        k: int,
        dim: int = 2,
        only_non_isomorphic: bool = False,
    ) -> Iterable[Graph]:
        return extensions.all_k_extensions(
            self, k=k, dim=dim, only_non_isomorphic=only_non_isomorphic
        )

    @doc_category("Graph manipulation")
    @copy_doc(extensions.all_extensions)
    def all_extensions(
        self,
        dim: int = 2,
        only_non_isomorphic: bool = False,
        k_min: int = 0,
        k_max: int | None = None,
    ) -> Iterable[Graph]:
        return extensions.all_extensions(
            self,
            dim=dim,
            only_non_isomorphic=only_non_isomorphic,
            k_min=k_min,
            k_max=k_max,
        )

    @doc_category("Generic rigidity")
    @copy_doc(extensions.extension_sequence)
    def extension_sequence(  # noqa: C901
        self, dim: int = 2, return_type: str = "extensions"
    ) -> list[Graph] | list | None:
        return extensions.extension_sequence(self, dim=dim, return_type=return_type)

    @doc_category("Generic rigidity")
    @copy_doc(extensions.has_extension_sequence)
    def has_extension_sequence(
        self,
        dim: int = 2,
    ) -> bool:
        return extensions.has_extension_sequence(self, dim=dim)

    @doc_category("Graph manipulation")
    def cone(self, inplace: bool = False, vertex: Vertex = None) -> Graph:
        """
        Return a coned version of the graph.

        Definitions
        -----------
        :prf:ref:`Cone graph <def-cone-graph>`

        Parameters
        ----------
        inplace:
            If ``True``, the graph is modified,
            otherwise a new modified graph is created,
            while the original graph remains unchanged (default).
        vertex:
            It is possible to give the added cone vertex a name using
            the keyword ``vertex``.

        Examples
        --------
        >>> G = Graph([(0,1)]).cone()
        >>> G.is_isomorphic(Graph([(0,1),(1,2),(0,2)]))
        True
        """
        if vertex in self.nodes:
            raise KeyError(f"Vertex {vertex} is already a vertex of the graph!")
        if vertex is None:
            vertex = self.number_of_nodes()
            while vertex in self.nodes:
                vertex += 1

        if inplace:
            self.add_edges([(u, vertex) for u in self.nodes])
            return self
        else:
            G = deepcopy(self)
            G.add_edges([(u, vertex) for u in G.nodes])
            return G

    @doc_category("Generic rigidity")
    def number_of_realizations(
        self,
        dim: int = 2,
        spherical: bool = False,
        check_min_rigid: bool = True,
        count_reflection: bool = False,
    ) -> int:
        """
        Count the number of complex realizations of a minimally ``dim``-rigid graph.

        Realizations in ``dim``-dimensional sphere
        can be counted using ``spherical=True``.

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
        dim:
            The dimension in which the realizations are counted.
            Currently, only ``dim=2`` is supported.
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

        Suggested Improvements
        ----------------------
        Implement the counting for ``dim=1``.
        """  # noqa: E501
        _input_check.dimension_for_algorithm(
            dim, [2], "the method number_of_realizations"
        )

        try:
            import lnumber

            if check_min_rigid and not generic_rigidity.is_min_rigid(self):
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
    @copy_doc(redundant_rigidity.is_vertex_redundantly_rigid)
    def is_vertex_redundantly_rigid(
        self, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
    ) -> bool:
        return redundant_rigidity.is_vertex_redundantly_rigid(
            self, dim=dim, algorithm=algorithm, prob=prob
        )

    @doc_category("Generic rigidity")
    @copy_doc(redundant_rigidity.is_k_vertex_redundantly_rigid)
    def is_k_vertex_redundantly_rigid(
        self,
        k: int,
        dim: int = 2,
        algorithm: str = "default",
        prob: float = 0.0001,
    ) -> bool:
        return redundant_rigidity.is_k_vertex_redundantly_rigid(
            self,
            k,
            dim=dim,
            algorithm=algorithm,
            prob=prob,
        )

    @doc_category("Generic rigidity")
    @copy_doc(redundant_rigidity.is_min_vertex_redundantly_rigid)
    def is_min_vertex_redundantly_rigid(
        self, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
    ) -> bool:
        return redundant_rigidity.is_min_vertex_redundantly_rigid(
            self, dim=dim, algorithm=algorithm, prob=prob
        )

    @doc_category("Generic rigidity")
    @copy_doc(redundant_rigidity.is_min_k_vertex_redundantly_rigid)
    def is_min_k_vertex_redundantly_rigid(
        self,
        k: int,
        dim: int = 2,
        algorithm: str = "default",
        prob: float = 0.0001,
    ) -> bool:
        return redundant_rigidity.is_min_k_vertex_redundantly_rigid(
            self, k, dim=dim, algorithm=algorithm, prob=prob
        )

    @doc_category("Generic rigidity")
    @copy_doc(redundant_rigidity.is_redundantly_rigid)
    def is_redundantly_rigid(
        self, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
    ) -> bool:
        return redundant_rigidity.is_redundantly_rigid(
            self, dim=dim, algorithm=algorithm, prob=prob
        )

    @doc_category("Generic rigidity")
    @copy_doc(redundant_rigidity.is_k_redundantly_rigid)
    def is_k_redundantly_rigid(
        self,
        k: int,
        dim: int = 2,
        algorithm: str = "default",
        prob: float = 0.0001,
    ) -> bool:
        return redundant_rigidity.is_k_redundantly_rigid(
            self, k, dim=dim, algorithm=algorithm, prob=prob
        )

    @doc_category("Generic rigidity")
    @copy_doc(redundant_rigidity.is_min_redundantly_rigid)
    def is_min_redundantly_rigid(
        self, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
    ) -> bool:
        return redundant_rigidity.is_min_redundantly_rigid(
            self, dim=dim, algorithm=algorithm, prob=prob
        )

    @doc_category("Generic rigidity")
    @copy_doc(redundant_rigidity.is_min_k_redundantly_rigid)
    def is_min_k_redundantly_rigid(
        self,
        k: int,
        dim: int = 2,
        algorithm: str = "default",
        prob: float = 0.0001,
    ) -> bool:
        return redundant_rigidity.is_min_k_redundantly_rigid(
            self, k, dim=dim, algorithm=algorithm, prob=prob
        )

    @doc_category("Generic rigidity")
    @copy_doc(generic_rigidity.is_rigid)
    def is_rigid(
        self,
        dim: int = 2,
        algorithm: str = "default",
        use_precomputed_pebble_digraph: bool = False,
        prob: float = 0.0001,
    ) -> bool:
        return generic_rigidity.is_rigid(
            self,
            dim=dim,
            algorithm=algorithm,
            use_precomputed_pebble_digraph=use_precomputed_pebble_digraph,
            prob=prob,
        )

    @doc_category("Generic rigidity")
    @copy_doc(generic_rigidity.is_min_rigid)
    def is_min_rigid(
        self,
        dim: int = 2,
        algorithm: str = "default",
        use_precomputed_pebble_digraph: bool = False,
        prob: float = 0.0001,
    ) -> bool:
        return generic_rigidity.is_min_rigid(
            self,
            dim=dim,
            algorithm=algorithm,
            use_precomputed_pebble_digraph=use_precomputed_pebble_digraph,
            prob=prob,
        )

    @doc_category("Generic rigidity")
    @copy_doc(global_rigidity.is_globally_rigid)
    def is_globally_rigid(
        self, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
    ) -> bool:
        return global_rigidity.is_globally_rigid(
            self, dim=dim, algorithm=algorithm, prob=prob
        )

    @doc_category("Rigidity Matroid")
    @copy_doc(matroidal_rigidity.is_Rd_dependent)
    def is_Rd_dependent(
        self,
        dim: int = 2,
        algorithm: str = "default",
        use_precomputed_pebble_digraph: bool = False,
    ) -> bool:
        return matroidal_rigidity.is_Rd_dependent(
            self,
            dim=dim,
            algorithm=algorithm,
            use_precomputed_pebble_digraph=use_precomputed_pebble_digraph,
        )

    @doc_category("Rigidity Matroid")
    @copy_doc(matroidal_rigidity.is_Rd_independent)
    def is_Rd_independent(
        self,
        dim: int = 2,
        algorithm: str = "default",
        use_precomputed_pebble_digraph: bool = False,
    ) -> bool:
        return matroidal_rigidity.is_Rd_independent(
            self,
            dim=dim,
            algorithm=algorithm,
            use_precomputed_pebble_digraph=use_precomputed_pebble_digraph,
        )

    @doc_category("Rigidity Matroid")
    @copy_doc(matroidal_rigidity.is_Rd_circuit)
    def is_Rd_circuit(  # noqa: C901
        self,
        dim: int = 2,
        algorithm: str = "default",
        use_precomputed_pebble_digraph: bool = False,
    ) -> bool:
        return matroidal_rigidity.is_Rd_circuit(
            self,
            dim=dim,
            algorithm=algorithm,
            use_precomputed_pebble_digraph=use_precomputed_pebble_digraph,
        )

    @doc_category("Rigidity Matroid")
    @copy_doc(matroidal_rigidity.is_Rd_closed)
    def is_Rd_closed(self, dim: int = 2, algorithm: str = "default") -> bool:
        return matroidal_rigidity.is_Rd_closed(self, dim=dim, algorithm=algorithm)

    @doc_category("Rigidity Matroid")
    @copy_doc(matroidal_rigidity.Rd_closure)
    def Rd_closure(self, dim: int = 2, algorithm: str = "default") -> list[Edge]:
        return matroidal_rigidity.Rd_closure(self, dim=dim, algorithm=algorithm)

    @doc_category("Generic rigidity")
    @copy_doc(generic_rigidity.rigid_components)
    def rigid_components(  # noqa: 901
        self, dim: int = 2, algorithm: str = "default", prob: float = 0.0001
    ) -> list[list[Vertex]]:
        return generic_rigidity.rigid_components(
            self, dim=dim, algorithm=algorithm, prob=prob
        )

    @doc_category("Generic rigidity")
    @copy_doc(generic_rigidity.max_rigid_dimension)
    def max_rigid_dimension(
        self, algorithm: str = "randomized", prob: float = 0.0001
    ) -> int | Inf:
        return generic_rigidity.max_rigid_dimension(
            self, algorithm=algorithm, prob=prob
        )

    @doc_category("General graph theoretical properties")
    def is_isomorphic(self, graph: Graph) -> bool:
        """
        Return whether two graphs are isomorphic.

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
        """
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
        _graph_input_check.no_loop(self)

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
        requires ``scipy``. To avoid unnecessary imports, the method is implemented here.
        """
        vertex_order = _graph_input_check.is_vertex_order(self, vertex_order)

        row_list = [
            [+((v1, v2) in self.edges) for v2 in vertex_order] for v1 in vertex_order
        ]

        return Matrix(row_list)

    @doc_category("Other")
    def random_framework(self, dim: int = 2, rand_range: int | Sequence[int] = None):
        # the return type is intentionally omitted to avoid circular import
        """
        Return a framework with random realization.

        This method calls :meth:`.Framework.Random`.
        """
        from pyrigi.framework import Framework

        return Framework.Random(self, dim, rand_range)

    @doc_category("Other")
    def to_tikz(
        self,
        layout_type: str = "spring",
        placement: dict[Vertex, Point] = None,
        vertex_style: str | dict[str, Sequence[Vertex]] = "gvertex",
        edge_style: str | dict[str, Sequence[Edge]] = "edge",
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
        --------
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

        >>> print(G.to_tikz(
        ...     layout_type = "circular",
        ...     vertex_style = "myvertex",
        ...     edge_style = "myedge")
        ... ) # doctest: +NORMALIZE_WHITESPACE
        \begin{tikzpicture}[]
            \node[myvertex] (0) at (1.0, 0.0) {};
            \node[myvertex] (1) at (-0.0, 1.0) {};
            \node[myvertex] (2) at (-1.0, -0.0) {};
            \node[myvertex] (3) at (0.0, -1.0) {};
            \draw[myedge] (0) to (1) (0) to (3) (1) to (2) (2) to (3);
        \end{tikzpicture}

        >>> print(G.to_tikz(
        ...     layout_type="circular",
        ...     edge_style={"red edge": [[1, 2]], "green edge": [[2, 3], [0, 1]]},
        ...     vertex_style={"red vertex": [0], "blue vertex": [2, 3]})
        ... ) # doctest: +NORMALIZE_WHITESPACE
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
    def sum_t(self, other_graph: Graph, edge: Edge, t: int = 2) -> Graph:
        """
        Return the ``t``-sum with ``other_graph`` along the given edge.

        Definitions
        -----------
        :prf:ref:`t-sum <def-t-sum>`

        Examples
        --------
        >>> G1 = Graph([[1,2],[2,3],[3,1],[3,4]])
        >>> G2 = Graph([[0,1],[1,2],[2,3],[3,1]])
        >>> H = G2.sum_t(G1, [1, 2], 3)
        >>> print(H)
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

    @doc_category("General graph theoretical properties")
    @copy_doc(apex.is_vertex_apex)
    def is_vertex_apex(self) -> bool:
        return apex.is_vertex_apex(self)

    @doc_category("General graph theoretical properties")
    @copy_doc(apex.is_k_vertex_apex)
    def is_k_vertex_apex(self, k: int) -> bool:
        return apex.is_k_vertex_apex(self, k)

    @doc_category("General graph theoretical properties")
    @copy_doc(apex.is_edge_apex)
    def is_edge_apex(self) -> bool:
        return apex.is_edge_apex(self)

    @doc_category("General graph theoretical properties")
    @copy_doc(apex.is_k_edge_apex)
    def is_k_edge_apex(self, k: int) -> bool:
        return apex.is_k_edge_apex(self, k)

    @doc_category("General graph theoretical properties")
    @copy_doc(apex.is_critically_vertex_apex)
    def is_critically_vertex_apex(self) -> bool:
        return apex.is_critically_vertex_apex(self)

    @doc_category("General graph theoretical properties")
    @copy_doc(apex.is_critically_k_vertex_apex)
    def is_critically_k_vertex_apex(self, k: int) -> bool:
        return apex.is_critically_k_vertex_apex(self, k)

    @doc_category("General graph theoretical properties")
    @copy_doc(apex.is_critically_edge_apex)
    def is_critically_edge_apex(self) -> bool:
        return apex.is_critically_edge_apex(self)

    @doc_category("General graph theoretical properties")
    @copy_doc(apex.is_critically_k_edge_apex)
    def is_critically_k_edge_apex(self, k: int) -> bool:
        return apex.is_critically_k_edge_apex(self, k)

    @doc_category("Graph manipulation")
    def intersection(self, other_graph: Graph) -> Graph:
        """
        Return the intersection with ``other_graph``.

        Parameters
        ----------
        other_graph: Graph

        Examples
        --------
        >>> H = Graph([[1,2],[2,3],[3,1],[3,4]])
        >>> G = Graph([[0,1],[1,2],[2,3],[3,1]])
        >>> graph = G.intersection(H)
        >>> print(graph)
        Graph with vertices [1, 2, 3] and edges [[1, 2], [1, 3], [2, 3]]
        >>> G = Graph([[0,1],[0,2],[1,2]])
        >>> G.add_vertex(3)
        >>> H = Graph([[0,1],[1,2],[2,4],[4,0]])
        >>> H.add_vertex(3)
        >>> graph = G.intersection(H)
        >>> print(graph)
        Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [1, 2]]
        """
        return Graph.from_vertices_and_edges(
            [v for v in self.nodes if v in other_graph.nodes],
            [e for e in self.edges if e in other_graph.edges],
        )

    @doc_category("General graph theoretical properties")
    @copy_doc(separating_set.is_stable_set)
    def is_stable_set(
        self,
        vertices: Collection[Vertex],
        certificate: bool = False,
    ) -> bool | tuple[bool, Optional[Edge]]:
        return separating_set.is_stable_set(
            self, vertices=vertices, certificate=certificate
        )

    @doc_category("General graph theoretical properties")
    @copy_doc(separating_set.is_separating_set)
    def is_separating_set(
        self,
        vertices: Collection[Vertex],
        use_copy: bool = True,
    ) -> bool:
        return separating_set.is_separating_set(
            self, vertices=vertices, use_copy=use_copy
        )

    @doc_category("General graph theoretical properties")
    @copy_doc(separating_set.is_uv_separating_set)
    def is_uv_separating_set(
        self,
        vertices: Collection[Vertex],
        u: Vertex,
        v: Vertex,
        use_copy: bool = True,
    ) -> bool:
        return separating_set.is_uv_separating_set(
            self, vertices=vertices, u=u, v=v, use_copy=use_copy
        )

    @doc_category("General graph theoretical properties")
    @copy_doc(separating_set.is_stable_separating_set)
    def is_stable_separating_set(
        self,
        vertices: Collection[Vertex],
        use_copy: bool = True,
    ) -> bool:
        return separating_set.is_stable_separating_set(
            self, vertices=vertices, use_copy=use_copy
        )

    @doc_category("General graph theoretical properties")
    @copy_doc(separating_set.stable_separating_set)
    def stable_separating_set(
        self,
        u: Optional[Vertex] = None,
        v: Optional[Vertex] = None,
        check_flexible: bool = True,
        check_connected: bool = True,
        check_distinct_rigid_components: bool = True,
    ) -> set[Vertex]:
        return separating_set.stable_separating_set(
            self,
            u=u,
            v=v,
            check_flexible=check_flexible,
            check_connected=check_connected,
            check_distinct_rigid_components=check_distinct_rigid_components,
        )

    @doc_category("Generic rigidity")
    @copy_doc(generic_rigidity.is_linked)
    def is_linked(self, u: Vertex, v: Vertex, dim: int = 2) -> bool:
        return generic_rigidity.is_linked(self, u, v, dim=dim)

    @doc_category("Generic rigidity")
    @copy_doc(global_rigidity.is_weakly_globally_linked)
    def is_weakly_globally_linked(self, u: Vertex, v: Vertex, dim: int = 2) -> bool:
        return global_rigidity.is_weakly_globally_linked(self, u, v, dim=dim)

    @doc_category("Other")
    def layout(self, layout_type: str = "spring") -> dict[Vertex, Point]:
        """
        Generate a placement of the vertices.

        This method is a wrapper for the functions
        :func:`~networkx.drawing.layout.spring_layout`,
        :func:`~networkx.drawing.layout.random_layout`,
        :func:`~networkx.drawing.layout.circular_layout`
        and :func:`~networkx.drawing.layout.planar_layout`.

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
    _generate_category_tables(
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
        add_attributes=False,
    ),
)
