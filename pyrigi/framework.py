"""

Module for the functionality concerning frameworks.

.. currentmodule:: pyrigi.framework

Classes:

.. autosummary::

    Framework

"""

from __future__ import annotations
from typing import List, Any, Dict, Union

from copy import deepcopy
from itertools import combinations
from random import randrange

import networkx as nx
import sympy as sp
from sympy import Matrix, flatten, binomial

from pyrigi.data_type import Vertex, Edge, Point, point_to_vector
from pyrigi.graph import Graph
from pyrigi.exception import LoopError
from pyrigi.graphDB import Complete as CompleteGraph
from pyrigi.misc import (
    doc_category,
    generate_category_tables,
    check_integrality_and_range,
    is_zero_vector,
)


class Framework(object):
    r"""
    This class provides the functionality for frameworks.

    Definitions
    -----------
     * :prf:ref:`Framework <def-framework>`
     * :prf:ref:`Realization <def-realization>`

    Parameters
    ----------
    graph:
        A graph without loops.
    realization:
        A dictionary mapping the vertices of the graph to points in $\RR^d$.
        The dimension ``d`` is retrieved from the points in realization.
        If ``graph`` is empty, and hence also the ``realization``,
        the dimension is set to 0 (:meth:`Framework.Empty`
        can be used to construct an empty framework with different dimension).

    Examples
    --------
    >>> F = Framework(Graph([[0,1]]), {0:[1,2], 1:[0,5]})
    >>> F
    Framework in 2-dimensional space consisting of:
    Graph with vertices [0, 1] and edges [[0, 1]]
    Realization {0:(1, 2), 1:(0, 5)}

    Notice that the realization of a vertex can be accessed using ``[ ]``:

    >>> F[0]
    Matrix([
    [1],
    [2]])

    TODO
    ----
    Use :meth:`~.Framework.set_realization` in the constructor.

    METHODS

    Notes
    -----
    Internally, the realization is represented as ``Dict[Vertex,Matrix]``.
    However, :meth:`~Framework.realization` can also return ``Dict[Vertex,Point]``.
    """

    def __init__(self, graph: Graph, realization: Dict[Vertex, Point]) -> None:
        if not isinstance(graph, Graph):
            raise TypeError("The graph has to be an instance of class Graph.")
        if nx.number_of_selfloops(graph) > 0:
            raise LoopError()
        if not len(realization.keys()) == graph.number_of_nodes():
            raise KeyError(
                "The length of realization has to be equal to "
                "the number of vertices of graph."
            )

        if realization:
            self._dim = len(list(realization.values())[0])
        else:
            self._dim = 0

        for v in graph.nodes:
            if v not in realization:
                raise KeyError(f"Vertex {v} is not contained in the realization.")
            if not len(realization[v]) == self._dim:
                raise ValueError(
                    f"The point {realization[v]} in the realization corresponding to "
                    f"vertex {v} does not have the right dimension."
                )

        self._realization = {v: Matrix(realization[v]) for v in graph.nodes}
        self._graph = deepcopy(graph)

    def __str__(self) -> str:
        """Return the string representation."""
        return (
            self.__class__.__name__
            + f" in {self.dim()}-dimensional space consisting of:\n{self._graph}\n"
            + "Realization {"
            + ", ".join(
                [
                    f"{v}:{tuple(self._realization[v])}"
                    for v in self._graph.vertex_list()
                ]
            )
            + "}"
        )

    def __repr__(self) -> str:
        """Return the representation"""
        return self.__str__()

    def __getitem__(self, vertex: Vertex) -> Matrix:
        """
        Return the coordinates corresponding to the image
        of a given vertex under the realization map.

        Examples
        --------
        >>> F = Framework(Graph([[0,1]]), {0:[1,2], 1:[0,5]})
        >>> F[0]
        Matrix([
        [1],
        [2]])
        """
        return self._realization[vertex]

    @doc_category("Attribute getters")
    def dim(self) -> int:
        """Return the dimension of the framework."""
        return self._dim

    @doc_category("Attribute getters")
    def dimension(self) -> int:
        """Alias for :meth:`~Framework.dim`"""
        return self.dim()

    @doc_category("Framework manipulation")
    def add_vertex(self, point: Point, vertex: Vertex = None) -> None:
        """
        Add a vertex to the framework with the corresponding coordinates.

        If no vertex is provided (``None``), then the smallest,
        free integer is chosen instead.

        Parameters
        ----------
        point:
            the realization of the new vertex
        vertex:
            the label of the new vertex

        Examples
        --------
        >>> F = Framework.Empty(dim=2)
        >>> F.add_vertex((1.5,2), 'a')
        >>> F.add_vertex((3,1))
        >>> F
        Framework in 2-dimensional space consisting of:
        Graph with vertices ['a', 1] and edges []
        Realization {a:(1.50000000000000, 2), 1:(3, 1)}
        """
        if vertex is None:
            candidate = self._graph.number_of_nodes()
            while candidate in self._graph.nodes:
                candidate += 1
            vertex = candidate

        if vertex in self._graph.nodes:
            raise KeyError(f"Vertex {vertex} is already a vertex of the graph!")

        self._realization[vertex] = Matrix(point)
        self._graph.add_node(vertex)

    @doc_category("Framework manipulation")
    def add_vertices(self, points: List[Point], vertices: List[Vertex] = []) -> None:
        r"""
        Add a list of vertices to the framework.

        Parameters
        ----------
        points:
            List of points consisting of coordinates in $\RR^d$. It is checked
            that all points lie in the same ambient space.
        vertices:
            List of vertices. If the list of vertices is empty, we generate a
            vertex that is not yet taken with the method :meth:`add_vertex`.
            Else, the list of vertices needs to have the same length as the
            list of points.

        Examples
        --------
        >>> F = Framework.Empty(dim=2)
        >>> F.add_vertices([(1.5,2), (3,1)], ['a',0])
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices ['a', 0] and edges []
        Realization {a:(1.50000000000000, 2), 0:(3, 1)}

        Notes
        -----
        For each vertex that has to be added, :meth:`add_vertex` is called.
        """
        if not (len(points) == len(vertices) or not vertices):
            raise IndexError("The vertex list does not have the correct length!")
        if not vertices:
            for point in points:
                self.add_vertex(point)
        else:
            for p, v in zip(points, vertices):
                self.add_vertex(p, v)

    @doc_category("Framework manipulation")
    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge to the framework.

        Notes
        -----
        This method only alters the graph attribute.
        """
        self._graph._check_edge_format(edge)
        self._graph.add_edge(*(edge))

    @doc_category("Framework manipulation")
    def add_edges(self, edges: List[Edge]) -> None:
        """
        Add a list of edges to the framework.

        Notes
        -----
        For each edge that has to be added, :meth:`add_edge` is called.
        """
        for edge in edges:
            self.add_edge(edge)

    @doc_category("Attribute getters")
    def graph(self) -> Graph:
        """
        Return a copy of the underlying graph.

        TODO
        ----
        example
        """
        return deepcopy(self._graph)

    @doc_category("Other")
    def plot(
        self,
        vertex_color="#ff8c00",
        edge_width=1.5,
        **kwargs,
    ) -> None:
        """
        Plot the framework.

        For various formatting options, see :meth:`.Graph.plot`.

        Parameters
        ----------


        TODO
        ----
        implement plotting also for other dimensions than 2
        """

        if self._dim != 2:
            raise NotImplementedError(
                "Plotting is currently supported only for 2-dimensional frameworks."
            )

        self._graph.plot(
            placement=self.realization(as_points=True, numerical=True),
            vertex_color=vertex_color,
            edge_width=edge_width,
            **kwargs,
        )

    @classmethod
    @doc_category("Class methods")
    def from_points(cls, points: List[Point]) -> Framework:
        """
        Generate a framework from a list of points.

        The list of vertices of the underlying graph
        is taken to be ``[0,...,len(points)-1]``.
        The underlying graph has no edges.

        Examples
        --------
        >>> F = Framework.from_points([(1,2), (2,3)])
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0, 1] and edges []
        Realization {0:(1, 2), 1:(2, 3)}
        """
        vertices = range(len(points))
        realization = {v: points[v] for v in vertices}
        return Framework(Graph.from_vertices(vertices), realization)

    @classmethod
    @doc_category("Class methods")
    def Random(
        cls, graph: Graph, dim: int = 2, rand_range: Union(int, List[int]) = None
    ) -> Framework:
        """
        Return a framework with random realization.

        Examples
        --------
        >>> F = Framework.Random(Graph([(0,1), (1,2), (0,2)]))
        >>> print(F) # doctest: +SKIP
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        Realization {0:(122, 57), 1:(27, 144), 2:(50, 98)}

        TODO
        ----
        Set the correct default range value.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if rand_range is None:
            b = 10 * graph.number_of_nodes() ** 2 * dim
            a = -b
        if isinstance(rand_range, list):
            if not len(rand_range) == 2:
                raise ValueError("If `rand_range` is a list, it must be of length 2.")
            a, b = rand_range
        if isinstance(rand_range, int):
            if rand_range <= 0:
                raise ValueError("If `rand_range` is an int, it must be positive")
            b = rand_range
            a = -b

        realization = {
            vertex: [randrange(a, b) for _ in range(dim)] for vertex in graph.nodes
        }

        return Framework(graph, realization)

    @classmethod
    @doc_category("Class methods")
    def Circular(cls, graph: Graph) -> Framework:
        """
        Return the framework with a regular unit circle realization in the plane.

        TODO
        ----
        example
        """
        n = graph.number_of_nodes()
        return Framework(
            graph,
            {
                v: [sp.cos(2 * i * sp.pi / n), sp.sin(2 * i * sp.pi / n)]
                for i, v in enumerate(graph.vertex_list())
            },
        )

    @classmethod
    @doc_category("Class methods")
    def Collinear(cls, graph: Graph, d: int = 1) -> Framework:
        """
        Return the framework with a realization on the x-axis in the d-dimensional space.

        TODO
        ----
        example
        """
        check_integrality_and_range(d, "dimension d", 1)
        return Framework(
            graph,
            {
                v: [i] + [0 for _ in range(d - 1)]
                for i, v in enumerate(graph.vertex_list())
            },
        )

    @classmethod
    @doc_category("Class methods")
    def Simplicial(cls, graph: Graph, d: int = None) -> Framework:
        """
        Return the framework with a realization on the d-simplex.

        Parameters
        ----------
        d:
            The dimension ``d`` has to be at least the number of vertices
            of the ``graph`` minus one.
            If ``d`` is not specified, then the least possible one is used.

        TODO
        ----
        examples
        """
        if d is None:
            d = graph.number_of_nodes() - 1
        check_integrality_and_range(
            d, "dimension d", max([1, graph.number_of_nodes() - 1])
        )
        return Framework(
            graph,
            {
                v: [1 if j == i - 1 else 0 for j in range(d)]
                for i, v in enumerate(graph.vertex_list())
            },
        )

    @classmethod
    @doc_category("Class methods")
    def Empty(cls, dim: int = 2) -> Framework:
        """
        Generate an empty framework.

        Parameters
        ----------
        dim:
            a natural number that determines the dimension
            in which the framework is realized

        TODO
        ----
        example
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        F = Framework(graph=Graph(), realization={})
        F._dim = dim
        return F

    @classmethod
    @doc_category("Class methods")
    def Complete(cls, points: List[Point]) -> Framework:
        """
        Generate a framework on the complete graph from a given list of points.

        The vertices of the underlying graph are taken
        to be the list ``[0,...,len(points)-1]``.

        Parameters
        ----------
        dim:
            a natural number that determines the dimension
            in which the framework is realized

        Examples
        --------
        >>> F = Framework.Complete([(1,),(2,),(3,),(4,)]); F
        Framework in 1-dimensional space consisting of:
        Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        Realization {0:(1,), 1:(2,), 2:(3,), 3:(4,)}
        """  # noqa: E501
        if not points:
            raise ValueError("The list of points cannot be empty.")

        Kn = CompleteGraph(len(points))
        return Framework(Kn, {v: Matrix(p) for v, p in zip(Kn.nodes, points)})

    @doc_category("Framework manipulation")
    def delete_vertex(self, vertex: Vertex) -> None:
        """
        Delete a vertex from the framework.
        """
        self._graph.delete_vertex(vertex)
        del self._realization[vertex]

    @doc_category("Framework manipulation")
    def delete_vertices(self, vertices: List[Vertex]) -> None:
        """
        Delete a list of vertices from the framework.
        """
        for vertex in vertices:
            self.delete_vertex(vertex)

    @doc_category("Framework manipulation")
    def delete_edge(self, edge: Edge) -> None:
        """
        Delete an edge from the framework.
        """
        self._graph.delete_edge(edge)

    @doc_category("Framework manipulation")
    def delete_edges(self, edges: List[Edge]) -> None:
        """
        Delete a list of edges from the framework.
        """
        self._graph.delete_edges(edges)

    @doc_category("Attribute getters")
    def realization(
        self, as_points: bool = False, numerical: bool = False
    ) -> Dict[Vertex, Point]:
        """
        Return a copy of the realization.

        Parameters
        ----------
        as_points:
            If ``True``, then the vertex positions type is Point,
            otherwise Matrix (default).
        numerical:
            If ``True``, the vertex positions are converted to floats.

        Examples
        --------
        >>> F = Framework.Complete([(0,0), (1,0), (1,1)])
        >>> F.realization(as_points=True)
        {0: [0, 0], 1: [1, 0], 2: [1, 1]}
        >>> F.realization()
        {0: Matrix([
        [0],
        [0]]), 1: Matrix([
        [1],
        [0]]), 2: Matrix([
        [1],
        [1]])}

        Notes
        -----
        The format returned by this method with ``as_points=True``
        can be read by networkx.
        """
        if not numerical:
            if not as_points:
                return deepcopy(self._realization)
            return {
                vertex: list(position) for vertex, position in self._realization.items()
            }
        else:
            if not as_points:
                {
                    vertex: Matrix([float(p) for p in position])
                    for vertex, position in self._realization.items()
                }
            return {
                vertex: [float(p) for p in position]
                for vertex, position in self._realization.items()
            }

    @doc_category("Framework properties")
    def is_quasi_injective(
        self, numerical: bool = False, tolerance: float = 1e-9
    ) -> bool:
        """
        Return whether the realization is :prf:ref:`quasi-injective <def-realization>`.

        For comparing whether two vectors are the same,
        :func:`.misc.is_zero_vector` is used.
        See its documentation for the description of the parameters.
        """

        for u, v in self._graph.edges:
            edge_vector = self[u] - self[v]
            if is_zero_vector(edge_vector, numerical, tolerance):
                return False
        return True

    @doc_category("Framework properties")
    def is_injective(self, numerical: bool = False, tolerance: float = 1e-9) -> bool:
        """
        Return whether the realization is injective.

        For comparing whether two vectors are the same,
        :func:`.misc.is_zero_vector` is used.
        See its documentation for the description of the parameters.
        """

        for u, v in combinations(self._graph.nodes, 2):
            edge_vector = self[u] - self[v]
            if is_zero_vector(edge_vector, numerical, tolerance):
                return False
        return True

    @doc_category("Framework manipulation")
    def set_realization(self, realization: Dict[Vertex, Point]) -> None:
        r"""
        Change the realization of the framework.

        Parameters
        ----------
        realization:
            a realization of the underlying graph of the framework

        Notes
        -----
        It is assumed that the realization contains all vertices from the
        underlying graph. Furthermore, all points in the realization need
        to be contained in $\RR^d$ for a fixed $d$.

        Examples
        --------
        >>> F = Framework.Complete([(0,0), (1,0), (1,1)])
        >>> F.set_realization({vertex:(vertex,vertex+1) for vertex in F.graph().vertex_list()})
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        Realization {0:(0, 1), 1:(1, 2), 2:(2, 3)}
        """  # noqa: E501
        if not len(realization) == self._graph.number_of_nodes():
            raise IndexError(
                "The realization does not contain the correct amount of vertices!"
            )
        for v in self._graph.nodes:
            if v not in realization:
                raise KeyError("Vertex {vertex} is not a key of the given realization!")
            if not len(realization[v]) == self.dimension():
                raise IndexError(
                    f"The element {realization[v]} does not have "
                    f"the dimension {self.dimension()}!"
                )
        self._realization = {v: Matrix(realization[v]) for v in realization.keys()}

    @doc_category("Framework manipulation")
    def set_vertex_pos(self, vertex: Vertex, point: Point) -> None:
        """
        Change the coordinates of a single given vertex.

        Examples
        --------
        >>> F = Framework.from_points([(0,0)])
        >>> F.set_vertex_pos(0, (6,2))
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0] and edges []
        Realization {0:(6, 2)}
        """
        if vertex not in self._realization:
            raise KeyError("Vertex {vertex} is not a key of the given realization!")
        if not len(point) == self.dimension():
            raise IndexError(
                f"The point {point} does not have the dimension {self.dimension()}!"
            )
        self._realization[vertex] = Matrix(point)

    @doc_category("Framework manipulation")
    def set_vertex_positions_from_lists(
        self, vertices: List[Vertex], points: List[Point]
    ) -> None:
        """
        Change the coordinates of a given list of vertices.

        TODO
        ----
        example

        Notes
        -----
        It is necessary that both lists have the same length.
        No vertex from ``vertices`` can be contained multiple times.
        We apply the method :meth:`~Framework.set_vertex_pos`
        to ``vertices`` and ``points``.
        """
        if len(list(set(vertices))) != len(list(vertices)):
            raise ValueError("Multiple Vertices with the same name were found!")
        if not len(vertices) == len(points):
            raise IndexError(
                "The list of vertices does not have the same length as the list of points"
            )
        self.set_vertex_positions({v: pos for v, pos in zip(vertices, points)})

    @doc_category("Framework manipulation")
    def set_vertex_positions(self, subset_of_realization: Dict[Vertex, Point]):
        """
        Change the coordinates of vertices given by a dictionary.

        TODO
        ----
        example
        """
        for v, pos in subset_of_realization.items():
            self.set_vertex_pos(v, pos)

    @doc_category("Infinitesimal rigidity")
    def rigidity_matrix(
        self,
        vertex_order: List[Vertex] = None,
        edge_order: List[Edge] = None,
    ) -> Matrix:
        r"""
        Construct the rigidity matrix of the framework.

        Definitions
        -----------
        * :prf:ref:`Rigidity matrix <def-rigidity-matrix>`

        Parameters
        ----------
        vertex_order:
            By listing vertices in the preferred order, the rigidity matrix
            can be computed in a way the user expects.
        edges_ordered:
            A Boolean indicating, whether the edges are assumed to be ordered (``True``),
            or whether they should be internally sorted (``False``).

        TODO
        ----
        tests

        Examples
        --------
        >>> F = Framework.Complete([(0,0),(2,0),(1,3)])
        >>> F.rigidity_matrix()
        Matrix([
        [-2,  0, 2,  0,  0, 0],
        [-1, -3, 0,  0,  1, 3],
        [ 0,  0, 1, -3, -1, 3]])
        """
        if vertex_order is None:
            vertex_order = self._graph.vertex_list()
        else:
            if not set(self._graph.nodes) == set(vertex_order):
                raise ValueError(
                    "vertex_order must contain "
                    "exactly the same vertices as the graph!"
                )
        if edge_order is None:
            edge_order = self._graph.edge_list()
        else:
            if not (
                set([set(e) for e in self._graph.edges])
                == set([set(e) for e in edge_order])
                and len(edge_order) == self._graph.number_of_edges()
            ):
                raise ValueError(
                    "edge_order must contain exactly the same edges as the graph!"
                )

        # ``delta`` is responsible for distinguishing the edges (i,j) and (j,i)
        def delta(e, w):
            # the parameter e represents an edge
            # the parameter w represents a vertex
            if w == e[0]:
                return 1
            if w == e[1]:
                return -1
            return 0

        return Matrix(
            [
                flatten(
                    [
                        delta(e, w)
                        * (self._realization[e[0]] - self._realization[e[1]])
                        for w in vertex_order
                    ]
                )
                for e in edge_order
            ]
        )

    def pinned_rigidity_matrix(
        self,
        pinned_vertices: Dict[Vertex, List[int]] = None,
        vertex_order: List[Vertex] = None,
        edge_order: List[Edge] = None,
    ) -> Matrix:
        r"""
        Construct the rigidity matrix of the framework.

        TODO
        ----
        definition of pinned rigidity matrix, tests

        Examples
        --------
        >>> F = Framework(Graph([[0, 1], [0, 2]]), {0: [0, 0], 1: [1, 0], 2: [1, 1]})
        >>> F.pinned_rigidity_matrix()
        Matrix([
        [-1,  0, 1, 0, 0, 0],
        [-1, -1, 0, 0, 1, 1],
        [ 1,  0, 0, 0, 0, 0],
        [ 0,  1, 0, 0, 0, 0],
        [ 0,  0, 1, 0, 0, 0]])
        """
        rigidity_matrix = self.rigidity_matrix(
            vertex_order=vertex_order, edge_order=edge_order
        )

        if vertex_order is None:
            vertex_order = self._graph.vertex_list()
        if edge_order is None:
            edge_order = self._graph.vertex_list()

        if pinned_vertices is None:
            freedom = self._dim * (self._dim + 1) // 2
            pinned_vertices = {}
            upper = self._dim + 1
            for v in vertex_order:
                upper -= 1
                frozen_coord = []
                for i in range(upper):
                    if freedom > 0:
                        frozen_coord.append(i)
                        freedom -= 1
                    else:
                        pinned_vertices[v] = frozen_coord
                        break
                pinned_vertices[v] = frozen_coord
        else:
            number_pinned = sum([len(coord) for coord in pinned_vertices.values()])
            if number_pinned > self._dim * (self._dim + 1) // 2:
                raise ValueError(
                    "The maximal number of coordinates that"
                    f"can be pinned is {self._dim * (self._dim + 1) // 2}, "
                    f"but you provided {number_pinned}."
                )
            for v in pinned_vertices:
                if min(pinned_vertices[v]) < 0 or max(pinned_vertices[v]) >= self._dim:
                    raise ValueError("Coordinate indices out of range.")

        pinning_rows = []
        for v in pinned_vertices:
            for coord in pinned_vertices[v]:
                idx = vertex_order.index(v)
                new_row = Matrix.zeros(1, self._dim * self._graph.number_of_nodes())
                new_row[idx * self._dim + coord] = 1
                pinning_rows.append(new_row)
        pinned_rigidity_matrix = Matrix.vstack(rigidity_matrix, *pinning_rows)
        return pinned_rigidity_matrix

    @doc_category("Waiting for implementation")
    def stress_matrix(
        self,
        data: Any,
        edge_order: List[Edge] = None,
    ) -> Matrix:
        r"""
        Construct the stress matrix from a stress of from its support.

        Definitions
        -----
        * :prf:ref:`Stress Matrix <def-stress-matrix>`

        """
        raise NotImplementedError()

    @doc_category("Infinitesimal rigidity")
    def trivial_inf_flexes(self) -> List[Matrix]:
        r"""
        Return a basis of the vector subspace of trivial infinitesimal flexes.

        Definitions
        -----------
        * :prf:ref:`Trivial infinitesimal flexes <def-trivial-inf-flex>`

        TODO
        ----
        more tests

        Examples
        --------
        >>> F = Framework.Complete([(0,0), (2,0), (0,2)])
        >>> F.trivial_inf_flexes()
        [Matrix([
        [1],
        [0],
        [1],
        [0],
        [1],
        [0]]), Matrix([
        [0],
        [1],
        [0],
        [1],
        [0],
        [1]]), Matrix([
        [ 0],
        [ 0],
        [ 0],
        [ 2],
        [-2],
        [ 0]])]
        """
        dim = self._dim
        translations = [
            Matrix.vstack(*[A for _ in self._graph.nodes])
            for A in Matrix.eye(dim).columnspace()
        ]
        basis_skew_symmetric = []
        for i in range(1, dim):
            for j in range(i):
                A = Matrix.zeros(dim)
                A[i, j] = 1
                A[j, i] = -1
                basis_skew_symmetric += [A]
        inf_rot = [
            Matrix.vstack(*[A * self._realization[v] for v in self._graph.nodes])
            for A in basis_skew_symmetric
        ]
        matrix_inf_flexes = Matrix.hstack(*(translations + inf_rot))
        return matrix_inf_flexes.transpose().echelon_form().transpose().columnspace()

    @doc_category("Infinitesimal rigidity")
    def nontrivial_inf_flexes(
        self,
    ) -> List[Matrix]:
        """
        Return non-trivial infinitesimal flexes.

        Definitions
        -----------
        :prf:ref:`Infinitesimal flex <def-inf-rigid-framework>`

        TODO
        ----
        tests, example

        Notes
        -----
        See :meth:`~Framework.trivial_inf_flexes`.
        """
        return self.inf_flexes(include_trivial=False)

    @doc_category("Infinitesimal rigidity")
    def inf_flexes(
        self,
        include_trivial: bool = False,
    ) -> List[Matrix]:
        r"""
        Return a basis of the space of infinitesimal flexes.

        Return a lift of a basis of the quotient of
        the vector space of infinitesimal flexes
        modulo trivial infinitesimal flexes, if ``include_trivial=False``.
        Return a basis of the vector space of infinitesimal flexes
        if ``include_trivial=True``.
        Else, return the entire kernel.

        TODO
        ----
        more tests

        Definitions
        -----------
        * :prf:ref:`Infinitesimal flex <def-inf-flex>`

        Parameters
        ----------
        include_trivial:
            Boolean that decides, whether the trivial motions should
            be included (``True``) or not (``False``)

        Examples
        --------
        >>> F = Framework.Complete([[0,0], [1,0], [1,1], [0,1]])
        >>> F.delete_edges([(0,2), (1,3)])
        >>> F.inf_flexes(include_trivial=False)
        [Matrix([
        [1],
        [0],
        [1],
        [0],
        [0],
        [0],
        [0],
        [0]])]
        """
        if include_trivial:
            return self.rigidity_matrix().nullspace()
        rigidity_matrix = self.rigidity_matrix()
        all_inf_flexes = rigidity_matrix.nullspace()
        trivial_inf_flexes = self.trivial_inf_flexes()
        s = len(trivial_inf_flexes)
        extend_basis_matrix = Matrix.hstack(*trivial_inf_flexes)
        tmp_matrix = Matrix.hstack(*trivial_inf_flexes)
        for v in all_inf_flexes:
            r = extend_basis_matrix.rank()
            tmp_matrix = Matrix.hstack(extend_basis_matrix, v)
            if not tmp_matrix.rank() == r:
                extend_basis_matrix = Matrix.hstack(extend_basis_matrix, v)
        basis = extend_basis_matrix.columnspace()
        return basis[s:]

    @doc_category("Waiting for implementation")
    def stresses(self) -> Any:
        r"""Return a basis of the space of stresses."""
        raise NotImplementedError()

    @doc_category("Infinitesimal rigidity")
    def rigidity_matrix_rank(self) -> int:
        """
        Compute the rank of the rigidity matrix.

        TODO
        ----
        example, tests
        """
        return self.rigidity_matrix().rank()

    @doc_category("Infinitesimal rigidity")
    def is_inf_rigid(self) -> bool:
        """
        Check whether the given framework is infinitesimally rigid.

        The check is based on :meth:`~Framework.rigidity_matrix_rank`.

        Definitions
        -----
        * :prf:ref:`Infinitesimal rigidity <def-inf-rigid-framework>`

        TODO
        ----
        example
        """
        if self._graph.number_of_nodes() <= self._dim + 1:
            return self.rigidity_matrix_rank() == binomial(
                self._graph.number_of_nodes(), 2
            )
        else:
            return (
                self.rigidity_matrix_rank()
                == self.dim() * self._graph.number_of_nodes()
                - binomial(self.dim() + 1, 2)
            )

    @doc_category("Infinitesimal rigidity")
    def is_inf_flexible(self) -> bool:
        """
        Check whether the given framework is infinitesimally flexible.

        See :meth:`~Framework.is_inf_rigid`
        """
        return not self.is_inf_rigid()

    @doc_category("Waiting for implementation")
    def is_inf_spanning(self) -> bool:
        raise NotImplementedError()

    @doc_category("Infinitesimal rigidity")
    def is_min_inf_rigid(self) -> bool:
        """
        Check whether a framework is minimally infinitesimally rigid.

        Definitions
        -----
        :prf:ref:`Minimal infinitesimal rigidity <def-min-rigid-framework>`

        Examples
        --------
        >>> F = Framework.Complete([[0,0], [1,0], [1,1], [0,1]])
        >>> F.is_min_inf_rigid()
        False
        >>> F.delete_edge((0,2))
        >>> F.is_min_inf_rigid()
        True
        """
        if not self.is_inf_rigid():
            return False
        for edge in self._graph.edge_list():
            self.delete_edge(edge)
            if self.is_inf_rigid():
                self.add_edge(edge)
                return False
            self.add_edge(edge)
        return True

    @doc_category("Waiting for implementation")
    def is_independent(self) -> bool:
        raise NotImplementedError()

    @doc_category("Waiting for implementation")
    def is_prestress_stable(self) -> bool:
        raise NotImplementedError()

    @doc_category("Infinitesimal rigidity")
    def is_redundantly_rigid(self) -> bool:
        """
        Check if the framework is infinitesimally redundantly rigid.

        Definitions
        -----------
        :prf:ref:`Redundant infinitesimal rigidity <def-redundantly-rigid-framework>`

        TODO
        ----
        tests

        Examples
        --------
        >>> F = Framework.Empty(dim=2)
        >>> F.add_vertices([(1,0), (1,1), (0,3), (-1,1)], ['a','b','c','d'])
        >>> F.add_edges([('a','b'), ('b','c'), ('c','d'), ('a','d'), ('a','c'), ('b','d')])
        >>> F.is_redundantly_rigid()
        True
        >>> F.delete_edge(('a','c'))
        >>> F.is_redundantly_rigid()
        False
        """  # noqa: E501
        for edge in self._graph.edge_list():
            self.delete_edge(edge)
            if not self.is_inf_rigid():
                self.add_edge(edge)
                return False
            self.add_edge(edge)
        return True

    @doc_category("Framework properties")
    def is_congruent_realization(
        self,
        other_realization: Dict[Vertex, Point],
        numerical: bool = False,
        tolerance: float = 10e-9,
    ) -> bool:
        """
        Return whether the given realization is congruent to self.

        Parameters
        ----------
        other_realization
            The realization for checking the congruence.
        numerical
            Whether the check is symbolic (default) or numerical.
        tolerance
            Used tolerance when checking numerically.
        """

        if set(self._graph.nodes) != set(other_realization.keys()):
            raise ValueError(
                "Not all vertices have a realization in the given dictionary."
            )

        for u, v in combinations(self._graph.nodes, 2):
            edge_vec = (self._realization[u]) - self._realization[v]
            dist_squared = (edge_vec.T * edge_vec)[0, 0]

            other_edge_vec = point_to_vector(other_realization[u]) - point_to_vector(
                other_realization[v]
            )
            otherdist_squared = (other_edge_vec.T * other_edge_vec)[0, 0]

            difference = sp.simplify(dist_squared - otherdist_squared)
            if not difference.is_zero:
                if not numerical:
                    return False
                elif numerical and sp.Abs(difference) > tolerance:
                    return False
        return True

    @doc_category("Framework properties")
    def is_congruent(
        self,
        other_framework: Framework,
        numerical: bool = False,
        tolerance: float = 10e-9,
    ) -> bool:
        """
        Return whether the given framework is congruent to self.

        Parameters
        ----------
        other_framework
            The framework for checking the congruence.
        numerical
            Whether the check is symbolic (default) or numerical.
        tolerance
            Used tolerance when checking numerically.
        """

        if not nx.utils.graphs_equal(self._graph, other_framework._graph):
            raise ValueError("Underlying graphs are not same.")

        return self.is_congruent_realization(
            other_framework._realization, numerical, tolerance
        )

    @doc_category("Framework properties")
    def is_equivalent_realization(
        self,
        other_realization: Dict[Vertex, Point],
        numerical: bool = False,
        tolerance: float = 10e-9,
    ) -> bool:
        """
        Return whether the given realization is equivalent to self.

        Parameters
        ----------
        other_realization
            The realization for checking the equivalence.
        numerical
            Whether the check is symbolic (default) or numerical.
        tolerance
            Used tolerance when checking numerically.
        """

        if set(self._graph.nodes) != set(other_realization.keys()):
            raise ValueError(
                "Not all vertices have a realization in the given dictionary."
            )

        for u, v in self._graph.edges:

            edge_vec = self._realization[u] - self._realization[v]
            dist_squared = (edge_vec.T * edge_vec)[0, 0]

            other_edge_vec = point_to_vector(other_realization[u]) - point_to_vector(
                other_realization[v]
            )
            otherdist_squared = (other_edge_vec.T * other_edge_vec)[0, 0]

            difference = sp.simplify(otherdist_squared - dist_squared)
            if not difference.is_zero:
                if not numerical:
                    return False
                elif numerical and sp.Abs(difference) > tolerance:
                    return False
        return True

    @doc_category("Framework properties")
    def is_equivalent(
        self,
        other_framework: Framework,
        numerical: bool = False,
        tolerance: float = 10e-9,
    ) -> bool:
        """
        Return whether the given framework is equivalent to self.

        Parameters
        ----------
        other_framework
            The framework for checking the equivalence.
        numerical
            Whether the check is symbolic (default) or numerical.
        tolerance
            Used tolerance when checking numerically.
        """

        if not nx.utils.graphs_equal(self._graph, other_framework._graph):
            raise ValueError("Underlying graphs are not same.")
        return self.is_equivalent_realization(
            other_framework._realization, numerical, tolerance
        )

    @doc_category("Framework manipulation")
    def translate(self, vector: Point, inplace: bool = True) -> Union[None, Framework]:
        """
        Translate the framework.

        Parameters
        ----------
        vector
            Translation vector
        inplace
            If True (default), then this framework is translated.
            Otherwise, a new translated framework is returned.
        """

        vector = point_to_vector(vector)

        if inplace:
            if vector.shape[0] != self.dim():
                raise ValueError(
                    "The dimension of the vector has to be the same as of the framework."
                )

            for v in self._realization.keys():
                self._realization[v] += vector
            return

        new_framework = deepcopy(self)
        new_framework.translate(vector, True)
        return new_framework

    @doc_category("Framework manipulation")
    def rotate2D(self, angle: float, inplace: bool = True) -> Union[None, Framework]:
        """
        Rotate the planar framework counter clockwise.

        Parameters
        ----------
        angle
            Rotation angle
        inplace
            If True (default), then this framework is rotated.
            Otherwise, a new rotated framework is returned.
        """

        if self.dim() != 2:
            raise ValueError("This realization is not in dimension 2!")

        rotation_matrix = Matrix(
            [[sp.cos(angle), -sp.sin(angle)], [sp.sin(angle), sp.cos(angle)]]
        )

        if inplace:
            for v, pos in self._realization.items():
                self._realization[v] = rotation_matrix * pos
            return

        new_framework = deepcopy(self)
        new_framework.rotate2D(angle, True)
        return new_framework


Framework.__doc__ = Framework.__doc__.replace(
    "METHODS",
    generate_category_tables(
        Framework,
        1,
        [
            "Attribute getters",
            "Framework properties",
            "Class methods",
            "Framework manipulation",
            "Infinitesimal rigidity",
            "Other",
            "Waiting for implementation",
        ],
        include_all=False,
    ),
)
