"""
Base module for the functionality concerning frameworks.
"""

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import numpy as np
import sympy as sp
from sympy import Matrix

import pyrigi._utils._input_check as _input_check
from pyrigi.graph import _general as graph_general
import pyrigi.graph._utils._input_check as _graph_input_check
from pyrigi._utils._doc import doc_category, generate_category_tables
from pyrigi.data_type import (
    Edge,
    Number,
    Point,
    Sequence,
    Vertex,
)
from pyrigi.graph import Graph


class FrameworkBase(object):
    r"""
    This class is a base class for :class:`.Framework`.

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
        The dimension $d$ is retrieved from the points in realization.
        If ``graph`` is empty, and hence also the ``realization``,
        the dimension is set to 0 (:meth:`.Empty`
        can be used to construct an empty framework with different dimension).

    Examples
    --------
    >>> F = FrameworkBase(Graph([[0,1]]), {0:[1,2], 1:[0,5]})
    >>> print(F)
    FrameworkBase in 2-dimensional space consisting of:
    Graph with vertices [0, 1] and edges [[0, 1]]
    Realization {0:(1, 2), 1:(0, 5)}

    Notice that the realization of a vertex can be accessed using ``[ ]``:

    >>> F[0]
    Matrix([
    [1],
    [2]])

    This the base class for :class:`.Framework`.

    >>> from pyrigi import Framework
    >>> issubclass(Framework, FrameworkBase)
    True

    METHODS

    Notes
    -----
    Internally, the realization is represented as ``dict[Vertex,Matrix]``.
    However, :meth:`~Framework.realization` can also return ``dict[Vertex,Point]``.
    """

    def __init__(self, graph: Graph, realization: dict[Vertex, Point]) -> None:
        if isinstance(graph, nx.Graph):
            graph = Graph(graph)

        if not isinstance(graph, Graph):
            raise TypeError("The graph has to be an instance of class Graph.")
        _graph_input_check.no_loop(graph)
        if not len(realization.keys()) == graph.number_of_nodes():
            raise KeyError(
                "The length of realization has to be equal to "
                "the number of vertices of graph."
            )

        if realization:
            self._dim = len(list(realization.values())[0])
        else:
            self._dim = 0

        self._graph = deepcopy(graph)
        self._realization = {}
        self.set_realization(realization)

    def __str__(self) -> str:
        """Return the string representation."""
        return (
            self.__class__.__name__
            + f" in {self.dim}-dimensional space consisting of:\n{self._graph}\n"
            + "Realization {"
            + ", ".join(
                [
                    f"{v}:{tuple(self._realization[v])}"
                    for v in graph_general.vertex_list(self._graph)
                ]
            )
            + "}"
        )

    def __repr__(self) -> str:
        """Return a representation of the framework."""
        str_realization = {
            v: [str(p) for p in pos]
            for v, pos in self.realization(as_points=True).items()
        }
        return f"{self.__class__.__name__}({repr(self.graph)}, {str_realization})"

    def __getitem__(self, vertex: Vertex) -> Matrix:
        """
        Return the coordinates of a given vertex in the realization.

        Parameters
        ----------
        vertex

        Examples
        --------
        >>> F = Framework(Graph([[0,1]]), {0:[1,2], 1:[0,5]})
        >>> F[0]
        Matrix([
        [1],
        [2]])
        """
        return self._realization[vertex]

    @property
    def dim(self) -> int:
        """Return the dimension of the framework."""
        return self._dim

    @property
    def graph(self) -> Graph:
        """
        Return a copy of the underlying graph.

        Examples
        ----
        >>> F = Framework.Random(Graph([(0,1), (1,2), (0,2)]))
        >>> print(F.graph)
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        """
        return deepcopy(self._graph)

    @doc_category("Framework manipulation")
    def add_vertex(self, point: Point, vertex: Vertex = None) -> None:
        """
        Add a vertex to the framework with the corresponding coordinates.

        If no vertex is provided (``None``),
        then an integer is chosen instead.

        Parameters
        ----------
        point:
            The realization of the new vertex.
        vertex:
            The label of the new vertex.

        Examples
        --------
        >>> F = Framework.Empty(dim=2)
        >>> F.add_vertex((1.5,2), 'a')
        >>> F.add_vertex((3,1))
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices ['a', 1] and edges []
        Realization {a:(1.50000000000000, 2), 1:(3, 1)}
        """
        if vertex is None:
            candidate = self._graph.number_of_nodes()
            while self._graph.has_node(candidate):
                candidate += 1
            vertex = candidate

        if self._graph.has_node(vertex):
            raise KeyError(f"Vertex {vertex} is already a vertex of the graph!")

        self._realization[vertex] = Matrix(point)
        self._graph.add_node(vertex)

    @doc_category("Framework manipulation")
    def add_vertices(
        self, points: Sequence[Point], vertices: Sequence[Vertex] = None
    ) -> None:
        r"""
        Add a list of vertices to the framework.

        Parameters
        ----------
        points:
            List of points consisting of coordinates in $\RR^d$. It is checked
            that all points lie in the same ambient space.
        vertices:
            List of vertices. If the list of vertices is empty, we generate
            vertices with the method :meth:`add_vertex`.
            Otherwise, the list of vertices needs to have the same length as the
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
        if vertices and not len(points) == len(vertices):
            raise IndexError("The vertex list does not have the correct length!")
        if not vertices:
            for point in points:
                self.add_vertex(point)
        else:
            for point, v in zip(points, vertices):
                self.add_vertex(point, v)

    @doc_category("Framework manipulation")
    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge to the framework.

        Parameters
        ----------
        edge:

        Notes
        -----
        This method only alters the graph attribute.
        """
        _graph_input_check.edge_format(self._graph, edge, loopfree=True)
        self._graph.add_edge(*edge)

    @doc_category("Framework manipulation")
    def add_edges(self, edges: Sequence[Edge]) -> None:
        """
        Add a list of edges to the framework.

        Parameters
        ----------
        edges:

        Notes
        -----
        For each edge that has to be added, :meth:`add_edge` is called.
        """
        for edge in edges:
            self.add_edge(edge)

    @doc_category("Framework manipulation")
    def delete_vertex(self, vertex: Vertex) -> None:
        """
        Delete a vertex from the framework.

        Parameters
        ----------
        vertex
        """
        self._graph.delete_vertex(vertex)
        del self._realization[vertex]

    @doc_category("Framework manipulation")
    def delete_vertices(self, vertices: Sequence[Vertex]) -> None:
        """
        Delete a list of vertices from the framework.

        Parameters
        ----------
        vertices
        """
        for vertex in vertices:
            self.delete_vertex(vertex)

    @doc_category("Framework manipulation")
    def delete_edge(self, edge: Edge) -> None:
        """
        Delete an edge from the framework.

        Parameters
        ----------
        edge
        """
        self._graph.delete_edge(edge)

    @doc_category("Framework manipulation")
    def delete_edges(self, edges: Sequence[Edge]) -> None:
        """
        Delete a list of edges from the framework.

        Parameters
        ----------
        edges
        """
        self._graph.delete_edges(edges)

    @doc_category("Attribute getters")
    def realization(
        self, as_points: bool = False, numerical: bool = False
    ) -> dict[Vertex, Point] | dict[Vertex, Matrix]:
        """
        Return a copy of the realization.

        Parameters
        ----------
        as_points:
            If ``True``, then the vertex positions type is
            :obj:`pyrigi.data_type.Point`,
            otherwise :obj:`Matrix <~sympy.matrices.dense.MutableDenseMatrix>` (default).
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
        """
        if not numerical:
            if not as_points:
                return deepcopy(self._realization)
            return {v: list(pos) for v, pos in self._realization.items()}
        else:
            if not as_points:
                return {
                    v: Matrix([float(coord) for coord in pos])
                    for v, pos in self._realization.items()
                }
            return {
                v: [float(coord) for coord in pos]
                for v, pos in self._realization.items()
            }

    @doc_category("Framework manipulation")
    def set_realization(self, realization: dict[Vertex, Point]) -> None:
        r"""
        Change the realization of the framework.

        Definitions
        -----------
        :prf:ref:`Realization <def-realization>`

        Parameters
        ----------
        realization:
            A realization of the underlying graph of the framework.
            It must contain all vertices from the underlying graph.
            Furthermore, all points in the realization need
            to be contained in $\RR^d$ for $d$ being
            the current dimension of the framework.

        Examples
        --------
        >>> F = Framework.Complete([(0,0), (1,0), (1,1)])
        >>> F.set_realization(
        ...     {vertex: (vertex, vertex + 1) for vertex in F.graph.vertex_list()}
        ... )
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        Realization {0:(0, 1), 1:(1, 2), 2:(2, 3)}
        """
        if not len(realization) == self._graph.number_of_nodes():
            raise IndexError(
                "The realization does not contain the correct amount of vertices!"
            )
        for v in self._graph.nodes:
            self._input_check_vertex_key(v, realization)
            self._input_check_point_dimension(realization[v])

        self._realization = {v: Matrix(pos) for v, pos in realization.items()}

    @doc_category("Framework manipulation")
    def set_vertex_pos(self, vertex: Vertex, point: Point) -> None:
        """
        Change the coordinates of a single given vertex.

        Parameters
        ----------
        vertex:
            A vertex whose position is changed.
        point:
            A new position of the ``vertex``.

        Examples
        --------
        >>> F = Framework.from_points([(0,0)])
        >>> F.set_vertex_pos(0, (6,2))
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0] and edges []
        Realization {0:(6, 2)}
        """
        self._input_check_vertex_key(vertex)
        self._input_check_point_dimension(point)

        self._realization[vertex] = Matrix(point)

    @doc_category("Framework manipulation")
    def set_vertex_positions_from_lists(
        self, vertices: Sequence[Vertex], points: Sequence[Point]
    ) -> None:
        """
        Change the coordinates of a given list of vertices.

        It is necessary that both lists have the same length.
        No vertex from ``vertices`` can be contained multiple times.
        We apply the method :meth:`~Framework.set_vertex_positions`
        to the corresponding pairs of ``vertices`` and ``points``.

        Parameters
        ----------
        vertices
        points

        Examples
        --------
        >>> F = Framework.Complete([(0,0),(0,0),(1,0),(1,0)])
        >>> F.realization(as_points=True)
        {0: [0, 0], 1: [0, 0], 2: [1, 0], 3: [1, 0]}
        >>> F.set_vertex_positions_from_lists([1,3], [(0,1),(1,1)])
        >>> F.realization(as_points=True)
        {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
        """
        if len(list(set(vertices))) != len(list(vertices)):
            raise ValueError("Multiple Vertices with the same name were found!")
        if not len(vertices) == len(points):
            raise IndexError(
                "The list of vertices does not have the same length "
                "as the list of points!"
            )
        self.set_vertex_positions({v: pos for v, pos in zip(vertices, points)})

    @doc_category("Framework manipulation")
    def set_vertex_positions(self, subset_of_realization: dict[Vertex, Point]) -> None:
        """
        Change the coordinates of vertices given by a dictionary.

        Parameters
        ----------
        subset_of_realization

        Examples
        --------
        >>> F = Framework.Complete([(0,0),(0,0),(1,0),(1,0)])
        >>> F.realization(as_points=True)
        {0: [0, 0], 1: [0, 0], 2: [1, 0], 3: [1, 0]}
        >>> F.set_vertex_positions({1:(0,1),3:(1,1)})
        >>> F.realization(as_points=True)
        {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
        """
        for v, pos in subset_of_realization.items():
            self.set_vertex_pos(v, pos)

    @doc_category("Other")
    def edge_lengths(self, numerical: bool = False) -> dict[Edge, Number]:
        """
        Return the dictionary of the edge lengths.

        Parameters
        -------
        numerical:
            If ``True``, numerical positions are used for the computation of the edge lengths.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (0,3)])
        >>> F = Framework(G, {0:[0,0], 1:[1,0], 2:[1,'1/2 * sqrt(5)'], 3:['1/2','4/3']})
        >>> F.edge_lengths(numerical=False)
        {(0, 1): 1, (0, 3): sqrt(73)/6, (1, 2): sqrt(5)/2, (2, 3): sqrt((-4/3 + sqrt(5)/2)**2 + 1/4)}
        >>> F.edge_lengths(numerical=True)
        {(0, 1): 1.0, (0, 3): 1.4240006242195884, (1, 2): 1.118033988749895, (2, 3): 0.5443838790578374}
        """  # noqa: E501
        if numerical:
            points = self.realization(as_points=True, numerical=True)
            return {
                tuple(e): float(
                    np.linalg.norm(np.array(points[e[0]]) - np.array(points[e[1]]))
                )
                for e in self._graph.edges
            }
        else:
            points = self.realization(as_points=True)
            return {
                tuple(e): sp.sqrt(
                    sum([(x - y) ** 2 for x, y in zip(points[e[0]], points[e[1]])])
                )
                for e in self._graph.edges
            }

    def _input_check_underlying_graphs(self, other_framework) -> None:
        """
        Check whether the underlying graphs of two frameworks are the same and
        raise an error otherwise.
        """
        if self._graph != other_framework._graph:
            raise ValueError("The underlying graphs are not same!")

    def _input_check_vertex_key(
        self, vertex: Vertex, realization: dict[Vertex, Point] = None
    ) -> None:
        """
        Check whether a vertex appears as key in a realization and
        raise an error otherwise.

        Parameters
        ----------
        vertex:
            The vertex to check.
        realization:
            The realization to check.
        """
        if realization is None:
            realization = self._realization
        if vertex not in realization:
            raise KeyError("Vertex {vertex} is not a key of the given realization!")

    def _input_check_point_dimension(self, point: Point) -> None:
        """
        Check whether a point has the right dimension and
        raise an error otherwise.

        Parameters
        ----------
        point:
        """
        if not len(point) == self.dim:
            raise ValueError(
                f"The point {point} does not have the dimension {self.dim}!"
            )

    @classmethod
    @doc_category("Class methods")
    def Empty(cls, dim: int = 2) -> FrameworkBase:
        """
        Generate an empty framework.

        Parameters
        ----------
        dim:
            A natural number that determines the dimension
            in which the framework is realized.

        Examples
        ----
        >>> F = Framework.Empty(dim=1); print(F)
        Framework in 1-dimensional space consisting of:
        Graph with vertices [] and edges []
        Realization {}
        """
        _input_check.dimension(dim)
        F = cls(graph=Graph(), realization={})
        F._dim = dim
        return F


FrameworkBase.__doc__ = FrameworkBase.__doc__.replace(
    "METHODS",
    generate_category_tables(
        FrameworkBase,
        1,
        [
            "Attribute getters",
            "Framework properties",
            "Class methods",
            "Framework manipulation",
            "Infinitesimal rigidity",
            "Plotting",
            "Other",
            "Waiting for implementation",
        ],
        include_all=False,
    ),
)
