"""

Module for the functionality concerning frameworks.

.. currentmodule:: pyrigi.framework

Classes:

.. autosummary::

    Framework

"""

from __future__ import annotations
from typing import List, Any, Dict

from copy import deepcopy
from random import randrange

import networkx as nx
from sympy import Matrix, flatten

from pyrigi.data_type import Vertex, Edge, Point
from pyrigi.graph import Graph
from pyrigi.misc import doc_category, generate_category_tables


class Framework(object):
    r"""
    This class provides the functionality for frameworks.


    Definitions
    -----------
     * :prf:ref:`Framework <def-framework>`
     * :prf:ref:`Realization <def-realization>`


    Parameters
    ----------
    graph
    realization:
        A dictionary mapping the vertices of the graph to points in $\RR^d$.
        The dimension `d` is retrieved from the points in realization.
        If `graph` is empty, and hence also the `realization`,
        the dimension is set to 0 (:meth:`Framework.Empty`
        can be used to construct an empty framework with different dimension).

    Examples
    --------
    >>> F = Framework(Graph([[0,1]]), {0:[1,2], 1:[0,5]})
    >>> F
    Framework in 2-dimensional space consisting of:
    Graph with vertices [0, 1] and edges [[0, 1]]
    Realization {0:(1, 2), 1:(0, 5)}

    METHODS

    Notes
    -----
    Internally, the realization is represented as a dictionary of
    matrices ("vectors"). However, there is a method available for
    transforming the realization to a more human-friendly format
    (see :meth:`~Framework.get_realization_list`)
    """

    def __init__(self, graph: Graph, realization: Dict[Vertex, Point]) -> None:
        if not isinstance(graph, Graph):
            raise TypeError("The graph has to be an instance of class Graph")
        if not len(realization.keys()) == graph.number_of_nodes():
            raise KeyError(
                "The length of realization has to be equal to "
                "the number of vertices of graph"
            )

        if realization:
            self._dim = len(list(realization.values())[0])
        else:
            self._dim = 0

        for v in graph.nodes:
            if v not in realization:
                raise KeyError(f"Vertex {v} is not contained in the realization")
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
        If no vertex is provided (`None`), then the smallest,
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

        Notes
        -----
        For each vertex that has to be added, :meth:`add_vertex` is called.

        Examples
        --------
        >>> F = Framework.Empty(dim=2)
        >>> F.add_vertices([(1.5,2), (3,1)], ['a',0])
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices ['a', 0] and edges []
        Realization {a:(1.50000000000000, 2), 0:(3, 1)}
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

        Parameters
        ----------
        edge:
            The edge is a tuple of vertices. It can either be passed as a tuple `(i,j)`
            or a list `[i,j]`.

        Notes
        -----
        This method only alters the graph attribute.
        """
        if not (len(edge)) == 2:
            raise TypeError(f"Edge {edge} does not have the correct length!")
        if not (edge[0] in self._graph.nodes and edge[1] in self._graph.nodes):
            raise ValueError(
                f"The adjacent vertices of {edge} are not contained in the graph!"
            )
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
    def underlying_graph(self) -> Graph:
        """
        Return a copy of the framework's underlying graph.
        """
        return deepcopy(self._graph)

    @doc_category("Attribute getters")
    def graph(self) -> Graph:
        """Alias for :meth:`~Framework.underlying_graph`"""
        return self.underlying_graph()

    @doc_category("Plotting")
    def draw_framework(self) -> None:
        """
        Plot the framework.

        Notes
        -----
        Use a networkx internal routine to plot the framework."""
        nx.draw(self._graph, pos=self.get_realization_list())

    @classmethod
    @doc_category("Class methods")
    def from_points(cls, points: List[Point]) -> None:
        """
        Generate a framework from a list of points.

        Notes
        -----
        The list of vertices of the underlying graph
        is taken to be `[0,...,len(points)]`.
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
        G = Graph()
        G.add_nodes_from(vertices)
        return Framework(graph=G, realization=realization)

    @classmethod
    @doc_category("Class methods")
    def from_graph(cls, graph: Graph, dim: int = 2) -> None:
        """
        Return a framework with random realization.

        Examples
        --------
        >>> F = Framework.from_graph(Graph([(0,1), (1,2), (0,2)]))
        >>> print(F) # doctest: +SKIP
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        Realization {0:(122, 57), 1:(27, 144), 2:(50, 98)}
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )

        N = 10 * graph.number_of_nodes() ** 2 * dim
        realization = {
            vertex: [randrange(1, N) for _ in range(dim)] for vertex in graph.nodes
        }

        return Framework(graph=graph, realization=realization)

    @classmethod
    @doc_category("Class methods")
    def Empty(cls, dim: int = 2) -> None:
        """
        Generate an empty framework.

        Parameters
        ----------
        dim:
            a natural number that determines the dimension
            in which the framework is realized
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
    def Complete(cls, points: List[Point]) -> None:
        """
        Generate a framework on the complete graph from a given list of points.

        Parameters
        ----------
        dim:
            a natural number that determines the dimension
            in which the framework is realized

        Notes
        -----
        The vertices of the underlying graph are taken
        to be the list `[0,...,len(points)]`.

        Examples
        --------
        >>> F = Framework.Complete([(1,),(2,),(3,),(4,)]); F
        Framework in 1-dimensional space consisting of:
        Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        Realization {0:(1,), 1:(2,), 2:(3,), 3:(4,)}
        """  # noqa: E501
        if not points:
            raise ValueError("The list of points cannot be empty.")

        Kn = Graph.Complete(len(points))
        realization = {
            (Kn.vertex_list())[i]: Matrix(points[i]) for i in range(len(points))
        }
        return Framework(graph=Kn, realization=realization)

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
    def get_realization_list(self) -> List[Point]:
        """
        Return the realization as a list of Point.

        Notes
        -----
        The format returned by this method can be read by networkx.

        Examples
        --------
        >>> F = Framework.Complete([(0,0), (1,0), (1,1)])
        >>> F.get_realization_list()
        {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0)}
        """
        return {
            vertex: tuple([float(point) for point in self._realization[vertex]])
            for vertex in self._graph.nodes
        }

    @doc_category("Attribute getters")
    def get_realization(self) -> Dict[Vertex, Point]:
        """
        Return a copy of the framework's realization.
        """
        return deepcopy(self._realization)

    @doc_category("Attribute getters")
    def realization(self) -> List[Point]:
        """Alias for :meth:`~Framework.get_realization`"""
        return self.get_realization()

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
    def change_vertex_coordinates(self, vertex: Vertex, point: Point) -> None:
        """
        Change the coordinates of a single given vertex.

        Examples
        --------
        >>> F = Framework.from_points([(0,0)])
        >>> F.change_vertex_coordinates(0, (6,2))
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
    def set_vertex_position(self, vertex: Vertex, point: Point) -> None:
        """Alias for :meth:`~Framework.change_vertex_coordinates`"""
        self.change_vertex_coordinates(vertex, point)

    @doc_category("Framework manipulation")
    def change_vertex_coordinates_list(
        self, vertices: List[Vertex], points: List[Point]
    ) -> None:
        """
        Change the coordinates of a given list of vertices.

        Notes
        -----
        It is necessary that both lists have the same length.
        No vertex from `vertices` can be contained multiple times.
        For an explanation of `vertices` and `points`,
        see :meth:`~Framework.add_vertices`.
        We apply the method :meth:`~Framework.change_vertex_coordinates`
        to `vertices` and `points`.
        """
        if list(set(vertices)).sort() != list(vertices).sort():
            raise ValueError("Mulitple Vertices with the same name were found!")
        if not len(vertices) == len(points):
            raise IndexError(
                "The list of vertices does not have the same length as the list of points"
            )
        for i in range(len(vertices)):
            self.change_vertex_coordinates(vertices[i], points[i])

    @doc_category("Framework manipulation")
    def set_vertex_positions(self, vertices: List[Vertex], points: List[Point]) -> None:
        """Alias for :meth:`~Framework.change_vertex_coordinates_list`"""
        self.change_vertex_coordinates_list(vertices, points)

    @doc_category("Framework manipulation")
    def change_realization(self, subset_of_realization: Dict[Vertex, Point]):
        """
        Change the coordinates of vertices given by a dictionary.
        """
        self.change_vertex_coordinates_list(
            subset_of_realization.keys(), subset_of_realization.values()
        )

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
            A Boolean indicating, whether the edges are assumed to be ordered (`True`),
            or whether they should be internally sorted (`False`).

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

        # `delta` is responsible for distinguishing the edges (i,j) and (j,i)
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

        Definitions
        -----------
        * :prf:ref:`Rigidity Matrix <def-rigidity-matrix>`

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
                for i in range(0, upper):
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
        Return a basis of the vector subspace of trivial infinitesimal flexes of the framework.

        Definitions
        -----------
        * :prf:ref:`Trivial infinitesimal flexes <def-trivial-inf-flexes>`

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
        """  # noqa: E501
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

        Definitions
        -----------
        * :prf:ref:`Infinitesimal flex <def-inf-flex>`

        Notes
        -----
        Return a lift of a basis of the quotient of the vector space of infinitesimal flexes
        modulo trivial infinitesimal flexes, if `include_trivial=False`.
        Return a basis of the vector space of infinitesimal flexes if `include_trivial=True`.

        Else, return the entire kernel.

        Parameters
        ----------
        include_trivial:
            Boolean that decides, whether the trivial motions should
            be included (`True`) or not (`False`)

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
        """  # noqa: E501
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
        """
        return (
            self.graph().number_of_nodes() <= 1
            or self.rigidity_matrix_rank()
            == self.dim() * self.graph().number_of_nodes()
            - (self.dim()) * (self.dim() + 1) // 2
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
        for edge in self.graph().edges:
            F = deepcopy(self)
            F.delete_edge(edge)
            if F.is_inf_rigid():
                return False
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

        >>> F = Framework.Empty(dim=2)
        >>> F.add_vertices([(1,0), (1,1), (0,3), (-1,1)], ['a','b','c','d'])
        >>> F.add_edges([('a','b'), ('b','c'), ('c','d'), ('a','d'), ('a','c'), ('b','d')])
        >>> F.is_redundantly_rigid()
        True
        >>> F.delete_edge(('a','c'))
        >>> F.is_redundantly_rigid()
        False
        """  # noqa: E501
        for edge in self._graph.edges:
            F = deepcopy(self)
            F.delete_edge(edge)
            if not F.is_inf_rigid():
                return False
        return True

    @doc_category("Waiting for implementation")
    def is_congruent(self, framework_) -> bool:
        raise NotImplementedError()

    @doc_category("Waiting for implementation")
    def is_equivalent(self, framework_) -> bool:
        raise NotImplementedError()


Framework.__doc__ = Framework.__doc__.replace(
    "METHODS",
    generate_category_tables(
        Framework,
        1,
        [
            "Attribute getters",
            "Class methods",
            "Framework manipulation",
            "Infinitesimal rigidity",
            "Plotting",
            "Waiting for implementation",
        ],
        include_all=False,
    ),
)
