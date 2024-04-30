"""

Module for the functionality concerning frameworks.

.. currentmodule:: pyrigi.framework

Classes:

.. autosummary::

    Framework

"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from random import randrange

import networkx as nx
from sympy import Matrix, flatten

from pyrigi.data_type import Vertex, Edge, Point, List, Any, Dict
from pyrigi.graph import Graph


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
    dim:
        The dimension is usually initialized by the realization. If
        the realization is empty, the dimension is 0 by default.

    Notes
    -----
    Internally, the realization is represented as a dictionary of
    matrices ("vectors"). However, there is a method available for
    transforming the realization to a more human-friendly format
    (see :meth:`~Framework.get_realization_list`)

    Examples
    -----
    >>> F = Framework(Graph([[0,1]]), {0:[1,2], 1:[0,5]})
    >>> print(F)
    Graph:          Vertices: [0, 1],       Edges: [(0, 1)]
    Realization:    {0: (1.0, 2.0), 1: (0.0, 5.0)}
    dim:            2
    """

    def __init__(self,
                 graph: Graph = Graph(),
                 realization: Dict[Vertex, Point] = {},
                 dim: int = 2) -> None:
        if not isinstance(graph, Graph):
            raise TypeError("The graph has to be an instance of class Graph")
        if not len(realization.keys()) == len(graph.vertices()):
            raise KeyError(
                "The length of realization has to be equal to the number of vertices of graph")
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")

        if len(realization.values()) == 0:
            dimension = dim
        else:
            dimension = len(list(realization.values())[0])

        for v in graph.vertices():
            if v not in realization:
                raise KeyError(
                    f"Vertex {v} is not contained in the realization")
            if not len(realization[v]) == dimension:
                raise ValueError(
                    f"The point {realization[v]} in the realization that vertex {v} corresponds to does not have the right dimension")

        self._realization = {v: Matrix(realization[v])
                             for v in graph.vertices()}
        self._graph = deepcopy(graph)
        self._dim = dimension

    def __str__(self) -> str:
        """
        Method to display the data inside the Framework. This overrides the `print` method. 
        
        Notes
        -----
        We try to order the vertices in the realization for the display. If this fails for
        whatever reason, the internal order is used instead.
        """
        try:
            realization_str = str({key: self.get_realization_list()[
                key] for key in sorted(self.get_realization_list())})
        except:
            realization_str = str({key: self.get_realization_list()[
                key] for key in self._graph.vertices()})
        return 'Graph:\t\t' + str(self._graph) + '\n' + 'Realization:\t' + realization_str + '\n' + 'dim:\t\t' + str(self.dim())

    def dim(self) -> int:
        """Return the dimension of the framework."""
        return self._dim

    def dimension(self) -> int:
        """Alias for :meth:`~Framework.dim`"""
        return self.dim()

    def add_vertex(self, point: Point, vertex: Vertex = None) -> None:
        """
        This method adds a vertex to the Framework with the cooresponding coordinates. 
        If no vertex is provided (`None`), then the smallest, free integer is chosen instead.

        Examples
        -----
        >>> F = Framework.Empty(dim=2)
        >>> F.add_vertex((1.5,2), 'a')
        >>> F.add_vertex((3,1))
        >>> print(F)
        Graph:          Vertices: ['a', 1],     Edges: []
        Realization:    {'a': (1.5, 2.0), 1: (3.0, 1.0)}
        dim:            2
        """
        if vertex is None:
            candidate = len(self._graph.vertices())
            while candidate in self._graph.vertices():
                candidate += 1
            vertex = candidate

        if vertex in self._graph.vertices():
            raise KeyError(f"Vertex {vertex} is already a vertex of the graph!")

        self._realization[vertex] = Matrix(point)
        self._graph.add_node(vertex)

    def add_vertices(self,
                     points: List[Point],
                     vertices: List[Vertex] = []) -> None:
        """
        In this method, a list of vertices is added. For each, `add_vertex` is called.

        Parameters
        -----
        points:
            List of points consisting of coordinates in $\RR^d$. It is checked 
            that all points lie in the same ambient space. 
        vertices:
            List of vertices. If the list of vertices is empty, we generate a 
            vertex that is not yet taken with the method :meth:`add_vertex`.
            Else, the list of vertices needs to have the same length as the
            list of points.

        Examples
        -----
        >>> F = Framework.Empty(dim=2)
        >>> F.add_vertices([(1.5,2), (3,1)], ['a',0])
        >>> print(F)
        Graph:          Vertices: ['a', 1],     Edges: []
        Realization:    {'a': (1.5, 2.0), 1: (3.0, 1.0)}
        dim:            2
        """
        if not (len(points) == len(vertices) or not vertices):
            raise IndexError(
                "The vertex list does not have the correct length!")
        if not vertices:
            for point in points:
                self.add_vertex(point)
        else:
            for p, v in zip(points, vertices):
                self.add_vertex(p, v)

    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge to the framework. This method only alters the graph attribute.
        
        Parameters
        -----
        edge:
            The edge is a tuple of vertices. It can either be passes as a tuple `(i,j)`
            or a list `[i,j]`.
        """
        if not (len(edge)) == 2:
            raise TypeError(f"Edge {edge} does not have the correct length!")
        if not (edge[0] in self._graph.nodes and edge[1] in self._graph.nodes):
            raise ValueError(
                f"The adjacent vertices of {edge} are not contained in the graph!")
        self._graph.add_edge(*(edge))

    def add_edges(self, edges: List[Edge]) -> None:
        """
        Call :meth:`add_edge` for each edge from a list of edges. 
        """
        for edge in edges:
            self.add_edge(edge)

    def underlying_graph(self) -> Graph:
        """        
        A deep copy of the framework's underlying graph is returned.
        Hence, modifying it does not affect the original framework.
        """
        return deepcopy(self._graph)

    def graph(self) -> Graph:
        """Alias for :meth:`~Framework.underlying_graph`"""
        return self.underlying_graph()

    def draw_framework(self) -> None:
        """Use the networkx-internal routine to plot the framework."""
        nx.draw(self._graph, pos=self.get_realization_list())

    @classmethod
    def from_points(cls, points: List[Point]) -> None:
        """
        Generate a framework from a list of points.

        Notes
        -----
        Since no vertices are provided, we generate the list `[0,...,len(points)]` instead

        Examples
        -----
        >>> F = Framework.from_points([(1,2), (2,3)])
        >>> print(F)
        Graph:          Vertices: [0, 1],       Edges: []
        Realization:    {0: (1.0, 2.0), 1: (2.0, 3.0)}
        dim:            2
        """
        vertices = range(len(points))
        realization = {v: points[v] for v in vertices}
        G = Graph()
        G.add_nodes_from(vertices)
        return Framework(graph=G, realization=realization)

    @classmethod
    def from_graph(cls, graph: Graph, dim: int) -> None:
        """
        Given a graph and a dimension, we create a random realization to create a framework.

        Examples
        -----
        >>> F = Framework.from_graph(Graph([(0,1), (1,2), (0,2)]), dim=2)
        >>> print(F)
        Graph:          Vertices: [0, 1, 2],    Edges: [(0, 1), (0, 2), (1, 2)]
        Realization:    {0: (65.0, 13.0), 1: (110.0, 64.0), 2: (54.0, 80.0)}
        dim:            2
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")

        N = 10 * len(graph.vertices())**2 * dim
        realization = {
                vertex: [randrange(1,N) for _ in range(dim)] 
            for vertex in graph.vertices()}

        return Framework(graph=graph, realization=realization)

    @classmethod
    def Empty(cls, dim: int) -> None:
        """
        Generate an empty framework.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")
        return Framework(graph=Graph(), realization={}, dim=dim)

    @classmethod
    def Complete(cls, points: List[Point] = [], dim: int = 2) -> None:
        """
        Generate a framework on the complete graph from a given list of points.

        Notes
        -----
        Since no vertices are provided, we generate the list `[0,...,len(points)]` instead.

        Examples
        -----
        >>> F = Framework.Complete([(1,),(2,),(3,),(4,)], dim=1)
        >>> print(F)
        Graph:          Vertices: [0, 1, 2, 3], Edges: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        Realization:    {0: (1.0,), 1: (2.0,), 2: (3.0,), 3: (4.0,)}
        dim:            1
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!")
        if not points:
            raise ValueError("The realization cannot be empty.")

        Kn = Graph.complete_graph(len(points))
        realization = {(Kn.vertices())[i]: Matrix(
            points[i]) for i in range(len(points))}
        return Framework(graph=Kn, realization=realization, dim=dim)

    def delete_vertex(self, vertex: Vertex) -> None:
        """
        Delete a vertex from the framework. (cf. :meth:`~Framework.add_vertex`)
        """
        self._graph.delete_vertex(vertex)
        del self._realization[vertex]

    def delete_vertices(self, vertices: List[Vertex]) -> None:
        """
        Delete a list of vertices from the framework. (cf. :meth:`~Framework.add_vertices`)
        """
        for vertex in vertices:
            self.delete_vertex(vertex)

    def delete_edge(self, edge: Edge) -> None:
        """Delete an edge from the framework. """
        self._graph.delete_edge(edge)

    def delete_edges(self, edges: List[Edge]) -> None:
        """Delete a list of edges from the framework."""
        self._graph.delete_edges(edges)

    def get_realization_list(self) -> List[Point]:
        """
        Rather than returning the internal matrix representation, this method returns the
        realization in the form of tuples. Conveniently, this format can be read by networkx.

        Examples
        -----
        >>> F = Framework.Complete([(0,0), (1,0), (1,1)])
        >>> F.get_realization_list()
        {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0)}
        """
        return {vertex: tuple([float(point) for point in self._realization[vertex]])
                for vertex in self._graph.vertices()}

    def get_realization(self) -> Dict[Vertex, Point]:
        """
        A copy of the framework's realization is returned. Hence, 
        modifying it does not affect the original framework.
        """
        return deepcopy(self._realization)

    def realization(self) -> List[Point]:
        """Alias for :meth:`~Framework.get_realization`"""
        return self.get_realization()

    def set_realization(self, realization: Dict[Vertex, Point]) -> None:
        """
        Provides a new realization for the framework. 

        Notes
        -----
        It is assumed that the realization contains all vertices from the
        underlying graph. Furthermore, all points in the realization need 
        to be contained in $\RR^d$ for a fixed $d$.

        Examples
        -----
        >>> F = Framework.Complete([(0,0), (1,0), (1,1)])
        >>> F.set_realization({vertex:(vertex,vertex+1) for vertex in F.graph().vertices()})
        >>> print(F)
        Graph:          Vertices: [0, 1, 2],    Edges: [(0, 1), (0, 2), (1, 2)]
        Realization:    {0: (0.0, 1.0), 1: (1.0, 2.0), 2: (2.0, 3.0)}
        dim:            2
        """
        if not len(realization) == len(self._graph.vertices()):
            raise IndexError(
                    "The realization does not contain the correct amount of vertices!")
        for v in self._graph.vertices():
            if v not in realization:
                raise KeyError(
                    "Vertex {vertex} is not a key of the given realization!")
            if not len(realization[v]) == self.dimension():
                raise IndexError(
                    f"The element {realization[v]} does not have the dimension {self.dimension()}!")
        self._realization = {
            v: Matrix(
                realization[v]) for v in realization.keys()}

    def change_vertex_coordinates(self, vertex: Vertex, point: Point) -> None:
        """
        Changes the coordinates of a single given vertex.

        Examples
        -----
        >>> F = Framework.from_points([(0,0)])
        >>> F.change_vertex_coordinates(0, (6,2))
        >>> print(F)
        Graph:          Vertices: [0],  Edges: []
        Realization:    {0: (6.0, 2.0)}
        dim:            2
        """
        if vertex not in self._realization:
            raise KeyError(
                "Vertex {vertex} is not a key of the given realization!")
        if not len(point) == self.dimension():
            raise IndexError(
                f"The point {point} does not have the dimension {self.dimension()}!")
        self._realization[vertex] = Matrix(point)

    def set_vertex_position(self, vertex: Vertex, point: Point) -> None:
        """Alias for :meth:`~Framework.change_vertex_coordinates`"""
        self.change_vertex_coordinates(vertex, point)

    def change_vertex_coordinates_list(
            self,
            vertices: List[Vertex],
            points: List[Point]) -> None:
        """
        Apply the method :meth:`~Framework.change_vertex_coordinates` to a list of vertices and points.
        
        Notes
        -----
        It is necessary that both lists have the same length. Also, no vertex from `vertices` can be 
        contained multiple times. For an explanation of `vertices` and `points`, see :meth:`~Framework.add_vertices`.
        """
        if list(set(vertices)).sort() != list(vertices).sort():
            raise ValueError(
                "Mulitple Vertices with the same name were found!")
        if not len(vertices) == len(points):
            raise IndexError(
                "The list of vertices does not have the same length as the list of points")
        for i in range(len(vertices)):
            self.change_vertex_coordinates(vertices[i], points[i])

    def set_vertex_positions(
            self,
            vertices: List[Vertex],
            points: List[Point]) -> None:
        """Alias for :meth:`~Framework.change_vertex_coordinates_list`"""
        self.change_vertex_coordinates_list(vertices, points)

    def change_realization(self, subset_of_realization: Dict[Vertex, Point]):
        """
        Apply the method :meth:`~Framework.change_vertex_coordinates_list` with a dictionary as input,
        rather than a list of vertices and points.
        """
        self.change_vertex_coordinates_list(
            subset_of_realization.keys(),
            subset_of_realization.values())

    def rigidity_matrix(
            self,
            vertex_order: List[Vertex] = None,
            pinned_vertices: Dict[Vertex, List[int]] = {},
            edges_ordered: bool = True) -> Matrix:
        r""" 
        Construct the rigidity matrix of the framework. 

        Definitions
        -----
        * :prf:ref:`Rigidity Matrix <def-rigidity-matrix>`
        
        Parameters
        -----
        vertex_order:
            By listing vertices in the preferred order, the rigidity matrix
            can be computed in a way the user expects.
        pinned_vertices:
            Dictionary of vertices and coordinates that do not contribute to the 
            computation of infinitesimal flexes. Each of the pinned vertex coordinates 
            adds a row given by the corresponding standard unit basis vector to 
            the rigidity matrix.
        edges_ordered:
            A Boolean indicating, whether the edges are assumed to be ordered (`True`), 
            or whether they should be internally sorted (`False).
        
        Examples
        -----
        >>> F = Framework.Complete([(0,0),(2,0),(1,3)])
        >>> F.rigidity_matrix(vertex_order=[2,1,0],pinned_vertices={0:[0], 1:[1]})
        Matrix([
        [ 0, 0, 2,  0, -2,  0],
        [ 1, 3, 0,  0, -1, -3],
        [-1, 3, 1, -3,  0,  0],
        [ 0, 0, 0,  1,  0,  0],
        [ 0, 0, 0,  0,  1,  0]])
        """
        try:
            if vertex_order is None:
                vertex_order = sorted(self._graph.vertices())
            else:
                if not set(
                        self._graph.vertices()) == set(vertex_order) or not len(
                        self._graph.vertices()) == len(vertex_order):
                    raise KeyError(
                        "The vertex_order needs to contain exactly the same vertices as the graph!")
        except TypeError as error:
            vertex_order = self._graph.vertices()

        for v in vertex_order:
            if v not in pinned_vertices:
                pinned_vertices[v] = []
        pinned_vertices = {v: pinned_vertices[v] for v in pinned_vertices.keys(
        ) if v in self._graph.vertices()}
        for v in pinned_vertices:
            if v not in self._graph.vertices():
                raise KeyError(
                    f"Vertex {v} in pinned_vertices is not a vertex of the graph!")
            if not len(pinned_vertices[v]) <= self.dim():
                raise IndexError(
                    f"The length of {pinned_vertices[v]} is larger than the dimension!")

        if not edges_ordered:
            edge_order = sorted(self._graph.edges())
        else:
            edge_order = self._graph.edges()

        """`delta` is responsible for distinguishing the edges (i,j) and (j,i)"""
        def delta(u, v, w):
            if w == u:
                return 1
            if w == v:
                return -1
            return 0

        """Add the column information about the pinned vertices, according to the `vertex_order`."""
        pinned_entries = flatten([[self.dim() * count + index for index in pinned_vertices[vertex_order[count]]]
                                  for count in range(len(vertex_order))])

        """Return the rigidity matrix with standard unit basis vectors added for each pinned coordinate."""
        return Matrix([flatten([delta(u, v, w)
                                * (self._realization[u] - self._realization[v])
                                for w in vertex_order])
                       for u, v in edge_order] +
                      [[1 if i == index else 0 for i in range(self.dim() * len(vertex_order))]
                       for index in pinned_entries])

    def stress_matrix(
            self,
            data: Any,
            pinned_vertices: Dict[Vertex, List[int]] = {},
            edge_order: List[Edge] = None) -> Matrix:
        r""" 
        Construct the stress matrix from a stress of from its support.
        """
        raise NotImplementedError()

    def trivial_infinitesimal_flexes(
            self, pinned_vertices: Dict[Vertex, List[int]] = {}) -> List[Matrix]:
        r"""
        The complete graph is infinitesimally rigid in all dimensions. 
        Thus, for computing the trivial infinitesimal flexes it suffices 
        to compute all infinitesimal flexes of the complete graph.
        
        Definitions
        -----
        * :prf:ref:`Trivial Motions <def-trivial-motions>`

        Parameters
        -----
        pinned_vertices:
            see :meth:`~Framework.rigidity_matrix`

        Examples
        -----
        >>> F = Framework.Complete([(0,0),(2,0),(1,3)])
        >>> F.trivial_infinitesimal_flexes()
        [Matrix([
            [ 3],
            [-1],
            [ 3],
            [ 1],
            [ 0],
            [ 0]]), 
        Matrix([
            [1],
            [0],
            [1],
            [0],
            [1],
            [0]]), 
        Matrix([
            [-3],
            [ 2],
            [-3],
            [ 0],
            [ 0],
            [ 1]])
        ]
        """
        vertices = self._graph.vertices()
        Kn = Graph.complete_graph_on_vertices(vertices)
        F_Kn = Framework(
            graph=Kn,
            realization=self.realization(),
            dim=self.dim())
        return F_Kn.infinitesimal_flexes(
            pinned_vertices=pinned_vertices,
            include_trivial=True)

    def nontrivial_infinitesimal_flexes(
            self, pinned_vertices: Dict[Vertex, List[int]] = {}) -> List[Matrix]:
        """
        Return the entries of the rigidity matrix' kernel that are not trivial infinitesimal flexes.
        See :meth:`~Framework.trivial_infinitesimal_flexes`
        """
        return self.infinitesimal_flexes(
            pinned_vertices=pinned_vertices,
            include_trivial=False)

    def infinitesimal_flexes(
            self,
            pinned_vertices: Dict[Vertex, List[int]] = {},
            include_trivial: bool = False) -> List[Matrix]:
        r""" 
        Returns a basis of the space of infinitesimal flexes. This is done by orthogonalizing the
        space of trivial and non-trivial flexes and subsequently forgetting the trivial flexes,
        provided that `include_trivial` is set to `False`. Else, return the entire kernel.

        Definitions
        -----
        * :prf:ref:`Infinitesimal Motion <def-infinitesimal-motion>`

        Parameters
        -----
        pinned_vertices:
            see :meth:`~Framework.rigidity_matrix`
        include_trivial:
            Boolean that decides, whether the trivial motions should be included `True` or not (`False`)

        Examples
        -----
        >>> F = Framework.Complete([[0,0], [1,0], [1,1], [0,1]])
        >>> F.delete_edges([(0,2), (1,3)])
        >>> F.infinitesimal_flexes(include_trivial=False)
        [Matrix([
        [ 1/4],
        [ 1/4],
        [ 1/4],
        [-1/4],
        [-1/4],
        [-1/4],
        [-1/4],
        [ 1/4]])]
        """
        if include_trivial:
            return self.rigidity_matrix(
                pinned_vertices=pinned_vertices).nullspace()
        trivial_flexes = self.trivial_infinitesimal_flexes(
            pinned_vertices=pinned_vertices)
        all_flexes = self.rigidity_matrix(
            pinned_vertices=pinned_vertices).nullspace()
        basis_flexspace = Matrix.orthogonalize(
            *(trivial_flexes + all_flexes), rankcheck=False)
        return basis_flexspace[len(trivial_flexes):len(all_flexes) + 1]

    def stresses(self) -> Any:
        r"""Return a basis of the space of stresses."""
        raise NotImplementedError()

    def rigidity_matrix_rank(
            self, pinned_vertices: Dict[Vertex, List[int]] = {}) -> int:
        """
        Compute the rank of the rigidity matrix. 

        Parameters
        -----
        pinned_vertices:
            see :meth:`~Framework.rigidity_matrix`
        """
        return self.rigidity_matrix(pinned_vertices=pinned_vertices).rank()

    def is_infinitesimally_rigid(self) -> bool:
        """
        Check whether the given framework is rigid.

        Definitions
        -----
        * :prf:ref:`Infinitesimal Rigidity <def-infinitesimal-rigidity>`

        Notes
        -----
        A framework is called infinitesimally rigid, if all infinitesimal flexes are trivial. 
        This is the case when either there is less than 1 vertex in the framework, making it 
        trivially rigid, or the rigidity matrix has rank $d \cdot |V| - {d\choose 2}$, since 
        there are ${d\choose 2}$ isometries of $\RR^d$.
        """
        return len(self.graph().vertices()) <= 1 or \
            self.rigidity_matrix_rank() == self.dim() * len(self.graph().vertices()) \
            - (self.dim()) * (self.dim() + 1) // 2
    
    def is_infinitesimally_flexible(self) -> bool:
        """
        Check whether the given framework is flexible.
        See :meth:`~Framework.is_infinitesimally_rigid`
        """
        return not self.is_infinitesimally_rigid()

    def is_infinitesimally_spanning(self) -> bool:
        raise NotImplementedError()

    def is_minimally_infinitesimally_rigid(self) -> bool:
        """
        Check whether a framework is minimally infinitesimally rigid.

        Notes
        -----
        A framework is called minimally infinitesimally rigid, if it is infinitesimally rigid
        and the removal of any edge results in an infinitesimally flexible graph.

        Examples
        -----
        >>> F = Framework.Complete([[0,0], [1,0], [1,1], [0,1]])
        >>> F.is_minimally_infinitesimally_rigid()
        False
        >>> F.delete_edge((0,2))
        >>> F.is_minimally_infinitesimally_rigid()
        True
        """
        if not self.is_infinitesimally_rigid():
            return False
        for edge in self.graph().edges:
            F = deepcopy(self)
            F.delete_edge(edge)
            print(F)
            if F.is_infinitesimally_rigid():
                return False
        return True

    def is_independent(self) -> bool:
        raise NotImplementedError()

    def is_prestress_stable(self) -> bool:
        raise NotImplementedError()

    def is_redundantly_rigid(self) -> bool:
        """
        Check if the framework is redundantly rigid.
        
        Definitions
        -----
        :prf:ref:`Redundant Rigidity <def-minimally-redundantly-rigid-framework>`

        >>> F = Framework.Empty(dim=2)
        >>> F.add_vertices([(1,0), (1,1), (0,3), (-1,1)], ['a','b','c','d'])
        >>> F.add_edges([('a','b'), ('b','c'), ('c','d'), ('a','d'), ('a','c'), ('b','d')])
        >>> F.is_redundantly_rigid()
        True
        >>> F.delete_edge(('a','c'))
        >>> F.is_redundantly_rigid()
        False
        """
        for edge in self._graph.edges:
            F = deepcopy(self)
            F.delete_edge(edge)
            if not F.is_infinitesimally_rigid():
                return False
        return True

    def is_congruent(self, framework_) -> bool:
        raise NotImplementedError()

    def is_equivalent(self, framework_) -> bool:
        raise NotImplementedError()
