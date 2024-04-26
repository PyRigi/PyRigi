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
    :prf:ref:`Framework <def-framework>`
        

    Parameters
    ----------
    graph
    realization: 
        A dictionary mapping the vertices of the graph to points in $\RR^n$.
    pinned_vertices: 
        A dictionary mapping vertices to lists of coordinate
        indices. We initialize this dictionary so that each index
        from the graph appears. This will make our life easier 
        later on.
    dim:
        The dimension is usually initialized by the realization. If
        the realization is empty, the dimension is 0 by default.

    Notes
    -----
    Internally, the realization is represented as a dictionary of
    matrices ("vectors").
    

    """
    # TODO override decorator for empty constructor?

    def __init__(self,
                 graph: Graph = Graph(),
                 realization: Dict[Vertex, Point] = {},
                 pinned_vertices: Dict[Vertex, List[int]] = {},
                 dim: int = 2) -> None:
        assert isinstance(graph, Graph)
        if len(realization.values()) == 0:
            dimension = dim
        else:
            dimension = len(list(realization.values())[0])

        assert len(realization.keys()) == len(graph.vertices())
        for v in graph.vertices():
            assert v in realization
            assert len(realization[v]) == dimension
            if not v in pinned_vertices:
                pinned_vertices[v] = []
        
#         TODO Resolve strange behavior that `pinned_vertices` is initialized as `{0:[], 1:[], ...}` when
#             realization is initialized as an empty dict
        pinned_vertices = {v:pinned_vertices[v] for v in pinned_vertices.keys() if v in graph.vertices()}
        for v in pinned_vertices:
            assert v in graph.vertices()
            assert len(pinned_vertices[v]) <= dimension

        self._realization = {v: Matrix(realization[v])
                             for v in graph.vertices()}
        self._graph = deepcopy(graph)
        self._dim = dimension
        self._pinned_vertices = pinned_vertices

    def dim(self) -> int:
        """Return the dimension of the framework."""
        return self._dim

    def dimension(self) -> int:
        """
        Alias for :meth:`~Framework.dim`
        """
        return self.dim()

    def add_vertex(self, point: Point, vertex: Vertex = None) -> None:
        if vertex is None:
            candidate = len(self._graph.vertices())
            while candidate in self._graph.vertices():
                candidate += 1
            vertex = candidate
        assert vertex not in self._graph.vertices()
        self._realization[vertex] = Matrix(point)
        self._graph.add_node(vertex)
        self._pinned_vertices[vertex] = []

    def add_vertices(self,
                     points: List[Point],
                     vertices: List[Vertex] = []) -> None:
        assert (len(points) == len(vertices) or not vertices)
        if not vertices:
            for point in points:
                self.add_vertex(point)
        else:
            for p, v in zip(points, vertices):
                self.add_vertex(p, v)

    def add_edge(self, edge: Edge) -> None:
        assert (len(edge)) == 2
        assert (edge[0] in self._graph.nodes and edge[1] in self._graph.nodes)
        self._graph.add_edge(*(edge))

    def add_edges(self, edges: List[Edge]) -> None:
        for edge in edges:
            self.add_edge(edge)

    def underlying_graph(self) -> Graph:
        """
        Return a copy of the underlying graph.

        A deep copy of the underlying graph of the framework is returned.
        Hence, modifying it does not effect the original framework.
        """
        return deepcopy(self._graph)

    def graph(self) -> Graph:
        """
        Alias for :meth:`~Framework.underlying_graph`
        """
        return self.underlying_graph()

    def print(self) -> None:
        """Method to display the data inside the Framework."""
        print('Graph:\t\t', self._graph)
        print('Realization:\t', {key:self.get_realization_list()[key] for key in sorted(self.get_realization_list())})
        print('Pinned coords:\t', {key:self.get_pinned_vertices()[key] for key in sorted(self.get_pinned_vertices())})
        print('dim:\t\t', self.dim())

    def draw_framework(self) -> None:
        nx.draw(self._graph, pos=self.get_realization_list())

    @classmethod
    def from_points(cls, points: List[Point]) -> None:
        """
        #TODO Generate a framework from a list of points
        """
        raise NotImplementedError()

    @classmethod
    def from_graph(cls, graph: Graph) -> None:
        """
        #TODO Framework with random coordinates?
        """
        raise NotImplementedError()

    @classmethod
    def Empty(cls, dim: int) -> None:
        """
        Generate an empty framework.
        """
        raise NotImplementedError()

    def delete_vertex(self, vertex: Vertex) -> None:
        self._graph.delete_vertex(vertex)
        del self._realization[vertex], self._pinned_vertices[vertex]

    def delete_vertices(self, vertices: List[Vertex]) -> None:
        for vertex in vertices:
            self.delete_vertex(vertex)

    def delete_edge(self, edge: Edge) -> None:
        self._graph.delete_edge(edge)

    def delete_edges(self, edges: List[Edge]) -> None:
        self._graph.delete_edges(edges)

    def get_realization_list(self) -> List[Point]:
        """
        Rather than returning the internal matrix representation, this method returns the
        realization in the form of tuples. This format can also be read by networkx.
        """
        return {vertex: tuple([float(point) for point in self._realization[vertex]])
                for vertex in self._graph.vertices()}

    def get_realization(self) -> Dict[Vertex, Point]:
        return deepcopy(self._realization)

    def realization(self) -> List[Point]:
        """
        Alias for :meth:`~Framework.get_realization`
        """
        return self.get_realization()

    def set_realization(self, realization: Dict[Vertex, Point]) -> None:
        for v in self._graph.vertices():
            assert v in realization
            assert len(realization[v]) == self.dimension()
        self._realization = realization

    def change_vertex_coordinates(self, vertex: Vertex, point: Point) -> None:
        assert vertex in self._realization
        assert len(point) == self._dim
        self._realization[vertex] = point

    def set_vertex_position(self, vertex: Vertex, point: Point) -> None:
        """
        Alias for :meth:`~Framework.change_vertex_coordinates`
        """
        self.change_vertex_coordinates(vertex, point)

    def change_vertex_coordinates_list(
            self,
            vertices: List[Vertex],
            points: List[Point]) -> None:
        if list(set(vertices)).sort() != list(vertices).sort():
            raise AttributeError(
                "Multiple Vertices with the same name were found!")
        assert len(vertices) == len(points)
        for i in range(0, len(vertices)):
            self.change_vertex_coordinates(vertices[i], points[i])
        
    def set_vertex_positions(
            self,
            vertices: List[Vertex],
            points: List[Point]) -> None:
        """
        Alias for :meth:`~Framework.change_vertex_coordinates_list`
        """
        self.change_vertex_coordinates_list(vertices, points)

    def change_realization(self, subset_of_realization: Dict[Vertex, Point]):
        self.change_vertex_coordinates_list(
            subset_of_realization.keys(),
            subset_of_realization.values())
        
    def pin_vertex(self, vertex: Vertex, indices: List[int]) -> None:
        assert vertex in self.graph().vertices()
        self._pinned_vertices[vertex] = indices

    def pin_vertices(self, vertices: List[Vertex], list_of_indices: List[List[int]]) -> None:
        for i in range(len(vertex)):
            self.pin_vertex(vertices[i], list_of_indices[i])

    def set_pinned_vertices(self, pinned_vertices: Dict[Vertex, List[int]]) -> None:
        for v in self._pinned_vertices.keys():
            if not v in pinned_vertices:
                pinned_vertices[v] = []
        assert set(self._pinned_vertices.keys()) == set(pinned_vertices.keys())
        self._pinned_vertices = pinned_vertices

    def get_pinned_vertices(self) -> Dict[Vertex, List[int]]:
        return deepcopy(self._pinned_vertices)

    def rigidity_matrix(
            self,
            vertex_order: List[Vertex] = None,
            edges_ordered: bool = True) -> Matrix:
        r""" Construct the rigidity matrix of the framework
        """
        try:
            if vertex_order is None:
                vertex_order = sorted(self._graph.vertices())
            else:
                assert set(self._graph.vertices()) == set(vertex_order)
        except TypeError as error:
            vertex_order = self._graph.vertices()

        if edges_ordered:
            edge_order = sorted(self._graph.edges())
        else:
            edge_order = self._graph.edges()

        def delta(u, v, w):
            if w == u:
                return 1
            if w == v:
                return -1
            return 0

        """Add the column information about the pinned vertices, according to the `vertex_order`."""
        pinned_entries = flatten([[self.dim()*count+index for index in self._pinned_vertices[vertex_order[count]]] 
                                  for count in range(len(vertex_order))])

        """Return the rigidity matrix with standard unit basis vectors added for each pinned coordinate."""
        return Matrix([flatten([delta(u, v, w)
                                * (self._realization[u] - self._realization[v])
                                for w in vertex_order])
                        for u, v in edge_order] +
                    [[1 if i == index else 0 for i in range(self.dim()*len(vertex_order))] 
                        for index in pinned_entries])

    def stress_matrix(
            self,
            data: Any,
            edge_order: List[Edge] = None) -> Matrix:
        r""" Construct the stress matrix from a stress of from its support
        """
        raise NotImplementedError()

    def trivial_infinitesimal_flexes(self) -> List[Matrix]:
        r"""The complete graph is infinitesimally rigid in all dimensions. Thus, for computing the trivial
        flexes it suffices to compute all infinitesimal flexes of the complete graph."""
        Kn = nx.complete_graph(len(self._graph.vertices()))
        F_Kn = Framework(
            graph=Graph(Kn.edges),
            realization=self.realization(),
            pinned_vertices=self.get_pinned_vertices(),
            dim=self.dim())
        return F_Kn.infinitesimal_flexes(include_trivial=True)

    def nontrivial_infinitesimal_flexes(self) -> List[Matrix]:
        return self.infinitesimal_flexes(include_trivial=False)

    def infinitesimal_flexes(
            self,
            include_trivial: bool = False) -> List[Matrix]:
        r""" Returns a basis of the space of infinitesimal flexes. This is done by orthogonalizing the
        space of trivial and non-trivial flexes and subsequently forgetting the trivial flexes.
        """
        if include_trivial:
            return self.rigidity_matrix().nullspace()
        trivial_flexes = self.trivial_infinitesimal_flexes()
        all_flexes = self.rigidity_matrix().nullspace()
        basis_flexspace = Matrix.orthogonalize(
            *(trivial_flexes + all_flexes), rankcheck=False)
        return basis_flexspace[len(trivial_flexes):len(all_flexes) + 1]

    def stresses(self) -> Any:
        r""" Return a basis of the space of stresses.
        """
        raise NotImplementedError()

    def rigidity_matrix_rank(self) -> int:
        return self.rigidity_matrix().rank()

    def is_infinitesimally_rigid(self) -> bool:
        return len(self.graph().vertices()) <= 1 or self.rigidity_matrix_rank() == self.dim(
        ) * len(self.graph().vertices()) - (self.dim()) * (self.dim() + 1) // 2

    def is_infinitesimally_spanning(self) -> bool:
        raise NotImplementedError()

    def is_minimally_infinitesimally_rigid(self) -> bool:
        """A framework is called minimally infinitesimally rigid, if it is infinitesimally rigid
        and the removal of any edge results in an infinitesimally flexible graph."""
        if not self.is_infinitesimally_rigid():
            return False
        for edge in self.graph().edges:
            F = deepcopy(self)
            F.delete_edge(edge)
            if F.is_infinitesimally_rigid():
                return False
        return True

    def is_infinitesimally_flexible(self) -> bool:
        return not self.is_infinitesimally_rigid()

    def is_independent(self) -> bool:
        raise NotImplementedError()

    def is_prestress_stable(self) -> bool:
        raise NotImplementedError()

    def is_redundantly_rigid(self) -> bool:
        """
        Check if the framework is :prf:ref:`redundantly rigid <def-minimally-redundantly-rigid-framework>`
        """
        for edge in self._graph.edges:
            F = deepcopy(self)
            F.delete_edge(edge)
            if not F.is_infinitesimally_rigid(F):
                return False
        return True

    def is_congruent(self, framework_) -> bool:
        raise NotImplementedError()

    def is_equivalent(self, framework_) -> bool:
        raise NotImplementedError()