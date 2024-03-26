"""

Module for the functionality concerning frameworks.

.. currentmodule:: pyrigi.framework

Classes:

.. autosummary::

    Framework

"""

from copy import deepcopy
from pyrigi.graph import Graph
from pyrigi.matrix import Matrix
from typing import List, Dict, Tuple, Any


class Framework(object):
    """
    This class provides the functionality for frameworks. By definition, it is a tuple of a graph and a realization.

    ATTRIBUTES
    ----------
    graph : Graph
    realization : dict
    dim : int

    """
    # TODO override decorator for empty constructor?

    def __init__(self, graph: Graph, realization: Dict[Any, List[float]]) -> None:
        # TODO: check that graph and realization is not empty
        assert isinstance(graph, Graph)
        dim = len(list(realization.values())[0])
        for v in graph.vertices():
            assert v in realization
            assert len(realization[v]) == dim
        self.realization = {v: realization[v] for v in graph.vertices()}
        self._graph = deepcopy(graph)
        self._graph._part_of_framework = True
        self.dim = dim

    def dim(self) -> int:
        return self.dim

    def dimension(self) -> int:
        return self.dim()

    def add_vertex(self, point: list[float], vertex: Any = None) -> None:
        if vertex is None:
            candidate = len(self._graph.vertices())
            while candidate in self._graph.vertices():
                candidate += 1
            vertex = candidate
        assert vertex not in self._graph.vertices()
        self.realization[vertex] = point
        self._graph.add_node(vertex)

    def add_vertices(self, points: List[List[float]], vertices: List[Any] = []) -> None:
        assert (len(points) == len(vertices) or not vertices)
        if not vertices:
            for point in points:
                self.add_vertex(point)
        else:
            for p, v in zip(points, vertices):
                self.add_vertex(p, v)

    def add_edge(self, edge: Tuple[int, int]) -> None:
        assert (len(edge)) == 2
        assert (edge[0] in self._graph.nodes and edge[1] in self._graph.nodes)
        self._graph.add_edge(*(edge))

    def add_edges(self, edges: List[Tuple[int, int]]) -> None:
        for edge in edges:
            self.add_edge(edge)

    def underlying_graph(self) -> Graph:
        """Return a copy of the graph object so that the graph object is not mutated"""
        return deepcopy(self._graph)

    def graph(self) -> Graph:
        """Return a copy of the graph object so that the graph object is not mutated"""
        return self.underlying_graph()

    def print(self) -> None:
        """Method to display the data inside the Framework."""
        print('Graph:\t\t', self._graph)
        print('Realization:\t', self.realization)

    @classmethod
    def from_points(cls, points: List[List[float]]) -> None:
        raise NotImplementedError()

    @classmethod
    def from_graph(cls, graph: Graph) -> None:
        raise NotImplementedError()

    @classmethod
    def empty(cls, dim: int) -> None:
        raise NotImplementedError()

    def delete_vertex(self, vertex: Any) -> None:
        raise NotImplementedError()

    def delete_vertices(self, vertices: List[Any]) -> None:
        raise NotImplementedError()

    def delete_edge(self, edge: Tuple[int, int]) -> None:
        raise NotImplementedError()

    def delete_edges(self, edges: List[Tuple[int, int]]) -> None:
        raise NotImplementedError()

    def set_vertex_position(self, vertex: Any, point: List[float]) -> None:
        raise NotImplementedError()

    def set_realization(self, realization: Dict[Any, List[float]]) -> None:
        """Add consistency check here"""
        raise NotImplementedError()

    def rigidity_matrix(self) -> Matrix:
        r""" Construct the rigidity matrix of the framework
        """
        raise NotImplementedError()

    def stress_matrix(self, data: Any) -> Matrix:
        r""" Construct the stress matrix from a stress of from its support
        """
        raise NotImplementedError()

    def infinitesimal_flexes(self, trivial: bool = False) -> Any:
        r""" Returns a basis of the space of infinitesimal flexes
        """
        raise NotImplementedError()

    def stresses(self) -> Any:
        r""" Returns a basis of the space of stresses
        """
        raise NotImplementedError()

    def rigidity_matrix_rank(self) -> int:
        raise NotImplementedError()

    def is_infinitesimally_rigid(self) -> bool:
        raise NotImplementedError()

    def is_infinitesimally_spanning(self) -> bool:
        raise NotImplementedError()

    def is_minimally_infinitesimally_rigid(self) -> bool:
        raise NotImplementedError()

    def is_infinitesimally_flexible(self) -> bool:
        raise NotImplementedError()

    def is_independent(self) -> bool:
        raise NotImplementedError()

    def is_prestress_stable(self) -> bool:
        raise NotImplementedError()

    def is_redundantly_rigid(self) -> bool:
        raise NotImplementedError()

    def is_congruent(self, framework_) -> bool:
        raise NotImplementedError()

    def is_equivalent(self, framework_) -> bool:
        raise NotImplementedError()

    def pin(self, vertices: List[Any]) -> None:
        raise NotImplementedError()

    def trivial_infinitesimal_flexes(self) -> Any:
        raise NotImplementedError()
