"""
This file contains functionality related to finite flexes.
"""

from pyrigi.graph import Graph
from pyrigi.data_type import Vertex, Point
from sympy import simplify
from pyrigi.misc import point_to_vector


class Motion(object):
    """
    Class representing a finite flex of a framework.
    """

    def __init__(self, graph: Graph) -> None:
        """
        Create an instance of a graph's motion.
        """

        self._graph = graph

    def __str__(self) -> str:
        return f"{self.__class__.__name__} of a " + self._graph.__str__()

    def __repr__(self) -> str:
        return self.__str__()


class ParametricMotion(Motion):
    """
    Class representing a parametric motion.
    """

    def __init__(
        self, graph: Graph, motion: dict[Vertex, Point], interval: tuple
    ) -> None:
        """
        Creates an instance.
        """

        super().__init__(graph)

        if not len(motion) == self._graph.number_of_nodes():
            raise IndexError(
                "The realization does not contain the correct amount of vertices!"
            )

        for v in self._graph.nodes:
            if v not in motion:
                raise KeyError("Vertex {vertex} is not a key of the given realization!")

        self._parametrization = {i: point_to_vector(v) for i, v in motion.items()}

        symbols = set()
        for vertex, position in self._parametrization.items():
            for coord in position:
                for symbol in coord.free_symbols:
                    if symbol.is_Symbol:
                        symbols.add(symbol)

        if len(symbols) != 1:
            raise ValueError(
                "Expected exactly one parameter in the motion! got: "
                f"{len(symbols)} parameters."
            )

        self._interval = interval
        self._parameter = symbols.pop()
        if not self.check_edge_lengths():
            raise ValueError("The given motion does not preserve edge lengths!")

    def check_edge_lengths(self) -> bool:
        """
        Check whether the saved motion preserves edge lengths.

        TODO
        ----
        Tests

        """

        for u, v in self._graph.edges:
            edge = self._parametrization[u] - self._parametrization[v]
            edge_len = edge.T * edge
            edge_len.simplify()
            if edge_len.has(self._parameter):
                return False
        return True

    def get_realization(self, value, numeric: bool = False) -> dict[Vertex:Point]:
        """
        Return specific realization for the given value of the parameter.

        TODO
        ----
        Tests

        """

        realization = {}
        for v in self._graph.nodes:
            placement = simplify(
                self._parametrization[v].subs({self._parameter: value})
            )
            if numeric:
                placement = placement.evalf()
            realization[v] = placement
        return realization

    def __str__(self) -> str:
        res = super().__str__() + " with motion defined for every vertex:"
        for vertex, param in self._parametrization.items():
            res = res + "\n" + str(vertex) + ": " + str(param)
        return res

    def animate(self) -> None:
        """
        Animation of the parametric motion for parameter
        in the range specified in the constructor.
        """

        raise NotImplementedError()
