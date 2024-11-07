"""
This file contains functionality related to finite flexes.
"""

from pyrigi.graph import Graph
from pyrigi.data_type import Vertex, Edge, Point
from sympy import sympify, Symbol, Matrix, vector, simplify
from pyrigi.misc import point_to_vector


class Motion:
    pass


class ParametricMotion(Motion):
    pass


class Motion(object):
    """
    Class representing a finite flex of a framework.
    """

    def __init__(self, graph: Graph) -> None:
        self._graph = graph

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} of a " + self._graph.__str__()
        )

    def __repr__(self) -> str:
        return self.__str__()


class ParametricMotion(Motion):
    def __init__(
        self, graph: Graph, motion: dict[Vertex, Point], interval: tuple
    ) -> None:
        """
        Creates an instance.
        """

        super().__init__(graph)
        self._parametrization = {i: point_to_vector(v) for i, v in motion.items()}
        symbols = set()
        for vertex, position in self._parametrization.items():
            for coord in position:
                for sym in coord.free_symbols:
                    if sym.is_Symbol:
                        symbols.add(sym)

        if len(symbols) != 1:
            raise ValueError(
                f"Expected exactly one parameter in the motion! got: {len(symbols)} parameters."
            )

        self._interval = interval
        self._parameter = symbols.pop()
        if not self.check_edge_lengths():
            raise ValueError("The given motion does not preserve edge lengths!")

    def check_edge_lengths(self) -> bool:
        """
        Check whether the saved motion preserves edge lengths.
        """

        for u, v in self._graph.edges:
            l = self._parametrization[u] - self._parametrization[v]
            l = l.T * l
            l.simplify()
            if l.has(self._parameter):
                return False
        return True

    def get_realization(self, value, numeric: bool = True) -> dict[Vertex:Point]:
        """
        Return specific realization for the given value of t.
        """

        res = {}
        for v in self._graph.nodes:
            if numeric:
                res[v] = self._parametrization[v].subs({self._parameter: value}.evalf())
            else:
                res[v] = simplify(
                    self._parametrization[v].subs({self._parameter: value})
                )
        return res

    def __str__(self) -> str:
        res = super().__str__() + " with motion defined for every vertex:"
        for vertex, param in self._parametrization.items():
            res = res + "\n" + str(vertex) + ": " + str(param)
        return res
        return (
            super().__str__() + " with motion defined for every vertex: " + str(self._parametrization)
        )
