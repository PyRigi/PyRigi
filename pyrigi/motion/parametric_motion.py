"""
This module contains functionality related to parametric motions.
"""

from copy import deepcopy
from typing import Any

import numpy as np
import sympy as sp
from sympy import simplify

from pyrigi._utils._conversion import point_to_vector, sympy_expr_to_float
from pyrigi.data_type import (
    Number,
    Point,
    Vertex,
)
from pyrigi.graph import Graph
from pyrigi.motion import Motion


class ParametricMotion(Motion):
    """
    Class representing a parametric motion.

    Definitions
    -----------
    :prf:ref:`Continuous flex (motion)<def-motion>`

    Parameters
    ----------
    graph:
    motion:
        A parametrization of a continuous flex using SymPy expressions,
        or strings that can be parsed by SymPy.
    interval:
        The interval in which the parameter is considered.

    Examples
    --------
    >>> from pyrigi import ParametricMotion
    >>> import sympy as sp
    >>> from pyrigi import graphDB as graphs
    >>> motion = ParametricMotion(
    ...     graphs.Cycle(4),
    ...     {
    ...         0: ("0", "0"),
    ...         1: ("1", "0"),
    ...         2: ("4 * (t**2 - 2) / (t**2 + 4)", "12 * t / (t**2 + 4)"),
    ...         3: (
    ...             "(t**4 - 13 * t**2 + 4) / (t**4 + 5 * t**2 + 4)",
    ...             "6 * (t**3 - 2 * t) / (t**4 + 5 * t**2 + 4)",
    ...         ),
    ...     },
    ...     [-sp.oo, sp.oo],
    ... )
    >>> print(motion)
    ParametricMotion of a Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 3], [1, 2], [2, 3]] with motion defined for every vertex:
    0: Matrix([[0], [0]])
    1: Matrix([[1], [0]])
    2: Matrix([[(4*t**2 - 8)/(t**2 + 4)], [12*t/(t**2 + 4)]])
    3: Matrix([[(t**4 - 13*t**2 + 4)/(t**4 + 5*t**2 + 4)], [(6*t**3 - 12*t)/(t**4 + 5*t**2 + 4)]])
    """  # noqa: E501

    def __init__(
        self,
        graph: Graph,
        motion: dict[Vertex, Point],
        interval: tuple[Number] | list[Number],
    ) -> None:
        """
        Create an instance of ``ParametricMotion``.
        """

        super().__init__(graph, len(list(motion.values())[0]))

        if not len(motion) == self._graph.number_of_nodes():
            raise ValueError(
                "The realization does not contain the correct amount of vertices!"
            )

        self._parametrization = {v: point_to_vector(pos) for v, pos in motion.items()}
        for v in self._graph.nodes:
            if v not in motion:
                raise KeyError(f"Vertex {v} is not a key of the given realization!")
            if len(self._parametrization[v]) != self._dim:
                raise ValueError(
                    f"The point {self._parametrization[v]} in the parametrization"
                    f" corresponding to vertex {v} does not have the right dimension."
                )

        if not interval[0] < interval[1]:
            raise ValueError("The given interval is not a valid interval!")

        symbols = set()
        for pos in self._parametrization.values():
            for coord in pos:
                for symbol in coord.free_symbols:
                    if symbol.is_Symbol:
                        symbols.add(symbol)

        if len(symbols) != 1:
            raise ValueError(
                "Expected exactly one parameter in the motion! got: "
                f"{len(symbols)} parameters."
            )

        self._interval = list(interval)
        self._parameter = symbols.pop()
        self._input_check_edge_lengths()

    def interval(self) -> list[Number]:
        """Return the underlying interval."""
        return deepcopy(self._interval)

    def parametrization(self, as_points: bool = False) -> dict[Vertex, Point]:
        """Return the parametrization."""
        if not as_points:
            return deepcopy(self._parametrization)
        return {v: list(pos) for v, pos in self._parametrization.items()}

    def _input_check_edge_lengths(self) -> None:
        """
        Check whether the motion preserves the edge lengths and
        raise an error otherwise.
        """
        for u, v in self._graph.edges:
            edge = self._parametrization[u] - self._parametrization[v]
            edge_len = edge.T * edge
            edge_len.simplify()
            if edge_len.has(self._parameter):
                raise ValueError("The given motion does not preserve edge lengths!")

    def realization(self, value: Number, numerical: bool = False) -> dict[Vertex:Point]:
        """
        Return a specific realization for the given ``value`` of the parameter.

        Parameters
        ----------
        value:
            The parameter of the deformation path is substituted by ``value``.
        numerical:
            Boolean determining whether the sympy expressions are supposed to be
            evaluated to numerical (``True``) or not (``False``).
        """

        realization = {}
        for v in self._graph.nodes:
            if numerical:
                _value = sympy_expr_to_float(value)
                placement = sympy_expr_to_float(
                    self._parametrization[v].subs({self._parameter: float(_value)})
                )
            else:
                placement = simplify(
                    self._parametrization[v].subs({self._parameter: value})
                )
            realization[v] = placement
        return realization

    def __str__(self) -> str:
        """Return the string representation."""
        res = super().__str__() + " with motion defined for every vertex:"
        for vertex, param in self._parametrization.items():
            res = res + "\n" + str(vertex) + ": " + str(param)
        return res

    def __repr__(self) -> str:
        """Return a representation of the parametric motion."""
        o_str = f"ParametricMotion({repr(self.graph)}, "
        str_parametrization = {
            v: [str(p) for p in pos]
            for v, pos in self.parametrization(as_points=True).items()
        }
        o_str += f"{str_parametrization}, {self.interval()})"
        return o_str

    def _realization_sampling(
        self, number_of_samples: int, use_tan: bool = False
    ) -> list[dict[Vertex, Point]]:
        """
        Return ``number_of_samples`` realizations for sampled values of the parameter.
        """
        realizations = []
        if not use_tan:
            for i in np.linspace(
                self._interval[0], self._interval[1], number_of_samples
            ):
                realizations.append(self.realization(i, numerical=True))
            return realizations

        newinterval = [
            sympy_expr_to_float(sp.atan(self._interval[0])),
            sympy_expr_to_float(sp.atan(self._interval[1])),
        ]
        for i in np.linspace(newinterval[0], newinterval[1], number_of_samples):
            realizations.append(self.realization(f"tan({i})", numerical=True))
        return realizations

    def animate(
        self,
        sampling: int = 50,
        **kwargs,
    ) -> Any:
        """
        Animate the parametric motion.

        See the parent method :meth:`~.Motion.animate` for a list of possible keywords.

        Parameters
        ----------
        sampling:
            The number of discrete points or frames used to approximate the motion in the
            animation. A higher value results in a smoother and more accurate
            representation of the motion, while a lower value can speed up rendering
            but may lead to a less precise or jerky animation. This parameter controls
            the resolution of the animation movement by setting the density of
            sampled data points between keyframes or time steps.
        """
        lower, upper = self._interval
        if lower == -np.inf or upper == np.inf:
            realizations = self._realization_sampling(sampling, use_tan=True)
        else:
            realizations = self._realization_sampling(sampling)

        return super().animate(
            realizations,
            None,
            **kwargs,
        )
