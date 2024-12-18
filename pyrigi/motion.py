"""
This module contains functionality related to motions (continuous flexes).
"""

from pyrigi.graph import Graph
from pyrigi.data_type import Vertex, Point
from sympy import simplify
from pyrigi.misc import point_to_vector
import numpy as np
import sympy as sp
from IPython.display import SVG


class Motion(object):
    """
    An abstract class representing a continuous flex of a framework.
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

    >>> from pyrigi.motion import ParametricMotion
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
    >>> motion
    ParametricMotion of a Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 3], [1, 2], [2, 3]] with motion defined for every vertex:
    0: Matrix([[0], [0]])
    1: Matrix([[1], [0]])
    2: Matrix([[(4*t**2 - 8)/(t**2 + 4)], [12*t/(t**2 + 4)]])
    3: Matrix([[(t**4 - 13*t**2 + 4)/(t**4 + 5*t**2 + 4)], [(6*t**3 - 12*t)/(t**4 + 5*t**2 + 4)]])
    """  # noqa: E501

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

        self._parametrization = {i: point_to_vector(v) for i, v in motion.items()}
        self._dim = len(list(self._parametrization.values())[0])
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

        """

        for u, v in self._graph.edges:
            edge = self._parametrization[u] - self._parametrization[v]
            edge_len = edge.T * edge
            edge_len.simplify()
            if edge_len.has(self._parameter):
                return False
        return True

    def realization(self, value, numeric: bool = False) -> dict[Vertex:Point]:
        """
        Return specific realization for the given value of the parameter.

        """

        realization = {}
        for v in self._graph.nodes:
            if numeric:
                placement = (
                    self._parametrization[v]
                    .subs({self._parameter: float(value)})
                    .evalf()
                )
            else:
                placement = simplify(
                    self._parametrization[v].subs({self._parameter: value})
                )
            realization[v] = placement
        return realization

    def __str__(self) -> str:
        res = super().__str__() + " with motion defined for every vertex:"
        for vertex, param in self._parametrization.items():
            res = res + "\n" + str(vertex) + ": " + str(param)
        return res

    def _realization_sampling(
        self, n: int, use_tan: bool = False
    ) -> list[dict[Vertex, Point]]:
        """
        Return n realizations for sampled values of the parameter.
        """

        realizations = []
        if not use_tan:
            for i in np.linspace(self._interval[0], self._interval[1], n):
                realizations.append(self.realization(i, numeric=True))
            return realizations

        newinterval = [
            sp.atan(self._interval[0]).evalf(),
            sp.atan(self._interval[1]).evalf(),
        ]
        for i in np.linspace(newinterval[0], newinterval[1], n):
            realizations.append(self.realization(f"tan({i})", numeric=True))
        return realizations

    @staticmethod
    def _normalize_realizations(
        realizations: list[dict[Vertex, Point]], width: int, height: int, spacing: int
    ) -> list[dict[Vertex, Point]]:
        """
        Normalize a given list of realizations
        so they fit exactly to the window with the given dimensions.
        """

        xmin = np.inf
        xmax = -np.inf
        ymin = np.inf
        ymax = -np.inf
        for r in realizations:
            for v, placement in r.items():
                xmin = min(xmin, placement[0])
                xmax = max(xmax, placement[0])
                ymin = min(ymin, placement[1])
                ymax = max(ymax, placement[1])

        xnorm = (width - spacing * 2) / (xmax - xmin)
        ynorm = (height - spacing * 2) / (ymax - ymin)
        norm_factor = min(xnorm, ynorm)

        realizations_normalized = []
        for r in realizations:
            r_norm = {}
            for v, placement in r.items():
                r_norm[v] = [
                    int((placement[0] - xmin) * norm_factor) + spacing,
                    int((placement[1] - ymin) * norm_factor) + spacing,
                ]
            realizations_normalized.append(r_norm)
        return realizations_normalized

    def animate(
        self,
        width: int = 500,
        height: int = 500,
        filename: str = None,
        sampling: int = 50,
        show_labels: bool = True,
        vertex_size: int = 5,
        duration: int = 8,
    ) -> None:
        """
        Animate the parametric motion.

        Parameters
        ----------
        width:
            The width of the window in the svg file.
        height:
            The height of the window in the svg file.
        filename:
            A name used to store the svg. If ``None```, the svg is not saved.
        sampling:
            The number of discrete points or frames used to approximate the motion in the
            animation. A higher value results in a smoother and more accurate
            representation of the motion, while a lower value can speed up rendering
            but may lead to a less precise or jerky animation. This parameter controls
            the resolution of the animation's movement by setting the density of
            sampled data points between keyframes or time steps.
        show_labels:
            If ``True``, the vertices will have a number label.
        vertex_size:
            The size of vertices in the animation.
        duration:
            The duration of one period of the animation in seconds.
        """

        if self._dim != 2:
            raise ValueError("This motion is not in dimension 2!")

        lower, upper = self._interval
        if lower == -np.inf or upper == np.inf:
            realizations = self._realization_sampling(sampling, use_tan=True)
        else:
            realizations = self._realization_sampling(sampling)

        realizations = self._normalize_realizations(realizations, width, height, 15)

        svg = f'<svg width="{width}" height="{height}" version="1.1" '
        svg += 'baseProfile="full" xmlns="http://www.w3.org/2000/svg" '
        svg += 'xmlns:xlink="http://www.w3.org/1999/xlink">\n'
        svg += '<rect width="100%" height="100%" fill="white"/>\n'

        v_to_int = {}
        for i, v in enumerate(self._graph.nodes):
            v_to_int[v] = i
            tmp = """<defs>\n"""

            tmp += f'\t<marker id="vertex{i}" viewBox="0 0 30 30" '
            tmp += f'refX="15" refY="15" markerWidth="{vertex_size}" '
            tmp += f'markerHeight="{vertex_size}">\n'
            tmp += '\t<circle cx="15" cy="15" r="13.5" fill="white" '
            tmp += 'stroke="black" stroke-width="2"/>\n'
            if show_labels:
                tmp += '\t<text x="15" y="22" font-size="22.5" '
                tmp += f'text-anchor="middle" fill="black">\n\t\t{i}\n\t</text>\n'
            tmp += "\t</marker>\n</defs>\n"
            svg = svg + "\n" + tmp

        inital_realization = realizations[0]
        for u, v in self._graph.edges:
            ru = inital_realization[u]
            rv = inital_realization[v]
            path = '<path fill="transparent" stroke="grey" stroke-width="5px" '
            path += f'id="edge{v_to_int[u]}-{v_to_int[v]}" d="M {ru[0]} {ru[1]} '
            path += f'L {rv[0]} {ru[1]}" marker-start="url(#vertex{v_to_int[u]})" '
            path += f'marker-end="url(#vertex{v_to_int[v]})" />'
            svg = svg + "\n" + path
        svg = svg + "\n"

        for u, v in self._graph.edges:
            positions_str = ""
            for r in realizations:
                ru = r[u]
                rv = r[v]
                positions_str += f" M {ru[0]} {ru[1]} L {rv[0]} {rv[1]};"
            animation = f'<animate href="#edge{v_to_int[u]}-{v_to_int[v]}" '
            animation += f'attributeName="d" dur="{duration}s" '
            animation += 'repeatCount="indefinite" calcMode="linear" '
            animation += f'values="{positions_str}"/>'
            svg = svg + "\n" + animation
        svg = svg + "\n</svg>"

        if filename is not None:
            if not filename.endswith(".svg"):
                filename = filename + ".svg"
            with open(filename, "wt") as file:
                file.write(svg)
        return SVG(data=svg)
