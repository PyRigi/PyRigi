"""
This file contains functionality related to finite flexes.
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
                raise KeyError(f"Vertex {v} is not a key of the given realization!")

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

    def _realization_sampling(self, n: int) -> list[dict[Vertex, Point]]:
        """ """

        realizations = []
        for i in np.linspace(self._interval[0], self._interval[1], n):
            realizations.append(self.realization(i, numeric=True))
        return realizations

    @staticmethod
    def _normalize_realizations(
        realizations: list[dict[Vertex, Point]], width: int, height: int, spacing: int
    ) -> list[dict[Vertex, Point]]:
        """
        Normalize given list of realizations
        so they fit exaxtly to the window with the given dimensions.
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
    ) -> None:
        """
        Animation of the parametric motion for parameter
        in the range specified in the constructor.

        Parameters
        ----------
        width:
            Width of the window in the svg file.
        height:
            Height of the window in the svg file.
        filename:
            where to store the svg. If None the svg is not saved.
        sampling:
            Number of discrete points or frames used to approximate the motion in the
            animation. A higher value results in a smoother and more accurate
            representation of the motion, while a lower value can speed up rendering
            but may lead to a less precise or jerky animation. This parameter controls
            the resolution of the animation's movement by setting the density of
            sampled data points between keyframes or time steps.
        """

        # for v, pos in self._parametrization.items():
        #     if len(pos) != 2:
        #         raise ValueError("Motion is not in 2D!")
        #     break

        if self._interval[0] == -sp.oo:  # !!!
            mot = self._parametrization
            for v, placement in mot.items():
                tan_placement = []
                for coord in placement:
                    tan_placement.append(
                        coord.replace(self._parameter, sp.sympify("tan(t)"))
                    )
                mot[v] = tan_placement
            mot = ParametricMotion(self._graph, mot, (-np.pi / 2, np.pi / 2))
            mot.animate(width, height, filename)
            return

        realizations = self._realization_sampling(sampling)
        realizations = self._normalize_realizations(realizations, width, height, 15)

        svg = f"""<svg width="{width}" height="{height}" version="1.1" baseProfile="full" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<rect width="100%" height="100%" fill="white"/>
"""

        v_to_int = {}
        for i, v in enumerate(self._graph.nodes):
            v_to_int[v] = i
            tmp = f"""<defs>
    <marker id="vertex{i}" viewBox="0 0 30 30" refX="15" refY="15" markerWidth="5" markerHeight="5">
    <circle cx="15" cy="15" r="13.5" fill="white" stroke="black" stroke-width="2"/>
    <text x="15" y="22" font-size="22.5" text-anchor="middle" fill="black">
        {i}
    </text>
    </marker>
</defs>
"""
            svg = svg + "\n" + tmp

        inital_realization = realizations[0]
        for u, v in self._graph.edges:
            ru = inital_realization[u]
            rv = inital_realization[v]
            path = f"""<path fill="transparent" stroke="grey" stroke-width="5px" id="edge{v_to_int[u]}-{v_to_int[v]}" d="M {ru[0]} {ru[1]} L {rv[0]} {ru[1]}" marker-start="url(#vertex{v_to_int[u]})"   marker-end="url(#vertex{v_to_int[v]})" />"""
            svg = svg + "\n" + path
        svg = svg + "\n"

        for u, v in self._graph.edges:
            positions_str = ""
            for r in realizations:
                ru = r[u]
                rv = r[v]
                positions_str += f" M {ru[0]} {ru[1]} L {rv[0]} {rv[1]};"
            animation = f"""<animate href="#edge{v_to_int[u]}-{v_to_int[v]}" attributeName="d" dur="8s" repeatCount="indefinite" calcMode="linear" values="{positions_str}"/>"""
            svg = svg + "\n" + animation
        svg = svg + "\n</svg>"

        if filename is not None:
            with open(filename, "wt") as file:
                file.write(svg)
        return SVG(data=svg)
