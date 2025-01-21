"""
This module contains functionality related to motions (continuous flexes).
"""

from pyrigi.graph import Graph
from pyrigi.framework import Framework
from pyrigi.data_type import Vertex, Point, Sequence, InfFlex, Number
from sympy import simplify
from pyrigi.misc import point_to_vector, normalize_flex
import numpy as np
import sympy as sp
from IPython.display import SVG
from typing import Any
from copy import deepcopy


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
    
    def graph(self) -> Graph:
        """
        Return a copy of the underlying graph.
        """
        return deepcopy(self._graph)

    @staticmethod
    def _normalize_realizations(
        realizations: Sequence[dict[Vertex, Point]],
        width: int,
        height: int,
        spacing: int,
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

    def _animate(
        self,
        realizations: Sequence[dict[Vertex, Point]],
        width: int = 500,
        height: int = 500,
        filename: str = None,
        show_labels: bool = True,
        vertex_size: int = 5,
        duration: int = 8,
    ) -> Any:
        """
        Animate the continuous motion.

        Parameters
        ----------
        realizations:
            A list of realization samples describing the motion.
        width:
            The width of the window in the saved svg file.
        height:
            The height of the window in the saved svg file.
        filename:
            A name used to store the svg. If ``None```, the svg is not saved.
        show_labels:
            If ``True``, the vertices will have a number label.
        vertex_size:
            The size of vertices in the animation.
        duration:
            The duration of one period of the animation in seconds.
        """
        if self._dim != 2:
            raise ValueError("Animations are supported only for motions in 2D.")
        if not (
            self.__class__.__name__ == "ApproximateMotion"
            or self.__class__.__name__ == "ParametricMotion"
        ):
            raise AttributeError(
                "The method `animate` is not yet implemented for "
                + "the class {self.__class__.__name__}"
            )

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
            raise ValueError(
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
        for _, position in self._parametrization.items():
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

    def realization(self, value: Number, numerical: bool = False) -> dict[Vertex:Point]:
        """
        Return specific realization for the given ``value`` of the parameter.

        Parameters
        ----------
        value:
            The parameter of the deformation path is substituted by ``value``.
        numerical:
            Boolean determining whether the sympy expressions are supposed to be
            evaluated (``True``) or not (``False``).
        """

        realization = {}
        for v in self._graph.nodes:
            if numerical:
                _value = sp.sympify(value).evalf()
                placement = (
                    self._parametrization[v]
                    .subs({self._parameter: float(_value)})
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

    def animate(self, sampling: int = 50, **kwargs) -> Any:
        """
        Animate the parametric motion.

        Parameters
        ----------
        sampling:
            The number of discrete points or frames used to approximate the motion in the
            animation. A higher value results in a smoother and more accurate
            representation of the motion, while a lower value can speed up rendering
            but may lead to a less precise or jerky animation. This parameter controls
            the resolution of the animation's movement by setting the density of
            sampled data points between keyframes or time steps.
        width:
            The width of the window in the saved svg file.
        height:
            The height of the window in the saved svg file.
        filename:
            A name used to store the svg. If ``None```, the svg is not saved.
        show_labels:
            If ``True``, the vertices will have a number label.
        vertex_size:
            The size of vertices in the animation.
        duration:
            The duration of one period of the animation in seconds.
        """
        lower, upper = self._interval
        if lower == -np.inf or upper == np.inf:
            realizations = self._realization_sampling(sampling, use_tan=True)
        else:
            realizations = self._realization_sampling(sampling)
        return self._animate(realizations, **kwargs)


class ApproximateMotion(Motion):
    """
    Class representing an approximated motion of a framework.

    Definitions
    -----------
    :prf:ref:`Continuous flex (motion)<def-motion>`

    Parameters
    ----------
    graph:
    steps:
        The amount of retraction steps that are performed. This number is equal to the
        amount of ``motion_samples`` that are computed.
    step_size:
        The step size of each retraction step. If the output seems too jumpy or instable,
        consider reducing the step size.
    chosen_flex:
        An integer indicating the ``i``-th flex from the list of :meth:`Framework.inf_flexes`
        for ``i=chosen_flex``.
    turning_threshold:
        Determines when the reflected infinitesimal flex at position ``chosen_flex``
        is taken instead of the regular one. To decide this, the distance from the
        previous Euler step is calculated using the Euclidean norm. If the current
        distance is at least ``turning_threshold`` times as large as the distance
        of the negative infinitesimal flex, then the latter one is chosen instead.
        If instead the animation is too slow, consider increasing this value.

    Attributes
    ----------
    edge_lengths:
        The edge lengths that ought to be preserved.
    motion_samples:
        A list of numerical configurations on the configuration space.

    Examples
    --------
    >>> from pyrigi.motion import ApproximateMotion
    >>> from pyrigi import graphDB as graphs
    >>> motion = ApproximateMotion(
    ...     graphs.Cycle(4),
    ...     {0:(0,0), 1:(1,0), 2:(1,1), 3:(0,1)},
    ...     10
    ... )
    >>> motion
    ApproximateMotion of a Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 3], [1, 2], [2, 3]] with starting configuration
    {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (0.0, 1.0)},
    10 retraction steps and initial step size 0.1.
    """  # noqa: E501

    def __init__(
        self,
        graph: Graph,
        starting_configuration: dict[Vertex, Point],
        steps: int,
        step_size: float = 0.1,
        chosen_flex: int = 0,
        turning_threshold: float = 1.5,
    ) -> None:
        """
        Creates an instance.
        """
        super().__init__(graph)

        if not len(starting_configuration) == graph.number_of_nodes():
            raise ValueError(
                "The realization does not contain the correct amount of vertices!"
            )

        self._starting_configuration = {
            v: tuple([float(sp.sympify(pt).evalf(15)) for pt in p])
            for v, p in starting_configuration.items()
        }
        p0 = self._starting_configuration[graph.vertex_list()[0]]
        # Translate to the origin
        self._starting_configuration = {
            v: tuple([pt[i] - p0[i] for i in range(len(pt))])
            for v, pt in self._starting_configuration.items()
        }
        self._dim = len(list(self._starting_configuration.values())[0])
        for v in graph.nodes:
            if v not in starting_configuration:
                raise KeyError(f"Vertex {v} is not a key of the given realization!")
            if len(self._starting_configuration[v]) != self._dim:
                raise ValueError(
                    f"The point {self._starting_configuration[v]} in the parametrization"
                    f" corresponding to vertex {v} does not have the right dimension."
                )

        self.motion_samples = [self._starting_configuration]
        cur_sol = self._starting_configuration
        self.steps = steps
        self.chosen_flex = chosen_flex
        self.step_size = step_size
        self._current_step_size = step_size
        F = Framework(graph, self._starting_configuration)
        self.edge_lengths = F.edge_lengths(numerical=True)
        cur_inf_flex = normalize_flex(
            F._transform_inf_flex_to_pointwise(F.inf_flexes()[chosen_flex]),
            numerical=True,
        )

        i = 1
        # To avoid an infinite loop, the step size rescaling is reduced if only too large
        # or too small step sizes are found Its value converges to 1.
        step_size_rescaling = 2
        jump_indicator = [False, False]
        while i < steps:
            euler_step, cur_inf_flex = self._euler_step(
                cur_inf_flex, cur_sol, turning_threshold
            )
            cur_sol = self._newton_steps(euler_step)
            self.motion_samples += [cur_sol]
            # Reject the step if the step size is not close to what we expect
            if (
                np.linalg.norm(
                    [
                        p1 - p2
                        for v in graph.nodes
                        for p1, p2 in zip(
                            self.motion_samples[-1][v],
                            self.motion_samples[-2][v],
                        )
                    ]
                )
                > self.step_size * 2
            ):
                self._current_step_size = self._current_step_size / step_size_rescaling
                self.motion_samples.pop()
                jump_indicator[0] = True
                if all(jump_indicator):
                    step_size_rescaling = step_size_rescaling ** (0.75)
                    jump_indicator = [False, False]
                continue
            elif (
                np.linalg.norm(
                    [
                        p1 - p2
                        for v in graph.nodes
                        for p1, p2 in zip(
                            self.motion_samples[-1][v],
                            self.motion_samples[-2][v],
                        )
                    ]
                )
                < self.step_size / 2
            ):
                self._current_step_size = self._current_step_size * step_size_rescaling
                self.motion_samples.pop()
                jump_indicator[1] = True
                if all(jump_indicator):
                    step_size_rescaling = step_size_rescaling ** (0.75)
                    jump_indicator = [False, False]
                continue
            jump_indicator = [False, False]
            i = i + 1

    @classmethod
    def from_framework(
        cls, F: Framework, steps: int, step_size: float = 0.1, chosen_flex: int = 0
    ):
        """
        Instantiates an ``ApproximateMotion`` from a ``Framework``.
        """
        return ApproximateMotion(
            F.graph(),
            F.realization(as_points=True, numerical=True),
            steps,
            step_size,
            chosen_flex,
        )

    def animate(self, **kwargs) -> Any:
        """
        Animate the approximate motion.

        Parameters
        ----------
        width:
            The width of the window in the saved svg file.
        height:
            The height of the window in the saved svg file.
        filename:
            A name used to store the svg. If ``None```, the svg is not saved.
        show_labels:
            If ``True``, the vertices will have a number label.
        vertex_size:
            The size of vertices in the animation.
        duration:
            The duration of one period of the animation in seconds.
        """
        realizations = self.motion_samples
        return self._animate(realizations, **kwargs)

    def _euler_step(
        self,
        old_inf_flex: InfFlex,
        realization: dict[Vertex, Point],
        turning_threshold: float,
    ) -> tuple[dict[Vertex, Point], InfFlex]:
        """
        Computes a single Euler step.

        This method returns the resulting configuration and the infinitesimal flex
        that was used in the computation as a tuple.
        """
        F = Framework(self._graph, realization)
        inf_flex = normalize_flex(
            F._transform_inf_flex_to_pointwise(F.inf_flexes()[self.chosen_flex]),
            numerical=True,
        )
        if np.linalg.norm(
            [
                q - w
                for v in inf_flex.keys()
                for q, w in zip(inf_flex[v], old_inf_flex[v])
            ]
        ) > turning_threshold * np.linalg.norm(
            [
                -q - w
                for v in inf_flex.keys()
                for q, w in zip(inf_flex[v], old_inf_flex[v])
            ]
        ):
            inf_flex = {v: [-pt for pt in p] for v, p in inf_flex.items()}
        point = self.motion_samples[-1]
        return {
            v: tuple(
                [
                    p[i] + self._current_step_size * inf_flex[v][i]
                    for i in range(len(point[v]))
                ]
            )
            for v, p in point.items()
        }, inf_flex

    def _newton_steps(self, realization: dict[Vertex, Point]) -> dict[Vertex, Point]:
        F = Framework(self._graph, realization)
        cur_sol = np.array(
            sum([list(realization[v]) for v in self._graph.vertex_list()], [])
        )
        cur_error = prev_error = sum(
            [
                np.abs(L - self.edge_lengths[e])
                for e, L in F.edge_lengths(numerical=True).items()
            ]
        )
        damping = 5e-2
        while not cur_error < 1e-4:
            mat = np.array(F.rigidity_matrix()).astype(np.float64)
            equations = [
                np.linalg.norm(
                    [
                        v - w
                        for v, w in zip(
                            cur_sol[(self._dim * e[0]) : (self._dim * (e[0] + 1))],
                            cur_sol[(self._dim * e[1]) : (self._dim * (e[1] + 1))],
                        )
                    ]
                )
                - self.edge_lengths[e]
                for e in self.edge_lengths.keys()
            ]
            newton_step = np.dot(np.linalg.pinv(mat), equations)
            new_sol = [
                cur_sol[i] - damping * newton_step[i] for i in range(len(cur_sol))
            ]
            cur_sol = sum(
                [
                    [new_sol[i + j] - new_sol[j] for j in range(self._dim)]
                    for i in range(0, len(new_sol), self._dim)
                ],
                [],
            )
            F = Framework(
                self._graph,
                {
                    i: [cur_sol[(self._dim * i) : (self._dim * (i + 1))]]
                    for i in range(len(realization.keys()))
                },
            )
            cur_error = sum(
                [
                    np.abs(L - self.edge_lengths[e])
                    for e, L in F.edge_lengths(numerical=True).items()
                ]
            )
            if cur_error <= prev_error:
                damping = damping * 1.25
            else:
                damping = damping / 2
            prev_error = cur_error

        return {
            self._graph.vertex_list()[i]: tuple(
                cur_sol[(self._dim * i) : (self._dim * (i + 1))]
            )
            for i in range(self._graph.number_of_nodes())
        }

    def __str__(self) -> str:
        res = super().__str__() + " with starting configuration\n"
        res += str(self.motion_samples[0]) + ",\n"
        res += str(self.steps) + " retraction steps and initial step size "
        res += str(self.step_size) + "."
        return res
