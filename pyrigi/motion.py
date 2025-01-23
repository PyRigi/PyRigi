"""
This module contains functionality related to motions (continuous flexes).
"""

from pyrigi.graph import Graph
from pyrigi.framework import Framework
from pyrigi.data_type import Vertex, Point, Sequence, InfFlex, Number, DirectedEdge
from pyrigi.plot_style import PlotStyle, PlotStyle2D
from sympy import simplify
from pyrigi.misc import point_to_vector, normalize_flex, vector_distance_pointwise
import numpy as np
import sympy as sp
from IPython.display import SVG
from typing import Any
from copy import deepcopy
from warnings import warn


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

    def _fix_origin(
        self,
        realizations: Sequence[dict[Vertex, Point]],
    ) -> list[dict[Vertex, Point]]:
        """
        Pin the first vertex to the origin.

        Parameters
        ----------
        realizations:
            A list of realization samples describing the motion.
        """
        _realizations = []
        u = self._graph.vertex_list()[0]
        for r in realizations:
            # Translate the realization to the origin
            _r = {v: [p[i] - r[u][i] for i in range(len(p))] for v, p in r.items()}
            _realizations.append(_r)
        return _realizations

    @staticmethod
    def _fix_edge(
        realizations: Sequence[dict[Vertex, Point]],
        fixed_edge: DirectedEdge,
        fixed_direction: Sequence[Number],
    ) -> list[dict[Vertex, Point]]:
        """
        Fix the edge ``fixed_edge`` for every entry of ``realizations``.

        Parameters
        ----------
        realizations:
            A list of realization samples describing the motion.
        fixed_edge:
            The edge of the underlying graph that should not move during
            the animation. By default, the first entry is pinned to the origin
            and the second is pinned to the `x`-axis.
        fixed_direction:
            Vector to which the first direction is fixed. By default, this is given by
            the first and second entry.
        """
        if len(fixed_edge) != 2:
            raise TypeError("The length of `fixed_edge` is not 2.")
        (v1, v2) = (fixed_edge[0], fixed_edge[1])
        if not (v1 in realizations[0] and v2 in realizations[0]):
            raise ValueError(
                "The vertices of the edge {realizations} are not part of the graph."
            )

        # Translate the realization to the origin
        _realizations = [
            {v: [p[i] - r[v1][i] for i in range(len(p))] for v, p in r.items()}
            for r in realizations
        ]
        if fixed_direction is None:
            fixed_direction = [
                q - p for p, q in zip(_realizations[0][v1], _realizations[0][v2])
            ]
            if np.isclose(np.linalg.norm(fixed_direction), 0, rtol=1e-6):
                warn(
                    f"The entries of the edge {fixed_edge} are too close to each "
                    + "other. Thus, `fixed_direction=(1,0)` is chosen instead."
                )
                fixed_direction = (1, 0)
            else:
                fixed_direction = [
                    p / np.linalg.norm(fixed_direction) for p in fixed_direction
                ]

        output_realizations = []
        for r in _realizations:
            if any([len(p) != 2 for p in r.values()]):
                raise ValueError(
                    "This method is not implemented for dimensions other than 2."
                )
            if len(fixed_direction) != 2 or np.linalg.norm(fixed_direction) != 1:
                raise ValueError("`fixed_direction` does not have the correct format.")

            v_dist = np.linalg.norm(r[v2])
            theta = np.arccos(
                np.dot([v_dist * t for t in fixed_direction], r[v2]) / v_dist**2
            )

            if r[v2][0] * r[v2][1] < 0:
                rotation_matrix = np.array(
                    [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
                )
            else:
                rotation_matrix = np.array(
                    [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
                )
            # Rotate the realization to the `fixed_direction`.
            r = {v: np.dot(rotation_matrix, p) for v, p in r.items()}
            output_realizations.append(r)
        return output_realizations

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
                    (placement[0] - xmin) * norm_factor + spacing,
                    (placement[1] - ymin) * norm_factor + spacing,
                ]
            realizations_normalized.append(r_norm)
        return realizations_normalized

    def animate(
        self,
        realizations: Sequence[dict[Vertex, Point]],
        plot_style: PlotStyle,
        fixed_edge: DirectedEdge = None,
        fixed_direction: Sequence[Number] = [1, 0],
        fix_origin: bool = True,
        filename: str = None,
        duration: int = 8,
        **kwargs,
    ) -> Any:
        """
        Animate the continuous motion.

        See :class:`~.PlotStyle2D` for a list of possible visualization keywords.
        Not necessarily all of them apply (e.g. keywords related to infinitesimal
        flexes are ignored).

        Parameters
        ----------
        realizations:
            A list of realization samples describing the motion.
        plot_style:
            An instance of the ``PlotStyle`` class that defines the visual style
            for plotting, see :class:`~.PlotStyle` for more details.
        fixed_edge:
            The edge of the underlying graph that should not move during
            the animation. The default is that only the origin is pinned and that
            the framework can rotate freely.
        fixed_direction:
            Vector to which the first direction is fixed. By default, it is the
            vector from ``fixed_edge[0]`` to ``fixed_edge[1]``.
        fix_origin:
            A boolean deciding whether the origin is fixed. This only has an effect
            if fixed_edge=None``.
        filename:
            A name used to store the svg. If ``None``, the svg is not saved.
        duration:
            The duration of one period of the animation in seconds.
        """
        if self._dim != 2:
            raise ValueError("Animations are supported only for motions in 2D.")

        if plot_style is None:
            plot_style = PlotStyle2D(
                vertex_size=6, canvas_width=500, canvas_height=500, edge_width=5
            )
        else:
            plot_style = PlotStyle2D.from_plot_style(plot_style)
        # Update the plot_style instance with any passed keyword arguments
        plot_style.update(**kwargs)

        width = plot_style.canvas_width
        height = plot_style.canvas_height

        if fixed_edge is not None:
            _realizations = self._fix_edge(realizations, fixed_edge, fixed_direction)
        elif fix_origin:
            _realizations = self._fix_origin(realizations)
        else:
            _realizations = realizations
        _realizations = self._normalize_realizations(_realizations, width, height, 15)

        svg = f'<svg width="{width}" height="{height}" version="1.1" '
        svg += 'baseProfile="full" xmlns="http://www.w3.org/2000/svg" '
        svg += 'xmlns:xlink="http://www.w3.org/1999/xlink">\n'
        svg += '<rect width="100%" height="100%" fill="white"/>\n'

        v_to_int = {}
        for i, v in enumerate(self._graph.nodes):
            v_to_int[v] = i
            tmp = """<defs>\n"""
            v_label = str(v)
            tmp += f'\t<marker id="vertex{i}" viewBox="0 0 30 30" '
            tmp += f'refX="15" refY="15" markerWidth="{plot_style.vertex_size}" '
            tmp += f'markerHeight="{plot_style.vertex_size}">\n'
            tmp += (
                f'\t<circle cx="15" cy="15" r="13.5" fill="{plot_style.vertex_color}" '
            )
            tmp += 'stroke="white" stroke-width="0"/>\n'
            if plot_style.vertex_labels:
                tmp += (
                    '\t<text x="15" y="22" font-size="22.5" font-family="DejaVuSans" '
                )
                tmp += f'text-anchor="middle" fill="{plot_style.font_color}">'
                tmp += f"\n\t\t{v_label}\n\t</text>\n"
            tmp += "\t</marker>\n</defs>\n"
            svg = svg + "\n" + tmp

        inital_realization = _realizations[0]
        for u, v in self._graph.edges:
            ru = inital_realization[u]
            rv = inital_realization[v]
            path = f'<path fill="transparent" stroke="{plot_style.edge_color}" '
            path += f'stroke-width="{plot_style.edge_width}px" '
            path += f'id="edge{v_to_int[u]}-{v_to_int[v]}" d="M {ru[0]} {ru[1]} '
            path += f'L {rv[0]} {ru[1]}" marker-start="url(#vertex{v_to_int[u]})" '
            path += f'marker-end="url(#vertex{v_to_int[v]})" />'
            svg = svg + "\n" + path
        svg = svg + "\n"

        for u, v in self._graph.edges:
            positions_str = ""
            for r in _realizations:
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
        Return a specific realization for the given ``value`` of the parameter.

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
                realizations.append(self.realization(i, numerical=True))
            return realizations

        newinterval = [
            sp.atan(self._interval[0]).evalf(),
            sp.atan(self._interval[1]).evalf(),
        ]
        for i in np.linspace(newinterval[0], newinterval[1], n):
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
            the resolution of the animation's movement by setting the density of
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
            fix_origin=False,
            **kwargs,
        )


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
        starting_realization: dict[Vertex, Point],
        steps: int,
        step_size: float = 0.1,
        chosen_flex: int = 0,
        turning_threshold: float = 1.5,
    ) -> None:
        """
        Creates an instance.
        """
        super().__init__(graph)

        if not len(starting_realization) == graph.number_of_nodes():
            raise ValueError(
                "The realization does not contain the correct amount of vertices!"
            )

        self._starting_realization = {
            v: tuple([float(sp.sympify(pt).evalf(15)) for pt in p])
            for v, p in starting_realization.items()
        }

        self._dim = len(list(self._starting_realization.values())[0])
        for v in graph.nodes:
            if v not in starting_realization:
                raise KeyError(f"Vertex {v} is not a key of the given realization!")
            if len(self._starting_realization[v]) != self._dim:
                raise ValueError(
                    f"The point {self._starting_realization[v]} in the parametrization"
                    f" corresponding to vertex {v} does not have the right dimension."
                )

        self.steps = steps
        self.chosen_flex = chosen_flex
        self.step_size = step_size
        self._current_step_size = step_size
        F = Framework(self._graph, self._starting_realization)
        self.edge_lengths = F.edge_lengths(numerical=True)
        self._compute_motion_samples(chosen_flex, turning_threshold)

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

    def _compute_motion_samples(
        self, chosen_flex: int, turning_threshold: float
    ) -> None:
        """
        Perform path-tracking to compute the attribute `motion_samples`.
        """
        F = Framework(self._graph, self._starting_realization)
        cur_inf_flex = normalize_flex(
            F._transform_inf_flex_to_pointwise(F.inf_flexes()[chosen_flex]),
            numerical=True,
        )

        cur_sol = self._starting_realization
        self.motion_samples = [cur_sol]
        i = 1
        # To avoid an infinite loop, the step size rescaling is reduced if only too large
        # or too small step sizes are found Its value converges to 1.
        step_size_rescaling = 2
        jump_indicator = [False, False]
        while i < self.steps:
            euler_step, cur_inf_flex = self._euler_step(
                cur_inf_flex, cur_sol, turning_threshold
            )
            cur_sol = self._newton_steps(euler_step)
            self.motion_samples += [cur_sol]
            # Reject the step if the step size is not close to what we expect
            if (
                vector_distance_pointwise(
                    self.motion_samples[-1], self.motion_samples[-2], numerical=True
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
                vector_distance_pointwise(
                    self.motion_samples[-1], self.motion_samples[-2], numerical=True
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

    def animate(
        self,
        **kwargs,
    ) -> Any:
        """
        Animate the approximate motion.

        See the parent method :meth:`~.Motion.animate` for a list of possible keywords.
        """
        realizations = self.motion_samples
        return super().animate(
            realizations,
            None,
            fix_origin=True,
            **kwargs,
        )

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
        reflected_inf_flex = {v: [-pt for pt in p] for v, p in inf_flex.items()}

        if vector_distance_pointwise(
            inf_flex, old_inf_flex, numerical=True
        ) > turning_threshold * vector_distance_pointwise(
            reflected_inf_flex,
            old_inf_flex,
            numerical=True,
        ):
            inf_flex = reflected_inf_flex
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
        """
        Computes a sequence of Newton steps to return to the constraint variety.

        Notes
        -----
        There are more robust implementations of Newton's method (using damped schemes and
        preconditioning), but that would blow method out of the current scope. Here, a
        naive damping scheme is implemented (so that the method is actually guaranteed
        to converge), but this means that in the case where dim(stresses=flexes), the
        damping goes to 0. MH has tested this so-called "damped Gauß-Newton scheme"
        extensively in two other packages. If the equations are randomized first, there
        are convergence and smoothness guarantees from numerical algebraic geometry,
        but that is currently out of the scope of the `ApproximateMotion` class.

        Suggested Improvements
        ----------------------
        Randomize the bar-length equations.
        """
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
            cur_sol = [
                cur_sol[i] - damping * newton_step[i] for i in range(len(cur_sol))
            ]
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
            # If the damping becomes too small, raise an exception.

            if damping < 1e-10:
                raise RuntimeError("Newton's method did not converge.")
            prev_error = cur_error

        return {
            v: tuple(cur_sol[(self._dim * i) : (self._dim * (i + 1))])
            for i, v in enumerate(self._graph.vertex_list())
        }

    def __str__(self) -> str:
        res = super().__str__() + " with starting configuration\n"
        res += str(self.motion_samples[0]) + ",\n"
        res += str(self.steps) + " retraction steps and initial step size "
        res += str(self.step_size) + "."
        return res
