"""
This module contains functionality related to motions (continuous flexes).
"""

import os
from copy import deepcopy
from typing import Any, Literal
from collections.abc import Callable
import warnings
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from IPython.display import SVG
from matplotlib.animation import FuncAnimation
from sympy import simplify

import pyrigi._input_check as _input_check
from pyrigi import _plot
from pyrigi.data_type import (
    Vertex,
    Point,
    Sequence,
    InfFlex,
    Number,
    DirectedEdge,
    Edge,
)
from pyrigi.graph import Graph
from pyrigi.framework import Framework
from pyrigi.plot_style import PlotStyle, PlotStyle2D, PlotStyle3D
from pyrigi.misc import (
    point_to_vector,
    _normalize_flex,
    _vector_distance_pointwise,
    sympy_expr_to_float,
    is_zero,
)
from pyrigi.warning import NumericalAlgorithmWarning


class Motion(object):
    """
    An abstract class representing a continuous flex of a framework.
    """

    def __init__(self, graph: Graph, dim: int) -> None:
        """
        Create an instance of a graph motion.
        """

        self._graph = graph
        self._dim = dim

    def __str__(self) -> str:
        """Return the string representation"""
        return f"{self.__class__.__name__} of a " + self._graph.__str__()

    def __repr__(self) -> str:
        """Return a representation of the motion."""
        return f"Motion({repr(self.graph)}, {self.dim})"

    @property
    def graph(self) -> Graph:
        """
        Return a copy of the underlying graph.
        """
        return deepcopy(self._graph)

    @property
    def dim(self) -> int:
        """
        Return the dimension of the motion.
        """
        return self._dim

    @staticmethod
    def _normalize_realizations(
        realizations: Sequence[dict[Vertex, Point]],
        x_width: Number,
        y_width: Number,
        z_width: Number = None,
        padding: Number = 0.01,
    ) -> list[dict[Vertex, Point]]:
        """
        Normalize a given list of realizations.

        The returned realizations fit exactly to the window with the given dimensions.

        Parameters
        ----------
        realizations:
            ``Sequence`` of realizations.
        x_width, y_width, z_width:
            Sizes of the underlying canvas.
        padding:
            Whitespace added on the boundaries of the canvas.

        Notes
        -----
        This is done by scaling the ``realizations`` and adding a
        padding so that the animation does not leave the predefined
        canvas.
        """

        xmin = ymin = zmin = np.inf
        xmax = ymax = zmax = -np.inf
        for realization in realizations:
            for v, point in realization.items():
                xmin, xmax = min(xmin, point[0]), max(xmax, point[0])
                ymin, ymax = min(ymin, point[1]), max(ymax, point[1])
                if z_width is not None:
                    zmin, zmax = min(zmin, point[2]), max(zmax, point[2])
        if not is_zero(xmax - xmin, numerical=True, tolerance=1e-6):
            xnorm = (x_width - padding * 2) / (xmax - xmin)
        else:
            xnorm = np.inf
        if not is_zero(ymax - ymin, numerical=True, tolerance=1e-6):
            ynorm = (y_width - padding * 2) / (ymax - ymin)
        else:
            ynorm = np.inf
        if z_width is not None:
            if not is_zero(zmax - zmin, numerical=True, tolerance=1e-6):
                znorm = (z_width - padding * 2) / (zmax - zmin)
            else:
                znorm = np.inf
            norm_factor = min(xnorm, ynorm, znorm)
        else:
            norm_factor = min(xnorm, ynorm)
        if norm_factor == np.inf:
            norm_factor = 1
        realizations_normalized = []
        for realization in realizations:
            realization_normalized = {}
            for v, point in realization.items():
                if z_width is not None:
                    realization_normalized[v] = [
                        (point[0] - xmin) * norm_factor + padding,
                        (point[1] - ymin) * norm_factor + padding,
                        (point[2] - zmin) * norm_factor + padding,
                    ]
                else:
                    realization_normalized[v] = [
                        (point[0] - xmin) * norm_factor + padding,
                        (point[1] - ymin) * norm_factor + padding,
                    ]
            realizations_normalized.append(realization_normalized)
        return realizations_normalized

    def animate3D(
        self,
        realizations: Sequence[dict[Vertex, Point]],
        plot_style: PlotStyle,
        edge_colors_custom: Sequence[Sequence[Edge]] | dict[str, Sequence[Edge]] = None,
        duration: float = 8,
        **kwargs,
    ) -> Any:
        """
        Animate the continuous motion in 3D.

        See :class:`~.PlotStyle3D` for a list of possible visualization keywords.
        Not necessarily all of them apply (e.g. keywords related to infinitesimal
        flexes are ignored).

        Parameters
        ----------
        realizations:
            A list of realization samples describing the motion.
        plot_style:
            An instance of the ``PlotStyle`` class that defines the visual style
            for plotting, see :class:`~.PlotStyle` for more details.
        edge_colors_custom:
            Optional parameter to specify the colors of edges. It can be
            a ``Sequence[Sequence[Edge]]`` to define groups of edges with the same color
            or a ``dict[str, Sequence[Edge]]`` where the keys are color strings and the
            values are lists of edges.
            The omitted edges are given the value ``plot_style.edge_color``.
        duration:
            The duration of one period of the animation in seconds.
        """
        _input_check.dimension_for_algorithm(self._dim, [3], "animate3D")
        if plot_style is None:
            # change some PlotStyle default values to fit 3D plotting better
            plot_style = PlotStyle3D(
                vertex_size=13.5, edge_width=1.5, dpi=150, vertex_labels=False
            )
        else:
            plot_style = PlotStyle3D.from_plot_style(plot_style)

        # Update the plot_style instance with any passed keyword arguments
        plot_style.update(**kwargs)

        delay = int(round(duration / len(realizations) * 1000))  # Set the delay in ms

        fig = plt.figure(dpi=plot_style.dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.grid(False)
        ax.set_axis_off()

        x_coords, y_coords, z_coords = [
            [
                realization[v][0]
                for v in self._graph.nodes
                for realization in realizations
            ]
            for i in range(3)
        ]
        min_val = min(x_coords + y_coords + z_coords) - plot_style.padding
        max_val = max(x_coords + y_coords + z_coords) + plot_style.padding
        aspect_ratio = plot_style.axis_scales
        ax.set_zlim(min_val * aspect_ratio[2], max_val * aspect_ratio[2])
        ax.set_ylim(min_val * aspect_ratio[1], max_val * aspect_ratio[1])
        ax.set_xlim(min_val * aspect_ratio[0], max_val * aspect_ratio[0])

        # Update the plot_style instance with any passed keyword arguments
        edge_color_array, edge_list_ref = _plot.resolve_edge_colors(
            self, plot_style.edge_color, edge_colors_custom
        )

        # Initializing points (vertices) and lines (edges) for display
        (vertices_plot,) = ax.plot(
            [],
            [],
            [],
            plot_style.vertex_shape,
            color=plot_style.vertex_color,
            markersize=plot_style.vertex_size,
        )
        lines = [
            ax.plot(
                [],
                [],
                [],
                c=edge_color_array[i],
                lw=plot_style.edge_width,
                linestyle=plot_style.edge_style,
            )[0]
            for i in range(len(edge_list_ref))
        ]
        annotated_text = []
        if plot_style.vertex_labels:
            annotated_text = [
                ax.text(
                    0,
                    0,
                    0,
                    f"{v}",
                    ha="center",
                    va="center",
                    color=plot_style.font_color,
                    size=plot_style.font_size,
                )
                for v in realizations[0].keys()
            ]

        # Animation initialization function.
        def init():
            vertices_plot.set_data([], [])  # Initial coordinates of vertices
            vertices_plot.set_3d_properties([])  # Initial 3D properties of vertices
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return [vertices_plot] + lines

        def update(frame):
            # Update vertices positions
            vertices_plot.set_data(
                [realizations[frame][v][0] for v in self._graph.nodes],
                [realizations[frame][v][1] for v in self._graph.nodes],
            )
            vertices_plot.set_3d_properties(
                [realizations[frame][v][2] for v in self._graph.nodes]
            )

            # Update the edges
            for i, (u, v) in enumerate(self._graph.edges):
                line = lines[i]
                line.set_data(
                    [realizations[frame][u][0], realizations[frame][v][0]],
                    [realizations[frame][u][1], realizations[frame][v][1]],
                )
                line.set_3d_properties(
                    [realizations[frame][u][2], realizations[frame][v][2]]
                )

            if plot_style.vertex_labels:
                for i in range(len(annotated_text)):
                    annotated_text[i].set_position(
                        (
                            realizations[frame][list(realizations[frame].keys())[i]][0],
                            realizations[frame][list(realizations[frame].keys())[i]][1],
                        )
                    )
                    annotated_text[i].set_3d_properties(
                        realizations[frame][list(realizations[frame].keys())[i]][2]
                    )
            return lines + [vertices_plot] + annotated_text

        ani = FuncAnimation(
            fig,
            update,
            frames=len(realizations),
            interval=delay,
            init_func=init,
            blit=True,
        )

        plt.tight_layout()
        # Checking if we are running from the terminal or from a notebook
        import sys

        if "ipykernel" in sys.modules:
            from IPython.display import HTML

            plt.close()
            return HTML(ani.to_jshtml())
        else:
            if "PYTEST_CURRENT_TEST" in os.environ:
                plt.show(block=False)
            else:
                plt.show()
            return

    def animate2D_plt(
        self,
        realizations: Sequence[dict[Vertex, Point]],
        plot_style: PlotStyle,
        edge_colors_custom: Sequence[Sequence[Edge]] | dict[str, Sequence[Edge]] = None,
        duration: float = 8,
        **kwargs,
    ) -> Any:
        r"""
        Animate the continuous motion in 2D.

        See :class:`~.PlotStyle2D` for a list of possible visualization keywords.
        Not necessarily all of them apply (e.g. keywords related to infinitesimal
        flexes are ignored).
        If the dimension of the motion is 1, then we embed it in $\RR^2$.

        Parameters
        ----------
        realizations:
            A list of realization samples describing the motion.
        plot_style:
            An instance of the ``PlotStyle`` class that defines the visual style
            for plotting, see :class:`~.PlotStyle` for more details.
        edge_colors_custom:
            Optional parameter to specify the colors of edges. It can be
            a ``Sequence[Sequence[Edge]]`` to define groups of edges with the same color
            or a ``dict[str, Sequence[Edge]]`` where the keys are color strings and the
            values are lists of edges.
            The omitted edges are given the value ``plot_style.edge_color``.
        duration:
            The duration of one period of the animation in seconds.
        """
        if self._dim == 1:
            realizations = [
                {v: [pos[0], 0] for v, pos in realization.items()}
                for realization in realizations
            ]
        _input_check.dimension_for_algorithm(self._dim, [1, 2], "animate2D_plt")

        delay = int(round(duration / len(realizations) * 1000))  # Set the delay in ms

        if plot_style is None:
            plot_style = PlotStyle2D(vertex_size=15)
        else:
            plot_style = PlotStyle2D.from_plot_style(plot_style)
        # Update the plot_style instance with any passed keyword arguments
        plot_style.update(**kwargs)

        fig, ax = plt.subplots()
        fig.set_figwidth(plot_style.canvas_width)
        fig.set_figheight(plot_style.canvas_height)
        ax.set_aspect(plot_style.aspect_ratio)
        ax.grid(False)
        ax.set_axis_off()

        x_min, y_min = [
            min(
                [pos[i] for realization in realizations for pos in realization.values()]
            )
            for i in range(2)
        ]
        x_max, y_max = [
            max(
                [pos[i] for realization in realizations for pos in realization.values()]
            )
            for i in range(2)
        ]

        ax.scatter(
            [x_min, x_max],
            [y_min, y_max],
            color="white",
            s=plot_style.vertex_size,
            marker=plot_style.vertex_shape,
        )

        # Update the plot_style instance with any passed keyword arguments
        edge_color_array, edge_list_ref = _plot.resolve_edge_colors(
            self, plot_style.edge_color, edge_colors_custom
        )

        # Initializing points (vertices) and lines (edges) for display
        lines = [
            ax.plot(
                [],
                [],
                c=edge_color_array[i],
                lw=plot_style.edge_width,
                linestyle=plot_style.edge_style,
            )[0]
            for i in range(len(edge_list_ref))
        ]
        (vertices_plot,) = ax.plot(
            [],
            [],
            plot_style.vertex_shape,
            color=plot_style.vertex_color,
            markersize=plot_style.vertex_size,
        )
        annotated_text = []
        if plot_style.vertex_labels:
            annotated_text = [
                ax.text(
                    0,
                    0,
                    f"{v}",
                    ha="center",
                    va="center",
                    color=plot_style.font_color,
                    size=plot_style.font_size,
                )
                for v in realizations[0].keys()
            ]

        # Animation initialization function.
        def init():
            vertices_plot.set_data([], [])  # Initial coordinates of vertices
            for line in lines:
                line.set_data([], [])
            return [vertices_plot] + lines

        def update(frame):
            # Update the edges
            for i, (u, v) in enumerate(self._graph.edges):
                line = lines[i]
                line.set_data(
                    [realizations[frame][u][0], realizations[frame][v][0]],
                    [realizations[frame][u][1], realizations[frame][v][1]],
                )
            # Update vertices positions
            vertices_plot.set_data(
                [realizations[frame][v][0] for v in self._graph.nodes],
                [realizations[frame][v][1] for v in self._graph.nodes],
            )

            if plot_style.vertex_labels:
                for i, (v, pos) in enumerate(realizations[frame].items()):
                    annotated_text[i].set_position(pos)
            return lines + [vertices_plot] + annotated_text

        ani = FuncAnimation(
            fig,
            update,
            frames=len(realizations),
            interval=delay,
            init_func=init,
            blit=True,
        )

        # Checking if we are running from the terminal or from a notebook
        import sys

        if "ipykernel" in sys.modules:
            from IPython.display import HTML

            plt.close()
            return HTML(ani.to_jshtml())
        else:
            if "PYTEST_CURRENT_TEST" in os.environ:
                plt.show(block=False)
            else:
                plt.show()
            return

    def animate2D_svg(
        self,
        realizations: Sequence[dict[Vertex, Point]],
        plot_style: PlotStyle,
        filename: str = None,
        duration: float = 8,
        **kwargs,
    ) -> Any:
        """
        Animate the motion as a ``.svg`` file.

        See :class:`~.PlotStyle2D` for a list of possible visualization keywords.
        Not necessarily all of them apply (e.g. keywords related to infinitesimal
        flexes are ignored).

        Parameters
        ----------
        plot_style:
            An instance of the ``PlotStyle`` class that defines the visual style
            for plotting, see :class:`~.PlotStyle` for more details.
        filename:
            A name used to store the svg. If ``None``, the svg is not saved.
        duration:
            The duration of one period of the animation in seconds.

        Notes
        -----
        Picking the value ``plot_style.vertex_size*5/plot_style.edge_width`` for
        the ``markerWidth`` and ``markerHeight`` ensures that the
        ``plot_style.edge_width`` does not rescale the vertex size
        (seems to be an odd, inherent behavior of `.svg`).
        """
        if self._dim == 1:
            realizations = [
                {v: [pos[0], 0] for v, pos in realization.items()}
                for realization in realizations
            ]
        _input_check.dimension_for_algorithm(self._dim, [1, 2], "animate2D_svg")

        if plot_style is None:
            plot_style = PlotStyle2D(
                vertex_size=7, canvas_width=500, canvas_height=500, edge_width=6
            )
        else:
            plot_style = PlotStyle2D.from_plot_style(plot_style)
        # Update the plot_style instance with any passed keyword arguments
        plot_style.update(**kwargs)

        width = plot_style.canvas_width
        height = plot_style.canvas_height

        _realizations = self._normalize_realizations(
            realizations,
            width,
            height,
            padding=plot_style.vertex_size * 5 / plot_style.edge_width
            + 2 * plot_style.edge_width,
        )

        svg = f'<svg width="{width}" height="{height}" version="1.1" '
        svg += 'baseProfile="full" xmlns="http://www.w3.org/2000/svg" '
        svg += 'xmlns:xlink="http://www.w3.org/1999/xlink">\n'
        svg += '<rect width="%" height="100%" fill="white"/>\n'

        v_to_int = {}
        for i, v in enumerate(self._graph.nodes):
            v_to_int[v] = i
            tmp = """<defs>\n"""
            v_label = str(v)
            tmp += f'\t<marker id="vertex{i}" viewBox="-2 -2 32 32" '
            tmp += 'refX="15" refY="15" '
            tmp += f'markerWidth="{plot_style.vertex_size*5/plot_style.edge_width}" '
            tmp += f'markerHeight="{plot_style.vertex_size*5/plot_style.edge_width}">\n'
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

        for u, v in self._graph.edges:
            ru = _realizations[0][u]
            rv = _realizations[0][v]
            path = f'<path fill="transparent" stroke="{plot_style.edge_color}" '
            path += f'stroke-width="{plot_style.edge_width}px" '
            path += f'id="edge{v_to_int[u]}-{v_to_int[v]}" d="M {ru[0]} {ru[1]} '
            path += f'L {rv[0]} {rv[1]}" marker-start="url(#vertex{v_to_int[u]})" '
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

    def animate(
        self,
        realizations: Sequence[dict[Vertex, Point]],
        plot_style: PlotStyle,
        animation_format: Literal["svg", "matplotlib"] = "svg",
        **kwargs,
    ) -> Any:
        """
        Animate the continuous motion.

        The motion can be animated only if its dimension is less than 3.
        This method calls :meth:`.animate2D` or
        :meth:`.animate3D`.
        For various formatting options, see :class:`.PlotStyle`.

        Parameters
        ----------
        realizations:
            A sequence of realizations of the underlying graph describing the motion.
        plot_style:
            An instance of the ``PlotStyle`` class that defines the visual style
            for plotting, see :class:`~.PlotStyle` for more details.
        animation_format:
            In dimension two, the ``animation_format`` can be set to determine,
            whether the output is in the ``.svg`` format or in the ``matplotlib`` format.
            The `"svg"` method is documented here: :meth:`~.Motion.animate2D_svg`.
            The method for `"matplotlib"` is documented here:
            :meth:`~.Motion.animate2D_plt`.
        """
        if self._dim == 3:
            return self.animate3D(realizations, plot_style=plot_style, **kwargs)
        _input_check.dimension_for_algorithm(self._dim, [1, 2, 3], "animate3D")

        if animation_format == "svg":
            return self.animate2D_svg(realizations, plot_style=plot_style, **kwargs)
        elif animation_format == "matplotlib":
            return self.animate2D_plt(realizations, plot_style=plot_style, **kwargs)
        else:
            raise ValueError(
                "The Literal `animation_format` needs to be "
                + 'either "svg" or "matplotlib".'
            )


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


class ApproximateMotion(Motion):
    """
    Class representing an approximated motion of a framework.

    When constructed, motion samples, i.e., realizations
    approximating a continuous flex of a given framework, are computed.

    Definitions
    -----------
    :prf:ref:`Continuous flex (motion)<def-motion>`

    Parameters
    ----------
    framework:
        A framework whose approximation of a continuous flex is computed.
    steps:
        The amount of retraction steps that are performed. This number is equal to the
        amount of motion samples that are computed.
    step_size:
        The step size of each retraction step. If the output seems too jumpy or instable,
        consider reducing the step size.
    chosen_flex:
        An integer indicating the ``i``-th flex from the list of :meth:`.Framework.inf_flexes`
        for ``i=chosen_flex``.
    tolerance:
        Tolerance for the Newton iteration.
    fixed_pair:
        Two vertices of the underlying graph that are fixed in each realization.
        By default, the first entry is pinned to the origin
        and the second is pinned to the ``x``-axis.
    fixed_direction:
        Vector to which the first direction is fixed. By default, this is given by
        the first and second entry of ``fixed_pair``.
    pinned_vertex:
        If the keyword ``fixed_pair`` is not set, we can use the keyword ``pinned_vertex``
        to pin one of the vertices to the origin instead during the motion.

    Examples
    --------
    >>> from pyrigi import ApproximateMotion
    >>> from pyrigi import graphDB as graphs
    >>> motion = ApproximateMotion.from_graph(
    ...     graphs.Cycle(4),
    ...     {0:(0,0), 1:(1,0), 2:(1,1), 3:(0,1)},
    ...     10
    ... )
    >>> print(motion)
    ApproximateMotion of a Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 3], [1, 2], [2, 3]] with starting configuration
    {0: [0.0, 0.0], 1: [1.0, 0.0], 2: [1.0, 1.0], 3: [0.0, 1.0]},
    10 retraction steps and initial step size 0.1.

    >>> F = Framework(graphs.Cycle(4), {0:(0,0), 1:(1,0), 2:(1,1), 3:(0,1)})
    >>> motion = ApproximateMotion(F, 10)
    >>> print(motion)
    ApproximateMotion of a Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 3], [1, 2], [2, 3]] with starting configuration
    {0: [0.0, 0.0], 1: [1.0, 0.0], 2: [1.0, 1.0], 3: [0.0, 1.0]},
    10 retraction steps and initial step size 0.1.
    """  # noqa: E501

    silence_numerical_alg_warns = False

    def __init__(
        self,
        framework: Framework,
        steps: int,
        step_size: float = 0.1,
        chosen_flex: int = 0,
        tolerance: float = 1e-5,
        fixed_pair: DirectedEdge = None,
        fixed_direction: Sequence[Number] = None,
        pinned_vertex: Vertex = None,
    ) -> None:
        """
        Create an instance of `ApproximateMotion`.
        """
        super().__init__(framework.graph, framework.dim)
        self._warn_numerical_alg(self.__init__)
        self._stress_length = len(framework.stresses())
        self._starting_realization = framework.realization(
            as_points=True, numerical=True
        )
        self._tolerance = tolerance
        self._steps = steps
        self._chosen_flex = chosen_flex
        self._step_size = step_size
        self._current_step_size = step_size
        self._edge_lengths = framework.edge_lengths(numerical=True)
        self._compute_motion_samples(chosen_flex)
        if fixed_pair is not None:
            _input_check.dimension_for_algorithm(
                self._dim, [2], "ApproximateMotion._fix_edge"
            )
            if fixed_direction is None:
                fixed_direction = [1] + [0 for _ in range(self._dim - 1)]
            if len(fixed_direction) != self._dim:
                raise ValueError(
                    "`fixed_direction` does not have the same length as the"
                    + f" motion's dimension, which is {self._dim}."
                )
            self._motion_samples = self._fix_edge(
                self._motion_samples, fixed_pair, fixed_direction
            )
            self._pinned_vertex = None
        elif pinned_vertex is None:
            pin_result = self._pin_origin(self._motion_samples, pinned_vertex)
            self._motion_samples = pin_result[0]
            self._pinned_vertex = pin_result[1]
        else:
            if pinned_vertex not in self._graph.nodes:
                raise ValueError(
                    f"The pinned vertex {pinned_vertex} is not part of the graph"
                )
            self._motion_samples = self._pin_origin(
                self._motion_samples, pinned_vertex
            )[0]
            self._pinned_vertex = pinned_vertex
        self._fixed_pair = fixed_pair
        self._fixed_direction = fixed_direction

    @classmethod
    def from_graph(
        cls,
        graph: Graph,
        realization: dict[Vertex, Point],
        steps: int,
        step_size: float = 0.1,
        chosen_flex: int = 0,
        tolerance: float = 1e-5,
        fixed_pair: DirectedEdge = None,
        fixed_direction: Sequence[Number] = None,
        pinned_vertex: Vertex = None,
    ) -> Motion:
        """
        Create an instance from a graph with a realization.
        """
        if not len(realization) == graph.number_of_nodes():
            raise ValueError(
                "The realization does not contain the correct amount of vertices!"
            )

        realization = {v: sympy_expr_to_float(pos) for v, pos in realization.items()}
        realization_0 = realization[list(realization.keys())[0]]
        for v in graph.nodes:
            if v not in realization:
                raise KeyError(f"Vertex {v} is not a key of the given realization!")
            if len(realization[v]) != len(realization_0):
                raise ValueError(
                    f"The point {realization[v]} in the parametrization"
                    f" corresponding to vertex {v} does not have the right dimension."
                )
        F = Framework(graph, realization)
        return ApproximateMotion(
            F,
            steps,
            step_size=step_size,
            chosen_flex=chosen_flex,
            tolerance=tolerance,
            fixed_pair=fixed_pair,
            fixed_direction=fixed_direction,
            pinned_vertex=pinned_vertex,
        )

    def __str__(self) -> str:
        """Return the string representation."""
        res = super().__str__() + " with starting configuration\n"
        res += str(self.motion_samples[0]) + ",\n"
        res += str(self.steps) + " retraction steps and initial step size "
        res += str(self.step_size) + "."
        return res

    def __repr__(self) -> str:
        """Return a representation of the approximate motion."""
        o_str = f"ApproximateMotion.from_graph({repr(self._graph)}, "
        o_str += f"{self._starting_realization}, {self._steps}, "
        o_str += f"step_size={self._step_size}, chosen_flex={self._chosen_flex}, "
        o_str += f"tolerance={self._tolerance}, fixed_pair={self._fixed_pair}, "
        o_str += f"fixed_direction={self._fixed_direction}, "
        o_str += f"pinned_vertex={self._pinned_vertex})"
        return o_str

    @property
    def edge_lengths(self) -> dict[Edge, Number]:
        """
        Return a copy of the edge lengths.
        """
        return deepcopy(self._edge_lengths)

    @property
    def motion_samples(self) -> list[dict[Vertex, Point]]:
        """
        Return a copy of the motion samples.
        """
        return deepcopy(self._motion_samples)

    @property
    def steps(self) -> int:
        """
        Return the number of steps.
        """
        return self._steps

    @property
    def tolerance(self) -> float:
        """
        Return the tolerance for Newton's method.
        """
        return self._tolerance

    @property
    def starting_realization(self) -> dict[Vertex, Point]:
        """
        Return the starting realization of the motion.
        """
        return deepcopy(self._starting_realization)

    @property
    def step_size(self) -> float:
        """
        Return the step size of the motion.
        """
        return self._step_size

    @property
    def fixed_pair(self) -> DirectedEdge:
        """
        Return the fixed pair of the motion.
        """
        return deepcopy(self._fixed_pair)

    @property
    def chosen_flex(self) -> int:
        """
        Return the chosen flex of the motion.
        """
        return self._chosen_flex

    @property
    def pinned_vertex(self) -> Vertex:
        """
        Return the pinned vertex of the motion.
        """
        return self._pinned_vertex

    @property
    def fixed_direction(self) -> Sequence[Number]:
        """
        Return the vector to which `fixed_pair` is fixed in the motion.
        """
        return deepcopy(self._fixed_direction)

    def fix_vertex(self, vertex: Vertex) -> None:
        """
        Pin ``vertex`` to the origin.

        Parameters
        ----------
        vertex:
            The vertex that is pinned to the origin.
        """
        if vertex is None or vertex not in self.graph.nodes:
            raise ValueError(f"The pinned vertex {vertex} is not part of the graph")
        self._pinned_vertex = vertex
        self._motion_samples = self._pin_origin(self.motion_samples, vertex)[0]

    def fix_pair_of_vertices(
        self, fixed_pair: DirectedEdge, fixed_direction: Sequence[Number] = None
    ) -> None:
        """
        Pin ``fixed_pair`` to the vector given by ``fixed_direction``.

        The default value for ``fixed_direction`` is the first standard unit
        vector ``[1,0,...,0]``.

        Parameters
        ----------
        fixed_pair:
            The pair of vertices that is fixed in the direction ``fixed_direction``.
        fixed_direction:
            A vector to which the  pair of vertices is fixed.
        """
        if (
            len(fixed_pair) == 2
            and fixed_pair[0] in self.graph.nodes
            and fixed_pair[1] in self.graph.nodes
        ):
            _input_check.dimension_for_algorithm(
                self._dim, [2], "ApproximateMotion.fix_edge"
            )
            if fixed_direction is None:
                fixed_direction = [1] + [0 for _ in range(self._dim - 1)]
            if len(fixed_direction) != self._dim:
                raise ValueError(
                    "`fixed_direction` does not have the same length as the"
                    + f" motion's dimension, which is {self._dim}."
                )
        else:
            raise ValueError(
                "`fixed_pair` does not have the correct format or "
                + "has entries that are not contained in the underlying graph."
            )
        self._fixed_pair = fixed_pair
        self._fixed_direction = fixed_direction
        self._motion_samples = self._fix_edge(
            self.motion_samples, fixed_pair, fixed_direction
        )

    @classmethod
    def _warn_numerical_alg(cls, method: Callable) -> None:
        """
        Raise a warning if a numerical algorithm is silently called.

        Parameters
        ----------
        method:
            Reference to the method that is called.
        """
        if not cls.silence_numerical_alg_warns:
            warnings.warn(NumericalAlgorithmWarning(method, class_off=cls))

    def _compute_motion_samples(self, chosen_flex: int) -> None:
        """
        Perform path-tracking to compute the attribute ``_motion_samples``.
        """
        F = Framework(self._graph, self._starting_realization)
        inf_flexes = F.inf_flexes(numerical=True, tolerance=self.tolerance)
        _input_check.integrality_and_range(
            chosen_flex, "chosen_flex", max_val=len(inf_flexes)
        )
        cur_inf_flex = _normalize_flex(
            F._transform_inf_flex_to_pointwise(inf_flexes[chosen_flex]),
            numerical=True,
        )

        cur_sol = self._starting_realization
        self._motion_samples = [cur_sol]
        i = 1
        # To avoid an infinite loop, the step size rescaling is reduced if only too large
        # or too small step sizes are found Its value converges to 1.
        step_size_rescaling = 2
        jump_indicator = [False, False]
        while i < self._steps:
            euler_step, cur_inf_flex = self._euler_step(cur_inf_flex, cur_sol)
            try:
                cur_sol = self._newton_steps(euler_step)
                self._current_step_size = self.step_size
            except RuntimeError:
                self._current_step_size = self._current_step_size / step_size_rescaling
                if self._current_step_size < self.step_size / 10:
                    raise RuntimeError(
                        "Newton's method did not converge. Potentially the "
                        + "given framework is not flexible?"
                    )
                continue
            self._motion_samples += [cur_sol]
            # Reject the step if the step size is not close to what we expect
            if (
                _vector_distance_pointwise(
                    self._motion_samples[-1], self._motion_samples[-2], numerical=True
                )
                > self.step_size * 2
            ):
                self._current_step_size = self._current_step_size / step_size_rescaling
                self._motion_samples.pop()
                jump_indicator[0] = True
                if all(jump_indicator):
                    step_size_rescaling = step_size_rescaling ** (0.75)
                    jump_indicator = [False, False]
                continue
            elif (
                _vector_distance_pointwise(
                    self._motion_samples[-1], self._motion_samples[-2], numerical=True
                )
                < self.step_size / 2
            ):
                self._current_step_size = self._current_step_size * step_size_rescaling
                self._motion_samples.pop()
                jump_indicator[1] = True
                if all(jump_indicator):
                    step_size_rescaling = step_size_rescaling ** (0.75)
                    jump_indicator = [False, False]
                continue
            jump_indicator = [False, False]
            i = i + 1

    def _pin_origin(
        self, realizations: Sequence[dict[Vertex, Point]], pinned_vertex: Vertex = None
    ) -> tuple[list[dict[Vertex, Point]], Vertex]:
        """
        Pin a vertex to the origin.

        Parameters
        ----------
        realizations:
            A list of realization samples describing the motion.
        pinned_vertex:
            Determines the vertex which is pinned to the origin.
        """
        _realizations = []
        if pinned_vertex is None:
            pinned_vertex = self._graph.vertex_list()[0]
        for realization in realizations:
            if pinned_vertex not in realization.keys():
                raise ValueError(
                    "The `pinned_vertex` does not have a value in the provided motion."
                )

            # Translate the realization to the origin
            _realization = {
                v: [pos[i] - realization[pinned_vertex][i] for i in range(len(pos))]
                for v, pos in realization.items()
            }
            _realizations.append(_realization)
        return _realizations, pinned_vertex

    @staticmethod
    def _fix_edge(
        realizations: Sequence[dict[Vertex, Point]],
        fixed_pair: DirectedEdge,
        fixed_direction: Sequence[Number],
    ) -> list[dict[Vertex, Point]]:
        """
        Fix the two vertices in ``fixed_pair`` for every entry of ``realizations``.

        Parameters
        ----------
        realizations:
            A list of realization samples describing the motion.
        fixed_pair:
            Two vertices of the underlying graph that should not move during
            the animation. By default, the first entry is pinned to the origin
            and the second is pinned to the `x`-axis.
        fixed_direction:
            Vector to which the first direction is fixed. By default, this is given by
            the first and second entry.
        """
        if len(fixed_pair) != 2:
            raise TypeError("The length of `fixed_pair` is not 2.")
        (v1, v2) = (fixed_pair[0], fixed_pair[1])
        if not (v1 in realizations[0] and v2 in realizations[0]):
            raise ValueError(
                "The vertices of the edge {realizations} are not part of the graph."
            )

        # Translate the realization to the origin
        _realizations = [
            {
                v: [pos[i] - realization[v1][i] for i in range(len(pos))]
                for v, pos in realization.items()
            }
            for realization in realizations
        ]
        if fixed_direction is None:
            fixed_direction = [
                x - y for x, y in zip(_realizations[0][v1], _realizations[0][v2])
            ]
            if is_zero(np.linalg.norm(fixed_direction), numerical=True, tolerance=1e-6):
                warnings.warn(
                    f"The entries of the edge {fixed_pair} are too close to each "
                    + "other. Thus, `fixed_direction=(1,0)` is chosen instead."
                )
                fixed_direction = [1] + [
                    0
                    for _ in range(
                        len(_realizations[list(_realizations.keys())[0]]) - 1
                    )
                ]
            else:
                fixed_direction = [
                    coord / np.linalg.norm(fixed_direction) for coord in fixed_direction
                ]

        output_realizations = []
        for realization in _realizations:
            if any([len(pos) not in [2, 3] for pos in realization.values()]):
                raise ValueError(
                    "This method ``_fix_edge`` is not implemented for "
                    + "dimensions other than 2 or 3."
                )
            if (
                len(fixed_direction) not in [2, 3]
                or np.linalg.norm(fixed_direction) != 1
            ):
                raise ValueError("`fixed_direction` does not have the correct format.")

            # Compute the signed angle `theta` between the `fixed_direction` and the
            # vector `realization[v2]`
            theta = np.arctan2(
                [fixed_direction[1], realization[v2][1]],
                [fixed_direction[0], realization[v2][0]],
            )[1]

            rotation_matrix = np.array(
                [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
            )
            # Rotate the realization to the `fixed_direction`.
            _realization = {
                v: np.dot(rotation_matrix, pos) for v, pos in realization.items()
            }
            output_realizations.append(_realization)
        return output_realizations

    def animate(
        self,
        **kwargs,
    ) -> Any:
        """
        Animate the approximate motion.

        See the parent method :meth:`~.Motion.animate` for the list of possible keywords.
        """
        realizations = self._motion_samples
        return super().animate(
            realizations,
            None,
            **kwargs,
        )

    def _euler_step(
        self,
        old_inf_flex: InfFlex,
        realization: dict[Vertex, Point],
    ) -> tuple[dict[Vertex, Point], InfFlex]:
        """
        Compute a single Euler step.

        This method returns the resulting configuration and the infinitesimal flex
        that was used in the computation as a tuple.

        Notes
        -----
        Choose the (normalized) infinitesimal flex with the smallest distance from the
        previous infinitesimal flex ``old_inf_flex``. This is given by computing the
        Moore-Penrose pseudoinverse.

        Suggested Improvements
        ----------------------
        * Add vector transport to ``old_inf_flex`` to more accurately compare the vectors.
        * Search the space of `inf_flexes` using a Least Squares approach rather than
        just searching a basis
        """
        F = Framework(self._graph, realization)

        inf_flex_space = np.vstack(
            F.inf_flexes(numerical=True, tolerance=self.tolerance)
        )
        old_inf_flex_matrix = np.reshape(
            sum([list(pos) for pos in old_inf_flex.values()], []), (-1, 1)
        )
        flex_coefficients = np.dot(
            np.linalg.pinv(inf_flex_space).transpose(), old_inf_flex_matrix
        )
        predicted_inf_flex = sum(
            np.dot(inf_flex_space.transpose(), flex_coefficients).tolist(), []
        )
        predicted_inf_flex = _normalize_flex(
            F._transform_inf_flex_to_pointwise(predicted_inf_flex), numerical=True
        )
        realization = self._motion_samples[-1]
        return {
            v: tuple(
                [
                    pos[i] + self._current_step_size * predicted_inf_flex[v][i]
                    for i in range(len(realization[v]))
                ]
            )
            for v, pos in realization.items()
        }, predicted_inf_flex

    def _newton_steps(self, realization: dict[Vertex, Point]) -> dict[Vertex, Point]:
        """
        Compute a sequence of Newton steps to return to the constraint variety.

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
                np.abs(length - self._edge_lengths[e])
                for e, length in F.edge_lengths(numerical=True).items()
            ]
        )
        damping = 1e-1
        rand_mat = np.random.rand(
            F._graph.number_of_edges() - self._stress_length, F._graph.number_of_edges()
        )
        while not cur_error < self.tolerance:
            rigidity_matrix = np.array(F.rigidity_matrix()).astype(np.float64)
            equations = [
                np.linalg.norm(
                    [
                        x - y
                        for x, y in zip(
                            cur_sol[(self._dim * e[0]) : (self._dim * (e[0] + 1))],
                            cur_sol[(self._dim * e[1]) : (self._dim * (e[1] + 1))],
                        )
                    ]
                )
                - length
                for e, length in self._edge_lengths.items()
            ]

            if self._stress_length > 0:
                equations = np.dot(rand_mat, equations)
                rigidity_matrix = np.dot(rand_mat, rigidity_matrix)
            newton_step = np.dot(np.linalg.pinv(rigidity_matrix), equations)

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
                    np.abs(length - self._edge_lengths[e])
                    for e, length in F.edge_lengths(numerical=True).items()
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
