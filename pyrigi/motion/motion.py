"""
This module contains functionality related to motions (continuous flexes).
"""

import os
from copy import deepcopy
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import SVG
from matplotlib.animation import FuncAnimation

import pyrigi._utils._input_check as _input_check
from pyrigi._utils._zero_check import is_zero
from pyrigi.data_type import (
    Edge,
    Number,
    Point,
    Sequence,
    Vertex,
)
from pyrigi.framework._plot import plot as framework_plot
from pyrigi.graph import Graph
from pyrigi.plot_style import PlotStyle, PlotStyle2D, PlotStyle3D


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
        filename: str = None,
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
        filename:
            A name under which the produced animation is saved. If ``None``, the animation
            is not saved. Otherwise, the ``Animation.save`` method from ``matplotlib`` is
            invoked, which uses external writers to create the ``.gif`` file, such as
            ``ffmpeg`` (default) or ``pillow``. The standard video codec is ``h264``.
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
                realization[v][i]
                for v in self._graph.nodes
                for realization in realizations
            ]
            for i in range(3)
        ]

        ax.scatter(
            [min(x_coords) - plot_style.padding, max(x_coords) + plot_style.padding],
            [min(y_coords) - plot_style.padding, max(y_coords) + plot_style.padding],
            [min(z_coords) - plot_style.padding, max(z_coords) + plot_style.padding],
            color="white",
            s=plot_style.vertex_size,
            marker=plot_style.vertex_shape,
        )

        ax.set_box_aspect(plot_style.axis_scales)

        # Update the plot_style instance with any passed keyword arguments
        edge_color_array, edge_list_ref = framework_plot._resolve_edge_colors(
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
        if filename is not None:
            if not filename.endswith(".gif"):
                filename = filename + ".gif"
            ani.save(f"{filename}", fps=30)
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
        filename: str = None,
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
        filename:
            A name under which the produced animation is saved. If ``None``, the animation
            is not saved. Otherwise, the ``Animation.save`` method from ``matplotlib`` is
            invoked, which uses external writers to create the ``.gif`` file, such as
            ``ffmpeg`` (default) or ``pillow``. The standard video codec is ``h264``.
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
        edge_color_array, edge_list_ref = framework_plot._resolve_edge_colors(
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

        if filename is not None:
            if not filename.endswith(".gif"):
                filename = filename + ".gif"
            ani.save(f"{filename}", fps=30)
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
        duration: float = 8,
        filename: str = None,
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
        duration:
            The duration of one period of the animation in seconds.
        filename:
            A name under which the produced animation is saved. If ``None``, the svg
            is not saved.

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
