"""
Module for the functionality concerning frameworks.
"""

from __future__ import annotations

import functools
from random import randrange
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy import Matrix

import pyrigi.misc._input_check as _input_check
from pyrigi.data_type import (
    DirectedEdge,
    Edge,
    InfFlex,
    Number,
    Point,
    Sequence,
    Stress,
    Vertex,
)
from pyrigi.framework.base import FrameworkBase
from pyrigi.graph import Graph
from pyrigi.graphDB import Complete as CompleteGraph
from pyrigi.misc._wrap import copy_doc
from pyrigi.misc.misc import _doc_category as doc_category
from pyrigi.misc.misc import _generate_category_tables
from pyrigi.plot_style import PlotStyle, PlotStyle2D, PlotStyle3D

from . import _general as general
from ._rigidity import infinitesimal as infinitesimal_rigidity
from ._rigidity import matroidal as matroidal_rigidity
from ._rigidity import redundant as redundant_rigidity
from ._rigidity import second_order as second_order_rigidity
from ._rigidity import stress as stress_rigidity
from ._transformations import transformations

__doctest_requires__ = {
    ("Framework.generate_stl_bars",): ["trimesh", "manifold3d", "pathlib"]
}


class Framework(FrameworkBase):
    r"""
    This class provides the functionality for frameworks.

    Definitions
    -----------
     * :prf:ref:`Framework <def-framework>`
     * :prf:ref:`Realization <def-realization>`

    Parameters
    ----------
    graph:
        A graph without loops.
    realization:
        A dictionary mapping the vertices of the graph to points in $\RR^d$.
        The dimension $d$ is retrieved from the points in realization.
        If ``graph`` is empty, and hence also the ``realization``,
        the dimension is set to 0 (:meth:`.Empty`
        can be used to construct an empty framework with different dimension).

    Examples
    --------
    >>> F = Framework(Graph([[0,1]]), {0:[1,2], 1:[0,5]})
    >>> print(F)
    Framework in 2-dimensional space consisting of:
    Graph with vertices [0, 1] and edges [[0, 1]]
    Realization {0:(1, 2), 1:(0, 5)}

    Notice that the realization of a vertex can be accessed using ``[ ]``:

    >>> F[0]
    Matrix([
    [1],
    [2]])

    METHODS

    Notes
    -----
    Internally, the realization is represented as ``dict[Vertex,Matrix]``.
    However, :meth:`~Framework.realization` can also return ``dict[Vertex,Point]``.
    """

    @doc_category("Plotting")
    def plot2D(
        self,
        plot_style: PlotStyle = None,
        projection_matrix: Matrix = None,
        random_seed: int = None,
        coordinates: Sequence[int] = None,
        inf_flex: int | InfFlex = None,
        stress: int | Stress = None,
        edge_colors_custom: Sequence[Sequence[Edge]] | dict[str, Sequence[Edge]] = None,
        stress_label_positions: dict[DirectedEdge, float] = None,
        arc_angles_dict: Sequence[float] | dict[DirectedEdge, float] = None,
        filename: str = None,
        dpi=300,
        **kwargs,
    ) -> None:
        """
        Plot the framework in 2D.

        If the framework is in dimensions higher than 2 and ``projection_matrix``
        with ``coordinates`` are ``None``, a random projection matrix
        containing two orthonormal vectors is generated and used for projection into 2D.
        For various formatting options, see :class:`.PlotStyle`.
        Only ``coordinates`` or ``projection_matrix`` parameter can be used, not both!

        Parameters
        ----------
        plot_style:
            An instance of the ``PlotStyle`` class that defines the visual style
            for plotting, see :class:`.PlotStyle` for more details.
        projection_matrix:
            The matrix used for projecting the realization
            when the dimension is greater than 2.
            The matrix must have dimensions ``(2, dim)``,
            where ``dim`` is the dimension of the framework.
            If ``None``, a random projection matrix is generated.
        random_seed:
            The random seed used for generating the projection matrix.
        coordinates:
            Indices of two coordinates to which the framework is projected.
        inf_flex:
            Optional parameter for plotting a given infinitesimal flex. The standard
            input format is a ``Matrix`` that is the output of e.g. the method
            :meth:`~.Framework.inf_flexes`. Alternatively, an ``int`` can be specified
            to directly choose the 0,1,2,...-th nontrivial infinitesimal flex (according
            to the method :meth:`~.Framework.nontrivial_inf_flexes`) for plotting.
            For these input types, it is important to use the same vertex order
            as the one from :meth:`.Graph.vertex_list`.
            If the vertex order needs to be specified, a
            ``dict[Vertex, Sequence[Number]]`` can be provided, which maps the
            vertex labels to vectors (i.e. a sequence of coordinates).
        stress:
            Optional parameter for plotting a given equilibrium stress. The standard
            input format is a ``Matrix`` that is the output of e.g. the method
            :meth:`~.Framework.stresses`. Alternatively, an ``int`` can be specified
            to directly choose the 0,1,2,...-th equilibrium stress (according
            to the method :meth:`~.Framework.stresses`) for plotting.
            For these input types, it is important to use the same edge order as the one
            from :meth:`.Graph.edge_list`.
            If the edge order needs to be specified, a ``Dict[Edge, Number]``
            can be provided, which maps the edges to numbers
            (i.e. coordinates).
        edge_colors_custom:
            Optional parameter to specify the colors of edges. It can be
            a ``Sequence[Sequence[Edge]]`` to define groups of edges with the same color
            or a ``dict[str, Sequence[Edge]]`` where the keys are color strings and the
            values are lists of edges.
            The ommited edges are given the value ``plot_style.edge_color``.
        stress_label_positions:
            Dictionary specifying the position of stress labels along the edges. Keys are
            ``DirectedEdge`` objects, and values are floats (e.g., 0.5 for midpoint).
            Ommited edges are given the value ``0.5``.
        arc_angles_dict:
            Optional parameter to specify custom arc angle for edges. Can be a
            ``Sequence[float]`` or a ``dict[Edge, float]`` where values define
            the curvature angle of edges in radians.
        filename:
            The filename under which the produced figure is saved. The default value is
            ``None`` which indicates that the figure is currently not saved.
            The figure is saved as a ``.png`` file using the ``save`` method from
            ``matplotlib``.
        dpi: Dots per inched in case the figure is saved. Default is 300 for producing
            a print-quality image.

        Examples
        --------
        >>> from pyrigi import Graph, Framework
        >>> G = Graph([(0,1), (1,2), (2,3), (0,3), (0,2), (1,3), (0,4)])
        >>> F = Framework(G, {0:(0,0), 1:(1,0), 2:(1,2), 3:(0,1), 4:(-1,0)})
        >>> from pyrigi import PlotStyle2D
        >>> style = PlotStyle2D(vertex_color="green", edge_color="blue")
        >>> F.plot2D(plot_style=style)

        Use keyword arguments

        >>> F.plot2D(vertex_color="red", edge_color="black", vertex_size=500)

        Specify stress and its labels positions

        >>> stress_label_positions = {(0, 1): 0.7, (1, 2): 0.2}
        >>> F.plot2D(stress=0, stress_label_positions=stress_label_positions)

        Specify infinitesimal flex

        >>> F.plot2D(inf_flex=0)

        Use both stress and infinitesimal flex

        >>> F.plot2D(stress=0, inf_flex=0)

        Use custom edge colors

        >>> edge_colors = {'red': [(0, 1), (1, 2)], 'blue': [(2, 3), (0, 3)]}
        >>> F.plot2D(edge_colors_custom=edge_colors)

        The following is just to close all figures after running the example:

        >>> import matplotlib.pyplot
        >>> matplotlib.pyplot.close("all")
        """
        if plot_style is None:
            plot_style = PlotStyle2D()
        else:
            plot_style = PlotStyle2D.from_plot_style(plot_style)

        # Update the plot_style instance with any passed keyword arguments
        plot_style.update(**kwargs)

        fig, ax = plt.subplots()
        ax.set_adjustable("datalim")
        fig.set_figwidth(plot_style.canvas_width)
        fig.set_figheight(plot_style.canvas_height)
        ax.set_aspect(plot_style.aspect_ratio)

        from pyrigi.framework.plot import _plot

        if self._dim == 1:
            placement = {
                vertex: [position[0], 0]
                for vertex, position in self.realization(
                    as_points=True, numerical=True
                ).items()
            }
            if hasattr(kwargs, "edges_as_arcs"):
                plot_style.update(edges_as_arcs=kwargs["edges_as_arcs"])
            else:
                plot_style.update(edges_as_arcs=True)

        elif self._dim == 2:
            placement = self.realization(as_points=True, numerical=True)

        else:
            placement, projection_matrix = self.projected_realization(
                projection_matrix=projection_matrix,
                coordinates=coordinates,
                proj_dim=2,
                random_seed=random_seed,
            )

        _plot.plot_with_2D_realization(
            self,
            ax,
            placement,
            plot_style=plot_style,
            edge_colors_custom=edge_colors_custom,
            arc_angles_dict=arc_angles_dict,
        )

        if inf_flex is not None:
            _plot.plot_inf_flex2D(
                self,
                ax,
                inf_flex,
                realization=placement,
                plot_style=plot_style,
                projection_matrix=projection_matrix,
            )
        if stress is not None:
            _plot.plot_stress2D(
                self,
                ax,
                stress,
                realization=placement,
                plot_style=plot_style,
                arc_angles_dict=arc_angles_dict,
                stress_label_positions=stress_label_positions,
            )

        if filename is not None:
            if not filename.endswith(".png"):
                filename = filename + ".png"
            plt.savefig(f"{filename}", dpi=dpi)

    @doc_category("Plotting")
    def animate3D_rotation(
        self,
        plot_style: PlotStyle = None,
        edge_colors_custom: Sequence[Sequence[Edge]] | dict[str, Sequence[Edge]] = None,
        total_frames: int = 100,
        delay: int = 75,
        rotation_axis: str | Sequence[Number] = None,
        **kwargs,
    ) -> Any:
        """
        Plot this framework in 3D and animate a rotation around an axis.

        For additional parameters and implementation details, see
        :meth:`~.Motion.animate3D`.

        Parameters
        ----------
        plot_style:
            An instance of the ``PlotStyle`` class that defines the visual style
            for plotting, see :class:`PlotStyle` for more details.
        edge_colors_custom:
            Optional parameter to specify the colors of edges. It can be
            a ``Sequence[Sequence[Edge]]`` to define groups of edges with the same color
            or a ``dict[str, Sequence[Edge]]`` where the keys are color strings and the
            values are lists of edges.
            The ommited edges are given the value ``plot_style.edge_color``.
        total_frames:
            Total number of frames for the animation sequence.
        delay:
            Delay between frames in milliseconds.
        rotation_axis:
            The user can input a rotation axis or vector. By default, a rotation around
            the z-axis is performed. This can either a character
            (``'x'``, ``'y'``, or ``'z'``) or a vector (e.g. ``[1, 0, 0]``).

        Examples
        --------
        >>> from pyrigi import frameworkDB
        >>> F = frameworkDB.Complete(4, dim=3)
        >>> F.animate3D_rotation()
        """
        _input_check.dimension_for_algorithm(self._dim, [3], "animate3D")
        if plot_style is None:
            # change some PlotStyle default values to fit 3D plotting better
            plot_style = PlotStyle3D(vertex_size=13.5, edge_width=1.5, dpi=150)
        else:
            plot_style = PlotStyle3D.from_plot_style(plot_style)

        # Update the plot_style instance with any passed keyword arguments
        plot_style.update(**kwargs)

        realization = self.realization(as_points=True, numerical=True)
        centroid_x, centroid_y, centroid_z = [
            sum([pos[i] for pos in realization.values()]) / len(realization)
            for i in range(3)
        ]
        realization = {
            v: [point[0] - centroid_x, point[1] - centroid_y, point[2] - centroid_z]
            for v, point in realization.items()
        }

        def _rotation_matrix(vector, frame):
            # Compute the rotation matrix Q
            vector = np.array(vector)
            vector = vector / np.linalg.norm(vector)
            angle = frame * np.pi / total_frames
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

            # Rodrigues' rotation matrix
            K = np.array(
                [
                    [0, -vector[2], vector[1]],
                    [vector[2], 0, -vector[0]],
                    [-vector[1], vector[0], 0],
                ]
            )
            Q = (
                np.eye(3) * cos_angle
                + K * sin_angle
                + np.outer(vector, vector) * (1 - cos_angle)
            )
            return Q

        match rotation_axis:
            case None | "z" | "Z":
                rotation_matrix = functools.partial(
                    _rotation_matrix, np.array([0, 0, 1])
                )
            case "x" | "X":
                rotation_matrix = functools.partial(
                    _rotation_matrix, np.array([1, 0, 0])
                )
            case "y" | "Y":
                rotation_matrix = functools.partial(
                    _rotation_matrix, np.array([0, 1, 0])
                )
            case _:  # Rotation around a custom axis
                if isinstance(rotation_axis, (np.ndarray, list, tuple)):
                    if len(rotation_axis) != 3:
                        raise ValueError("The rotation_axis must have length 3.")
                    rotation_matrix = functools.partial(
                        _rotation_matrix, np.array(rotation_axis)
                    )
                else:
                    raise ValueError(
                        "The rotation_axis must be of one of the following "
                        + "types: np.ndarray, list, tuple."
                    )

        rotating_realizations = [
            {
                v: np.dot(pos, rotation_matrix(frame).T).tolist()
                for v, pos in realization.items()
            }
            for frame in range(2 * total_frames)
        ]
        pinned_vertex = self._graph.vertex_list()[0]
        _realizations = []
        for rotated_realization in rotating_realizations:
            # Translate the realization to the origin
            _realizations.append(
                {
                    v: [
                        pos[i] - rotated_realization[pinned_vertex][i]
                        for i in range(len(pos))
                    ]
                    for v, pos in rotated_realization.items()
                }
            )

        from pyrigi import Motion

        motion = Motion(self.graph, self.dim)
        duration = 2 * total_frames * delay / 1000
        return motion.animate3D(
            _realizations,
            plot_style=plot_style,
            edge_colors_custom=edge_colors_custom,
            duration=duration,
            **kwargs,
        )

    @doc_category("Plotting")
    def plot3D(
        self,
        plot_style: PlotStyle = None,
        projection_matrix: Matrix = None,
        random_seed: int = None,
        coordinates: Sequence[int] = None,
        inf_flex: int | InfFlex = None,
        stress: int | Stress = None,
        edge_colors_custom: Sequence[Sequence[Edge]] | dict[str, Sequence[Edge]] = None,
        stress_label_positions: dict[DirectedEdge, float] = None,
        filename: str = None,
        dpi=300,
        **kwargs,
    ) -> None:
        """
        Plot the provided framework in 3D.

        If the framework is in a dimension higher than 3 and ``projection_matrix``
        with ``coordinates`` are ``None``, a random projection matrix
        containing three orthonormal vectors is generated and used for projection into 3D.
        For various formatting options, see :class:`.PlotStyle`.
        Only the parameter ``coordinates`` or ``projection_matrix`` can be used,
        not both at the same time.

        Parameters
        ----------
        plot_style:
            An instance of the ``PlotStyle`` class that defines the visual style
            for plotting, see :class:`.PlotStyle` for more details.
        projection_matrix:
            The matrix used for projecting the realization
            when the dimension is greater than 3.
            The matrix must have dimensions ``(3, dim)``,
            where ``dim`` is the dimension of the framework.
            If ``None``, a random projection matrix is generated.
        random_seed:
            The random seed used for generating the projection matrix.
        coordinates:
            Indices of two coordinates to which the framework is projected.
        inf_flex:
            Optional parameter for plotting a given infinitesimal flex. The standard
            input format is a ``Matrix`` that is the output of e.g. the method
            :meth:`~.Framework.inf_flexes`. Alternatively, an ``int`` can be specified
            to directly choose the 0,1,2,...-th nontrivial infinitesimal flex (according
            to the method :meth:`~.Framework.nontrivial_inf_flexes`) for plotting.
            For these input types, is important to use the same vertex order as the one
            from :meth:`.Graph.vertex_list`.
            If the vertex order needs to be specified, a
            ``dict[Vertex, Sequence[Number]]`` can be provided, which maps the
            vertex labels to vectors (i.e. a sequence of coordinates).
        stress:
            Optional parameter for plotting a given equilibrium stress. The standard
            input format is a ``Matrix`` that is the output of e.g. the method
            :meth:`~.Framework.stresses`. Alternatively, an ``int`` can be specified
            to directly choose the 0,1,2,...-th equilibrium stress (according
            to the method :meth:`~.Framework.stresses`) for plotting.
            For these input types, is important to use the same edge order as the one
            from :meth:`.Graph.edge_list`.
            If the edge order needs to be specified, a ``Dict[Edge, Number]``
            can be provided, which maps the edges to numbers
            (i.e. coordinates).
        edge_colors_custom:
            Optional parameter to specify the colors of edges. It can be
            a ``Sequence[Sequence[Edge]]`` to define groups of edges with the same color
            or a ``dict[str, Sequence[Edge]]`` where the keys are color strings and the
            values are lists of edges.
            The ommited edges are given the value ``plot_style.edge_color``.
        stress_label_positions:
            Dictionary specifying the position of stress labels along the edges. Keys are
            ``DirectedEdge`` objects, and values are floats (e.g., 0.5 for midpoint).
            Ommited edges are given the value ``0.5``.
        filename:
            The filename under which the produced figure is saved. The default value is
            ``None`` which indicates that the figure is currently not saved.
            The figure is saved as a ``.png`` file using the ``save`` method from
            ``matplotlib``.
        dpi: Dots per inched in case the figure is saved. Default is 300 for producing
            a print-quality image.

        Examples
        --------
        >>> from pyrigi import frameworkDB
        >>> F = frameworkDB.Octahedron(realization="Bricard_plane")
        >>> F.plot3D()

        >>> from pyrigi import PlotStyle3D
        >>> style = PlotStyle3D(vertex_color="green", edge_color="blue")
        >>> F.plot3D(plot_style=style)

        Use keyword arguments

        >>> F.plot3D(vertex_color="red", edge_color="black", vertex_size=500)

        Specify stress and its positions

        >>> stress_label_positions = {(0, 2): 0.7, (1, 2): 0.2}
        >>> F.plot3D(stress=0, stress_label_positions=stress_label_positions)

        Specify infinitesimal flex

        >>> F.plot3D(inf_flex=0)

        Use both stress and infinitesimal flex

        >>> F.plot3D(stress=0, inf_flex=0)

        Use custom edge colors

        >>> edge_colors = {'red': [(5, 1), (1, 2)], 'blue': [(2, 4), (4, 3)]}
        >>> F.plot3D(edge_colors_custom=edge_colors)

        The following is just to close all figures after running the example:

        >>> import matplotlib.pyplot
        >>> matplotlib.pyplot.close("all")
        """
        if plot_style is None:
            # change some PlotStyle default values to fit 3D plotting better
            plot_style = PlotStyle3D(vertex_size=175, flex_length=0.2)
        else:
            plot_style = PlotStyle3D.from_plot_style(plot_style)

        # Update the plot_style instance with any passed keyword arguments
        plot_style.update(**kwargs)

        fig = plt.figure(dpi=plot_style.dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.grid(False)
        ax.set_axis_off()

        placement = self.realization(as_points=True, numerical=True)
        if self._dim in [1, 2]:
            placement = {
                v: list(p) + [0 for _ in range(3 - self._dim)]
                for v, p in placement.items()
            }

        elif self._dim == 3:
            placement = self.realization(as_points=True, numerical=True)

        else:
            placement, projection_matrix = self.projected_realization(
                projection_matrix=projection_matrix,
                coordinates=coordinates,
                proj_dim=3,
                random_seed=random_seed,
            )

        # Center the realization
        centroid = [
            sum([pos[i] for pos in placement.values()]) / len(placement)
            for i in range(3)
        ]
        _placement = {
            v: [pos[0] - centroid[0], pos[1] - centroid[1], pos[2] - centroid[2]]
            for v, pos in placement.items()
        }

        from pyrigi.framework.plot import _plot

        _plot.plot_with_3D_realization(
            self,
            ax,
            _placement,
            plot_style,
            edge_colors_custom=edge_colors_custom,
        )

        if inf_flex is not None:
            _plot.plot_inf_flex3D(
                self,
                ax,
                inf_flex,
                realization=_placement,
                plot_style=plot_style,
                projection_matrix=projection_matrix,
            )

        if stress is not None:
            _plot.plot_stress3D(
                self,
                ax,
                stress,
                realization=_placement,
                plot_style=plot_style,
                stress_label_positions=stress_label_positions,
            )

        if filename is not None:
            if not filename.endswith(".png"):
                filename = filename + ".png"
            plt.savefig(f"{filename}", dpi=dpi)

    @doc_category("Plotting")
    def plot(
        self,
        plot_style: PlotStyle = None,
        **kwargs,
    ) -> None:
        """
        Plot the framework.

        The framework can be plotted only if its dimension is less than 3.
        For plotting a projection of a higher dimensional framework,
        use :meth:`.plot2D` or :meth:`.plot3D` instead.
        For various formatting options, see :class:`.PlotStyle`.
        """
        if self._dim == 3:
            self.plot3D(plot_style=plot_style, **kwargs)
        elif self._dim > 3:
            raise ValueError(
                "This framework is in higher dimension than 3!"
                + " For projection into 2D use F.plot2D(),"
                + " for projection into 3D use F.plot3D()."
            )
        else:
            self.plot2D(plot_style=plot_style, **kwargs)

    @doc_category("Other")
    def to_tikz(
        self,
        vertex_style: str | dict[str, Sequence[Vertex]] = "fvertex",
        edge_style: str | dict[str, Sequence[Edge]] = "edge",
        label_style: str = "labelsty",
        figure_opts: str = "",
        vertex_in_labels: bool = False,
        vertex_out_labels: bool = False,
        default_styles: bool = True,
    ) -> str:
        r"""
        Create a TikZ code for the framework.

        The framework must have dimension 2.
        For using it in ``LaTeX`` you need to use the ``tikz`` package.
        For more examples on formatting options, see also :meth:`.Graph.to_tikz`.

        Parameters
        ----------
        vertex_style:
            If a single style is given as a string,
            then all vertices get this style.
            If a dictionary from styles to a list of vertices is given,
            vertices are put in style accordingly.
            The vertices missing in the dictionary do not get a style.
        edge_style:
            If a single style is given as a string,
            then all edges get this style.
            If a dictionary from styles to a list of edges is given,
            edges are put in style accordingly.
            The edges missing in the dictionary do not get a style.
        label_style:
            The style for labels that are placed next to vertices.
        figure_opts:
            Options for the tikzpicture environment.
        vertex_in_labels
            A bool on whether vertex names should be put as labels on the vertices.
        vertex_out_labels
            A bool on whether vertex names should be put next to vertices.
        default_styles
            A bool on whether default style definitions should be put to the options.

        Examples
        ----------
        >>> G = Graph([(0, 1), (1, 2), (2, 3), (0, 3)])
        >>> F=Framework(G,{0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1]})
        >>> print(F.to_tikz()) # doctest: +NORMALIZE_WHITESPACE
        \begin{tikzpicture}[fvertex/.style={circle,inner sep=0pt,minimum size=3pt,fill=white,draw=black,double=white,double distance=0.25pt,outer sep=1pt},edge/.style={line width=1.5pt,black!60!white}]
           \node[fvertex] (0) at (0, 0) {};
           \node[fvertex] (1) at (1, 0) {};
           \node[fvertex] (2) at (1, 1) {};
           \node[fvertex] (3) at (0, 1) {};
           \draw[edge] (0) to (1) (0) to (3) (1) to (2) (2) to (3);
        \end{tikzpicture}

        >>> print(F.to_tikz(vertex_in_labels=True)) # doctest: +NORMALIZE_WHITESPACE
        \begin{tikzpicture}[fvertex/.style={circle,inner sep=1pt,minimum size=3pt,fill=white,draw=black,double=white,double distance=0.25pt,outer sep=1pt,font=\scriptsize},edge/.style={line width=1.5pt,black!60!white}]
           \node[fvertex] (0) at (0, 0) {$0$};
           \node[fvertex] (1) at (1, 0) {$1$};
           \node[fvertex] (2) at (1, 1) {$2$};
           \node[fvertex] (3) at (0, 1) {$3$};
           \draw[edge] (0) to (1) (0) to (3) (1) to (2) (2) to (3);
        \end{tikzpicture}
        """  # noqa: E501

        # check dimension
        if self.dim != 2:
            raise ValueError(
                "TikZ code is only generated for frameworks in dimension 2."
            )

        # strings for tikz styles
        if vertex_out_labels and default_styles:
            lstyle_str = r"labelsty/.style={font=\scriptsize,black!70!white}"
        else:
            lstyle_str = ""

        if vertex_style == "fvertex" and default_styles:
            if vertex_in_labels:
                vstyle_str = (
                    "fvertex/.style={circle,inner sep=1pt,minimum size=3pt,"
                    "fill=white,draw=black,double=white,double distance=0.25pt,"
                    r"outer sep=1pt,font=\scriptsize}"
                )
            else:
                vstyle_str = (
                    "fvertex/.style={circle,inner sep=0pt,minimum size=3pt,fill=white,"
                    "draw=black,double=white,double distance=0.25pt,outer sep=1pt}"
                )
        else:
            vstyle_str = ""
        if edge_style == "edge" and default_styles:
            estyle_str = "edge/.style={line width=1.5pt,black!60!white}"
        else:
            estyle_str = ""

        figure_str = [figure_opts, vstyle_str, estyle_str, lstyle_str]
        figure_str = [fs for fs in figure_str if fs != ""]
        figure_str = ",".join(figure_str)

        return self.graph.to_tikz(
            placement=self.realization(),
            figure_opts=figure_str,
            vertex_style=vertex_style,
            edge_style=edge_style,
            label_style=label_style,
            vertex_in_labels=vertex_in_labels,
            vertex_out_labels=vertex_out_labels,
            default_styles=False,
        )

    @classmethod
    @doc_category("Class methods")
    def from_points(cls, points: Sequence[Point]) -> Framework:
        """
        Generate a framework from a list of points.

        The list of vertices of the underlying graph
        is taken to be ``[0,...,len(points)-1]``.
        The underlying graph has no edges.

        Parameters
        ----------
        points:
            The realization of the framework that this method outputs
            is provided as a list of points.

        Examples
        --------
        >>> F = Framework.from_points([(1,2), (2,3)])
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0, 1] and edges []
        Realization {0:(1, 2), 1:(2, 3)}
        """
        vertices = range(len(points))
        realization = {v: points[v] for v in vertices}
        return Framework(Graph.from_vertices(vertices), realization)

    @classmethod
    @doc_category("Class methods")
    def Random(
        cls,
        graph: Graph,
        dim: int = 2,
        rand_range: int | Sequence[int] = None,
        numerical: bool = False,
    ) -> Framework:
        """
        Return a framework with random realization.

        Depending on the parameter ``numerical``, the realization either
        consists of random integers (``numerical=False``) or random floats
        (``numerical=True``).

        Parameters
        ----------
        dim:
            The dimension of the constructed framework.
        graph:
            Graph for which the random realization should be constructed.
        rand_range:
            Sets the range of random numbers from which the realization is
            sampled. The format is either an interval ``(a,b)`` or a single
            integer ``a``, which produces the range ``(-a,a)``.
            If ``rand_range=None``, then the range is set to ``(-a,a)`` for
            ``a = 10^4 * n * dim`` in the case that ``numerical=False``, where
            ``n`` is the number of vertices. For ``numerical=True``, we set the
            default interval to ``(-1,1)``.
        numerical:
            A boolean indicating whether numerical coordinates should be used.

        Examples
        --------
        >>> F = Framework.Random(Graph([(0,1), (1,2), (0,2)]))
        >>> print(F) # doctest: +SKIP
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        Realization {0:(122, 57), 1:(27, 144), 2:(50, 98)}
        """
        _input_check.dimension(dim)
        if rand_range is None:
            if numerical:
                a, b = -1, 1
            else:
                b = 10**4 * graph.number_of_nodes() * dim
                a = -b
        elif isinstance(rand_range, list | tuple):
            if not len(rand_range) == 2:
                raise ValueError("If `rand_range` is a list, it must be of length 2.")
            a, b = rand_range
        elif isinstance(rand_range, int):
            if rand_range <= 0:
                raise ValueError("If `rand_range` is an int, it must be positive")
            b = rand_range
            a = -b
        else:
            raise TypeError("`rand_range` must be either a list or a single int.")

        if numerical:
            realization = {
                v: [a + np.random.rand() * (b - a) for _ in range(dim)]
                for v in graph.nodes
            }
        else:
            realization = {
                v: [randrange(a, b) for _ in range(dim)] for v in graph.nodes
            }

        return Framework(graph, realization)

    @classmethod
    @doc_category("Class methods")
    def Circular(cls, graph: Graph) -> Framework:
        """
        Return the framework with a regular unit circle realization in the plane.

        Parameters
        ----------
        graph:
            Underlying graph on which the framework is constructed.

        Examples
        ----
        >>> import pyrigi.graphDB as graphs
        >>> F = Framework.Circular(graphs.CompleteBipartite(4, 2))
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0, 1, 2, 3, 4, 5] and edges ...
        Realization {0:(1, 0), 1:(1/2, sqrt(3)/2), ...
        """
        n = graph.number_of_nodes()
        return Framework(
            graph,
            {
                v: [sp.cos(2 * i * sp.pi / n), sp.sin(2 * i * sp.pi / n)]
                for i, v in enumerate(graph.vertex_list())
            },
        )

    @classmethod
    @doc_category("Class methods")
    def Collinear(cls, graph: Graph, dim: int = 1) -> Framework:
        """
        Return the framework with a realization on the x-axis.

        Parameters
        ----------
        dim:
            The dimension of the space in which the framework is constructed.
        graph:
            Underlying graph on which the framework is constructed.

        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> print(Framework.Collinear(graphs.Complete(3), dim=2))
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        Realization {0:(0, 0), 1:(1, 0), 2:(2, 0)}
        """
        _input_check.dimension(dim)
        return Framework(
            graph,
            {
                v: [i] + [0 for _ in range(dim - 1)]
                for i, v in enumerate(graph.vertex_list())
            },
        )

    @classmethod
    @doc_category("Class methods")
    def Simplicial(cls, graph: Graph, dim: int = None) -> Framework:
        """
        Return the framework with a realization on the ``dim``-simplex.

        Parameters
        ----------
        graph:
            Underlying graph on which the framework is constructed.
        dim:
            The dimension ``dim`` has to be at least the number of vertices
            of the ``graph`` minus one.
            If ``dim`` is not specified, then the least possible one is used.

        Examples
        ----
        >>> F = Framework.Simplicial(Graph([(0,1), (1,2), (2,3), (0,3)]), 4)
        >>> F.realization(as_points=True)
        {0: [0, 0, 0, 0], 1: [1, 0, 0, 0], 2: [0, 1, 0, 0], 3: [0, 0, 1, 0]}
        >>> F = Framework.Simplicial(Graph([(0,1), (1,2), (2,3), (0,3)]))
        >>> F.realization(as_points=True)
        {0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}
        """
        if dim is None:
            dim = graph.number_of_nodes() - 1
        _input_check.integrality_and_range(
            dim, "dimension d", max([1, graph.number_of_nodes() - 1])
        )
        return Framework(
            graph,
            {
                v: [1 if j == i - 1 else 0 for j in range(dim)]
                for i, v in enumerate(graph.vertex_list())
            },
        )

    @classmethod
    @doc_category("Class methods")
    def Complete(cls, points: Sequence[Point]) -> Framework:
        """
        Generate a framework on the complete graph from a given list of points.

        The vertices of the underlying graph are taken
        to be the list ``[0,...,len(points)-1]``.

        Parameters
        ----------
        points:
            The realization of the framework that this method outputs
            is provided as a list of points.

        Examples
        --------
        >>> F = Framework.Complete([(1,),(2,),(3,),(4,)]); print(F)
        Framework in 1-dimensional space consisting of:
        Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        Realization {0:(1,), 1:(2,), 2:(3,), 3:(4,)}
        """  # noqa: E501
        if not points:
            raise ValueError("The list of points cannot be empty!")

        Kn = CompleteGraph(len(points))
        return Framework(Kn, {v: pos for v, pos in zip(Kn.nodes, points)})

    @doc_category("Framework properties")
    @copy_doc(general.is_quasi_injective)
    def is_quasi_injective(
        self, numerical: bool = False, tolerance: float = 1e-9
    ) -> bool:
        return general.is_quasi_injective(
            self, numerical=numerical, tolerance=tolerance
        )

    @doc_category("Framework properties")
    @copy_doc(general.is_injective)
    def is_injective(self, numerical: bool = False, tolerance: float = 1e-9) -> bool:
        return general.is_injective(self, numerical=numerical, tolerance=tolerance)

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.rigidity_matrix)
    def rigidity_matrix(
        self,
        vertex_order: Sequence[Vertex] = None,
        edge_order: Sequence[Edge] = None,
    ) -> Matrix:
        return infinitesimal_rigidity.rigidity_matrix(
            self, vertex_order=vertex_order, edge_order=edge_order
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(stress_rigidity.is_dict_stress)
    def is_dict_stress(self, dict_stress: dict[Edge, Number], **kwargs) -> bool:
        return stress_rigidity.is_dict_stress(self, dict_stress=dict_stress, **kwargs)

    @doc_category("Infinitesimal rigidity")
    @copy_doc(stress_rigidity.is_vector_stress)
    def is_vector_stress(
        self,
        stress: Sequence[Number],
        edge_order: Sequence[Edge] = None,
        numerical: bool = False,
        tolerance=1e-9,
    ) -> bool:
        return stress_rigidity.is_vector_stress(
            self,
            stress=stress,
            edge_order=edge_order,
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(stress_rigidity.is_stress)
    def is_stress(self, stress: Stress, **kwargs) -> bool:
        return stress_rigidity.is_stress(self, stress=stress, **kwargs)

    @doc_category("Infinitesimal rigidity")
    @copy_doc(stress_rigidity.stress_matrix)
    def stress_matrix(
        self,
        stress: Stress,
        edge_order: Sequence[Edge] = None,
        vertex_order: Sequence[Vertex] = None,
    ) -> Matrix:
        return stress_rigidity.stress_matrix(
            self, stress=stress, edge_order=edge_order, vertex_order=vertex_order
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.trivial_inf_flexes)
    def trivial_inf_flexes(self, vertex_order: Sequence[Vertex] = None) -> list[Matrix]:
        return infinitesimal_rigidity.trivial_inf_flexes(
            self, vertex_order=vertex_order
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.nontrivial_inf_flexes)
    def nontrivial_inf_flexes(self, **kwargs) -> list[Matrix]:
        return infinitesimal_rigidity.nontrivial_inf_flexes(self, **kwargs)

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.inf_flexes)
    def inf_flexes(
        self,
        include_trivial: bool = False,
        vertex_order: Sequence[Vertex] = None,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> list[Matrix] | list[list[float]]:
        return infinitesimal_rigidity.inf_flexes(
            self,
            include_trivial=include_trivial,
            vertex_order=vertex_order,
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(stress_rigidity.stresses)
    def stresses(
        self,
        edge_order: Sequence[Edge] = None,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> list[Matrix] | list[list[float]]:
        return stress_rigidity.stresses(
            self, edge_order=edge_order, numerical=numerical, tolerance=tolerance
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.rigidity_matrix_rank)
    def rigidity_matrix_rank(
        self, numerical: bool = False, tolerance: bool = 1e-9
    ) -> int:
        return infinitesimal_rigidity.rigidity_matrix_rank(
            self, numerical=numerical, tolerance=tolerance
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_inf_rigid)
    def is_inf_rigid(self, numerical: bool = False, tolerance: bool = 1e-9) -> bool:
        return infinitesimal_rigidity.is_inf_rigid(
            self, numerical=numerical, tolerance=tolerance
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_inf_flexible)
    def is_inf_flexible(self, **kwargs) -> bool:
        return infinitesimal_rigidity.is_inf_flexible(self, **kwargs)

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_min_inf_rigid)
    def is_min_inf_rigid(self, use_copy: bool = True, **kwargs) -> bool:
        return infinitesimal_rigidity.is_min_inf_rigid(
            self, use_copy=use_copy, **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(matroidal_rigidity.is_independent)
    def is_independent(self, **kwargs) -> bool:
        return matroidal_rigidity.is_independent(self, **kwargs)

    @doc_category("Infinitesimal rigidity")
    @copy_doc(matroidal_rigidity.is_dependent)
    def is_dependent(self, **kwargs) -> bool:
        return matroidal_rigidity.is_dependent(self, **kwargs)

    @doc_category("Infinitesimal rigidity")
    @copy_doc(matroidal_rigidity.is_isostatic)
    def is_isostatic(self, **kwargs) -> bool:
        return matroidal_rigidity.is_isostatic(self, **kwargs)

    @doc_category("Other")
    @copy_doc(second_order_rigidity.is_prestress_stable)
    def is_prestress_stable(
        self,
        numerical: bool = False,
        tolerance: float = 1e-9,
        inf_flexes: Sequence[InfFlex] = None,
        stresses: Sequence[Stress] = None,
    ) -> bool:
        return second_order_rigidity.is_prestress_stable(
            self,
            numerical=numerical,
            tolerance=tolerance,
            inf_flexes=inf_flexes,
            stresses=stresses,
        )

    @doc_category("Other")
    @copy_doc(second_order_rigidity.is_second_order_rigid)
    def is_second_order_rigid(
        self,
        numerical: bool = False,
        tolerance: float = 1e-9,
        inf_flexes: Sequence[InfFlex] = None,
        stresses: Sequence[Stress] = None,
    ) -> bool:
        return second_order_rigidity.is_second_order_rigid(
            self,
            numerical=numerical,
            tolerance=tolerance,
            inf_flexes=inf_flexes,
            stresses=stresses,
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(redundant_rigidity.is_redundantly_inf_rigid)
    def is_redundantly_inf_rigid(self, use_copy: bool = True, **kwargs) -> bool:
        return redundant_rigidity.is_redundantly_inf_rigid(
            self, use_copy=use_copy, **kwargs
        )

    @doc_category("Framework properties")
    @copy_doc(general.is_congruent_realization)
    def is_congruent_realization(
        self,
        other_realization: dict[Vertex, Point] | dict[Vertex, Matrix],
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        return general.is_congruent_realization(
            self,
            other_realization=other_realization,
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Framework properties")
    @copy_doc(general.is_congruent)
    def is_congruent(
        self,
        other_framework: Framework,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        return general.is_congruent(
            self,
            other_framework=other_framework,
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Framework properties")
    @copy_doc(general.is_equivalent_realization)
    def is_equivalent_realization(
        self,
        other_realization: dict[Vertex, Point] | dict[Vertex, Matrix],
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        return general.is_equivalent_realization(
            self,
            other_realization=other_realization,
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Framework properties")
    @copy_doc(general.is_equivalent)
    def is_equivalent(
        self,
        other_framework: Framework,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        return general.is_equivalent(
            self,
            other_framework=other_framework,
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Framework manipulation")
    @copy_doc(transformations.translate)
    def translate(
        self, vector: Point | Matrix, inplace: bool = True
    ) -> None | Framework:
        return transformations.translate(self, vector, inplace=inplace)

    @doc_category("Framework manipulation")
    @copy_doc(transformations.rescale)
    def rescale(self, factor: Number, inplace: bool = True) -> None | Framework:
        return transformations.rescale(self, factor, inplace=inplace)

    @doc_category("Framework manipulation")
    @copy_doc(transformations.rotate2D)
    def rotate2D(
        self, angle: float, rotation_center: Point = [0, 0], inplace: bool = True
    ) -> None | Framework:
        return transformations.rotate2D(
            self, angle, rotation_center=rotation_center, inplace=inplace
        )

    @doc_category("Framework manipulation")
    @copy_doc(transformations.rotate3D)
    def rotate3D(
        self,
        angle: Number,
        axis_direction: Sequence[Number] = [0, 0, 1],
        axis_shift: Point = [0, 0, 0],
        inplace: bool = True,
    ) -> None | Framework:
        return transformations.rotate3D(
            self,
            angle,
            axis_direction=axis_direction,
            axis_shift=axis_shift,
            inplace=inplace,
        )

    @doc_category("Framework manipulation")
    @copy_doc(transformations.rotate)
    def rotate(self, **kwargs) -> None | Framework:
        return transformations.rotate(self, **kwargs)

    @doc_category("Framework manipulation")
    @copy_doc(transformations.projected_realization)
    def projected_realization(
        self,
        proj_dim: int = None,
        projection_matrix: Matrix = None,
        random_seed: int = None,
        coordinates: Sequence[int] = None,
    ) -> tuple[dict[Vertex, Point], Matrix]:
        return transformations.projected_realization(
            self,
            proj_dim=proj_dim,
            projection_matrix=projection_matrix,
            random_seed=random_seed,
            coordinates=coordinates,
        )

    @doc_category("Other")
    @copy_doc(general.edge_lengths)
    def edge_lengths(self, numerical: bool = False) -> dict[Edge, Number]:
        return general.edge_lengths(self, numerical=numerical)

    @staticmethod
    def _generate_stl_bar(
        holes_distance: float,
        holes_diameter: float,
        bar_width: float,
        bar_height: float,
        filename="bar.stl",
    ) -> Any:
        """
        Generate an STL file for a bar.

        The method uses ``Trimesh`` and ``Manifold3d`` packages to create
        a model of a bar with two holes at the ends.
        The method returns the bar as a ``Trimesh`` object and saves it as an STL file.

        Parameters
        ----------
        holes_distance:
            Distance between the centers of the holes.
        holes_diameter:
            Diameter of the holes.
        bar_width:
            Width of the bar.
        bar_height:
            Height of the bar.
        filename:
            Name of the output STL file.
        """
        try:
            from trimesh.creation import box as trimesh_box
            from trimesh.creation import cylinder as trimesh_cylinder
        except ImportError:
            raise ImportError(
                "To create meshes of bars that can be exported as STL files, "
                "the packages 'trimesh' and 'manifold3d' are required. "
                "To install PyRigi including trimesh and manifold3d, "
                "run 'pip install pyrigi[meshing]'!"
            )

        _input_check.greater(holes_distance, 0, "holes_distance")
        _input_check.greater(holes_diameter, 0, "holes_diameter")
        _input_check.greater(bar_width, 0, "bar_width")
        _input_check.greater(bar_height, 0, "bar_height")

        _input_check.greater(bar_width, holes_diameter, "bar_width", "holes_diameter")
        _input_check.greater(
            holes_distance,
            2 * holes_diameter,
            "holes_distance",
            "twice the holes_diameter",
        )

        # Create the main bar as a box
        bar = trimesh_box(extents=[holes_distance, bar_width, bar_height])

        # Define the positions of the holes (relative to the center of the bar)
        hole_position_1 = [-holes_distance / 2, 0, 0]
        hole_position_2 = [holes_distance / 2, 0, 0]

        # Create cylindrical shapes at the ends of the bar
        rounding_1 = trimesh_cylinder(radius=bar_width / 2, height=bar_height)
        rounding_1.apply_translation(hole_position_1)
        rounding_2 = trimesh_cylinder(radius=bar_width / 2, height=bar_height)
        rounding_2.apply_translation(hole_position_2)

        # Use boolean union to combine the bar and the roundings
        bar = bar.union([rounding_1, rounding_2])

        # Create cylindrical holes
        hole_1 = trimesh_cylinder(radius=holes_diameter / 2, height=bar_height)
        hole_1.apply_translation(hole_position_1)
        hole_2 = trimesh_cylinder(radius=holes_diameter / 2, height=bar_height)
        hole_2.apply_translation(hole_position_2)

        # Use boolean subtraction to create holes in the bar
        bar_mesh = bar.difference([hole_1, hole_2])

        # Export to STL
        bar_mesh.export(filename)
        return bar_mesh

    @doc_category("Other")
    def generate_stl_bars(
        self,
        scale: float = 1.0,
        width_of_bars: float = 8.0,
        height_of_bars: float = 3.0,
        holes_diameter: float = 4.3,
        filename_prefix: str = "bar_",
        output_dir: str = "stl_output",
    ) -> None:
        """
        Generate STL files for the bars of the framework.

        The STL files are generated in the working folder.
        The naming convention for the files is ``bar_i-j.stl``,
        where ``i`` and ``j`` are the vertices of an edge.

        Parameters
        ----------
        scale:
            Scale factor for the lengths of the edges, default is 1.0.
        width_of_bars:
            Width of the bars, default is 8.0 mm.
        height_of_bars:
            Height of the bars, default is 3.0 mm.
        holes_diameter:
            Diameter of the holes at the ends of the bars, default is 4.3 mm.
        filename_prefix:
            Prefix for the filenames of the generated STL files, default is ``bar_``.
        output_dir:
            Name or path of the folder where the STL files are saved,
            default is ``stl_output``. Relative to the working directory.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (0,3)])
        >>> F = Framework(G, {0:[0,0], 1:[1,0], 2:[1,'1/2 * sqrt(5)'], 3:[1/2,'4/3']})
        >>> F.generate_stl_bars(scale=20)
        STL files for the bars have been generated in the folder `stl_output`.
        """
        from pathlib import Path as plPath

        # Create the folder if it does not exist
        folder_path = plPath(output_dir)
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)

        edges_with_lengths = self.edge_lengths()

        for edge, length in edges_with_lengths.items():
            scaled_length = length * scale
            f_name = (
                output_dir
                + "/"
                + filename_prefix
                + str(edge[0])
                + "-"
                + str(edge[1])
                + ".stl"
            )

            self._generate_stl_bar(
                holes_distance=scaled_length,
                holes_diameter=holes_diameter,
                bar_width=width_of_bars,
                bar_height=height_of_bars,
                filename=f_name,
            )

        print(
            f"STL files for the bars have been generated in the folder `{output_dir}`."
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_vector_inf_flex)
    def is_vector_inf_flex(
        self,
        inf_flex: Sequence[Number],
        vertex_order: Sequence[Vertex] = None,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        return infinitesimal_rigidity.is_vector_inf_flex(
            self,
            inf_flex=inf_flex,
            vertex_order=vertex_order,
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_dict_inf_flex)
    def is_dict_inf_flex(
        self, vert_to_flex: dict[Vertex, Sequence[Number]], **kwargs
    ) -> bool:
        return infinitesimal_rigidity.is_dict_inf_flex(
            self, vert_to_flex=vert_to_flex, **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_vector_nontrivial_inf_flex)
    def is_vector_nontrivial_inf_flex(
        self,
        inf_flex: Sequence[Number],
        vertex_order: Sequence[Vertex] = None,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        return infinitesimal_rigidity.is_vector_nontrivial_inf_flex(
            self,
            inf_flex=inf_flex,
            vertex_order=vertex_order,
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_dict_nontrivial_inf_flex)
    def is_dict_nontrivial_inf_flex(
        self, vert_to_flex: dict[Vertex, Sequence[Number]], **kwargs
    ) -> bool:
        return infinitesimal_rigidity.is_dict_nontrivial_inf_flex(
            self, vert_to_flex=vert_to_flex, **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_nontrivial_flex)
    def is_nontrivial_flex(
        self,
        inf_flex: InfFlex,
        **kwargs,
    ) -> bool:
        return infinitesimal_rigidity.is_nontrivial_flex(
            self, inf_flex=inf_flex, **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_vector_trivial_inf_flex)
    def is_vector_trivial_inf_flex(self, inf_flex: Sequence[Number], **kwargs) -> bool:
        return infinitesimal_rigidity.is_vector_trivial_inf_flex(
            self, inf_flex=inf_flex, **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_dict_trivial_inf_flex)
    def is_dict_trivial_inf_flex(
        self, inf_flex: dict[Vertex, Sequence[Number]], **kwargs
    ) -> bool:
        return infinitesimal_rigidity.is_dict_trivial_inf_flex(
            self, inf_flex=inf_flex, **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_trivial_flex)
    def is_trivial_flex(
        self,
        inf_flex: InfFlex,
        **kwargs,
    ) -> bool:
        return infinitesimal_rigidity.is_trivial_flex(self, inf_flex=inf_flex, **kwargs)


Framework.__doc__ = Framework.__doc__.replace(
    "METHODS",
    _generate_category_tables(
        Framework,
        1,
        [
            "Attribute getters",
            "Framework properties",
            "Class methods",
            "Framework manipulation",
            "Infinitesimal rigidity",
            "Plotting",
            "Other",
            "Waiting for implementation",
        ],
        include_all=False,
    ),
)
