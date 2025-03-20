"""
Module for the functionality concerning frameworks.
"""

from __future__ import annotations

import functools
from copy import deepcopy
from itertools import combinations
from random import randrange
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy import Matrix, flatten, binomial

from pyrigi.data_type import (
    Vertex,
    Edge,
    Point,
    InfFlex,
    Stress,
    Sequence,
    Number,
    DirectedEdge,
)
from pyrigi.graph import Graph
from pyrigi.graphDB import Complete as CompleteGraph
from pyrigi.misc import (
    _generate_category_tables,
    is_zero,
    is_zero_vector,
    _generate_two_orthonormal_vectors,
    _generate_three_orthonormal_vectors,
    sympy_expr_to_float,
    point_to_vector,
    _null_space,
)
from pyrigi.misc import _doc_category as doc_category
import pyrigi._input_check as _input_check


__doctest_requires__ = {
    ("Framework.generate_stl_bars",): ["trimesh", "manifold3d", "pathlib"]
}

from pyrigi.plot_style import PlotStyle, PlotStyle2D, PlotStyle3D


class Framework(object):
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

    def __init__(self, graph: Graph, realization: dict[Vertex, Point]) -> None:
        if not isinstance(graph, Graph):
            raise TypeError("The graph has to be an instance of class Graph.")
        graph._input_check_no_loop()
        if not len(realization.keys()) == graph.number_of_nodes():
            raise KeyError(
                "The length of realization has to be equal to "
                "the number of vertices of graph."
            )

        if realization:
            self._dim = len(list(realization.values())[0])
        else:
            self._dim = 0

        self._graph = deepcopy(graph)
        self._realization = {}
        self.set_realization(realization)

    def __str__(self) -> str:
        """Return the string representation."""
        return (
            self.__class__.__name__
            + f" in {self.dim}-dimensional space consisting of:\n{self._graph}\n"
            + "Realization {"
            + ", ".join(
                [
                    f"{v}:{tuple(self._realization[v])}"
                    for v in self._graph.vertex_list()
                ]
            )
            + "}"
        )

    def __repr__(self) -> str:
        """Return a representation of the framework."""
        str_realization = {
            v: [str(p) for p in pos]
            for v, pos in self.realization(as_points=True).items()
        }
        return f"Framework({repr(self.graph)}, {str_realization})"

    def __getitem__(self, vertex: Vertex) -> Matrix:
        """
        Return the coordinates of a given vertex in the realization.

        Parameters
        ----------
        vertex

        Examples
        --------
        >>> F = Framework(Graph([[0,1]]), {0:[1,2], 1:[0,5]})
        >>> F[0]
        Matrix([
        [1],
        [2]])
        """
        return self._realization[vertex]

    @property
    def dim(self) -> int:
        """Return the dimension of the framework."""
        return self._dim

    @property
    def graph(self) -> Graph:
        """
        Return a copy of the underlying graph.

        Examples
        ----
        >>> F = Framework.Random(Graph([(0,1), (1,2), (0,2)]))
        >>> print(F.graph)
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        """
        return deepcopy(self._graph)

    @doc_category("Framework manipulation")
    def add_vertex(self, point: Point, vertex: Vertex = None) -> None:
        """
        Add a vertex to the framework with the corresponding coordinates.

        If no vertex is provided (``None``),
        then an integer is chosen instead.

        Parameters
        ----------
        point:
            The realization of the new vertex.
        vertex:
            The label of the new vertex.

        Examples
        --------
        >>> F = Framework.Empty(dim=2)
        >>> F.add_vertex((1.5,2), 'a')
        >>> F.add_vertex((3,1))
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices ['a', 1] and edges []
        Realization {a:(1.50000000000000, 2), 1:(3, 1)}
        """
        if vertex is None:
            candidate = self._graph.number_of_nodes()
            while self._graph.has_node(candidate):
                candidate += 1
            vertex = candidate

        if self._graph.has_node(vertex):
            raise KeyError(f"Vertex {vertex} is already a vertex of the graph!")

        self._realization[vertex] = Matrix(point)
        self._graph.add_node(vertex)

    @doc_category("Framework manipulation")
    def add_vertices(
        self, points: Sequence[Point], vertices: Sequence[Vertex] = None
    ) -> None:
        r"""
        Add a list of vertices to the framework.

        Parameters
        ----------
        points:
            List of points consisting of coordinates in $\RR^d$. It is checked
            that all points lie in the same ambient space.
        vertices:
            List of vertices. If the list of vertices is empty, we generate
            vertices with the method :meth:`add_vertex`.
            Otherwise, the list of vertices needs to have the same length as the
            list of points.

        Examples
        --------
        >>> F = Framework.Empty(dim=2)
        >>> F.add_vertices([(1.5,2), (3,1)], ['a',0])
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices ['a', 0] and edges []
        Realization {a:(1.50000000000000, 2), 0:(3, 1)}

        Notes
        -----
        For each vertex that has to be added, :meth:`add_vertex` is called.
        """
        if vertices and not len(points) == len(vertices):
            raise IndexError("The vertex list does not have the correct length!")
        if not vertices:
            for point in points:
                self.add_vertex(point)
        else:
            for point, v in zip(points, vertices):
                self.add_vertex(point, v)

    @doc_category("Framework manipulation")
    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge to the framework.

        Parameters
        ----------
        edge:

        Notes
        -----
        This method only alters the graph attribute.
        """
        self._graph._input_check_edge_format(edge, loopfree=True)
        self._graph.add_edge(*edge)

    @doc_category("Framework manipulation")
    def add_edges(self, edges: Sequence[Edge]) -> None:
        """
        Add a list of edges to the framework.

        Parameters
        ----------
        edges:

        Notes
        -----
        For each edge that has to be added, :meth:`add_edge` is called.
        """
        for edge in edges:
            self.add_edge(edge)

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

        from pyrigi import _plot

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
        edge_colors_custom: (
            Sequence[Sequence[Edge]] | dict[str : Sequence[Edge]]
        ) = None,
        stress_label_positions: dict[DirectedEdge, float] = None,
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

        from pyrigi import _plot

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
        vertex_style: str | dict[str : Sequence[Vertex]] = "fvertex",
        edge_style: str | dict[str : Sequence[Edge]] = "edge",
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
        cls, graph: Graph, dim: int = 2, rand_range: int | Sequence[int] = None
    ) -> Framework:
        """
        Return a framework with random realization with integral coordinates.

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
            ``a = 10^4 * n * dim``, where ``n`` is the number of vertices.

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

        realization = {v: [randrange(a, b) for _ in range(dim)] for v in graph.nodes}

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
    def Empty(cls, dim: int = 2) -> Framework:
        """
        Generate an empty framework.

        Parameters
        ----------
        dim:
            A natural number that determines the dimension
            in which the framework is realized.

        Examples
        ----
        >>> F = Framework.Empty(dim=1); print(F)
        Framework in 1-dimensional space consisting of:
        Graph with vertices [] and edges []
        Realization {}
        """
        _input_check.dimension(dim)
        F = Framework(graph=Graph(), realization={})
        F._dim = dim
        return F

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

    @doc_category("Framework manipulation")
    def delete_vertex(self, vertex: Vertex) -> None:
        """
        Delete a vertex from the framework.

        Parameters
        ----------
        vertex
        """
        self._graph.delete_vertex(vertex)
        del self._realization[vertex]

    @doc_category("Framework manipulation")
    def delete_vertices(self, vertices: Sequence[Vertex]) -> None:
        """
        Delete a list of vertices from the framework.

        Parameters
        ----------
        vertices
        """
        for vertex in vertices:
            self.delete_vertex(vertex)

    @doc_category("Framework manipulation")
    def delete_edge(self, edge: Edge) -> None:
        """
        Delete an edge from the framework.

        Parameters
        ----------
        edge
        """
        self._graph.delete_edge(edge)

    @doc_category("Framework manipulation")
    def delete_edges(self, edges: Sequence[Edge]) -> None:
        """
        Delete a list of edges from the framework.

        Parameters
        ----------
        edges
        """
        self._graph.delete_edges(edges)

    @doc_category("Attribute getters")
    def realization(
        self, as_points: bool = False, numerical: bool = False
    ) -> dict[Vertex, Point] | dict[Vertex, Matrix]:
        """
        Return a copy of the realization.

        Parameters
        ----------
        as_points:
            If ``True``, then the vertex positions type is
            :obj:`pyrigi.data_type.Point`,
            otherwise :obj:`Matrix <~sympy.matrices.dense.MutableDenseMatrix>` (default).
        numerical:
            If ``True``, the vertex positions are converted to floats.

        Examples
        --------
        >>> F = Framework.Complete([(0,0), (1,0), (1,1)])
        >>> F.realization(as_points=True)
        {0: [0, 0], 1: [1, 0], 2: [1, 1]}
        >>> F.realization()
        {0: Matrix([
        [0],
        [0]]), 1: Matrix([
        [1],
        [0]]), 2: Matrix([
        [1],
        [1]])}
        """
        if not numerical:
            if not as_points:
                return deepcopy(self._realization)
            return {v: list(pos) for v, pos in self._realization.items()}
        else:
            if not as_points:
                return {
                    v: Matrix([float(coord) for coord in pos])
                    for v, pos in self._realization.items()
                }
            return {
                v: [float(coord) for coord in pos]
                for v, pos in self._realization.items()
            }

    @doc_category("Framework properties")
    def is_quasi_injective(
        self, numerical: bool = False, tolerance: float = 1e-9
    ) -> bool:
        """
        Return whether the realization is quasi-injective.

        Definitions
        -----------
        :prf:ref:`Quasi-injectivity <def-realization>`

        Parameters
        ----------
        numerical:
            Whether the check is symbolic (default) or numerical.
        tolerance:
            Used tolerance when checking numerically.

        Notes
        -----
        For comparing whether two vectors are the same,
        :func:`.misc.is_zero_vector` is used.
        See its documentation for the description of the parameters.
        """

        for u, v in self.graph.edges:
            edge_vector = self[u] - self[v]
            if is_zero_vector(edge_vector, numerical, tolerance):
                return False
        return True

    @doc_category("Framework properties")
    def is_injective(self, numerical: bool = False, tolerance: float = 1e-9) -> bool:
        """
        Return whether the realization is injective.

        Parameters
        ----------
        numerical:
            Whether the check is symbolic (default) or numerical.
        tolerance:
            Used tolerance when checking numerically.

        Notes
        -----
        For comparing whether two vectors are the same,
        :func:`.misc.is_zero_vector` is used.
        See its documentation for the description of the parameters.
        """

        for u, v in combinations(self._graph.nodes, 2):
            edge_vector = self[u] - self[v]
            if is_zero_vector(edge_vector, numerical, tolerance):
                return False
        return True

    @doc_category("Framework manipulation")
    def set_realization(self, realization: dict[Vertex, Point]) -> None:
        r"""
        Change the realization of the framework.

        Definitions
        -----------
        :prf:ref:`Realization <def-realization>`

        Parameters
        ----------
        realization:
            A realization of the underlying graph of the framework.
            It must contain all vertices from the underlying graph.
            Furthermore, all points in the realization need
            to be contained in $\RR^d$ for $d$ being
            the current dimension of the framework.

        Examples
        --------
        >>> F = Framework.Complete([(0,0), (1,0), (1,1)])
        >>> F.set_realization(
        ...     {vertex: (vertex, vertex + 1) for vertex in F.graph.vertex_list()}
        ... )
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        Realization {0:(0, 1), 1:(1, 2), 2:(2, 3)}
        """
        if not len(realization) == self._graph.number_of_nodes():
            raise IndexError(
                "The realization does not contain the correct amount of vertices!"
            )
        for v in self._graph.nodes:
            self._input_check_vertex_key(v, realization)
            self._input_check_point_dimension(realization[v])

        self._realization = {v: Matrix(pos) for v, pos in realization.items()}

    @doc_category("Framework manipulation")
    def set_vertex_pos(self, vertex: Vertex, point: Point) -> None:
        """
        Change the coordinates of a single given vertex.

        Parameters
        ----------
        vertex:
            A vertex whose position is changed.
        point:
            A new position of the ``vertex``.

        Examples
        --------
        >>> F = Framework.from_points([(0,0)])
        >>> F.set_vertex_pos(0, (6,2))
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0] and edges []
        Realization {0:(6, 2)}
        """
        self._input_check_vertex_key(vertex)
        self._input_check_point_dimension(point)

        self._realization[vertex] = Matrix(point)

    @doc_category("Framework manipulation")
    def set_vertex_positions_from_lists(
        self, vertices: Sequence[Vertex], points: Sequence[Point]
    ) -> None:
        """
        Change the coordinates of a given list of vertices.

        It is necessary that both lists have the same length.
        No vertex from ``vertices`` can be contained multiple times.
        We apply the method :meth:`~Framework.set_vertex_positions`
        to the corresponding pairs of ``vertices`` and ``points``.

        Parameters
        ----------
        vertices
        points

        Examples
        --------
        >>> F = Framework.Complete([(0,0),(0,0),(1,0),(1,0)])
        >>> F.realization(as_points=True)
        {0: [0, 0], 1: [0, 0], 2: [1, 0], 3: [1, 0]}
        >>> F.set_vertex_positions_from_lists([1,3], [(0,1),(1,1)])
        >>> F.realization(as_points=True)
        {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
        """
        if len(list(set(vertices))) != len(list(vertices)):
            raise ValueError("Multiple Vertices with the same name were found!")
        if not len(vertices) == len(points):
            raise IndexError(
                "The list of vertices does not have the same length "
                "as the list of points!"
            )
        self.set_vertex_positions({v: pos for v, pos in zip(vertices, points)})

    @doc_category("Framework manipulation")
    def set_vertex_positions(self, subset_of_realization: dict[Vertex, Point]) -> None:
        """
        Change the coordinates of vertices given by a dictionary.

        Parameters
        ----------
        subset_of_realization

        Examples
        --------
        >>> F = Framework.Complete([(0,0),(0,0),(1,0),(1,0)])
        >>> F.realization(as_points=True)
        {0: [0, 0], 1: [0, 0], 2: [1, 0], 3: [1, 0]}
        >>> F.set_vertex_positions({1:(0,1),3:(1,1)})
        >>> F.realization(as_points=True)
        {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
        """
        for v, pos in subset_of_realization.items():
            self.set_vertex_pos(v, pos)

    @doc_category("Infinitesimal rigidity")
    def rigidity_matrix(
        self,
        vertex_order: Sequence[Vertex] = None,
        edge_order: Sequence[Edge] = None,
    ) -> Matrix:
        r"""
        Construct the rigidity matrix of the framework.

        Definitions
        -----------
        * :prf:ref:`Rigidity matrix <def-rigidity-matrix>`

        Parameters
        ----------
        vertex_order:
            A list of vertices, providing the ordering for the columns
            of the rigidity matrix.
            If none is provided, the list from :meth:`.Graph.vertex_list` is taken.
        edge_order:
            A list of edges, providing the ordering for the rows
            of the rigidity matrix.
            If none is provided, the list from :meth:`.Graph.edge_list` is taken.

        Examples
        --------
        >>> F = Framework.Complete([(0,0),(2,0),(1,3)])
        >>> F.rigidity_matrix()
        Matrix([
        [-2,  0, 2,  0,  0, 0],
        [-1, -3, 0,  0,  1, 3],
        [ 0,  0, 1, -3, -1, 3]])
        """
        vertex_order = self._graph._input_check_vertex_order(vertex_order)
        edge_order = self._graph._input_check_edge_order(edge_order)

        # ``delta`` is responsible for distinguishing the edges (i,j) and (j,i)
        def delta(e, w):
            # the parameter e represents an edge
            # the parameter w represents a vertex
            if w == e[0]:
                return 1
            if w == e[1]:
                return -1
            return 0

        return Matrix(
            [
                flatten(
                    [
                        delta(e, w)
                        * (self._realization[e[0]] - self._realization[e[1]])
                        for w in vertex_order
                    ]
                )
                for e in edge_order
            ]
        )

    @doc_category("Infinitesimal rigidity")
    def is_dict_stress(self, dict_stress: dict[Edge, Number], **kwargs) -> bool:
        """
        Return whether a dictionary specifies an equilibrium stress of the framework.

        Definitions
        -----------
        :prf:ref:`Equilibrium Stress <def-equilibrium-stress>`

        Parameters
        ----------
        dict_stress:
            Dictionary that maps the edges to stress values.

        Examples
        --------
        >>> F = Framework.Complete([[0,0], [1,0], ['1/2',0]])
        >>> F.is_dict_stress({(0,1):'-1/2', (0,2):1, (1,2):1})
        True
        >>> F.is_dict_stress({(0,1):1, (1,2):'-1/2', (0,2):1})
        False

        Notes
        -----
        See :meth:`.is_vector_stress`.
        """
        stress_edge_list = [tuple(e) for e in list(dict_stress.keys())]
        self._graph._input_check_edge_order(stress_edge_list, "dict_stress")
        graph_edge_list = [tuple(e) for e in self._graph.edge_list()]
        dict_to_list = []

        for e in graph_edge_list:
            dict_to_list += [
                (
                    dict_stress[e]
                    if e in stress_edge_list
                    else dict_stress[tuple([e[1], e[0]])]
                )
            ]

        return self.is_vector_stress(
            dict_to_list, edge_order=self._graph.edge_list(), **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    def is_vector_stress(
        self,
        stress: Sequence[Number],
        edge_order: Sequence[Edge] = None,
        numerical: bool = False,
        tolerance=1e-9,
    ) -> bool:
        r"""
        Return whether a vector is an equilibrium stress.

        Definitions
        -----------
        :prf:ref:`Equilibrium stress <def-equilibrium-stress>`

        Parameters
        ----------
        stress:
            A vector to be checked whether it is a stress of the framework.
        edge_order:
            A list of edges, providing the ordering for the entries of the ``stress``.
            If none is provided, the list from :meth:`.Graph.edge_list` is taken.
        numerical:
            A Boolean determining whether the evaluation of the product of the ``stress``
            and the rigidity matrix is symbolic or numerical.
        tolerance:
            Absolute tolerance that is the threshold for acceptable equilibrium
            stresses. This parameter is used to determine the number of digits,
            to which accuracy the symbolic expressions are evaluated.

        Examples
        --------
        >>> G = Graph([[0,1],[0,2],[0,3],[1,2],[2,3],[3,1]])
        >>> pos = {0: (0, 0), 1: (0,1), 2: (-1,-1), 3: (1,-1)}
        >>> F = Framework(G, pos)
        >>> omega1 = [-8, -4, -4, 2, 2, 1]
        >>> F.is_stress(omega1)
        True
        >>> omega1[0] = 0
        >>> F.is_stress(omega1)
        False
        >>> from pyrigi import frameworkDB
        >>> F = frameworkDB.Complete(5, dim=2)
        >>> stresses=F.stresses()
        >>> F.is_stress(stresses[0])
        True
        """
        edge_order = self._graph._input_check_edge_order(edge_order=edge_order)
        return is_zero_vector(
            Matrix(stress).transpose() * self.rigidity_matrix(edge_order=edge_order),
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Infinitesimal rigidity")
    def is_stress(self, stress: Stress, **kwargs) -> bool:
        """
        Alias for :meth:`.is_vector_stress` and
        :meth:`.is_dict_stress`.

        One of the alias methods is called depending on the type of the input.

        Parameters
        ----------
        stress
        """
        if isinstance(stress, list | Matrix):
            return self.is_vector_stress(stress, **kwargs)
        elif isinstance(stress, dict):
            return self.is_dict_stress(stress, **kwargs)
        else:
            raise TypeError(
                "The `stress` must be specified either by a list/Matrix or a dictionary!"
            )

    @doc_category("Infinitesimal rigidity")
    def stress_matrix(
        self,
        stress: Stress,
        edge_order: Sequence[Edge] = None,
        vertex_order: Sequence[Vertex] = None,
    ) -> Matrix:
        r"""
        Construct the stress matrix of a stress.

        Definitions
        -----
        * :prf:ref:`Stress Matrix <def-stress-matrix>`

        Parameters
        ----------
        stress:
            A stress of the framework given as a vector.
        edge_order:
            A list of edges, providing the ordering of edges in ``stress``.
            If ``None``, :meth:`.Graph.edge_list` is assumed.
        vertex_order:
            Specification of row/column order of the stress matrix.
            If ``None``, :meth:`.Graph.vertex_list` is assumed.

        Examples
        --------
        >>> G = Graph([[0,1],[0,2],[0,3],[1,2],[2,3],[3,1]])
        >>> pos = {0: (0, 0), 1: (0,1), 2: (-1,-1), 3: (1,-1)}
        >>> F = Framework(G, pos)
        >>> omega = [-8, -4, -4, 2, 2, 1]
        >>> F.stress_matrix(omega)
        Matrix([
        [-16,  8,  4,  4],
        [  8, -4, -2, -2],
        [  4, -2, -1, -1],
        [  4, -2, -1, -1]])
        """
        vertex_order = self._graph._input_check_vertex_order(vertex_order)
        edge_order = self._graph._input_check_edge_order(edge_order)
        if not self.is_stress(stress, edge_order=edge_order, numerical=True):
            raise ValueError(
                "The provided stress does not lie in the cokernel of the rigidity matrix!"
            )
        # creation of a zero |V|x|V| matrix
        stress_matr = sp.zeros(len(self._graph))
        v_to_i = {v: i for i, v in enumerate(vertex_order)}

        for edge, edge_stress in zip(edge_order, stress):
            for v in edge:
                stress_matr[v_to_i[v], v_to_i[v]] += edge_stress

        for e, stressval in zip(edge_order, stress):
            i, j = v_to_i[e[0]], v_to_i[e[1]]
            stress_matr[i, j] = -stressval
            stress_matr[j, i] = -stressval

        return stress_matr

    @doc_category("Infinitesimal rigidity")
    def trivial_inf_flexes(self, vertex_order: Sequence[Vertex] = None) -> list[Matrix]:
        r"""
        Return a basis of the vector subspace of trivial infinitesimal flexes.

        Definitions
        -----------
        :prf:ref:`Trivial infinitesimal flexes <def-trivial-inf-flex>`

        Parameters
        ----------
        vertex_order:
            A list of vertices, providing the ordering for the entries
            of the infinitesimal flexes.

        Examples
        --------
        >>> F = Framework.Complete([(0,0), (2,0), (0,2)])
        >>> F.trivial_inf_flexes()
        [Matrix([
        [1],
        [0],
        [1],
        [0],
        [1],
        [0]]), Matrix([
        [0],
        [1],
        [0],
        [1],
        [0],
        [1]]), Matrix([
        [ 0],
        [ 0],
        [ 0],
        [ 2],
        [-2],
        [ 0]])]
        """
        vertex_order = self._graph._input_check_vertex_order(vertex_order)
        dim = self._dim
        translations = [
            Matrix.vstack(*[A for _ in vertex_order])
            for A in Matrix.eye(dim).columnspace()
        ]
        basis_skew_symmetric = []
        for i in range(1, dim):
            for j in range(i):
                A = Matrix.zeros(dim)
                A[i, j] = 1
                A[j, i] = -1
                basis_skew_symmetric += [A]
        inf_rot = [
            Matrix.vstack(*[A * self._realization[v] for v in vertex_order])
            for A in basis_skew_symmetric
        ]
        matrix_inf_flexes = Matrix.hstack(*(translations + inf_rot))
        return matrix_inf_flexes.transpose().echelon_form().transpose().columnspace()

    @doc_category("Infinitesimal rigidity")
    def nontrivial_inf_flexes(self, **kwargs) -> list[Matrix]:
        """
        Return non-trivial infinitesimal flexes.

        See :meth:`~Framework.inf_flexes` for possible keywords.

        Definitions
        -----------
        :prf:ref:`Infinitesimal flex <def-inf-rigid-framework>`

        Parameters
        ----------
        vertex_order:
            A list of vertices, providing the ordering for the entries
            of the infinitesimal flexes.
            If ``None``, the list from :meth:`.Graph.vertex_list` is taken.

        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> F = Framework.Circular(graphs.CompleteBipartite(3, 3))
        >>> F.nontrivial_inf_flexes()
        [Matrix([
        [       3/2],
        [-sqrt(3)/2],
        [         1],
        [         0],
        [         0],
        [         0],
        [       3/2],
        [-sqrt(3)/2],
        [         1],
        [         0],
        [         0],
        [         0]])]
        """
        return self.inf_flexes(include_trivial=False, **kwargs)

    @doc_category("Infinitesimal rigidity")
    def inf_flexes(
        self,
        include_trivial: bool = False,
        vertex_order: Sequence[Vertex] = None,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> list[Matrix] | list[list[float]]:
        r"""
        Return a basis of the space of infinitesimal flexes.

        Return a lift of a basis of the quotient of
        the vector space of infinitesimal flexes
        modulo trivial infinitesimal flexes, if ``include_trivial=False``.
        Return a basis of the vector space of infinitesimal flexes
        if ``include_trivial=True``.

        Definitions
        -----------
        :prf:ref:`Infinitesimal flex <def-inf-flex>`

        Parameters
        ----------
        include_trivial:
            Boolean that decides, whether the trivial flexes should
            be included.
        vertex_order:
            A list of vertices, providing the ordering for the entries
            of the infinitesimal flexes.
            If none is provided, the list from :meth:`.Graph.vertex_list` is taken.
        numerical:
            Determines whether the output is symbolic (default) or numerical.
        tolerance
            Used tolerance when computing the infinitesimal flex numerically.

        Examples
        --------
        >>> F = Framework.Complete([[0,0], [1,0], [1,1], [0,1]])
        >>> F.delete_edges([(0,2), (1,3)])
        >>> F.inf_flexes(include_trivial=False)
        [Matrix([
        [1],
        [0],
        [1],
        [0],
        [0],
        [0],
        [0],
        [0]])]
        >>> F = Framework(
        ...     Graph([[0, 1], [0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]]),
        ...     {0: [0, 0], 1: [0, 1], 2: [0, 2], 3: [1, 2], 4: [-1, 2]},
        ... )
        >>> F.inf_flexes()
        [Matrix([
        [0],
        [0],
        [0],
        [0],
        [0],
        [1],
        [0],
        [0],
        [0],
        [0]])]
        """
        vertex_order = self._graph._input_check_vertex_order(vertex_order)
        if include_trivial:
            if not numerical:
                return self.rigidity_matrix(vertex_order=vertex_order).nullspace()
            else:
                F = Framework(
                    self._graph, self.realization(as_points=True, numerical=True)
                )
                return _null_space(
                    np.array(F.rigidity_matrix(vertex_order=vertex_order)).astype(
                        np.float64
                    )
                )

        if not numerical:
            rigidity_matrix = self.rigidity_matrix(vertex_order=vertex_order)

            all_inf_flexes = rigidity_matrix.nullspace()
            trivial_inf_flexes = self.trivial_inf_flexes(vertex_order=vertex_order)
            s = len(trivial_inf_flexes)
            extend_basis_matrix = Matrix.hstack(*trivial_inf_flexes)
            for inf_flex in all_inf_flexes:
                tmp_matrix = Matrix.hstack(extend_basis_matrix, inf_flex)
                if not tmp_matrix.rank() == extend_basis_matrix.rank():
                    extend_basis_matrix = Matrix.hstack(extend_basis_matrix, inf_flex)
            basis = extend_basis_matrix.columnspace()
            return basis[s:]
        else:
            F = Framework(self._graph, self.realization(as_points=True, numerical=True))
            inf_flexes = _null_space(
                np.array(F.rigidity_matrix(vertex_order=vertex_order)).astype(
                    np.float64
                ),
                tolerance=tolerance,
            )
            inf_flexes = [inf_flexes[:, i] for i in range(inf_flexes.shape[1])]
            Kn = Framework(
                CompleteGraph(len(self._graph)),
                self.realization(as_points=True, numerical=True),
            )
            inf_flexes_trivial = _null_space(
                np.array(Kn.rigidity_matrix(vertex_order=vertex_order)).astype(
                    np.float64
                ),
                tolerance=tolerance,
            )
            s = inf_flexes_trivial.shape[1]
            extend_basis_matrix = inf_flexes_trivial
            for inf_flex in inf_flexes:
                inf_flex = np.reshape(inf_flex, (-1, 1))
                tmp_matrix = np.hstack((inf_flexes_trivial, inf_flex))
                if not np.linalg.matrix_rank(
                    tmp_matrix, tol=tolerance
                ) == np.linalg.matrix_rank(inf_flexes_trivial, tol=tolerance):
                    extend_basis_matrix = np.hstack((extend_basis_matrix, inf_flex))
            Q, R = np.linalg.qr(extend_basis_matrix)
            Q = Q[:, s : np.linalg.matrix_rank(R, tol=tolerance)]
            return [list(Q[:, i]) for i in range(Q.shape[1])]

    @doc_category("Infinitesimal rigidity")
    def stresses(
        self,
        edge_order: Sequence[Edge] = None,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> list[Matrix] | list[list[float]]:
        r"""
        Return a basis of the space of equilibrium stresses.

        Definitions
        -----------
        :prf:ref:`Equilibrium stress <def-equilibrium-stress>`

        Parameters
        ----------
        edge_order:
            A list of edges, providing the ordering for the entries of the stresses.
            If none is provided, the list from :meth:`.Graph.edge_list` is taken.
        numerical:
            Determines whether the output is symbolic (default) or numerical.
        tolerance:
            Used tolerance when computing the stresses numerically.

        Examples
        --------
        >>> G = Graph([[0,1],[0,2],[0,3],[1,2],[2,3],[3,1]])
        >>> pos = {0: (0, 0), 1: (0,1), 2: (-1,-1), 3: (1,-1)}
        >>> F = Framework(G, pos)
        >>> F.stresses()
        [Matrix([
        [-8],
        [-4],
        [-4],
        [ 2],
        [ 2],
        [ 1]])]
        """
        if not numerical:
            return self.rigidity_matrix(edge_order=edge_order).transpose().nullspace()
        F = Framework(self._graph, self.realization(as_points=True, numerical=True))
        stresses = _null_space(
            np.array(F.rigidity_matrix(edge_order=edge_order).transpose()).astype(
                np.float64
            ),
            tolerance=tolerance,
        )
        return [list(stresses[:, i]) for i in range(stresses.shape[1])]

    @doc_category("Infinitesimal rigidity")
    def rigidity_matrix_rank(self) -> int:
        """
        Return the rank of the rigidity matrix.

        Definitions
        -----------
        :prf:ref:`Rigidity matrix <def-rigidity-matrix>`

        Examples
        --------
        >>> K4 = Framework.Complete([[0,0], [1,0], [1,1], [0,1]])
        >>> K4.rigidity_matrix_rank()   # the complete graph is a circuit
        5
        >>> K4.delete_edge([0,1])
        >>> K4.rigidity_matrix_rank()   # deleting a bar gives full rank
        5
        >>> K4.delete_edge([2,3])
        >>> K4.rigidity_matrix_rank()   #so now deleting an edge lowers the rank
        4
        """
        return self.rigidity_matrix().rank()

    @doc_category("Infinitesimal rigidity")
    def is_inf_rigid(self) -> bool:
        """
        Return whether the framework is infinitesimally rigid.

        The check is based on :meth:`~Framework.rigidity_matrix_rank`.

        Definitions
        -----------
        :prf:ref:`Infinitesimal rigidity <def-inf-rigid-framework>`

        Examples
        --------
        >>> from pyrigi import frameworkDB
        >>> F1 = frameworkDB.CompleteBipartite(4,4)
        >>> F1.is_inf_rigid()
        True
        >>> F2 = frameworkDB.Cycle(4,dim=2)
        >>> F2.is_inf_rigid()
        False
        """
        if self._graph.number_of_nodes() <= self._dim + 1:
            return self.rigidity_matrix_rank() == binomial(
                self._graph.number_of_nodes(), 2
            )
        else:
            return (
                self.rigidity_matrix_rank()
                == self.dim * self._graph.number_of_nodes() - binomial(self.dim + 1, 2)
            )

    @doc_category("Infinitesimal rigidity")
    def is_inf_flexible(self) -> bool:
        """
        Return whether the framework is infinitesimally flexible.

        Definitions
        -----------
        :prf:ref:`Infinitesimal rigidity <def-inf-rigid-framework>`
        """
        return not self.is_inf_rigid()

    @doc_category("Infinitesimal rigidity")
    def is_min_inf_rigid(self) -> bool:
        """
        Return whether the framework is minimally infinitesimally rigid.

        Definitions
        -----
        :prf:ref:`Minimal infinitesimal rigidity <def-min-rigid-framework>`

        Examples
        --------
        >>> F = Framework.Complete([[0,0], [1,0], [1,1], [0,1]])
        >>> F.is_min_inf_rigid()
        False
        >>> F.delete_edge((0,2))
        >>> F.is_min_inf_rigid()
        True
        """
        if not self.is_inf_rigid():
            return False
        for edge in self._graph.edge_list():
            self.delete_edge(edge)
            if self.is_inf_rigid():
                self.add_edge(edge)
                return False
            self.add_edge(edge)
        return True

    @doc_category("Infinitesimal rigidity")
    def is_independent(self) -> bool:
        """
        Return whether the framework is independent.

        Definitions
        -----------
        :prf:ref:`Independent framework <def-independent-framework>`

        Examples
        --------
        >>> F = Framework.Complete([[0,0], [1,0], [1,1], [0,1]])
        >>> F.is_independent()
        False
        >>> F.delete_edge((0,2))
        >>> F.is_independent()
        True
        """
        return self.rigidity_matrix_rank() == self._graph.number_of_edges()

    @doc_category("Infinitesimal rigidity")
    def is_dependent(self) -> bool:
        """
        Return whether the framework is dependent.

        See also :meth:`~.Framework.is_independent`.

        Definitions
        -----------
        :prf:ref:`Dependent framework <def-independent-framework>`
        """
        return not self.is_independent()

    @doc_category("Infinitesimal rigidity")
    def is_isostatic(self) -> bool:
        """
        Return whether the framework is isostatic.

        Definitions
        -----------
        :prf:ref:`Isostatic framework <def-isostatic-frameworks>`
        """
        return self.is_independent() and self.is_inf_rigid()

    @doc_category("Other")
    def is_prestress_stable(
        self,
        numerical: bool = False,
        tolerance: float = 1e-9,
        inf_flexes: Sequence[InfFlex] = None,
        stresses: Sequence[Stress] = None,
    ) -> bool:
        """
        Return whether the framework is prestress stable.

        See also :meth:`.is_second_order_rigid`.

        Definitions
        ----------
        :prf:ref:`Prestress stability <def-prestress-stability>`

        Parameters
        -------
        numerical:
            If ``True``, numerical infinitesimal flexes and stresses
            are used in the check for prestress stability.
            In case that ``numerical=False``, this method only
            properly works for symbolic coordinates.
        tolerance:
            Numerical tolerance used for the check that something is
            an approximate zero.
        inf_flexes, stresses:
            Precomputed infinitesimal flexes and equilibrium stresses can be provided
            to avoid recomputation. If not provided, they are computed here.

        Examples
        --------
        >>> from pyrigi import frameworkDB as fws
        >>> F = fws.Frustum(3)
        >>> F.is_prestress_stable()
        True
        """
        edges = self._graph.edge_list(as_tuples=True)
        inf_flexes = self._process_list_of_inf_flexes(
            inf_flexes, numerical=numerical, tolerance=tolerance
        )
        if len(inf_flexes) == 0:
            return True
        stresses = self._process_list_of_stresses(
            stresses, numerical=numerical, tolerance=tolerance
        )
        if len(stresses) == 0:
            return False

        if len(inf_flexes) == 1:
            q = inf_flexes[0]
            stress_energy_list = []
            for stress in stresses:
                stress_energy_list.append(
                    sum(
                        [
                            stress[(u, v)]
                            * sum(
                                [
                                    (q1 - q2) ** 2
                                    for q1, q2 in zip(
                                        q[u],
                                        q[v],
                                    )
                                ]
                            )
                            for u, v in edges
                        ]
                    )
                )
            return any(
                [
                    not is_zero(Q, numerical=numerical, tolerance=tolerance)
                    for Q in stress_energy_list
                ]
            )

        if len(stresses) == 1:
            a = sp.symbols("a0:%s" % len(inf_flexes), real=True)
            stress_energy = 0
            stress_energy += sum(
                [
                    stresses[0][(u, v)]
                    * sum(
                        [
                            (
                                sum(
                                    [
                                        a[i]
                                        * (inf_flexes[i][u][j] - inf_flexes[i][v][j])
                                        for i in range(len(inf_flexes))
                                    ]
                                )
                                ** 2
                            )
                            for j in range(self._dim)
                        ]
                    )
                    for u, v in edges
                ]
            )

            coefficients = {
                (i, j): sp.Poly(stress_energy, a).coeff_monomial(a[i] * a[j])
                for i in range(len(inf_flexes))
                for j in range(i, len(inf_flexes))
            }
            #  We then apply the SONC criterion.
            if numerical:
                return all(
                    [
                        coefficients[(i, j)] ** 2
                        < sympy_expr_to_float(
                            4 * coefficients[(i, i)] * coefficients[(j, j)]
                        )
                        for i in range(len(inf_flexes))
                        for j in range(i + 1, len(inf_flexes))
                    ]
                )
            sonc_expressions = [
                sp.simplify(
                    sp.cancel(
                        4 * coefficients[(i, i)] * coefficients[(j, j)]
                        - coefficients[(i, j)] ** 2
                    )
                )
                for i in range(len(inf_flexes))
                for j in range(i + 1, len(inf_flexes))
            ]
            if any(expr is None for expr in sonc_expressions):
                raise RuntimeError(
                    "It could not be determined by `sympy.simplify` "
                    + "whether the given sympy expression can be simplified."
                    + "Please report this as an issue on Github "
                    + "(https://github.com/PyRigi/PyRigi/issues)."
                )
            sonc_expressions = [expr.is_positive for expr in sonc_expressions]
            if any(expr is None for expr in sonc_expressions):
                raise RuntimeError(
                    "It could not be determined by `sympy.is_positive` "
                    + "whether the given sympy expression is positive."
                    + "Please report this as an issue on Github "
                    + "(https://github.com/PyRigi/PyRigi/issues)."
                )
            return all(sonc_expressions)

        raise ValueError(
            "Prestress stability is not yet implemented for the general case."
        )

    @doc_category("Other")
    def is_second_order_rigid(
        self,
        numerical: bool = False,
        tolerance: float = 1e-9,
        inf_flexes: Sequence[InfFlex] = None,
        stresses: Sequence[Stress] = None,
    ) -> bool:
        """
        Return whether the framework is second-order rigid.

        Checking second-order-rigidity for a general framework is computationally hard.
        If there is only one stress or only one infinitesimal flex, second-order rigidity
        is identical to :prf:ref:`prestress stability <def-prestress-stability>`,
        so we can apply :meth:`.is_prestress_stable`. See also
        :prf:ref:`thm-second-order-implies-prestress-stability`.

        Definitions
        ----------
        :prf:ref:`Second-order rigidity <def-second-order-rigid>`

        Parameters
        -------
        numerical:
            If ``True``, numerical infinitesimal flexes and stresses
            are used in the check for prestress stability.
            In case that ``numerical=False``, this method only
            properly works for symbolic coordinates.
        tolerance:
            Numerical tolerance used for the check that something is
            an approximate zero.
        inf_flexes, stresses:
            Precomputed infinitesimal flexes and equilibrium stresses can be provided
            to avoid recomputation. If not provided, they are computed here.

        Examples
        --------
        >>> from pyrigi import frameworkDB as fws
        >>> F = fws.Frustum(3)
        >>> F.is_second_order_rigid()
        True
        """
        inf_flexes = self._process_list_of_inf_flexes(
            inf_flexes, numerical=numerical, tolerance=tolerance
        )
        if len(inf_flexes) == 0:
            return True
        stresses = self._process_list_of_stresses(
            stresses, numerical=numerical, tolerance=tolerance
        )
        if len(stresses) == 0:
            return False

        if len(stresses) == 1 or len(inf_flexes) == 1:
            return self.is_prestress_stable(
                numerical=numerical,
                tolerance=tolerance,
                inf_flexes=inf_flexes,
                stresses=stresses,
            )

        raise ValueError("Second-order rigidity is not implemented for this framework.")

    def _process_list_of_inf_flexes(
        self,
        inf_flexes: Sequence[InfFlex],
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> list[dict[Vertex, Point]]:
        """
        Process the input infinitesimal flexes for the second-order methods.

        If any of the input is not a nontrivial flex, an error is thrown.
        Otherwise, the infinitesimal flexes are transformed to a ``list`` of
        ``dict``.

        Parameters
        ----------
        inf_flexes:
            The infinitesimal flexes to be processed.
        numerical:
            If ``True``, the check is numerical.
        tolerance:
            Numerical tolerance used for the check that something is
            a nontrivial infinitesimal flex.
        """
        if inf_flexes is None:
            inf_flexes = self.inf_flexes(numerical=numerical, tolerance=tolerance)
            if len(inf_flexes) == 0:
                return inf_flexes
        elif any(
            not self.is_nontrivial_flex(
                inf_flex, numerical=numerical, tolerance=tolerance
            )
            for inf_flex in inf_flexes
        ):
            raise ValueError(
                "Some of the provided `inf_flexes` are not "
                + "nontrivial infinitesimal flexes!"
            )
        if len(inf_flexes) == 0:
            raise ValueError("No infinitesimal flexes were provided.")
        if all(isinstance(inf_flex, list | tuple | Matrix) for inf_flex in inf_flexes):
            inf_flexes = [self._transform_inf_flex_to_pointwise(q) for q in inf_flexes]
        elif not all(isinstance(inf_flex, dict) for inf_flex in inf_flexes):
            raise ValueError(
                "The provided `inf_flexes` do not have the correct format."
            )
        return inf_flexes

    def _process_list_of_stresses(
        self,
        stresses: Sequence[Stress],
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> list[dict[Edge, Number]]:
        """
        Process the input equilibrium stresses for the second-order methods.

        If any of the input is not an equilibrium stress, an error is thrown.
        Otherwise, the equilibrium stresses are transformed to a list of
        ``dict``.

        Parameters
        ----------
        stresses:
            The equilibrium stresses to be processed.
        numerical:
            If ``True``, the check is numerical.
        tolerance:
            Numerical tolerance used for the check that something is
            an equilibrium stress.
        """
        edges = self._graph.edge_list(as_tuples=True)
        if stresses is None:
            stresses = self.stresses(numerical=numerical, tolerance=tolerance)
            if len(stresses) == 0:
                return stresses
        elif any(
            not self.is_stress(stress, numerical=numerical, tolerance=tolerance)
            for stress in stresses
        ):
            raise ValueError(
                "Some of the provided `stresses` are not equilibrium stresses!"
            )
        if len(stresses) == 0:
            raise ValueError("No equilibrium stresses were provided.")
        if all(isinstance(stress, list | tuple | Matrix) for stress in stresses):
            stresses = [
                self._transform_stress_to_edgewise(stress, edge_order=edges)
                for stress in stresses
            ]
        elif not all(isinstance(stress, dict) for stress in stresses):
            raise ValueError("The provided `stresses` do not have the correct format.")
        return stresses

    @doc_category("Infinitesimal rigidity")
    def is_redundantly_inf_rigid(self) -> bool:
        """
        Return if the framework is infinitesimally redundantly rigid.

        Definitions
        -----------
        :prf:ref:`Redundant infinitesimal rigidity <def-redundantly-rigid-framework>`

        Examples
        --------
        >>> F = Framework.Empty(dim=2)
        >>> F.add_vertices([(1,0), (1,1), (0,3), (-1,1)], ['a','b','c','d'])
        >>> F.add_edges([('a','b'), ('b','c'), ('c','d'), ('a','d'), ('a','c'), ('b','d')])
        >>> F.is_redundantly_inf_rigid()
        True
        >>> F.delete_edge(('a','c'))
        >>> F.is_redundantly_inf_rigid()
        False
        """  # noqa: E501
        for edge in self._graph.edge_list():
            self.delete_edge(edge)
            if not self.is_inf_rigid():
                self.add_edge(edge)
                return False
            self.add_edge(edge)
        return True

    @doc_category("Framework properties")
    def is_congruent_realization(
        self,
        other_realization: dict[Vertex, Point] | dict[Vertex, Matrix],
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        """
        Return whether the given realization is congruent to self.

        Definitions
        -----------
        :prf:ref:`Congruent frameworks <def-equivalent-framework>`

        Parameters
        ----------
        other_realization
            The realization for checking the congruence.
        numerical
            Whether the check is symbolic (default) or numerical.
        tolerance
            Used tolerance when checking numerically.
        """
        self._graph._input_check_vertex_order(
            list(other_realization.keys()), "other_realization"
        )

        for u, v in combinations(self._graph.nodes, 2):
            edge_vec = (self._realization[u]) - self._realization[v]
            dist_squared = (edge_vec.T * edge_vec)[0, 0]

            other_edge_vec = point_to_vector(other_realization[u]) - point_to_vector(
                other_realization[v]
            )
            otherdist_squared = (other_edge_vec.T * other_edge_vec)[0, 0]

            difference = sp.simplify(dist_squared - otherdist_squared)
            if not is_zero(difference, numerical=numerical, tolerance=tolerance):
                return False
        return True

    @doc_category("Framework properties")
    def is_congruent(
        self,
        other_framework: Framework,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        """
        Return whether the given framework is congruent to self.

        Definitions
        -----------
        :prf:ref:`Congruent frameworks <def-equivalent-framework>`

        Parameters
        ----------
        other_framework
            The framework for checking the congruence.
        numerical
            Whether the check is symbolic (default) or numerical.
        tolerance
            Used tolerance when checking numerically.
        """

        self._input_check_underlying_graphs(other_framework)

        return self.is_congruent_realization(
            other_framework._realization, numerical, tolerance
        )

    @doc_category("Framework properties")
    def is_equivalent_realization(
        self,
        other_realization: dict[Vertex, Point] | dict[Vertex, Matrix],
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        """
        Return whether the given realization is equivalent to self.

        Definitions
        -----------
        :prf:ref:`Equivalent frameworks <def-equivalent-framework>`

        Parameters
        ----------
        other_realization
            The realization for checking the equivalence.
        numerical
            Whether the check is symbolic (default) or numerical.
        tolerance
            Used tolerance when checking numerically.
        """
        self._graph._input_check_vertex_order(
            list(other_realization.keys()), "other_realization"
        )

        for u, v in self._graph.edges:
            edge_vec = self._realization[u] - self._realization[v]
            dist_squared = (edge_vec.T * edge_vec)[0, 0]

            other_edge_vec = point_to_vector(other_realization[u]) - point_to_vector(
                other_realization[v]
            )
            otherdist_squared = (other_edge_vec.T * other_edge_vec)[0, 0]

            difference = sp.simplify(otherdist_squared - dist_squared)
            if not is_zero(difference, numerical=numerical, tolerance=tolerance):
                return False
        return True

    @doc_category("Framework properties")
    def is_equivalent(
        self,
        other_framework: Framework,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        """
        Return whether the given framework is equivalent to self.

        Definitions
        -----------
        :prf:ref:`Equivalent frameworks <def-equivalent-framework>`

        Parameters
        ----------
        other_framework
            The framework for checking the equivalence.
        numerical
            Whether the check is symbolic (default) or numerical.
        tolerance
            Used tolerance when checking numerically.
        """

        self._input_check_underlying_graphs(other_framework)

        return self.is_equivalent_realization(
            other_framework._realization, numerical, tolerance
        )

    @doc_category("Framework manipulation")
    def translate(
        self, vector: Point | Matrix, inplace: bool = True
    ) -> None | Framework:
        """
        Translate the framework.

        Parameters
        ----------
        vector
            Translation vector
        inplace
            If ``True`` (default), then this framework is translated.
            Otherwise, a new translated framework is returned.
        """
        vector = point_to_vector(vector)

        if inplace:
            if vector.shape[0] != self.dim:
                raise ValueError(
                    "The dimension of the vector has to be the same as of the framework!"
                )

            for v in self._realization.keys():
                self._realization[v] += vector
            return

        new_framework = deepcopy(self)
        new_framework.translate(vector, True)
        return new_framework

    @doc_category("Framework manipulation")
    def rescale(self, factor: Number, inplace: bool = True) -> None | Framework:
        """
        Scale the framework.

        Parameters
        ----------
        factor:
            Scaling factor
        inplace:
            If ``True`` (default), then this framework is translated.
            Otherwise, a new translated framework is returned.
        """
        if isinstance(factor, str):
            factor = sp.sympify(factor)
        if inplace:
            for v in self._realization.keys():
                self._realization[v] = self._realization[v] * factor
            return

        new_framework = deepcopy(self)
        new_framework.rescale(factor, True)
        return new_framework

    @doc_category("Framework manipulation")
    def rotate2D(self, angle: float, inplace: bool = True) -> None | Framework:
        """
        Rotate the planar framework counterclockwise.

        Parameters
        ----------
        angle:
            Rotation angle
        inplace:
            If ``True`` (default), then this framework is rotated.
            Otherwise, a new rotated framework is returned.
        """

        if self.dim != 2:
            raise ValueError("This realization is not in dimension 2!")

        rotation_matrix = Matrix(
            [[sp.cos(angle), -sp.sin(angle)], [sp.sin(angle), sp.cos(angle)]]
        )

        if inplace:
            for v, pos in self._realization.items():
                self._realization[v] = rotation_matrix * pos
            return

        new_framework = deepcopy(self)
        new_framework.rotate2D(angle, True)
        return new_framework

    @doc_category("Framework manipulation")
    def projected_realization(
        self,
        proj_dim: int = None,
        projection_matrix: Matrix = None,
        random_seed: int = None,
        coordinates: Sequence[int] = None,
    ) -> tuple[dict[Vertex, Point], Matrix]:
        """
        Return the realization projected to a lower dimension and the projection matrix.

        Parameters
        ----------
        proj_dim:
            The dimension to which the framework is projected.
            This is determined from ``projection_matrix`` if it is provided.
        projection_matrix:
            The matrix used for projecting the placement of vertices.
            The matrix must have dimensions ``(proj_dim, dim)``,
            where ``dim`` is the dimension of the framework ``self``.
            If ``None``, a numerical random projection matrix is generated.
        random_seed:
            The random seed used for generating the projection matrix.
        coordinates:
            Indices of coordinates to which projection is applied.
            Providing the parameter overrides the previous ones.

        Suggested Improvements
        ----------------------
        Generate random projection matrix over symbolic rationals.
        """
        if coordinates is not None:
            if not isinstance(coordinates, tuple) and not isinstance(coordinates, list):
                raise TypeError(
                    "The parameter ``coordinates`` must be a tuple or a list."
                )
            if max(coordinates) >= self._dim:
                raise ValueError(
                    f"Index {np.max(coordinates)} out of range"
                    + f" with placement in dim: {self._dim}."
                )
            if isinstance(proj_dim, int) and len(coordinates) != proj_dim:
                raise ValueError(
                    f"The number of coordinates ({len(coordinates)}) does not match"
                    + f" proj_dim ({proj_dim})."
                )
            matrix = np.zeros((len(coordinates), self._dim))
            for i, coord in enumerate(coordinates):
                matrix[i, coord] = 1

            return (
                {
                    v: tuple([pos[coord] for coord in coordinates])
                    for v, pos in self._realization.items()
                },
                Matrix(matrix),
            )

        if projection_matrix is not None:
            projection_matrix = np.array(projection_matrix)
            if projection_matrix.shape[1] != self._dim:
                raise ValueError(
                    "The projection matrix has wrong number of columns."
                    + f"{projection_matrix.shape[1]} instead of {self._dim}."
                )
            if isinstance(proj_dim, int) and projection_matrix.shape[0] != proj_dim:
                raise ValueError(
                    "The projection matrix has wrong number of rows."
                    + f"{projection_matrix.shape[0]} instead of {self._dim}."
                )

        if projection_matrix is None:
            if proj_dim == 2:
                projection_matrix = _generate_two_orthonormal_vectors(
                    self._dim, random_seed=random_seed
                )
            elif proj_dim == 3:
                projection_matrix = _generate_three_orthonormal_vectors(
                    self._dim, random_seed=random_seed
                )
            else:
                raise ValueError(
                    "An automatically generated random matrix is supported"
                    + f" only in dimension 2 or 3. {proj_dim} was given instead."
                )
            projection_matrix = projection_matrix.T
        return (
            {
                v: tuple(
                    [float(s[0]) for s in np.dot(projection_matrix, np.array(pos))]
                )
                for v, pos in self.realization(as_points=False, numerical=True).items()
            },
            projection_matrix,
        )

    @doc_category("Other")
    def edge_lengths(self, numerical: bool = False) -> dict[Edge, Number]:
        """
        Return the dictionary of the edge lengths.

        Parameters
        -------
        numerical:
            If ``True``, numerical positions are used for the computation of the edge lengths.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (0,3)])
        >>> F = Framework(G, {0:[0,0], 1:[1,0], 2:[1,'1/2 * sqrt(5)'], 3:['1/2','4/3']})
        >>> F.edge_lengths(numerical=False)
        {(0, 1): 1, (0, 3): sqrt(73)/6, (1, 2): sqrt(5)/2, (2, 3): sqrt((-4/3 + sqrt(5)/2)**2 + 1/4)}
        >>> F.edge_lengths(numerical=True)
        {(0, 1): 1.0, (0, 3): 1.4240006242195884, (1, 2): 1.118033988749895, (2, 3): 0.5443838790578374}
        """  # noqa: E501
        if numerical:
            points = self.realization(as_points=True, numerical=True)
            return {
                tuple(e): float(
                    np.linalg.norm(np.array(points[e[0]]) - np.array(points[e[1]]))
                )
                for e in self._graph.edges
            }
        else:
            points = self.realization(as_points=True)
            return {
                tuple(e): sp.sqrt(
                    sum([(x - y) ** 2 for x, y in zip(points[e[0]], points[e[1]])])
                )
                for e in self._graph.edges
            }

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

    def _transform_inf_flex_to_pointwise(
        self, inf_flex: Matrix, vertex_order: Sequence[Vertex] = None
    ) -> dict[Vertex, list[Number]]:
        r"""
        Transform the natural data type of a flex (``Matrix``) to a
        dictionary that maps a vertex to a ``Sequence`` of coordinates
        (i.e. a vector).

        Parameters
        ----------
        inf_flex:
            An infinitesimal flex in the form of a ``Matrix``.
        vertex_order:
            If ``None``, the :meth:`.Graph.vertex_list`
            is taken as the vertex order.

        Examples
        ----
        >>> F = Framework.from_points([(0,0), (1,0), (0,1)])
        >>> F.add_edges([(0,1),(0,2)])
        >>> flex = F.nontrivial_inf_flexes()[0]
        >>> F._transform_inf_flex_to_pointwise(flex)
        {0: [1, 0], 1: [1, 0], 2: [0, 0]}

        Notes
        ----
        For example, this method can be used for generating an
        infinitesimal flex for plotting purposes.
        """
        vertex_order = self._graph._input_check_vertex_order(vertex_order)
        return {
            vertex_order[i]: [inf_flex[i * self.dim + j] for j in range(self.dim)]
            for i in range(len(vertex_order))
        }

    def _transform_stress_to_edgewise(
        self, stress: Matrix, edge_order: Sequence[Edge] = None
    ) -> dict[Edge, Number]:
        r"""
        Transform the natural data type of a stress (``Matrix``) to a
        dictionary that maps an edge to a coordinate.

        Parameters
        ----------
        stress:
            An equilibrium stress in the form of a ``Matrix``.
        edge_order:
            If ``None``, the :meth:`.Graph.edge_list`
            is taken as the edge order.

        Examples
        ----
        >>> F = Framework.Complete([(0,0),(1,0),(1,1),(0,1)])
        >>> stress = F.stresses()[0]
        >>> F._transform_stress_to_edgewise(stress)
        {(0, 1): 1, (0, 2): -1, (0, 3): 1, (1, 2): 1, (1, 3): -1, (2, 3): 1}

        Notes
        ----
        For example, this method can be used for generating an
        equilibrium stresss for plotting purposes.
        """
        edge_order = self._graph._input_check_edge_order(edge_order)
        return {tuple(edge_order[i]): stress[i] for i in range(len(edge_order))}

    @doc_category("Infinitesimal rigidity")
    def is_vector_inf_flex(
        self,
        inf_flex: Sequence[Number],
        vertex_order: Sequence[Vertex] = None,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        r"""
        Return whether a vector is an infinitesimal flex of the framework.

        Definitions
        -----------
        * :prf:ref:`Infinitesimal flex <def-inf-flex>`
        * :prf:ref:`Rigidity Matrix <def-rigidity-matrix>`

        Parameters
        ----------
        inf_flex:
            An infinitesimal flex of the framework specified by a vector.
        vertex_order:
            A list of vertices specifying the order in which ``inf_flex`` is given.
            If none is provided, the list from :meth:`~.Graph.vertex_list` is taken.
        numerical:
            A Boolean determining whether the evaluation of the product of
            the ``inf_flex`` and the rigidity matrix is symbolic or numerical.
        tolerance:
            Absolute tolerance that is the threshold for acceptable numerical flexes.
            This parameter is used to determine the number of digits, to which
            accuracy the symbolic expressions are evaluated.

        Examples
        --------
        >>> from pyrigi import frameworkDB as fws
        >>> F = fws.Square()
        >>> q = [0,0,0,0,-2,0,-2,0]
        >>> F.is_vector_inf_flex(q)
        True
        >>> q[0] = 1
        >>> F.is_vector_inf_flex(q)
        False
        >>> F = Framework.Complete([[0,0], [1,1]])
        >>> F.is_vector_inf_flex(["sqrt(2)","-sqrt(2)",0,0], vertex_order=[1,0])
        True
        """
        vertex_order = self._graph._input_check_vertex_order(vertex_order)
        return is_zero_vector(
            self.rigidity_matrix(vertex_order=vertex_order) * Matrix(inf_flex),
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Infinitesimal rigidity")
    def is_dict_inf_flex(
        self, vert_to_flex: dict[Vertex, Sequence[Number]], **kwargs
    ) -> bool:
        """
        Return whether a dictionary specifies an infinitesimal flex of the framework.

        Definitions
        -----------
        :prf:ref:`Infinitesimal flex <def-inf-flex>`

        Parameters
        ----------
        vert_to_flex:
            Dictionary that maps the vertex labels to
            vectors of the same dimension as the framework is.

        Examples
        --------
        >>> F = Framework.Complete([[0,0], [1,1]])
        >>> F.is_dict_inf_flex({0:[0,0], 1:[-1,1]})
        True
        >>> F.is_dict_inf_flex({0:[0,0], 1:["sqrt(2)","-sqrt(2)"]})
        True

        Notes
        -----
        See :meth:`.is_vector_inf_flex`.
        """
        self._graph._input_check_vertex_order(list(vert_to_flex.keys()), "vert_to_flex")

        dict_to_list = []
        for v in self._graph.vertex_list():
            dict_to_list += list(vert_to_flex[v])

        return self.is_vector_inf_flex(
            dict_to_list, vertex_order=self._graph.vertex_list(), **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    def is_vector_nontrivial_inf_flex(
        self,
        inf_flex: Sequence[Number],
        vertex_order: Sequence[Vertex] = None,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        r"""
        Return whether an infinitesimal flex is nontrivial.

        Definitions
        -----------
        :prf:ref:`Nontrivial infinitesimal flex <def-trivial-inf-flex>`

        Parameters
        ----------
        inf_flex:
            An infinitesimal flex of the framework.
        vertex_order:
            A list of vertices specifying the order in which ``inf_flex`` is given.
            If none is provided, the list from :meth:`.Graph.vertex_list` is taken.
        numerical:
            A Boolean determining whether the evaluation of the product of the `inf_flex`
            and the rigidity matrix is symbolic or numerical.
        tolerance:
            Absolute tolerance that is the threshold for acceptable numerical flexes.
            This parameter is used to determine the number of digits, to which
            accuracy the symbolic expressions are evaluated.

        Examples
        --------
        >>> from pyrigi import frameworkDB as fws
        >>> F = fws.Square()
        >>> q = [0,0,0,0,-2,0,-2,0]
        >>> F.is_vector_nontrivial_inf_flex(q)
        True
        >>> q = [1,-1,1,1,-1,1,-1,-1]
        >>> F.is_vector_inf_flex(q)
        True
        >>> F.is_vector_nontrivial_inf_flex(q)
        False

        Notes
        -----
        This is done by solving a linear system composed of a matrix $A$ whose columns
        are given by a basis of the trivial flexes and the vector $b$ given by the
        input flex. $b$ is trivial if and only if there is a linear combination of
        the columns in $A$ producing $b$. In other words, when there is a solution to
        $Ax=b$, then $b$ is a trivial infinitesimal motion. Otherwise, $b$ is
        nontrivial.

        In the ``numerical=True`` case we compute a least squares solution $x$ of the
        overdetermined linear system and compare the values in $Ax$ to the values
        in $b$.
        """
        vertex_order = self._graph._input_check_vertex_order(vertex_order)
        if not self.is_vector_inf_flex(
            inf_flex,
            vertex_order=vertex_order,
            numerical=numerical,
            tolerance=tolerance,
        ):
            return False

        if not numerical:
            Q_trivial = Matrix.hstack(
                *(self.trivial_inf_flexes(vertex_order=vertex_order))
            )
            system = Q_trivial, Matrix(inf_flex)
            return sp.linsolve(system) == sp.EmptySet
        else:
            Q_trivial = np.array(
                [
                    sympy_expr_to_float(flex, tolerance=tolerance)
                    for flex in self.trivial_inf_flexes(vertex_order=vertex_order)
                ]
            ).transpose()
            b = np.array(sympy_expr_to_float(inf_flex, tolerance=tolerance)).transpose()
            x = np.linalg.lstsq(Q_trivial, b, rcond=None)[0]
            return not is_zero_vector(
                np.dot(Q_trivial, x) - b, numerical=True, tolerance=tolerance
            )

    @doc_category("Infinitesimal rigidity")
    def is_dict_nontrivial_inf_flex(
        self, vert_to_flex: dict[Vertex, Sequence[Number]], **kwargs
    ) -> bool:
        r"""
        Return whether a dictionary specifies an infinitesimal flex which is nontrivial.

        See :meth:`.is_vector_nontrivial_inf_flex` for details,
        particularly concerning the possible parameters.

        Definitions
        -----------
        :prf:ref:`Nontrivial infinitesimal flex <def-trivial-inf-flex>`

        Parameters
        ----------
        vert_to_flex:
            An infinitesimal flex of the framework in the form of a dictionary.

        Examples
        --------
        >>> from pyrigi import frameworkDB as fws
        >>> F = fws.Square()
        >>> q = {0:[0,0], 1: [0,0], 2:[-2,0], 3:[-2,0]}
        >>> F.is_dict_nontrivial_inf_flex(q)
        True
        >>> q = {0:[1,-1], 1: [1,1], 2:[-1,1], 3:[-1,-1]}
        >>> F.is_dict_nontrivial_inf_flex(q)
        False
        """
        self._graph._input_check_vertex_order(list(vert_to_flex.keys()), "vert_to_flex")

        dict_to_list = []
        for v in self._graph.vertex_list():
            dict_to_list += list(vert_to_flex[v])

        return self.is_vector_nontrivial_inf_flex(
            dict_to_list, vertex_order=self._graph.vertex_list(), **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    def is_nontrivial_flex(
        self,
        inf_flex: InfFlex,
        **kwargs,
    ) -> bool:
        """
        Alias for :meth:`.is_vector_nontrivial_inf_flex` and
        :meth:`.is_dict_nontrivial_inf_flex`.

        It is distinguished between instances of ``list`` and instances of ``dict`` to
        call one of the alias methods.

        Definitions
        -----------
        :prf:ref:`Nontrivial infinitesimal flex <def-trivial-inf-flex>`

        Parameters
        ----------
        inf_flex
        """
        if isinstance(inf_flex, list | tuple | Matrix):
            return self.is_vector_nontrivial_inf_flex(inf_flex, **kwargs)
        elif isinstance(inf_flex, dict):
            return self.is_dict_nontrivial_inf_flex(inf_flex, **kwargs)
        else:
            raise TypeError(
                "The `inf_flex` must be specified either by a vector or a dictionary!"
            )

    @doc_category("Infinitesimal rigidity")
    def is_vector_trivial_inf_flex(self, inf_flex: Sequence[Number], **kwargs) -> bool:
        r"""
        Return whether an infinitesimal flex is trivial.

        See also :meth:`.is_nontrivial_vector_inf_flex` for details,
        particularly concerning the possible parameters.

        Definitions
        -----------
        :prf:ref:`Trivial infinitesimal flex <def-trivial-inf-flex>`

        Parameters
        ----------
        inf_flex:
            An infinitesimal flex of the framework.

        Examples
        --------
        >>> from pyrigi import frameworkDB as fws
        >>> F = fws.Square()
        >>> q = [0,0,0,0,-2,0,-2,0]
        >>> F.is_vector_trivial_inf_flex(q)
        False
        >>> q = [1,-1,1,1,-1,1,-1,-1]
        >>> F.is_vector_trivial_inf_flex(q)
        True
        """
        if not self.is_vector_inf_flex(inf_flex, **kwargs):
            return False
        return not self.is_vector_nontrivial_inf_flex(inf_flex, **kwargs)

    @doc_category("Infinitesimal rigidity")
    def is_dict_trivial_inf_flex(
        self, inf_flex: dict[Vertex, Sequence[Number]], **kwargs
    ) -> bool:
        r"""
        Return whether an infinitesimal flex specified by a dictionary is trivial.

        See :meth:`.is_vector_trivial_inf_flex` for details,
        particularly concerning the possible parameters.

        Definitions
        -----------
        :prf:ref:`Trivial infinitesimal flex <def-trivial-inf-flex>`

        Parameters
        ----------
        inf_flex:
            An infinitesimal flex of the framework in the form of a dictionary.

        Examples
        --------
        >>> from pyrigi import frameworkDB as fws
        >>> F = fws.Square()
        >>> q = {0:[0,0], 1: [0,0], 2:[-2,0], 3:[-2,0]}
        >>> F.is_dict_trivial_inf_flex(q)
        False
        >>> q = {0:[1,-1], 1: [1,1], 2:[-1,1], 3:[-1,-1]}
        >>> F.is_dict_trivial_inf_flex(q)
        True
        """
        self._graph._input_check_vertex_order(list(inf_flex.keys()), "vert_to_flex")

        dict_to_list = []
        for v in self._graph.vertex_list():
            dict_to_list += list(inf_flex[v])

        return self.is_vector_trivial_inf_flex(
            dict_to_list, vertex_order=self._graph.vertex_list(), **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    def is_trivial_flex(
        self,
        inf_flex: InfFlex,
        **kwargs,
    ) -> bool:
        """
        Alias for :meth:`.is_vector_trivial_inf_flex` and
        :meth:`.is_dict_trivial_inf_flex`.

        Ii is distinguished between instances of ``list`` and instances of ``dict`` to
        call one of the alias methods.

        Definitions
        -----------
        :prf:ref:`Trivial infinitesimal flex <def-trivial-inf-flex>`

        Parameters
        ----------
        inf_flex
        """
        if isinstance(inf_flex, list | tuple | Matrix):
            return self.is_vector_trivial_inf_flex(inf_flex, **kwargs)
        elif isinstance(inf_flex, dict):
            return self.is_dict_trivial_inf_flex(inf_flex, **kwargs)
        else:
            raise TypeError(
                "The `inf_flex` must be specified either by a vector or a dictionary!"
            )

    def _input_check_underlying_graphs(self, other_framework) -> None:
        """
        Check whether the underlying graphs of two frameworks are the same and
        raise an error otherwise.
        """
        if self._graph != other_framework._graph:
            raise ValueError("The underlying graphs are not same!")

    def _input_check_vertex_key(
        self, vertex: Vertex, realization: dict[Vertex, Point] = None
    ) -> None:
        """
        Check whether a vertex appears as key in a realization and
        raise an error otherwise.
        """
        if realization is None:
            realization = self._realization
        if vertex not in realization:
            raise KeyError("Vertex {vertex} is not a key of the given realization!")

    def _input_check_point_dimension(self, point: Point) -> None:
        """
        Check whether a point has the right dimension and
        raise an error otherwise.
        """
        if not len(point) == self.dim:
            raise ValueError(
                f"The point {point} does not have the dimension {self.dim}!"
            )


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
