"""

Module for the functionality concerning frameworks.

.. currentmodule:: pyrigi.framework

Classes:

.. autosummary::

    Framework

"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from random import randrange

import networkx as nx
import sympy as sp
import numpy as np
import functools

from sympy import Matrix, flatten, binomial

from pyrigi.data_type import (
    Vertex,
    Edge,
    Point,
    InfFlex,
    Stress,
    point_to_vector,
    Sequence,
    Number,
    DirectedEdge,
)

from pyrigi.graph import Graph
from pyrigi.graphDB import Complete as CompleteGraph
from pyrigi.misc import (
    doc_category,
    generate_category_tables,
    is_zero_vector,
    generate_two_orthonormal_vectors,
    generate_three_orthonormal_vectors,
    eval_sympy_vector,
)
import pyrigi._input_check as _input_check

from typing import Any
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
        The dimension ``d`` is retrieved from the points in realization.
        If ``graph`` is empty, and hence also the ``realization``,
        the dimension is set to 0 (:meth:`Framework.Empty`
        can be used to construct an empty framework with different dimension).

    Examples
    --------
    >>> F = Framework(Graph([[0,1]]), {0:[1,2], 1:[0,5]})
    >>> F
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
        self.set_realization(realization)

    def __str__(self) -> str:
        """Return the string representation."""
        return (
            self.__class__.__name__
            + f" in {self.dim()}-dimensional space consisting of:\n{self._graph}\n"
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
        """Return the representation"""
        return self.__str__()

    def __getitem__(self, vertex: Vertex) -> Matrix:
        """
        Return the coordinates corresponding to the image
        of a given vertex under the realization map.

        Examples
        --------
        >>> F = Framework(Graph([[0,1]]), {0:[1,2], 1:[0,5]})
        >>> F[0]
        Matrix([
        [1],
        [2]])
        """
        return self._realization[vertex]

    @doc_category("Attribute getters")
    def dim(self) -> int:
        """Return the dimension of the framework."""
        return self._dim

    @doc_category("Attribute getters")
    def dimension(self) -> int:
        """Alias for :meth:`~Framework.dim`"""
        return self.dim()

    @doc_category("Framework manipulation")
    def add_vertex(self, point: Point, vertex: Vertex = None) -> None:
        """
        Add a vertex to the framework with the corresponding coordinates.

        If no vertex is provided (``None``), then the smallest,
        free integer is chosen instead.

        Parameters
        ----------
        point:
            the realization of the new vertex
        vertex:
            the label of the new vertex

        Examples
        --------
        >>> F = Framework.Empty(dim=2)
        >>> F.add_vertex((1.5,2), 'a')
        >>> F.add_vertex((3,1))
        >>> F
        Framework in 2-dimensional space consisting of:
        Graph with vertices ['a', 1] and edges []
        Realization {a:(1.50000000000000, 2), 1:(3, 1)}
        """
        if vertex is None:
            candidate = self._graph.number_of_nodes()
            while candidate in self._graph.nodes:
                candidate += 1
            vertex = candidate

        if vertex in self._graph.nodes:
            raise KeyError(f"Vertex {vertex} is already a vertex of the graph!")

        self._realization[vertex] = Matrix(point)
        self._graph.add_node(vertex)

    @doc_category("Framework manipulation")
    def add_vertices(
        self, points: Sequence[Point], vertices: Sequence[Vertex] = []
    ) -> None:
        r"""
        Add a list of vertices to the framework.

        Parameters
        ----------
        points:
            List of points consisting of coordinates in $\RR^d$. It is checked
            that all points lie in the same ambient space.
        vertices:
            List of vertices. If the list of vertices is empty, we generate a
            vertex that is not yet taken with the method :meth:`add_vertex`.
            Else, the list of vertices needs to have the same length as the
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
        if not (len(points) == len(vertices) or not vertices):
            raise IndexError("The vertex list does not have the correct length!")
        if not vertices:
            for point in points:
                self.add_vertex(point)
        else:
            for p, v in zip(points, vertices):
                self.add_vertex(p, v)

    @doc_category("Framework manipulation")
    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge to the framework.

        Notes
        -----
        This method only alters the graph attribute.
        """
        self._graph._input_check_edge_format(edge)
        self._graph.add_edge(*(edge))

    @doc_category("Framework manipulation")
    def add_edges(self, edges: Sequence[Edge]) -> None:
        """
        Add a list of edges to the framework.

        Notes
        -----
        For each edge that has to be added, :meth:`add_edge` is called.
        """
        for edge in edges:
            self.add_edge(edge)

    @doc_category("Attribute getters")
    def graph(self) -> Graph:
        """
        Return a copy of the underlying graph.

        Examples
        ----
        >>> F = Framework.Random(Graph([(0,1), (1,2), (0,2)]))
        >>> F.graph()
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        """
        return deepcopy(self._graph)

    @doc_category("Plotting")
    def plot2D(
        self,
        plot_style: PlotStyle = None,
        projection_matrix: Matrix = None,
        random_seed: int = None,
        coordinates: Sequence[int] = None,
        inf_flex: int | InfFlex = None,
        stress: int | Stress = None,
        edge_coloring: Sequence[Sequence[Edge]] | dict[str, Sequence[Edge]] = None,
        stress_label_positions: dict[DirectedEdge, float] = None,
        arc_angles_dict: Sequence[float] | dict[DirectedEdge, float] = None,
        **kwargs,
    ) -> None:
        """
        Plot this framework in 2D.

        If this framework is in dimensions higher than 2 and ``projection_matrix``
        with ``coordinates`` are ``None``, a random projection matrix
        containing two orthonormal vectors is generated and used for projection into 2D.
        For various formatting options, see :meth:`.PlotStyle`.
        Only ``coordinates`` or ``projection_matrix`` parameter can be used, not both!

        Parameters
        ----------
        plot_style:
            An instance of the ``PlotStyle`` class that defines the visual style
            for plotting, see :class:`PlotStyle` for more details.
        projection_matrix:
            The matrix used for projecting the placement of vertices
            only when they are in dimension higher than 2.
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
        edge_coloring:
            Optional parameter to specify the coloring of edges. It can be
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
        >>> from pyrigi.plot_style import PlotStyle2D
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

        Use edge coloring
        >>> edge_coloring = {'red': [(0, 1), (1, 2)], 'blue': [(2, 3), (0, 3)]}
        >>> F.plot2D(edge_coloring=edge_coloring)

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
            edge_coloring=edge_coloring,
            arc_angles_dict=arc_angles_dict,
        )

        if inf_flex is not None:
            _plot.plot_inf_flex2D(
                self,
                ax,
                inf_flex,
                points=placement,
                plot_style=plot_style,
                projection_matrix=projection_matrix,
            )
        if stress is not None:
            _plot.plot_stress2D(
                self,
                ax,
                stress,
                points=placement,
                plot_style=plot_style,
                arc_angles_dict=arc_angles_dict,
                stress_label_positions=stress_label_positions,
            )

    @doc_category("Plotting")
    def animate3D(
        self,
        vertex_color: str = "#ff8c00",
        vertex_shape: str = "o",
        vertex_size: int = 13.5,
        edge_color: str = "black",
        edge_coloring: Sequence[Sequence[Edge]] | dict[str, Sequence[Edge]] = None,
        edge_width: float = 1.5,
        edge_style: str = "solid",
        equal_aspect_ratio: bool = True,
        total_frames: int = 50,
        delay: int = 75,
        rotation_axis: str | Sequence[Number] = None,
        dpi: int = 150,
    ) -> Any:
        """
        Plot this framework in 3D and animate a rotation around an axis.

        Parameters
        ----------
        vertex_color, vertex_shape, vertex_size, edge_color, edge_width, edge_style:
            The user can choose differen colors etc. both for edges and vertices.
        total_frames:
            Number of frames used for the animation. The higher this number,
            the smoother the resulting animation.
        equal_aspect_ratio:
            Determines whether the aspect ratio of the plot is equal in all space
            directions or whether it is adjusted depending on the framework's size
            in `x`, `y` and `z`-direction individually.
        delay:
            Delay between frames in milliseconds.
        rotation_axis:
            The user can input a rotation axis or vector. By default, a rotation around
            the z-axis is performed. This can either be done in the form of a char
            ('x', 'y', 'z') or as a vector (e.g. [1, 0, 0]).
        dpi:
            Dots per inch of the produced animation.

        Examples
        --------
        >>> from pyrigi import frameworkDB
        >>> F = frameworkDB.Complete(4, dim=3)
        >>> F.animate3D()
        """
        _input_check.dimension_for_algorithm(self._dim, [3], "animate3D")

        # Creation of the figure
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.grid(False)
        ax.set_axis_off()

        from pyrigi import _plot

        edge_color_array, edge_list_ref = _plot.resolve_edge_colors(
            self, edge_color, edge_coloring
        )
        realization = self.realization(as_points=True, numerical=True)
        centroid_x = sum([p[0] for p in realization.values()]) / len(realization)
        centroid_y = sum([p[1] for p in realization.values()]) / len(realization)
        centroid_z = sum([p[2] for p in realization.values()]) / len(realization)
        realization = {
            v: [p[0] - centroid_x, p[1] - centroid_y, p[2] - centroid_z]
            for v, p in realization.items()
        }
        # Limits of the axes
        vertices = np.array(list(realization.values()))
        # Initializing points (vertices) and lines (edges) for display
        (vertices_plot,) = ax.plot(
            [], [], [], vertex_shape, color=vertex_color, markersize=vertex_size
        )
        lines = [
            ax.plot(
                [], [], [], c=edge_color_array[i], lw=edge_width, linestyle=edge_style
            )[0]
            for i in range(len(edge_list_ref))
        ]

        # Animation initialization function.
        def init():
            vertices_plot.set_data([], [])  # Initial coordinates of vertices
            vertices_plot.set_3d_properties([])  # Initial 3D properties of vertices
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return [vertices_plot] + lines

        def _rotation_matrix(v, frame):
            # Compute the rotation matrix Q
            v = np.array(v)
            v = v / np.linalg.norm(v)
            angle = frame * np.pi / total_frames
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

            # Rodrigues' rotation matrix
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            Q = np.eye(3) * cos_angle + K * sin_angle + np.outer(v, v) * (1 - cos_angle)
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

        rot_vertices = sum(
            [
                vertices.dot(rotation_matrix(frame).T).tolist()
                for frame in range(2 * total_frames)
            ],
            [],
        )
        if equal_aspect_ratio:
            min_val = min([min(pt) for pt in rot_vertices]) - 0.01
            max_val = max([max(pt) for pt in rot_vertices]) + 0.01
            ax.set_zlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_xlim(min_val, max_val)
        else:
            ax.set_zlim(
                min([pt[2] for pt in rot_vertices]) - 0.01,
                max([pt[2] for pt in rot_vertices]) + 0.01,
            )
            ax.set_ylim(
                min([pt[1] for pt in rot_vertices]) - 0.01,
                max([pt[1] for pt in rot_vertices]) + 0.01,
            )
            ax.set_xlim(
                min([pt[0] for pt in rot_vertices]) - 0.01,
                max([pt[0] for pt in rot_vertices]) + 0.01,
            )

        # Function to update data at each frame
        def update(frame):
            one_rotation_matrix = rotation_matrix(frame)
            rotated_vertices = vertices.dot(one_rotation_matrix.T)

            # Update vertices positions
            vertices_plot.set_data(rotated_vertices[:, 0], rotated_vertices[:, 1])
            vertices_plot.set_3d_properties(rotated_vertices[:, 2])

            # Update the edges
            for i, (start, end) in enumerate(self._graph.edges):
                line = lines[i]
                line.set_data(
                    [rotated_vertices[start, 0], rotated_vertices[end, 0]],
                    [rotated_vertices[start, 1], rotated_vertices[end, 1]],
                )
                line.set_3d_properties(
                    [rotated_vertices[start, 2], rotated_vertices[end, 2]]
                )

            return [vertices_plot] + lines

        # Creating the animation
        ani = FuncAnimation(
            fig,
            update,
            frames=total_frames * 2,
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
            plt.show()
            return

    @doc_category("Plotting")
    def plot3D(
        self,
        plot_style: PlotStyle = None,
        projection_matrix: Matrix = None,
        random_seed: int = None,
        coordinates: Sequence[int] = None,
        inf_flex: int | InfFlex = None,
        stress: int | Stress = None,
        edge_coloring: Sequence[Sequence[Edge]] | dict[str : Sequence[Edge]] = None,
        stress_label_positions: dict[DirectedEdge, float] = None,
        **kwargs,
    ) -> None:
        """
        Plot the provided framework in 3D.

        If the framework is in a dimension higher than 3 and ``projection_matrix``
        with ``coordinates`` are ``None``, a random projection matrix
        containing three orthonormal vectors is generated and used for projection into 3D.
        For various formatting options, see :meth:`.PlotStyle`.
        Only the parameter ``coordinates`` or ``projection_matrix`` can be used,
        not both at the same time.

        Parameters
        ----------
        plot_style:
            An instance of the ``PlotStyle`` class that defines the visual style
            for plotting, see :class:`PlotStyle` for more details.
        projection_matrix:
            The matrix used for projecting the placement of vertices
            only when they are in dimension higher than 2.
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
        edge_coloring:
            Optional parameter to specify the coloring of edges. It can be
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

        >>> from pyrigi.plot_style import PlotStyle3D
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

        Use edge coloring
        >>> edge_coloring = {'red': [(5, 1), (1, 2)], 'blue': [(2, 4), (4, 3)]}
        >>> F.plot3D(edge_coloring=edge_coloring)

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
            sum([p[i] for p in placement.values()]) / len(placement) for i in range(3)
        ]
        _placement = {
            v: [p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2]]
            for v, p in placement.items()
        }

        from pyrigi import _plot

        _plot.plot_with_3D_realization(
            self,
            ax,
            _placement,
            plot_style,
            edge_coloring=edge_coloring,
        )

        if inf_flex is not None:
            _plot.plot_inf_flex3D(
                self,
                ax,
                inf_flex,
                points=_placement,
                plot_style=plot_style,
                projection_matrix=projection_matrix,
            )

        if stress is not None:
            _plot.plot_stress3D(
                self,
                ax,
                stress,
                points=_placement,
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
        use :meth:`.Framework.plot2D` or :meth:`.Framework.plot3D` instead.
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
        Works for dimension 2 only.

        For using it in ``LaTeX`` you need to use the ``tikz`` package.

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

        For more examples on formatting options, see also :meth:`.Graph.to_tikz`.
        """  # noqa: E501

        # check dimension
        if self.dimension() != 2:
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

        return self.graph().to_tikz(
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
        Return a framework with random realization.

        Parameters
        ----------
        dim:
            The dimension of the constructed framework.
        graph:
            Graph on which the random realization should be constructed.
        rand_range:
            Sets the range of random numbers from which the realization is
            sampled. The format is either an interval ``(a,b)`` or a single
            integer ``a``, which produces the range ``(-a,a)``.

        Examples
        --------
        >>> F = Framework.Random(Graph([(0,1), (1,2), (0,2)]))
        >>> print(F) # doctest: +SKIP
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        Realization {0:(122, 57), 1:(27, 144), 2:(50, 98)}

        Notes
        -----
        If ``rand_range=None``, then the range is set to ``(-a,a)`` for
        ``a = 10^4 * n * d``.
        """
        _input_check.dimension(dim)
        if rand_range is None:
            b = 10**4 * graph.number_of_nodes() * dim
            a = -b
        if isinstance(rand_range, list | tuple):
            if not len(rand_range) == 2:
                raise ValueError("If `rand_range` is a list, it must be of length 2.")
            a, b = rand_range
        if isinstance(rand_range, int):
            if rand_range <= 0:
                raise ValueError("If `rand_range` is an int, it must be positive")
            b = rand_range
            a = -b

        realization = {
            vertex: [randrange(a, b) for _ in range(dim)] for vertex in graph.nodes
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
        Return the framework with a realization on the x-axis in the d-dimensional space.

        Parameters
        ----------
        dim:
            The dimension of the constructed framework.
        graph:
            Underlying graph on which the framework is constructed.

        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> Framework.Collinear(graphs.Complete(3), dim=2)
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
        Return the framework with a realization on the d-simplex.

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
            a natural number that determines the dimension
            in which the framework is realized

        Examples
        ----
        >>> F = Framework.Empty(dim=1); F
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
        >>> F = Framework.Complete([(1,),(2,),(3,),(4,)]); F
        Framework in 1-dimensional space consisting of:
        Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        Realization {0:(1,), 1:(2,), 2:(3,), 3:(4,)}
        """  # noqa: E501
        if not points:
            raise ValueError("The list of points cannot be empty!")

        Kn = CompleteGraph(len(points))
        return Framework(Kn, {v: Matrix(p) for v, p in zip(Kn.nodes, points)})

    @doc_category("Framework manipulation")
    def delete_vertex(self, vertex: Vertex) -> None:
        """
        Delete a vertex from the framework.
        """
        self._graph.delete_vertex(vertex)
        del self._realization[vertex]

    @doc_category("Framework manipulation")
    def delete_vertices(self, vertices: Sequence[Vertex]) -> None:
        """
        Delete a list of vertices from the framework.
        """
        for vertex in vertices:
            self.delete_vertex(vertex)

    @doc_category("Framework manipulation")
    def delete_edge(self, edge: Edge) -> None:
        """
        Delete an edge from the framework.
        """
        self._graph.delete_edge(edge)

    @doc_category("Framework manipulation")
    def delete_edges(self, edges: Sequence[Edge]) -> None:
        """
        Delete a list of edges from the framework.
        """
        self._graph.delete_edges(edges)

    @doc_category("Attribute getters")
    def realization(
        self, as_points: bool = False, numerical: bool = False
    ) -> dict[Vertex, Point]:
        """
        Return a copy of the realization.

        Parameters
        ----------
        as_points:
            If ``True``, then the vertex positions type is Point,
            otherwise Matrix (default).
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

        Notes
        -----
        The format returned by this method with ``as_points=True``
        can be read by networkx.
        """
        if not numerical:
            if not as_points:
                return deepcopy(self._realization)
            return {
                vertex: list(position) for vertex, position in self._realization.items()
            }
        else:
            if not as_points:
                {
                    vertex: Matrix([float(p) for p in position])
                    for vertex, position in self._realization.items()
                }
            return {
                vertex: [float(p) for p in position]
                for vertex, position in self._realization.items()
            }

    @doc_category("Framework properties")
    def is_quasi_injective(
        self, numerical: bool = False, tolerance: float = 1e-9
    ) -> bool:
        """
        Return whether the realization is :prf:ref:`quasi-injective <def-realization>`.

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

        for u, v in self._graph.edges:
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
        numerical
            Whether the check is symbolic (default) or numerical.
        tolerance
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

        Parameters
        ----------
        realization:
            a realization of the underlying graph of the framework

        Notes
        -----
        It is assumed that the realization contains all vertices from the
        underlying graph. Furthermore, all points in the realization need
        to be contained in $\RR^d$ for a fixed $d$.

        Examples
        --------
        >>> F = Framework.Complete([(0,0), (1,0), (1,1)])
        >>> F.set_realization({vertex:(vertex,vertex+1) for vertex in F.graph().vertex_list()})
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        Realization {0:(0, 1), 1:(1, 2), 2:(2, 3)}
        """  # noqa: E501
        if not len(realization) == self._graph.number_of_nodes():
            raise IndexError(
                "The realization does not contain the correct amount of vertices!"
            )
        for v in self._graph.nodes:
            self._input_check_vertex_key(v, realization)
            self._input_check_point_dimension(realization[v])

        self._realization = {v: Matrix(realization[v]) for v in realization.keys()}

    @doc_category("Framework manipulation")
    def set_vertex_pos(self, vertex: Vertex, point: Point) -> None:
        """
        Change the coordinates of a single given vertex.

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

        Examples
        ----
        >>> F = Framework.Complete([(0,0),(0,0),(1,0),(1,0)])
        >>> F.realization(as_points=True)
        {0: [0, 0], 1: [0, 0], 2: [1, 0], 3: [1, 0]}
        >>> F.set_vertex_positions_from_lists([1,3], [(0,1),(1,1)])
        >>> F.realization(as_points=True)
        {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}

        Notes
        -----
        It is necessary that both lists have the same length.
        No vertex from ``vertices`` can be contained multiple times.
        We apply the method :meth:`~Framework.set_vertex_pos`
        to ``vertices`` and ``points``.
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
    def set_vertex_positions(self, subset_of_realization: dict[Vertex, Point]):
        """
        Change the coordinates of vertices given by a dictionary.

        Examples
        ----
        >>> F = Framework.Complete([(0,0),(0,0),(1,0),(1,0)])
        >>> F.realization(as_points=True)
        {0: [0, 0], 1: [0, 0], 2: [1, 0], 3: [1, 0]}
        >>> F.set_vertex_positions({1:(0,1),3:(1,1)})
        >>> F.realization(as_points=True)
        {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}

        Notes
        -----
        See :meth:`~Framework.set_vertex_pos`.
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
            If none is provided, the list from :meth:`~Graph.vertex_list` is taken.
        edge_order:
            A list of edges, providing the ordering for the rows
            of the rigidity matrix.
            If none is provided, the list from :meth:`~Graph.edge_list` is taken.

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
            Dictionary that maps the edge labels to coordinates.

        Notes
        -----
        See :meth:`.Framework.is_vector_stress`.

        Examples
        --------
        >>> F = Framework.Complete([[0,0], [1,0], ['1/2',0]])
        >>> F.is_dict_stress({(0,1):'-1/2', (0,2):1, (1,2):1})
        True
        >>> F.is_dict_stress({(0,1):1, (1,2):'-1/2', (0,2):1})
        False
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
        Return whether a vector is a stress.

        Definitions
        -----------
        :prf:ref:`Equilibrium stress <def-equilibrium-stress>`

        Parameters
        ----------
        stress:
            A vector to be checked whether it is a stress of the framework.
        edge_order:
            A list of edges, providing the ordering for the entries of the ``stress``.
            If none is provided, the list from :meth:`~Graph.edge_list` is taken.
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
        Alias for :meth:`Framework.is_vector_stress` and
        :meth:`Framework.is_dict_stress`.

        Notes
        -----
        We distinguish between instances of ``List`` and instances of
        ``Dict`` to call one of the alias methods.

        """
        if isinstance(stress, list | Matrix):
            return self.is_vector_stress(stress, **kwargs)
        elif isinstance(stress, dict):
            return self.is_dict_stress(stress, **kwargs)
        else:
            raise TypeError(
                "The `stress` must be specified either by a vector or a dictionary!"
            )

    @doc_category("Infinitesimal rigidity")
    def stress_matrix(
        self,
        stress: Stress,
        edge_order: Sequence[Edge] = None,
        vertex_order: Sequence[Vertex] = None,
    ) -> Matrix:
        r"""
        Construct the stress matrix from a stress of from its support.

        The matrix order is the one from :meth:`~.Framework.vertex_list`.

        Definitions
        -----
        * :prf:ref:`Stress Matrix <def-stress-matrix>`

        Parameters
        ----------
        stress:
            A stress of the framework.
        edge_order:
            A list of edges, providing the ordering for the rows
            of the stress matrix.
        vertex_order:
            By listing vertices in the preferred order, the rigidity matrix
            can be computed in a way the user expects.

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
        * :prf:ref:`Trivial infinitesimal flexes <def-trivial-inf-flex>`

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
    def nontrivial_inf_flexes(
        self, vertex_order: Sequence[Vertex] = None
    ) -> list[Matrix]:
        """
        Return non-trivial infinitesimal flexes.

        Definitions
        -----------
        :prf:ref:`Infinitesimal flex <def-inf-rigid-framework>`

        Parameters
        ----------
        vertex_order:
            A list of vertices, providing the ordering for the entries
            of the infinitesimal flexes.
            If none is provided, the list from :meth:`~Graph.vertex_list` is taken.

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

        Notes
        -----
        See :meth:`~Framework.trivial_inf_flexes`.
        """
        return self.inf_flexes(include_trivial=False, vertex_order=vertex_order)

    @doc_category("Infinitesimal rigidity")
    def inf_flexes(
        self, include_trivial: bool = False, vertex_order: Sequence[Vertex] = None
    ) -> list[Matrix]:
        r"""
        Return a basis of the space of infinitesimal flexes.

        Return a lift of a basis of the quotient of
        the vector space of infinitesimal flexes
        modulo trivial infinitesimal flexes, if ``include_trivial=False``.
        Return a basis of the vector space of infinitesimal flexes
        if ``include_trivial=True``.
        Else, return the entire kernel.

        Definitions
        -----------
        * :prf:ref:`Infinitesimal flex <def-inf-flex>`

        Parameters
        ----------
        include_trivial:
            Boolean that decides, whether the trivial flexes should
            be included (``True``) or not (``False``)
        vertex_order:
            A list of vertices, providing the ordering for the entries
            of the infinitesimal flexes.
            If none is provided, the list from :meth:`~Graph.vertex_list` is taken.

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
        >>> F = Framework(Graph([[0, 1], [0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]]), {0: [0, 0], 1: [0, 1], 2: [0, 2], 3: [1, 2], 4: [-1, 2]})
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
        """  # noqa: E501
        vertex_order = self._graph._input_check_vertex_order(vertex_order)
        if include_trivial:
            return self.rigidity_matrix(vertex_order=vertex_order).nullspace()
        rigidity_matrix = self.rigidity_matrix(vertex_order=vertex_order)

        all_inf_flexes = rigidity_matrix.nullspace()
        trivial_inf_flexes = self.trivial_inf_flexes(vertex_order=vertex_order)
        s = len(trivial_inf_flexes)
        extend_basis_matrix = Matrix.hstack(*trivial_inf_flexes)
        tmp_matrix = Matrix.hstack(*trivial_inf_flexes)
        for v in all_inf_flexes:
            r = extend_basis_matrix.rank()
            tmp_matrix = Matrix.hstack(extend_basis_matrix, v)
            if not tmp_matrix.rank() == r:
                extend_basis_matrix = Matrix.hstack(extend_basis_matrix, v)
        basis = extend_basis_matrix.columnspace()
        return basis[s:]

    @doc_category("Infinitesimal rigidity")
    def stresses(self, edge_order: Sequence[Edge] = None) -> list[Matrix]:
        r"""
        Return a basis of the space of equilibrium stresses.

        Definitions
        -----------
        :prf:ref:`Equilibrium stress <def-equilibrium-stress>`

        Parameters
        ----------
        edge_order:
            A list of edges, providing the ordering for the entries of the stresses.
            If none is provided, the list from :meth:`~Graph.edge_list` is taken.

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
        return self.rigidity_matrix(edge_order=edge_order).transpose().nullspace()

    @doc_category("Infinitesimal rigidity")
    def rigidity_matrix_rank(self) -> int:
        """
        Compute the rank of the rigidity matrix.

        Examples
        ----
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
        Check whether the given framework is infinitesimally rigid.

        The check is based on :meth:`~Framework.rigidity_matrix_rank`.

        Definitions
        -----
        * :prf:ref:`Infinitesimal rigidity <def-inf-rigid-framework>`

        Examples
        ----
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
                == self.dim() * self._graph.number_of_nodes()
                - binomial(self.dim() + 1, 2)
            )

    @doc_category("Infinitesimal rigidity")
    def is_inf_flexible(self) -> bool:
        """
        Check whether the given framework is infinitesimally flexible.

        See :meth:`~Framework.is_inf_rigid`
        """
        return not self.is_inf_rigid()

    @doc_category("Infinitesimal rigidity")
    def is_min_inf_rigid(self) -> bool:
        """
        Check whether a framework is minimally infinitesimally rigid.

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
        Check whether the framework is :prf:ref:`independent <def-independent-framework>`.

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
        Check whether the framework is :prf:ref:`dependent <def-independent-framework>`.

        Notes
        -----
        See also :meth:`~.Framework.is_independent`.
        """
        return not self.is_independent()

    @doc_category("Infinitesimal rigidity")
    def is_isostatic(self) -> bool:
        """
        Check whether the framework is :prf:ref:`independent <def-independent-framework>`
        and :prf:ref:`infinitesimally rigid <def-inf-rigid-framework>`.
        """
        return self.is_independent() and self.is_inf_rigid()

    @doc_category("Waiting for implementation")
    def is_prestress_stable(self) -> bool:
        """
        TODO
        ----
        Implement
        """
        raise NotImplementedError()

    @doc_category("Infinitesimal rigidity")
    def is_redundantly_rigid(self) -> bool:
        """
        Check if the framework is infinitesimally redundantly rigid.

        Definitions
        -----------
        :prf:ref:`Redundant infinitesimal rigidity <def-redundantly-rigid-framework>`

        Examples
        --------
        >>> F = Framework.Empty(dim=2)
        >>> F.add_vertices([(1,0), (1,1), (0,3), (-1,1)], ['a','b','c','d'])
        >>> F.add_edges([('a','b'), ('b','c'), ('c','d'), ('a','d'), ('a','c'), ('b','d')])
        >>> F.is_redundantly_rigid()
        True
        >>> F.delete_edge(('a','c'))
        >>> F.is_redundantly_rigid()
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
        other_realization: dict[Vertex, Point],
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        """
        Return whether the given realization is congruent to self.

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
            if not difference.is_zero:
                if not numerical:
                    return False
                elif numerical and sp.Abs(difference) > tolerance:
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
        other_realization: dict[Vertex, Point],
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        """
        Return whether the given realization is equivalent to self.

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
            if not difference.is_zero:
                if not numerical:
                    return False
                elif numerical and sp.Abs(difference) > tolerance:
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
    def translate(self, vector: Point, inplace: bool = True) -> None | Framework:
        """
        Translate the framework.

        Parameters
        ----------
        vector
            Translation vector
        inplace
            If True (default), then this framework is translated.
            Otherwise, a new translated framework is returned.
        """
        vector = point_to_vector(vector)

        if inplace:
            if vector.shape[0] != self.dim():
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
    def rotate2D(self, angle: float, inplace: bool = True) -> None | Framework:
        """
        Rotate the planar framework counter clockwise.

        Parameters
        ----------
        angle
            Rotation angle
        inplace
            If True (default), then this framework is rotated.
            Otherwise, a new rotated framework is returned.
        """

        if self.dim() != 2:
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
                projection_matrix = generate_two_orthonormal_vectors(
                    self._dim, random_seed=random_seed
                )
            elif proj_dim == 3:
                projection_matrix = generate_three_orthonormal_vectors(
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
                vertex: tuple(np.dot(projection_matrix, np.array(position)))
                for vertex, position in self.realization(
                    as_points=False, numerical=True
                ).items()
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
                tuple(pair): float(
                    np.linalg.norm(
                        np.array(points[pair[0]]) - np.array(points[pair[1]])
                    )
                )
                for pair in self._graph.edges
            }
        else:
            points = self.realization(as_points=True)
            return {
                tuple(pair): sp.sqrt(
                    sum(
                        [(v - w) ** 2 for v, w in zip(points[pair[0]], points[pair[1]])]
                    )
                )
                for pair in self._graph.edges
            }

    @staticmethod
    def _generate_stl_bar(
        holes_distance: float,
        holes_diameter: float,
        bar_width: float,
        bar_height: float,
        filename="bar.stl",
    ):
        """
        Generate an STL file for a bar.

        The method uses Trimesh and Manifold3d packages to create a model of a bar
        with two holes at the ends. The bar is saved as an STL file.

        Parameters
        ----------
        holes_distance : float
            Distance between the centers of the holes.
        holes_diameter : float
            Diameter of the holes.
        bar_width : float
            Width of the bar.
        bar_height : float
            Height of the bar.
        filename : str
            Name of the output STL file.

        Returns
        -------
        bar_mesh : trimesh.base.Trimesh
            The bar as a Trimesh object.
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

        Generates STL files for the bars of the framework. The files are generated
        in the working folder. The naming convention for the files is ``bar_i-j.stl``,
        where i and j are the vertices of an edge.

        Parameters
        ----------
        scale
            Scale factor for the lengths of the edges, default is 1.0.
        width_of_bars
            Width of the bars, default is 8.0 mm.
        height_of_bars
            Height of the bars, default is 3.0 mm.
        holes_diameter
            Diameter of the holes at the ends of the bars, default is 4.3 mm.
        filename_prefix
            Prefix for the filenames of the generated STL files, default is ``bar_``.
        output_dir
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
        Transform the natural data type of a flex (Matrix) to a
        dictionary that maps a vertex to a Sequence of coordinates
        (i.e. a vector).

        Parameters
        ----------
        inf_flex:
            An infinitesimal flex in the form of a `Matrix`.
        vertex_order:
            If ``None``, the :meth:`.Graph.vertex_list`
            is taken as the vertex order.

        Notes
        ----
        For example, this method can be used for generating an
        infinitesimal flex for plotting purposes.

        Examples
        ----
        >>> F = Framework.from_points([(0,0), (1,0), (0,1)])
        >>> F.add_edges([(0,1),(0,2)])
        >>> flex = F.nontrivial_inf_flexes()[0]
        >>> F._transform_inf_flex_to_pointwise(flex)
        {0: [1, 0], 1: [1, 0], 2: [0, 0]}

        """
        vertex_order = self._graph._input_check_vertex_order(vertex_order)
        return {
            vertex_order[i]: [inf_flex[i * self.dim() + j] for j in range(self.dim())]
            for i in range(len(vertex_order))
        }

    def _transform_stress_to_edgewise(
        self, stress: Matrix, edge_order: Sequence[Edge] = None
    ) -> dict[Edge, Number]:
        r"""
        Transform the natural data type of a stress (Matrix) to a
        dictionary that maps an edge to a coordinate.

        Parameters
        ----------
        stress:
            An equilibrium stress in the form of a `Matrix`.
        edge_order:
            If ``None``, the :meth:`.Graph.edge_list`
            is taken as the edge order.

        Notes
        ----
        For example, this method can be used for generating an
        equilibrium stresss for plotting purposes.

        Examples
        ----
        >>> F = Framework.Complete([(0,0),(1,0),(1,1),(0,1)])
        >>> stress = F.stresses()[0]
        >>> F._transform_stress_to_edgewise(stress)
        {(0, 1): 1, (0, 2): -1, (0, 3): 1, (1, 2): 1, (1, 3): -1, (2, 3): 1}

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
        :prf:ref:`Infinitesimal Flex <def-inf-flex>`
        :prf:ref:`Rigidity Matrix <def-rigidity-matrix>`

        Parameters
        ----------
        inf_flex:
            An infinitesimal flex of the framework specified by a vector.
        vertex_order:
            A list of vertices specifying the order in which ``inf_flex`` is given.
            If none is provided, the list from :meth:`~Graph.vertex_list` is taken.
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

        Notes
        -----
        See :meth:`.Framework.is_vector_inf_flex`.

        Examples
        --------
        >>> F = Framework.Complete([[0,0], [1,1]])
        >>> F.is_dict_inf_flex({0:[0,0], 1:[-1,1]})
        True
        >>> F.is_dict_inf_flex({0:[0,0], 1:["sqrt(2)","-sqrt(2)"]})
        True
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
        :prf:ref:`Nontrivial infinitesimal Flex <def-trivial-inf-flex>`

        Parameters
        ----------
        inf_flex:
            An infinitesimal flex of the framework.
        vertex_order:
            A list of vertices specifying the order in which ``inf_flex`` is given.
            If none is provided, the list from :meth:`~Graph.vertex_list` is taken.
        numerical:
            A Boolean determining whether the evaluation of the product of the `inf_flex`
            and the rigidity matrix is symbolic or numerical.
        tolerance:
            Absolute tolerance that is the threshold for acceptable numerical flexes.
            This parameter is used to determine the number of digits, to which
            accuracy the symbolic expressions are evaluated.

        Notes
        -----
        This is done by solving a linear system composed of a matrix `A` whose columns
        are given by a basis of the trivial flexes and the vector `b` given by the
        input flex. `b` is trivial if and only if there is a linear combination of
        the columns in `A` producing `b`. In other words, when there is a solution to
        `Ax=b`, then `b` is a trivial infinitesimal motion. Otherwise, `b` is
        nontrivial.

        In the `numerical=True` case we compute a least squares solution `x` of the
        overdetermined linear system and compare the values in `Ax` to the values
        in `b`.

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
                    eval_sympy_vector(flex, tolerance=tolerance)
                    for flex in self.trivial_inf_flexes(vertex_order=vertex_order)
                ]
            ).transpose()
            b = np.array(eval_sympy_vector(inf_flex, tolerance=tolerance)).transpose()
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

        Definitions
        -----------
        :prf:ref:`Nontrivial infinitesimal Flex <def-trivial-inf-flex>`

        Parameters
        ----------
        vert_to_flex:
            An infinitesimal flex of the framework in the form of a dictionary.

        Notes
        -----
        See :meth:`Framework.is_vector_nontrivial_inf_flex` for details,
        particularly concerning the possible parameters.

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
        Alias for :meth:`Framework.is_vector_nontrivial_inf_flex` and
        :meth:`Framework.is_dict_nontrivial_inf_flex`.

        Notes
        -----
        We distinguish between instances of ``list`` and instances of ``dict`` to
        call one of the alias methods.
        """
        if isinstance(inf_flex, list | tuple):
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

        Definitions
        -----------
        :prf:ref:`Trivial infinitesimal Flex <def-trivial-inf-flex>`

        Parameters
        ----------
        inf_flex:
            An infinitesimal flex of the framework.

        Notes
        -----
        See :meth:`Framework.is_nontrivial_vector_inf_flex` for details,
        particularly concerning the possible parameters.

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
        self, vert_to_flex: dict[Vertex, Sequence[Number]], **kwargs
    ) -> bool:
        r"""
        Return whether an infinitesimal flex specified by a dictionary is trivial.

        Definitions
        -----------
        :prf:ref:`Trivial infinitesimal flex <def-trivial-inf-flex>`

        Parameters
        ----------
        vert_to_flex:
            An infinitesimal flex of the framework in the form of a dictionary.

        Notes
        -----
        See :meth:`Framework.is_vector_trivial_inf_flex` for details,
        particularly concerning the possible parameters.

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
        self._graph._input_check_vertex_order(list(vert_to_flex.keys()), "vert_to_flex")

        dict_to_list = []
        for v in self._graph.vertex_list():
            dict_to_list += list(vert_to_flex[v])

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
        Alias for :meth:`Framework.is_vector_trivial_inf_flex` and
        :meth:`Framework.is_dict_trivial_inf_flex`.

        Notes
        -----
        We distinguish between instances of ``list`` and instances of ``dict`` to
        call one of the alias methods.
        """
        if isinstance(inf_flex, list | tuple):
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
        if not nx.utils.graphs_equal(self._graph, other_framework._graph):
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
        if not len(point) == self.dimension():
            raise ValueError(
                f"The point {point} does not have the dimension {self.dimension()}!"
            )


Framework.__doc__ = Framework.__doc__.replace(
    "METHODS",
    generate_category_tables(
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
