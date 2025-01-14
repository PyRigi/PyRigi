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
import distinctipy

import networkx as nx
import sympy as sp
import numpy as np
import functools

from sympy import Matrix, flatten, binomial, sympify

from pyrigi.data_type import (
    Vertex,
    Edge,
    Point,
    Stress,
    Inf_Flex,
    point_to_vector,
    Sequence,
    Coordinate,
    DirectedEdge,
)

from pyrigi.graph import Graph
from pyrigi.exception import LoopError
from pyrigi.graphDB import Complete as CompleteGraph
from pyrigi.misc import (
    doc_category,
    generate_category_tables,
    check_integrality_and_range,
    is_zero_vector,
    generate_two_orthonormal_vectors,
    generate_three_orthonormal_vectors,
    eval_sympy_vector,
)

from typing import Optional, Any
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation

__doctest_requires__ = {
    ("Framework.generate_stl_bars",): ["trimesh", "manifold3d", "pathlib"]
}


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

    TODO
    ----
    Use :meth:`~.Framework.set_realization` in the constructor.

    METHODS

    Notes
    -----
    Internally, the realization is represented as ``dict[Vertex,Matrix]``.
    However, :meth:`~Framework.realization` can also return ``dict[Vertex,Point]``.
    """

    def __init__(self, graph: Graph, realization: dict[Vertex, Point]) -> None:
        if not isinstance(graph, Graph):
            raise TypeError("The graph has to be an instance of class Graph.")
        if nx.number_of_selfloops(graph) > 0:
            raise LoopError()
        if not len(realization.keys()) == graph.number_of_nodes():
            raise KeyError(
                "The length of realization has to be equal to "
                "the number of vertices of graph."
            )

        if realization:
            self._dim = len(list(realization.values())[0])
        else:
            self._dim = 0

        for v in graph.nodes:
            if v not in realization:
                raise KeyError(f"Vertex {v} is not contained in the realization.")
            if not len(realization[v]) == self._dim:
                raise ValueError(
                    f"The point {realization[v]} in the realization corresponding to "
                    f"vertex {v} does not have the right dimension."
                )

        self._realization = {v: Matrix(realization[v]) for v in graph.nodes}
        self._graph = deepcopy(graph)

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
        self._graph._check_edge_format(edge)
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
    def _plot_with_2D_realization(
        self,
        ax: Axes,
        realization: dict[Vertex, Point],
        vertex_size: int = 300,
        vertex_color: str = "#ff8c00",
        vertex_shape: str = "o",
        vertex_labels: bool = True,
        edge_width: float = 2.5,
        edge_color: (
            str | Sequence[Sequence[Edge]] | dict[str : Sequence[Edge]]
        ) = "black",
        edge_style: str = "solid",
        font_size: int = 12,
        font_color: str = "whitesmoke",
        curved_edges: bool = False,
        connection_style: float | Sequence[float] | dict[Edge, float] = np.pi / 6,
    ) -> None:
        """
        Plot the graph of the framework with the given realization in the plane.

        For description of other parameters see :meth:`.Framework.plot`.

        Parameters
        ----------
        realization:
            The realization in the plane used for plotting.
        inf_flex:
            Optional parameter for plotting a given infinitesimal flex. It is
            important to use the same vertex order as the one
            from :meth:`.Graph.vertex_list`.
            Alternatively, an ``int`` can be specified to choose the 0,1,2,...-th
            nontrivial infinitesimal flex for plotting.
            Lastly, a ``dict[Vertex, Sequence[Coordinate]]`` can be provided, which
            maps the vertex labels to vectors (i.e. a sequence of coordinates).
        stress:
            Optional parameter for plotting an equilibrium stress. We expect
            it to have the format `Dict[Edge, Coordinate]`.
        """
        edge_color_array, edge_list_ref = self._resolve_edge_colors(edge_color)

        if not curved_edges:
            nx.draw(
                self._graph,
                pos=realization,
                ax=ax,
                node_size=vertex_size,
                node_color=vertex_color,
                node_shape=vertex_shape,
                with_labels=vertex_labels,
                width=edge_width,
                edge_color=edge_color_array,
                font_color=font_color,
                font_size=font_size,
                edgelist=edge_list_ref,
                style=edge_style,
            )
        else:
            newGraph = nx.MultiDiGraph()
            connection_style = self._resolve_connection_style(connection_style)
            for e, style in connection_style.items():
                newGraph.add_edge(e[0], e[1], weight=style)
            plt.box(False)  # Manually removes the frame of the plot
            plt.tick_params(
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
                bottom=False,
            )  # Removes the ticks
            nx.draw_networkx_nodes(
                newGraph,
                realization,
                ax=ax,
                node_size=vertex_size,
                node_color=vertex_color,
                node_shape=vertex_shape,
            )
            nx.draw_networkx_labels(
                newGraph, realization, ax=ax, font_color=font_color, font_size=font_size
            )
            for edge in newGraph.edges(data=True):
                nx.draw_networkx_edges(
                    newGraph,
                    realization,
                    ax=ax,
                    width=edge_width,
                    edge_color=edge_color_array,
                    arrows=True,
                    arrowstyle="-",
                    edgelist=[(edge[0], edge[1])],
                    connectionstyle=f"Arc3, rad = {edge[2]['weight']}",
                )

    @doc_category("Plotting")
    def plot2D(  # noqa: C901
        self,
        projection_matrix: Matrix = None,
        random_seed: int = None,
        coordinates: Sequence[int] = None,
        inf_flex: int | Inf_Flex = None,
        stress: int | Stress = None,
        return_matrix: bool = False,
        vertex_size: int = 300,
        vertex_color: str = "#ff8c00",
        vertex_shape: str = "o",
        vertex_labels: bool = True,
        edge_width: float = 2.5,
        edge_color: (
            str | Sequence[Sequence[Edge]] | dict[str : Sequence[Edge]]
        ) = "black",
        edge_style: str = "solid",
        flex_width: float = 1.5,
        flex_length: float = 0.15,
        flex_color: (
            str | Sequence[Sequence[Edge]] | dict[str : Sequence[Edge]]
        ) = "limegreen",
        flex_style: str = "solid",
        flex_arrowsize: int = 20,
        stress_color: str = "orangered",
        stress_fontsize: int = 10,
        stress_label_pos: float | dict[DirectedEdge, float] = 0.5,
        stress_rotate_labels: bool = True,
        stress_normalization: bool = False,
        font_size: int = 12,
        font_color: str = "whitesmoke",
        canvas_width: float = 6.4,
        canvas_height: float = 4.8,
        aspect_ratio: float = 1.0,
        curved_edges: bool = False,
        connection_style: float | Sequence[float] | dict[Edge, float] = np.pi / 6,
    ) -> Optional[Matrix]:
        """
        Plot this framework in 2D.

        If this framework is in dimensions higher than 2 and projection_matrix
        with coordinates are None a random projection matrix
        containing two orthonormal vectors is generated and used for projection into 2D.
        This matrix is then returned.
        For various formatting options, see :meth:`.Graph.plot`.
        Only coordinates or projection_matrix parameter can be used, not both!

        Parameters
        ----------
        projection_matrix:
            The matrix used for projecting the placement of vertices
            only when they are in dimension higher than 2.
            The matrix must have dimensions (2, dim),
            where dim is the dimension of the currect placements of vertices.
            If None, a random projection matrix is generated.
        random_seed:
            The random seed used for generating the projection matrix.
            When the same value is provided, the framework will plot exactly same.
        coordinates:
            Indices of two coordinates that will be used as the placement in 2D.
        inf_flex:
            Optional parameter for plotting a given infinitesimal flex. The standard
            input format is a ``Matrix`` that is the output of e.g. the method
            ``Framework.inf_flexes``. Alternatively, an ``int`` can be specified
            to directly choose the 0,1,2,...-th nontrivial infinitesimal flex (according
            to the method ``Framework.nontrivial_inf_flexes``) for plotting.
            For these input types, is important to use the same vertex order as the one
            from :meth:`.Graph.vertex_list`.
            If the vertex order needs to be specified, a
            ``dict[Vertex, Sequence[Coordinate]]`` can be provided, which maps the
            vertex labels to vectors (i.e. a sequence of coordinates).
        stress:
            Optional parameter for plotting a given equilibrium stress. The standard
            input format is a ``Matrix`` that is the output of e.g. the method
            ``Framework.stresses``. Alternatively, an ``int`` can be specified
            to directly choose the 0,1,2,...-th equilibrium stress (according
            to the method ``Framework.stresses``) for plotting.
            For these input types, is important to use the same edge order as the one
            from :meth:`.Graph.edge_list`.
            If the edge order needs to be specified, a ``Dict[Edge, Coordinate]``
            can be provided, which maps the edges to numbers
            (i.e. coordinates).
        return_matrix:
            If True the matrix used for projection into 2D is returned.
        vertex_size:
            The size of the vertices.
        vertex_color:
            The color of the vertices. The color can be a string or an rgb (or rgba)
            tuple of floats from 0-1.
        vertex_shape:
            The shape of the vertices specified as as matplotlib.scatter
            marker, one of ``so^>v<dph8``.
        vertex_labels:
            If ``True`` (default), vertex labels are displayed.
        edge_width:
        edge_color:
            If a single color is given as a string or rgb (or rgba) tuple
            of floats from 0-1, then all edges get this color.
            If a (possibly incomplete) partition of the edges is given,
            then each part gets a different color.
            If a dictionary from colors to a list of edge is given,
            edges are colored accordingly.
            The edges missing in the partition/dictionary, are colored black.
        edge_style:
            Edge line style: ``-``/``solid``, ``--``/``dashed``,
            ``-.``/``dashdot`` or ``:``/``dotted``. By default '-'.
        flex_width:
            The width of the infinitesimal flex's arrow tail.
        flex_color:
            The color of the infinitesimal flex is by default 'limegreen'.
        flex_style:
            Line Style: ``-``/``solid``, ``--``/``dashed``,
            ``-.``/``dashdot`` or ``:``/``dotted``. By default '-'.
        flex_length:
            Length of the displayed flex relative to the total canvas
            diagonal in percent. By default 15%.
        flex_arrowsize:
            Size of the arrowhead's length and width.
        stress_color:
            Color of the font used to label the edges with stresses.
        stress_fontsize:
            Fontsize of the stress labels.
        stress_label_pos:
            Position of the stress label along the edge. `float` numbers
            from the interval `[0,1]` are allowed. `0` represents the head
            of the edge, `0.5` the center and `1` the edge's tail. The position
            can either be specified for all edges equally or as a
            `dict[Edge, float]` of ordered edges. Omitted edges are set to `0.5`.
        stress_rotate_labels:
            A boolean indicating whether the stress label should be rotated.
        stress_normalization:
            A boolean indicating whether the stress values should be turned into
            floating point numbers. If ``True``, the stress is automatically normalized.
        font_size:
            The size of the font used for the labels.
        font_color:
            The color of the font used for the labels.
        canvas_width:
            The width of the canvas in inches.
        canvas_height:
            The height of the canvas in inches.
        aspect_ratio:
            The ratio of y-unit to x-unit. By default 1.0.
        curved_edges:
            If the edges are too close to each other, we can decide to
            visualize them as arcs.
        connection_style:
            In case of curvilinear plotting (``curved_edges=True``), the edges
            are displayed as arcs. With this parameter, we can set the
            pitch of these arcs and it is in radians. It can either be
            specified for each arc (``connection_style=0.5``) or individually
            as a ``list`` and ``dict``
            (``connection_style={(0,1):0.5, (1,2):-0.5}``). It is possible to
            provide fewer edges when the input is a ``dict``; the remaining
            edges are padded with zeros in that case.

        Examples
        --------
        >>> from pyrigi import frameworkDB
        >>> F = frameworkDB.Complete(4, dim=1)
        >>> F.plot2D(stress=0);
        >>> F = frameworkDB.Frustum(3)
        >>> F.plot2D(inf_flex=0, stress=0);
        """
        plotting_args = {
            "vertex_size": vertex_size,
            "vertex_color": vertex_color,
            "vertex_shape": vertex_shape,
            "vertex_labels": vertex_labels,
            "edge_width": edge_width,
            "edge_color": edge_color,
            "edge_style": edge_style,
            "font_size": font_size,
            "font_color": font_color,
            "connection_style": connection_style,
        }
        flex_args = {
            "flex_width": flex_width,
            "flex_length": flex_length,
            "flex_color": flex_color,
            "flex_style": flex_style,
            "flex_arrowsize": flex_arrowsize,
        }
        stress_args = {
            "stress_color": stress_color,
            "stress_fontsize": stress_fontsize,
            "stress_label_pos": stress_label_pos,
            "stress_rotate_labels": stress_rotate_labels,
            "stress_normalization": stress_normalization,
            "connection_style": connection_style,
        }

        fig, ax = plt.subplots()
        ax.set_adjustable("datalim")
        fig.set_figwidth(canvas_width)
        fig.set_figheight(canvas_height)
        ax.set_aspect(aspect_ratio)

        if self._dim == 1:
            placement = {}
            for vertex, position in self.realization(
                as_points=True, numerical=True
            ).items():
                placement[vertex] = np.append(np.array(position), 0)

            self._plot_with_2D_realization(
                ax, placement, curved_edges=True, **plotting_args
            )

            if inf_flex is not None:
                self._plot_inf_flex(ax, inf_flex, points=placement, **flex_args)
            if stress is not None:
                self._plot_stress(
                    ax, stress, points=placement, curved_edges=True, **stress_args
                )
            return

        placement = self.realization(as_points=True, numerical=True)
        if self._dim == 2:
            self._plot_with_2D_realization(
                ax, placement, curved_edges=curved_edges, **plotting_args
            )
            if inf_flex is not None:
                self._plot_inf_flex(ax, inf_flex, **flex_args)
            if stress is not None:
                self._plot_stress(ax, stress, curved_edges=curved_edges, **stress_args)
            return

        # dim > 2 -> use projection to 2D
        if coordinates is not None:
            if (
                not isinstance(coordinates, tuple)
                and not isinstance(coordinates, list)
                or len(coordinates) != 2
            ):
                raise ValueError(
                    "coordinates must have length 2!"
                    + " Exactly Two coordinates are necessary for plotting in 2D."
                )
            if np.max(coordinates) >= self._dim:
                raise ValueError(
                    f"Index {np.max(coordinates)} out of range"
                    + f" with placement in dim: {self._dim}."
                )
            projection_matrix = np.zeros((2, self._dim))
            projection_matrix[0, coordinates[0]] = 1
            projection_matrix[1, coordinates[1]] = 1

        if projection_matrix is not None:
            projection_matrix = np.array(projection_matrix)
            if projection_matrix.shape != (2, self._dim):
                raise ValueError(
                    f"The projection matrix has wrong dimensions! \
                    {projection_matrix.shape} instead of (2, {self._dim})."
                )
        if projection_matrix is None:
            projection_matrix = generate_two_orthonormal_vectors(
                self._dim, random_seed=random_seed
            )
            projection_matrix = projection_matrix.T

        for vertex, position in self.realization(
            as_points=False, numerical=True
        ).items():
            placement[vertex] = np.dot(projection_matrix, np.array(position))

        self._plot_with_2D_realization(
            ax, placement, curved_edges=curved_edges, **plotting_args
        )
        if inf_flex is not None:
            self._plot_inf_flex(
                ax,
                inf_flex,
                points=placement,
                projection_matrix=projection_matrix,
                **flex_args,
            )
        if stress is not None:
            self._plot_stress(
                ax,
                stress,
                points=placement,
                projection_matrix=projection_matrix,
                curved_edges=curved_edges,
                **stress_args,
            )
        if return_matrix:
            return projection_matrix

    @doc_category("Plotting")
    def animate3D(
        self,
        vertex_color: str = "#ff8c00",
        vertex_shape: str = "o",
        vertex_size: int = 13.5,
        edge_color: (
            str | Sequence[Sequence[Edge]] | dict[str : Sequence[Edge]]
        ) = "black",
        edge_width: float = 1.1,
        edge_style: str = "solid",
        equal_aspect_ratio: bool = True,
        total_frames: int = 50,
        delay: int = 75,
        rotation_axis: str | Sequence[Coordinate] = None,
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

        Examples
        --------
        >>> from pyrigi import frameworkDB
        >>> F = frameworkDB.Complete(4, dim=3)
        >>> F.animate3D();
        """
        # Creation of the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.grid(False)
        ax.set_axis_off()

        edge_color_array, edge_list_ref = self._resolve_edge_colors(edge_color)

        # Limits of the axes
        abs_list = [list(abs(i)) for i in self._realization.values()]
        abs_list = [max(abs_list[i]) for i in range(len(abs_list))]

        vertices = np.array(
            [
                list(list(self.realization(numerical=True).values())[i])
                for i in range(self._graph.number_of_nodes())
            ]
        )

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
        projection_matrix: Matrix = None,
        random_seed: int = None,
        coordinates: Sequence[int] = None,
        inf_flex: int | Inf_Flex = None,
        stress: int | Stress = None,
        return_matrix: bool = False,
        vertex_size: int = 200,
        vertex_color: str = "#ff8c00",
        vertex_shape: str = "o",
        vertex_labels: bool = True,
        edge_width: float = 2.5,
        edge_color: (
            str | Sequence[Sequence[Edge]] | dict[str : Sequence[Edge]]
        ) = "black",
        edge_style: str = "solid",
        flex_width: float = 2.5,
        flex_length: float = 0.75,
        flex_color: (
            str | Sequence[Sequence[Edge]] | dict[str : Sequence[Edge]]
        ) = "limegreen",
        flex_style: str = "solid",
        flex_arrowsize: int = 20,
        stress_color: str = "orangered",
        stress_fontsize: int = 8,
        stress_label_pos: float | dict[DirectedEdge, float] = 0.5,
        stress_rotate_labels: bool = True,
        stress_normalization: bool = False,
        font_size: int = 12,
        font_color: str = "whitesmoke",
        equal_aspect_ratio: bool = True,
        padding: float = 0.01,
    ) -> Optional[Matrix]:
        """
        Plot the provided framework in 3D.

        If the framework is in a dimension higher than 3 and ``projection_matrix``
        with ``coordinates`` are ``None``, a random projection matrix
        containing three orthonormal vectors is generated and used for projection into 3D.
        This particular matrix is then returned.
        For various formatting options, see :meth:`.Graph.plot`.
        Only the parameter `coordinates` or `projection_matrix` can be used,
        not both at the same time.

        Parameters
        ----------
        projection_matrix:
            The matrix used for projecting the realization of vertices
            when the dimension is higher than 3.
            The matrix must have dimensions ``(3, dim)``,
            where ``dim`` is the dimension of the framework.
            If ``None``, a random projection matrix is generated.
        random_seed:
            The seed used for generating a random projection matrix.
        coordinates:
            Indices of three coordinates to which the framework is projected.
        return_matrix:
            If ``True``, the matrix used for projection into 3D is returned.

        Notes
        -----
        See :meth:`.Framework._plot_using_projection_matrix_3D` for a full
        list of parameters.

        Examples
        --------
        >>> from pyrigi import frameworkDB
        >>> F = frameworkDB.Octahedron(realization="Bricard_plane")
        >>> F.plot3D(inf_flex=0, stress=0);
        """
        plotting_args = {
            "vertex_size": vertex_size,
            "vertex_color": vertex_color,
            "vertex_shape": vertex_shape,
            "vertex_labels": vertex_labels,
            "edge_width": edge_width,
            "edge_color": edge_color,
            "edge_style": edge_style,
            "font_size": font_size,
            "font_color": font_color,
        }
        flex_args = {
            "flex_width": flex_width,
            "flex_length": flex_length,
            "flex_color": flex_color,
            "flex_style": flex_style,
            "flex_arrowsize": flex_arrowsize,
        }
        stress_args = {
            "stress_color": stress_color,
            "stress_fontsize": stress_fontsize,
            "stress_label_pos": stress_label_pos,
            "stress_rotate_labels": stress_rotate_labels,
            "stress_normalization": stress_normalization,
        }

        if self._dim == 1 or self._dim == 2:
            return self.plot2D(
                inf_flex=inf_flex,
                stress=stress,
                **(plotting_args | flex_args | stress_args),
            )

        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111, projection="3d")
        ax.grid(False)
        ax.set_axis_off()
        placement = self.realization(as_points=True, numerical=True)
        if self._dim == 3:
            self._plot_with_3D_realization(
                ax,
                placement,
                equal_aspect_ratio=equal_aspect_ratio,
                padding=padding,
                **plotting_args,
            )
            if inf_flex is not None:
                self._plot_inf_flex(ax, inf_flex, points=placement, **flex_args)
            if stress is not None:
                self._plot_stress(ax, stress, points=placement, **stress_args)
            return

        # dim > 3 -> use projection to 3D
        if coordinates is not None:
            if (
                not isinstance(coordinates, tuple)
                and not isinstance(coordinates, list)
                or len(coordinates) != 3
            ):
                raise ValueError(
                    "The parameter `coordinates` must have length 3!"
                    + " Exactly three coordinates are necessary for plotting in 3D."
                )
            if np.max(coordinates) >= self._dim:
                raise ValueError(
                    f"Index {np.max(coordinates)} out of range"
                    + f" with placement in dim: {self._dim}."
                )
            projection_matrix = np.zeros((3, self._dim))
            projection_matrix[0, coordinates[0]] = 1
            projection_matrix[1, coordinates[1]] = 1
            projection_matrix[2, coordinates[2]] = 1

        if projection_matrix is not None:
            projection_matrix = np.array(projection_matrix)
            if projection_matrix.shape != (3, self._dim):
                raise ValueError(
                    f"The projection matrix has wrong dimensions! \
                    {projection_matrix.shape} instead of (3, {self._dim})."
                )
        else:
            projection_matrix = generate_three_orthonormal_vectors(
                self._dim, random_seed=random_seed
            )
            projection_matrix = projection_matrix.T
        for vertex, position in self.realization(
            as_points=False, numerical=True
        ).items():
            placement[vertex] = np.dot(projection_matrix, np.array(position))

        self._plot_with_3D_realization(
            ax,
            placement,
            equal_aspect_ratio=equal_aspect_ratio,
            padding=padding,
            **plotting_args,
        )
        if inf_flex is not None:
            self._plot_inf_flex(
                ax,
                inf_flex,
                points=placement,
                projection_matrix=projection_matrix,
                **flex_args,
            )
        if stress is not None:
            self._plot_stress(
                ax,
                stress,
                points=placement,
                projection_matrix=projection_matrix,
                **stress_args,
            )
        if return_matrix:
            return projection_matrix

    @doc_category("Plotting")
    def _plot_with_3D_realization(
        self,
        ax: Axes,
        realization: dict[Vertex, Point],
        vertex_color: str = "#ff8c00",
        vertex_size: int = 200,
        vertex_shape: str = "o",
        vertex_labels: bool = True,
        font_color: str = "whitesmoke",
        font_size: int = 10,
        edge_width: float = 2.5,
        edge_color: (
            str | Sequence[Sequence[Edge]] | dict[str : Sequence[Edge]]
        ) = "black",
        edge_style: str = "solid",
        equal_aspect_ratio: bool = True,
        padding: float = 0.01,
    ) -> None:
        """
        Plot the graph of the framework with the given realization in the plane.

        For description of other parameters see :meth:`.Framework.plot`.

        Parameters
        ----------
        inf_flex:
            Optional parameter for plotting a given infinitesimal flex. It is
            important to use the same vertex order as the one
            from :meth:`.Graph.vertex_list`.
            Alternatively, an ``int`` can be specified to choose the 0,1,2,...-th
            nontrivial infinitesimal flex for plotting.
            Lastly, a ``Dict[Vertex, Sequence[Coordinate]]`` can be provided, which
            maps the vertex labels to vectors (i.e. a sequence of coordinates).
        projection_matrix:
            The matrix used for projection.
            The matrix must have dimensions ``(3, dim)``,
            where ``dim`` is the dimension of the framework.
        vertex_color:
            The color of the vertices. The color can be a string or an rgb (or rgba)
            tuple of floats from 0-1.
        vertex_size:
            The size of the vertices.
        vertex_shape:
            The shape of the vertices specified as as matplotlib.scatter
            marker, one of ``so^>v<dph8``.
        vertex_labels:
            If ``True`` (default), vertex labels are displayed.
        font_size:
            The size of the font used for the labels.
        font_color:
            The color of the font used for the labels.
        edge_width:
        edge_color:
            If a single color is given as a string or rgb (or rgba) tuple
            of floats from 0-1, then all edges get this color.
            If a (possibly incomplete) partition of the edges is given,
            then each part gets a different color.
            If a dictionary from colors to a list of edge is given,
            edges are colored accordingly.
            The edges missing in the partition/dictionary, are colored black.
        edge_style:
            Edge line style: ``-``/``solid``, ``--``/``dashed``,
            ``-.``/``dashdot`` or ``:``/``dotted``. By default '-'.
        equal_aspect_ratio:
            Determines whether the aspect ratio of the plot is equal in all space
            directions or whether it is adjusted depending on the framework's size
            in `x`, `y` and `z`-direction individually.
        padding:
            Specifies the white space around the framework.

        Notes
        -----
        The parameters for `inf_flex`-plotting are listed in
        the API reference.
        """
        # Create a figure for the rapresentation of the framework

        edge_color_array, edge_list_ref = self._resolve_edge_colors(edge_color)

        # Draw the vertices as points in the 3D enviroment
        x_nodes = [realization[node][0] for node in self._graph.nodes]
        y_nodes = [realization[node][1] for node in self._graph.nodes]
        z_nodes = [realization[node][2] for node in self._graph.nodes]
        ax.scatter(
            x_nodes,
            y_nodes,
            z_nodes,
            c=vertex_color,
            s=vertex_size,
            marker=vertex_shape,
        )
        if equal_aspect_ratio:
            min_val = min(x_nodes + y_nodes + z_nodes) - padding
            max_val = max(x_nodes + y_nodes + z_nodes) + padding
            ax.set_zlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_xlim(min_val, max_val)
        else:
            ax.set_zlim(min(z_nodes) - padding, max(z_nodes) + padding)
            ax.set_ylim(min(y_nodes) - padding, max(y_nodes) + padding)
            ax.set_xlim(min(x_nodes) - padding, max(x_nodes) + padding)

        for i in range(len(edge_list_ref)):
            edge = edge_list_ref[i]
            x = [realization[edge[0]][0], realization[edge[1]][0]]
            y = [realization[edge[0]][1], realization[edge[1]][1]]
            z = [realization[edge[0]][2], realization[edge[1]][2]]
            ax.plot(x, y, z, c=edge_color_array[i], lw=edge_width, linestyle=edge_style)
        for node in self._graph.nodes:
            x, y, z, *others = realization[node]
            # To show the name of the vertex
            if vertex_labels:
                ax.text(
                    x,
                    y,
                    z,
                    str(node),
                    color=font_color,
                    fontsize=font_size,
                    ha="center",
                    va="center",
                )

    @doc_category("Plotting")
    def _plot_inf_flex(  # noqa: C901
        self,
        ax: Axes,
        inf_flex: Matrix | Inf_Flex,
        points: dict[Vertex, Point] = None,
        flex_width: float = 2.5,
        flex_length: float = 0.65,
        flex_color: (
            str | Sequence[Sequence[Edge]] | dict[str : Sequence[Edge]]
        ) = "limegreen",
        flex_style: str = "solid",
        flex_arrowsize: int = 20,
        projection_matrix: Matrix = None,
        **kwargs,
    ) -> None:
        """
        Adds infinitesimal flexes based in the `points` as vectors to the axis `ax`.

        Parameters
        ----------
        ax:
        inf_flex:
            Optional parameter for plotting a given infinitesimal flex. It is
            important to use the same vertex order as the one
            from :meth:`.Graph.vertex_list`.
            Alternatively, an ``int`` can be specified to choose the 0,1,2,...-th
            nontrivial infinitesimal flex for plotting.
            Lastly, a ``dict[Vertex, Sequence[Coordinate]]`` can be provided, which
            maps the vertex labels to vectors (i.e. a sequence of coordinates).
        flex_width:
            Width of the infinitesimal flex's arrowtail.
        flex_length:
            Length of the displayed flex relative to the total canvas
            diagonal in percent. By default 15%.
        flex_color:
            The color of the infinitesimal flex is by default 'limegreen'.
        flex_style:
            Line Style: ``-``/``solid``, ``--``/``dashed``,
            ``-.``/``dashdot`` or ``:``/``dotted``. By default '-'.
        projection_matrix:
        """
        inf_flex_pointwise = None
        if isinstance(inf_flex, int) and inf_flex >= 0:
            inf_flex_basis = self.nontrivial_inf_flexes()
            if inf_flex >= len(inf_flex_basis):
                raise IndexError(
                    "The value of inf_flex exceeds "
                    + "the dimension of the space "
                    + "of infinitesimal flexes."
                )
            inf_flex_pointwise = self._transform_inf_flex_to_pointwise(
                inf_flex_basis[inf_flex]
            )
        elif isinstance(inf_flex, Matrix):
            inf_flex_pointwise = self._transform_inf_flex_to_pointwise(inf_flex)
        elif isinstance(inf_flex, dict) and all(
            isinstance(inf_flex[key], Sequence) for key in inf_flex.keys()
        ):
            inf_flex_pointwise = inf_flex
        else:
            raise TypeError(
                "inf_flex does not have the correct Type or the `int` is too large."
            )

        if not self.is_dict_inf_flex(inf_flex_pointwise):
            raise ValueError("The provided `inf_flex` is not an infinitesimal flex.")
        if self._dim == 1:
            inf_flex_pointwise = {
                v: [inf_flex_pointwise[v], 0] for v in inf_flex_pointwise.keys()
            }
        if projection_matrix is not None:
            inf_flex_pointwise = {
                v: np.dot(projection_matrix, np.array(flex))
                for v, flex in inf_flex_pointwise.items()
            }

        if points is None:
            points = self.realization(as_points=True, numerical=True)
        elif not isinstance(points, dict):
            raise TypeError("Realization has the wrong type!")
        elif not all(
            [
                len(points[v]) == len(points[list(points.keys())[0]])
                and len(points[v]) in [2, 3]
                for v in self._graph.nodes
            ]
        ):
            raise ValueError(
                "Not all values in the realization have the same"
                + "length and the dimension needs to be 2 or 3."
            )

        magnidutes = []
        for flex_key in inf_flex_pointwise.keys():
            if len(inf_flex_pointwise[flex_key]) != len(points[list(points.keys())[0]]):
                raise ValueError(
                    "The infinitesimal flex needs to be "
                    + f"in dimension {len(points[list(points.keys())[0]])}."
                )
            inf_flex = [float(x) for x in inf_flex_pointwise[flex_key]]
            magnidutes.append(np.linalg.norm(inf_flex))

        # normalize the edge lengths by the Euclidean norm of the longest one
        flex_mag = max(magnidutes)
        for flex_key in inf_flex_pointwise.keys():
            if not all(entry == 0 for entry in inf_flex_pointwise[flex_key]):
                inf_flex_pointwise[flex_key] = tuple(
                    flex / flex_mag for flex in inf_flex_pointwise[flex_key]
                )
        # Delete the edges with zero length
        inf_flex_pointwise = {
            flex_key: np.array(inf_flex_pointwise[flex_key], dtype=float)
            for flex_key in inf_flex_pointwise.keys()
            if not all(entry == 0 for entry in inf_flex_pointwise[flex_key])
        }

        if len(inf_flex_pointwise[list(inf_flex_pointwise.keys())[0]]) == 2:
            x_canvas_width = ax.get_xlim()[1] - ax.get_xlim()[0]
            y_canvas_width = ax.get_ylim()[1] - ax.get_ylim()[0]
            arrow_length = np.sqrt(x_canvas_width**2 + y_canvas_width**2) * flex_length
            H = nx.DiGraph([(v, str(v) + "_flex") for v in inf_flex_pointwise.keys()])
            H_placement = {
                str(v)
                + "_flex": np.array(
                    [
                        points[v][0] + arrow_length * inf_flex_pointwise[v][0],
                        points[v][1] + arrow_length * inf_flex_pointwise[v][1],
                    ],
                    dtype=float,
                )
                for v in inf_flex_pointwise.keys()
            }
            H_placement.update(
                {v: np.array(points[v], dtype=float) for v in inf_flex_pointwise.keys()}
            )
            if (
                not isinstance(flex_color, str | list)
                or isinstance(flex_color, list)
                and not len(flex_color) == len(inf_flex_pointwise)
            ):
                raise TypeError(
                    "`flex_color` must either be a `str` specifying"
                    + "a color or a list of strings with the same"
                    + "number as the nonzero flexes."
                )
            nx.draw(
                H,
                pos=H_placement,
                ax=ax,
                arrows=True,
                arrowsize=flex_arrowsize,
                node_size=0,
                node_color="white",
                width=flex_width,
                edge_color=flex_color,
                style=flex_style,
                **kwargs,
            )
        elif self._dim == 3:
            for v in inf_flex_pointwise.keys():
                ax.quiver(
                    points[v][0],
                    points[v][1],
                    points[v][2],
                    inf_flex_pointwise[v][0],
                    inf_flex_pointwise[v][1],
                    inf_flex_pointwise[v][2],
                    color=flex_color,
                    lw=flex_width,
                    linestyle=flex_style,
                    length=flex_length,
                    arrow_length_ratio=0.25,
                )
        else:
            raise ValueError(
                "The dimension of the infinitesimal flex needs to be between 1 and 3."
            )

    @doc_category("Plotting")
    def _plot_stress(  # noqa: C901
        self,
        ax: Axes,
        stress: Matrix | Stress,
        points: dict[Vertex, Point] = None,
        stress_color: str = "orangered",
        stress_fontsize: int = 8,
        stress_label_pos: float | dict[DirectedEdge, float] = 0.5,
        stress_rotate_labels: bool = True,
        stress_normalization: bool = False,
        connection_style: float | dict[DirectedEdge, float] = 0.5,
        curved_edges: bool = False,
        **kwargs,
    ) -> None:
        """
        ax:
        stress:
            Optional parameter for plotting a given equilibrium stress. The standard
            input format is a ``Matrix`` that is the output of e.g. the method
            ``Framework.stresses``. Alternatively, an ``int`` can be specified
            to directly choose the 0,1,2,...-th equilibrium stress (according
            to the method ``Framework.stresses``) for plotting.
            For these input types, is important to use the same edge order as the one
            from :meth:`.Graph.edge_list`.
            If the edge order needs to be specified, a ``Dict[Edge, Coordinate]``
            can be provided, which maps the edges to numbers
            (i.e. coordinates).
        points:
            It is possible to provide an alternative realization.
        stress_color:
            Color of the font used to label the edges with stresses.
        stress_fontsize:
            Fontsize of the stress labels.
        stress_label_pos:
            Position of the stress label along the edge. `float` numbers
            from the interval `[0,1]` are allowed. `0` represents the head
            of the edge, `0.5` the center and `1` the edge's tail. The position
            can either be specified for all edges equally or as a
            `dict[Edge, float]` of ordered edges. Omitted edges are set to `0.5`.
        stress_rotate_labels:
            A boolean indicating whether the stress label should be rotated.
        stress_normalization:
            A boolean indicating whether the stress values should be turned into
            floating point numbers. If ``True``, the stress is automatically normalized.
        """
        stress_edgewise = None
        if isinstance(stress, int) and stress >= 0:
            stresses = self.stresses()
            if stress >= len(stresses):
                raise IndexError(
                    "The value of `stress` exceeds "
                    + "the dimension of the space "
                    + "of equilibrium stresses."
                )
            stress_edgewise = self._transform_stress_to_edgewise(stresses[stress])
        elif isinstance(stress, Matrix):
            stress_edgewise = self._transform_stress_to_edgewise(stress)
        elif isinstance(stress, dict) and all(
            isinstance(stress[key], int | float | str) for key in stress.keys()
        ):
            stress_edgewise = stress
        else:
            raise TypeError("`stress` does not have the correct Type.")

        if not self.is_dict_stress(stress_edgewise):
            raise ValueError("The provided `stress` is not an equilibrium stress.")

        if points is None:
            points = self.realization(as_points=True, numerical=True)

        if stress_normalization:
            numerical_stress = {
                edge: float(sympify(w).evalf(10)) for edge, w in stress_edgewise.items()
            }
            _stress = {
                edge: round(w / np.linalg.norm(list(numerical_stress.values())), 2)
                for edge, w in numerical_stress.items()
            }
        else:
            _stress = stress_edgewise
        if isinstance(stress_label_pos, dict):
            if not all([self._graph.has_edge(*e) for e in stress_label_pos.keys()]):
                raise ValueError(
                    "The `stress_label_pos` dictionary must contain the same "
                    + "edges as the stress dictionary."
                )
            for edge in self._graph.edge_list(as_tuples=True):
                stress_keys = [set(e) for e in stress_label_pos.keys()]
                if set(edge) not in stress_keys:
                    stress_label_pos[edge] = 0.5
        elif isinstance(stress_label_pos, float):
            label_float = stress_label_pos
            stress_label_pos = {}
            for edge in self._graph.edge_list(as_tuples=True):
                stress_label_pos[edge] = label_float
        else:
            raise TypeError(
                "`stress_label_pos` must be either a float or a dictionary."
            )
        if len(points[list(points.keys())[0]]) == 2:
            if curved_edges:
                newGraph = nx.MultiDiGraph()
                connection_style = self._resolve_connection_style(connection_style)
                for e, style in connection_style.items():
                    newGraph.add_edge(e[0], e[1], weight=style)
                plt.box(False)  # Manually removes the frame of the plot
                for e in newGraph.edges(data=True):
                    edge = tuple([e[0], e[1]])
                    nx.draw_networkx_edge_labels(
                        newGraph,
                        ax=ax,
                        pos=points,
                        edge_labels={edge: _stress[edge]},
                        font_color=stress_color,
                        font_size=stress_fontsize,
                        label_pos=stress_label_pos[edge],
                        rotate=stress_rotate_labels,
                        connectionstyle=f"Arc3, rad = {e[2]['weight']}",
                        **kwargs,
                    )
            else:
                for edge in self._graph.edges:
                    nx.draw_networkx_edge_labels(
                        self._graph,
                        ax=ax,
                        pos=points,
                        edge_labels={edge: _stress[edge]},
                        font_color=stress_color,
                        font_size=stress_fontsize,
                        label_pos=stress_label_pos[edge],
                        rotate=stress_rotate_labels,
                        **kwargs,
                    )
        elif len(points[list(points.keys())[0]]) == 3:
            for edge in stress_label_pos.keys():
                pos = [
                    points[edge[0]][i]
                    + stress_label_pos[edge] * (points[edge[1]][i] - points[edge[0]][i])
                    for i in range(3)
                ]
                ax.text(
                    pos[0],
                    pos[1],
                    pos[2],
                    str(_stress[edge]),
                    color=stress_color,
                    fontsize=stress_fontsize,
                    ha="center",
                    va="center",
                    **kwargs,
                )
        else:
            raise ValueError(
                "The method `_plot_stress` is currently only implemented"
                + " for frameworks in 1 up to 3 dimensions."
            )

    @doc_category("Plotting")
    def plot(
        self,
        **kwargs,
    ) -> Optional[Matrix]:
        """
        Plot the framework.

        Notes
        -----
        If the dimension of the framework is greater than 3, ``ValueError`` is raised,
        use :meth:`.Framework.plot2D` or :meth:`.Framework.plot3D` instead.
        For various formatting options, see :meth:`.Graph.plot`.

        TODO
        ----
        Implement plotting in dimension 3 and
        """
        if self._dim == 3:
            return self.plot3D(**kwargs)
        elif self._dim > 3:
            raise ValueError(
                "This framework is in higher dimension than 3!"
                + " For projection into 2D use F.plot2D(),"
                + " for projection into 3D use F.plot3D()."
            )
        else:
            return self.plot2D(**kwargs)

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
        If ``rand_range=None``, then the range is set to ``(-10 * n^2 * d)``.

        TODO
        ----
        Set the correct default range value.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if rand_range is None:
            b = 10**4 * graph.number_of_nodes() ** 2 * dim
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
        check_integrality_and_range(dim, "dimension d", 1)
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
        >>> F = Framework.Simplicial(Graph([(0,1), (1,2), (2,3), (0,3)]), 4);
        >>> F.realization(as_points=True)
        {0: [0, 0, 0, 0], 1: [1, 0, 0, 0], 2: [0, 1, 0, 0], 3: [0, 0, 1, 0]}
        >>> F = Framework.Simplicial(Graph([(0,1), (1,2), (2,3), (0,3)]));
        >>> F.realization(as_points=True)
        {0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}
        """
        if dim is None:
            dim = graph.number_of_nodes() - 1
        check_integrality_and_range(
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
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
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
            raise ValueError("The list of points cannot be empty.")

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
            if v not in realization:
                raise KeyError("Vertex {vertex} is not a key of the given realization!")
            if not len(realization[v]) == self.dimension():
                raise IndexError(
                    f"The element {realization[v]} does not have "
                    f"the dimension {self.dimension()}!"
                )
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
        if vertex not in self._realization:
            raise KeyError("Vertex {vertex} is not a key of the given realization!")
        if not len(point) == self.dimension():
            raise IndexError(
                f"The point {point} does not have the dimension {self.dimension()}!"
            )
        self._realization[vertex] = Matrix(point)

    @doc_category("Framework manipulation")
    def set_vertex_positions_from_lists(
        self, vertices: Sequence[Vertex], points: Sequence[Point]
    ) -> None:
        """
        Change the coordinates of a given list of vertices.

        Examples
        ----
        >>> F = Framework.Complete([(0,0),(0,0),(1,0),(1,0)]);
        >>> F.realization(as_points=True)
        {0: [0, 0], 1: [0, 0], 2: [1, 0], 3: [1, 0]}
        >>> F.set_vertex_positions_from_lists([1,3], [(0,1),(1,1)]);
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
                "The list of vertices does not have the same length as the list of points"
            )
        self.set_vertex_positions({v: pos for v, pos in zip(vertices, points)})

    @doc_category("Framework manipulation")
    def set_vertex_positions(self, subset_of_realization: dict[Vertex, Point]):
        """
        Change the coordinates of vertices given by a dictionary.

        Examples
        ----
        >>> F = Framework.Complete([(0,0),(0,0),(1,0),(1,0)]);
        >>> F.realization(as_points=True)
        {0: [0, 0], 1: [0, 0], 2: [1, 0], 3: [1, 0]}
        >>> F.set_vertex_positions({1:(0,1),3:(1,1)});
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
        vertex_order = self._check_vertex_order(vertex_order)
        edge_order = self._check_edge_order(edge_order)

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

    def pinned_rigidity_matrix(
        self,
        pinned_vertices: dict[Vertex, Sequence[int]] = None,
        vertex_order: Sequence[Vertex] = None,
        edge_order: Sequence[Edge] = None,
    ) -> Matrix:
        r"""
        Construct the rigidity matrix of the framework.

        Parameters
        ----------
        vertex_order:
            A list of vertices, providing the ordering for the columns
            of the rigidity matrix.
        edge_order:
            A list of edges, providing the ordering for the rows
            of the rigidity matrix.

        TODO
        ----
        definition of pinned rigidity matrix, tests

        Examples
        --------
        >>> F = Framework(Graph([[0, 1], [0, 2]]), {0: [0, 0], 1: [1, 0], 2: [1, 1]})
        >>> F.pinned_rigidity_matrix()
        Matrix([
        [-1,  0, 1, 0, 0, 0],
        [-1, -1, 0, 0, 1, 1],
        [ 1,  0, 0, 0, 0, 0],
        [ 0,  1, 0, 0, 0, 0],
        [ 0,  0, 1, 0, 0, 0]])
        """
        vertex_order = self._check_vertex_order(vertex_order)
        edge_order = self._check_edge_order(edge_order)
        rigidity_matrix = self.rigidity_matrix(
            vertex_order=vertex_order, edge_order=edge_order
        )

        if pinned_vertices is None:
            freedom = self._dim * (self._dim + 1) // 2
            pinned_vertices = {}
            upper = self._dim + 1
            for v in vertex_order:
                upper -= 1
                frozen_coord = []
                for i in range(upper):
                    if freedom > 0:
                        frozen_coord.append(i)
                        freedom -= 1
                    else:
                        pinned_vertices[v] = frozen_coord
                        break
                pinned_vertices[v] = frozen_coord
        else:
            number_pinned = sum([len(coord) for coord in pinned_vertices.values()])
            if number_pinned > self._dim * (self._dim + 1) // 2:
                raise ValueError(
                    "The maximal number of coordinates that"
                    f"can be pinned is {self._dim * (self._dim + 1) // 2}, "
                    f"but you provided {number_pinned}."
                )
            for v in pinned_vertices:
                if min(pinned_vertices[v]) < 0 or max(pinned_vertices[v]) >= self._dim:
                    raise ValueError("Coordinate indices out of range.")

        pinning_rows = []
        for v in pinned_vertices:
            for coord in pinned_vertices[v]:
                idx = vertex_order.index(v)
                new_row = Matrix.zeros(1, self._dim * self._graph.number_of_nodes())
                new_row[idx * self._dim + coord] = 1
                pinning_rows.append(new_row)
        pinned_rigidity_matrix = Matrix.vstack(rigidity_matrix, *pinning_rows)
        return pinned_rigidity_matrix

    @doc_category("Infinitesimal rigidity")
    def is_dict_stress(self, dict_stress: dict[Edge, Coordinate], **kwargs) -> bool:
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
        self._check_edge_order(stress_edge_list)
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
        stress: Stress,
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
        edge_order = self._check_edge_order(edge_order=edge_order)
        return is_zero_vector(
            Matrix(stress).transpose() * self.rigidity_matrix(edge_order=edge_order),
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Infinitesimal rigidity")
    def is_stress(self, stress: Stress | dict[Edge, Coordinate], **kwargs) -> bool:
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
        vertex_order = self._check_vertex_order(vertex_order)
        edge_order = self._check_edge_order(edge_order)
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
        vertex_order = self._check_vertex_order(vertex_order)
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
        vertex_order = self._check_vertex_order(vertex_order)
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

        TODO
        ----
        tests
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

        TODO
        ----
        tests

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
        self._check_vertex_order(list(other_realization.keys()))

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

        if not nx.utils.graphs_equal(self._graph, other_framework._graph):
            raise ValueError("Underlying graphs are not same.")

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
        self._check_vertex_order(list(other_realization.keys()))

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

        if not nx.utils.graphs_equal(self._graph, other_framework._graph):
            raise ValueError("Underlying graphs are not same.")
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
                    "The dimension of the vector has to be the same as of the framework."
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

    @doc_category("Other")
    def edge_lengths(self, numerical: bool = False) -> dict[Edge, Coordinate]:
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
                "run 'pip install pyrigi[meshing]'"
            )

        if (
            holes_distance <= 0
            or holes_diameter <= 0
            or bar_width <= 0
            or bar_height <= 0
        ):
            raise ValueError("Use only positive values for the parameters.")

        if bar_width <= holes_diameter:
            raise ValueError("The bar width must be greater than the holes diameter.")

        if holes_distance <= 2 * holes_diameter:
            raise ValueError(
                "The distance between the holes must be greater "
                "than twice the holes diameter."
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

    @doc_category("Other")
    def _transform_inf_flex_to_pointwise(
        self, inf_flex: Matrix, vertex_order: Sequence[Vertex] = None
    ) -> dict[Vertex, list[Coordinate]]:
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
        vertex_order = self._check_vertex_order(vertex_order)
        return {
            vertex_order[i]: [inf_flex[i * self.dim() + j] for j in range(self.dim())]
            for i in range(len(vertex_order))
        }

    @doc_category("Other")
    def _transform_stress_to_edgewise(
        self, stress: Matrix, edge_order: Sequence[Edge] = None
    ) -> dict[Edge, Coordinate]:
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
        edge_order = self._check_edge_order(edge_order)
        return {tuple(edge_order[i]): stress[i] for i in range(len(edge_order))}

    @doc_category("Infinitesimal rigidity")
    def is_vector_inf_flex(
        self,
        inf_flex: Sequence[Coordinate],
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
        vertex_order = self._check_vertex_order(vertex_order)
        return is_zero_vector(
            self.rigidity_matrix(vertex_order=vertex_order) * Matrix(inf_flex),
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Infinitesimal rigidity")
    def is_dict_inf_flex(
        self, vert_to_flex: dict[Vertex, Sequence[Coordinate]], **kwargs
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
        self._check_vertex_order(list(vert_to_flex.keys()))
        dict_to_list = []

        for v in self._graph.vertex_list():
            if v not in vert_to_flex:
                raise ValueError(
                    f"Vertex {v} must be in the dictionary `vert_to_flex`."
                )
            dict_to_list += list(vert_to_flex[v])

        return self.is_vector_inf_flex(
            dict_to_list, vertex_order=self._graph.vertex_list(), **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    def is_vector_nontrivial_inf_flex(
        self,
        inf_flex: Sequence[Coordinate],
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
        vertex_order = self._check_vertex_order(vertex_order)
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
        self, vert_to_flex: dict[Vertex, Sequence[Coordinate]], **kwargs
    ) -> bool:
        r"""
        Return whether a dictionary specifies an infinitesimal flex which is nontrivial.

        Definitions
        -----------
        :prf:ref:`Nontrivial infinitesimal Flex <def-trivial-inf-flex>`

        Parameters
        ----------
        inf_flex:
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
        self._check_vertex_order(list(vert_to_flex.keys()))
        dict_to_list = []

        for v in self._graph.vertex_list():
            if v not in vert_to_flex:
                raise ValueError(
                    f"Vertex {v} must be in the dictionary `vert_to_flex`."
                )
            dict_to_list += list(vert_to_flex[v])

        return self.is_vector_nontrivial_inf_flex(
            dict_to_list, vertex_order=self._graph.vertex_list(), **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    def is_nontrivial_flex(
        self,
        inf_flex: Sequence[Coordinate] | dict[Vertex, Sequence[Coordinate]],
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
    def is_vector_trivial_inf_flex(
        self, inf_flex: Sequence[Coordinate], **kwargs
    ) -> bool:
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
        self, vert_to_flex: dict[Vertex, Sequence[Coordinate]], **kwargs
    ) -> bool:
        r"""
        Return whether an infinitesimal flex specified by a dictionary is trivial.

        Definitions
        -----------
        :prf:ref:`Trivial infinitesimal flex <def-trivial-inf-flex>`

        Parameters
        ----------
        inf_flex:
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
        self._check_vertex_order(list(vert_to_flex.keys()))
        dict_to_list = []

        for v in self._graph.vertex_list():
            if v not in vert_to_flex:
                raise ValueError(
                    f"Vertex {v} must be in the dictionary `vert_to_flex`."
                )
            dict_to_list += list(vert_to_flex[v])

        return self.is_vector_trivial_inf_flex(
            dict_to_list, vertex_order=self._graph.vertex_list(), **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    def is_trivial_flex(
        self,
        inf_flex: Sequence[Coordinate] | dict[Vertex, Sequence[Coordinate]],
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

    @doc_category("Other")
    def _check_vertex_order(self, vertex_order=Sequence[Vertex]) -> list[Vertex]:
        """
        Checks whether the provided `vertex_order` contains the same elements
        as the graph's vertex set.

        Parameters
        ----------
        vertex_order:
            List of vertices in the preferred order

        Notes
        -----
        Throws an error if the vertices in `vertex_order` do not agree with the
        underlying graphs's vertices.
        """
        if vertex_order is None:
            return self._graph.vertex_list()
        else:
            if not self._graph.number_of_nodes() == len(vertex_order) or not set(
                self._graph.vertex_list()
            ) == set(vertex_order):
                raise ValueError(
                    "New vertex set must contain exactly "
                    + "the same vertices as the underlying graph!"
                )
            return list(vertex_order)

    @doc_category("Other")
    def _check_edge_order(self, edge_order=Sequence[Edge]) -> list[Edge]:
        """
        Checks whether the provided `edge_order` contains the same elements
        as the graph's edge set.

        Parameters
        ----------
        edge_order:
            List of edges in the preferred order

        Notes
        -----
        Throws an error if the edges in `edge_order` do not agree with the
        underlying graphs's edges.
        """
        if edge_order is None:
            return self._graph.edge_list()
        else:
            if not self._graph.number_of_edges() == len(edge_order) or not all(
                [
                    set(e) in [set(e) for e in edge_order]
                    for e in self._graph.edge_list()
                ]
            ):
                raise ValueError(
                    "edge_order must contain exactly the same edges as the graph!"
                )
            return list(edge_order)

    @doc_category("Plotting")
    def _resolve_connection_style(self, connection_style: str) -> str:
        """
        Resolve the connection style for the visualization of the framework.

        Parameters
        ----------
        connection_style:
            The connection style for the visualization of the framework.
        """
        G = self._graph
        if isinstance(connection_style, float):
            connection_style = {
                e: connection_style for e in G.edge_list(as_tuples=True)
            }
        elif isinstance(connection_style, list):
            if not G.number_of_edges() == len(connection_style):
                raise AttributeError(
                    "The provided `connection_style` doesn't have the correct length."
                )
            connection_style = {
                e: style
                for e, style in zip(G.edge_list(as_tuples=True), connection_style)
            }
        elif isinstance(connection_style, dict):
            if (
                not all(
                    [
                        isinstance(e, tuple)
                        and len(e) == 2
                        and isinstance(v, float | int)
                        for e, v in connection_style.items()
                    ]
                )
                or not all(
                    [
                        set(key) in [set([e[0], e[1]]) for e in G.edge_list()]
                        for key in connection_style.keys()
                    ]
                )
                or any(
                    [set(key) for key in connection_style.keys()].count(e) > 1
                    for e in [set(key) for key in connection_style.keys()]
                )
            ):
                raise AttributeError(
                    "The provided `connection_style` contains different edges "
                    + "than the underlying graph or has an incorrect format."
                )
            connection_style = {
                e: 0
                for e in G.edge_list(as_tuples=True)
                if not (
                    e in connection_style.keys()
                    or tuple([e[1], e[0]]) in connection_style.keys()
                )
            } | {
                (tuple(e) if e in G.edge_list() else tuple([e[1], e[0]])): style
                for e, style in connection_style.items()
            }
        else:
            raise TypeError(
                "The provided `connection_style` does not have the appropriate type."
            )
        return connection_style

    @doc_category("Plotting")
    def _resolve_edge_colors(
        self, edge_color: str | Sequence[Sequence[Edge]] | dict[str : Sequence[Edge]]
    ) -> tuple[list, list]:
        """
        Return the lists of colors and edges in the format for plotting.
        """
        G = self._graph
        edge_list = G.edge_list()
        edge_list_ref = []
        edge_color_array = []

        if isinstance(edge_color, str):
            return [edge_color for _ in edge_list], edge_list
        if isinstance(edge_color, list):
            edges_partition = edge_color
            colors = distinctipy.get_colors(
                len(edges_partition), colorblind_type="Deuteranomaly", pastel_factor=0.2
            )
            for i, part in enumerate(edges_partition):
                for e in part:
                    if not G.has_edge(e[0], e[1]):
                        raise ValueError(
                            "The input includes a pair that is not an edge."
                        )
                    edge_color_array.append(colors[i])
                    edge_list_ref.append(tuple(e))
        elif isinstance(edge_color, dict):
            color_edges_dict = edge_color
            for color, edges in color_edges_dict.items():
                for e in edges:
                    if not G.has_edge(e[0], e[1]):
                        raise ValueError(
                            "The input includes an edge that is not part of the framework"
                        )
                    edge_color_array.append(color)
                    edge_list_ref.append(tuple(e))
        else:
            raise ValueError("The input color_edge has none of the supported formats.")
        for e in edge_list:
            if (e[0], e[1]) not in edge_list_ref and (e[1], e[0]) not in edge_list_ref:
                edge_color_array.append("black")
                edge_list_ref.append(e)
        if len(edge_list_ref) > G.number_of_edges():
            multiple_colored = [
                e
                for e in edge_list_ref
                if (edge_list_ref.count(e) > 1 or (e[1], e[0]) in edge_list_ref)
            ]
            duplicates = []
            for e in multiple_colored:
                if not (e in duplicates or (e[1], e[0]) in duplicates):
                    duplicates.append(e)
            raise ValueError(
                f"The color of the edges in the following list"
                f"was specified multiple times: {duplicates}."
            )
        return edge_color_array, edge_list_ref


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
