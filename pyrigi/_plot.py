import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sympy import Matrix, sympify
import distinctipy

from pyrigi.framework import Framework
from pyrigi.data_type import (
    Vertex,
    Edge,
    Point,
    InfFlex,
    Stress,
    Sequence,
    DirectedEdge,
)

def plot_inf_flex(  # noqa: C901
        framework: Framework,
        ax: Axes,
        inf_flex: Matrix | InfFlex,
        points: dict[Vertex, Point] = None,
        flex_width: float = 2.5,
        flex_length: float = 0.65,
        flex_color: (
                str | Sequence[Sequence[Edge]] | dict[str: Sequence[Edge]]
        ) = "limegreen",
        flex_style: str = "solid",
        flex_arrowsize: int = 20,
        projection_matrix: Matrix = None,
        **kwargs,
) -> None:
    """
    Add an infinitesimal flex based in the `points` as vectors to the axis `ax`.

    Parameters
    ----------
    ax:
    inf_flex:
        Optional parameter for plotting a given infinitesimal flex. It is
        important to use the same vertex order as the one
        from :meth:`.Graph.vertex_list`.
        Alternatively, an ``int`` can be specified to choose the 0,1,2,...-th
        nontrivial infinitesimal flex for plotting.
        Lastly, a ``dict[Vertex, Sequence[Number]]`` can be provided, which
        maps the vertex labels to vectors (i.e. a sequence of Numbers).
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
        inf_flex_basis = framework.nontrivial_inf_flexes()
        if inf_flex >= len(inf_flex_basis):
            raise IndexError(
                "The value of inf_flex exceeds "
                + "the dimension of the space "
                + "of infinitesimal flexes."
            )
        inf_flex_pointwise = framework._transform_inf_flex_to_pointwise(
            inf_flex_basis[inf_flex]
        )
    elif isinstance(inf_flex, Matrix):
        inf_flex_pointwise = framework._transform_inf_flex_to_pointwise(inf_flex)
    elif isinstance(inf_flex, dict) and all(
            isinstance(inf_flex[key], Sequence) for key in inf_flex.keys()
    ):
        inf_flex_pointwise = inf_flex
    else:
        raise TypeError(
            "inf_flex does not have the correct Type or the `int` is too large."
        )

    if not framework.is_dict_inf_flex(inf_flex_pointwise):
        raise ValueError("The provided `inf_flex` is not an infinitesimal flex.")
    if framework._dim == 1:
        inf_flex_pointwise = {
            v: [inf_flex_pointwise[v], 0] for v in inf_flex_pointwise.keys()
        }
    if projection_matrix is not None:
        inf_flex_pointwise = {
            v: np.dot(projection_matrix, np.array(flex))
            for v, flex in inf_flex_pointwise.items()
        }

    if points is None:
        points = framework.realization(as_points=True, numerical=True)
    elif not isinstance(points, dict):
        raise TypeError("Realization has the wrong type!")
    elif not all(
            [
                len(points[v]) == len(points[list(points.keys())[0]])
                and len(points[v]) in [2, 3]
                for v in framework._graph.nodes
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
        arrow_length = np.sqrt(x_canvas_width ** 2 + y_canvas_width ** 2) * flex_length
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
    elif framework._dim == 3:
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
                arrow_length_ratio=0.35,
            )
    else:
        raise ValueError(
            "The dimension of the infinitesimal flex needs to be between 1 and 3."
        )


def plot_stress(  # noqa: C901
        framework: Framework,
        ax: Axes,
        stress: Matrix | Stress,
        points: dict[Vertex, Point] = None,
        stress_color: str = "orangered",
        stress_fontsize: int = 10,
        stress_label_pos: float | dict[DirectedEdge, float] = 0.5,
        stress_rotate_labels: bool = True,
        stress_normalization: bool = False,
        connection_style: float | dict[DirectedEdge, float] = 0.5,
        curved_edges: bool = False,
        **kwargs,
) -> None:
    """
    Add an equilibrium stress based in the `edges` as numbers to the axis `ax`.

    Parameters
    ----------
    ax:
    stress:
        Optional parameter for plotting a given equilibrium stress. The standard
        input format is a ``Matrix`` that is the output of e.g. the method
        ``Framework.stresses``. Alternatively, an ``int`` can be specified
        to directly choose the 0,1,2,...-th equilibrium stress (according
        to the method ``Framework.stresses``) for plotting.
        For these input types, is important to use the same edge order as the one
        from :meth:`.Graph.edge_list`.
        If the edge order needs to be specified, a ``Dict[Edge, Number]``
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
        stresses = framework.stresses()
        if stress >= len(stresses):
            raise IndexError(
                "The value of `stress` exceeds "
                + "the dimension of the space "
                + "of equilibrium stresses."
            )
        stress_edgewise = framework._transform_stress_to_edgewise(stresses[stress])
    elif isinstance(stress, Matrix):
        stress_edgewise = framework._transform_stress_to_edgewise(stress)
    elif isinstance(stress, dict) and all(
            isinstance(stress[key], int | float | str) for key in stress.keys()
    ):
        stress_edgewise = stress
    else:
        raise TypeError("`stress` does not have the correct Type.")

    if not framework.is_dict_stress(stress_edgewise):
        raise ValueError("The provided `stress` is not an equilibrium stress.")

    if points is None:
        points = framework.realization(as_points=True, numerical=True)

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
        if not all([framework._graph.has_edge(*e) for e in stress_label_pos.keys()]):
            raise ValueError(
                "The `stress_label_pos` dictionary must contain the same "
                + "edges as the stress dictionary."
            )
        for edge in framework._graph.edge_list(as_tuples=True):
            stress_keys = [set(e) for e in stress_label_pos.keys()]
            if set(edge) not in stress_keys:
                stress_label_pos[edge] = 0.5
    elif isinstance(stress_label_pos, float):
        label_float = stress_label_pos
        stress_label_pos = {}
        for edge in framework._graph.edge_list(as_tuples=True):
            stress_label_pos[edge] = label_float
    else:
        raise TypeError(
            "`stress_label_pos` must be either a float or a dictionary."
        )
    if len(points[list(points.keys())[0]]) == 2:
        if curved_edges:
            newGraph = nx.MultiDiGraph()
            connection_style = resolve_connection_style(framework, connection_style)
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
            for edge in framework._graph.edges:
                nx.draw_networkx_edge_labels(
                    framework._graph,
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
            "The method `_plot_stress` is currently implemented only"
            + " for frameworks in 1, 2, and 3 dimensions."
        )

def plot_with_3D_realization(
    framework: Framework,
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
        Lastly, a ``Dict[Vertex, Sequence[Number]]`` can be provided, which
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

    edge_color_array, edge_list_ref = resolve_edge_colors(framework, edge_color)

    # Draw the vertices as points in the 3D enviroment
    x_nodes = [realization[node][0] for node in framework._graph.nodes]
    y_nodes = [realization[node][1] for node in framework._graph.nodes]
    z_nodes = [realization[node][2] for node in framework._graph.nodes]
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
    for node in framework._graph.nodes:
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

def resolve_connection_style(framework: Framework, connection_style: str) -> str:
    """
    Resolve the connection style for the visualization of the framework.

    Parameters
    ----------
    connection_style:
        The connection style for the visualization of the framework.
    """
    G = framework._graph
    if isinstance(connection_style, float):
        connection_style = {
            e: connection_style for e in G.edge_list(as_tuples=True)
        }
    elif isinstance(connection_style, list):
        if not G.number_of_edges() == len(connection_style):
            raise ValueError(
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
            raise ValueError(
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

def resolve_edge_colors(
    framework: Framework, edge_color: str | Sequence[Sequence[Edge]] | dict[str : Sequence[Edge]]
) -> tuple[list, list]:
    """
    Return the lists of colors and edges in the format for plotting.
    """
    G = framework._graph
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


def plot_with_2D_realization(
    framework: Framework,
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
    **kwargs,
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
        Lastly, a ``dict[Vertex, Sequence[Number]]`` can be provided, which
        maps the vertex labels to vectors (i.e. a sequence of coordinates).
    stress:
        Optional parameter for plotting an equilibrium stress. We expect
        it to have the format `Dict[Edge, Number]`.
    """
    edge_color_array, edge_list_ref = resolve_edge_colors(framework, edge_color)

    if not curved_edges:
        nx.draw(
            framework._graph,
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
        connection_style = resolve_connection_style(framework, connection_style)
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
