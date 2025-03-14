"""
Module for plotting functionality.
"""

import networkx as nx
import numpy as np
import distinctipy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sympy import Matrix

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
from pyrigi.framework import Framework
from pyrigi.plot_style import PlotStyle, PlotStyle2D, PlotStyle3D
from pyrigi.misc import sympy_expr_to_float, is_zero_vector


def resolve_inf_flex(
    framework: Framework,
    inf_flex: int | Matrix | InfFlex,
    realization: dict[Vertex, Point] = None,
    projection_matrix: Matrix = None,
) -> dict[Vertex, Point]:
    """
    Resolve an infinitesimal flex from various datatypes.

    Parameters
    ----------
    framework:
    inf_flex:
        The infinitesimal flex to resolve.
    realization:
        A realization.
        If ``None``, the framework realization is used.
    projection_matrix:
        A matrix used for projection to a lower dimension.
    """
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
        raise TypeError("inf_flex does not have the correct Type.")

    if not framework.is_dict_inf_flex(inf_flex_pointwise, numerical=True):
        raise ValueError("The provided `inf_flex` is not an infinitesimal flex.")
    if framework.dim == 1:
        inf_flex_pointwise = {
            v: [v_flex, 0] for v, v_flex in inf_flex_pointwise.items()
        }
    if projection_matrix is not None:
        inf_flex_pointwise = {
            v: np.dot(projection_matrix, np.array(flex))
            for v, flex in inf_flex_pointwise.items()
        }

    if realization is None:
        realization = framework.realization(as_points=True, numerical=True)
    elif not isinstance(realization, dict):
        raise TypeError("Realization has the wrong type!")
    elif not all(
        [
            len(realization[v]) == len(realization[list(realization.keys())[0]])
            and len(realization[v]) in [2, 3]
            for v in framework._graph.nodes
        ]
    ):
        raise ValueError(
            "Not all values in the realization have the same"
            + "length and the dimension needs to be 2 or 3."
        )

    magnidutes = []
    for flex_key in inf_flex_pointwise.keys():
        if len(inf_flex_pointwise[flex_key]) != len(
            realization[list(realization.keys())[0]]
        ):
            raise ValueError(
                "The infinitesimal flex needs to be "
                + f"in dimension {len(realization[list(realization.keys())[0]])}."
            )
        inf_flex = [float(x) for x in inf_flex_pointwise[flex_key]]
        magnidutes.append(np.linalg.norm(inf_flex))

    # normalize the edge lengths by the Euclidean norm of the longest one
    flex_mag = max(magnidutes)
    for v, flex in inf_flex_pointwise.items():
        if not is_zero_vector(inf_flex):
            inf_flex_pointwise[v] = tuple(coord / flex_mag for coord in flex)
    # Delete the edges with zero length
    inf_flex_pointwise = {
        v: np.array(flex, dtype=float)
        for v, flex in inf_flex_pointwise.items()
        if not is_zero_vector(flex)
    }

    return inf_flex_pointwise


def plot_inf_flex2D(
    framework: Framework,
    ax: Axes,
    inf_flex: int | Matrix | InfFlex,
    realization: dict[Vertex, Point] = None,
    projection_matrix: Matrix = None,
    plot_style: PlotStyle2D = None,
) -> None:
    """
    Plot a 2D infinitesimal flex on the canvas.

    Parameters
    ----------
    framework:
    ax:
        The matplotlib axis on which the flex is drawn.
    inf_flex:
        The infinitesimal flex to plot.
    projection_matrix:
        A matrix used to project the infinitesimal flex to 2D.
    plot_style:
    """
    inf_flex_pointwise = resolve_inf_flex(
        framework, inf_flex, realization, projection_matrix
    )

    x_canvas_width = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_canvas_width = ax.get_ylim()[1] - ax.get_ylim()[0]
    arrow_length = (
        np.sqrt(x_canvas_width**2 + y_canvas_width**2) * plot_style.flex_length
    )
    H = nx.DiGraph([(v, str(v) + "_flex") for v in inf_flex_pointwise.keys()])
    H_realization = {
        str(v)
        + "_flex": np.array(
            [
                realization[v][0] + arrow_length * inf_flex_pointwise[v][0],
                realization[v][1] + arrow_length * inf_flex_pointwise[v][1],
            ],
            dtype=float,
        )
        for v in inf_flex_pointwise.keys()
    }
    H_realization.update(
        {v: np.array(realization[v], dtype=float) for v in inf_flex_pointwise.keys()}
    )
    if not isinstance(plot_style.flex_color, str):
        raise TypeError("`flex_color` must be a `str` specifying a color.")
    nx.draw(
        H,
        pos=H_realization,
        ax=ax,
        arrows=True,
        arrowsize=plot_style.flex_arrow_size,
        node_size=0,
        node_color="white",
        width=plot_style.flex_width,
        edge_color=plot_style.flex_color,
        style=plot_style.flex_style,
    )


def plot_inf_flex3D(
    framework: Framework,
    ax: Axes,
    inf_flex: int | Matrix | InfFlex,
    realization: dict[Vertex, Point] = None,
    projection_matrix: Matrix = None,
    plot_style: PlotStyle3D = None,
) -> None:
    """
    Plot a 3D infinitesimal flex on the canvas.

    Parameters
    ----------
    framework:
    ax:
        The matplotlib axis on which the flex is drawn.
    inf_flex:
        The infinitesimal flex to plot.
    projection_matrix:
        A matrix used to project the infinitesimal flex to 3D.
    plot_style:
    """
    inf_flex_pointwise = resolve_inf_flex(
        framework, inf_flex, realization, projection_matrix
    )
    x_canvas_width = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_canvas_width = ax.get_ylim()[1] - ax.get_ylim()[0]
    z_canvas_width = ax.get_zlim()[1] - ax.get_zlim()[0]
    arrow_length = (
        np.sqrt(x_canvas_width**2 + y_canvas_width**2 + z_canvas_width**2)
        * plot_style.flex_length
    )

    for v in inf_flex_pointwise.keys():
        ax.quiver(
            realization[v][0],
            realization[v][1],
            realization[v][2],
            inf_flex_pointwise[v][0],
            inf_flex_pointwise[v][1],
            inf_flex_pointwise[v][2],
            color=plot_style.flex_color,
            lw=plot_style.flex_width,
            linestyle=plot_style.flex_style,
            length=arrow_length,
            arrow_length_ratio=0.3,
        )


def resolve_stress(
    framework: Framework,
    stress: Matrix | Stress,
    plot_style: PlotStyle,
    stress_label_positions: dict[DirectedEdge, float] = None,
) -> tuple[dict[Edge, Number], dict[DirectedEdge, float]]:
    """
    Resolve an equilibrium stress from various datatypes and
    position of the labels on edges.

    The method returns a tuple with two dictionaries:
    one for the stress values and another for the label positions.

    Parameters
    ----------
    framework:
    stress:
        The equilibrium stress to resolve.
    plot_style:
    stress_label_positions:
        A dictionary mapping directed edges
        to values determining the position of stress labels.
    """

    if stress_label_positions is None:
        stress_label_positions = {}

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

    if not framework.is_dict_stress(stress_edgewise, numerical=True):
        raise ValueError("The provided `stress` is not an equilibrium stress.")

    if plot_style.stress_normalization:
        numerical_stress = {
            edge: sympy_expr_to_float(w) for edge, w in stress_edgewise.items()
        }
        _stress = {
            edge: round(w / np.linalg.norm(list(numerical_stress.values())), 2)
            for edge, w in numerical_stress.items()
        }
    else:
        _stress = stress_edgewise
    if not isinstance(stress_label_positions, dict):
        raise TypeError("`stress_label_positions` must be a dictionary.")

    if not all([framework._graph.has_edge(*e) for e in stress_label_positions.keys()]):
        raise ValueError(
            "The `stress_label_positions` dictionary must contain the same "
            + "edges as the stress dictionary."
        )

    for edge in framework._graph.edge_list(as_tuples=True):
        if edge in stress_label_positions:
            stress_label_positions[edge] = stress_label_positions[edge]
        elif edge[::-1] in stress_label_positions:
            stress_label_positions[edge] = 1 - stress_label_positions[edge[::-1]]
        else:
            stress_label_positions[edge] = 0.5

    return _stress, stress_label_positions


def plot_stress2D(
    framework: Framework,
    ax: Axes,
    stress: Matrix | Stress,
    plot_style: PlotStyle2D,
    realization: dict[Vertex, Point] = None,
    arc_angles_dict: dict[Edge, float] = None,
    stress_label_positions: dict[Edge, float] = None,
) -> None:
    """
    Plot a 2D equilibrium stress on the canvas.

    Parameters
    ----------
    framework:
    ax:
        The matplotlib axis on which the stress is drawn.
    stress:
        The equilibrium stress to draw.
    plot_style:
    realization:
        A dictionary mapping vertices to their points in the realization.
    arc_angles_dict:
        A dictionary specifying arc angles for curved edges.
    stress_label_positions:
        A dictionary mapping edges to label position floats.
    """
    stress_edgewise, stress_label_positions = resolve_stress(
        framework, stress, plot_style, stress_label_positions
    )
    if plot_style.edges_as_arcs:
        new_graph = nx.MultiDiGraph()
        arc_angles = resolve_arc_angles(
            framework, plot_style.arc_angle, arc_angles_dict
        )
        for e, style in arc_angles.items():
            new_graph.add_edge(e[0], e[1], weight=style)
        plt.box(False)  # Manually removes the frame of the plot
        for e in new_graph.edges(data=True):
            edge = tuple([e[0], e[1]])
            nx.draw_networkx_edge_labels(
                new_graph,
                ax=ax,
                pos=realization,
                edge_labels={
                    edge: (
                        stress_edgewise[edge]
                        if edge in stress_edgewise
                        else stress_edgewise[tuple(edge[::-1])]
                    )
                },
                font_color=plot_style.stress_color,
                font_size=plot_style.stress_fontsize,
                label_pos=(
                    stress_label_positions[edge]
                    if edge in stress_label_positions
                    else 1.0 - stress_label_positions[tuple(edge[::-1])]
                ),
                rotate=plot_style.stress_rotate_labels,
                connectionstyle=f"Arc3, rad = {e[2]['weight']}",
            )
    else:
        for e in framework._graph.edges:
            nx.draw_networkx_edge_labels(
                framework._graph,
                ax=ax,
                pos=realization,
                edge_labels={
                    e: (
                        stress_edgewise[e]
                        if e in stress_edgewise
                        else stress_edgewise[tuple(e[::-1])]
                    )
                },
                font_color=plot_style.stress_color,
                font_size=plot_style.stress_fontsize,
                label_pos=(
                    stress_label_positions[e]
                    if e in stress_label_positions
                    else 1 - stress_label_positions[tuple(e[::-1])]
                ),
                rotate=plot_style.stress_rotate_labels,
            )


def plot_stress3D(
    framework: Framework,
    ax: Axes,
    stress: Matrix | Stress,
    plot_style: PlotStyle,
    realization: dict[Vertex, Point] = None,
    stress_label_positions: dict[Edge, float] = None,
) -> None:
    """
    Plot a 3D equilibrium stress on the canvas.

    Parameters
    ----------
    framework:
    ax:
        The matplotlib axis on which the stress is drawn.
    stress:
        The equilibrium stress to draw.
    plot_style:
    realization:
        A dictionary mapping vertices to their points in the realization.
    stress_label_positions:
        A dictionary mapping edges to label position floats.
    """
    stress_edgewise, stress_label_positions = resolve_stress(
        framework, stress, plot_style, stress_label_positions
    )
    for edge, edge_stress in stress_label_positions.items():
        label_pos = [
            realization[edge[0]][i]
            + edge_stress * (realization[edge[1]][i] - realization[edge[0]][i])
            for i in range(3)
        ]
        ax.text(
            label_pos[0],
            label_pos[1],
            label_pos[2],
            str(
                stress_edgewise[edge]
                if edge in stress_edgewise
                else stress_edgewise[tuple(edge[::-1])]
            ),
            color=plot_style.stress_color,
            fontsize=plot_style.stress_fontsize,
            ha="center",
            va="center",
        )


def plot_with_2D_realization(
    framework: Framework,
    ax: Axes,
    realization: dict[Vertex, Point],
    plot_style: PlotStyle2D,
    edge_colors_custom: Sequence[Sequence[Edge]] | dict[str, Sequence[Edge]] = None,
    arc_angles_dict: Sequence[float] | dict[Edge, float] = None,
) -> None:
    """
    Plot the graph of the framework with the given realization in the plane.

    Parameters
    ----------
    framework:
    ax:
        The matplotlib axis on which the stress is drawn.
    realization:
        A dictionary mapping vertices to their points in the realization.
    plot_style:
    edge_colors_custom:
        It is possible to provide custom edge colors through this parameter.
        They can either be provided through a partition of edges or a
        dictionary with ``str`` color keywords that map to lists of edges.
    arc_angles_dict:
        A dictionary specifying arc angles for curved edges.
    """
    edge_color_array, edge_list_ref = resolve_edge_colors(
        framework, plot_style.edge_color, edge_colors_custom
    )

    if not plot_style.edges_as_arcs:
        nx.draw(
            framework._graph,
            pos=realization,
            ax=ax,
            node_size=plot_style.vertex_size,
            node_color=plot_style.vertex_color,
            node_shape=plot_style.vertex_shape,
            with_labels=plot_style.vertex_labels,
            width=plot_style.edge_width,
            edge_color=edge_color_array,
            font_color=plot_style.font_color,
            font_size=plot_style.font_size,
            edgelist=edge_list_ref,
            style=plot_style.edge_style,
        )
    else:
        newGraph = nx.MultiDiGraph()
        arc_angles = resolve_arc_angles(
            framework, plot_style.arc_angle, arc_angles_dict
        )
        for e, style in arc_angles.items():
            newGraph.add_edge(e[0], e[1], weight=style)
        edge_to_color = {
            frozenset(edge): color
            for edge, color in zip(edge_list_ref, edge_color_array)
        }
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
            node_size=plot_style.vertex_size,
            node_color=plot_style.vertex_color,
            node_shape=plot_style.vertex_shape,
        )
        nx.draw_networkx_labels(
            newGraph,
            realization,
            ax=ax,
            font_color=plot_style.font_color,
            font_size=plot_style.font_size,
        )
        for edge in newGraph.edges(data=True):
            nx.draw_networkx_edges(
                newGraph,
                realization,
                ax=ax,
                width=plot_style.edge_width,
                edge_color=edge_to_color[frozenset(edge[0:2])],
                arrows=True,
                arrowstyle="-",
                edgelist=[(edge[0], edge[1])],
                connectionstyle=f"Arc3, rad = {edge[2]['weight']}",
            )


def plot_with_3D_realization(
    framework: Framework,
    ax: Axes,
    realization: dict[Vertex, Point],
    plot_style: PlotStyle3D,
    edge_colors_custom: Sequence[Sequence[Edge]] | dict[str, Sequence[Edge]] = None,
) -> None:
    """
    Plot the framework with the given realization in the 3-space.

    Parameters
    ----------
    framework:
    ax:
        The matplotlib axis on which the stress is drawn.
    realization:
        A dictionary mapping vertices to their points in the realization.
    plot_style:
    edge_colors_custom:
        It is possible to provide custom edge colors through this parameter.
        They can either be provided through a partition of edges or a
        dictionary with ``str`` color keywords that map to lists of edges.
    """
    # Create a figure for the representation of the framework

    edge_color_array, edge_list_ref = resolve_edge_colors(
        framework, plot_style.edge_color, edge_colors_custom
    )

    # Center the realization
    x_coords, y_coords, z_coords = [
        [realization[u][i] for u in framework._graph.nodes] for i in range(3)
    ]
    min_coord = min(x_coords + y_coords + z_coords) - plot_style.padding
    max_coord = max(x_coords + y_coords + z_coords) + plot_style.padding
    aspect_ratio = plot_style.axis_scales
    ax.set_zlim(min_coord * aspect_ratio[0], max_coord * aspect_ratio[0])
    ax.set_ylim(min_coord * aspect_ratio[1], max_coord * aspect_ratio[1])
    ax.set_xlim(min_coord * aspect_ratio[2], max_coord * aspect_ratio[2])
    ax.scatter(
        x_coords,
        y_coords,
        z_coords,
        c=plot_style.vertex_color,
        s=plot_style.vertex_size,
        marker=plot_style.vertex_shape,
    )

    for i in range(len(edge_list_ref)):
        edge = edge_list_ref[i]
        x = [realization[edge[0]][0], realization[edge[1]][0]]
        y = [realization[edge[0]][1], realization[edge[1]][1]]
        z = [realization[edge[0]][2], realization[edge[1]][2]]
        ax.plot(
            x,
            y,
            z,
            c=edge_color_array[i],
            lw=plot_style.edge_width,
            linestyle=plot_style.edge_style,
        )
    # To show the name of the vertex
    if plot_style.vertex_labels:
        for u in framework._graph.nodes:
            x, y, z, *others = realization[u]
            ax.text(
                x,
                y,
                z,
                str(u),
                color=plot_style.font_color,
                fontsize=plot_style.font_size,
                ha="center",
                va="center",
            )


def resolve_arc_angles(
    framework: Framework,
    arc_angle: float,
    arc_angles_dict: Sequence[float] | dict[Edge, float] = None,
) -> dict[Edge, float]:
    """
    Resolve the arc angles style for the visualization of the framework.
    """
    G = framework._graph

    if arc_angles_dict is None:
        arc_angles_dict = {}

    if isinstance(arc_angles_dict, list):
        if not G.number_of_edges() == len(arc_angles_dict):
            raise ValueError(
                "The provided `arc_angles_dict` don't have the correct length."
            )
        res = {
            e: style for e, style in zip(G.edge_list(as_tuples=True), arc_angles_dict)
        }
    elif isinstance(arc_angles_dict, dict):
        if (
            not all(
                [
                    isinstance(e, tuple) and len(e) == 2 and isinstance(v, float | int)
                    for e, v in arc_angles_dict.items()
                ]
            )
            or not all(
                [
                    set(edge) in [set([e[0], e[1]]) for e in G.edge_list()]
                    for edge in arc_angles_dict.keys()
                ]
            )
            or any(
                [set(edge) for edge in arc_angles_dict.keys()].count(e) > 1
                for e in [set(edge) for edge in arc_angles_dict.keys()]
            )
        ):
            raise ValueError(
                "The provided `arc_angles_dict` contain different edges "
                + "than the underlying graph or has an incorrect format."
            )
        res = {e: style for e, style in arc_angles_dict.items() if G.has_edge(*e)}
        for e in G.edges:
            if not (tuple(e) in res or tuple([e[1], e[0]]) in res):
                res[tuple(e)] = arc_angle
    else:
        raise TypeError(
            "The provided `arc_angles_dict` do not have the appropriate type."
        )
    return res


def resolve_edge_colors(
    framework: Framework,
    edge_color: str,
    edge_colors_custom: Sequence[Sequence[Edge]] | dict[str, Sequence[Edge]] = None,
) -> tuple[list, list]:
    """
    Return the lists of colors and edges in the format for plotting.
    """
    G = framework._graph
    edge_list = G.edge_list()
    edge_list_ref = []
    edge_color_array = []

    if edge_colors_custom is None:
        edge_colors_custom = {}

    if not isinstance(edge_color, str):
        raise TypeError("The provided `edge_color` is not a string. ")

    if isinstance(edge_colors_custom, list):
        edges_partition = edge_colors_custom
        colors = distinctipy.get_colors(
            len(edges_partition), colorblind_type="Deuteranomaly", pastel_factor=0.2
        )
        for i, part in enumerate(edges_partition):
            for e in part:
                if not G.has_edge(e[0], e[1]):
                    raise ValueError("The input includes a pair that is not an edge.")
                edge_color_array.append(colors[i])
                edge_list_ref.append(tuple(e))
    elif isinstance(edge_colors_custom, dict):
        color_edges_dict = edge_colors_custom
        for color, edges in color_edges_dict.items():
            for e in edges:
                if not G.has_edge(e[0], e[1]):
                    raise ValueError(
                        "The input includes an edge that is not part of the framework"
                    )
                edge_color_array.append(color)
                edge_list_ref.append(tuple(e))
    else:
        raise ValueError(
            "The input edge_colors_custom has none of the supported formats."
        )
    for e in edge_list:
        if (e[0], e[1]) not in edge_list_ref and (e[1], e[0]) not in edge_list_ref:
            edge_color_array.append(edge_color)
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
