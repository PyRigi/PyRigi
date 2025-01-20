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
    Number,
    DirectedEdge,
)
from pyrigi.plot_style import PlotStyle, PlotStyle2D, PlotStyle3D


def plot_inf_flex(  # noqa: C901
    framework: Framework,
    ax: Axes,
    inf_flex: int | Matrix | InfFlex,
    points: dict[Vertex, Point] = None,
    projection_matrix: Matrix = None,
    plot_style: PlotStyle = None,
    **kwargs,
) -> None:
    """
    Add an infinitesimal flex based in the `points` as vectors to the axis `ax`.
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

    if not framework.is_dict_inf_flex(inf_flex_pointwise):
        raise ValueError("The provided `inf_flex` is not an infinitesimal flex.")
    if framework.dim() == 1:
        inf_flex_pointwise = {
            v: [v_flex, 0] for v, v_flex in inf_flex_pointwise.items()
        }
    if projection_matrix is not None:
        # TODO use random projection matrix from plot_with_2D_realization
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

    if len(list(inf_flex_pointwise.values())[0]) == 2:
        x_canvas_width = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_canvas_width = ax.get_ylim()[1] - ax.get_ylim()[0]
        arrow_length = (
            np.sqrt(x_canvas_width**2 + y_canvas_width**2) * plot_style.flex_length
        )
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
        if not isinstance(plot_style.flex_color, str):
            raise TypeError("`flex_color` must be a `str` specifying a color.")
        nx.draw(
            H,
            pos=H_placement,
            ax=ax,
            arrows=True,
            arrowsize=plot_style.flex_arrowsize,
            node_size=0,
            node_color="white",
            width=plot_style.flex_width,
            edge_color=plot_style.flex_color,
            style=plot_style.flex_style,
            **kwargs,
        )
    elif framework.dim() == 3:
        for v in inf_flex_pointwise.keys():
            ax.quiver(
                points[v][0],
                points[v][1],
                points[v][2],
                inf_flex_pointwise[v][0],
                inf_flex_pointwise[v][1],
                inf_flex_pointwise[v][2],
                color=plot_style.flex_color,
                lw=plot_style.flex_width,
                linestyle=plot_style.flex_style,
                length=plot_style.flex_length,
                arrow_length_ratio=0.35,
            )
    else:
        raise ValueError(
            "The dimension of the infinitesimal flex needs to be between 1 and 3."
        )


def resolve_stress(
    framework: Framework,
    stress: Matrix | Stress,
    plot_style: PlotStyle,
    stress_label_positions: dict[Edge, float] = None,
) -> tuple[dict[Edge, Number], dict[DirectedEdge, float]]:
    """
    Add an equilibrium stress based in the `edges` as numbers to the axis `ax`.
    """
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

    if plot_style.stress_normalization:
        numerical_stress = {
            edge: float(sympify(w).evalf(10)) for edge, w in stress_edgewise.items()
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

    if stress_label_positions is None:
        stress_label_positions = {}
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
    points: dict[Vertex, Point] = None,
    connection_styles: dict[Edge, float] = None,
    stress_label_positions: dict[Edge, float] = None,
    **kwargs,
) -> None:
    """
    Add an equilibrium stress based in the `edges` as numbers to the axis `ax`.
    """
    stress_edgewise, stress_label_positions = resolve_stress(
        framework, stress, plot_style, stress_label_positions
    )

    if plot_style.curved_edges:
        new_graph = nx.MultiDiGraph()
        connection_style = resolve_connection_style(
            framework, plot_style.connection_style, connection_styles
        )
        for e, style in connection_style.items():
            new_graph.add_edge(e[0], e[1], weight=style)
        plt.box(False)  # Manually removes the frame of the plot
        for e in new_graph.edges(data=True):
            edge = tuple([e[0], e[1]])
            nx.draw_networkx_edge_labels(
                new_graph,
                ax=ax,
                pos=points,
                edge_labels={edge: stress_edgewise[edge]},
                font_color=plot_style.stress_color,
                font_size=plot_style.stress_fontsize,
                label_pos=stress_label_positions[edge],
                rotate=plot_style.stress_rotate_labels,
                connectionstyle=f"Arc3, rad = {e[2]['weight']}",
                **kwargs,
            )
    else:
        for edge in framework._graph.edges:
            nx.draw_networkx_edge_labels(
                framework._graph,
                ax=ax,
                pos=points,
                edge_labels={edge: stress_edgewise[edge]},
                font_color=plot_style.stress_color,
                font_size=plot_style.stress_fontsize,
                label_pos=stress_label_positions[edge],
                rotate=plot_style.stress_rotate_labels,
                **kwargs,
            )


def plot_stress3D(
    framework: Framework,
    ax: Axes,
    stress: Matrix | Stress,
    plot_style: PlotStyle,
    points: dict[Vertex, Point] = None,
    stress_label_positions: dict[Edge, float] = None,
    **kwargs,
) -> None:
    stress_edgewise, stress_label_positions = resolve_stress(
        framework, stress, plot_style, stress_label_positions
    )
    for edge, edge_stress in stress_label_positions.items():
        pos = [
            points[edge[0]][i] + edge_stress * (points[edge[1]][i] - points[edge[0]][i])
            for i in range(3)
        ]
        ax.text(
            pos[0],
            pos[1],
            pos[2],
            str(stress_edgewise[edge]),
            color=plot_style.stress_color,
            fontsize=plot_style.stress_fontsize,
            ha="center",
            va="center",
            **kwargs,
        )


def plot_with_3D_realization(
    framework: Framework,
    ax: Axes,
    realization: dict[Vertex, Point],
    plot_style: PlotStyle3D,
    edge_coloring: Sequence[Sequence[Edge]] | dict[str, Sequence[Edge]] = None,
) -> None:
    """
    Plot the framework with the given realization in the 3-space.
    """
    # Create a figure for the representation of the framework

    edge_color_array, edge_list_ref = resolve_edge_colors(
        framework, plot_style.edge_color, edge_coloring
    )

    # Draw the vertices as points in the 3D environment
    x_nodes = [realization[node][0] for node in framework._graph.nodes]
    y_nodes = [realization[node][1] for node in framework._graph.nodes]
    z_nodes = [realization[node][2] for node in framework._graph.nodes]
    ax.scatter(
        x_nodes,
        y_nodes,
        z_nodes,
        c=plot_style.vertex_color,
        s=plot_style.vertex_size,
        marker=plot_style.vertex_shape,
    )

    min_val = min(x_nodes + y_nodes + z_nodes) - plot_style.padding
    max_val = max(x_nodes + y_nodes + z_nodes) + plot_style.padding
    ax.set_zlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlim(min_val, max_val)

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
        for node in framework._graph.nodes:
            x, y, z, *others = realization[node]
            ax.text(
                x,
                y,
                z,
                str(node),
                color=plot_style.font_color,
                fontsize=plot_style.font_size,
                ha="center",
                va="center",
            )


def resolve_connection_style(
    framework: Framework,
    connection_style: float,
    connection_styles: Sequence[float] | dict[Edge, float] = None,
) -> dict[Edge, float]:
    """
    Resolve the connection style for the visualization of the framework.
    """
    G = framework._graph

    if connection_styles is None:
        connection_styles = {}

    if isinstance(connection_styles, list):
        if not G.number_of_edges() == len(connection_styles):
            raise ValueError(
                "The provided `connection_styles` don't have the correct length."
            )
        res = {
            e: style for e, style in zip(G.edge_list(as_tuples=True), connection_styles)
        }
    elif isinstance(connection_styles, dict):
        if (
            not all(
                [
                    isinstance(e, tuple) and len(e) == 2 and isinstance(v, float | int)
                    for e, v in connection_styles.items()
                ]
            )
            or not all(
                [
                    set(key) in [set([e[0], e[1]]) for e in G.edge_list()]
                    for key in connection_styles.keys()
                ]
            )
            or any(
                [set(key) for key in connection_styles.keys()].count(e) > 1
                for e in [set(key) for key in connection_styles.keys()]
            )
        ):
            raise ValueError(
                "The provided `connection_styles` contain different edges "
                + "than the underlying graph or has an incorrect format."
            )
        res = {e: style for e, style in connection_styles.items() if G.has_edge(*e)}
        for e in G.edges:
            if not (tuple(e) in res or tuple([e[1], e[0]]) in res):
                res[tuple(e)] = connection_style
    else:
        raise TypeError(
            "The provided `connection_styles` do not have the appropriate type."
        )
    return res


def resolve_edge_colors(
    framework: Framework,
    edge_color: str,
    edge_coloring: Sequence[Sequence[Edge]] | dict[str, Sequence[Edge]] = None,
) -> tuple[list, list]:
    """
    Return the lists of colors and edges in the format for plotting.
    """
    G = framework._graph
    edge_list = G.edge_list()
    edge_list_ref = []
    edge_color_array = []

    if edge_coloring is None:
        edge_coloring = {}

    if not isinstance(edge_color, str):
        raise TypeError("The provided `edge_color` is not a string. ")

    if isinstance(edge_coloring, list):
        edges_partition = edge_coloring
        colors = distinctipy.get_colors(
            len(edges_partition), colorblind_type="Deuteranomaly", pastel_factor=0.2
        )
        for i, part in enumerate(edges_partition):
            for e in part:
                if not G.has_edge(e[0], e[1]):
                    raise ValueError("The input includes a pair that is not an edge.")
                edge_color_array.append(colors[i])
                edge_list_ref.append(tuple(e))
    elif isinstance(edge_coloring, dict):
        color_edges_dict = edge_coloring
        for color, edges in color_edges_dict.items():
            for e in edges:
                if not G.has_edge(e[0], e[1]):
                    raise ValueError(
                        "The input includes an edge that is not part of the framework"
                    )
                edge_color_array.append(color)
                edge_list_ref.append(tuple(e))
    else:
        raise ValueError("The input edge_coloring has none of the supported formats.")
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


def plot_with_2D_realization(
    framework: Framework,
    ax: Axes,
    realization: dict[Vertex, Point],
    plot_style: PlotStyle2D,
    edge_coloring: Sequence[Sequence[Edge]] | dict[str, Sequence[Edge]] = None,
    connection_styles: Sequence[float] | dict[Edge, float] = None,
    **kwargs,
) -> None:
    """
    Plot the graph of the framework with the given realization in the plane.
    """
    edge_color_array, edge_list_ref = resolve_edge_colors(
        framework, plot_style.edge_color, edge_coloring
    )

    if not plot_style.curved_edges:
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
        connection_style = resolve_connection_style(
            framework, plot_style.connection_style, connection_styles
        )
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
                edge_color=edge_color_array,
                arrows=True,
                arrowstyle="-",
                edgelist=[(edge[0], edge[1])],
                connectionstyle=f"Arc3, rad = {edge[2]['weight']}",
            )
