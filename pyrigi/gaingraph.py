from pyrigi.data_type import Vertex, Sequence, DirectedEdge, Point
import pyrigi._input_check as _input_check
import networkx as nx
from pyrigi.plot_style import PlotStyle2D
from matplotlib import pyplot as plt
import distinctipy


class GainGraph(nx.MultiDiGraph):
    """
    Class representing a gain graph.

    Parameters
    ----------
    group_order:
        Order of the underlying group.
    vertex_partition:
        Partition of the vertex set.

    Examples
    --------
    >>> from pyrigi import GainGraph
    >>> G = GainGraph()
    >>> print(G)
    Graph with vertices [] and edges []

    This class inherits the class :class:`networkx.MultiDiGraph`.
    """

    # def __init__(gains: Sequence[Sequence[DirectedEdge | GroupElement]], group: Group):
    def __str__(self) -> str:
        """
        Return the string representation.
        """
        return (
            self.__class__.__name__
            + f" with vertices {list(self.nodes)} and gains "
            + f"{[{(e[0],e[1]):e[2]} for e in self.edges(data="gain", keys=False)]}"
        )

    def __repr__(self) -> str:
        """
        Return a representation.
        """
        return self.__str__()

    @classmethod
    def from_cyclic_group(
        cls,
        gains: (
            Sequence[Sequence[DirectedEdge | int]] | dict[DirectedEdge, Sequence[int]]
        ),
        group_order: int,
        vertex_partition: tuple[Sequence[Vertex], Sequence[Vertex]] = None,
    ):
        """
        Create a gain graph for a cyclic group of order ``group_order``
        """
        _input_check.integrality_and_range(group_order, "group_order", 2)

        if isinstance(gains, Sequence):
            _gains = {}
            for gain in gains:
                if (
                    not len(gain) == 2
                    or not isinstance(gain[0], tuple | list)
                    or len(gain[0]) != 2
                    or not isinstance(gain[1], int)
                ):
                    raise ValueError("The gains do not have the correct format.")
                if tuple(gain[0]) in _gains:
                    _gains[tuple(gain[0])] += [gain[0] % group_order]
                else:
                    _gains[tuple(gain[0])] = [gain[0] % group_order]
        elif isinstance(gains, dict):
            _gains = gains
            for e in _gains.keys():
                if (
                    not len(e) == 2
                    or not isinstance(_gains[e], tuple | list)
                    or not all([isinstance(gain, int) for gain in _gains[e]])
                ):
                    raise ValueError("The gains do not have the correct format.")
                _gains[e] = [gain % group_order for gain in _gains[e]]
        else:
            raise TypeError(
                f"The gains have the type {type(gains)}, "
                + "even though we expect a `list` or a `dict`."
            )

        for edge, gain_list in _gains.items():
            if len(gain_list) != len(set(gain_list)):
                raise ValueError("The gains are not unique on parallel edges.")
            if edge[0] == edge[1] and 0 in gain_list:
                raise ValueError("Loops cannot have the identity element as gain.")
        for edge1, gain_list1 in _gains.items():
            for edge2, gain_list_2 in _gains.items():
                if edge1[0] == edge2[1] and edge1[1] == edge2[0] and not edge1[0]==edge1[1]:
                    for gain in gain_list1:
                        if (-gain) % group_order in gain_list_2:
                            raise ValueError(
                                "Parallel edges cannot have inverse gains."
                            )

        G = GainGraph()
        for edge, gain in _gains.items():
            G.add_edges_from([(edge[0], edge[1], {"gain": val}) for val in gain])
        G.group_order = group_order

        if vertex_partition is not None:
            if len(vertex_partition) != 2:
                raise ValueError("`vertex_partition` does not have the correct length.")
            if any(
                [
                    len(vertex_partition[i]) != len(set(vertex_partition[i]))
                    for i in range(2)
                ]
            ):
                raise ValueError(
                    "There are duplicate vertices in the `vertex_partition`."
                )
            for v in G.nodes:
                if v not in vertex_partition[0] + vertex_partition[1]:
                    raise ValueError(
                        "The `vertex_partition` does not "
                        + "cover all of the multigraph's vertices."
                    )
            for v in vertex_partition[0] + vertex_partition[1]:
                if v not in G.nodes:
                    G.add_node(v)

            G.vertex_partition = vertex_partition
        else:
            G.vertex_partition = [list(G.nodes),[]]

        return G

    def plot(
        self,
        plot_style: PlotStyle2D = None,
        placement: dict[Vertex, Point] = None,
        arc_angles_dict: (
            Sequence[float] | dict[Sequence[DirectedEdge, int], float]
        ) = None,
        edge_colors_custom: Sequence[Sequence[DirectedEdge, int]] | dict[Sequence[DirectedEdge, int], str] = None,
        **kwargs,
    ):
        if plot_style is None:
            plot_style = PlotStyle2D(vertex_color="#4169E1")
        plot_style.update(**kwargs)

        if placement is None:
            placement = nx.shell_layout(self)
        if (
            set(placement.keys()) != set(self.nodes)
            or len(placement.keys()) != len(self.nodes)
            or any(
                [
                    len(p) != len(placement[list(placement.keys())[0]])
                    for p in placement.values()
                ]
            )
        ):
            raise TypeError("The placement does not have the correct format!")

        edge_color_array, edge_list_ref = self.resolve_edge_colors(
            plot_style.edge_color, edge_colors_custom
        )
        arc_angles = self.resolve_arc_angles(
            plot_style.arc_angle, arc_angles_dict
        )

        fig, ax = plt.subplots()
        ax.set_adjustable("datalim")
        fig.set_figwidth(plot_style.canvas_width)
        fig.set_figheight(plot_style.canvas_height)
        ax.set_aspect(plot_style.aspect_ratio)

        newGraph = nx.MultiDiGraph()
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
            placement,
            nodelist=self.vertex_partition[0],
            ax=ax,
            node_size=plot_style.vertex_size,
            node_color=plot_style.vertex_color,
            node_shape=plot_style.vertex_shape,
        )
        nx.draw_networkx_nodes(
            newGraph,
            placement,
            nodelist=self.vertex_partition[1],
            ax=ax,
            node_size=plot_style.vertex_size,
            node_color="red",
            node_shape=plot_style.vertex_shape,
        )
        nx.draw_networkx_labels(
            newGraph,
            placement,
            ax=ax,
            font_color=plot_style.font_color,
            font_size=plot_style.font_size,
        )
        for edge in newGraph.edges(data=True):
            nx.draw_networkx_edges(
                newGraph,
                placement,
                ax=ax,
                width=plot_style.edge_width,
                edge_color=edge_to_color[frozenset(edge[0:2])],
                arrows=True,
                arrowstyle="->",
                edgelist=[(edge[0], edge[1])],
                connectionstyle=f"Arc3, rad = {edge[2]['weight']}",
            )
        labels = {
            tuple((edge[0],edge[1],edge[2])): edge[3]
            for edge in self.edges(data="gain", keys=True)
        }
        for e in newGraph.edges(data=True,keys=True):
            nx.draw_networkx_edge_labels(
                newGraph,
                placement,
                {(e[0],e[1],e[2]): labels[(e[0],e[1],e[2])]},
                connectionstyle=f"Arc3, rad = {e[3]['weight']}" ,
                font_color=plot_style.stress_color,
                font_size=plot_style.font_size,
                ax=ax,
            )

    def resolve_edge_colors(
        self,
        edge_color: str,
        edge_colors_custom: Sequence[Sequence[DirectedEdge, int]] | dict[Sequence[DirectedEdge, int], str] = None,
    ) -> tuple[list, list]:
        """
        Return the lists of colors and edges in the format for plotting.
        """
        edge_list = self.edges()
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
                    if not self.has_edge(e[0], e[1]):
                        raise ValueError("The input includes a pair that is not an edge.")
                    edge_color_array.append(colors[i])
                    edge_list_ref.append(tuple(e))
        elif isinstance(edge_colors_custom, dict):
            color_edges_dict = edge_colors_custom
            for color, edges in color_edges_dict.items():
                for e in edges:
                    if not self.has_edge(e[0], e[1]):
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
            edge_color_array.append(edge_color)
            edge_list_ref.append(e)
        if len(edge_list_ref) > self.number_of_edges():
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
    
    def resolve_arc_angles(
        self,
        arc_angle: float,
        arc_angles_dict: Sequence[float] | dict[Sequence[DirectedEdge,int], float] = None,
    ) -> dict[Sequence[DirectedEdge,int], float]:
        """
        Resolve the arc angles style for the visualization of the framework.
        """
        if arc_angles_dict is None:
            arc_angles_dict = {}

        if isinstance(arc_angles_dict, list):
            if not self.number_of_edges() == len(arc_angles_dict):
                raise ValueError(
                    "The provided `arc_angles_dict` don't have the correct length."
                )
            res = {
            e: style for e, style in zip(self.edges(data="gain", keys=False), arc_angles_dict)
            }
        elif isinstance(arc_angles_dict, dict):
            if (
                not all(
                    [
                        isinstance(e, tuple) and len(e) == 3 and isinstance(v, float | int)
                        for e, v in arc_angles_dict.items()
                    ]
                )
                or not all(
                    [
                        set(key) in [set([e[0], e[1], e[2]]) for e in self.edges(data="gain", keys=False)]
                        for key in arc_angles_dict.keys()
                    ]
                )
                or any(
                    [set(key) for key in arc_angles_dict.keys()].count(e) > 2
                    for e in [set(key) for key in arc_angles_dict.keys()]
                )
            ):
                raise ValueError(
                    "The provided `arc_angles_dict` contain different edges "
                    + "than the underlying graph or has an incorrect format."
                )
            res = {e: style for e, style in arc_angles_dict.items() if self.has_edge(*e)}
            for e in self.edges(data="gain", keys=False):
                if not (tuple(e) in res or tuple([e[1], e[0],e[2]]) in res):
                    res[tuple(e)] = arc_angle
        else:
            raise TypeError(
                "The provided `arc_angles_dict` do not have the appropriate type."
            )
        return res