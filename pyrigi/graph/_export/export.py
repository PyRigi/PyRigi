"""
This module provides exports for graphs.
"""

import networkx as nx

from pyrigi._utils import _input_check as _input_check
from pyrigi.data_type import Edge, Point, Sequence, Vertex
from pyrigi.exception import NotSupportedValueError
from pyrigi.graph import _general as general
from pyrigi.graph._utils import _input_check as _graph_input_check


def to_tikz(
    graph: nx.Graph,
    layout_type: str = "spring",
    placement: dict[Vertex, Point] = None,
    vertex_style: str | dict[str, Sequence[Vertex]] = "gvertex",
    edge_style: str | dict[str, Sequence[Edge]] = "edge",
    label_style: str = "labelsty",
    figure_opts: str = "",
    vertex_in_labels: bool = False,
    vertex_out_labels: bool = False,
    default_styles: bool = True,
) -> str:
    r"""
    Create a TikZ code for the graph.

    For using it in ``LaTeX`` you need to use the ``tikz`` package.

    Parameters
    ----------
    placement:
        If ``placement`` is not specified,
        then it is generated depending on parameter ``layout``.
    layout_type:
        The possibilities are ``spring`` (default), ``circular``,
        ``random`` or ``planar``, see also :meth:`~Graph.layout`.
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
    --------
    >>> G = Graph([(0,1), (1,2), (2,3), (0,3)])
    >>> print(G.to_tikz()) # doctest: +SKIP
    \begin{tikzpicture}[gvertex/.style={fill=black,draw=white,circle,inner sep=0pt,minimum size=4pt},edge/.style={line width=1.5pt,black!60!white}]
        \node[gvertex] (0) at (-0.98794, -0.61705) {};
        \node[gvertex] (1) at (0.62772, -1.0) {};
        \node[gvertex] (2) at (0.98514, 0.62151) {};
        \node[gvertex] (3) at (-0.62492, 0.99554) {};
        \draw[edge] (0) to (1) (0) to (3) (1) to (2) (2) to (3);
    \end{tikzpicture}

    >>> print(G.to_tikz(layout_type = "circular")) # doctest: +NORMALIZE_WHITESPACE
    \begin{tikzpicture}[gvertex/.style={fill=black,draw=white,circle,inner sep=0pt,minimum size=4pt},edge/.style={line width=1.5pt,black!60!white}]
        \node[gvertex] (0) at (1.0, 0.0) {};
        \node[gvertex] (1) at (-0.0, 1.0) {};
        \node[gvertex] (2) at (-1.0, -0.0) {};
        \node[gvertex] (3) at (0.0, -1.0) {};
        \draw[edge] (0) to (1) (0) to (3) (1) to (2) (2) to (3);
    \end{tikzpicture}

    >>> print(G.to_tikz(placement = {0:[0, 0], 1:[1, 1], 2:[2, 2], 3:[3, 3]})) # doctest: +NORMALIZE_WHITESPACE
    \begin{tikzpicture}[gvertex/.style={fill=black,draw=white,circle,inner sep=0pt,minimum size=4pt},edge/.style={line width=1.5pt,black!60!white}]
        \node[gvertex] (0) at (0, 0) {};
        \node[gvertex] (1) at (1, 1) {};
        \node[gvertex] (2) at (2, 2) {};
        \node[gvertex] (3) at (3, 3) {};
        \draw[edge] (0) to (1) (0) to (3) (1) to (2) (2) to (3);
    \end{tikzpicture}

    >>> print(G.to_tikz(layout_type = "circular", vertex_out_labels = True)) # doctest: +NORMALIZE_WHITESPACE
    \begin{tikzpicture}[gvertex/.style={fill=black,draw=white,circle,inner sep=0pt,minimum size=4pt},edge/.style={line width=1.5pt,black!60!white},labelsty/.style={font=\scriptsize,black!70!white}]
        \node[gvertex,label={[labelsty]right:$0$}] (0) at (1.0, 0.0) {};
        \node[gvertex,label={[labelsty]right:$1$}] (1) at (-0.0, 1.0) {};
        \node[gvertex,label={[labelsty]right:$2$}] (2) at (-1.0, -0.0) {};
        \node[gvertex,label={[labelsty]right:$3$}] (3) at (0.0, -1.0) {};
        \draw[edge] (0) to (1) (0) to (3) (1) to (2) (2) to (3);
    \end{tikzpicture}

    >>> print(G.to_tikz(layout_type = "circular", vertex_in_labels = True)) # doctest: +NORMALIZE_WHITESPACE
    \begin{tikzpicture}[gvertex/.style={white,fill=black,draw=black,circle,inner sep=1pt,font=\scriptsize},edge/.style={line width=1.5pt,black!60!white}]
        \node[gvertex] (0) at (1.0, 0.0) {$0$};
        \node[gvertex] (1) at (-0.0, 1.0) {$1$};
        \node[gvertex] (2) at (-1.0, -0.0) {$2$};
        \node[gvertex] (3) at (0.0, -1.0) {$3$};
        \draw[edge] (0) to (1) (0) to (3) (1) to (2) (2) to (3);
    \end{tikzpicture}

    >>> print(G.to_tikz(
    ...     layout_type = "circular",
    ...     vertex_style = "myvertex",
    ...     edge_style = "myedge")
    ... ) # doctest: +NORMALIZE_WHITESPACE
    \begin{tikzpicture}[]
        \node[myvertex] (0) at (1.0, 0.0) {};
        \node[myvertex] (1) at (-0.0, 1.0) {};
        \node[myvertex] (2) at (-1.0, -0.0) {};
        \node[myvertex] (3) at (0.0, -1.0) {};
        \draw[myedge] (0) to (1) (0) to (3) (1) to (2) (2) to (3);
    \end{tikzpicture}

    >>> print(G.to_tikz(
    ...     layout_type="circular",
    ...     edge_style={"red edge": [[1, 2]], "green edge": [[2, 3], [0, 1]]},
    ...     vertex_style={"red vertex": [0], "blue vertex": [2, 3]})
    ... ) # doctest: +NORMALIZE_WHITESPACE
    \begin{tikzpicture}[]
        \node[red vertex] (0) at (1.0, 0.0) {};
        \node[blue vertex] (2) at (-1.0, -0.0) {};
        \node[blue vertex] (3) at (0.0, -1.0) {};
        \node[] (1) at (-0.0, 1.0) {};
        \draw[red edge] (1) to (2);
        \draw[green edge] (2) to (3) (0) to (1);
        \draw[] (0) to (3);
    \end{tikzpicture}
    """  # noqa: E501

    # strings for tikz styles
    if vertex_out_labels and default_styles:
        label_style_str = r"labelsty/.style={font=\scriptsize,black!70!white}"
    else:
        label_style_str = ""

    if vertex_style == "gvertex" and default_styles:
        if vertex_in_labels:
            vertex_style_str = (
                "gvertex/.style={white,fill=black,draw=black,circle,"
                r"inner sep=1pt,font=\scriptsize}"
            )
        else:
            vertex_style_str = (
                "gvertex/.style={fill=black,draw=white,circle,inner sep=0pt,"
                "minimum size=4pt}"
            )
    else:
        vertex_style_str = ""
    if edge_style == "edge" and default_styles:
        edge_style_str = "edge/.style={line width=1.5pt,black!60!white}"
    else:
        edge_style_str = ""

    figure_str = [figure_opts, vertex_style_str, edge_style_str, label_style_str]
    figure_str = [fs for fs in figure_str if fs != ""]
    figure_str = ",".join(figure_str)

    # tikz for edges
    edge_style_dict = {}
    if type(edge_style) is str:
        edge_style_dict[edge_style] = general.edge_list(graph)
    else:
        dict_edges = []
        for estyle, elist in edge_style.items():
            cdict_edges = [ee for ee in elist if graph.has_edge(*ee)]
            edge_style_dict[estyle] = cdict_edges
            dict_edges += cdict_edges
        remaining_edges = [
            ee
            for ee in general.edge_list(graph)
            if not ((list(ee) in dict_edges) or (list(ee)[::-1] in dict_edges))
        ]
        edge_style_dict[""] = remaining_edges

    edges_str = ""
    for estyle, elist in edge_style_dict.items():
        edges_str += (
            f"\t\\draw[{estyle}] "
            + " ".join([" to ".join([f"({v})" for v in e]) for e in elist])
            + ";\n"
        )

    # tikz for vertices
    if placement is None:
        placement = layout(graph, layout_type)

    vertex_style_dict = {}
    if type(vertex_style) is str:
        vertex_style_dict[vertex_style] = general.vertex_list(graph)
    else:
        dict_vertices = []
        for style, vertex_list in vertex_style.items():
            cdict_vertices = [v for v in vertex_list if (v in graph.nodes)]
            vertex_style_dict[style] = cdict_vertices
            dict_vertices += cdict_vertices
        remaining_vertices = [v for v in graph.nodes if (v not in dict_vertices)]
        vertex_style_dict[""] = remaining_vertices

    vertices_str = ""
    for style, vertex_list in vertex_style_dict.items():
        vertices_str += "".join(
            [
                "\t\\node["
                + style
                + (
                    ("," if vertex_style != "" else "")
                    + f"label={{[{label_style}]right:${v}$}}"
                    if vertex_out_labels
                    else ""
                )
                + f"] ({v}) at "
                + f"({round(placement[v][0], 5)}, {round(placement[v][1], 5)}) {{"
                + (f"${v}$" if vertex_in_labels else "")
                + "};\n"
                for v in vertex_list
            ]
        )
    return (
        "\\begin{tikzpicture}["
        + figure_str
        + "]\n"
        + vertices_str
        + edges_str
        + "\\end{tikzpicture}"
    )


def layout(graph: nx.Graph, layout_type: str = "spring") -> dict[Vertex, Point]:
    """
    Generate a placement of the vertices.

    This method is a wrapper for the functions
    :func:`~networkx.drawing.layout.spring_layout`,
    :func:`~networkx.drawing.layout.random_layout`,
    :func:`~networkx.drawing.layout.circular_layout`
    and :func:`~networkx.drawing.layout.planar_layout`.

    Parameters
    ----------
    layout_type:
        The supported layouts are ``circular``, ``planar``,
        ``random`` and ``spring`` (default).
    """
    if layout_type == "circular":
        return nx.drawing.layout.circular_layout(graph)
    elif layout_type == "planar":
        return nx.drawing.layout.planar_layout(graph)
    elif layout_type == "random":
        return nx.drawing.layout.random_layout(graph)
    elif layout_type == "spring":
        return nx.drawing.layout.spring_layout(graph)
    else:
        raise NotSupportedValueError(layout_type, "layout_type", layout)


def to_int(graph: nx.Graph, vertex_order: Sequence[Vertex] = None) -> int:
    """
    Return the integer representation of the graph.

    The graph integer representation is the integer whose binary
    expansion is given by the sequence obtained by concatenation
    of the rows of the upper triangle of the adjacency matrix,
    excluding the diagonal.

    Parameters
    ----------
    vertex_order:
        By listing vertices in the preferred order, the adjacency matrix
        is computed with the given order. If no vertex order is
        provided, :meth:`~.Graph.vertex_list()` is used.

    Examples
    --------
    >>> G = Graph([(0,1), (1,2)])
    >>> G.adjacency_matrix()
    Matrix([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]])
    >>> G.to_int()
    5

    Suggested Improvements
    ----------------------
    Implement taking canonical before computing the integer representation.
    """
    _input_check.greater_equal(graph.number_of_edges(), 1, "number of edges")
    if general.min_degree(graph) == 0:
        raise ValueError(
            "The integer representation only works "
            "for graphs without isolated vertices!"
        )
    _graph_input_check.no_loop(graph)

    adj_matrix = general.adjacency_matrix(graph, vertex_order)
    upper_diag = [
        str(b) for i, row in enumerate(adj_matrix.tolist()) for b in row[i + 1 :]
    ]
    return int("".join(upper_diag), 2)
