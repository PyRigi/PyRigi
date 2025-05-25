"""
This module provides exports for frameworks.
"""

from pyrigi.data_type import (
    Edge,
    Sequence,
    Vertex,
)
from pyrigi.framework.base import FrameworkBase
from pyrigi.graph._export import export as graph_export


def to_tikz(
    framework: FrameworkBase,
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
    if framework.dim != 2:
        raise ValueError("TikZ code is only generated for frameworks in dimension 2.")

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

    return graph_export.to_tikz(
        framework._graph,
        placement=framework.realization(),
        figure_opts=figure_str,
        vertex_style=vertex_style,
        edge_style=edge_style,
        label_style=label_style,
        vertex_in_labels=vertex_in_labels,
        vertex_out_labels=vertex_out_labels,
        default_styles=False,
    )
