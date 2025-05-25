"""
This module provides exports for frameworks.
"""

from typing import Any

import pyrigi._utils._input_check as _input_check
from pyrigi.data_type import (
    Edge,
    Sequence,
    Vertex,
)
from pyrigi.framework.base import FrameworkBase
from pyrigi.graph._export import export as graph_export

__doctest_requires__ = {("generate_stl_bars",): ["trimesh", "manifold3d", "pathlib"]}


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


def generate_stl_bars(
    framework: FrameworkBase,
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

    edges_with_lengths = framework.edge_lengths()

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

        _generate_stl_bar(
            holes_distance=scaled_length,
            holes_diameter=holes_diameter,
            bar_width=width_of_bars,
            bar_height=height_of_bars,
            filename=f_name,
        )

    print(f"STL files for the bars have been generated in the folder `{output_dir}`.")
