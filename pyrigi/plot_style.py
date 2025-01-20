import numpy as np


class PlotStyle(object):
    """
    Class for defining the plot style.

    Parameters
    ----------
    vertex_size:
        The size of the vertices.
    vertex_color:
        The color of the vertices. The color can be a string or rgb (or rgba)
        tuple of floats from 0-1.
    vertex_shape:
        The shape of the vertices specified as matplotlib.scatter
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
    """

    def __init__(
        self,
        vertex_size: int = 300,
        vertex_color: str = "#ff8c00",
        vertex_shape: str = "o",
        vertex_labels: bool = True,
        edge_width: float = 2.5,
        edge_color: str = "black",
        edge_style: str = "solid",
        flex_width: float = 1.5,
        flex_length: float = 0.15,
        flex_color: str = "limegreen",
        flex_style: str = "solid",
        flex_arrowsize: int = 20,
        stress_color: str = "orangered",
        stress_fontsize: int = 10,
        stress_label_pos: float = 0.5,
        stress_rotate_labels: bool = True,
        stress_normalization: bool = False,
        font_size: int = 12,
        font_color: str = "whitesmoke",
        canvas_width: float = 6.4,
        canvas_height: float = 4.8,
        aspect_ratio: float = 1.0,
        curved_edges: bool = False,
        connection_style: float = np.pi / 6,
        padding: float = 0.01,
        dpi: int = 200,
    ):
        self.vertex_size = vertex_size
        self.vertex_color = vertex_color
        self.vertex_shape = vertex_shape
        self.vertex_labels = vertex_labels
        self.edge_width = edge_width
        self.edge_color = edge_color
        self.edge_style = edge_style
        self.flex_width = flex_width
        self.flex_length = flex_length
        self.flex_color = flex_color
        self.flex_style = flex_style
        self.flex_arrowsize = flex_arrowsize
        self.stress_color = stress_color
        self.stress_fontsize = stress_fontsize
        self.stress_label_pos = stress_label_pos
        self.stress_rotate_labels = stress_rotate_labels
        self.stress_normalization = stress_normalization
        self.font_size = font_size
        self.font_color = font_color
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.aspect_ratio = aspect_ratio
        self.curved_edges = curved_edges
        self.connection_style = connection_style
        self.padding = padding
        self.dpi = dpi

    def update(self, **kwargs):
        """
        Update the plot style attributes from the keyword arguments.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"PlotStyle does not have the attribute {key}.")
