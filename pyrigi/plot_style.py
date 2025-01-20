import numpy as np


class PlotStyle(object):
    """
    Class for defining the plot style.

    For options specific to 2D or 3D plots,
    see :class:`.PlotStyle2D` and :class:`.PlotStyle3D`.

    Parameters
    ----------
    vertex_size:
        The size of the vertices.
    vertex_color:
        The color of the vertices. The color can be a string or rgb (or rgba)
        tuple of floats from 0-1.
    vertex_labels:
        If ``True`` (default), vertex labels are displayed.
    vertex_shape:
        The shape of the vertices specified as matplotlib.scatter
        marker, one of ``so^>v<dph8``.
    edge_width:
    edge_color:
        A color given as a string (name like ``"green"`` or  hex ``"#00ff00"``).
        For specifying a different color for each edge,
        see parameter ``edge_coloring`` in :meth:`.Framework.plot2D`.
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
    dpi:
        DPI (dots per inch) for the plot.
    """

    def __init__(
        self,
        vertex_size: int = 300,
        vertex_color: str = "#ff8c00",
        vertex_labels: bool = True,
        vertex_shape: str = "o",
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
        stress_rotate_labels: bool = True,
        stress_normalization: bool = False,
        font_size: int = 12,
        font_color: str = "whitesmoke",
        canvas_width: float = 6.4,
        canvas_height: float = 4.8,
        dpi: int = 175,
    ):
        self.vertex_size = vertex_size
        self.vertex_color = vertex_color
        self.vertex_labels = vertex_labels
        self.vertex_shape = vertex_shape
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
        self.stress_rotate_labels = stress_rotate_labels
        self.stress_normalization = stress_normalization
        self.font_size = font_size
        self.font_color = font_color
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
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


class PlotStyle2D(PlotStyle):
    """
    Class for defining the 2D plot style.

    Inherits from PlotStyle.

    Parameters
    ----------
    stress_label_pos:
        Position of the stress label along the edge. `float` numbers
        from the interval `[0,1]` are allowed. `0` represents the head
        of the edge, `0.5` the center and `1` the edge's tail.
    aspect_ratio:
        The ratio of y-unit to x-unit. By default 1.0.
    edges_as_arcs:
        If the edges are too close to each other, we can decide to
        visualize them as arcs.
    arc_angle:
        In case of curvilinear plotting (``edges_as_arcs=True``), the edges
        are displayed as arcs. With this parameter, we can set the
        pitch of these arcs in radians.
        For setting different values for individual edges,
        see ``arc_angles_dict` in :meth:`.Framework.plot2D`.
    """

    def __init__(
        self,
        stress_label_pos: float = 0.5,
        aspect_ratio: float = 1.0,
        edges_as_arcs: bool = False,
        arc_angle: float = np.pi / 6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stress_label_pos = stress_label_pos
        self.aspect_ratio = aspect_ratio
        self.edges_as_arcs = edges_as_arcs
        self.arc_angle = arc_angle

    @classmethod
    def from_plot_style(cls, plot_style: PlotStyle):
        """
        Construct an instance of ``PlotStyle2D`` from a given instance of ``PlotStyle``.

        Parameters
        ----------
        plot_style: PlotStyle
            The PlotStyle instance to copy attributes from.
        """
        return cls(**plot_style.__dict__)


class PlotStyle3D(PlotStyle):
    """
    Class for defining the 3D plot style.

    Parameters
    ----------

    axis_scales:
        Triple indicating the scaling of the three axes.
    padding:
        Padding value for the plot.
    """

    def __init__(
        self,
        padding: float = 0.01,
        axis_scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.padding = padding
        self.axis_scales = axis_scales

    @classmethod
    def from_plot_style(cls, plot_style: PlotStyle):
        """
        Construct an instance of ``PlotStyle3D`` from a given instance of ``PlotStyle``.

        Parameters
        ----------
        plot_style: PlotStyle
            The PlotStyle instance to copy attributes from.
        """
        return cls(**plot_style.__dict__)
