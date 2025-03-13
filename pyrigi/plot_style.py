"""
This module provides classes for defining style of plots.
"""

from collections.abc import Sequence

import numpy as np


class PlotStyle(object):
    """
    Class for defining the plot style.

    For options specific to 2D or 3D plots,
    see :class:`.PlotStyle2D` and :class:`.PlotStyle3D`.
    For specifying different colors for each edge etc.,
    see :meth:`.Framework.plot2D` and :meth:`.Framework.plot3D`.

    Note that the parameters can be used directly in plot methods
    like :meth:`.Graph.plot` or :meth:`.Framework.plot2D`.

    Parameters
    ----------
    vertex_size:
        The size of the vertices.
    vertex_color:
        The color of the vertices given by name like ``"green"`` or  hex ``"#00ff00"``.
    vertex_labels:
        If ``True`` (default), vertex labels are displayed.
    vertex_shape:
        The shape of the vertices specified as :meth:`matplotlib.pyplot.scatter`
        marker, one of ``so^>v<dph8``.
    edge_width:
    edge_color:
        The color of all edges given as a string
        (name like ``"green"`` or  hex ``"#00ff00"``).
        For specifying a different color for each edge,
        see parameter ``edge_colors_custom`` in :meth:`.Framework.plot2D`.
    edge_style:
        Edge line style: ``-``/``solid``, ``--``/``dashed``,
        ``-.``/``dashdot`` or ``:``/``dotted``.
    flex_width:
        The width of the infinitesimal flex's arrow tail.
    flex_color:
        The color of the infinitesimal flex.
    flex_style:
        Line Style: ``-``/``solid``, ``--``/``dashed``,
        ``-.``/``dashdot`` or ``:``/``dotted``.
    flex_length:
        Length of the displayed flex relative to the total canvas
        diagonal in percent.
    flex_arrow_size:
        The size of the arrowhead's length and width.
    stress_color:
        The color of the font used to label the edges with stresses.
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

    Examples
    --------
    >>> from pyrigi import Graph
    >>> G = Graph([(0,1), (1,2), (2,3), (0,3)])
    >>> plot_style = PlotStyle(vertex_color="#FF0000", edge_color="black", vertex_size=50)
    >>> G.plot(plot_style)

    To change the plot style later, use the :meth:`.update` method:

    >>> plot_style.update(vertex_color="#00FF00")
    >>> G.plot(plot_style)

    Or assign to the attributes:

    >>> plot_style.vertex_color = "blue"
    >>> G.plot(plot_style)
    """

    def __init__(
        self,
        vertex_size: float | int = 300,
        vertex_color: str = "#ff8c00",
        vertex_labels: bool = True,
        vertex_shape: str = "o",
        edge_width: float | int = 2.5,
        edge_color: str = "black",
        edge_style: str = "solid",
        flex_width: float | int = 1.5,
        flex_length: float | int = 0.15,
        flex_color: str = "limegreen",
        flex_style: str = "solid",
        flex_arrow_size: int = 20,
        stress_color: str = "orangered",
        stress_fontsize: int = 10,
        stress_rotate_labels: bool = True,
        stress_normalization: bool = False,
        font_size: int = 12,
        font_color: str = "whitesmoke",
        canvas_width: float | int = 6.4,
        canvas_height: float | int = 4.8,
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
        self.flex_arrow_size = flex_arrow_size
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

    @property
    def vertex_size(self) -> float | int:
        return self._vertex_size

    @vertex_size.setter
    def vertex_size(self, value) -> None:
        if isinstance(value, float | int):
            self._vertex_size = value
        else:
            raise TypeError("vertex_size must be a float or integer.")

    @property
    def vertex_color(self) -> str:
        return self._vertex_color

    @vertex_color.setter
    def vertex_color(self, value) -> None:
        if isinstance(value, str):
            self._vertex_color = value
        else:
            raise TypeError("vertex_color must be a string.")

    @property
    def vertex_labels(self) -> bool:
        return self._vertex_labels

    @vertex_labels.setter
    def vertex_labels(self, value) -> None:
        if isinstance(value, bool):
            self._vertex_labels = value
        else:
            raise TypeError("vertex_labels must be a boolean.")

    @property
    def vertex_shape(self) -> str:
        return self._vertex_shape

    @vertex_shape.setter
    def vertex_shape(self, value) -> None:
        if isinstance(value, str):
            self._vertex_shape = value
        else:
            raise TypeError("vertex_shape must be a string.")

    @property
    def edge_width(self) -> float | int:
        return self._edge_width

    @edge_width.setter
    def edge_width(self, value) -> None:
        if isinstance(value, float | int):
            self._edge_width = value
        else:
            raise TypeError("edge_width must be a float or integer.")

    @property
    def edge_color(self) -> str:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, value) -> None:
        if isinstance(value, str):
            self._edge_color = value
        else:
            raise TypeError("edge_color must be a string.")

    @property
    def edge_style(self) -> str:
        return self._edge_style

    @edge_style.setter
    def edge_style(self, value) -> None:
        if isinstance(value, str):
            self._edge_style = value
        else:
            raise TypeError("edge_style must be a string.")

    @property
    def flex_width(self) -> float | int:
        return self._flex_width

    @flex_width.setter
    def flex_width(self, value) -> None:
        if isinstance(value, float | int):
            self._flex_width = value
        else:
            raise TypeError("flex_width must be a float or integer.")

    @property
    def flex_length(self) -> float | int:
        return self._flex_length

    @flex_length.setter
    def flex_length(self, value) -> None:
        if isinstance(value, float | int):
            self._flex_length = value
        else:
            raise TypeError("flex_length must be a float or integer.")

    @property
    def flex_color(self) -> str:
        return self._flex_color

    @flex_color.setter
    def flex_color(self, value) -> None:
        if isinstance(value, str):
            self._flex_color = value
        else:
            raise TypeError("flex_color must be a string.")

    @property
    def flex_style(self) -> str:
        return self._flex_style

    @flex_style.setter
    def flex_style(self, value) -> None:
        if isinstance(value, str):
            self._flex_style = value
        else:
            raise TypeError("flex_style must be a string.")

    @property
    def flex_arrow_size(self) -> int:
        return self._flex_arrow_size

    @flex_arrow_size.setter
    def flex_arrow_size(self, value) -> None:
        if isinstance(value, int):
            self._flex_arrow_size = value
        else:
            raise TypeError("flex_arrow_size must be an int.")

    @property
    def stress_color(self) -> str:
        return self._stress_color

    @stress_color.setter
    def stress_color(self, value) -> None:
        if isinstance(value, str):
            self._stress_color = value
        else:
            raise TypeError("stress_color must be a string.")

    @property
    def stress_fontsize(self) -> int:
        return self._stress_fontsize

    @stress_fontsize.setter
    def stress_fontsize(self, value) -> None:
        if isinstance(value, int):
            self._stress_fontsize = value
        else:
            raise TypeError("stress_fontsize must be an int.")

    @property
    def stress_rotate_labels(self) -> bool:
        return self._stress_rotate_labels

    @stress_rotate_labels.setter
    def stress_rotate_labels(self, value) -> None:
        if isinstance(value, bool):
            self._stress_rotate_labels = value
        else:
            raise TypeError("stress_rotate_labels must be a boolean.")

    @property
    def stress_normalization(self) -> bool:
        return self._stress_normalization

    @stress_normalization.setter
    def stress_normalization(self, value) -> None:
        if isinstance(value, bool):
            self._stress_normalization = value
        else:
            raise TypeError("stress_normalization must be a boolean.")

    @property
    def font_size(self) -> int:
        return self._font_size

    @font_size.setter
    def font_size(self, value) -> None:
        if isinstance(value, int):
            self._font_size = value
        else:
            raise TypeError("font_size must be an int.")

    @property
    def font_color(self) -> str:
        return self._font_color

    @font_color.setter
    def font_color(self, value) -> None:
        if isinstance(value, str):
            self._font_color = value
        else:
            raise TypeError("font_color must be a string.")

    @property
    def canvas_width(self) -> float | int:
        return self._canvas_width

    @canvas_width.setter
    def canvas_width(self, value) -> None:
        if isinstance(value, float | int):
            self._canvas_width = value
        else:
            raise TypeError("canvas_width must be a float or integer.")

    @property
    def canvas_height(self) -> float | int:
        return self._canvas_height

    @canvas_height.setter
    def canvas_height(self, value) -> None:
        if isinstance(value, float | int):
            self._canvas_height = value
        else:
            raise TypeError("canvas_height must be a float or integer.")

    @property
    def dpi(self) -> int:
        return self._dpi

    @dpi.setter
    def dpi(self, value) -> None:
        if isinstance(value, int):
            self._dpi = value
        else:
            raise TypeError("dpi must be an int.")


class PlotStyle2D(PlotStyle):
    """
    Class for defining the 2D plot style.

    Parameters
    ----------
    aspect_ratio:
        The ratio of y-unit to x-unit.
    edges_as_arcs:
        If ``True`` (default), the edges are displayed as arcs.
    arc_angle:
        Only if ``edges_as_arcs=True``:
        the pitch of the edge arcs in radians.
        For setting different values for individual edges,
        see ``arc_angles_dict`` in :meth:`.Framework.plot2D`.

    Examples
    --------
    >>> from pyrigi import Framework, PlotStyle2D
    >>> F = Framework.Complete([(0,1), (1,2), (0,2)])
    >>> plot_style_2d = PlotStyle2D(aspect_ratio=1, edges_as_arcs=True, arc_angle=np.pi/6)
    >>> F.plot2D(plot_style_2d)

    To update the plot style, you can assign to the attributes:

    >>> plot_style_2d.aspect_ratio = 0.75
    >>> plot_style_2d.edges_as_arcs = False
    >>> F.plot2D(plot_style_2d)

    Or use the :meth:`.update` method:

    >>> plot_style_2d.update(aspect_ratio=1.0, edges_as_arcs=True, arc_angle=np.pi/4)
    >>> F.plot2D(plot_style_2d)
    """

    def __init__(
        self,
        aspect_ratio: float | int = 1.0,
        edges_as_arcs: bool = False,
        arc_angle: float | int = np.pi / 6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.aspect_ratio = aspect_ratio
        self.edges_as_arcs = edges_as_arcs
        self.arc_angle = arc_angle

    @property
    def aspect_ratio(self) -> float | int:
        return self._aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, value) -> None:
        if isinstance(value, float | int):
            self._aspect_ratio = value
        else:
            raise TypeError("aspect_ratio must be a float or an int.")

    @property
    def edges_as_arcs(self) -> bool:
        return self._edges_as_arcs

    @edges_as_arcs.setter
    def edges_as_arcs(self, value) -> None:
        if isinstance(value, bool):
            self._edges_as_arcs = value
        else:
            raise TypeError("edges_as_arcs must be a boolean.")

    @property
    def arc_angle(self) -> float | int:
        return self._arc_angle

    @arc_angle.setter
    def arc_angle(self, value) -> None:
        if isinstance(value, float | int):
            self._arc_angle = value
        else:
            raise TypeError("arc_angle must be a float or integer.")

    @classmethod
    def from_plot_style(cls, plot_style: PlotStyle) -> PlotStyle:
        """
        Construct an instance from a given instance of :class:`.PlotStyle`.

        Parameters
        ----------
        plot_style:
            A :class:`.PlotStyle` instance to copy attributes from.
        """
        return cls(**{key[1:]: val for key, val in plot_style.__dict__.items()})


class PlotStyle3D(PlotStyle):
    """
    Class for defining the 3D plot style.

    Parameters
    ----------
    axis_scales:
        A triple indicating the scaling of the three axes.
    padding:
        Padding value for the plot.

    Examples
    --------
    >>> from pyrigi import Framework, PlotStyle3D
    >>> F = Framework.Complete([(0,1,2), (1,2,3), (2,3,0), (0,3,1)])
    >>> plot_style_3d = PlotStyle3D(padding=0.05, axis_scales=(2.0, 2.0, 2.0))
    >>> F.plot(plot_style_3d)

    To update the plot style, you can assign to the attributes:

    >>> plot_style_3d.padding = 0.10
    >>> plot_style_3d.axis_scales = (1.0, 2, 1.0)
    >>> F.plot(plot_style_3d)

    Or use the :meth:`.update` method:

    >>> plot_style_3d.update(padding=0.15, axis_scales=(1.0, 1.0, 3))
    """

    def __init__(
        self,
        padding: float | int = 0.01,
        axis_scales: Sequence[float | int] = (1.0, 1.0, 1.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.padding = padding
        self.axis_scales = axis_scales

    @property
    def padding(self) -> float | int:
        return self._padding

    @padding.setter
    def padding(self, value) -> None:
        if not isinstance(value, float | int):
            raise TypeError("Padding must be a float or integer.")
        self._padding = value

    @property
    def axis_scales(self) -> Sequence[float | int]:
        return self._axis_scales

    @axis_scales.setter
    def axis_scales(self, scales) -> None:
        if not isinstance(scales, tuple | list):
            raise TypeError(
                f"Axis_scales must be a tuple or a list, not {type(scales).__name__}."
            )
        if len(scales) != 3:
            raise ValueError(
                f"Axis_scales must contain exactly three elements, not {len(scales)}."
            )
        if not all(isinstance(scale, (int, float)) for scale in scales):
            raise TypeError("All elements of axis_scales must be of type int or float.")
        self._axis_scales = scales

    @classmethod
    def from_plot_style(cls, plot_style: PlotStyle) -> PlotStyle:
        """
        Construct an instance from a given instance of :class:`.PlotStyle`.

        Parameters
        ----------
        plot_style:
            A :class:`.PlotStyle` instance to copy attributes from.
        """
        return cls(**{key[1:]: val for key, val in plot_style.__dict__.items()})
