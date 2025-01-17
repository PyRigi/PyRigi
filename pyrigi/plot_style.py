import numpy as np


class PlotStyle(object):
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
