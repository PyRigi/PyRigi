import numpy as np
import pytest
from pyrigi.plot_style import PlotStyle, PlotStyle2D, PlotStyle3D


def test_PlotStyle():
    # creation of PlotStyle
    plot_style = PlotStyle()
    assert isinstance(plot_style, PlotStyle)


def test_PlotStyle_arguments():
    plot_style = PlotStyle(
        vertex_size=300,
        vertex_color="#ff8c00",
        vertex_labels=True,
        vertex_shape="o",
        edge_width=2.5,
        edge_color="black",
        edge_style="solid",
        flex_width=1.5,
        flex_length=0.15,
        flex_color="limegreen",
        flex_style="solid",
        flex_arrow_size=20,
        stress_color="orangered",
        stress_fontsize=10,
        stress_rotate_labels=True,
        stress_normalization=False,
        font_size=12,
        font_color="whitesmoke",
        canvas_width=6.4,
        canvas_height=4.8,
        dpi=175,
    )

    assert plot_style.vertex_size == 300
    assert plot_style.vertex_color == "#ff8c00"
    assert plot_style.vertex_labels == True
    assert plot_style.vertex_shape == "o"
    assert plot_style.edge_width == 2.5
    assert plot_style.edge_color == "black"
    assert plot_style.edge_style == "solid"
    assert plot_style.flex_width == 1.5
    assert plot_style.flex_length == 0.15
    assert plot_style.flex_color == "limegreen"
    assert plot_style.flex_style == "solid"
    assert plot_style.flex_arrow_size == 20
    assert plot_style.stress_color == "orangered"
    assert plot_style.stress_fontsize == 10
    assert plot_style.stress_rotate_labels == True
    assert plot_style.stress_normalization == False
    assert plot_style.font_size == 12
    assert plot_style.font_color == "whitesmoke"
    assert plot_style.canvas_width == 6.4
    assert plot_style.canvas_height == 4.8
    assert plot_style.dpi == 175


def test_update():
    # proper attribute update
    plot_style = PlotStyle(vertex_color="blue", edge_color="red")
    assert plot_style.vertex_color == "blue"
    assert plot_style.edge_color == "red"

    plot_style.update(vertex_color="green", edge_color="yellow")

    assert plot_style.vertex_color == "green"
    assert plot_style.edge_color == "yellow"

    # introduction of non-existing attribute should raise ValueError
    with pytest.raises(ValueError):
        plot_style.update(non_existing_attribute="value")


def test_PlotStyle2D():
    # creation of PlotStyle2D
    plot_style_2d = PlotStyle2D()
    assert isinstance(plot_style_2d, PlotStyle2D)

    # check inheritance from PlotStyle
    assert issubclass(PlotStyle2D, PlotStyle)
    assert isinstance(plot_style_2d, PlotStyle)


def test_PlotStyle2D_arguments():
    # Create instance of PlotStyle2D including an inherited argument
    plot_style_2d = PlotStyle2D(
        aspect_ratio=1.5, edges_as_arcs=True, arc_angle=np.pi / 4, edge_color="blue"
    )

    # Check given attributes are correct
    assert plot_style_2d.aspect_ratio == 1.5
    assert plot_style_2d.edges_as_arcs == True
    assert plot_style_2d.arc_angle == np.pi / 4
    assert plot_style_2d.edge_color == "blue"

    # Test method from_plot_style
    plot_style = PlotStyle(vertex_color="red")
    converted_plot_style = PlotStyle2D.from_plot_style(plot_style)

    assert isinstance(converted_plot_style, PlotStyle2D)
    assert converted_plot_style.vertex_color == "red"


def test_PlotStyle3D():
    # creation of PlotStyle3D
    plot_style_3d = PlotStyle3D()
    assert isinstance(plot_style_3d, PlotStyle3D)

    # check inheritance from PlotStyle
    assert issubclass(PlotStyle3D, PlotStyle)
    assert isinstance(plot_style_3d, PlotStyle)

    # invalid axis_scales should raise ValueError
    with pytest.raises(ValueError):
        PlotStyle3D(axis_scales=("invalid", 1, "invalid"))


def test_PlotStyle3D_arguments():
    # Create instance of PlotStyle3D
    plot_style_3d = PlotStyle3D(padding=0.02, axis_scales=(2.0, 2.0, 2.0))

    # Check given attributes are correct
    assert plot_style_3d.padding == 0.02
    assert plot_style_3d.axis_scales == (2.0, 2.0, 2.0)

    # Test method from_plot_style
    plot_style = PlotStyle(vertex_color="red")
    converted_plot_style = PlotStyle3D.from_plot_style(plot_style)

    assert isinstance(converted_plot_style, PlotStyle3D)
    assert converted_plot_style.vertex_color == "red"
