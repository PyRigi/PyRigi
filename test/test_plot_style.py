import numpy as np
import pytest
from pyrigi.plot_style import PlotStyle, PlotStyle2D, PlotStyle3D


def test_PlotStyle():
    plot_style = PlotStyle()
    assert isinstance(plot_style, PlotStyle)

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
    assert plot_style.vertex_labels is True
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
    assert plot_style.stress_rotate_labels is True
    assert plot_style.stress_normalization is False
    assert plot_style.font_size == 12
    assert plot_style.font_color == "whitesmoke"
    assert plot_style.canvas_width == 6.4
    assert plot_style.canvas_height == 4.8
    assert plot_style.dpi == 175


def test_update():
    plot_style = PlotStyle(vertex_color="blue", edge_color="red")
    assert plot_style.vertex_color == "blue"
    assert plot_style.edge_color == "red"

    plot_style.update(vertex_color="green", edge_color="yellow")

    assert plot_style.vertex_color == "green"
    assert plot_style.edge_color == "yellow"

    # introduction of non-existing attribute should raise ValueError
    with pytest.raises(ValueError):
        plot_style.update(non_existing_attribute="value")


def test_PlotStyle_getters_setters():
    plot_style = PlotStyle(vertex_color="red")

    # Test vertex_size property
    plot_style.vertex_size = 400
    assert plot_style.vertex_size == 400
    with pytest.raises(TypeError):
        plot_style.vertex_size = "400"

    # Test vertex_color property
    assert plot_style.vertex_color == "red"
    plot_style.vertex_color = "blue"
    assert plot_style.vertex_color == "blue"
    with pytest.raises(TypeError):
        plot_style.vertex_color = 255

    # Test vertex_shape property
    plot_style.vertex_shape = "s"
    assert plot_style.vertex_shape == "s"
    with pytest.raises(TypeError):
        plot_style.vertex_shape = 22

    # Test edge_width property
    plot_style.edge_width = 3.5
    assert plot_style.edge_width == 3.5
    with pytest.raises(TypeError):
        plot_style.edge_width = "3.5"

    # Test edge_color property
    plot_style.edge_color = "blue"
    assert plot_style.edge_color == "blue"
    with pytest.raises(TypeError):
        plot_style.edge_color = 255

    # Test edge_style property
    plot_style.edge_style = "dashed"
    assert plot_style.edge_style == "dashed"
    with pytest.raises(TypeError):
        plot_style.edge_style = False

    # Test flex_width property
    plot_style.flex_width = 2.5
    assert plot_style.flex_width == 2.5
    with pytest.raises(TypeError):
        plot_style.flex_width = "2.5"

    # Test flex_length property
    plot_style.flex_length = 0.25
    assert plot_style.flex_length == 0.25
    with pytest.raises(TypeError):
        plot_style.flex_length = "0.25"

    # Test flex_color property
    plot_style.flex_color = "blue"
    assert plot_style.flex_color == "blue"
    with pytest.raises(TypeError):
        plot_style.flex_color = 255

    # Test flex_style property
    plot_style.flex_style = "dashed"
    assert plot_style.flex_style == "dashed"
    with pytest.raises(TypeError):
        plot_style.flex_style = False

    # Test flex_arrow_size property
    plot_style.flex_arrow_size = 30
    assert plot_style.flex_arrow_size == 30
    with pytest.raises(TypeError):
        plot_style.flex_arrow_size = "30"

    # Test stress_color property
    plot_style.stress_color = "blue"
    assert plot_style.stress_color == "blue"
    with pytest.raises(TypeError):
        plot_style.stress_color = 255

    # Test stress_fontsize property
    plot_style.stress_fontsize = 15
    assert plot_style.stress_fontsize == 15
    with pytest.raises(TypeError):
        plot_style.stress_fontsize = "15"

    # Test stress_rotate_labels property
    plot_style.stress_rotate_labels = False
    assert plot_style.stress_rotate_labels is False
    with pytest.raises(TypeError):
        plot_style.stress_rotate_labels = "False"

    # Test stress_normalization property
    plot_style.stress_normalization = True
    assert plot_style.stress_normalization is True
    with pytest.raises(TypeError):
        plot_style.stress_normalization = "True"

    # Test font_size property
    plot_style.font_size = 16
    assert plot_style.font_size == 16
    with pytest.raises(TypeError):
        plot_style.font_size = "16"

    # Test font_color property
    plot_style.font_color = "black"
    assert plot_style.font_color == "black"
    with pytest.raises(TypeError):
        plot_style.font_color = 255

    # Test canvas_width property
    plot_style.canvas_width = 8.4
    assert plot_style.canvas_width == 8.4
    with pytest.raises(TypeError):
        plot_style.canvas_width = "8.4"

    # Test canvas_height property
    plot_style.canvas_height = 5.8
    assert plot_style.canvas_height == 5.8
    with pytest.raises(TypeError):
        plot_style.canvas_height = "5.8"

    # Test dpi property
    plot_style.dpi = 200
    assert plot_style.dpi == 200
    with pytest.raises(TypeError):
        plot_style.dpi = "200"  #


def test_PlotStyle2D():
    plot_style_2d = PlotStyle2D()
    assert isinstance(plot_style_2d, PlotStyle2D)

    # check inheritance from PlotStyle
    assert issubclass(PlotStyle2D, PlotStyle)
    assert isinstance(plot_style_2d, PlotStyle)

    plot_style_2d = PlotStyle2D(
        aspect_ratio=1.5, edges_as_arcs=True, arc_angle=np.pi / 4, edge_color="blue"
    )

    assert plot_style_2d.aspect_ratio == 1.5
    assert plot_style_2d.edges_as_arcs is True
    assert plot_style_2d.arc_angle == np.pi / 4
    assert plot_style_2d.edge_color == "blue"

    plot_style = PlotStyle(vertex_color="red")
    converted_plot_style = PlotStyle2D.from_plot_style(plot_style)

    assert isinstance(converted_plot_style, PlotStyle2D)
    assert converted_plot_style.vertex_color == "red"


def test_PlotStyle2D_getters_setters():
    plot_style = PlotStyle2D(
        vertex_size=400,
        vertex_color="red",
        aspect_ratio=1.0,
        edges_as_arcs=True,
        arc_angle=3.14,
    )

    # Test aspect_ratio property
    plot_style.aspect_ratio = 2.0
    assert plot_style.aspect_ratio == 2.0
    with pytest.raises(TypeError):
        plot_style.aspect_ratio = "three point one four"

    # Test edges_as_arcs property
    plot_style.edges_as_arcs = False
    assert plot_style.edges_as_arcs is False
    with pytest.raises(TypeError):
        plot_style.edges_as_arcs = "False"

    # Test arc_angle property
    plot_style.arc_angle = 6.28
    assert plot_style.arc_angle - 6.28 < 1e-5
    with pytest.raises(TypeError):
        plot_style.arc_angle = "six point two eight"

    # Test base class getters and setters too
    plot_style.vertex_size = 500
    assert plot_style.vertex_size == 500
    with pytest.raises(TypeError):
        plot_style.vertex_size = "five hundred"


def test_PlotStyle3D():
    plot_style_3d = PlotStyle3D()
    assert isinstance(plot_style_3d, PlotStyle3D)

    # check inheritance from PlotStyle
    assert issubclass(PlotStyle3D, PlotStyle)
    assert isinstance(plot_style_3d, PlotStyle)

    # invalid axis_scales should raise ValueError
    with pytest.raises(TypeError):
        PlotStyle3D(axis_scales=("invalid", 1, "invalid"))

    # Create instance of PlotStyle3D
    plot_style_3d = PlotStyle3D(padding=0.02, axis_scales=(2.0, 2.0, 2.0))

    # Check given attributes are correct
    assert plot_style_3d.padding == 0.02
    assert plot_style_3d.axis_scales == (2.0, 2.0, 2.0)

    # Test method from_plot_style
    plot_style = PlotStyle(vertex_color="red")
    converted_plot_style = PlotStyle3D.from_plot_style(plot_style)
    converted_plot_style_twice = PlotStyle3D.from_plot_style(converted_plot_style)

    assert isinstance(converted_plot_style, PlotStyle3D)
    assert converted_plot_style.vertex_color == "red"
    assert converted_plot_style_twice.vertex_color == "red"


def test_PlotStyle3D_setters_and_getters():
    plot_style_3d = PlotStyle3D(padding=0.02, axis_scales=(2.0, 2.0, 2.0))

    # Check initial attributes are correct
    assert plot_style_3d.padding == 0.02
    assert plot_style_3d.axis_scales == (2.0, 2.0, 2.0)

    # Test setters and getters
    new_padding = 0.05
    new_axis_scales = (1.0, 1.0, 1.0)

    plot_style_3d.padding = new_padding
    plot_style_3d.axis_scales = new_axis_scales

    assert plot_style_3d.padding == new_padding
    assert plot_style_3d.axis_scales == new_axis_scales

    with pytest.raises(ValueError):
        plot_style_3d.axis_scales = (1.0, 1.0)
    with pytest.raises(TypeError):
        plot_style_3d.axis_scales = "1.0, 1.0, 1.0"
    with pytest.raises(TypeError):
        plot_style_3d.axis_scales = (1.0, "1.0", 1.0)
