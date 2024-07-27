"""
Module for graph drawing on a canvas in jupyter notebook.
"""

from ipywidgets import (
    Output,
    RadioButtons,
    ColorPicker,
    HBox,
    VBox,
    IntSlider,
    Checkbox,
)
from ipycanvas import Canvas, hold_canvas
from IPython.display import display
from pyrigi.graph import Graph
import time


class GraphDrawer(object):
    """
    Class for the drawer. An instance of this class creates a canvas and takes mouse inputs in order to construct a graph.
    The vertices of the graph will be labelled using non-negative integers. Supported inputs
    - Edit Type -> Vertex
    1. Click on an empty place on canvas -> add a vertex at the clicked point.
    2. Click on an existing vertex -> select the clicked vertex for repositioning.
    3. Move mouse pointer when a vertex is selected -> reposition the vertex according to the pointer.
    4. Double click on an existing vertex -> remove the vertex.

    - Edit Type -> Edge
    1. Click on an existing vertex and move the pointer to another vertex while holding the mouse button down - add / remove and edge between the two vertices.

    Examples
    --------
    >>> from pyrigi.graphdrawer import GraphDrawer
    >>> Drawer = GraphDrawer()
    >>> Drawer.graph()
    Graph with vertices [] and edges []

    """

    def __init__(self) -> None:
        """
        Constructor of the class.

        TODO
        ---
        Add width/height parameters to canvas. Currently canvas has fixed width=600 and height=600.

        """
        self._edit_type = "Vertex"
        self._radius = 10
        self._ewidth = 2
        self._v_color = "blue"
        self._e_color = "black"
        self._last_click_time = -1
        self._selected_vertex = None
        self._next_vertex_label = 0
        self._show_vlabels = True

        self._G = Graph()  # the graph on canvas
        self._out = Output()  # can later be used to represent some properties

        # setting canvas properties
        self._canvas = Canvas(width=600, height=600)
        self._canvas.stroke_rect(0, 0, self._canvas.width, self._canvas.height)
        self._canvas.on_mouse_down(self._handle_mouse_down)
        self._canvas.on_mouse_up(self._handle_mouse_up)
        self._canvas.on_mouse_move(self._handle_mouse_move)
        self._canvas.on_mouse_out(self._handle_mouse_out)
        self._canvas.font = "12px serif"
        self._canvas.text_align = "center"
        self._canvas.text_baseline = "middle"

        ##### menu items #######
        # Edit Type radio buttons
        self._radio_buttons = RadioButtons(
            options=["Vertex", "Edge"],
            value="Vertex",
            description="Edit Type:",
            disabled=False,
        )
        self._radio_buttons.observe(self._on_edit_type_change)

        # color picker for the new vertices
        self._vcolor_picker = ColorPicker(
            concise=False, description="V-Color", value=self._v_color, disabled=False
        )

        self._vcolor_picker.observe(self._on_vcolor_change)

        # color picker for the new edges
        self._ecolor_picker = ColorPicker(
            concise=False, description="E-Color", value=self._e_color, disabled=False
        )
        self._ecolor_picker.observe(self._on_ecolor_change)

        # setting radius for vertices
        self._vradius_slider = IntSlider(
            value=self._radius,
            min=8,
            max=20,
            step=1,
            description="V-Size:",
            disabled=False,
            continuous_update=True,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        self._vradius_slider.observe(self._on_vradius_change)

        # setting line width for the edges
        self._ewidth_slider = IntSlider(
            value=self._ewidth,
            min=1,
            max=10,
            step=1,
            description="E-Size:",
            disabled=False,
            continuous_update=True,
            orientation="horizontal",
            readout=True,
            readour_format="d",
        )
        self._ewidth_slider.observe(self._on_ewidth_change)

        self._vlabel_checkbox = Checkbox(
            value=True, description="Show V-Labels", disabled=False, indent=False
        )
        self._vlabel_checkbox.observe(self._on_show_vlabel_change)

        # combining the menu and canvas
        right_box = VBox(
            [
                self._radio_buttons,
                self._vcolor_picker,
                self._ecolor_picker,
                self._vradius_slider,
                self._ewidth_slider,
                self._vlabel_checkbox,
            ]
        )
        box = HBox([self._canvas, right_box])

        # displaying the combined menu and canvas, and the output
        display(box)
        display(self._out)

    def _on_edit_type_change(self, change) -> None:
        """
        Handler of the Edit Type(Vertex / Edge) radio buttons.
        """

        if change["type"] == "change" and change["name"] == "value":
            self._edit_type = change["new"]

    def _on_vcolor_change(self, change) -> None:
        """
        Handler of the color picker for the new vertices.
        """

        if change["type"] == "change" and change["name"] == "value":
            self._v_color = change["new"]

    def _on_ecolor_change(self, change) -> None:
        """
        Handler of the color picker for the new edges.
        """

        if change["type"] == "change" and change["name"] == "value":
            self._e_color = change["new"]

    def _on_vradius_change(self, change) -> None:
        """
        Handler of the vertex size slider.
        """

        if change["type"] == "change" and change["name"] == "value":
            self._radius = change["new"]
            with hold_canvas():
                self._canvas.clear()
                self._redraw_graph()

    def _on_ewidth_change(self, change) -> None:
        """
        Handler of the edge width slider.
        """
        if change["type"] == "change" and change["name"] == "value":
            self._ewidth = change["new"]
            with hold_canvas():
                self._canvas.clear()
                self._redraw_graph()

    def _on_show_vlabel_change(self, change) -> None:
        """
        Handler of the vertex labels checkbox.
        """
        if change["type"] == "change" and change["name"] == "value":
            self._show_vlabels = change["new"]
            with hold_canvas():
                self._canvas.clear()
                self._redraw_graph()

    def _handle_mouse_down(self, x, y):
        """
        Handler for :meth:`ipycanvas.Canvas.on_mouse_down`.
        """
        for vertex in self._G.vertex_list():
            if (self._G.nodes[vertex]["pos"][0] - x) ** 2 + (
                self._G.nodes[vertex]["pos"][1] - y
            ) ** 2 < self._radius**2:
                if self._is_double_clicked() and self._edit_type == "Vertex":
                    self._G.remove_node(vertex)
                    break
                self._selected_vertex = vertex
                break
        else:
            if self._edit_type == "Vertex":
                self._G.add_node(
                    self._next_vertex_label, color=self._v_color, pos=[x, y]
                )
                # self.vertex_pos_dict[self.next_vertex_label] = (x, y)
                self._selected_vertex = self._next_vertex_label
                self._next_vertex_label += 1
                with hold_canvas():
                    self._canvas.clear()
                    self._redraw_graph()

    def _handle_mouse_move(self, x, y):
        """
        Handler for :meth:`ipycanvas.Canvas.on_mouse_move`.
        """
        vertex = self._selected_vertex
        if isinstance(vertex, int) and self._edit_type == "Edge":
            if (
                isinstance(self._collided_vertex(x, y), int)
                and self._collided_vertex(x, y) != vertex
            ):
                neighbour = self._collided_vertex(x, y)
                if sorted((vertex, neighbour)) in self._G.edge_list():
                    self._G.remove_edge(vertex, neighbour)
                else:
                    self._G.add_edge(vertex, neighbour, color=self._e_color)
                self._selected_vertex = neighbour
            with hold_canvas():
                self._canvas.clear()
                self._canvas.stroke_style = self._e_color
                self._canvas.line_width = self._ewidth
                self._canvas.stroke_line(
                    self._G.nodes[vertex]["pos"][0],
                    self._G.nodes[vertex]["pos"][1],
                    x,
                    y,
                )
                self._redraw_graph()

        elif isinstance(vertex, int) and self._edit_type == "Vertex":
            self._G.nodes[vertex]["pos"] = [x, y]
            with hold_canvas():
                self._canvas.clear()
                self._redraw_graph()

    def _handle_mouse_up(self, x, y):
        """
        Handler for :meth:`ipycanvas.Canvas.on_mouse_up`.
        """
        self._last_click_time = time.time()
        self._selected_vertex = None
        with hold_canvas():
            self._canvas.clear()
            self._redraw_graph()

    def _handle_mouse_out(self, x, y):
        """
        Handler for :meth:`ipycanvas.Canvas.on_mouse_out`.
        """
        self._selected_vertex = None
        with hold_canvas():
            self._canvas.clear()
            self._redraw_graph()

    def _is_double_clicked(self) -> bool:
        """
        A method for checking whether the click is a double click.
        """
        # ipcanvas package does not support double click events. This method is a simple way of adding
        # this support. If it does not seem good enough, ipyevents package can be used.

        if time.time() - self._last_click_time < 0.35:
            return True
        return False

    def _collided_vertex(self, x, y) -> int | None:
        """
        Return the vertex containing the point (x,y) on canvas.
        """
        for vertex in self._G.vertex_list():
            if (self._G.nodes[vertex]["pos"][0] - x) ** 2 + (
                self._G.nodes[vertex]["pos"][1] - y
            ) ** 2 < self._radius**2:
                return vertex
        return None

    def _redraw_graph(self) -> None:
        """
        Update the graph on canvas to illustrate the latest changes.
        """
        self._canvas.line_width = self._ewidth
        for u, v in self._G.edge_list():

            self._canvas.stroke_style = self._G[u][v]["color"]
            self._canvas.stroke_line(
                self._G.nodes[u]["pos"][0],
                self._G.nodes[u]["pos"][1],
                self._G.nodes[v]["pos"][0],
                self._G.nodes[v]["pos"][1],
            )

        for vertex in self._G.vertex_list():
            self._canvas.fill_style = self._G.nodes[vertex]["color"]
            [x, y] = self._G.nodes[vertex]["pos"]
            self._canvas.fill_circle(x, y, self._radius)
            if self._show_vlabels:
                self._canvas.fill_style = "white"
                self._canvas.fill_text(str(vertex), x, y)
        self._canvas.stroke_style = "black"
        self._canvas.line_width = 1
        self._canvas.stroke_rect(0, 0, self._canvas.width, self._canvas.height)

    def graph(self) -> Graph:
        """
        Return a copy of the current graph on the canvas.
        """
        H = Graph()
        H.add_nodes_from(self._G.nodes)
        H.add_edges_from(self._G.edges)
        return H
