"""
Module for graph drawing on a canvas in jupyter notebook.
"""

from collections.abc import Sequence

import networkx as nx
import numpy as np
from IPython.display import display
from ipycanvas import MultiCanvas, hold_canvas
from ipywidgets import Output, ColorPicker, HBox, VBox, IntSlider, Checkbox, Label
from ipyevents import Event
from sympy import Rational

from pyrigi.data_type import Edge
from pyrigi.graph import Graph
from pyrigi.framework import Framework


class GraphDrawer(object):
    """
    Class for graph drawing.

    An instance of this class creates a canvas and
    takes mouse inputs in order to construct a graph. The vertices of the graph
    are labeled using non-negative integers.

    Supported inputs are:

    - Press mouse button on an empty place on canvas:
        Add a vertex at the pointer position.
    - Press mouse button on an existing vertex (or empty space) and
        release the mouse button on another vertex (or empty space):
        Add/remove an edge between the two vertices.
    - Drag a vertex with ``Ctrl`` being pressed:
        Reposition the vertex.
    - Double-click on an existing vertex:
        Remove the corresponding vertex.
    - Double-click on an existing edge:
        Remove the corresponding edge.

    Parameters
    ----------
    graph:
        A graph without loops which is to be drawn on canvas
        when the object is created. The non-integer labels are relabeled
    size:
        Width and height of the canvas, defaults to ``[600,600]``.
        The width and height are adjusted so that they are multiples of 100
        with minimum value 400 and maximum value 1000.
    layout_type:
        Layout type to visualise the ``graph``.
        For supported layout types see :meth:`.Graph.layout`.
        The default is ``spring``.
        If ``graph`` is ``None`` or empty, this parameter has no effect.
    place:
        The part of the canvas that is used for drawing ``graph``.
        Options are ``all`` (default, use all canvas), ``E`` (use the east part),
        ``W`` (use the west part), ``N`` (use the north part), ``S`` (use the south part),
        and also ``NE``, ``NW``, ``SE`` and ``SW``.
        If ``graph`` is ``None`` or empty, this parameter has no effect.

    Examples
    --------
    >>> from pyrigi import GraphDrawer
    >>> Drawer = GraphDrawer()
    HBox(children=(MultiCanvas(height=600, width=600)...
    >>> print(Drawer.graph())
    Graph with vertices [] and edges []
    """

    def __init__(
        self,
        graph: Graph = None,
        size: Sequence[int] = (600, 600),
        layout_type: str = "spring",
        place: str = "all",
    ) -> None:
        """
        Constructor of the class.
        """

        self._radius = 10
        self._edge_width = 2
        self._vertex_color = "blue"  # default color for vertices
        self._edge_color = "black"  # default color for edges

        self._selected_vertex = None  # this determines what vertex to update on canvas
        self._vertex_labels = True
        self._mouse_down = False
        self._vertex_move_on = False
        self._grid_size = 50

        self._graph = Graph()  # the graph on canvas
        self._out = Output()  # can later be used to represent some properties

        # setting multicanvas properties

        if not isinstance(size, Sequence) or not len(size) == 2:
            raise ValueError("The parameter `size` must be a list of two integers")
        # arrange width and height of the canvas so that they are in [300,1000]
        for i in range(2):
            if size[i] < 400:
                size[i] = 400
            if size[i] > 1000:
                size[i] = 1000
        # convert members of size to closest multiple of 100
        size = [int(round(x / 100) * 100) for x in size]

        self._mcanvas = MultiCanvas(5, width=size[0], height=size[1])
        self._mcanvas[0].stroke_rect(0, 0, self._mcanvas.width, self._mcanvas.height)
        self._mcanvas[2].font = "12px serif"
        self._mcanvas[2].text_align = "center"
        self._mcanvas[2].text_baseline = "middle"
        self._mcanvas[3].font = "12px serif"
        self._mcanvas[3].text_align = "center"
        self._mcanvas[3].text_baseline = "middle"
        self._mcanvas.on_mouse_down(self._handle_mouse_down)
        self._mcanvas.on_mouse_up(self._handle_mouse_up)
        self._mcanvas.on_mouse_move(self._handle_mouse_move)
        self._mcanvas.on_mouse_out(self._handle_mouse_out)

        # IpyEvents Part
        self._events = Event()
        self._events.source = self._mcanvas
        self._events.watched_events = ["keydown", "keyup", "dblclick"]
        self._events.on_dom_event(self._handle_event)

        # color picker for the new vertices
        self._vertex_color_picker = ColorPicker(
            concise=False,
            description="V-Color",
            value=self._vertex_color,
            disabled=False,
        )

        self._vertex_color_picker.observe(self._on_vertex_color_change)

        # color picker for the new edges
        self._edge_color_picker = ColorPicker(
            concise=False, description="E-Color", value=self._edge_color, disabled=False
        )
        self._edge_color_picker.observe(self._on_edge_color_change)

        # setting radius for vertices
        self._vertex_radius_slider = IntSlider(
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
        self._vertex_radius_slider.observe(self._on_vertex_radius_change)

        # setting line width for the edges
        self._edge_width_slider = IntSlider(
            value=self._edge_width,
            min=1,
            max=10,
            step=1,
            description="E-Size:",
            disabled=False,
            continuous_update=True,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        self._edge_width_slider.observe(self._on_edge_width_change)

        # setting checkbox for showing vertex labels
        self._vertex_label_checkbox = Checkbox(
            value=True, description="Show V-Labels", disabled=False, indent=False
        )
        self._vertex_label_checkbox.observe(self._on_show_vertex_label_change)

        self._grid_checkbox = Checkbox(
            value=False, description="Show Grid", disabled=False, indent=False
        )
        self._grid_checkbox.observe(self._on_grid_checkbox_change)

        self._grid_snap_checkbox = Checkbox(
            value=False,
            description="Grid Snapping",
            disabled=True,
            indent=False,
        )

        self._grid_size_slider = IntSlider(
            value=self._grid_size,
            min=10,
            max=50,
            step=5,
            description="Grid Size:",
            disabled=True,
            continuous_update=True,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        self._grid_size_slider.observe(self._on_grid_size_change)

        # combining the menu and canvas
        right_box = VBox(
            [
                self._vertex_color_picker,
                self._edge_color_picker,
                self._vertex_radius_slider,
                self._edge_width_slider,
                self._vertex_label_checkbox,
                self._grid_checkbox,
                self._grid_snap_checkbox,
                self._grid_size_slider,
            ]
        )
        # instructions
        instruction_dict = {
            "- Add vertex:": "Mouse press",
            "- Add edge:": "Drag between endpoints",
            "- Remove Edge-1:": "Double click",
            "- Remove Edge-2:": "Drag between endpoints",
            "- Remove Vertex:": "Double click",
            "- Move vertex:": "Hold ctrl and drag",
        }
        for instruction in instruction_dict:
            label_action = Label(value=instruction)
            label_description = Label(value=instruction_dict[instruction])
            box = HBox([label_action, label_description])
            right_box.children += (box,)

        box = HBox([self._mcanvas, right_box])

        if isinstance(graph, Graph) and graph.number_of_nodes() > 0:
            self._set_graph(graph, layout_type, place)
            with hold_canvas():
                self._mcanvas[1].clear()
                self._redraw_graph()

        # displaying the combined menu and canvas, and the output
        display(box)
        display(self._out)

    def _handle_event(self, event) -> None:
        """
        Handle keyboard events and double-click events using ``ipyevents``.
        """
        if event["event"] == "keydown":
            self._vertex_move_on = event["ctrlKey"]
        elif event["event"] == "keyup":
            self._vertex_move_on = event["ctrlKey"]
        elif event["event"] == "dblclick":
            x = (
                (event["clientX"] - event["boundingRectLeft"])
                / (event["boundingRectRight"] - event["boundingRectLeft"])
                * self._mcanvas.width
            )
            y = (
                (event["clientY"] - event["boundingRectTop"])
                / (event["boundingRectBottom"] - event["boundingRectTop"])
                * self._mcanvas.height
            )
            self._handle_dblclick(x, y)

    def _assign_pos(self, x, y, place) -> list[int]:
        """
        Convert layout positions which are between -1 and 1
        to canvas positions according to the chosen place by scaling.
        """
        width = self._mcanvas.width
        height = self._mcanvas.height
        r = self._radius

        # -3 is used below so that the vertices do not touch the edges of the multicanvas
        if place == "all":
            return [
                int(width / 2 + x * (width / 2 - r - 3)),
                int(height / 2 + y * (height / 2 - r - 3)),
            ]
        if place == "N":
            return [
                int(width / 2 + x * (width / 2 - r - 3)),
                int(height / 4 + y * (height / 4 - r - 3)),
            ]
        if place == "S":
            return [
                int(width / 2 + x * (width / 2 - r - 3)),
                int(height * 3 / 4 + y * (height / 4 - r - 3)),
            ]
        if place == "W":
            return [
                int(width / 4 + x * (width / 4 - r - 3)),
                int(height / 2 + y * (height / 2 - r - 3)),
            ]
        if place == "E":
            return [
                int(width * 3 / 4 + x * (width / 4 - r - 3)),
                int(height / 2 + y * (height / 2 - r - 3)),
            ]
        if place == "NE":
            return [
                int(width * 3 / 4 + x * (width / 4 - r - 3)),
                int(height / 4 + y * (height / 4 - r - 3)),
            ]
        if place == "NW":
            return [
                int(width / 4 + x * (width / 4 - r - 3)),
                int(height / 4 + y * (height / 4 - r - 3)),
            ]
        if place == "SE":
            return [
                int(width * 3 / 4 + x * (width / 4 - r - 3)),
                int(height * 3 / 4 + y * (height / 4 - r - 3)),
            ]
        if place == "SW":
            return [
                int(width / 4 + x * (width / 4 - r - 3)),
                int(height * 3 / 4 + y * (height / 4 - r - 3)),
            ]

    def _set_graph(self, graph: Graph, layout_type, place) -> None:
        """
        Set up a ``graph`` with specified layout and place it on the canvas.

        See :obj:`GraphDrawer` for the parameters.
        """
        vertex_map = {}
        for vertex in graph:
            if not isinstance(vertex, int) or vertex < 0:
                for i in range(graph.number_of_nodes()):
                    if not graph.has_node(i) and i not in vertex_map.values():
                        vertex_map[vertex] = i
                        break
        graph = nx.relabel_nodes(graph, vertex_map, copy=True)
        placement = graph.layout(layout_type)

        # random layout assigns coordinates between 0 and 1.
        # adjust the coordinates to between -1 and 1 as other layouts
        if layout_type == "random":
            for vertex in placement:
                placement[vertex] = [2 * x - 1 for x in placement[vertex]]

        # add vertices to the graph of the graphdrawer by scaling the coordinates
        # from [-1,1] to [self._mcanvas.width, self._mcanvas.height]
        for vertex in graph.nodes:
            px, py = placement[vertex]
            self._graph.add_node(
                vertex, color=self._vertex_color, pos=self._assign_pos(px, py, place)
            )
        for edge in graph.edges:
            self._graph.add_edge(edge[0], edge[1], color=self._edge_color)

        if len(vertex_map) != 0:
            with self._out:
                print("relabeled vertices:", vertex_map)

    def _on_grid_checkbox_change(self, change: dict[str, str]) -> None:
        """
        Handle the grid checkbox.
        """
        if change["type"] == "change" and change["name"] == "value":
            self._update_background(change["new"])
            self._grid_snap_checkbox.disabled = change["old"]
            self._grid_size_slider.disabled = change["old"]
            if change["new"] is False:
                self._grid_snap_checkbox.value = False

    def _on_grid_size_change(self, change: dict[str, str]) -> None:
        """
        Handle the grid size slider.
        """
        if change["type"] == "change" and change["name"] == "value":
            self._grid_size = change["new"]
            self._update_background(grid_on=self._grid_checkbox.value)

    def _on_vertex_color_change(self, change: dict[str, str]) -> None:
        """
        Handle the color picker for the new vertices.
        """
        if change["type"] == "change" and change["name"] == "value":
            self._vertex_color = change["new"]

    def _on_edge_color_change(self, change: dict[str, str]) -> None:
        """
        Handle the color picker for the new edges.
        """
        if change["type"] == "change" and change["name"] == "value":
            self._edge_color = change["new"]

    def _on_vertex_radius_change(self, change: dict[str, str]) -> None:
        """
        Handle the vertex size slider.
        """
        if change["type"] == "change" and change["name"] == "value":
            self._radius = change["new"]
            with hold_canvas():
                self._mcanvas[2].clear()
                self._redraw_graph()

    def _on_edge_width_change(self, change: dict[str, str]) -> None:
        """
        Handle the edge width slider.
        """
        if change["type"] == "change" and change["name"] == "value":
            self._edge_width = change["new"]
            with hold_canvas():
                self._mcanvas[2].clear()
                self._redraw_graph()

    def _on_show_vertex_label_change(self, change: dict[str, str]) -> None:
        """
        Handle the vertex labels checkbox.
        """
        if change["type"] == "change" and change["name"] == "value":
            self._vertex_labels = change["new"]
            with hold_canvas():
                self._mcanvas[2].clear()
                self._redraw_graph()

    def _update_background(self, grid_on: bool):
        """
        Update the background of a canvas.

        A grid can be added, if desired.

        Parameters
        ----------
        grid_on:
            Boolean determining whether a grid should be added to the canvas.
        """
        self._mcanvas[0].clear()
        self._mcanvas[0].line_width = 1
        self._mcanvas[0].stroke_style = "black"
        self._mcanvas[0].stroke_rect(0, 0, self._mcanvas.width, self._mcanvas.height)
        self._mcanvas[0].stroke_style = "grey"
        if not grid_on:
            return
        size = self._grid_size
        # self._mcanvas[0].set_line_dash([2,2])

        # add lines from center to sides so that center
        # of the canvas is always at a corner
        for i in range(0, int(self._mcanvas.width / 2), size):
            self._mcanvas[0].stroke_line(
                self._mcanvas.width / 2 + i,
                0,
                self._mcanvas.width / 2 + i,
                self._mcanvas.height,
            )
            if i != 0:
                self._mcanvas[0].stroke_line(
                    self._mcanvas.width / 2 - i,
                    0,
                    self._mcanvas.width / 2 - i,
                    self._mcanvas.height,
                )

        for i in range(0, int(self._mcanvas.height / 2), size):
            self._mcanvas[0].stroke_line(
                0,
                self._mcanvas.height / 2 + i,
                self._mcanvas.width,
                self._mcanvas.height / 2 + i,
            )
            if i != 0:
                self._mcanvas[0].stroke_line(
                    0,
                    self._mcanvas.height / 2 - i,
                    self._mcanvas.width,
                    self._mcanvas.height / 2 - i,
                )
        # add a red dot at the origin
        self._mcanvas[0].fill_style = "red"
        self._mcanvas[0].fill_circle(
            self._mcanvas.width / 2, self._mcanvas.height / 2, 2
        )

    def _handle_mouse_down(self, x, y) -> None:
        """
        Handle :meth:`ipycanvas.MultiCanvas.on_mouse_down`.

        It determines what to do when mouse button is pressed.
        """
        location = [int(x), int(y)]
        self._selected_vertex = self._collided_vertex(location[0], location[1])
        # if there is no vertex at pointer pos and grid snap is on
        # check if there is a vertex at the closest grid corner.
        if self._grid_snap_checkbox.value and self._selected_vertex is None:
            gridpoint = self._closest_grid_coordinate(x, y)
            location = self._grid_to_canvas_point(gridpoint[0], gridpoint[1])
            self._selected_vertex = self._collided_vertex(
                location[0], location[1]
            )  # select the vertex containing the mouse pointer position
        if self._selected_vertex is None and self._collided_edge(x, y) is None:
            # add a new vertex if no vertex is selected and
            # no edge contains the mouse pointer position
            vertex = self._least_available_label()
            self._graph.add_node(vertex, color=self._vertex_color, pos=location)
            self._selected_vertex = vertex
        with hold_canvas():
            # redraw graph and send the edges incident with selected vertex to layer 1
            # and the selected vertex to layer 3 for possible continuous update.
            self._mcanvas[2].clear()
            self._redraw_graph(self._selected_vertex)

        self._mouse_down = True

    def _handle_mouse_up(self, x, y) -> None:
        """
        Handle :meth:`ipycanvas.MultiCanvas.on_mouse_up`.

        It determines what to do when mouse button is released.
        """
        location = [int(x), int(y)]
        vertex = self._collided_vertex(location[0], location[1])
        # if there is no vertex at the pointer pos and grid snap is on
        # check if there is a vertex at the closest grid corner.
        if self._grid_snap_checkbox.value and vertex is None:
            gridpoint = self._closest_grid_coordinate(x, y)
            location = self._grid_to_canvas_point(gridpoint[0], gridpoint[1])
            vertex = self._collided_vertex(location[0], location[1])

        if self._selected_vertex is None:
            # This is to ignore the case when mousebutton is pressed
            # outside multicanvas and released on multicanvas
            return
        if vertex is None:
            # if there is no existing vertex containing the mouse pointer position,
            # add a new vertex and an edge between the new vertex and the selected vertex
            vertex = self._least_available_label()

            self._graph.add_node(vertex, color=self._vertex_color, pos=location)
            self._graph.add_edge(vertex, self._selected_vertex, color=self._edge_color)
        elif vertex is not None and vertex is not self._selected_vertex:
            # if there is a vertex containing mouse pointer position other than
            # the selected vertex, add / remove edge between these two vertices.
            if self._graph.has_edge(vertex, self._selected_vertex):
                self._graph.remove_edge(vertex, self._selected_vertex)
            else:
                self._graph.add_edge(
                    vertex, self._selected_vertex, color=self._edge_color
                )

        with hold_canvas():
            self._mcanvas[1].clear()
            self._mcanvas[2].clear()
            self._mcanvas[3].clear()
            self._redraw_graph()
        self._mouse_down = False

    def _handle_dblclick(self, x, y) -> None:
        """
        Handle double click event (using ipyevents).

        Double-clicking on a vertex or edge removes the vertex or the edge, respectively.
        """
        edge = self._collided_edge(x, y)
        vertex = self._collided_vertex(x, y)
        if vertex is not None and vertex == self._selected_vertex:
            self._graph.remove_node(self._selected_vertex)
        elif edge is not None:
            self._graph.remove_edge(edge[0], edge[1])

        with hold_canvas():
            self._mcanvas[2].clear()
            self._redraw_graph()
        self._selected_vertex = None

    def _handle_mouse_move(self, x, y) -> None:
        """
        Handle :meth:`ipycanvas.MultiCanvas.on_mouse_move`.

        It determines what to do when mouse pointer is moving on multicanvas.
        """
        location = [int(x), int(y)]
        collided_vertex = self._collided_vertex(x, y)
        self._mcanvas[4].clear()
        if self._grid_snap_checkbox.value and (
            collided_vertex is None or collided_vertex is self._selected_vertex
        ):
            gridpoint = self._closest_grid_coordinate(x, y)
            location = self._grid_to_canvas_point(gridpoint[0], gridpoint[1])

        if self._selected_vertex is None or not self._mouse_down:
            # do nothing if no vertex is selected or mouse button is not down
            if self._grid_snap_checkbox.value:
                with hold_canvas():
                    self._mcanvas[4].fill_style = "cyan"
                    self._mcanvas[4].fill_circle(location[0], location[1], 3)
            return

        if not self._vertex_move_on:
            # add a line segment between selected vertex and the mouse pointer position
            # and update layer 1 of multicanvas
            with hold_canvas():
                self._mcanvas[1].clear()
                self._mcanvas[1].stroke_style = self._edge_color
                self._mcanvas[1].line_width = self._edge_width
                self._mcanvas[1].stroke_line(
                    self._graph.nodes[self._selected_vertex]["pos"][0],
                    self._graph.nodes[self._selected_vertex]["pos"][1],
                    location[0],
                    location[1],
                )
                self._redraw_vertex(self._selected_vertex)
        else:
            # move vertex to mouse pointer position
            # and update layer 1 and 3 of multicanvas
            self._graph.nodes[self._selected_vertex]["pos"] = location
            with hold_canvas():
                self._mcanvas[1].clear()
                self._mcanvas[3].clear()
                self._redraw_vertex(self._selected_vertex)

    def _handle_mouse_out(self, x, y) -> None:
        """
        Handle :meth:`ipycanvas.MultiCanvas.on_mouse_out`.

        It determines what to do when the mouse leaves multicanvas.
        """
        _, _ = x, y  # To avoid unused variable warning
        self._selected_vertex = None
        self._vertex_move_on = False
        with hold_canvas():
            self._mcanvas[1].clear()
            self._mcanvas[2].clear()
            self._mcanvas[3].clear()
            self._redraw_graph()

    def _collided_vertex(self, x, y) -> int | None:
        """
        Return the vertex containing the point ``(x,y)`` on canvas.
        """
        for vertex in self._graph.nodes:
            if (self._graph.nodes[vertex]["pos"][0] - x) ** 2 + (
                self._graph.nodes[vertex]["pos"][1] - y
            ) ** 2 < self._radius**2:
                return vertex
        return None

    def _collided_edge(self, x, y) -> Edge | None:
        """
        Return the edge containing the point ``(x,y)`` on canvas.
        """
        for edge in self._graph.edges:
            if (
                self._point_distance_to_segment(
                    self._graph.nodes[edge[0]]["pos"],
                    self._graph.nodes[edge[1]]["pos"],
                    [x, y],
                )
                < self._edge_width / 2 + 1
            ):
                return edge
        return None

    @staticmethod
    def _point_distance_to_segment(a, b, point) -> float:
        """
        Return the distance between ``point`` and line segment given by ``a`` and ``b``.
        """
        a = np.asarray(a)
        b = np.asarray(b)
        point = np.asarray(point)
        ap = point - a
        ab = b - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = max(0, min(1, t))
        closest_point = a + t * ab
        return np.linalg.norm(point - closest_point)

    def _redraw_vertex(self, vertex) -> None:
        """
        Update the position of a specific vertex and its incident edges.

        It is used when repositioning a vertex and
        adding/removing a new edge so that only the parts related to
        the vertex are updated on canvas. The incident edges with vertex
        are drawn on layer 1 and the vertex itself is drawn on layer 3
        of the multicanvas.
        This is to make sure that edges do not show up above other vertices.
        """
        self._mcanvas[1].line_width = self._edge_width
        for v in self._graph[vertex]:
            self._mcanvas[1].stroke_style = self._graph[vertex][v]["color"]
            self._mcanvas[1].stroke_line(
                self._graph.nodes[vertex]["pos"][0],
                self._graph.nodes[vertex]["pos"][1],
                self._graph.nodes[v]["pos"][0],
                self._graph.nodes[v]["pos"][1],
            )
        self._mcanvas[3].fill_style = self._graph.nodes[vertex]["color"]
        x, y = self._graph.nodes[vertex]["pos"]
        self._mcanvas[3].fill_circle(x, y, self._radius)
        if self._vertex_labels:
            self._mcanvas[3].fill_style = "white"
            self._mcanvas[3].fill_text(str(vertex), x, y)

    def _redraw_graph(self, vertex=None) -> None:
        """
        Redraw the whole graph.

        If ``vertex`` is not None,
        then the edges incident with ``vertex`` are drawn on layer 1,
        ``vertex`` is drawn on layer 3 and all other vertices and edges are drawn
        on layer 2. This is to prepare multicanvas for adding/removing edges
        incident with ``vertex`` and repositioning ``vertex`` while keeping other vertices
        and edges fixed on multicanvas in layer 2.
        """
        self._mcanvas[1].line_width = self._edge_width
        self._mcanvas[2].line_width = self._edge_width
        for u, v in self._graph.edges:
            # i below is the index of the layer to be used.
            # if the edge is incident with ``vertex``,
            # draw this edge on layer 1 of multicanvas.
            # otherwise draw it on layer 2.
            if vertex in [u, v]:
                i = 1
            else:
                i = 2

            self._mcanvas[i].stroke_style = self._graph[u][v]["color"]
            self._mcanvas[i].stroke_line(
                self._graph.nodes[u]["pos"][0],
                self._graph.nodes[u]["pos"][1],
                self._graph.nodes[v]["pos"][0],
                self._graph.nodes[v]["pos"][1],
            )

        for v in self._graph.nodes:
            # i below is the index of the layer to be used.
            # draw ``vertex`` on layer 3 and other vertices on layer 2
            # so that moving ``vertex`` shows up above others.
            if vertex == v:
                i = 3
            else:
                i = 2
            self._mcanvas[i].fill_style = self._graph.nodes[v]["color"]
            x, y = self._graph.nodes[v]["pos"]
            self._mcanvas[i].fill_circle(x, y, self._radius)
            if self._vertex_labels:
                self._mcanvas[i].fill_style = "white"
                self._mcanvas[i].fill_text(str(v), x, y)

    def _grid_to_canvas_point(self, x, y):
        """
        Return the canvas coordinates for the given grid point ``(x,y)``.
        """
        # gridpoint = self._closest_grid_coordinate(x,y)

        return [
            self._mcanvas.width / 2 + x * self._grid_size,
            self._mcanvas.height / 2 - y * self._grid_size,
        ]

    def _closest_grid_coordinate(self, x, y):
        """
        Return the closest grid coordinates on canvas of the given point ``(x,y)``.
        """
        grid_x = int(round((x - self._mcanvas.width / 2) / self._grid_size))
        grid_y = int(round((self._mcanvas.height / 2 - y) / self._grid_size))

        # make sure that the coordinates do not exceed canvas size
        if grid_x < -1 * (self._mcanvas.width / 2) / self._grid_size:
            grid_x += 1
        elif grid_x > (self._mcanvas.width / 2) / self._grid_size:
            grid_x += -1
        if grid_y < -1 * (self._mcanvas.height / 2) / self._grid_size:
            grid_y += 1
        elif grid_y > (self._mcanvas.height / 2) / self._grid_size:
            grid_y += -1
        return [grid_x, grid_y]

    def _least_available_label(self):
        """
        Return the least non-negative integer available for the new vertex label.
        """
        if self._graph.number_of_nodes() == 0:
            return 0

        # the following is enough as there has to be
        # an available label from 0 to the number of vertices.
        for i in range(self._graph.number_of_nodes() + 1):
            if not self._graph.has_node(i):
                return i

    def graph(self) -> Graph:
        """
        Return a copy of the current graph on the multicanvas.
        """
        return Graph.from_vertices_and_edges(self._graph.nodes, self._graph.edges)

    def framework(self, grid: bool = False) -> Framework:
        """
        Return a copy of the current 2D framework on the multicanvas.

        Parameters
        ---------
        grid:
            If ``True`` and *Grid Snapping* is checked,
            the realization is scaled so that the grid points
            correspond to integral points.
        """
        H = self.graph()
        # create the realisation map where the origin is the center of the canvas
        posdict = {
            v: [
                int(self._graph.nodes[v]["pos"][0]) - int(self._mcanvas.width / 2),
                int(self._mcanvas.height / 2) - int(self._graph.nodes[v]["pos"][1]),
            ]
            for v in H.nodes
        }
        # when grid is True update (assing grid coordinates) the positions
        # of the vertices
        if self._grid_checkbox.value and grid:
            for v in H.nodes:
                posdict[v] = [Rational(x, self._grid_size) for x in posdict[v]]
        return Framework(graph=H, realization=posdict)
