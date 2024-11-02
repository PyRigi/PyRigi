"""
Module for graph drawing on a canvas in jupyter notebook.

.. currentmodule:: pyrigi.graph_drawer

Classes:

.. autosummary::

    GraphDrawer
"""

from ipywidgets import Output, ColorPicker, HBox, VBox, IntSlider, Checkbox
from ipycanvas import MultiCanvas, hold_canvas
from IPython.display import display
from pyrigi.graph import Graph
from pyrigi.framework import Framework
from ipyevents import Event
import networkx as nx
import numpy as np


class GraphDrawer(object):
    """
    Class for graph drawing.

    An instance of this class creates a canvas and
    takes mouse inputs in order to construct a graph. The vertices of the graph
    will be labeled using non-negative integers. Supported inputs are listed below.

    - Press mouse button on an empty place on canvas:
        Add a vertex at the pointer position.
    - Press mouse button on an existing vertex (or empty space) and release the mouse button on another vertex (or empty space):
        Add/remove an edge between the two vertices.
    - Drag a vertex with ``Ctrl`` is being pressed:
        Reposition the vertex.
    - Double click on an existing vertex:
        Remove the corresponding vertex.
    - Double click on an existing edge:
        Remove the corresponding edge.

    Parameters
    ----------
    graph:
        (optional) A graph without loops which is to be drawn on canvas
        when the object is created. The non-integer labels are relabeled
    layout_type:
        Layout type to visualise the ``graph``.
        For supported layout types see :meth:`.Graph.layout`.
        The default is ``spring``.
        If ``graph`` is ``None`` or empty this parameter will not have any effect.
    place:
        The part of the canvas that will be used for drawing ``graph``.
        Options are ``all`` (default, use all canvas), ``E`` (use the east part),
        ``W`` (use the west part), ``N`` (use the north part), ``S`` (use the south part),
        and also ``NE``, ``NW``, ``SE`` and ``SW``.
        If ``graph`` is ``None`` or empty this parameter will not have any effect.


    Examples
    --------
    >>> from pyrigi import GraphDrawer
    >>> Drawer = GraphDrawer()
    HBox(children=(MultiCanvas(height=600, width=600), VBox(children=(ColorPicker(value='blue', description='V-Color'), ColorPicker(value='black', description='E-Color'), IntSlider(value=10, description='V-Size:', max=20, min=8), IntSlider(value=2, description='E-Size:', max=10, min=1), Checkbox(value=True, description='Show V-Labels', indent=False)))))
    Output()
    press and hold ctrl key to move vertices around with mouse.
    >>> Drawer.graph()
    Graph with vertices [] and edges []

    TODO
    ----
    - Add width/height parameters to canvas. Currently width=600 and height=600 are fixed.
    - Add a background grid option.

    """  # noqa: E501

    def __init__(
        self, graph: Graph = None, layout_type: str = "spring", place: str = "all"
    ) -> None:
        """
        Constructor of the class.
        """

        self._radius = 10
        self._ewidth = 2
        self._v_color = "blue"  # default color for vertices
        self._e_color = "black"  # default color for edges

        self._selected_vertex = None  # this determines what vertex to update on canvas
        self._next_vertex_label = 0  # label for next vertex
        self._show_vlabels = True
        self._mouse_down = False
        self._vertexmove_on = False

        self._graph = Graph()  # the graph on canvas
        self._out = Output()  # can later be used to represent some properties

        # setting multicanvas properties
        self._mcanvas = MultiCanvas(4, width=600, height=600)
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
            readout_format="d",
        )
        self._ewidth_slider.observe(self._on_ewidth_change)

        # setting checkbox for showing vertex labels
        self._vlabel_checkbox = Checkbox(
            value=True, description="Show V-Labels", disabled=False, indent=False
        )
        self._vlabel_checkbox.observe(self._on_show_vlabel_change)

        # combining the menu and canvas
        right_box = VBox(
            [
                self._vcolor_picker,
                self._ecolor_picker,
                self._vradius_slider,
                self._ewidth_slider,
                self._vlabel_checkbox,
            ]
        )
        box = HBox([self._mcanvas, right_box])

        if isinstance(graph, Graph) and graph.number_of_nodes() > 0:
            self._set_graph(graph, layout_type, place)
            with hold_canvas():
                self._mcanvas[1].clear()
                self._redraw_graph()

        # displaying the combined menu and canvas, and the output
        display(box)
        display(self._out)
        with self._out:
            print("press and hold ctrl key to move vertices around with mouse.")

    def _handle_event(self, event):
        """
        This function handles keyboard events and double click event using ``ipyevents``.
        """
        if event["event"] == "keydown":
            self._vertexmove_on = event["ctrlKey"]
        elif event["event"] == "keyup":
            self._vertexmove_on = event["ctrlKey"]

        x, y = event["relativeX"], event["relativeY"]
        if event["event"] == "dblclick":
            self._handle_dblclick(x, y)

    def _assign_pos(self, x, y, place):
        """
        This function converts layout positions which are between -1 and 1
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

    def _set_graph(self, graph: Graph, layout_type, place):
        vertex_map = {}
        for vertex in graph:
            if not isinstance(vertex, int) or vertex < 0:
                for k in range(graph.number_of_nodes()):
                    if not graph.has_node(k) and k not in vertex_map.values():
                        vertex_map[vertex] = k
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
                vertex, color=self._v_color, pos=self._assign_pos(px, py, place)
            )
        for edge in graph.edges:
            self._graph.add_edge(edge[0], edge[1], color=self._e_color)

        self._next_vertex_label = max(self._graph.nodes) + 1
        if len(vertex_map) != 0:
            with self._out:
                print("relabeled vertices:", vertex_map)

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
                self._mcanvas[2].clear()
                self._redraw_graph()

    def _on_ewidth_change(self, change) -> None:
        """
        Handler of the edge width slider.
        """
        if change["type"] == "change" and change["name"] == "value":
            self._ewidth = change["new"]
            with hold_canvas():
                self._mcanvas[2].clear()
                self._redraw_graph()

    def _on_show_vlabel_change(self, change) -> None:
        """
        Handler of the vertex labels checkbox.
        """
        if change["type"] == "change" and change["name"] == "value":
            self._show_vlabels = change["new"]
            with hold_canvas():
                self._mcanvas[2].clear()
                self._redraw_graph()

    def _handle_mouse_down(self, x, y):
        """
        Handler for :meth:`ipycanvas.MultiCanvas.on_mouse_down`.

        It determines what to do when mouse button is pressed.
        """

        self._selected_vertex = self._collided_vertex(
            x, y
        )  # select the vertex containing the mouse pointer position
        if self._selected_vertex is None and self._collided_edge(x, y) is None:
            # add a new vertex if no vertex is selected and
            # no edge contains the mouse pointer position
            self._graph.add_node(
                self._next_vertex_label, color=self._v_color, pos=[int(x), int(y)]
            )
            self._selected_vertex = self._next_vertex_label
            self._next_vertex_label += 1
        with hold_canvas():
            # redraw graph and send the edges incident with selected vertex to layer 1
            # and the selected vertex to layer 3 for possible continuous update.
            self._mcanvas[2].clear()
            self._redraw_graph(self._selected_vertex)

        self._mouse_down = True

    def _handle_mouse_up(self, x, y):
        """
        Handler for :meth:`ipycanvas.MultiCanvas.on_mouse_up`.

        It determines what to do when mouse button is released.
        """
        vertex = self._collided_vertex(x, y)
        s_vertex = self._selected_vertex

        if s_vertex is None:
            # This is to ignore the case when mousebutton is pressed
            # outside multicanvas and released on multicanvas
            return
        if vertex is None:
            # if there is no existing vertex containing the mouse pointer position,
            # add a new vertex and an edge between the new vertex and the selected vertex
            vertex = self._next_vertex_label
            self._graph.add_node(vertex, color=self._v_color, pos=[int(x), int(y)])
            self._graph.add_edge(vertex, s_vertex, color=self._e_color)
            self._next_vertex_label += 1
        elif vertex is not None and vertex is not s_vertex:
            # if there is a vertex containing mouse pointer position other than
            # the selected vertex, add / remove edge between these two vertices.
            if self._graph.has_edge(vertex, s_vertex):
                self._graph.remove_edge(vertex, s_vertex)
            else:
                self._graph.add_edge(vertex, s_vertex, color=self._e_color)

        with hold_canvas():
            self._mcanvas[1].clear()
            self._mcanvas[2].clear()
            self._mcanvas[3].clear()
            self._redraw_graph()
        self._mouse_down = False

    def _handle_dblclick(self, x, y):
        """
        This function is the handler for double click event (using ipyevents).

        Double clicking on a vertex or edge will remove the vertex or the edge, resp.
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

    def _handle_mouse_move(self, x, y):
        """
        Handler for :meth:`ipycanvas.MultiCanvas.on_mouse_move`.

        It determines what to do when mouse pointer is moving on multicanvas.
        """

        vertex = self._selected_vertex

        if vertex is None or not self._mouse_down:
            # do nothing if no vertex is selected or mouse button is not down
            return

        if not self._vertexmove_on:
            # add a line segment between selected vertex and the mouse pointer position
            # and update layer 1 of multicanvas
            with hold_canvas():
                self._mcanvas[1].clear()
                self._mcanvas[1].stroke_style = self._e_color
                self._mcanvas[1].line_width = self._ewidth
                self._mcanvas[1].stroke_line(
                    self._graph.nodes[vertex]["pos"][0],
                    self._graph.nodes[vertex]["pos"][1],
                    x,
                    y,
                )
                self._redraw_vertex(vertex)
        else:
            # move vertex to mouse pointer position
            # and update layer 1 and 3 of multicanvas
            self._graph.nodes[vertex]["pos"] = [int(x), int(y)]
            with hold_canvas():
                self._mcanvas[1].clear()
                self._mcanvas[3].clear()
                self._redraw_vertex(vertex)

    def _handle_mouse_out(self, x, y):
        """
        Handler for :meth:`ipycanvas.MultiCanvas.on_mouse_out`.

        It determines what to do when the mouse leaves multicanvas.
        """
        self._selected_vertex = None
        self._vertexmove_on = False
        with hold_canvas():
            self._mcanvas[1].clear()
            self._mcanvas[2].clear()
            self._mcanvas[3].clear()
            self._redraw_graph()

    def _collided_vertex(self, x, y) -> int | None:
        """
        Return the vertex containing the point (x,y) on canvas.
        """
        for vertex in self._graph.nodes:
            if (self._graph.nodes[vertex]["pos"][0] - x) ** 2 + (
                self._graph.nodes[vertex]["pos"][1] - y
            ) ** 2 < self._radius**2:
                return vertex
        return None

    def _collided_edge(self, x, y):
        """
        Return the edge containing the point (x,y) on canvas.
        """
        for edge in self._graph.edges:
            if (
                self._point_distance_to_segment(
                    self._graph.nodes[edge[0]]["pos"],
                    self._graph.nodes[edge[1]]["pos"],
                    [x, y],
                )
                < self._ewidth / 2 + 1
            ):
                return edge
        return None

    def _point_distance_to_segment(self, a, b, p):
        """
        Return the distance between point 'p' and line segment given by 'a' and 'b'.
        """
        a = np.asarray(a)
        b = np.asarray(b)
        p = np.asarray(p)
        ap = p - a
        ab = b - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = max(0, min(1, t))
        closest_point = a + t * ab
        return np.linalg.norm(p - closest_point)

    def _redraw_vertex(self, vertex):
        """
        Update the position of a specific vertex and its incident edges

        It is used when repositioning a vertex and
        adding/removing a new edge so that only the parts related to
        the vertex are updated on canvas. The incident edges with vertex
        are drawn on layer 1 and the vertex itself is drawn on layer 3
        of the multicanvas.
        This is to make sure that edges do not show up above other vertices.
        """
        self._mcanvas[1].line_width = self._ewidth
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
        if self._show_vlabels:
            self._mcanvas[3].fill_style = "white"
            self._mcanvas[3].fill_text(str(vertex), x, y)

    def _redraw_graph(self, hvertex=None) -> None:
        """
        Redraw the whole graph.

        If hvertex is not None,
        then the edges incident with hvertex are drawn on layer 1,
        hvertex is drawn on layer 3 and all other vertices and edges are drawn
        on layer 2. This is to prepare multicanvas for adding/removing edges
        incident with hvertex and repositioning hvertex while keeping other vertices
        and edges fixed on multicanvas in layer 2.
        """
        self._mcanvas[1].line_width = self._ewidth
        self._mcanvas[2].line_width = self._ewidth
        for u, v in self._graph.edges:
            # n below is the index of the layer to be used.
            # if the edge is incident with hvertex,
            # draw this edge on layer 1 of multicanvas.
            # otherwise draw it on layer 2.
            if hvertex in [u, v]:
                n = 1
            else:
                n = 2

            self._mcanvas[n].stroke_style = self._graph[u][v]["color"]
            self._mcanvas[n].stroke_line(
                self._graph.nodes[u]["pos"][0],
                self._graph.nodes[u]["pos"][1],
                self._graph.nodes[v]["pos"][0],
                self._graph.nodes[v]["pos"][1],
            )

        for vertex in self._graph.nodes:
            # n below is the index of the layer to be used.
            # draw hvertex on layer 3 and other vertices on layer 2
            # so that moving vertex (hvertex) will show up above others.
            if hvertex == vertex:
                n = 3
            else:
                n = 2
            self._mcanvas[n].fill_style = self._graph.nodes[vertex]["color"]
            x, y = self._graph.nodes[vertex]["pos"]
            self._mcanvas[n].fill_circle(x, y, self._radius)
            if self._show_vlabels:
                self._mcanvas[n].fill_style = "white"
                self._mcanvas[n].fill_text(str(vertex), x, y)

    def graph(self) -> Graph:
        """
        Return a copy of the current graph on the multicanvas.
        """
        H = Graph()
        H.add_nodes_from(self._graph.nodes)
        H.add_edges_from(self._graph.edges)
        return H

    def framework(self) -> Framework:
        """
        Return a copy of the current 2D framework on the multicanvas.
        """
        H = self.graph()
        posdict = {
            v: [
                self._graph.nodes[v]["pos"][0] - int(self._mcanvas.width / 2),
                int(self._mcanvas.height / 2) - self._graph.nodes[v]["pos"][1],
            ]
            for v in H.nodes
        }
        return Framework(graph=H, realization=posdict)
