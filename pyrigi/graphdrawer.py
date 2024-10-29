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
    interact
)
from ipycanvas import Canvas, hold_canvas
from IPython.display import display
from pyrigi.graph import Graph
from ipyevents import Event
import networkx as nx
import numpy as np
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

    def __init__(
        self, graph: Graph = None, layout_type: str = "planar", place="all"
    ) -> None:
        """
        Constructor of the class.

        TODO
        ---
        Add width/height parameters to canvas. Currently canvas has fixed width=600 and height=600.

        """
        self._edit_type = "Edge"
        self._radius = 10
        self._ewidth = 2
        self._v_color = "blue"
        self._e_color = "black"

        #self._last_mdown_time = -5  # last time mouse was down
        #self._last_mup_time = -3  # last time mouse was up
        #self._last_click_pos = [-1, -1]

        #self._last_click_time = -1
        self._selected_vertex = None
        self._next_vertex_label = 0
        self._show_vlabels = True
        self._mouse_down = False
        #self._mouse_pos = [0, 0]

        self._G = Graph()  # the graph on canvas
        self._out = Output()  # can later be used to represent some properties

        # setting canvas properties
        self._canvas = Canvas(width=600, height=600)
        self._canvas.stroke_rect(0, 0, self._canvas.width, self._canvas.height)

        #self._canvas.on_key_down(self._on_keyboard_event)
        #self._canvas.on_mouse_down(self._handle_mouse_down)
        #self._canvas.on_mouse_up(self._handle_mouse_up)
        #self._canvas.on_mouse_move(self._handle_mouse_move)
        #self._canvas.on_mouse_out(self._handle_mouse_out)
        self._canvas.font = "12px serif"
        self._canvas.text_align = "center"
        self._canvas.text_baseline = "middle"

        ##### IpyEvents Part ###
        self._events = Event(wait=10)
        self._events.source = self._canvas
        self._events.watched_events=['click', 'mousedown','dblclick','mousemove','mouseup','mouseleave']
        self._events.on_dom_event(self._handle_event)
        self._events.ignore_modifier_key_events=True

        ##### menu items #######
        # # Edit Type radio buttons
        # self._radio_buttons = RadioButtons(
        #     options=["Vertex", "Edge"],
        #     value="Vertex",
        #     description="Edit Type:",
        #     disabled=False,
        # )
        # self._radio_buttons.observe(self._on_edit_type_change)

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
        box = HBox([self._canvas, right_box])

        if isinstance(graph, Graph) and graph.number_of_nodes() > 0:
            self._set_G(graph, layout_type, place)
            with hold_canvas():
                self._canvas.clear()
                self._redraw_graph()

        # displaying the combined menu and canvas, and the output
        display(box)
        display(self._out)

    def _handle_event(self,event):
        x,y = event['relativeX'],event['relativeY']
        if event['event']=='mousemove':
            self._handle_mouse_move(x,y,event['ctrlKey'])
        elif event['event']=='mousedown':
            self._handle_mouse_down(x,y)
        elif event['event']=='dblclick':
            self._handle_dblclick(x,y)
        elif event['event']=='mouseup':
            self._handle_mouse_up(x,y)
        elif event['event']=='mouseleave':
            self._handle_mouse_out(x,y)
        #with self._out:
            #print(event['relativeX'],event['relativeY'],event['shiftKey'],event['event'])

    def _assign_pos(self, x, y, place):
        """
        This function converts layout positions which are between -1 and 1 to canvas positions according to the chosen place by scaling.
        """
        width = self._canvas.width
        height = self._canvas.height
        r = self._radius

        # -3 is used below so that the vertices do not touch the edges of the canvas
        if place == "all":
            return [
                width / 2 + x * (width / 2 - r - 3),
                height / 2 + y * (height / 2 - r - 3),
            ]
        if place == "N":
            return [
                width / 2 + x * (width / 2 - r - 3),
                height / 4 + y * (height / 4 - r - 3),
            ]
        if place == "S":
            return [
                width / 2 + x * (width / 2 - r - 3),
                height * 3 / 4 + y * (height / 4 - r - 3),
            ]
        if place == "W":
            return [
                width / 4 + x * (width / 4 - r - 3),
                height / 2 + y * (height / 2 - r - 3),
            ]
        if place == "E":
            return [
                width * 3 / 4 + x * (width / 4 - r - 3),
                height / 2 + y * (height / 2 - r - 3),
            ]
        if place == "NE":
            return [
                width * 3 / 4 + x * (width / 4 - r - 3),
                height / 4 + y * (height / 4 - r - 3),
            ]
        if place == "NW":
            return [
                width / 4 + x * (width / 4 - r - 3),
                height / 4 + y * (height / 4 - r - 3),
            ]
        if place == "SE":
            return [
                width * 3 / 4 + x * (width / 4 - r - 3),
                height * 3 / 4 + y * (height / 4 - r - 3),
            ]
        if place == "SW":
            return [
                width / 4 + x * (width / 4 - r - 3),
                height * 3 / 4 + y * (height / 4 - r - 3),
            ]

    def _set_G(self, graph: Graph, layout_type, place):
        map = {}
        for vertex in graph:
            if not isinstance(vertex, int) or vertex < 0:
                for k in range(graph.number_of_nodes()):
                    if not graph.has_node(k) and k not in map.values():
                        map[vertex] = k
                        break
        graph = nx.relabel_nodes(graph, map, copy=True)
        placement = graph.layout(layout_type)

        # random layout assigns coordinates between 0 and 1. adjust the coordinates to between -1 and 1 as other layouts
        if layout_type == "random":
            for vertex in placement:
                placement[vertex] = [2 * x - 1 for x in placement[vertex]]

        # add vertices to the graph of the graphdrawer by scaling the coordinates from [-1,1] to [self._canvas.width, self._canvas.height]
        for vertex in graph.vertex_list():
            [px, py] = placement[vertex]
            self._G.add_node(
                vertex, color=self._v_color, pos=self._assign_pos(px, py, place)
            )
        for edge in graph.edge_list():
            self._G.add_edge(edge[0], edge[1], color=self._e_color)

        self._next_vertex_label = max(self._G.vertex_list()) + 1
        if len(map) != 0:
            with self._out:
                print("relabeled vertices:", map)

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

        # self._edit_type = 'Edge'
        self._selected_vertex = self._collided_vertex(x, y)

        if self._selected_vertex == None and self._collided_edge(x, y) == None:
            self._G.add_node(self._next_vertex_label, color=self._v_color, pos=[x, y])
            # self.vertex_pos_dict[self.next_vertex_label] = (x, y)
            self._selected_vertex = self._next_vertex_label
            self._next_vertex_label += 1
            with hold_canvas():
                self._canvas.clear()
                self._redraw_graph()
        #self._last_mdown_time = time.time()
        self._mouse_down = True

    def _handle_mouse_up(self,x,y):
        vertex = self._collided_vertex(x,y)
        s_vertex = self._selected_vertex

        if s_vertex is None:
            # This is to ignore the case when mousebutton is pressed outside canvas and released on canvas
            return 
        if vertex is None:
            vertex = self._next_vertex_label
            self._G.add_node(vertex, color=self._v_color, pos=[x, y])
            # self.vertex_pos_dict[self.next_vertex_label] = (x, y)
            #self._selected_vertex = self._next_vertex_label
            self._G.add_edge(vertex, s_vertex, color=self._e_color)
            self._next_vertex_label += 1
        elif vertex is not None and vertex is not s_vertex:
            # if isinstance(collided, int) and collided != vertex:
            #     neighbour = collided
            if sorted((vertex, s_vertex)) in self._G.edge_list():
                self._G.remove_edge(vertex, s_vertex)
            else:
                self._G.add_edge(vertex, s_vertex, color=self._e_color)

            #     self._selected_vertex = neighbour

        #self._selected_vertex = None
        with hold_canvas():
                self._canvas.clear()
                self._redraw_graph()
        self._mouse_down = False
        with hold_canvas():
            self._canvas.clear()
            self._redraw_graph()



    def _handle_dblclick(self,x,y):
        edge = self._collided_edge(x, y)
        vertex = self._collided_vertex(x, y)
        if vertex != None and vertex == self._selected_vertex:
            self._G.remove_node(self._selected_vertex)
        elif edge != None:
            self._G.remove_edge(edge[0], edge[1])
        
        with hold_canvas():
            self._canvas.clear()
            self._redraw_graph()
        self._selected_vertex = None


    def _handle_mouse_move(self, x, y, vertexmove_on):

        vertex = self._selected_vertex

        if vertex is None or not self._mouse_down:
            # do nothing if no vertex is selected or mouse button is not down
            return
        #self._mouse_pos = [x, y]

        if not vertexmove_on:
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

        else:
            self._G.nodes[vertex]["pos"] = [x, y]
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

 #   def _on_keyboard_event(self, key, shift_key, ctrl_key, meta_key):
 #       if (
 #           ctrl_key == True
 #           and not self._edge_draw
 #           and self._collided_vertex(self._mouse_pos[0], self._mouse_pos[1]) != None
 #       ):
 #           self._edit_type = "Vertex"
 #       else:
 #           self._edit_type = "Edge"

    def _collided_edge(self, x, y):
        """
        Return the edge containing the point (x,y) on canvas.
        """
        for edge in self._G.edge_list():
            if (
                self._point_distance_to_segment(
                    self._G.nodes[edge[0]]["pos"], self._G.nodes[edge[1]]["pos"], [x, y]
                )
                < self._ewidth / 2 + 1
            ):
                return edge
        return None

    def _point_distance_to_segment(self, a, b, p):
        """
        Return the distance between a line segment with endpoints 'a' and 'b', and a point 'p'
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

    def _redraw_graph(self) -> None:
        """
        Update the graph on canvas to illustrate the latest changes.
        """

        """ BELOW IS FOR SNAPPING FEATURE
        self._canvas.stroke_style = 'gray'
        self._canvas.line_width = 1
        for n in range(30, self._canvas.height, 30):
            self._canvas.stroke_line(0,n,self._canvas.width,n)
        for n in range(30, self._canvas.width, 30):
            self._canvas.stroke_line(n,0,n,self._canvas.height)
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
