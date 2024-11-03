import networkx as nx

from pyrigi.data_type import Vertex, Edge

# from pyrigi.misc import doc_category, generate_category_tables
# from pyrigi.exception import LoopError

"""
Auxilary class for directed graph used in pebble game style algorithms.
"""


class PebbleDiGraph(nx.MultiDiGraph):
    """
    Class representing a directed graph
    that keeps all necessary data for pebble game algorithm.

    All nx methods in use need a wrapper - to make future developments easier.
    """

    def __init__(self, K: int = None, L: int = None, *args, **kwargs) -> None:
        """
        Initialisation, in which we can set up the graph and the values of K and L,
        used for the pebble game algorithm.
        """
        # We allow not defining them yet
        if K is not None and L is not None:
            self._check_K_and_L(K, L)

        self._K = K
        self._L = L

        super().__init__(*args, **kwargs)

    def _check_K_and_L(self, K: int, L: int) -> None:
        """
        Check if K and L satisfy the conditions of
        K > 0, 0 <= L < 2K
        """
        # Check that K and L are integers
        if not (isinstance(K, int) and isinstance(L, int)):
            raise TypeError("K and L need to be integers!")

        # Check the conditions
        if 0 >= K:
            raise ValueError("K must be positive")

        if 0 > L:
            raise ValueError("L must be non-negative")

        if L >= 2 * K:
            raise ValueError("L<2K must hold")

    @property
    def K(self) -> int:
        """
        Get the value of K.

        K is integer and 0 < K. Also, L < 2K.
        """
        return self._K

    @K.setter
    def K(self, value: int) -> None:
        """
        Set K outside of the constructor.

        This will invalidate the current directions of the edges.

        Parameters
        ----------
        K: K must be integer and 0 < K. Also, L < 2K.
        """
        self._check_K_and_L(value, self._L)
        self._K = value

    @property
    def L(self) -> int:
        """
        Get the value of L.

        L is integer and 0 <= L. Also, L < 2K.
        """
        return self._L

    @L.setter
    def L(self, value: int) -> None:
        """
        Set L outside of the constructor.

        This will invalidate the current directions of the edges.

        Parameters
        ----------
        L: L must be integer and 0 <= L. Also, L < 2K.
        """
        self._check_K_and_L(self._K, value)
        self._L = value

    def set_K_and_L(self, K: int, L: int) -> None:
        """
        Set K and L together outside of the constructor.

        This will invalidate the current directions of the edges.

        Parameters
        ----------
        K: K is integer and 0 < K.
        L: L is integer and 0 <= L.
        Also, L < 2K.
        """
        self._check_K_and_L(K, L)

        self._K = K
        self._L = L

    def number_of_edges(self) -> int:
        """
        Number of directed edges
        """
        return len(super().edges)

    def in_degree(self, vertex: Vertex) -> int:
        """
        Number of edges leading to vertex.

        Parameters
        ----------
        vertex: Vertex, that we wish to know the indegree.
        TODO check if vertex exists
        """
        return super().in_degree(vertex)

    def out_degree(self, vertex: Vertex) -> int:
        """
        Number of edges leading out from a vertex.

        Parameters
        ----------
        vertex: Vertex, that we wish to know the outdegree.
        TODO check if vertex exists
        """
        return super().out_degree(vertex)

    def redirect_edge_to_head(self, edge: Edge, vertex_to: Vertex) -> None:
        """
        Redirect given edge to the given head.

        Parameters
        ----------
        edge: Edge to redirect.
        vertex_to: Vertex to which the Edge will point to.
                 Vertex must be part of the Edge.
        """
        if self.has_node(vertex_to) and vertex_to in edge:
            tail = edge[0]
            head = edge[1]
            self.remove_edge(tail, head)
            self.add_edge(head, vertex_to)

    def fundamental_circuit(self, u: Vertex, v: Vertex) -> {set[Vertex]}:
        """
        Return the fundamental (matroid) cycle of the edge uv.

        If the edge uv is independent, returns None


        Parameters
        ----------
        u, v: vertices to add edge between.
        If u or v is not present in the graph, Error is raised.
        """

        def dfs(
            vertex: Vertex, visited: set, edge_path: list[Edge], current_edge=None
        ) -> {bool, set}:
            """
            Run depth first search to find vertices
            that can be reached from u or v.

            Returns whether any of these has outdegree < self._K
            and the set of reachable vertices.
            It will also turn edges around by this path.

            Parameters
            ----------
            vertex: Vertex, starting position of the dfs
            visited: set of Vertex. Contains the vertices already reached.
            edge_path: list of Edge. Contains the used edges in the transversal.
            current_edge: Edge. The edge through we reached this vertex.
            """
            visited.add(vertex)
            if current_edge:
                edge_path.append(current_edge)

            # Check if the stopping criteria is met
            if vertex != u and vertex != v and self.out_degree(vertex) < self.K:
                # turn around edges via path
                for edge in edge_path:
                    self.redirect_edge_to_head(edge, edge[0])

                return True, visited

            for out_edge in self.out_edges(vertex):
                found = False
                next_vertex = out_edge[-1]
                if next_vertex not in visited:
                    found, visited = dfs(next_vertex, visited, edge_path, out_edge)
                if found:
                    return True, visited
            if edge_path:
                edge_path.pop()
            return False, visited

        max_degree_u_v_together = 2 * self.K - self.L - 1

        if not self.has_node(u):
            raise ValueError(
                "Cannot check if edge can be added, since Vertex "
                + u
                + " is not present in graph."
            )

        if not self.has_node(v):
            raise ValueError(
                "Cannot check if edge can be added, since Vertex "
                + v
                + " is not present in graph."
            )

        while self.out_degree(u) + self.out_degree(v) > max_degree_u_v_together:
            visited_vertices = {u, v}

            edge_path_u, edge_path_v = [], []

            # Perform DFS from u
            found_from_u, visited_vertices = dfs(u, visited_vertices, edge_path_u)

            if found_from_u:
                continue

            # Perform DFS from v
            found_from_v, visited_vertices = dfs(v, visited_vertices, edge_path_v)

            if found_from_v:
                continue

            # not found_from_u and not found_from_v
            # so we reached the maximal extent of the reachable points
            # which will be the fundamental circuit
            break

        can_add_edge = (
            self.out_degree(u) + self.out_degree(v) <= max_degree_u_v_together
        )
        if can_add_edge:
            # The edge is independent
            return None

        return visited_vertices

    def can_add_edge_between_vertices(self, u: Vertex, v: Vertex) -> bool:
        """
        Check whether one can add the edge between the vertices u and v,
        so that it still respects the vertex degrees?

        Parameters
        ----------
        u, v: vertices, between an edge is proposed.

        If u or v is not present in the graph, Error is raised.

        """
        return self.fundamental_circuit(u, v) is None

    def add_edge_maintaining_digraph(self, u: Vertex, v: Vertex) -> bool:
        """
        Add the given edge to the directed graph, if possible.

        This will also check the possibility of adding the edge and return
        True or False depending on it.

        Parameters
        ----------
        u, v: vertices, between an edge is proposed
        """
        # if the vertex u is not present (yet), then it has outdegree 0
        # => it is ok to add the directed edge from there
        if not self.has_node(u):
            self.add_edges_from([(u, v)])
            return True
        # if the vertex v is not present (yet), then it has outdegree 0
        # => it is ok to add the directed edge from there
        if not self.has_node(v):
            self.add_edges_from([(v, u)])
            return True

        # heuristics: point it out from the one with the fewer outdegrees
        if self.can_add_edge_between_vertices(u, v):
            if self.out_degree(u) < self.out_degree(v):
                self.add_edges_from([(u, v)])
            else:
                self.add_edges_from([(v, u)])
            return True
        else:  # if not possible to add, just don't add
            return False

    def add_edges_maintaining_digraph(self, edges: list[Edge]) -> None:
        """
        Add a list of edges to the directed graph
        so that it will choose the correct orientations of them and
        constructs the corresponding pebble graph.

        ! Note that this might not add all the edges, only the edges that
        ! take part of the maximal sparse subgraph

        Parameters
        ----------
        edges: List of Edge to add
        """
        for edge in edges:
            self.add_edge_maintaining_digraph(edge[0], edge[1])
