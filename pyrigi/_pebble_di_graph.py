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
        Raises an error, if K and L don't satisfy the value constraints:
        K, L are integers, 0 < K, 0 <= L < 2K
        """
        if not (isinstance(K, int) and isinstance(L, int)):
            raise TypeError("K and L need to be integers!")

        if 0 >= K:
            raise ValueError("K must be positive")

        if 0 > L:
            raise ValueError("L must be non-negative")

        if L >= 2 * K:
            raise ValueError("L<2K must hold")

    def set_K(self, K: int) -> None:
        """
        Set K outside of the constructor.

        This will invalidate the current directions of the edges.

        Parameters
        ----------
        K: K must be integer and 0 < K. Also, L < 2K.
        """
        self._check_K_and_L(K, self.get_L())
        self._K = K

    def set_L(self, L: int) -> None:
        """
        Set L outside of the constructor.

        This will invalidate the current directions of the edges.

        Parameters
        ----------
        L: L must be integer and 0 <= L. Also, L < 2K.
        """
        self._check_K_and_L(self.get_K(), L)

        self._L = L

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

    def get_K(self) -> int:
        """
        Get the value of K.

        K is integer and 0 < K. Also, L < 2K.
        """
        return self._K

    def get_L(self) -> int:
        """
        Get the value of L.

        L is integer and 0 <= L. Also, L < 2K.
        """
        return self._L

    def number_of_edges(self) -> int:
        """
        Number of directed edges
        """
        return len(super().edges)

    def in_degree(self, node: Vertex) -> int:
        """
        Number of edges leading to node.

        Parameters
        ----------
        node: Vertex, that we wish to know the indegree.
        TODO check if vertex exists
        """
        return super().in_degree(node)

    def out_degree(self, node: Vertex) -> int:
        """
        Number of edges leading out from a node.

        Parameters
        ----------
        node: Vertex, that we wish to know the outdegree.
        TODO check if vertex exists
        """
        return super().out_degree(node)

    def point_edge_head_to(self, edge: Edge, node_to: Vertex) -> None:
        """
        Redirect given edge to the given head.

        Parameters
        ----------
        edge: Edge to redirect.
        node_to: Vertex to which the Edge will point to.
        """
        if self.has_node(node_to):
            tail = edge[0]
            head = edge[1]
            self.remove_edge(tail, head)
            self.add_edge(head, node_to)

    def added_edge_between(self, u: Vertex, v: Vertex) -> {bool, set}:
        """
        Check if edge can be added between the vertices u and v

        Return whether the given edge can be added
        and the fundamental (matroid) cycle of the edge uv.

        Parameters
        ----------
        u, v: vertices to add edge between.
        If u or v is not present in the graph, Error is raised.
        """

        def dfs(
            node: Vertex, visited: set, edge_path: list[Edge], current_edge=None
        ) -> {bool, set}:
            """
            Run depth first search to find vertices
            that can be reached from u or v.

            Returns whether any of these has outdegree < self._K
            and the set of reachable vertices.
            It will also turn edges around by this path.

            Parameters
            ----------
            node: Vertex, starting position of the dfs
            visited: set of Vertex. Contains the vertices already reached.
            edge_path: list of Edge. Contains the used edges in the transversal.
            current_edge: Edge. The edge through we reached this node.
            """
            visited.add(node)
            if current_edge:
                edge_path.append(current_edge)

            # Check if the stopping criteria is met
            if node != u and node != v and self.out_degree(node) < self.get_K():
                # turn around edges via path
                for edge in edge_path:
                    self.point_edge_head_to(edge, edge[0])

                return True, visited

            for out_edge in self.out_edges(node):
                found = False
                next_node = out_edge[-1]
                if next_node not in visited:
                    found, visited = dfs(next_node, visited, edge_path, out_edge)
                if found:
                    return True, visited
            if edge_path:
                edge_path.pop()
            return False, visited

        max_degree_for_u_and_v_together = 2 * self.get_K() - self.get_L() - 1

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

        while self.out_degree(u) + self.out_degree(v) > max_degree_for_u_and_v_together:
            visited_nodes = {u, v}

            edge_path_u, edge_path_v = [], []

            # Perform DFS from u
            found_from_u, visited_nodes = dfs(u, visited_nodes, edge_path_u)

            if found_from_u:
                continue

            # Perform DFS from v
            found_from_v, visited_nodes = dfs(v, visited_nodes, edge_path_v)

            if found_from_v:
                continue

            # not found_from_u and not found_from_v
            # so we reached the maximal extent of the reachable points
            # which will be the fundamental circuit
            break

        can_add_edge = (
            self.out_degree(u) + self.out_degree(v) <= max_degree_for_u_and_v_together
        )
        if can_add_edge:
            # Then the fundamental circuit is {u,v}
            visited_nodes = {u, v}

        return can_add_edge, visited_nodes

    def fundamental_circuit(self, u: Vertex, v: Vertex) -> set:
        """
        Get the list of nodes that form the fundamental circuit of uv

        These are the vertices that are
        accessible from u and v at the last passing of the dfs.

        Parameters
        ----------
        u, v: vertices, between the edge is formed,
        which we look for the fundamental circuit.

        If u or v is not present in the graph, Error is raised.
        """
        can_add_edge, fundamental_circuit = self.added_edge_between(u, v)
        if can_add_edge:
            return {u, v}
        else:
            return fundamental_circuit

    def can_add_edge_between_nodes(self, u: Vertex, v: Vertex) -> bool:
        """
        Check whether one can add the edge between the nodes u and v,
        so that it still respects the node degrees?

        Parameters
        ----------
        u, v: vertices, between an edge is proposed.

        If u or v is not present in the graph, Error is raised.

        """
        can_add_edge, fundamental_circuit = self.added_edge_between(u, v)
        return can_add_edge

    def add_edge_to_maintain_digraph_if_possible(self, u: Vertex, v: Vertex) -> bool:
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
        if self.can_add_edge_between_nodes(u, v):
            if self.out_degree(u) < self.out_degree(v):
                self.add_edges_from([(u, v)])
            else:
                self.add_edges_from([(v, u)])
            return True
        else:  # if not possible to add, just don't add
            return False

    def add_edges_to_maintain_out_degrees(self, edges: list[Edge]) -> None:
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
            self.add_edge_to_maintain_digraph_if_possible(edge[0], edge[1])
