import networkx as nx

#from pyrigi.data_type import Vertex, Edge
#from pyrigi.misc import doc_category, generate_category_tables
#from pyrigi.exception import LoopError

"""
Auxilary class for directed graph used in pebble game style algorithms.
"""

class MultiDiGraph(nx.MultiDiGraph):
    """
    Class representing a directed graph.
    All nx methods in use need a wrapper - to make future developments easier. 
    """
    def __init__(self, K=None, L=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__K = K
        self.__L = L

    def set_K(self, K):
        self.__K = K

    def set_L(self, L):
        self.__L = L

    def get_K(self):
        return self.__K

    def get_L(self):
        return self.__L

    def get_number_of_edges(self):
        return len(super().edges)

    def in_degree(self, node):
        return super().in_degree(node)
    
    def out_degree(self, node):
        return super().out_degree(node)
    
    # Redirect edge to the given head
    def point_edge_head_to(self, edge, node_to):
        # placeholder 
        tail = edge[0]
        head = edge[1]
        self.remove_edge(tail, head)
        self.add_edge(head, node_to)

    # Checks if you can add edge between the the vertices u and v
    # It returns if the given edge can be added 
    # and the fundamental (matroid) cycle of the edge u, v. 
    def added_edge_between(self, u, v):
        # Running depth first search to find vertices that can be reached
        # returns if any of these has outdegree < self._K
        # It will also turn edges around by this path.
        def dfs(node, visited, edge_path, current_edge = None):
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

        while self.out_degree(u) + self.out_degree(v) > max_degree_for_u_and_v_together:
            visited_nodes = {u,v}

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

        can_add_edge = self.out_degree(u) + self.out_degree(v) <= max_degree_for_u_and_v_together 
        if can_add_edge:
            # Then the fundamental circuit is {u,v}
            visited_nodes = {u,v}

        return can_add_edge, visited_nodes

    # Get the list of nodes that form the fundamental circuit of {uv}
    # These are the vertices that are 
    # accessible from u and v at the last passing of the dfs
    def fundamental_circuit(self, u, v):
        can_add_edge, fundamental_circuit = self.added_edge_between(u,v)
        if can_add_edge:
            return {u,v}
        else:
            return fundamental_circuit

    # Can you add the edge between the nodes u and v, 
    # so that it still respects the node degrees?   
    def can_add_edge_between_nodes(self, u, v):
        can_add_edge, fundamental_circuit = self.added_edge_between(u,v)
        return can_add_edge

    # Function to add the given edge to the directed graph, if possible. 
    # 
    # This will also check the possibility of adding the edge and return 
    # True or False depending on it.  
    def add_edge_to_maintain_digraph_if_possible(self, u, v):
        # if the vertex u is not present (yet), then it has outdegree 0
        # => it is ok to add the directed edge from there 
        if u not in self.nodes():
            self.add_edges_from([(u,v)])
            return
        # if the vertex v is not present (yet), then it has outdegree 0 
        # => it is ok to add the directed edge from there
        if v not in self.nodes():
            self.add_edges_from([(v,u)])
            return
        
        # heuristics: point it out from the one with the fewer outdegrees
        if self.can_add_edge_between_nodes(u, v):
            if self.out_degree(u) < self.out_degree(v):
                self.add_edges_from([(u,v)])
            else:
                self.add_edges_from([(v,u)])
            return True
        else: # if not possible to add, just don't add
            return False
        
    # Simple way to add a list of edges to the directed graph
    # so that it will choose the correct orientations of them and 
    # constructs the corresponding pebble graph. 
    # !Note that this might not add all the edges, only the edges that 
    # ! take part of the maximal sparse subgraph  
    def add_edges_to_maintain_out_degrees(self, edges):
        for edge in edges:
            self.add_edge_to_maintain_digraph_if_possible(edge[0], edge[1])
