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

    def get_number_of_edges(self):
        return len(super().edges)

    def in_degree(self, node):
        return super().in_degree(node)
    
    def out_degree(self, node):
        return super().out_degree(node)
    
    def point_edge_head_to(self, edge, node_to):
        # placeholder 
        tail = edge[0]
        head = edge[1]
        self.remove_edge(tail, head)
        self.add_edge(head, node_to)

    def added_edge_between(self, u, v, K, L):
        def dfs(node, visited, edge_path, current_edge = None):
            visited.add(node)
            if current_edge:
                edge_path.append(current_edge)

            # Check if the stopping criteria is met
            if node != u and node != v and self.out_degree(node) < K:
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

        max_degree_for_u_and_v_together = 2 * K - L - 1
        visited_nodes = {u,v}

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
            break 

        return (self.out_degree(u) + self.out_degree(v) <= max_degree_for_u_and_v_together), visited_nodes

    # get the list of nodes accessible from u and v 
    # that respects the outdegree rules
    def reachable_nodes(self, u, v, K, L):
        can_add_edge, reached_vertices = self.added_edge_between(u,v,K,L)
        if can_add_edge:
            return {u,v}
        else:
            return reached_vertices

    # can you add the edge between the nodes u and v, 
    # so that it still respects the node degrees?   
    def can_add_edge_between_nodes(self, u, v, K, L):
        can_add_edge, reached_vertices = self.added_edge_between(u,v,K,L)
        return can_add_edge

# Example usage
G = MultiDiGraph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
u, v, K, L = 2, 4, 2, 3
result = G.can_add_edge_between_nodes(u, v, K, L)
print(result)

result = G.can_add_edge_between_nodes(1, 3, K, L)
print(result)

G.add_edges_from([(2, 4)])

result = G.can_add_edge_between_nodes(u, v, K, L)
print(result)
result = G.can_add_edge_between_nodes(1, 3, K, L)
print(result)




