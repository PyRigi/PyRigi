from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import List, Union, Iterable

import networkx as nx
from sympy import Matrix
import math

#from pyrigi.data_type import Vertex, Edge
#from pyrigi.misc import doc_category, generate_category_tables
#from pyrigi.exception import LoopError

"""
Auxilary class for directed graph used in pebble game style algorithms.
"""

"""
TODO imports are copied, might not need all of them. 
"""

class MultiDiGraph(nx.MultiDiGraph):
    """
    Class representing a directed graph.
    All nx methods need a wrapper - to make future developments easier. 
    Extra features needed on top of nx.MultiDiGraph:
    - indegree of nodes
    """

    def in_degree(self, node):
        return super().in_degree(node)
    
    def reverse_edge(self, edge, node_to):
        # edge.get_edge_data()
        # self.remove_edge()
        # add reversed edge
        return 0
    
    # runs a DFS from v1 and v2 with K and L 
    def run_DFS(self, v1, v2, K, L):
        return 0

    # get the list of nodes accessible from v1 and v2 
    # that respects the indegree rules 
    def get_accessible_nodes(self, v1, v, K, L):
        return {} 

    # can you add the edge between the nodes v1 and v2, 
    # so that it still respects the node degrees?   
    def can_add_edge_between_nodes(self, u, v, K, L):
        def dfs(node, visited, path):
            visited.add(node)
            path.append(node)

            # Check if the stopping criteria is met
            if node != u and node != v and G.out_degree(node) < K:
                for i in range(len(path) - 1):
                    self.remove_edge(path[i], path[i + 1])
                    self.add_edge(path[i + 1], path[i])
                return True

            for neighbor in self.neighbors(node):
                found = False
                if neighbor not in visited:
                    found = dfs(neighbor, visited, path)
                if found:
                    return True

            path.pop()
            return False

        max_degree_for_u_and_v_together = 2 * K - L - 1
        while self.out_degree(u) + self.out_degree(v) > max_degree_for_u_and_v_together:
            print(self.edges())
            visited_u, visited_v = set(), set()
            path_u, path_v = [], []

            # Perform DFS from u
            found_from_u = dfs(u, visited_u, path_u)
            if found_from_u:
                continue

            # Perform DFS from v
            found_from_v = dfs(v, visited_v, path_v)
            if found_from_v:
                continue
            # not found_from_u and not found_from_v
            break 

        return self.out_degree(u) + self.out_degree(v) <= max_degree_for_u_and_v_together

    

#G = MultiDiGraph([(1, 2), (2, 3), (3, 4),(4, 1)] )

# Example usage
G = MultiDiGraph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
u, v, K, L = 2, 4, 2, 3
result = G.can_add_edge_between_nodes(u, v, K, L)
print(result)

G.add_edges_from([(2, 4)])
result = G.can_add_edge_between_nodes(u, v, K, L)
print(result)



