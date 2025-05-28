import networkx as nx


def is_isomorphic_graph_list(list1: list[nx.Graph], list2: list[nx.Graph]) -> bool:
    """
    Return whether two lists of graphs are the same up to permutations
    and graph isomorphism.
    """
    if len(list1) != len(list2):
        return False
    for graph1 in list1:
        count_copies = 0
        for grapht in list1:
            if nx.is_isomorphic(graph1, grapht):
                count_copies += 1
        count_found = 0
        for graph2 in list2:
            if nx.is_isomorphic(graph1, graph2):
                count_found += 1
                if count_found == count_copies:
                    break
        else:
            return False
    return True
