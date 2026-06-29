import networkx as nx

import pyrigi.graphDB as graphs
from pyrigi.graph._constructions import constructions as graph_constructions
from pyrigi.graph._general import max_degree


def test_cone():
    G = graph_constructions.cone(nx.Graph(graphs.Complete(5)))
    assert set(G.nodes) == set([0, 1, 2, 3, 4, 5]) and len(G.nodes) == 6
    G = graph_constructions.cone(nx.Graph(graphs.Complete(4)), vertex="a")
    assert "a" in G.nodes
    G = graph_constructions.cone(nx.Graph(graphs.Cycle(4)))
    assert G.number_of_nodes() == max_degree(G) + 1
