import networkx as nx

import pyrigi.graphDB as graphs
from pyrigi.graph._constructions import constructions as graph_constructions
from test import TEST_WRAPPED_FUNCTIONS


def test_cone():
    G = graphs.Complete(5).cone()
    assert set(G.nodes) == set([0, 1, 2, 3, 4, 5]) and len(G.nodes) == 6
    G = graphs.Complete(4).cone(vertex="a")
    assert "a" in G.nodes
    G = graphs.Cycle(4).cone()
    assert G.number_of_nodes() == G.max_degree() + 1
    if TEST_WRAPPED_FUNCTIONS:
        G = graph_constructions.cone(nx.Graph(graphs.Complete(5)))
        assert set(G.nodes) == set([0, 1, 2, 3, 4, 5]) and len(G.nodes) == 6
        G = graph_constructions.cone(nx.Graph(graphs.Complete(4)), vertex="a")
        assert "a" in G.nodes
        G = graph_constructions.cone(nx.Graph(graphs.Cycle(4)))
        assert G.number_of_nodes() == max([deg for _, deg in G.degree()]) + 1
