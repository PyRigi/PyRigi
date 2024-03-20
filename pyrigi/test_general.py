from pyrigi.graph import Graph
from pyrigi.framework import Framework

def test_vertex_addition():
    G = Graph()
    G.add_node(0)
    F = Framework(G, {0:[1,2]})
    F_ = Framework(G, {0:[1,2]})
    F.add_vertex([1,1], 1)
    F.add_vertex([0,0], 2)
    F_.add_vertices([[1,1], [0,0]])
    assert F.realization==F_.realization and F.graph().vertices()==F_.graph().vertices() and F.dim==F_.dim
