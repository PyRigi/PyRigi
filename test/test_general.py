from pyrigi.graph import Graph
from pyrigi.framework import Framework


def test_vertex_addition():
    G = Graph()
    G.add_node(0)
    F = Framework(G, {0: [1, 2]})
    F_ = Framework()
    F.add_vertex([1, 1], 1)
    F.add_vertex([0, 0], 2)
    F_.add_vertices([[1, 2], [1, 1], [0, 0]])
    assert (
        F.realization == F_.realization
        and F.graph().vertices() == F_.graph().vertices()
        and F.dim == F_.dim
    )
    assert (
        list(F.graph().vertices()) == [0, 1, 2]
        and len(F.graph().edges()) == 0
    )
    F.add_edge([0, 1])

def test_minimal_maximal_rigid_subgraphs():
    G = Graph()
    G.add_nodes_from([0,1,2,3,4,5,6,'a','b'])
    G.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,0), 
                      (0,3), (1,4), (2,5),
                      (0,'a'), (0,'b'), ('a','b')])
    max_subgraphs = G.maximal_rigid_subgraphs()
    assert max_subgraphs == [G.subgraph([0,1,2,3,4,5,6]), G.subgraph([0,'a','b'])]
    min_subgraphs = G.minimal_rigid_subgraphs()
    print(min_subgraphs)
    assert min_subgraphs == [G.subgraph([0,1,2,3,4,5,6]), G.subgraph([0,'a','b'])]