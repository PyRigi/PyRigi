from pyrigi.graph import Graph

def test_minimal_maximal_rigid_subgraphs():
    G = Graph()
    G.add_nodes_from([0,1,2,3,4,5,6,'a','b'])
    G.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,0),
                      (0,3), (1,4), (2,5),
                      (0,'a'), (0,'b'), ('a','b')])
    max_subgraphs = G.maximal_rigid_subgraphs()
    assert max_subgraphs == [G.subgraph([0,1,2,3,4,5,6]), G.subgraph([0,'a','b'])]
    min_subgraphs = G.minimal_rigid_subgraphs()
    assert min_subgraphs == [G.subgraph([0,1,2,3,4,5,6]), G.subgraph([0,'a','b'])]