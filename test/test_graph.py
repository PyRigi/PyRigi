from pyrigi.graph import Graph
import pytest

@pytest.mark.slow
def test_min_max_rigid_subgraphs():
    G = Graph()
    G.add_nodes_from([0,1,2,3,4,5,'a','b'])
    G.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0),
                      (0,3), (1,4), (2,5),
                      (0,'a'), (0,'b'), ('a','b')])
    max_subgraphs = G.max_rigid_subgraphs()
    assert(
        len(max_subgraphs) == 2
        and len(max_subgraphs[0].vertices()) in [3,6] 
        and len(max_subgraphs[1].vertices()) in [3,6]
        and len(max_subgraphs[0].edges) in [3,9]
        and len(max_subgraphs[1].edges) in [3,9]
    ) 
    min_subgraphs = G.min_rigid_subgraphs()
    print(min_subgraphs[0])
    print(min_subgraphs[1])
    assert(
        len(min_subgraphs) == 2
        and len(min_subgraphs[0].vertices()) in [3,6] 
        and len(min_subgraphs[1].vertices()) in [3,6]
        and len(min_subgraphs[0].edges) in [3,9]
        and len(min_subgraphs[1].edges) in [3,9]
    ) 

def test_graph_rigidity_and_sparsity():
    G = Graph()
    G.add_nodes_from([0,1,2,3])
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0)])
    assert(
        G.is_sparse(2,3)
        and not G.is_rigid(dim=2, combinatorial=True)
    ) 
    G.add_edge(0,2)
    assert(
        G.is_tight(2,3)
        and G.is_rigid(dim=2, combinatorial=True)
        and G.is_min_rigid(dim=2, combinatorial=True)
        and not G.is_globally_rigid(dim=2)
    ) 
    G.add_edge(1,3)
    assert(
        not G.is_tight(2,3)
        and G.is_rigid(dim=2, combinatorial=True)
        and G.is_globally_rigid(dim=2)
    )
