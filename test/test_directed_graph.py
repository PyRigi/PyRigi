from pyrigi.directed_graph import MultiDiGraph
from pyrigi.exception import LoopError

import pytest
import networkx as nx

"""
Database of directed graphs:
"""

def DirectedCycle(n:int):
    """Return the directed cycle on n vertices."""
    G = MultiDiGraph()
    for i in range(n):
        G.add_edges_from([(i,(i+1)%n)])
    return G


def DirectedPath(n:int):
    """Return the directed path from vertex 0 to vertex n-1."""
    G = MultiDiGraph()
    for i in range(n-1):
        G.add_edges_from([(i, i+1)])
    return G

def OutStar(n:int):
    """Return a K_{1,n} all edges directed out of the centre."""
    G = MultiDiGraph()
    for i in range(1,n+1):
        G.add_edges_from([(0,i)])
    return G


def InStar(n:int):
    """Return a K_{1,n} all edges directed in to the centre."""
    G = MultiDiGraph()
    for i in range(1,n+1):
        G.add_edges_from([(i,0)])
    return G



"""
Unit tests
"""

@pytest.mark.parametrize(
    "graphs",
    [
        DirectedCycle(5),
        OutStar(5),
        InStar(5)
    ],
)

def test_dir_graph_edges(graphs):
    assert 5 == graphs.get_number_of_edges()

def test_redirect_edge():
    Cycle_graph = DirectedCycle(3)
    Cycle_graph.point_edge_head_to([0,1], 0)
    assert 2 == Cycle_graph.in_degree(0)
    assert 0 == Cycle_graph.out_degree(0)
    assert 2 == Cycle_graph.out_degree(1)

def test_can_add_edge_between_nodes():
    Cycle_graph = DirectedCycle(4)
    for i in range(4): # we can add cyclicly to any of them in any direction
        assert Cycle_graph.can_add_edge_between_nodes(i,(i+2)%4,2,3)
        assert Cycle_graph.can_add_edge_between_nodes((i+2)%4, i, 2, 3)

    Disjoint_graph = MultiDiGraph()
    Disjoint_graph.add_edges_from([(0,1),(2,3)])
    assert Disjoint_graph.can_add_edge_between_nodes(0,2, 1, 1)
    assert Disjoint_graph.can_add_edge_between_nodes(3,0, 1, 1)
    assert Disjoint_graph.can_add_edge_between_nodes(1,2, 1, 1)



def test_can_not_add_edge_between_nodes():
    Cycle_graph = DirectedCycle(4)
    for i in range(4): # we can't add cyclicly to any of them in any direction
        assert not Cycle_graph.can_add_edge_between_nodes(i,(i+1)%4,2,3)
        assert not Cycle_graph.can_add_edge_between_nodes((i+1)%4, i, 2, 3)

    Path_graph = DirectedPath(4)
    for i in range(4):
        assert not Path_graph.can_add_edge_between_nodes(i, (i+2)%4,1,1) 