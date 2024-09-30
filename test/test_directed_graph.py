from pyrigi.directed_graph import MultiDiGraph
from pyrigi.exception import LoopError

import pytest

"""
Database of directed graphs:
"""
def DirectedCycle(n: int) -> MultiDiGraph:
    """Return the directed cycle on n vertices."""
    G = MultiDiGraph()
    G.add_edges_from((i, (i + 1) % n) for i in range(n))
    return G

def DirectedPath(n: int) -> MultiDiGraph:
    """Return the directed path from vertex 0 to vertex n-1."""
    G = MultiDiGraph()
    G.add_edges_from((i, i + 1) for i in range(n - 1))
    return G

def OutStar(n: int) -> MultiDiGraph:
    """Return a K_{1,n} with all edges directed out of the center."""
    G = MultiDiGraph()
    G.add_edges_from((0, i) for i in range(1, n + 1))
    return G

def InStar(n: int) -> MultiDiGraph:
    """Return a K_{1,n} with all edges directed into the center."""
    G = MultiDiGraph()
    G.add_edges_from((i, 0) for i in range(1, n + 1))
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


def test_in_degree():
    graph = InStar(2)
    assert graph.in_degree(0) == 2
    assert graph.in_degree(1) == 0
    assert graph.in_degree(2) == 0

def test_out_degree():    
    graph = OutStar(2)
    assert graph.out_degree(0) == 2
    assert graph.out_degree(1) == 0
    assert graph.out_degree(2) == 0

def test_redirect_edge():
    Cycle_graph = DirectedCycle(3)
    Cycle_graph.point_edge_head_to([0,1], 0)
    assert 2 == Cycle_graph.in_degree(0)
    assert 0 == Cycle_graph.out_degree(0)
    assert 2 == Cycle_graph.out_degree(1)

def test_point_edge_head_to():
    graph = DirectedPath(3)
    graph.point_edge_head_to((0, 1), 2)
    assert (0, 1) not in graph.edges
    assert (1, 2) in graph.edges
    assert (1, 2) in graph.edges

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

    Path_graph = DirectedPath(4)
    assert Path_graph.can_add_edge_between_nodes(0, 3, 2, 1)


def test_can_not_add_edge_between_nodes():
    Cycle_graph = DirectedCycle(4)
    for i in range(4): # we can't add cyclicly to any of them in any direction
        assert not Cycle_graph.can_add_edge_between_nodes(i,(i+1)%4,2,3)
        assert not Cycle_graph.can_add_edge_between_nodes((i+1)%4, i, 2, 3)

    Path_graph = DirectedPath(4)
    for i in range(4):
        assert not Path_graph.can_add_edge_between_nodes(i, (i+2)%4,1,1) 

def test_added_edge_between():
    Path_graph = DirectedPath(4)
    can_add, visited = Path_graph.added_edge_between(0, 3, 2, 1)
    assert can_add is True
    assert visited == {0, 3}

def test_reachable_nodes():
    graph = MultiDiGraph()
    graph.add_edges_from([(0,1), (1,2), (2,3),(3,0), (0,2), (3,4)])
    reachable = graph.reachable_nodes(0, 1, 2, 3)
    assert reachable == {0, 1, 2, 3}
