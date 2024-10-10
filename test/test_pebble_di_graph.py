from pyrigi._pebble_di_graph import PebbleDiGraph

import pytest

"""
Create the most used directed graphs. Note that the edges added here are in fact directed.
If you want to add them in a way so that they necessarily keep the pebble game directions
Use other methods.
"""


def DirectedCycle(n: int, K: int, L: int) -> PebbleDiGraph:
    """Return the directed cycle on n vertices."""
    G = PebbleDiGraph(K, L)
    G.add_edges_from((i, (i + 1) % n) for i in range(n))
    return G


def DirectedPath(n: int, K: int, L: int) -> PebbleDiGraph:
    """Return the directed path from vertex 0 to vertex n-1."""
    G = PebbleDiGraph(K, L)
    G.add_edges_from((i, i + 1) for i in range(n - 1))
    return G


def OutStar(n: int, K: int, L: int) -> PebbleDiGraph:
    """Return a K_{1,n} with all edges directed out of the center."""
    G = PebbleDiGraph(K, L)
    G.add_edges_from((0, i) for i in range(1, n + 1))
    return G


def InStar(n: int, K: int, L: int) -> PebbleDiGraph:
    """Return a K_{1,n} with all edges directed into the center."""
    G = PebbleDiGraph(K, L)
    G.add_edges_from((i, 0) for i in range(1, n + 1))
    return G


"""
Unit tests
"""


def test_set_K_and_L_check_setup():
    graph = PebbleDiGraph(2, 3)
    assert graph.K == 2
    assert graph.L == 3
    with pytest.raises(ValueError):
        graph.L = 4
    with pytest.raises(TypeError):
        graph.L = (2, 3)
    with pytest.raises(ValueError):
        graph.K = 1
    with pytest.raises(ValueError):
        graph.set_K_and_L(1, 2)
    assert graph.K == 2
    assert graph.L == 3
    graph.set_K_and_L(4, 5)
    assert graph.K == 4
    assert graph.L == 5
    graph.K = 3
    assert graph.K == 3
    assert graph.L == 5
    graph.L = 0
    assert graph.L == 0
    with pytest.raises(ValueError):
        graph.K = 0
    with pytest.raises(ValueError):
        graph.L = -1
    with pytest.raises(TypeError):
        graph.L = 1.5


@pytest.mark.parametrize(
    "graphs",
    [DirectedCycle(5, 2, 3), OutStar(5, 2, 3), InStar(5, 2, 3)],
)
def test_dir_graph_edges(graphs):
    assert 5 == graphs.number_of_edges()


def test_in_degree():
    graph = InStar(2, 2, 3)
    assert graph.in_degree(0) == 2
    assert graph.in_degree(1) == 0
    assert graph.in_degree(2) == 0


def test_out_degree():
    graph = OutStar(2, 1, 1)
    assert graph.out_degree(0) == 2
    assert graph.out_degree(1) == 0
    assert graph.out_degree(2) == 0


def test_redirect_edge():
    Cycle_graph = DirectedCycle(3, 1, 0)
    Cycle_graph.point_edge_head_to([0, 1], 0)
    assert 2 == Cycle_graph.in_degree(0)
    assert 0 == Cycle_graph.out_degree(0)
    assert 2 == Cycle_graph.out_degree(1)


def test_point_edge_head_to():
    graph = DirectedPath(3, 2, 2)
    graph.point_edge_head_to((0, 1), 2)
    assert (0, 1) not in graph.edges
    assert (1, 2) in graph.edges
    assert (1, 2) in graph.edges


def test_can_add_edge_between_nodes():
    Cycle_graph = DirectedCycle(4, 2, 3)
    for i in range(4):  # we can add cyclicly to any of them in any direction
        assert Cycle_graph.can_add_edge_between_nodes(i, (i + 2) % 4)
        assert Cycle_graph.can_add_edge_between_nodes((i + 2) % 4, i)

    Disjoint_graph = PebbleDiGraph(1, 1)
    Disjoint_graph.add_edges_from([(0, 1), (2, 3)])
    assert Disjoint_graph.can_add_edge_between_nodes(0, 2)
    assert Disjoint_graph.can_add_edge_between_nodes(3, 0)
    assert Disjoint_graph.can_add_edge_between_nodes(1, 2)

    Path_graph = DirectedPath(4, 2, 1)

    assert Path_graph.can_add_edge_between_nodes(0, 3)

    with pytest.raises(ValueError):
        Path_graph.can_add_edge_between_nodes(0, "A")


def test_can_not_add_edge_between_nodes():
    Cycle_graph = DirectedCycle(4, 2, 3)

    for i in range(4):  # we can't add cyclicly to any of them in any direction
        assert not Cycle_graph.can_add_edge_between_nodes(i, (i + 1) % 4)
        assert not Cycle_graph.can_add_edge_between_nodes((i + 1) % 4, i)

    Path_graph = DirectedPath(4, 1, 1)
    for i in range(4):
        assert not Path_graph.can_add_edge_between_nodes(i, (i + 2) % 4)


def test_added_edge_between():
    Path_graph = DirectedPath(4, 2, 1)
    can_add, visited = Path_graph.added_edge_between(0, 3)
    assert can_add is True
    assert visited == {0, 3}


def test_reachable_nodes():
    graph = PebbleDiGraph(2, 3)
    graph.add_edges_to_maintain_out_degrees(
        [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (3, 4)]
    )

    # not addable edge, get base circuit
    reachable = graph.fundamental_circuit(1, 3)
    assert reachable == {0, 1, 2, 3}

    # Edge can be added: get back the two vertices
    reachable = graph.fundamental_circuit(2, 4)
    assert reachable == {2, 4}

    # Edge already exist: get back the two vertices
    reachable = graph.fundamental_circuit(2, 3)
    assert reachable == {2, 3}
