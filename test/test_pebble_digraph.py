from pyrigi._pebble_digraph import PebbleDiGraph

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


def test_redirect_edge_to_head():
    graph = DirectedPath(3, 2, 2)

    # Turn edge around
    graph.redirect_edge_to_head((0, 1), 0)
    assert (0, 1) not in graph.edges
    assert (1, 2) in graph.edges
    assert (1, 0) in graph.edges

    Cycle_graph = DirectedCycle(3, 1, 0)
    # Turn edge around
    Cycle_graph.redirect_edge_to_head([0, 1], 0)
    assert 2 == Cycle_graph.in_degree(0)
    assert 0 == Cycle_graph.out_degree(0)
    assert 2 == Cycle_graph.out_degree(1)

    # only possible if the proposed head is part of the edge itself
    Cycle_graph = DirectedCycle(3, 1, 0)
    # Does nothing
    Cycle_graph.redirect_edge_to_head([0, 1], 2)
    assert 1 == Cycle_graph.in_degree(0)
    assert 1 == Cycle_graph.out_degree(0)
    assert 1 == Cycle_graph.out_degree(1)

    graph = DirectedPath(3, 2, 2)
    # Does nothing
    graph.redirect_edge_to_head((0, 1), "A")
    assert (0, 1) in graph.edges
    assert (1, 2) in graph.edges
    assert 2 == len(graph.edges)


def test_can_add_edge_between_nodes():
    Cycle_graph = DirectedCycle(4, 2, 3)
    for i in range(4):  # we can add cyclicly to any of them in any direction
        assert Cycle_graph.can_add_edge_between_vertices(i, (i + 2) % 4)
        assert Cycle_graph.can_add_edge_between_vertices((i + 2) % 4, i)

    Disjoint_graph = PebbleDiGraph(1, 1)
    Disjoint_graph.add_edges_from([(0, 1), (2, 3)])
    assert Disjoint_graph.can_add_edge_between_vertices(0, 2)
    assert Disjoint_graph.can_add_edge_between_vertices(3, 0)
    assert Disjoint_graph.can_add_edge_between_vertices(1, 2)

    Path_graph = DirectedPath(4, 2, 1)

    assert Path_graph.can_add_edge_between_vertices(0, 3)

    with pytest.raises(ValueError):
        Path_graph.can_add_edge_between_vertices(0, "A")


def test_can_not_add_edge_between_nodes():
    Cycle_graph = DirectedCycle(4, 2, 3)

    for i in range(4):  # we can't add cyclicly to any of them in any direction
        assert not Cycle_graph.can_add_edge_between_vertices(i, (i + 1) % 4)
        assert not Cycle_graph.can_add_edge_between_vertices((i + 1) % 4, i)

    Path_graph = DirectedPath(4, 1, 1)
    for i in range(4):
        assert not Path_graph.can_add_edge_between_vertices(i, (i + 2) % 4)


def test_add_independent_edge():
    Path_graph = DirectedPath(4, 2, 1)
    fund_circle = Path_graph.fundamental_circuit(0, 3)
    assert fund_circle is None


def test_nodes_in_fundamental_circuit():
    graph = PebbleDiGraph(2, 3)
    graph.add_edges_maintaining_digraph(
        [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (3, 4)]
    )

    # not addable edge, get base circuit
    fund_circuit = graph.fundamental_circuit(1, 3)
    assert fund_circuit == {0, 1, 2, 3}

    # Edge can be added: fundamental cycle is none
    fund_circuit = graph.fundamental_circuit(2, 4)
    assert fund_circuit is None

    # Edge already exist: get back the two vertices
    fund_circuit = graph.fundamental_circuit(2, 3)
    assert fund_circuit == {2, 3}


def test_nodes_in_fundamental_circuit_with_2_2_graph():
    graph = PebbleDiGraph(2, 2)
    # double the path
    graph.add_edges_maintaining_digraph(
        [(0, 1), (0, 1), (1, 2), (1, 2), (2, 3), (2, 3)]
    )

    # not addable edge, get base circuit
    fund_circuit = graph.fundamental_circuit(0, 3)
    assert fund_circuit == {0, 1, 2, 3}

    # not addable edge, get base circuit, but not all vertices
    fund_circuit = graph.fundamental_circuit(2, 3)
    assert fund_circuit == {2, 3}
