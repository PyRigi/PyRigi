import networkx as nx
import pytest
from test_graph import TEST_WRAPPED_FUNCTIONS, read_sparsity

import pyrigi.graphDB as graphs
import pyrigi.matroidal_rigidity as matroidal_rigidity
from pyrigi.graph import Graph
from pyrigi.warning import RandomizedAlgorithmWarning

Rd_algorithms_all_d = ["default", "randomized"]
Rd_algorithms_d1 = Rd_algorithms_all_d + ["graphic"]
Rd_algorithms_d2 = Rd_algorithms_all_d + ["sparsity"]

is_Rd_closed_algorithms_all_d = ["default", "randomized"]
is_Rd_closed_algorithms_d1 = is_Rd_closed_algorithms_all_d + ["graphic"]
is_Rd_closed_algorithms_d2 = is_Rd_closed_algorithms_all_d + ["pebble"]


###############################################################
# is_Rd_circuit
###############################################################
@pytest.mark.parametrize(
    "graph",
    [graphs.Cycle(n) for n in range(3, 7)]
    + [Graph.from_vertices_and_edges([0, 1, 2, 3], [[0, 1], [0, 2], [1, 2]])],
)
@pytest.mark.parametrize("algorithm", Rd_algorithms_d1)
def test_is_Rd_circuit_d1(graph, algorithm):
    assert graph.is_Rd_circuit(dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert matroidal_rigidity.is_Rd_circuit(
            nx.Graph(graph), dim=1, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        Graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]),
        Graph([(0, 1), (2, 3)]),
        graphs.Complete(2),
        graphs.Diamond(),
        graphs.K33plusEdge(),
        graphs.ThreePrism(),
        graphs.ThreePrismPlusEdge(),
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.Path(3),
        graphs.K66MinusPerfectMatching(),
    ],
)
@pytest.mark.parametrize("algorithm", Rd_algorithms_d1)
def test_is_not_Rd_circuit_d1(graph, algorithm):
    assert not graph.is_Rd_circuit(dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not matroidal_rigidity.is_Rd_circuit(
            nx.Graph(graph), dim=1, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(4),
        graphs.ThreePrismPlusEdge(),
        graphs.K33plusEdge(),
        Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (3, 0), (3, 1), (2, 4)]),
        pytest.param(read_sparsity("K4"), marks=pytest.mark.slow_main),
        pytest.param(read_sparsity("circle_5_8"), marks=pytest.mark.slow_main),
        pytest.param(read_sparsity("circle_10_18"), marks=pytest.mark.slow_main),
        pytest.param(read_sparsity("circle_20_38"), marks=pytest.mark.slow_main),
        pytest.param(read_sparsity("circle_30_58"), marks=pytest.mark.slow_main),
    ]
    + [graphs.Wheel(n) for n in range(3, 7)],
)
@pytest.mark.parametrize("algorithm", Rd_algorithms_d2)
def test_is_Rd_circuit_d2(graph, algorithm):
    assert graph.is_Rd_circuit(dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert matroidal_rigidity.is_Rd_circuit(
            nx.Graph(graph), dim=2, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Complete(5),
        graphs.Diamond(),
        graphs.ThreePrism(),
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.Path(3),
        graphs.Cycle(4),
        graphs.K66MinusPerfectMatching(),
        Graph(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 0),
                (0, 3),
                (0, 2),
                (1, 3),
                (3, 5),
            ]
        ),
        graphs.Complete(4) + Graph([(3, 4), (4, 5), (5, 6), (6, 3), (3, 5), (4, 6)]),
        pytest.param(read_sparsity("not_circle_5_7"), marks=pytest.mark.slow_main),
        pytest.param(read_sparsity("not_circle_10_18"), marks=pytest.mark.slow_main),
        pytest.param(read_sparsity("not_circle_20_39"), marks=pytest.mark.slow_main),
        pytest.param(read_sparsity("not_circle_30_58"), marks=pytest.mark.slow_main),
    ],
)
@pytest.mark.parametrize("algorithm", Rd_algorithms_d2)
def test_is_not_Rd_circuit_d2(graph, algorithm):
    assert not graph.is_Rd_circuit(dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not matroidal_rigidity.is_Rd_circuit(
            nx.Graph(graph), dim=2, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(5),
        pytest.param(graphs.ThreeConnectedR3Circuit(), marks=pytest.mark.slow_main),
        graphs.DoubleBanana(),
    ],
)
@pytest.mark.parametrize("algorithm", Rd_algorithms_all_d)
def test_is_Rd_circuit_d3(graph, algorithm):
    assert graph.is_Rd_circuit(dim=3, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert matroidal_rigidity.is_Rd_circuit(
            nx.Graph(graph), dim=3, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Path(5),
        graphs.Complete(4),
        graphs.Cycle(6),
        graphs.ThreePrism(),
        graphs.K33plusEdge(),
    ],
)
@pytest.mark.parametrize("algorithm", Rd_algorithms_all_d)
def test_is_not_Rd_circuit_d3(graph, algorithm):
    assert not graph.is_Rd_circuit(dim=3, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not matroidal_rigidity.is_Rd_circuit(
            nx.Graph(graph), dim=3, algorithm=algorithm
        )


###############################################################
# is_Rd_closed
###############################################################
@pytest.mark.parametrize(
    "graph, dim",
    [[graphs.Complete(n + 1), n] for n in range(1, 5)]
    + [
        pytest.param(
            graphs.DoubleBanana() + Graph([(0, 1)]), 3, marks=pytest.mark.slow_main
        )
    ],
)
@pytest.mark.parametrize("algorithm", is_Rd_closed_algorithms_all_d)
def test_is_Rd_closed(graph, dim, algorithm):
    assert graph.is_Rd_closed(dim=dim, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert matroidal_rigidity.is_Rd_closed(
            nx.Graph(graph), dim=dim, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.Octahedral(), 3],
        pytest.param(graphs.DoubleBanana(), 3, marks=pytest.mark.slow_main),
    ],
)
@pytest.mark.parametrize("algorithm", is_Rd_closed_algorithms_all_d)
def test_is_not_Rd_closed(graph, dim, algorithm):
    assert not graph.is_Rd_closed(dim=dim, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not matroidal_rigidity.is_Rd_closed(
            nx.Graph(graph), dim=dim, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        Graph([(0, 1), (2, 3)]),
        Graph([(0, 1), (1, 2), (0, 2), (3, 4)]),
    ],
)
@pytest.mark.parametrize("algorithm", is_Rd_closed_algorithms_d1)
def test_is_Rd_closed_d1(graph, algorithm):
    assert graph.is_Rd_closed(dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert matroidal_rigidity.is_Rd_closed(
            nx.Graph(graph), dim=1, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Path(4),
    ],
)
@pytest.mark.parametrize("algorithm", is_Rd_closed_algorithms_d1)
def test_is_not_Rd_closed_d1(graph, algorithm):
    assert not graph.is_Rd_closed(dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not matroidal_rigidity.is_Rd_closed(
            nx.Graph(graph), dim=1, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(4),
        graphs.Cycle(4),
        Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 4), (2, 5)]),
    ],
)
@pytest.mark.parametrize("algorithm", is_Rd_closed_algorithms_d2)
def test_is_Rd_closed_d2(graph, algorithm):
    assert graph.is_Rd_closed(dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert matroidal_rigidity.is_Rd_closed(
            nx.Graph(graph), dim=2, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.ThreePrism(),
        graphs.ThreePrismPlusEdge(),
        pytest.param(graphs.K66MinusPerfectMatching(), marks=pytest.mark.slow_main),
    ],
)
@pytest.mark.parametrize("algorithm", is_Rd_closed_algorithms_d2)
def test_is_not_Rd_closed_d2(graph, algorithm):
    assert not graph.is_Rd_closed(dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not matroidal_rigidity.is_Rd_closed(
            nx.Graph(graph), dim=2, algorithm=algorithm
        )


###############################################################
# is_Rd_(in)dependent
###############################################################
@pytest.mark.parametrize(
    "graph",
    [
        graphs.Diamond(),
        graphs.K33plusEdge(),
        graphs.ThreePrism(),
        graphs.ThreePrismPlusEdge(),
        graphs.CompleteBipartite(2, 3),
        graphs.K66MinusPerfectMatching(),
    ]
    + [graphs.Cycle(n) for n in range(3, 7)],
)
@pytest.mark.parametrize("algorithm", Rd_algorithms_d1)
def test_is_Rd_dependent_d1(graph, algorithm):
    assert graph.is_Rd_dependent(dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not matroidal_rigidity.is_Rd_independent(
            nx.Graph(graph), dim=1, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.CompleteBipartite(1, 3),
        graphs.Path(3),
    ],
)
@pytest.mark.parametrize("algorithm", Rd_algorithms_d1)
def test_is_Rd_independent_d1(graph, algorithm):
    assert graph.is_Rd_independent(dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert matroidal_rigidity.is_Rd_independent(
            nx.Graph(graph), dim=1, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(4),
        graphs.ThreePrismPlusEdge(),
        graphs.K33plusEdge(),
        graphs.Complete(5),
        graphs.CompleteBipartite(3, 4),
        graphs.K66MinusPerfectMatching(),
    ]
    + [graphs.Wheel(n) for n in range(3, 8)],
)
@pytest.mark.parametrize("algorithm", Rd_algorithms_d2)
def test_is_Rd_dependent_d2(graph, algorithm):
    assert graph.is_Rd_dependent(dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not matroidal_rigidity.is_Rd_independent(
            nx.Graph(graph), dim=2, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Diamond(),
        graphs.ThreePrism(),
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.CompleteBipartite(3, 3),
        graphs.Path(3),
        graphs.Cycle(4),
    ],
)
@pytest.mark.parametrize("algorithm", Rd_algorithms_d2)
def test_is_Rd_independent_d2(graph, algorithm):
    assert graph.is_Rd_independent(dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert matroidal_rigidity.is_Rd_independent(
            nx.Graph(graph), dim=2, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [graphs.Complete(5), graphs.ThreeConnectedR3Circuit(), graphs.DoubleBanana()],
)
@pytest.mark.parametrize("algorithm", Rd_algorithms_all_d)
def test_is_Rd_dependent_d3(graph, algorithm):
    assert graph.is_Rd_dependent(dim=3, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not matroidal_rigidity.is_Rd_independent(
            nx.Graph(graph), dim=3, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(4),
        graphs.Cycle(6),
        graphs.ThreePrism(),
        graphs.K33plusEdge(),
        graphs.K66MinusPerfectMatching(),
        graphs.Path(5),
    ],
)
@pytest.mark.parametrize("algorithm", Rd_algorithms_all_d)
def test_is_Rd_independent_d3(graph, algorithm):
    assert graph.is_Rd_independent(dim=3, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert matroidal_rigidity.is_Rd_independent(
            nx.Graph(graph), dim=3, algorithm=algorithm
        )


def test_is_Rd_independent_d3_warning():
    G = graphs.K33plusEdge()
    with pytest.warns(RandomizedAlgorithmWarning):
        G.is_Rd_independent(dim=3)
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.warns(RandomizedAlgorithmWarning):
            matroidal_rigidity.is_Rd_independent(nx.Graph(G), dim=3)
