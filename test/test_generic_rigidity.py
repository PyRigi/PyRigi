import math
from itertools import combinations

import networkx as nx
import pytest
from test_graph import (
    TEST_WRAPPED_FUNCTIONS,
    is_rigid_algorithms_all_d,
    is_rigid_algorithms_d1,
    is_rigid_algorithms_d2,
    relabeled_inc,
)

import pyrigi.generic_rigidity as generic_rigidity
import pyrigi.graphDB as graphs
import pyrigi.sparsity
from pyrigi.graph import Graph
from pyrigi.warning import RandomizedAlgorithmWarning

is_min_rigid_algorithms_all_d = ["default", "randomized"]
is_min_rigid_algorithms_d2 = is_min_rigid_algorithms_all_d + [
    "extension_sequence",
    "sparsity",
]
is_min_rigid_algorithms_d1 = is_min_rigid_algorithms_all_d + ["graphic"]


###############################################################
# is_rigid
###############################################################
@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.K66MinusPerfectMatching(), 3],
        pytest.param(graphs.Icosahedral(), 3, marks=pytest.mark.long_local),
    ]
    + [[graphs.Complete(n), d] for d in range(1, 5) for n in range(1, d + 2)],
)
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_all_d)
def test_is_rigid(graph, dim, algorithm):
    assert graph.is_rigid(dim, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert generic_rigidity.is_rigid(nx.Graph(graph), dim, algorithm=algorithm)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Diamond(),
        graphs.K33plusEdge(),
        graphs.ThreePrism(),
        graphs.ThreePrismPlusEdge(),
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.Cycle(4),
        graphs.Path(3),
        graphs.Dodecahedral(),
    ],
)
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d1)
def test_is_rigid_d1(graph, algorithm):
    assert graph.is_rigid(dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert generic_rigidity.is_rigid(nx.Graph(graph), dim=1, algorithm=algorithm)


@pytest.mark.parametrize(
    "graph",
    [
        Graph.from_vertices(range(3)),
        Graph([[0, 1], [2, 3]]),
        graphs.Cycle(3) + relabeled_inc(graphs.Cycle(3)),
    ],
)
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d1)
def test_is_not_rigid_d1(graph, algorithm):
    assert not graph.is_rigid(dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not generic_rigidity.is_rigid(
            nx.Graph(graph), dim=1, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Complete(3),
        graphs.Complete(4),
        graphs.CompleteBipartite(3, 3),
        graphs.CompleteBipartite(3, 4),
        graphs.CompleteBipartite(4, 4),
        graphs.Diamond(),
        graphs.K33plusEdge(),
        graphs.ThreePrism(),
        graphs.ThreePrismPlusEdge(),
        graphs.K66MinusPerfectMatching(),
    ],
)
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d2)
def test_is_rigid_d2(graph, algorithm):
    assert graph.is_rigid(dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert generic_rigidity.is_rigid(nx.Graph(graph), dim=2, algorithm=algorithm)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Path(3),
        graphs.Path(4),
        graphs.Cycle(4),
        graphs.Cycle(5),
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
    ],
)
@pytest.mark.parametrize("algorithm", is_rigid_algorithms_d2)
def test_not_is_rigid_d2(graph, algorithm):
    assert not graph.is_rigid(dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not generic_rigidity.is_rigid(
            nx.Graph(graph), dim=2, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "method, params",
    [
        ["is_min_rigid", [3]],
        ["is_rigid", [3]],
    ],
)
def test_is_rigid_dimension_sparsity_error(method, params):
    G = graphs.DoubleBanana()
    with pytest.raises(ValueError):
        func = getattr(G, method)
        func(*params, algorithm="sparsity")
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.raises(ValueError):
            func = getattr(generic_rigidity, method)
            func(nx.Graph(G), *params, algorithm="sparsity")


###############################################################
# is_min_rigid
###############################################################
@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.CompleteBipartite(1, 3),
        graphs.Path(3),
        Graph.from_int(102),  # a tree on 5 vertices
    ],
)
@pytest.mark.parametrize("algorithm", is_min_rigid_algorithms_d1)
def test_is_min_rigid_d1(graph, algorithm):
    assert graph.is_min_rigid(dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert generic_rigidity.is_min_rigid(
            nx.Graph(graph), dim=1, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        Graph.from_vertices(range(3)),
        Graph([[0, 1], [2, 3]]),
        graphs.Complete(3),
        graphs.Diamond(),
        graphs.K33plusEdge(),
        graphs.ThreePrism(),
        graphs.ThreePrismPlusEdge(),
        graphs.CompleteBipartite(2, 3),
        graphs.Cycle(4),
    ],
)
@pytest.mark.parametrize("algorithm", is_min_rigid_algorithms_d1)
def test_is_not_min_rigid_d1(graph, algorithm):
    assert not graph.is_min_rigid(dim=1, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not generic_rigidity.is_min_rigid(
            nx.Graph(graph), dim=1, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Complete(3),
        graphs.CompleteBipartite(3, 3),
        graphs.Diamond(),
        graphs.ThreePrism(),
    ],
)
@pytest.mark.parametrize("algorithm", is_min_rigid_algorithms_d2)
def test_is_min_rigid_d2(graph, algorithm):
    assert graph.is_min_rigid(dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert generic_rigidity.is_min_rigid(
            nx.Graph(graph), dim=2, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(4),
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.CompleteBipartite(3, 4),
        graphs.CompleteBipartite(4, 4),
        graphs.Cycle(4),
        graphs.Cycle(5),
        graphs.K33plusEdge(),
        graphs.Path(3),
        graphs.Path(4),
        graphs.ThreePrismPlusEdge(),
        pytest.param(graphs.Dodecahedral(), marks=pytest.mark.long_local),
    ],
)
@pytest.mark.parametrize("algorithm", is_min_rigid_algorithms_d2)
def test_is_not_min_rigid_d2(graph, algorithm):
    assert not graph.is_min_rigid(dim=2, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not generic_rigidity.is_min_rigid(
            nx.Graph(graph), dim=2, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        Graph.from_vertices_and_edges([0, 1], [[0, 1]]),
        graphs.Complete(3),
        graphs.Complete(4),
        graphs.Octahedral(),
        pytest.param(graphs.K66MinusPerfectMatching(), marks=pytest.mark.slow_main),
        pytest.param(graphs.Icosahedral(), marks=pytest.mark.long_local),
    ],
)
@pytest.mark.parametrize("algorithm", is_min_rigid_algorithms_all_d)
def test_is_min_rigid_d3(graph, algorithm):
    assert graph.is_min_rigid(dim=3, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert generic_rigidity.is_min_rigid(
            nx.Graph(graph), dim=3, algorithm=algorithm
        )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(5),
        graphs.CubeWithDiagonal(),
        graphs.CompleteBipartite(5, 5),
        graphs.DoubleBanana(dim=3),
        pytest.param(graphs.ThreeConnectedR3Circuit(), marks=pytest.mark.long_local),
        graphs.Dodecahedral(),
    ],
)
@pytest.mark.parametrize("algorithm", is_min_rigid_algorithms_all_d)
def test_is_not_min_rigid_d3(graph, algorithm):
    assert not graph.is_min_rigid(dim=3, algorithm=algorithm)
    if TEST_WRAPPED_FUNCTIONS:
        assert not generic_rigidity.is_min_rigid(
            nx.Graph(graph), dim=3, algorithm=algorithm
        )


###############################################################
# rigid_components
###############################################################
@pytest.mark.parametrize(
    "graph, components, dim",
    [
        [graphs.Path(6), [[0, 1, 2, 3, 4, 5]], 1],
        [graphs.Path(3) + relabeled_inc(graphs.Path(3), 3), [[0, 1, 2], [3, 4, 5]], 1],
        [graphs.Path(5), [[i, i + 1] for i in range(4)], 2],
        [
            graphs.CompleteBipartite(3, 3) + Graph([(0, "a"), (0, "b"), ("a", "b")]),
            [[0, "a", "b"], [0, 1, 2, 3, 4, 5]],
            2,
        ],
        [graphs.Cycle(3) + relabeled_inc(graphs.Cycle(3)), [[0, 1, 2], [3, 4, 5]], 2],
        [
            graphs.Cycle(3) + relabeled_inc(graphs.Cycle(3), 2),
            [[0, 1, 2], [2, 3, 4]],
            2,
        ],
        [graphs.Complete(3) + Graph.from_vertices([3]), [[0, 1, 2], [3]], 2],
        [graphs.ThreePrism(), [[i for i in range(6)]], 2],
        [graphs.DoubleBanana(), [[0, 1, 2, 3, 4], [0, 1, 5, 6, 7]], 3],
        [
            graphs.Diamond() + relabeled_inc(graphs.Diamond()) + Graph([[2, 6]]),
            [[0, 1, 2, 3], [4, 5, 6, 7], [2, 6]],
            2,
        ],
        [
            # graphs.ThreeConnectedR3Circuit with 0 removed
            # and then each vertex label decreased by 1
            Graph.from_int(64842845087398392615),
            [[0, 1, 2, 3], [0, 9, 10, 11], [3, 4, 5, 6], [6, 7, 8, 9]],
            2,
        ],
    ],
)
def test_rigid_components(graph, components, dim):
    def to_sets(comps):
        return set([frozenset(comp) for comp in comps])

    comps_set = to_sets(components)

    if dim == 1:
        assert (
            to_sets(graph.rigid_components(dim=dim, algorithm="graphic")) == comps_set
        )
        if TEST_WRAPPED_FUNCTIONS:
            assert (
                to_sets(
                    generic_rigidity.rigid_components(
                        nx.Graph(graph), dim=dim, algorithm="graphic"
                    )
                )
                == comps_set
            )
    elif dim == 2:
        assert to_sets(graph.rigid_components(dim=dim, algorithm="pebble")) == comps_set
        if TEST_WRAPPED_FUNCTIONS:
            assert (
                to_sets(
                    generic_rigidity.rigid_components(
                        nx.Graph(graph), dim=dim, algorithm="pebble"
                    )
                )
                == comps_set
            )
        if graph.number_of_nodes() <= 8:  # since it runs through all subgraphs
            assert (
                to_sets(graph.rigid_components(dim=dim, algorithm="subgraphs-pebble"))
                == comps_set
            )
            if TEST_WRAPPED_FUNCTIONS:
                assert (
                    to_sets(
                        generic_rigidity.rigid_components(
                            nx.Graph(graph), dim=dim, algorithm="subgraphs-pebble"
                        )
                    )
                    == comps_set
                )

    # randomized algorithm is tested for all dimensions for graphs
    # with at most 8 vertices (since it runs through all subgraphs)
    if graph.number_of_nodes() <= 8:
        assert (
            to_sets(graph.rigid_components(dim=dim, algorithm="randomized"))
            == comps_set
        )
        assert (
            to_sets(graph.rigid_components(dim=dim, algorithm="numerical")) == comps_set
        )
        if TEST_WRAPPED_FUNCTIONS:
            assert (
                to_sets(
                    generic_rigidity.rigid_components(
                        nx.Graph(graph), dim=dim, algorithm="randomized"
                    )
                )
                == comps_set
            )
            assert (
                to_sets(
                    generic_rigidity.rigid_components(
                        nx.Graph(graph), dim=dim, algorithm="numerical"
                    )
                )
                == comps_set
            )


@pytest.mark.parametrize(
    "graph",
    [
        Graph(nx.gnp_random_graph(20, 0.1)),
        Graph(nx.gnm_random_graph(30, 62)),
        pytest.param(Graph(nx.gnm_random_graph(25, 46)), marks=pytest.mark.slow_main),
        pytest.param(Graph(nx.gnm_random_graph(40, 80)), marks=pytest.mark.slow_main),
        pytest.param(
            Graph(nx.gnm_random_graph(100, 230)), marks=pytest.mark.long_local
        ),
        pytest.param(
            Graph(nx.gnm_random_graph(100, 190)), marks=pytest.mark.long_local
        ),
    ],
)
def test_rigid_components_pebble_random_graphs(graph):
    rigid_components = graph.rigid_components(dim=2, algorithm="pebble")

    # Check that all components are rigid
    for c in rigid_components:
        new_graph = graph.subgraph(c)
        assert new_graph.is_rigid(dim=2, algorithm="sparsity")

    # Check that vertex-pairs that are not in a component are not in a rigid component
    # check every vertex pairs in the graph
    for u, v in list(combinations(graph.nodes, 2)):
        # if there is no component from rigid components that contains u and v together
        # the edge u,v can be added
        if not any([u in c and v in c for c in rigid_components]):
            pebble_digraph = pyrigi.sparsity._get_pebble_digraph(graph, 2, 3)
            assert pebble_digraph.can_add_edge_between_vertices(u, v)


###############################################################
# max_rigid_dimension
###############################################################
@pytest.mark.parametrize(
    "graph, k",
    [
        [graphs.Cycle(4), 1],
        [graphs.Diamond(), 2],
        [graphs.Complete(4), math.inf],
        [Graph([(0, 1), (2, 3)]), 0],
        [graphs.Complete(5), math.inf],
        [graphs.Frustum(3), 2],
        [graphs.ThreePrism(), 2],
        [graphs.DoubleBanana(), 2],
        [graphs.CompleteMinusOne(5), 3],
        [graphs.Octahedral(), 3],
        [graphs.K66MinusPerfectMatching(), 3],
    ],
)
def test_max_rigid_dimension(graph, k):
    assert graph.max_rigid_dimension() == k
    assert graph.max_rigid_dimension(algorithm="numerical") == k
    if TEST_WRAPPED_FUNCTIONS:
        assert generic_rigidity.max_rigid_dimension(nx.Graph(graph)) == k
        assert (
            generic_rigidity.max_rigid_dimension(nx.Graph(graph), algorithm="numerical")
            == k
        )


def test_max_rigid_dimension_warning():
    graph = graphs.K66MinusPerfectMatching()
    with pytest.warns(RandomizedAlgorithmWarning):
        graph.max_rigid_dimension()
        graph.max_rigid_dimension(algorithm="numerical")
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.warns(RandomizedAlgorithmWarning):
            generic_rigidity.max_rigid_dimension(nx.Graph(graph))
            generic_rigidity.max_rigid_dimension(nx.Graph(graph), algorithm="numerical")
