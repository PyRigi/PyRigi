import itertools
import random

import networkx as nx
import pytest

from pyrigi.graph._sparsity._flow import (
    _max_flow,
    _k_indegree_bounded_orientation,
    _build_flow_network,
    _indegree_F,
    _is_arc_joining_F,
    _build_auxiliary_graph,
    _augment_F,
    _search_augmenting_path,
    _k_forest_packing,
    _k_forest_decomposition,
    _k_bounded_forest_decomposition,
    _eta_rooted_term_connectivity,
    _eta_rooted_arc_connectivity,
    _build_rooted_digraph,
    _check_terminal_superset_sparsity,
    _check_sparsity_for_l_less_than_k,
    _check_sparsity_for_l_between_k_and_2k,
    _check_sparsity_for_l_equal_2k,
    _check_sparsity_for_l_between_2k_and_3k,
)


def test_max_flow():
    D = nx.DiGraph()
    D.add_edge("s", "a", capacity=1)
    D.add_edge("a", "t", capacity=1)
    x, S = _max_flow(D, "s", "t", "capacity")
    assert x["s"]["a"] == 1
    assert x["a"]["t"] == 1
    assert S == {"a"}


def test_build_flow_network():
    G = nx.path_graph(3)  # edges (0,1), (1,2)
    N = _build_flow_network(G, 2)
    # s, t, 3 vertices, 2 edge nodes
    assert N.number_of_nodes() == 2 + 3 + 2
    # per edge: s->z, z->u, z->v; per vertex: v->t
    assert N.number_of_edges() == 3 * 2 + 3
    for v in G.nodes():
        assert N[v]["t"]["capacity"] == 2
    for e in G.edges():
        assert N["s"][f"z_{e}"]["capacity"] == 1


@pytest.mark.parametrize(
    "G, k",
    [
        (nx.path_graph(4), 1),
        (nx.cycle_graph(3), 1),
        (nx.complete_graph(4), 2),
        (nx.complete_graph(3), 2),
    ],
)
def test_orientation_exists(G, k):
    X, D = _k_indegree_bounded_orientation(G, k)
    assert X is None
    assert D.number_of_edges() == G.number_of_edges()
    assert all(D.in_degree(v) <= k for v in D.nodes())
    for u, v in G.edges():
        assert D.has_edge(u, v) != D.has_edge(v, u)


def test_orientation_too_many_edges():
    X, D = _k_indegree_bounded_orientation(nx.complete_graph(4), 1)
    assert D is None
    assert X == set(range(4))


def test_orientation_does_not_exist_with_slack():
    # K4 has 6 edges > 4 vertices, but isolated vertices make the
    # trivial edge-count check pass, so the flow must detect the failure.
    G = nx.complete_graph(4)
    G.add_nodes_from([4, 5, 6])
    X, D = _k_indegree_bounded_orientation(G, 1)
    assert D is None
    assert X is not None


def test_indegree_F():
    F = {(0, 1): 0, (1, 2): 0, (0, 3): 1, (1, 3): 1, (2, 3): None}
    assert _indegree_F(0, F) == 0
    assert _indegree_F(1, F) == 1
    assert _indegree_F(2, F) == 1
    assert _indegree_F(3, F) == 2


def test_is_arc_joining_F():
    F = {(0, 1): 0, (1, 2): 0, (0, 3): 1, (1, 3): 1, (2, 3): None}
    assert not _is_arc_joining_F((0, 1), F, 0)
    assert not _is_arc_joining_F((1, 2), F, 0)
    assert not _is_arc_joining_F((0, 3), F, 1)
    assert not _is_arc_joining_F((1, 3), F, 1)
    assert _is_arc_joining_F((2, 3), F, 0)
    assert not _is_arc_joining_F((0, 2), F, 0)
    assert _is_arc_joining_F((0, 2), F, 1)


def test_build_auxiliary_graph_nodes():
    G = nx.DiGraph([(0, 1), (1, 2)])
    F = {(0, 1): None, (1, 2): None}
    D = _build_auxiliary_graph(G, F, {}, 1)
    assert set(D.nodes()) == {0, 1, 2, (0, 1), (1, 2), "t"}
    assert D.has_edge((0, 1), "t")
    assert D.has_edge((1, 2), "t")
    assert D.has_edge(1, (0, 1))
    assert D.has_edge(2, (1, 2))


def test_augment_F_shifts_labels_along_path():
    e1, e2, e3 = (0, 1), (1, 2), (2, 3)
    F = {e1: None, e2: 0, e3: 1}
    P = [1, e1, e2, e3, "t"]
    assert _is_arc_joining_F(e3, F, 0)

    result = _augment_F(P, F, 2)

    assert result == {e1: 0, e2: 1, e3: 0}
    assert all(result[e] is not None for e in P[1:-1])


def test_augment_F_single_edge_path():
    F = {(0, 1): None, (1, 2): 0}
    P = [1, (0, 1), "t"]
    result = _augment_F(P, F, 2)
    assert result[(0, 1)] == 0
    assert result[(1, 2)] == 0


def test_search_augmenting_path():
    F = {(0, 1): 0, (0, 2): 0, (1, 2): None}
    C = [0, 1, 2]
    G = G = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    deficits = {0: 0, 1: 0, 2: 1}
    assert _search_augmenting_path(C, G, F, {}, 2, deficits)
    assert F[(1, 2)] == 1
    assert deficits[2] == 0


def test_k_forest_packing():
    G = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    F = {(0, 1): None, (1, 2): None, (0, 2): None}
    tau = {0: 0, 1: 0, 2: 0}
    result = _k_forest_packing(G, F, 2, tau)

    # every edge is labelled and each label class is a forest
    assert set(result) == set(G.edges())
    assert all(label in (0, 1) for label in result.values())
    for i in (0, 1):
        H = nx.Graph([e for e, label in result.items() if label == i])
        assert nx.is_forest(H)


def test_k_forest_packing_2():
    G = nx.DiGraph([(0, 3), (1, 3), (2, 3)])
    F = {(0, 3): None, (1, 3): None, (2, 3): None}
    tau = {0: 0, 1: 0, 2: 0, 3: 0}
    k = 2
    result = _k_forest_packing(G, F, k, tau)
    assigned_edges = {e: label for e, label in result.items() if label is not None}

    # not all edges are assigned, because the graph is not 2-forest-decomposable
    assert len(assigned_edges) == 2
    assert set(assigned_edges.keys()) < set(G.edges())
    for i in (0, 1):
        edges_in_forest = [e for e, label in assigned_edges.items() if label == i]
        assert not edges_in_forest or nx.is_forest(nx.Graph(edges_in_forest))


@pytest.mark.parametrize(
    "G, k",
    [
        (nx.path_graph(5), 1),
        (nx.cycle_graph(4), 2),
        (nx.complete_graph(4), 2),
        (nx.complete_graph(6), 3),
    ],
)
def test_forest_decomposition_exists(G, k):
    X, *F = _k_forest_decomposition(G, k)
    assert X is None
    assert len(F) == k
    edges = [frozenset(e) for forest in F for e in forest]
    assert len(edges) == len(set(edges)) == G.number_of_edges()
    assert set(edges) == {frozenset(e) for e in G.edges()}
    for forest in F:
        assert not forest or nx.is_forest(nx.Graph(list(forest)))


@pytest.mark.parametrize(
    "G, k",
    [
        (nx.cycle_graph(3), 1),
        (nx.complete_graph(5), 2),
        (nx.complete_graph(7), 3),
    ],
)
def test_forest_decomposition_does_not_exist(G, k):
    X, *F = _k_forest_decomposition(G, k)
    assert X is not None
    assert G.subgraph(X).number_of_edges() > k * len(X) - k
    assert len(F) == k
    assert all(len(forest) == 0 for forest in F)


def test_forest_decomposition_violating_set_with_pendant_path():
    G = nx.complete_graph(5)
    G.add_edges_from([(4, 5), (5, 6)])
    X, *_ = _k_forest_decomposition(G, 2)
    assert X is not None
    assert G.subgraph(X).number_of_edges() > 2 * len(X) - 2


def test_k_bounded_forest_decomposition():
    D = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    X, *F = _k_bounded_forest_decomposition(D, 2)
    assert X is None
    assert set().union(*F) == set(D.edges())
    assert not set(F[0]) & set(F[1])


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("k", [1, 2, 3])
def test_violating_set_is_valid_random(seed, k):
    G = nx.gnm_random_graph(8, 6 * k, seed=seed)
    X, *F = _k_forest_decomposition(G, k)
    if X is None:
        assert all(not f or nx.is_forest(nx.Graph(list(f))) for f in F)
    else:
        assert len(X) >= 2
        assert G.subgraph(X).number_of_edges() > k * len(X) - k


def _entering(D, X):
    return sum(1 for u, v in D.edges() if v in X and u not in X)


def _violating_set_exists(D, eta, s, Z):
    others = [v for v in D.nodes if v != s]
    return any(
        set(X) & Z and _entering(D, set(X)) < eta
        for r in range(1, len(others) + 1)
        for X in itertools.combinations(others, r)
    )


@pytest.mark.parametrize("seed", range(5))
def test_eta_rooted_term_connectivity_random(seed):
    rng = random.Random(seed)
    for _ in range(100):
        n = rng.randint(2, 6)
        D = nx.MultiDiGraph()
        D.add_nodes_from(range(n))
        for _ in range(rng.randint(0, 14)):
            D.add_edge(*rng.sample(range(n), 2))
        Z = set(rng.sample(range(1, n), rng.randint(0, n - 1)))
        for eta in range(6):
            X = _eta_rooted_term_connectivity(D, eta, 0, Z)
            if X is None:
                assert not _violating_set_exists(D, eta, 0, Z)
            else:
                assert X and 0 not in X and X & Z
                assert _entering(D, X) < eta


@pytest.mark.parametrize("seed", range(5))
def test_eta_rooted_arc_connectivity_random(seed):
    rng = random.Random(seed)
    for _ in range(100):
        n = rng.randint(1, 6)
        D = nx.MultiDiGraph()
        D.add_nodes_from(range(n))
        if n > 1:
            for _ in range(rng.randint(0, 16)):
                D.add_edge(*rng.sample(range(n), 2))
        Z = set(range(1, n))
        for eta in range(7):
            X = _eta_rooted_arc_connectivity(D, eta, 0)
            if X is None:
                assert not _violating_set_exists(D, eta, 0, Z)
            else:
                assert X and 0 not in X
                assert _entering(D, X) < eta


def test_build_rooted_digraph():
    D0 = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    D, s = _build_rooted_digraph(D0, {0}, 2)
    assert set(D.nodes()) == {1, 2, s}
    assert D.number_of_edges(1, 2) == 1
    assert D.number_of_edges(s, 1) == 1
    assert D.number_of_edges(s, 2) == 0


def test_build_rooted_digraph_invalid():
    with pytest.raises(ValueError):
        _build_rooted_digraph(nx.DiGraph([(0, 1)]), {1}, 1)
    with pytest.raises(ValueError):
        _build_rooted_digraph(nx.DiGraph([(0, 2), (1, 2)]), set(), 1)


def test_build_rooted_digraph_structure():
    D0 = nx.DiGraph([(0, 1), (0, 2), (1, 2), (2, 3)])
    D, s = _build_rooted_digraph(D0, {0}, 3)
    assert isinstance(D, nx.MultiDiGraph)
    assert set(D.nodes()) == {1, 2, 3, s}
    # arcs leaving U0 are dropped, the others are kept
    assert D.number_of_edges(1, 2) == 1
    assert D.number_of_edges(2, 3) == 1
    assert D.number_of_edges() == 2 + (3 - 1) + (3 - 2) + (3 - 1)
    # parallel arcs from the root: k - indegree in D0
    assert D.number_of_edges(s, 1) == 2
    assert D.number_of_edges(s, 2) == 1
    assert D.number_of_edges(s, 3) == 2
    assert D.in_degree(s) == 0


def test_build_rooted_digraph_indegree_is_k():
    # several vertices in U0; arcs from the root fill indegrees up to k
    D0 = nx.DiGraph([(0, 1), (1, 2), (2, 1), (3, 2)])
    D, s = _build_rooted_digraph(D0, {0, 3}, 2)
    assert set(D.nodes()) == {1, 2, s}
    assert D.number_of_edges(s, 1) == 0
    assert D.number_of_edges(s, 2) == 0
    assert D.number_of_edges(1, 2) == 1
    assert D.number_of_edges(2, 1) == 1


def test_build_rooted_digraph_does_not_modify_input():
    D0 = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    edges = set(D0.edges())
    _build_rooted_digraph(D0, {0}, 2)
    assert set(D0.nodes()) == {0, 1, 2}
    assert set(D0.edges()) == edges


def test_build_rooted_digraph_saturated():
    # k-regular indegree: no arcs from the root
    D0 = nx.DiGraph([(1, 2), (2, 1)])
    D0.add_node(0)
    D, s = _build_rooted_digraph(D0, {0}, 1)
    assert D.out_degree(s) == 0


def _edges_inside(D, Y):
    return sum(1 for u, v in D.edges() if u in Y and v in Y)


def _terminal_violation_exists(D, c, Z, K):
    others = [v for v in D.nodes if v != c]
    for r in range(1, len(others) + 1):
        for X in itertools.combinations(others, r):
            Y = set(X) | {c}
            if set(X) & Z and _edges_inside(D, Y) > K * len(Y) - 2 * K:
                return True
    return False


def test_check_terminal_superset_sparsity_none():
    # c isolated, a path on the others: no set containing c violates (1,2)-sparsity
    D = nx.DiGraph([(1, 2), (2, 3)])
    D.add_node(0)
    assert _check_terminal_superset_sparsity(D, 0, {1, 2, 3}, 1) is None


def test_check_terminal_superset_sparsity_violation():
    # triangle 0-1-2 with c = 0 a source: 3 edges > 2*3 - 2*2 for K = 2
    D = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    X = _check_terminal_superset_sparsity(D, 0, {1, 2}, 2)
    assert X is not None and 0 in X and X != {0} and X & {1, 2}
    assert _edges_inside(D, X) > 2 * len(X) - 4


def test_check_terminal_superset_sparsity_terminals_matter():
    # K = 2: {0,1,2} is violating, but 3 is the only terminal, so a violating
    # set must contain 3; {0,1,2,3} has 5 edges > 2*4 - 4
    D = nx.DiGraph([(0, 1), (0, 2), (1, 2), (2, 3), (1, 3)])
    X = _check_terminal_superset_sparsity(D, 0, {3}, 2)
    assert X is not None and 0 in X and 3 in X
    # without the extra edge only {0,1,2} violates, which misses the terminal
    D.remove_edge(1, 3)
    assert _check_terminal_superset_sparsity(D, 0, {3}, 2) is None


@pytest.mark.parametrize("seed", range(5))
def test_check_terminal_superset_sparsity_random(seed):
    rng = random.Random(seed)
    for _ in range(100):
        K = rng.randint(1, 3)
        n = rng.randint(2, 6)
        D = nx.DiGraph()
        D.add_nodes_from(range(n))
        # c = 0 is a source; indegrees bounded by K
        for v in range(1, n):
            for u in rng.sample(range(n), rng.randint(0, min(K, n - 1))):
                if u != v:
                    D.add_edge(u, v)
        Z = set(rng.sample(range(1, n), rng.randint(0, n - 1)))
        X = _check_terminal_superset_sparsity(D, 0, Z, K)
        if X is None:
            assert not _terminal_violation_exists(D, 0, Z, K)
        else:
            assert 0 in X and X != {0} and X & Z
            assert _edges_inside(D, X) > K * len(X) - 2 * K


@pytest.mark.parametrize(
    "G, K, L",
    [
        (nx.complete_graph(5), 2, 1),
        #  5 complete graph with pendant
        (nx.Graph(list(nx.complete_graph(5).edges) + [(5, 6)]), 2, 1),
    ],
)
def test_check_sparsity_for_l_less_than_k_false(G, K, L):
    X = _check_sparsity_for_l_less_than_k(G, K, L)
    assert X


@pytest.mark.parametrize(
    "G, K, L",
    [(nx.complete_graph(4), 2, 1), (nx.path_graph(5), 2, 1)],
)
def test_check_sparsity_for_l_less_than_k_true(G, K, L):
    X = _check_sparsity_for_l_less_than_k(G, K, L)
    assert X is None


@pytest.mark.parametrize(
    "G, K, L",
    [
        (nx.complete_graph(4), 2, 3),
        (nx.Graph(list(nx.complete_graph(4).edges) + [(4, 5)]), 2, 3),
    ],
)
def test_check_sparsity_for_l_between_k_and_2k_false(G, K, L):
    X = _check_sparsity_for_l_between_k_and_2k(G, K, L)
    assert X


@pytest.mark.parametrize(
    "G, K, L",
    [
        # Diamond graph: V=4, E=5.
        (nx.diamond_graph(), 2, 3),
        (nx.path_graph(5), 2, 3),
    ],
)
def test_check_sparsity_for_l_between_k_and_2k_true(G, K, L):
    X = _check_sparsity_for_l_between_k_and_2k(G, K, L)
    assert X is None


@pytest.mark.parametrize(
    "G, K, L",
    [
        (nx.diamond_graph(), 2, 4),
        (nx.Graph(list(nx.diamond_graph().edges) + [(3, 4)]), 2, 4),
    ],
)
def test_check_sparsity_for_l_equal_2k_false(G, K, L):
    X = _check_sparsity_for_l_equal_2k(G, K, L)
    assert X


@pytest.mark.parametrize(
    "G, K, L",
    [(nx.cycle_graph(4), 2, 4), (nx.cycle_graph(5), 2, 4)],
)
def test_check_sparsity_for_l_equal_2k_true(G, K, L):
    X = _check_sparsity_for_l_equal_2k(G, K, L)
    assert X is None


C5_with_chord = nx.cycle_graph(5)
C5_with_chord.add_edge(0, 2)


@pytest.mark.parametrize(
    "G, K, L",
    [
        (C5_with_chord.copy(), 2, 5),
        (nx.Graph(list(C5_with_chord.edges) + [(4, 5)]), 2, 5),
        (nx.cycle_graph(5), 2, 5),
        (nx.path_graph(6), 2, 5),
    ],
)
def test_check_sparsity_for_l_between_2k_and_3k_false(G, K, L):
    X = _check_sparsity_for_l_between_2k_and_3k(G, K, L)
    assert X


@pytest.mark.parametrize(
    "G, K, L",
    [
        (nx.empty_graph(5), 2, 5),
        (nx.Graph([(0, 1), (2, 3), (4, 5)]), 2, 5),
        (nx.path_graph(3), 2, 4),
    ],
)
def test_check_sparsity_for_l_between_2k_and_3k_true(G, K, L):
    X = _check_sparsity_for_l_between_2k_and_3k(G, K, L)
    assert X is None
