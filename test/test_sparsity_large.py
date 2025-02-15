import pytest
from pyrigi.graph import Graph
import networkx as nx


# can be run with pytest -m slow_main


def read_from_sparse6(filename):
    return Graph(nx.read_sparse6(filename))


@pytest.mark.slow_main
def test_rigid_in_d2():
    graph = read_from_sparse6("test/input_graphs/sparsity/K4.s6")
    assert graph.is_rigid(dim=2, algorithm="sparsity")

    # (2,3)-tight graph on 1000 vertices and 1997 edges
    graph = read_from_sparse6("test/input_graphs/sparsity/huge_tight_2_3.s6")
    assert graph.is_kl_tight(K=2, L=3, algorithm="pebble")


@pytest.mark.slow_main
def test_big_random_tight_graphs():
    # (6,8)-tight graph on 50 vertices and 292 edges
    graph = read_from_sparse6("test/input_graphs/sparsity/tight_6_8.s6")
    assert graph.is_kl_tight(K=6, L=8, algorithm="pebble")

    # (7,3)-tight graph on 70 vertices and 487 edges
    graph = read_from_sparse6("test/input_graphs/sparsity/tight_7_3.s6")
    assert graph.is_kl_tight(K=7, L=3, algorithm="pebble")

    # (5,9)-tight graph on 40 vertices and 191 edges
    graph = read_from_sparse6("test/input_graphs/sparsity/tight_5_9.s6")
    assert graph.is_kl_tight(K=5, L=9, algorithm="pebble")

    # (13,14)-tight graph on 20 vertices and 246 edges
    graph = read_from_sparse6("test/input_graphs/sparsity/tight_13_14.s6")
    assert graph.is_kl_tight(K=13, L=14, algorithm="pebble")

    # (2,3)-tight graph on 1000 vertices and 1997 edges
    graph = read_from_sparse6("test/input_graphs/sparsity/huge_tight_2_3.s6")
    assert graph.is_kl_tight(K=2, L=3, algorithm="pebble")


@pytest.mark.slow_main
def test_big_random_sparse_graphs():
    # (3,1)-sparse graph on 20 vertices and 58 edges
    graph = read_from_sparse6("test/input_graphs/sparsity/sparse_3_1.s6")
    assert graph.is_kl_sparse(K=3, L=1, algorithm="pebble")

    # (1,1)-sparse graph on 40 vertices and 38 edges - one edge broken
    graph = read_from_sparse6("test/input_graphs/sparsity/sparse_1_1.s6")
    assert graph.is_kl_sparse(K=1, L=1, algorithm="pebble")

    # (17,30)-sparse graph on 50 vertices and 38 edges
    graph = read_from_sparse6("test/input_graphs/sparsity/sparse_17_30.s6")
    assert graph.is_kl_sparse(K=17, L=30, algorithm="pebble")

    graph = read_from_sparse6("test/input_graphs/sparsity/sparse_4_6.s6")
    assert graph.is_kl_sparse(K=4, L=6, algorithm="pebble")

    # (2,3)-sparse graph on 1000 vertices and 1996 edges
    graph = read_from_sparse6("test/input_graphs/sparsity/huge_sparse_2_3.s6")
    assert graph.is_kl_sparse(K=2, L=3, algorithm="pebble")


@pytest.mark.slow_main
def test_big_random_not_sparse_graphs():
    # Dense graph on 20 vertices
    graph = read_from_sparse6("test/input_graphs/sparsity/not_sparse_5_2.s6")
    assert not graph.is_kl_sparse(K=5, L=2, algorithm="pebble")

    # (7,7)-tight graph plus one edge on 40 vertices (274 edges)
    graph = read_from_sparse6("test/input_graphs/sparsity/not_sparse_7_7.s6")
    assert not graph.is_kl_sparse(K=7, L=7, algorithm="pebble")

    # few edges in graph on 30 vertices, but has a (3,5)-connected circle
    graph = read_from_sparse6("test/input_graphs/sparsity/not_sparse_3_5.s6")
    assert not graph.is_kl_sparse(K=3, L=5, algorithm="pebble")

    # random large graph on 70 vertices, not sparse
    graph = read_from_sparse6("test/input_graphs/sparsity/not_sparse_6_6.s6")
    assert not graph.is_kl_sparse(K=6, L=6, algorithm="pebble")


@pytest.mark.slow_main
def test_Rd_circuit_graphs():
    graph = read_from_sparse6("test/input_graphs/sparsity/K4.s6")
    assert graph.is_Rd_circuit(dim=2, algorithm="sparsity")

    graph = read_from_sparse6("test/input_graphs/sparsity/circle_5_8.s6")
    assert graph.is_Rd_circuit(dim=2, algorithm="sparsity")

    graph = read_from_sparse6("test/input_graphs/sparsity/circle_10_18.s6")
    assert graph.is_Rd_circuit(dim=2, algorithm="sparsity")

    graph = read_from_sparse6("test/input_graphs/sparsity/circle_20_38.s6")
    assert graph.is_Rd_circuit(dim=2, algorithm="sparsity")

    graph = read_from_sparse6("test/input_graphs/sparsity/circle_30_58.s6")
    assert graph.is_Rd_circuit(dim=2, algorithm="sparsity")


@pytest.mark.slow_main
def test_Rd_not_circuit_graphs():
    graph = read_from_sparse6("test/input_graphs/sparsity/not_circle_5_7.s6")
    assert not graph.is_Rd_circuit(dim=2, algorithm="sparsity")

    graph = read_from_sparse6("test/input_graphs/sparsity/not_circle_10_18.s6")
    assert not graph.is_Rd_circuit(dim=2, algorithm="sparsity")

    graph = read_from_sparse6("test/input_graphs/sparsity/not_circle_20_39.s6")
    assert not graph.is_Rd_circuit(dim=2, algorithm="sparsity")

    graph = read_from_sparse6("test/input_graphs/sparsity/not_circle_30_58.s6")
    assert not graph.is_Rd_circuit(dim=2, algorithm="sparsity")
