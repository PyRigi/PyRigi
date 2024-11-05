import pytest
from pyrigi.graph import Graph
import pyrigi.graphDB as graphs


# can be run with pytest -m large


# utility function to read a graph from a file
# Graph is in the following format:
# N M
# v1 v2
#
# v_n v_m
# where N is n the number of vertices and M the number of edges
# the N vertices are [0,1,2,...,N-1]
def read_graph_from_file(filename):
    with open(filename) as f:
        n, m = [int(x) for x in next(f).split()]
        g = Graph()
        for i in range(n):
            g.add_vertex(i)
        for i in range(m):
            v1, v2 = [int(x) for x in next(f).split()]
            g.add_edge(v1, v2)
        return g


@pytest.mark.large
def test_rigid_in_d2():
    graph = read_graph_from_file("test/input_graphs/K4.txt")
    assert graph.is_rigid(dim=2, combinatorial=True)


@pytest.mark.large
def test_big_random_tight_graphs():
    # (6,8)-tight graph on 50 vertices and 292 edges
    graph = read_graph_from_file("test/input_graphs/tight_6_8.txt")
    assert graph.is_tight(K=6, L=8, algorithm="pebble")

    # (7,3)-tight graph on 70 vertices and 487 edges
    graph = read_graph_from_file("test/input_graphs/tight_7_3.txt")
    assert graph.is_tight(K=7, L=3, algorithm="pebble")

    # (5,9)-tight graph on 40 vertices and 191 edges
    graph = read_graph_from_file("test/input_graphs/tight_5_9.txt")
    assert graph.is_tight(K=5, L=9, algorithm="pebble")

    # (13,14)-tight graph on 20 vertices and 246 edges
    graph = read_graph_from_file("test/input_graphs/tight_13_14.txt")
    assert graph.is_tight(K=13, L=14, algorithm="pebble")


@pytest.mark.large
def test_big_random_sparse_graphs():
    # (3,1)-sparse graph on 20 vertices and 58 edges
    graph = read_graph_from_file("test/input_graphs/sparse_3_1.txt")
    assert graph.is_sparse(K=3, L=1, algorithm="pebble")

    # (1,1)-sparse graph on 40 vertices and 38 edges - one edge broken
    graph = read_graph_from_file("test/input_graphs/sparse_1_1.txt")
    assert graph.is_sparse(K=1, L=1, algorithm="pebble")

    # (17,30)-sparse graph on 50 vertices and 38 edges
    graph = read_graph_from_file("test/input_graphs/sparse_17_30.txt")
    assert graph.is_sparse(K=17, L=30, algorithm="pebble")

    graph = read_graph_from_file("test/input_graphs/sparse_4_6.txt")
    assert graph.is_sparse(K=4, L=6, algorithm="pebble")


@pytest.mark.large
def test_big_random_not_sparse_graphs():
    # Dense graph on 20 vertices
    graph = read_graph_from_file("test/input_graphs/not_sparse_5_2.txt")
    assert not graph.is_sparse(K=5, L=2, algorithm="pebble")

    # (7,7)-tight graph plus one edge on 40 vertices (274 edges)
    graph = read_graph_from_file("test/input_graphs/not_sparse_7_7.txt")
    assert not graph.is_sparse(K=7, L=7, algorithm="pebble")

    # few edges in graph on 30 vertices, but has a (3,5)-connected circle
    graph = read_graph_from_file("test/input_graphs/not_sparse_3_5.txt")
    assert not graph.is_sparse(K=3, L=5, algorithm="pebble")

    # random large graph on 70 vertices, not sparse
    graph = read_graph_from_file("test/input_graphs/not_sparse_6_6.txt")
    assert not graph.is_sparse(K=6, L=6, algorithm="pebble")
