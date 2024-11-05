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
    graph = read_graph_from_file("test/input_graphs/random_6_8.txt")
    assert graph.is_tight(K=6, L=8, algorithm="pebble")

    # (7,3)-tight graph on 70 vertices and 487 edges
    graph = read_graph_from_file("test/input_graphs/random_7_3.txt")
    assert graph.is_tight(K=7, L=3, algorithm="pebble")

    # (5,9)-tight graph on 40 vertices and 191 edges
    graph = read_graph_from_file("test/input_graphs/random_5_9.txt")
    assert graph.is_tight(K=5, L=9, algorithm="pebble")

    # (13,14)-tight graph on 20 vertices and 246 edges
    graph = read_graph_from_file("test/input_graphs/random_13_14.txt")
    assert graph.is_tight(K=13, L=14, algorithm="pebble")

