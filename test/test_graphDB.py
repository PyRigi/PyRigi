from pyrigi.graph import Graph
from pyrigi.framework import Framework
import pyrigi.graphDB as graphs

import pytest


def test_check_vertex_and_edge_order():
    F = Framework.Random(Graph([("a", 1.8), ("a", "#"), ("#", 0), (0, 1.8)]))
    vertex_order = ["a", "#", 0, 1.8]
    edge_order = [(0, "#"), ("a", 1.8), (0, 1.8), ("#", "a")]
    assert F._check_vertex_order(vertex_order) and F._check_edge_order(edge_order)
    vertex_order = ["a", "#", 0, "s"]
    edge_order = [("#", "#"), ("a", 1.8), (0, 1.8), ("#", "a")]
    with pytest.raises(ValueError):
        F._check_vertex_order(vertex_order)
    with pytest.raises(ValueError):
        F._check_edge_order(edge_order)
