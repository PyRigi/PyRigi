from pyrigi.graph import Graph
from pyrigi.framework import Framework
import pytest

def test_always_passes():
    assert True

def test_always_fails():
    assert False


def test_vertex_addition():
    G = Graph()
    G.add_vertex(0)
    F = Framework(G, {0:[1,2]})
    F.add_vertex([1,1], 1)
    F.add_vertex([0,0], 2)
    print(F)
