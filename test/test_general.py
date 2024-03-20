import pytest

def test_always_passes():
    assert True

def test_always_fails():
    assert False

from .. import pyrigi as pr

def test_vertex_addition():
    G = pr.Graph()
    G.add_vertex(0)
    F = pr.Framework(G, {0:[1,2]})
    F.add_vertex([1,1], 1)
    F.add_vertex([0,0], 2)
    print(F)
