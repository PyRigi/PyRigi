from pyrigi.graph import Graph
from pyrigi.framework import Framework

def test_dimension():
    F = Framework(Graph([[0,1]]),{0:[1,2], 1:[0,5]})
    assert F.dim == F.dimension()
    assert F.dim == 2
    F = Framework(dim=3)
    assert F.dim == 3