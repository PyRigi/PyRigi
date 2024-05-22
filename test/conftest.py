import pytest

from pyrigi.graph import Graph


@pytest.fixture
def C4():
    """4-cycle graph"""
    return Graph([(0, 1), (1, 2), (2, 3), (3, 0)])


@pytest.fixture
def Diamond():
    """K4 minus edge graph"""
    return Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])


@pytest.fixture
def K4():
    """Complete graph on 4 vertices"""
    return Graph.Complete(4)
