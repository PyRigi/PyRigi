import pytest

from pyrigi.graph import Graph


@pytest.fixture
def Empty():
    """Empty graph"""
    return Graph()


@pytest.fixture
def C4():
    """4-cycle graph"""
    return Graph([(0, 1), (1, 2), (2, 3), (3, 0)])


@pytest.fixture
def Diamond():
    """K4 minus edge graph"""
    return Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])


@pytest.fixture
def K2():
    """Complete graph on 2 vertices"""
    return Graph.Complete(2)


@pytest.fixture
def K3():
    """Complete graph on 3 vertices"""
    return Graph.Complete(3)


@pytest.fixture
def K4():
    """Complete graph on 4 vertices"""
    return Graph.Complete(4)


@pytest.fixture
def P2():
    """Path graph with 2 edges"""
    return Graph([[i, i + 1] for i in range(2)])


@pytest.fixture
def P3():
    """Path graph with 3 edges"""
    return Graph([[i, i + 1] for i in range(3)])
