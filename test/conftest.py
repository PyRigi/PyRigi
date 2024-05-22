import pytest

from pyrigi.graph import Graph
from pyrigi.framework import Framework


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


# --------------------------------------------------------------
# The following are framework fixtures.
# To distinguish them from graphs, we append _d<dimension>.
# After another underscore, the shape of realization might be indicated.
# --------------------------------------------------------------


# Empty
@pytest.fixture
def Empty_d1(Empty):
    """1-dimensional empty framework"""
    return Framework.Empty(1)


@pytest.fixture
def Empty_d2(Empty):
    """2-dimensional empty framework"""
    return Framework.Empty(2)


# C4
@pytest.fixture
def C4_d1(C4):
    """Framework of the 4-cycle with square realization in the plane"""
    return Framework(C4, {0: [0], 1: [1], 2: [2], 3: [3]})


@pytest.fixture
def C4_d2_square(C4):
    """Framework of the 4-cycle with square realization in the plane"""
    return Framework(C4, {0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1]})


# Diamond
@pytest.fixture
def Diamond_d2_square(Diamond):
    """Framework of the diamond with square realization in the plane"""
    return Framework(Diamond, {0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1]})


# K2
@pytest.fixture
def K2_d1(K2):
    """Framework of K2 on the line"""
    return Framework(K2, {0: [0, 0], 1: [1, 0]})


@pytest.fixture
def K2_d2(K2):
    """Framework of K2 with the edge on the x-axis in the plane"""
    return Framework(K2, {0: [0, 0], 1: [1, 0]})


# K3
@pytest.fixture
def K3_d1(K3):
    """Framework of K3 on the line"""
    return Framework(K3, {0: [0], 1: [1], 2: [2]})


@pytest.fixture
def K3_d2_rightangle(K3):
    """Framework of K3 with right-angle triangle realization in the plane"""
    return Framework(K3, {0: [0, 0], 1: [1, 0], 2: [0, 1]})


# K4
@pytest.fixture
def K4_d2(K4):
    """Framework of K4 with square realization in the plane"""
    return Framework(K4, {0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1]})


# P2
@pytest.fixture
def P2_d1(P2):
    """Framework of P2 with 2 edges on the x-axis in the plane"""
    return Framework([[i, i + 1] for i in range(2)], {i: [i] for i in range(2)})


@pytest.fixture
def P2_d2_linear(P2):
    """Framework of P2 with 2 edges on the x-axis in the plane"""
    return Framework([[i, i + 1] for i in range(2)], {i: [i, 0] for i in range(2)})


# P3
@pytest.fixture
def P3_d1(P3):
    """Framework of P3 with 3 edges on the x-axis in the plane"""
    return Framework([[i, i + 1] for i in range(2)], {i: [i] for i in range(2)})


@pytest.fixture
def P3_d2(P3):
    """Framework of P3 with 3 edges on the x-axis in the plane"""
    return Framework([[i, i + 1] for i in range(2)], {i: [i, 0] for i in range(2)})
