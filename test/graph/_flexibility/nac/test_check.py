import pytest

import networkx as nx

from pyrigi.graph._flexibility.nac.core import IntEdge
from pyrigi.graph._flexibility import nac
from pyrigi import graphDB


@pytest.mark.parametrize(
    ("graph", "coloring", "result"),
    [
        (
            graphDB.DiamondWithZeroExtension(),
            (set([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]), set([(1, 4), (3, 4)])),
            True,
        ),
        (
            graphDB.DiamondWithZeroExtension(),
            (set([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 4), (3, 4)]), set([])),
            False,
        ),
        (
            graphDB.DiamondWithZeroExtension(),
            (set([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 4)]), set([(3, 4)])),
            False,
        ),
        (
            graphDB.DiamondWithZeroExtension(),
            (set([(0, 1), (1, 2), (3, 0), (0, 2)]), set([(2, 3), (1, 4), (3, 4)])),
            False,
        ),
        (
            graphDB.ThreePrism(),
            (
                set([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]),
                set([(0, 3), (1, 4), (2, 5)]),
            ),
            True,
        ),
    ],
)
def test_is_NAC_coloring(
    graph: nx.Graph, coloring: tuple[set[IntEdge], set[IntEdge]], result: bool
):
    red, blue = coloring
    assert nac.is_NAC_coloring(graph, (red, blue)) == result
    assert nac.is_NAC_coloring(graph, (blue, red)) == result
