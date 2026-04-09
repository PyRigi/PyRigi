import networkx as nx
import pytest

import pyrigi.graph._flexibility.nac as nac
import pyrigi.graphDB as graphs
from pyrigi.graph import Graph

algorithms = ["default", "naive", "subgraphs"] + [
    "subgraphs-{}-{}-{}".format(split, merge, size)
    for split in ["none", "neighbors", "neighbors_degree"]
    for merge in ["linear", "shared_vertices"]
    for size in [1, 4]
]


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Path(3),
        graphs.Cycle(4),
        graphs.Cycle(5),
        graphs.CompleteBipartite(3, 3),
        graphs.CompleteBipartite(3, 4),
        graphs.K33plusEdge(),
        graphs.ThreePrism(),
        graphs.ThreePrismPlusTriangleOnSide(),
        graphs.DiamondWithZeroExtension(),
        graphs.Frustum(4),
        graphs.CubeWithDiagonal(),
        graphs.Dodecahedral(),
        graphs.DoubleBanana(),
        graphs.Grid(3, 3),
        graphs.K66MinusPerfectMatching(),
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    algorithms,
)
def test_single_and_has_NAC_coloring(graph: nx.Graph, algorithm: str):
    graph = nx.Graph(graph)
    NAC_col = nac.single_NAC_coloring(graph, algorithm=algorithm)
    assert NAC_col is not None
    assert nac.is_NAC_coloring(graph, NAC_col)
    assert nac.has_NAC_coloring(graph, algorithm=algorithm)


@pytest.mark.parametrize(
    "graph",
    [
        Graph.from_vertices([0]),
        Graph.from_vertices([0, 1]),
        graphs.Complete(2),
        graphs.Complete(3),
        graphs.Complete(4),
        graphs.Complete(5),
        graphs.CompleteMinusOne(5),
        graphs.Diamond(),
        graphs.ThreePrismPlusEdge(),
        graphs.Octahedral(),
        graphs.Icosahedral(),
        graphs.Wheel(4),
        graphs.Wheel(5),
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    algorithms,
)
def test_single_and_has_no_NAC_coloring(graph: nx.Graph, algorithm: str):
    graph = nx.Graph(graph)
    assert nac.single_NAC_coloring(graph, algorithm=algorithm) is None
    assert not nac.has_NAC_coloring(graph, algorithm=algorithm)
