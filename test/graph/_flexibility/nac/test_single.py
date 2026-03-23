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
        graphs.ThreePrism(),
        graphs.DiamondWithZeroExtension(),
        graphs.Frustum(4),
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    algorithms,
)
def test_single_and_has_NAC_coloring(graph: nx.Graph, algorithm: str):
    assert nac.single_NAC_coloring(nx.Graph(graph), algorithm=algorithm) is not None
    assert nac.has_NAC_coloring(nx.Graph(graph), algorithm=algorithm)


@pytest.mark.parametrize(
    "graph",
    [
        Graph.from_vertices([0]),
        Graph.from_vertices([0, 1]),
        Graph.from_vertices_and_edges([0, 1], [(0, 1)]),
        nx.complete_graph(5),
        graphs.Cycle(3),
        graphs.Complete(5),
        graphs.Diamond(),
        graphs.ThreePrismPlusEdge(),
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    algorithms,
)
def test_single_and_has_no_NAC_coloring(graph: nx.Graph, algorithm: str):
    assert nac.single_NAC_coloring(nx.Graph(graph), algorithm=algorithm) is None
    assert not nac.has_NAC_coloring(nx.Graph(graph), algorithm=algorithm)
