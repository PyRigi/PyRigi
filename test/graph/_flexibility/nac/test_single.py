import networkx as nx
import pytest

import pyrigi.graph._flexibility.nac as nac
import pyrigi.graphDB as graphs
from pyrigi.graph import Graph


@pytest.mark.parametrize(
    ("graph", "result"),
    [
        (Graph.from_vertices([0]), False),
        (Graph.from_vertices([0, 1]), False),
        (Graph.from_vertices_and_edges([0, 1], [(0, 1)]), False),
        (nx.complete_graph(5), False),
        (graphs.Path(3), True),
        (graphs.Cycle(3), False),
        (graphs.Cycle(4), True),
        (graphs.Cycle(5), True),
        (graphs.Complete(5), False),
        (graphs.CompleteBipartite(3, 4), True),
        (graphs.Diamond(), False),
        (graphs.ThreePrism(), True),
        (graphs.ThreePrismPlusEdge(), False),
        (graphs.DiamondWithZeroExtension(), True),
    ],
    ids=[
        "singleton",
        "two_vertices_no_edge",
        "single_edge",
        "complete_graph",
        "path",
        "cycle3",
        "cycle4",
        "cycle5",
        "complete5",
        "bipartite5",
        "diamond",
        "prism",
        "prismPlus",
        "minimallyRigid",
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    ["default", "naive", "subgraphs"]
    + [
        "subgraphs-{}-{}-{}".format(split, merge, size)
        for split in ["none", "neighbors", "neighbors_degree"]
        for merge in ["linear", "shared_vertices"]
        for size in [1, 4]
    ],
)
def test_single_and_has_NAC_coloring(graph: nx.Graph, algorithm: str, result: bool):
    assert result == (nac.single_NAC_coloring(graph, algorithm=algorithm) is not None)
    assert result == nac.has_NAC_coloring(graph, algorithm=algorithm)
