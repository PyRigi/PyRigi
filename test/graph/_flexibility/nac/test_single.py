import networkx as nx
import pytest

import pyrigi.graph._flexibility.nac as nac
import pyrigi.graphDB as graphs


@pytest.mark.parametrize(
    ("graph", "result"),
    [
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
def test_sinlge_and_has_NAC_coloring(graph: nx.Graph, result: bool):
    assert result == (nac.single_NAC_coloring(graph) is not None)
    assert result == nac.has_NAC_coloring(
        graph,
    )
