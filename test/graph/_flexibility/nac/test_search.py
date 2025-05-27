import random
from dataclasses import dataclass

import networkx as nx
import pytest

import pyrigi.graphDB as graphs
from pyrigi.graph._flexibility.nac import MonoClassType, is_NAC_coloring
from pyrigi.graph._flexibility.nac.search import NAC_colorings_impl
from pyrigi.graph.graph import Graph


@dataclass
class NACTestCase:
    """
    Used for NAC-coloring and Cartesian NAC-coloring testing.
    """

    __slots__ = ("name", "graph", "no_normal", "no_cartesian")

    name: str
    graph: nx.Graph
    no_normal: int | None
    no_cartesian: int | None


NAC_TEST_CASES: list[NACTestCase] = [
    NACTestCase("path", graphs.Path(3), 2, 2),
    NACTestCase(
        "path_and_single_vertex",
        Graph.from_vertices_and_edges([0, 1, 2, 3], [(0, 1), (1, 2)]),
        2,
        2,
    ),
    NACTestCase("cycle3", graphs.Cycle(3), 0, 0),
    NACTestCase("cycle4", graphs.Cycle(4), 6, 2),
    NACTestCase("cycle5", graphs.Cycle(5), 20, 10),
    NACTestCase("complete5", graphs.Complete(5), 0, 0),
    NACTestCase("bipartite1x3", graphs.CompleteBipartite(1, 3), 6, 6),
    NACTestCase(
        "bipartite1x3-improved",
        Graph.from_vertices_and_edges([0, 1, 2, 3], [(0, 1), (0, 2), (0, 3), (2, 3)]),
        2,
        2,
    ),
    NACTestCase("bipartite1x4", graphs.CompleteBipartite(1, 4), 14, 14),
    NACTestCase(
        "bipartite1x4-improved",
        Graph.from_vertices_and_edges(
            [0, 1, 2, 3, 4], [(0, 1), (0, 2), (0, 3), (0, 4), (3, 4)]
        ),
        6,
        6,
    ),
    NACTestCase("bipartite2x3", graphs.CompleteBipartite(2, 3), 14, 0),
    NACTestCase("bipartite2x4", graphs.CompleteBipartite(2, 4), 30, 0),
    NACTestCase("bipartite3x3", graphs.CompleteBipartite(3, 3), 30, 0),
    NACTestCase("bipartite3x4", graphs.CompleteBipartite(3, 4), 62, 0),
    NACTestCase("diamond", graphs.Diamond(), 0, 0),
    NACTestCase("prism", graphs.ThreePrism(), 2, 2),
    NACTestCase("prismPlus", graphs.ThreePrismPlusEdge(), 0, 0),
    NACTestCase("minimallyRigid", graphs.DiamondWithZeroExtension(), 2, 0),
    NACTestCase(
        "smaller_problemist",
        Graph.from_vertices_and_edges(
            [0, 1, 2, 3, 4, 5, 6],
            [(0, 3), (0, 6), (1, 2), (1, 6), (2, 5), (3, 5), (4, 5), (4, 6)],
        ),
        108,
        30,
    ),
    NACTestCase(
        "large_problemist",
        Graph.from_vertices_and_edges(
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [(0, 3), (0, 4), (1, 2), (1, 8), (2, 7), (3, 8)]
            + [(4, 7), (5, 7), (5, 8), (6, 7), (6, 8)],
        ),
        472,
        54,
    ),
    NACTestCase(
        "3-squares-and-connectig-edge",
        Graph.from_vertices_and_edges(
            list(range(10)),
            [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7)]
            + [(7, 4), (0, 8), (4, 8), (0, 9), (4, 9), (1, 5)],
        ),
        606,
        30,
    ),
    NACTestCase(
        "square-2-pendagons-and-connectig-edge",
        Graph.from_vertices_and_edges(
            list(range(12)),
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (5, 6), (6, 7)]
            + [(7, 8), (8, 9), (9, 5), (0, 10), (5, 10), (0, 11)]
            + [(5, 11), (1, 6)],
        ),
        None,  # 4596,
        286,
    ),
    NACTestCase(
        "diconnected-problemist",
        Graph.from_vertices_and_edges(
            list(range(15)),
            [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (7, 8), (9, 10)]
            + [(0, 13), (5, 13), (0, 14), (5, 14), (1, 6)],
        ),
        1214,
        254,
    ),
    NACTestCase(
        "brachiosaurus",
        Graph.from_vertices_and_edges(
            list(range(10)),
            [(0, 7), (0, 8), (0, 9), (1, 7), (1, 8), (1, 9), (2, 7)]
            + [(2, 8), (2, 9), (3, 7), (3, 8), (3, 9), (4, 5), (4, 8)]
            + [(4, 9), (5, 6), (5, 9), (6, 7), (6, 8), (6, 9)],
        ),
        126,
        None,  # unknown, yet
    ),
    NACTestCase(
        "cycles_destroyer",
        Graph.from_vertices_and_edges(
            list(range(14 + 1)),
            [(0, 3), (0, 8), (0, 12), (0, 14), (0, 9), (0, 5), (1, 13)]
            + [(1, 2), (1, 8), (1, 10), (2, 11), (2, 7), (2, 9), (3, 6)]
            + [(4, 14), (4, 13), (5, 9), (5, 6), (6, 9), (6, 10), (6, 8)]
            + [(7, 10), (7, 11), (8, 10), (9, 10), (10, 12), (11, 13)]
            + [(11, 12), (12, 14)],
        ),
        68,
        None,  # unknown, yet
    ),
    NACTestCase(
        "square_and_line",
        Graph.from_vertices_and_edges(
            list(range(15)),
            [(7, 8), (0, 13), (5, 13), (0, 14), (5, 14)],
        ),
        14,
        6,
    ),
]

NAC_ALGORITHMS = ["naive", "subgraphs"] + [
    "subgraphs-{}-{}-{}".format(split, merge, size)
    for split in ["none", "neighbors", "neighbors_degree"]
    for merge in ["linear", "shared_vertices"]
    for size in [1, 4]
]


@pytest.mark.parametrize(
    ("graph", "colorings_no"),
    [
        (case.graph, case.no_normal)
        for case in NAC_TEST_CASES
        if case.no_normal is not None
    ],
    ids=[case.name for case in NAC_TEST_CASES if case.no_normal is not None],
)
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS)
@pytest.mark.parametrize("use_decompositions", [True, False])
@pytest.mark.parametrize("use_cycles", [True, False])
@pytest.mark.parametrize(
    "class_type",
    [MonoClassType.TRI_CONNECTED, MonoClassType.TRI_EXTENDED],
)
@pytest.mark.parametrize("seed", [42, random.randint(0, 2**30)])
def test_all_NAC_colorings(
    graph: nx.Graph,
    colorings_no: int,
    algorithm: str,
    use_decompositions: bool,
    use_cycles: bool,
    class_type: MonoClassType,
    seed: int,
):
    # This configuration is supported only for the naive algorithm
    if not use_cycles and algorithm != "naive":
        return

    coloring_list = list(
        NAC_colorings_impl(
            graph,
            algorithm=algorithm,
            mono_class_type=class_type,
            use_cycles_optimization=use_cycles,
            use_blocks_decomposition=use_decompositions,
            seed=seed,
        )
    )

    no_duplicates = {
        (tuple(sorted(coloring[0])), tuple(sorted(coloring[1])))
        for coloring in coloring_list
    }
    assert len(coloring_list) == len(no_duplicates)

    assert colorings_no == len(coloring_list)

    for coloring in coloring_list:
        assert is_NAC_coloring(graph, coloring)
