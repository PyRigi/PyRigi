import networkx as nx
import pytest

from pyrigi import graphDB
from pyrigi.graph._flexibility.nac.core import IntEdge
from pyrigi.graph._flexibility.nac.mono_classes import MonoClassType, find_mono_classes
from pyrigi.graph.graph import Graph


@pytest.mark.parametrize(
    (
        "graph",
        "triangle_connected_comp",
        "monochromatic_classes",
    ),
    [
        (
            graphDB.ThreePrism(),
            [
                [(0, 1), (1, 2), (2, 0)],
                [(3, 4), (4, 5), (5, 3)],
                [(0, 3)],
                [(1, 4)],
                [(2, 5)],
            ],
            None,
        ),
        (
            Graph.from_vertices_and_edges(
                range(4), [(0, 1), (2, 0), (2, 1), (0, 3), (1, 3), (2, 3)]
            ),
            [[(0, 1), (2, 0), (2, 1), (0, 3), (1, 3), (2, 3)]],
            None,
        ),
        # Basic example of an edge in one triangle component
        (
            Graph.from_vertices_and_edges(
                range(7),
                [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (0, 4),
                    (1, 4),
                    (1, 5),
                    (2, 5),
                    (2, 6),
                    (3, 6),
                    (4, 5),
                    (5, 6),
                    (0, 3),
                ],
            ),
            [
                [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (0, 4),
                    (1, 4),
                    (1, 5),
                    (2, 5),
                    (2, 6),
                    (3, 6),
                    (4, 5),
                    (5, 6),
                ],
                [(0, 3)],
            ],
            [
                [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (0, 4),
                    (1, 4),
                    (1, 5),
                    (2, 5),
                    (2, 6),
                    (3, 6),
                    (4, 5),
                    (5, 6),
                    (0, 3),
                ]
            ],
        ),
        # Two edges over a component need to share the came class
        (
            Graph.from_vertices_and_edges(
                range(8),
                [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (0, 4),
                    (1, 4),
                    (1, 5),
                    (2, 5),
                    (2, 6),
                    (3, 6),
                    (4, 5),
                    (5, 6),
                    (0, 7),
                    (3, 7),
                ],
            ),
            [
                [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (0, 4),
                    (1, 4),
                    (1, 5),
                    (2, 5),
                    (2, 6),
                    (3, 6),
                    (4, 5),
                    (5, 6),
                ],
                [(0, 7)],
                [(3, 7)],
            ],
            [
                [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (0, 4),
                    (1, 4),
                    (1, 5),
                    (2, 5),
                    (2, 6),
                    (3, 6),
                    (4, 5),
                    (5, 6),
                ],
                [(0, 7), (3, 7)],
            ],
        ),
    ],
)
def test_find_monochromatic_classes(
    graph: nx.Graph,
    triangle_connected_comp: list[list[IntEdge]],
    monochromatic_classes: list[list[IntEdge]] | None,
):
    if monochromatic_classes is None:
        monochromatic_classes = triangle_connected_comp

    triang_edge_to_comp, triangle_comp_to_edges = find_mono_classes(
        graph, MonoClassType.TRI_CONNECTED
    )
    mono_edge_to_comp, mono_comp_to_edges = find_mono_classes(
        graph, MonoClassType.TRI_EXTENDED
    )

    for edge in graph.edges:
        assert edge in triangle_comp_to_edges[triang_edge_to_comp[edge]]
        assert edge in mono_comp_to_edges[mono_edge_to_comp[edge]]

    def normalize(classes: list[list[IntEdge]]) -> set[frozenset[frozenset]]:
        return set(frozenset(frozenset(edge) for edge in edges) for edges in classes)

    assert normalize(triangle_connected_comp) == normalize(triangle_comp_to_edges)
    assert normalize(monochromatic_classes) == normalize(mono_comp_to_edges)
