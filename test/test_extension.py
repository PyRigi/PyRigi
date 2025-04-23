import pytest
from networkx import Graph as NXGraph

import pyrigi.extension as extension
import pyrigi.generic_rigidity as generic_rigidity
import pyrigi.graphDB as graphs
import pyrigi.misc as misc
from pyrigi.graph import Graph
from test_graph import TEST_WRAPPED_FUNCTIONS
from pyrigi.exception import NotSupportedValueError


def test_k_extension():
    assert graphs.Complete(2).zero_extension([0, 1]) == graphs.Complete(3)
    assert graphs.Complete(2).zero_extension([1], dim=1) == graphs.Path(3)
    assert graphs.Complete(4).one_extension([0, 1, 2], (0, 1)) == Graph(
        [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
    )
    assert graphs.CompleteBipartite(3, 2).one_extension(
        [0, 1, 2, 3, 4], (0, 3), dim=4
    ) == Graph(
        [
            (0, 4),
            (0, 5),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 5),
            (4, 5),
        ]
    )

    assert graphs.CompleteBipartite(3, 2).k_extension(
        2, [0, 1, 3], [(0, 3), (1, 3)], dim=1
    ) == Graph([(0, 4), (0, 5), (1, 4), (1, 5), (2, 3), (2, 4), (3, 5)])
    assert graphs.CompleteBipartite(3, 2).k_extension(
        2, [0, 1, 3, 4], [(0, 3), (1, 3)]
    ) == Graph([(0, 4), (0, 5), (1, 4), (1, 5), (2, 3), (2, 4), (3, 5), (4, 5)])
    assert graphs.Cycle(6).k_extension(
        4, [0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4)], dim=1
    ) == Graph([(0, 5), (0, 6), (1, 6), (2, 6), (3, 6), (4, 5), (4, 6)])
    if TEST_WRAPPED_FUNCTIONS:
        assert extension.k_extension(
            graphs.CompleteBipartite(3, 2), 2, [0, 1, 3], [(0, 3), (1, 3)], dim=1
        ) == Graph([(0, 4), (0, 5), (1, 4), (1, 5), (2, 3), (2, 4), (3, 5)])
        assert extension.k_extension(graphs.CompleteBipartite(3, 2),
        2, [0, 1, 3, 4], [(0, 3), (1, 3)]
    ) == Graph([(0, 4), (0, 5), (1, 4), (1, 5), (2, 3), (2, 4), (3, 5), (4, 5)])
        assert extension.k_extension(graphs.Cycle(6),
        4, [0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4)], dim=1
    ) == Graph([(0, 5), (0, 6), (1, 6), (2, 6), (3, 6), (4, 5), (4, 6)])


def test_all_k_extensions():
    for ext in graphs.Complete(4).all_k_extensions(1, 1):
        assert ext in [
            Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3]]),
            Graph([[0, 1], [0, 3], [0, 4], [1, 2], [1, 3], [2, 3], [2, 4]]),
            Graph([[0, 1], [0, 2], [0, 4], [1, 2], [1, 3], [2, 3], [3, 4]]),
            Graph([[0, 1], [0, 2], [0, 3], [1, 3], [1, 4], [2, 3], [2, 4]]),
            Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 3], [3, 4]]),
            Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 4], [3, 4]]),
        ]
    for ext in graphs.Complete(4).all_k_extensions(
        2, 2, only_non_isomorphic=True
    ):
        assert ext in [
            Graph([[0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]),
            Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 4], [3, 4]]),
        ]
    all_diamond_0_2 = list(
        graphs.Diamond().all_k_extensions(0, 2, only_non_isomorphic=True)
    )
    assert (
        len(all_diamond_0_2) == 3
        and all_diamond_0_2[0]
        == Graph([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 4], [2, 3]])
        and all_diamond_0_2[1]
        == Graph([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [2, 4]])
        and all_diamond_0_2[2]
        == Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 3], [3, 4]])
    )
    all_diamond_1_2 = graphs.Diamond().all_k_extensions(1, 2, only_non_isomorphic=True)
    assert next(all_diamond_1_2) == Graph(
        [[0, 2], [0, 3], [0, 4], [1, 2], [1, 4], [2, 3], [2, 4]]
    ) and next(all_diamond_1_2) == Graph(
        [[0, 2], [0, 3], [0, 4], [1, 2], [1, 4], [2, 3], [3, 4]]
    )
    if TEST_WRAPPED_FUNCTIONS:
        for ext in extension.all_k_extensions(graphs.Complete(4), 1, 1):
            assert ext in [
                Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3]]),
                Graph([[0, 1], [0, 3], [0, 4], [1, 2], [1, 3], [2, 3], [2, 4]]),
                Graph([[0, 1], [0, 2], [0, 4], [1, 2], [1, 3], [2, 3], [3, 4]]),
                Graph([[0, 1], [0, 2], [0, 3], [1, 3], [1, 4], [2, 3], [2, 4]]),
                Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 3], [3, 4]]),
                Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 4], [3, 4]]),
            ]
        for ext in extension.all_k_extensions(
            graphs.Complete(4), 2, 2, only_non_isomorphic=True
        ):
            assert ext in [
                Graph([[0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]),
                Graph([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 4], [3, 4]]),
            ]
        all_diamond_0_2 = list(
            extension.all_k_extensions(graphs.Diamond(), 0, 2, only_non_isomorphic=True)
        )
        assert (
            len(all_diamond_0_2) == 3
            and all_diamond_0_2[0]
            == Graph([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 4], [2, 3]])
            and all_diamond_0_2[1]
            == Graph([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [2, 4]])
            and all_diamond_0_2[2]
            == Graph([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 3], [3, 4]])
        )
        all_diamond_1_2 = extension.all_k_extensions(graphs.Diamond(), 1, 2, only_non_isomorphic=True)
        assert next(all_diamond_1_2) == Graph(
            [[0, 2], [0, 3], [0, 4], [1, 2], [1, 4], [2, 3], [2, 4]]
        ) and next(all_diamond_1_2) == Graph(
            [[0, 2], [0, 3], [0, 4], [1, 2], [1, 4], [2, 3], [3, 4]]
        )


@pytest.mark.parametrize(
    "graph, k, dim, sol",
    [
        [Graph.from_int(254), 1, 2, [3934, 4011, 6891, 7672, 7916]],
        [graphs.Diamond(), 0, 2, [223, 239, 254]],
        [graphs.Complete(4), 0, 3, [511]],
        [graphs.CompleteMinusOne(5), 0, 1, [1535, 8703]],
        [
            Graph.from_int(16350),
            2,
            3,
            [257911, 260603, 376807, 384943, 1497823, 1973983],
        ],
        [graphs.CompleteMinusOne(5), 2, 3, [4095, 7679, 7935, 8187]],
    ],
)
def test_all_k_extensions2(graph, k, dim, sol):
    assert misc.is_isomorphic_graph_list(
        list(graph.all_k_extensions(k, dim, only_non_isomorphic=True)),
        [Graph.from_int(igraph) for igraph in sol],
    )
    if TEST_WRAPPED_FUNCTIONS:
        assert misc.is_isomorphic_graph_list(
            list(extension.all_k_extensions(graph, k, dim, only_non_isomorphic=True)),
            [Graph.from_int(igraph) for igraph in sol],
        )


@pytest.mark.parametrize(
    "graph, k, vertices, edges, dim",
    [
        [graphs.Complete(6), 2, [0, 1, 2], [[0, 1], [0, 2]], -1],
        [graphs.Complete(6), 2, [0, 1, 6], [[0, 1], [0, 6]], 1],
        [graphs.Complete(6), 2, [0, 1, 2], [[0, 1]], 1],
        [graphs.Complete(3), -1, [0], [], 2],
        [graphs.CompleteBipartite(2, 3), 2, [0, 1, 2], [[0, 1], [0, 2]], 1],
    ],
)
def test_k_extension_dim_error(graph, k, vertices, edges, dim):
    with pytest.raises(ValueError):
        graph.k_extension(k, vertices, edges, dim)
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.raises(ValueError):
            extension.k_extension(graph, k, vertices, edges, dim)


@pytest.mark.parametrize(
    "graph, k, vertices, edges",
    [
        [
            Graph.from_vertices_and_edges([1, 2, 3], [[1, 2], [2, 3], [3, 3]]),
            1,
            [1, 2, 3],
            [[3, 3]],
        ],
        [graphs.Complete(6), 2, [1, 2, 3, 4], [[1, 2], [1, 2]]],
        [graphs.Complete(6), 2, [1, 2, 3, 4], [[1, 2], [2, 1]]],
        [graphs.Complete(6), 2, [1, 2, 3, 4], [(1, 2), [1, 2]]],
        [graphs.Complete(6), 2, [1, 2, 3, 4], [(1, 2), [2, 1]]],
        [graphs.Complete(6), 2, [1, 2, 3, 4], [(1, 2), (1, 2)]],
        [graphs.Complete(6), 2, [1, 2, 3, 4], [(1, 2), (2, 1)]],
        [graphs.Complete(6), 3, [1, 2, 3, 4, 5], [[1, 2], [2, 3], [1, 2]]],
    ],
)
def test_k_extension_error(graph, k, vertices, edges):
    with pytest.raises(ValueError):
        graph.k_extension(k, vertices, edges)
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.raises(ValueError):
            extension.k_extension(graph, k, vertices, edges)


def test_all_k_extension_error():
    with pytest.raises(ValueError):
        list(Graph.from_vertices([0, 1, 2]).all_k_extensions(1, 1))
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.raises(ValueError):
            list(extension.all_k_extensions(Graph.from_vertices([0, 1, 2]), 1, 1))

@pytest.mark.parametrize(
    "graph, dim, sol",
    [
        [Graph.from_int(254), 2, [3326, 3934, 4011, 6891, 7672, 7916, 10479, 12511]],
        [graphs.Diamond(), 2, [223, 239, 254]],
        [graphs.Complete(4), 3, [511]],
        [graphs.Complete(1), 1, [1]],
        [graphs.CompleteMinusOne(5), 1, [1535, 8703]],
        [
            Graph.from_int(16350),
            3,
            [257911, 260603, 376807, 384943, 515806, 981215, 1497823, 1973983],
        ],
        [graphs.CompleteMinusOne(5), 3, [4095, 7679, 7935, 8187, 16350]],
    ],
)
def test_all_extensions(graph, dim, sol):
    assert misc.is_isomorphic_graph_list(
        list(graph.all_extensions(dim, only_non_isomorphic=True)),
        [Graph.from_int(igraph) for igraph in sol],
    )
    if TEST_WRAPPED_FUNCTIONS:
        assert misc.is_isomorphic_graph_list(
            list(extension.all_extensions(graph, dim, only_non_isomorphic=True)),
            [Graph.from_int(igraph) for igraph in sol],
        )


@pytest.mark.parametrize(
    "graph, dim",
    [
        [Graph.from_int(254), 2],
        [graphs.Diamond(), 2],
        [graphs.ThreePrism(), 2],
        [graphs.Complete(2), 1],
        [graphs.Complete(1), 1],
        [graphs.CompleteMinusOne(5), 1],
        pytest.param(Graph.from_int(16350), 3, marks=pytest.mark.slow_main),
        [graphs.CompleteMinusOne(5), 3],
    ],
)
def test_all_extensions_single(graph, dim):
    for k in range(0, dim):
        assert misc.is_isomorphic_graph_list(
            list(graph.all_extensions(dim, only_non_isomorphic=True, k_min=k, k_max=k)),
            list(graph.all_k_extensions(k, dim, only_non_isomorphic=True)),
        )
        assert misc.is_isomorphic_graph_list(
            list(graph.all_extensions(dim, k_min=k, k_max=k)),
            list(graph.all_k_extensions(k, dim)),
        )
    if TEST_WRAPPED_FUNCTIONS:
        for k in range(0, dim):
            assert misc.is_isomorphic_graph_list(
                list(extension.all_extensions(graph, dim, only_non_isomorphic=True, k_min=k, k_max=k)),
                list(extension.all_k_extensions(graph, k, dim, only_non_isomorphic=True)),
            )
            assert misc.is_isomorphic_graph_list(
                list(extension.all_extensions(graph, dim, k_min=k, k_max=k)),
                list(extension.all_k_extensions(graph, k, dim)),
            )


@pytest.mark.parametrize(
    "graph, dim, k_min, k_max",
    [
        [graphs.Diamond(), 2, -1, 0],
        [graphs.ThreePrism(), 2, 0, -1],
        [graphs.Diamond(), 2, 2, 1],
        [graphs.Diamond(), 2, 3, None],
        [graphs.Complete(4), 3, -2, -1],
        [graphs.CompleteMinusOne(5), 1, 5, 4],
        [graphs.Complete(3), 3, 5, None],
    ],
)
def test_all_extensions_value_error(graph, dim, k_min, k_max):
    with pytest.raises(ValueError):
        list(graph.all_extensions(dim=dim, k_min=k_min, k_max=k_max))
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.raises(ValueError):
            list(extension.all_extensions(graph, dim=dim, k_min=k_min, k_max=k_max))

@pytest.mark.parametrize(
    "graph, dim, k_min, k_max",
    [
        [graphs.Diamond(), 2, 0, 1.4],
        [graphs.Diamond(), 2, 0.2, 2],
        [graphs.Diamond(), 1.2, 2, 1],
        [graphs.Diamond(), "2", 2, 1],
        [graphs.Diamond(), 1, 2, "1"],
        [graphs.Diamond(), 2, 3 / 2, None],
        [graphs.Diamond(), 2, "2", None],
        [graphs.Diamond(), None, 2, 1],
        [graphs.Diamond(), 1, None, 1],
    ],
)
def test_all_extensions_type_error(graph, dim, k_min, k_max):
    with pytest.raises(TypeError):
        list(graph.all_extensions(dim=dim, k_min=k_min, k_max=k_max))
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.raises(TypeError):
            list(extension.all_extensions(graph, dim=dim, k_min=k_min, k_max=k_max))


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Complete(3),
        graphs.CompleteBipartite(3, 3),
        graphs.Diamond(),
        graphs.ThreePrism(),
        graphs.CubeWithDiagonal(),
        Graph.from_int(6462968),
        Graph.from_int(69380589),
        Graph.from_int(19617907),
        Graph.from_int(170993054),
        Graph.from_int(173090142),
    ],
)
def test_has_extension_sequence(graph):
    assert graph.has_extension_sequence()
    if TEST_WRAPPED_FUNCTIONS:
        assert extension.has_extension_sequence(graph)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Path(3),
        graphs.CompleteBipartite(1, 2),
        graphs.Complete(4),
        graphs.Cycle(6),
        graphs.K33plusEdge(),
        graphs.ThreePrismPlusEdge(),
        Graph.from_int(2269176),
        Graph.from_int(19650659),
        Graph.from_vertices([0]),
        Graph.from_vertices([]),
    ],
)
def test_has_not_extension_sequence(graph):
    assert not graph.has_extension_sequence()
    if TEST_WRAPPED_FUNCTIONS:
        assert not extension.has_extension_sequence(graph)


def test_extension_sequence_solution():
    assert graphs.Complete(2).extension_sequence(return_type="graphs") == [
        Graph([[0, 1]]),
    ]

    assert graphs.Complete(3).extension_sequence(return_type="graphs") == [
        Graph([[1, 2]]),
        Graph([[0, 1], [0, 2], [1, 2]]),
    ]

    solution = [
        Graph([[3, 4]]),
        Graph([[2, 3], [2, 4], [3, 4]]),
        Graph([[1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]),
        Graph([[1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4]]),
        Graph(
            [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]],
        ),
    ]
    assert (
        graphs.CompleteBipartite(3, 3).extension_sequence(return_type="graphs")
        == solution
    )

    solution_ext = [
        [0, [3, 4], [], 2],  # k, vertices, edges, new_vertex
        [0, [3, 4], [], 1],
        [0, [1, 2], [], 5],
        [1, [3, 4, 5], [(3, 4)], 0],
    ]
    G = Graph([[3, 4]])
    for i in range(len(solution)):
        assert solution[i] == G
        if i < len(solution_ext):
            G.k_extension(*solution_ext[i], dim=2, inplace=True)

    assert graphs.Diamond().extension_sequence(return_type="graphs") == [
        Graph([[2, 3]]),
        Graph([[0, 2], [0, 3], [2, 3]]),
        Graph([[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]]),
    ]

    result = graphs.ThreePrism().extension_sequence(return_type="graphs")
    solution = [
        Graph([[4, 5]]),
        Graph([[3, 4], [3, 5], [4, 5]]),
        Graph([[1, 3], [1, 4], [3, 4], [3, 5], [4, 5]]),
        Graph([[1, 2], [1, 3], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]]),
        Graph(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]],
        ),
    ]
    assert solution == result
    solution_ext = [
        [0, [4, 5], [], 3],  # k, vertices, edges, new_vertex
        [0, [3, 4], [], 1],
        [0, [1, 5], [], 2],
        [1, [1, 2, 3], [(1, 3)], 0],
    ]
    G = Graph([[4, 5]])
    for i in range(len(result)):
        assert result[i] == G
        if i < len(solution_ext):
            G.k_extension(*solution_ext[i], dim=2, inplace=True)
    if TEST_WRAPPED_FUNCTIONS:
        assert extension.extension_sequence(graphs.Complete(2), return_type="graphs") == [
            Graph([[0, 1]]),
        ]

        assert extension.extension_sequence(graphs.Complete(3), return_type="graphs") == [
            Graph([[1, 2]]),
            Graph([[0, 1], [0, 2], [1, 2]]),
        ]

        solution = [
            Graph([[3, 4]]),
            Graph([[2, 3], [2, 4], [3, 4]]),
            Graph([[1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]),
            Graph([[1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4]]),
            Graph(
                [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]],
            ),
        ]
        assert (
            extension.extension_sequence(graphs.CompleteBipartite(3, 3), return_type="graphs")
            == solution
        )

        solution_ext = [
            [0, [3, 4], [], 2],  # k, vertices, edges, new_vertex
            [0, [3, 4], [], 1],
            [0, [1, 2], [], 5],
            [1, [3, 4, 5], [(3, 4)], 0],
        ]
        G = Graph([[3, 4]])
        for i in range(len(solution)):
            assert solution[i] == G
            if i < len(solution_ext):
                extension.k_extension(G, *solution_ext[i], dim=2, inplace=True)

        assert extension.extension_sequence(graphs.Diamond(), return_type="graphs") == [
            Graph([[2, 3]]),
            Graph([[0, 2], [0, 3], [2, 3]]),
            Graph([[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]]),
        ]

        result = extension.extension_sequence(graphs.ThreePrism(), return_type="graphs")
        solution = [
            Graph([[4, 5]]),
            Graph([[3, 4], [3, 5], [4, 5]]),
            Graph([[1, 3], [1, 4], [3, 4], [3, 5], [4, 5]]),
            Graph([[1, 2], [1, 3], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]]),
            Graph(
                [[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]],
            ),
        ]
        assert solution == result
        solution_ext = [
            [0, [4, 5], [], 3],  # k, vertices, edges, new_vertex
            [0, [3, 4], [], 1],
            [0, [1, 5], [], 2],
            [1, [1, 2, 3], [(1, 3)], 0],
        ]
        G = Graph([[4, 5]])
        for i in range(len(result)):
            assert result[i] == G
            if i < len(solution_ext):
                extension.k_extension(G, *solution_ext[i], dim=2, inplace=True)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Complete(3),
        graphs.CompleteBipartite(3, 3),
        graphs.Diamond(),
        graphs.ThreePrism(),
        graphs.CubeWithDiagonal(),
        Graph.from_int(6462968),
        Graph.from_int(69380589),
        Graph.from_int(19617907),
        Graph.from_int(170993054),
        Graph.from_int(173090142),
    ],
)
def test_extension_sequence(graph):
    ext = graph.extension_sequence(return_type="both")
    assert ext is not None
    current = ext[0]
    for i in range(1, len(ext)):
        current = current.k_extension(*ext[i][1])
        assert current == ext[i][0]
    if TEST_WRAPPED_FUNCTIONS:
        ext = extension.extension_sequence(graph, return_type="both")
        assert ext is not None
        current = ext[0]
        for i in range(1, len(ext)):
            current = extension.k_extension(current, *ext[i][1])
            assert current == ext[i][0]


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.Complete(2), 2],
        [graphs.Complete(3), 2],
        [graphs.CompleteBipartite(3, 3), 2],
        [graphs.Diamond(), 2],
        [graphs.ThreePrism(), 2],
        [graphs.CubeWithDiagonal(), 2],
        [Graph.from_int(6462968), 2],
        [Graph.from_int(69380589), 2],
        [Graph.from_int(19617907), 2],
        [Graph.from_int(170993054), 2],
        [Graph.from_int(173090142), 2],
        [graphs.Complete(2), 1],
        [Graph.from_int(75), 1],
        [Graph.from_int(77), 1],
        [Graph.from_int(86), 1],
        [graphs.Complete(1), 1],
        [graphs.Complete(4), 3],
        [graphs.CompleteMinusOne(5), 3],
        [Graph.from_int(16350), 3],
        [Graph.from_int(4095), 3],
        [graphs.DoubleBanana(), 3],
    ],
)
def test_extension_sequence_dim(graph, dim):
    ext = graph.extension_sequence(dim=dim, return_type="both")
    assert ext is not None
    current = ext[0]
    for i in range(1, len(ext)):
        current = current.k_extension(*ext[i][1], dim=dim)
        assert current == ext[i][0]
    if TEST_WRAPPED_FUNCTIONS:
        ext = extension.extension_sequence(graph, dim=dim, return_type="both")
        assert ext is not None
        current = ext[0]
        for i in range(1, len(ext)):
            current = extension.k_extension(current, *ext[i][1], dim=dim)
            assert current == ext[i][0]



@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.Complete(2), 2],
        [graphs.Complete(3), 2],
        [graphs.CompleteBipartite(3, 3), 2],
        [graphs.Diamond(), 2],
        [graphs.ThreePrism(), 2],
        [graphs.CubeWithDiagonal(), 2],
        [Graph.from_int(6462968), 2],
        [Graph.from_int(69380589), 2],
        [Graph.from_int(19617907), 2],
        [Graph.from_int(170993054), 2],
        [Graph.from_int(173090142), 2],
        [graphs.Complete(2), 1],
        [Graph.from_int(75), 1],
        [Graph.from_int(77), 1],
        [Graph.from_int(86), 1],
        [graphs.Complete(1), 1],
    ],
)
def test_extension_sequence_min_rigid(graph, dim):
    ext = graph.extension_sequence(dim=dim, return_type="graphs")
    assert ext is not None
    for current in ext:
        assert current.is_min_rigid(dim)
    if TEST_WRAPPED_FUNCTIONS:
        ext = extension.extension_sequence(graph, dim=dim, return_type="graphs")
        assert ext is not None
        for current in ext:
            assert generic_rigidity.is_min_rigid(current, dim)


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Path(3),
        graphs.CompleteBipartite(1, 2),
        graphs.Complete(4),
        graphs.Cycle(6),
        graphs.K33plusEdge(),
        graphs.ThreePrismPlusEdge(),
        Graph.from_int(2269176),
        Graph.from_int(19650659),
        Graph.from_vertices([0]),
        Graph.from_vertices([]),
    ],
)
def test_extension_sequence_none(graph):
    assert graph.extension_sequence() is None
    if TEST_WRAPPED_FUNCTIONS:
        assert extension.extension_sequence(graph) is None


@pytest.mark.parametrize(
    "graph, dim",
    [
        [graphs.Path(3), 2],
        [graphs.CompleteBipartite(1, 2), 2],
        [graphs.Complete(4), 2],
        [graphs.Cycle(6), 2],
        [graphs.K33plusEdge(), 2],
        [graphs.ThreePrismPlusEdge(), 2],
        [Graph.from_int(2269176), 2],
        [Graph.from_int(19650659), 2],
        [Graph.from_vertices([0]), 2],
        [Graph.from_vertices([]), 2],
        [graphs.Cycle(3), 1],
        [graphs.Complete(4), 1],
        [graphs.Complete(6), 3],
    ],
)
def test_extension_sequence_dim_none(graph, dim):
    assert graph.extension_sequence(dim) is None
    if TEST_WRAPPED_FUNCTIONS:
        assert extension.extension_sequence(graph, dim) is None


def test_extension_sequence_error():
    with pytest.raises(NotSupportedValueError):
        graphs.Complete(3).extension_sequence(return_type="Test")
    if TEST_WRAPPED_FUNCTIONS:
        with pytest.raises(NotSupportedValueError):
            extension.extension_sequence(graphs.Complete(3), return_type="Test")
