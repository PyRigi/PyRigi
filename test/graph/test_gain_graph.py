"""Tests for the GainGraph class."""

import networkx as nx
import pytest

from pyrigi.graph import GainGraph
from pyrigi.graph.gain_graph import GainGraphZd

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_init_empty():
    G = GainGraph()

    assert G.dim is None
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


def test_init_with_dim():
    G = GainGraph(dim=3)

    assert G.dim == 3
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


def test_init_from_edges():
    G = GainGraph(
        [
            (0, 1, (1, 0)),
            (0, 1, (0, 1)),
            (1, 2, (1, 1)),
            (2, 2, (2, 0)),
        ]
    )

    assert G.dim == 2
    assert set(G.vertex_list()) == {0, 1, 2}
    assert G.number_of_edges() == 4


def test_init_from_edge_generator():
    edges = [
        edge
        for edge in [
            (0, 1, (1, 0)),
            (1, 2, (0, 1)),
        ]
    ]

    G = GainGraph(edges)
    assert G.dim == 2
    assert G.number_of_edges() == 2


def test_init_from_networkx_simple_graph():
    H = nx.Graph()
    H.add_edge(0, 1, gain=(1, 2))

    G = GainGraph(H)

    assert G.dim == 2
    assert G.gains_between(0, 1) == [(1, 2)]


def test_init_from_edges_infer_dimension():
    G = GainGraph([(0, 1, (1, 2, 3))])

    assert G.dim == 3


def test_init_dispatches_gain_edges_to_zd_class():
    G = GainGraph([(0, 1, [1, 2])])

    assert type(G) is GainGraphZd
    assert G.dim == 2
    assert G.number_of_edges() == 1


def test_init_falls_back_to_multigraph_data():
    G = GainGraph([(0, 1)])

    assert type(G) is GainGraphZd
    assert isinstance(G, nx.MultiGraph)
    assert G.number_of_edges() == 1


def test_init_from_empty_edges_with_dim():
    G = GainGraph([], dim=2)

    assert G.dim == 2
    assert G.number_of_edges() == 0


@pytest.mark.parametrize("dim", [0, -1])
def test_init_invalid_dim_value(dim):
    with pytest.raises(ValueError):
        GainGraph(dim=dim)


@pytest.mark.parametrize("dim", [True, 1.5, "2"])
def test_init_invalid_dim_type(dim):
    with pytest.raises(TypeError):
        GainGraph(dim=dim)


@pytest.mark.parametrize("graph_class", [GainGraph, GainGraphZd])
def test_from_vertices_and_edges_accepts_generator(graph_class):
    edges = (
        (u, v, gain)
        for u, v, gain in [
            (0, 1, (1, 0)),
            (1, 2, (0, 1)),
        ]
    )

    G = graph_class.from_vertices_and_edges([0, 1, 2], edges)

    assert G.number_of_edges() == 2
    assert G.edge_list() == [
        (0, 1, (1, 0)),
        (1, 2, (0, 1)),
    ]


@pytest.mark.parametrize("graph_class", [GainGraph, GainGraphZd])
def test_from_vertices_and_edges_invalid_edge(graph_class):
    with pytest.raises(TypeError):
        graph_class.from_vertices_and_edges(
            [0, 1],
            [(0, 1)],
        )


# ---------------------------------------------------------------------------
# Gain validation
# ---------------------------------------------------------------------------


def test_add_edge_invalid_gain_dimension():
    G = GainGraph(dim=2)

    with pytest.raises(TypeError):
        G.add_edge(0, 1, gain=(1,))


def test_add_edge_invalid_gain_type():
    G = GainGraph(dim=2)

    with pytest.raises(TypeError):
        G.add_edge(0, 1, gain=(1, 2.5))


def test_add_edge_invalid_boolean_gain():
    G = GainGraph(dim=2)

    with pytest.raises(TypeError):
        G.add_edge(0, 1, gain=(True, 0))


def test_add_edge_empty_gain():
    G = GainGraph()

    with pytest.raises(ValueError):
        G.add_edge(0, 1, gain=())


def test_add_edge_scalar_gain():
    G = GainGraph()

    with pytest.raises(TypeError):
        G.add_edge(0, 1, gain=1)


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_add_edge_without_gain(dim):
    G = GainGraph(dim=dim)

    G.add_edge(0, 1)

    assert G.edge_list() == [(0, 1, (0,) * dim)]


@pytest.mark.parametrize(
    ("dim", "edges"),
    [
        (1, [(0, 1), (1, 2)]),
        (2, [(0, 1), (1, 2), (2, 3)]),
        (3, [(0, 1), (1, 2)]),
    ],
)
def test_add_edges_without_explicit_gains(dim, edges):
    G = GainGraph(dim=dim)

    G.add_edges(edges)

    expected_edges = [(u, v, (0,) * dim) for u, v in edges]

    assert G.edge_list() == sorted(expected_edges)


def test_add_edges_without_dimension_infers_two_dimensions():
    G = GainGraph()

    G.add_edges([(0, 1), (1, 2)])

    assert G.dim == 2
    assert G.edge_list() == [
        (0, 1, (0, 0)),
        (1, 2, (0, 0)),
    ]


def test_dimension_inferred_on_first_edge():
    G = GainGraph()

    G.add_edge(0, 1, gain=(1, 2, 3))

    assert G.dim == 3


def test_dimension_consistent_across_edges():
    G = GainGraph(dim=2)

    G.add_edge(0, 1, gain=(1, 2))

    with pytest.raises(TypeError):
        G.add_edge(1, 2, gain=(1, 2, 3))


# ---------------------------------------------------------------------------
# Adding edges
# ---------------------------------------------------------------------------


def test_add_edge():
    G = GainGraph(dim=2)

    G.add_edge(0, 1, gain=(1, 2))

    assert G.number_of_edges() == 1
    assert G.gains_between(0, 1)[0] == (1, 2)


def test_add_multiple_identical_edges():
    G = GainGraph(dim=2)

    G.add_edge(0, 1, gain=(1, 2))
    G.add_edge(0, 1, gain=(1, 2))
    G.add_edge(1, 0, gain=(-1, -2))

    assert G.number_of_edges() == 1


def test_add_gain_edge():
    G = GainGraph(dim=2)

    G.add_edge(0, 1, (1, 2))

    assert G.gains_between(0, 1)[0] == (1, 2)


def test_add_parallel_edges():
    G = GainGraph(dim=2)

    G.add_edge(0, 1, gain=(1, 0))
    G.add_edge(0, 1, gain=(0, 1))

    assert G.number_of_edges() == 2
    assert G.number_of_edges(0, 1) == 2
    assert set(G.gains_between(0, 1)) == {(1, 0), (0, 1)}


def test_add_parallel_edges_with_same_gain():
    G = GainGraph(dim=2)

    G.add_edge(0, 1, gain=(1, 0))
    G.add_edge(0, 1, gain=(1, 0))

    assert G.number_of_edges() == 1
    assert G.gains_between(0, 1) == [(1, 0)]


def test_add_edges():
    G = GainGraph(dim=2)

    G.add_edges(
        [
            (0, 1, (1, 0)),
            (0, 1, (0, 1)),
            (1, 2, (1, 1)),
        ]
    )

    assert G.number_of_edges() == 3
    assert G.number_of_nodes() == 3


@pytest.mark.parametrize(
    "edge",
    [
        None,
        1,
        (0,),
        (0, 1, (1,), "extra"),
    ],
)
def test_add_edges_invalid_input(edge):
    G = GainGraph(dim=2)

    with pytest.raises(nx.NetworkXError):
        G.add_edges([edge])


@pytest.mark.parametrize(
    "edge",
    [
        None,
        1,
        "01",
        (0, 1, (1, 0), "extra"),
    ],
)
def test_delete_edge_invalid_input(edge):
    G = GainGraph(dim=2)

    with pytest.raises(TypeError):
        G.delete_edge(edge)


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------


def test_add_loop():
    G = GainGraph(dim=2)

    G.add_edge(0, 0, gain=(1, 2))

    assert G.number_of_edges() == 1
    assert G.number_of_edges(0, 0) == 1
    assert G.gains_between(0, 0)[0] == (1, 2)


def test_add_multiple_loops():
    G = GainGraph(dim=2)

    G.add_edge(0, 0, gain=(1, 0))
    G.add_edge(0, 0, gain=(0, 1))

    assert G.number_of_edges() == 2
    assert set(G.gains_between(0, 0)) == {(1, 0), (0, 1)}


def test_loop_with_nonzero_gain():
    G = GainGraph([(0, 0, (1, 2))])

    assert G.gains_between(0, 0)[0] == (1, 2)


# ---------------------------------------------------------------------------
# Edge getters and orientation
# ---------------------------------------------------------------------------


def test_edge_list():
    G = GainGraph(
        [
            (0, 1, (1, 0)),
            (0, 1, (0, 1)),
            (1, 2, (1, 1)),
        ]
    )

    assert G.edge_list() == [
        (0, 1, (0, 1)),
        (0, 1, (1, 0)),
        (1, 2, (1, 1)),
    ]


def test_edge_gain_forward_orientation():
    G = GainGraph([(0, 1, (1, 2))])

    assert G.gains_between(0, 1)[0] == (1, 2)


def test_edge_gain_reverse_orientation():
    G = GainGraph([(0, 1, (1, 2))])

    assert G.gains_between(1, 0)[0] == (-1, -2)


def test_edge_gain_missing_edge():
    G = GainGraph(dim=2)

    with pytest.raises(IndexError):
        G.gains_between(0, 1)[0]


def test_gains_between():
    G = GainGraph(
        [
            (0, 1, (1, 0)),
            (0, 1, (0, 1)),
        ]
    )

    assert G.gains_between(0, 1) == [(1, 0), (0, 1)]


def test_gains_between_reverse_orientation():
    G = GainGraph(
        [
            (0, 1, (1, 0)),
            (0, 1, (0, 1)),
        ]
    )

    assert G.gains_between(1, 0) == [(-1, 0), (0, -1)]


# ---------------------------------------------------------------------------
# Deleting edges
# ---------------------------------------------------------------------------


def test_delete_edge_by_endpoints():
    G = GainGraph(
        [
            (0, 1, (1, 0)),
            (0, 1, (0, 1)),
        ]
    )

    G.delete_edge((0, 1))

    assert G.number_of_edges() == 1


def test_delete_edge():
    G = GainGraph(dim=2)

    G.add_edge(0, 1, gain=(1, 0))
    G.add_edge(0, 1, gain=(0, 1))

    G.delete_edge((0, 1, (1, 0)))

    assert G.number_of_edges() == 1
    assert G.gains_between(0, 1)[0] == (0, 1)


def test_delete_edge_by_gain():
    G = GainGraph(
        [
            (0, 1, (1, 0)),
            (0, 1, (0, 1)),
        ]
    )

    G.delete_edge((0, 1, (1, 0)))

    assert G.number_of_edges() == 1
    assert G.gains_between(0, 1)[0] == (0, 1)


def test_delete_edge_by_reverse_orientation():
    G = GainGraph([(0, 1, (1, 2))])

    G.delete_edge((1, 0, (-1, -2)))

    assert G.number_of_edges() == 0


def test_delete_loop_by_gain():
    G = GainGraph([(0, 0, (1, 2))])

    G.delete_edge((0, 0, (-1, -2)))

    assert G.number_of_edges() == 0


def test_delete_edge_missing():
    G = GainGraph([(0, 1, (1, 0))])

    with pytest.raises(ValueError):
        G.delete_edge((0, 1, (0, 1)))


def test_delete_edges():
    G = GainGraph(
        [
            (0, 1, (1, 0)),
            (0, 1, (0, 1)),
            (1, 2, (1, 1)),
        ]
    )

    G.delete_edges([(0, 1, (1, 0)), (1, 2, (1, 1))])

    assert G.number_of_edges() == 1
    assert G.gains_between(0, 1)[0] == (0, 1)


def test_delete_loops():
    G = GainGraph(
        [
            (0, 0, (1, 0)),
            (1, 1, (0, 1)),
            (0, 1, (1, 1)),
        ]
    )

    G.delete_loops()

    assert G.number_of_edges() == 1
    assert G.edge_list() == [(0, 1, (1, 1))]


# ---------------------------------------------------------------------------
# Vertices
# ---------------------------------------------------------------------------


def test_from_vertices():
    G = GainGraphZd.from_vertices([0, 1, 2], dim=2)

    assert set(G.vertex_list()) == {0, 1, 2}
    assert G.number_of_edges() == 0
    assert G.dim == 2


def test_from_vertices_and_edges():
    G = GainGraphZd.from_vertices_and_edges(
        [0, 1, 2],
        [
            (0, 1, (1, 0)),
            (0, 1, (0, 1)),
            (2, 2, (1, 1)),
        ],
    )

    assert set(G.vertex_list()) == {0, 1, 2}
    assert G.number_of_edges() == 3
    assert G.dim == 2


def test_from_vertices_and_edges_missing_vertex():
    with pytest.raises(ValueError):
        GainGraphZd.from_vertices_and_edges(
            [0, 1],
            [(0, 2, (1, 0))],
        )


def test_add_vertices():
    G = GainGraph(dim=2)

    G.add_vertices([0, 1, 2])

    assert set(G.vertex_list()) == {0, 1, 2}


def test_add_vertex():
    G = GainGraph(dim=2)

    G.add_vertex(0)

    assert G.vertex_list() == [0]


def test_delete_vertex():
    G = GainGraph(
        [
            (0, 1, (1, 0)),
            (1, 2, (0, 1)),
        ]
    )

    G.delete_vertex(1)

    assert set(G.vertex_list()) == {0, 2}
    assert G.number_of_edges() == 0


def test_delete_vertices():
    G = GainGraph(
        [
            (0, 1, (1, 0)),
            (1, 2, (0, 1)),
            (2, 3, (1, 1)),
        ]
    )

    G.delete_vertices([1, 2])

    assert set(G.vertex_list()) == {0, 3}
    assert G.number_of_edges() == 0


# ---------------------------------------------------------------------------
# NetworkX compatibility
# ---------------------------------------------------------------------------


def test_is_multigraph():
    G = GainGraph(dim=2)

    assert isinstance(G, nx.MultiGraph)
    assert G.is_multigraph()


def test_degree_with_loops():
    G = GainGraph(
        [
            (0, 0, (1, 0)),
            (0, 1, (0, 1)),
            (0, 1, (1, 1)),
        ]
    )

    # NetworkX counts a loop twice in the degree.
    assert G.degree(0) == 4


def test_init_from_networkx_multigraph():
    H = nx.MultiGraph()
    H.add_edge(0, 1, gain=(1, 0))
    H.add_edge(0, 1, gain=(0, 1))
    H.add_edge(1, 1, gain=(1, 1))
    H.add_edge(1, 1, gain=(-1, -1))
    H.add_edge(1, 0, gain=(0, -1))
    """
    INFO: note that contrary to pyrigi, networkx
    sorts the vertices first. Therefore, the last
    method cal gives a different result than pyrigi.
    """

    G = GainGraph(H)

    print(G.edge_list)
    assert G.number_of_edges() == 4
    assert G.dim == 2
    assert G.gains_between(0, 1)[0] == (1, 0)
    assert G.gains_between(0, 1)[1] == (0, 1)
    assert G.gains_between(0, 1)[2] == (0, -1)
    assert G.gains_between(1, 1)[0] == (1, 1)


# ---------------------------------------------------------------------------
# Copying and equality
# ---------------------------------------------------------------------------


def test_copy():
    G = GainGraph(
        [
            (0, 1, (1, 0)),
            (0, 1, (0, 1)),
            (1, 1, (1, 1)),
        ]
    )

    H = G.copy()

    assert H == G
    assert H is not G
    assert H.dim == G.dim


def test_copy_preserves_attributes():
    G = GainGraph(dim=2)
    G.add_node(0)
    G.add_edge(0, 1, gain=(1, 0))

    H = G.copy()

    assert H.graph == G.graph


def test_equality():
    G = GainGraph(
        [
            (0, 1, (1, 0)),
            (0, 1, (0, 1)),
        ]
    )

    H = GainGraph(
        [
            (0, 1, (1, 0)),
            (0, 1, (0, 1)),
        ]
    )

    assert G == H


def test_equality_reversed_edge_orientation():
    G = GainGraph([(0, 1, (1, 2))])
    H = GainGraph([(1, 0, (-1, -2))])

    assert G == H


def test_equality_reversed_loop_orientation():
    G = GainGraph([(0, 0, (1, 2))])
    H = GainGraph([(0, 0, (-1, -2))])

    assert G == H


def test_equality_different_multiplicity():
    G = GainGraph(
        [
            (0, 1, (1, 0)),
            (0, 1, (1, 0)),
        ]
    )
    H = GainGraph([(0, 1, (1, 0))])

    assert G == H


def test_inequality_different_gains():
    G = GainGraph([(0, 1, (1, 0))])
    H = GainGraph([(0, 1, (0, 1))])

    assert G != H


def test_inequality_different_dimension():
    G = GainGraph([(0, 1, (1, 0))])
    H = GainGraph([(0, 1, (1, 0, 0))])

    assert G != H


def test_inequality_different_vertices():
    G = GainGraph([(0, 1, (1, 0))])
    H = GainGraph([(0, 2, (1, 0))])

    assert G != H


def test_repr_roundtrip():
    G = GainGraph(
        [
            (0, 1, (1, 0)),
            (0, 1, (0, 1)),
            (1, 1, (1, 1)),
        ]
    )

    H = eval(repr(G), {"GainGraphZd": GainGraphZd})

    assert H == G


@pytest.mark.parametrize(
    ("left_edges", "right_edges"),
    [
        pytest.param(
            [(0, 1, (1, 0))],
            [(1, 2, (0, 1))],
            id="2D-adjacent-edges",
        ),
        pytest.param(
            [("a", "b", (1,))],
            [("b", "c", (2,))],
            id="1D-string-vertices",
        ),
        pytest.param(
            [(0, 0, (1, 0))],
            [(0, 1, (0, 1))],
            id="2D-loop-and-edge",
        ),
    ],
)
def test_parametrized_graph_addition(left_edges, right_edges):
    G = GainGraph(left_edges)
    H = GainGraph(right_edges)

    G_before = G.copy()
    H_before = H.copy()

    K = G + H
    L = H + G

    # Addition should be commutative.
    assert K == L

    # Addition should not modify either operand.
    assert G == G_before
    assert H == H_before

    # All vertices and edges should be present.
    assert set(K.nodes) == set(G.nodes) | set(H.nodes)
    assert K.number_of_edges() == len(set(G.edge_list()) | set(H.edge_list()))


def test_add_graph_with_empty_graph():
    G = GainGraph([(0, 1, (1, 2))])
    H = GainGraph(dim=2)

    K = G + H

    assert K == G
    assert K.dim == 2


def test_add_empty_graphs_with_different_known_dimensions():
    G = GainGraph(dim=2)
    H = GainGraph(dim=2)

    K = G + H

    assert K.dim == 2
    assert K.number_of_nodes() == 0
    assert K.number_of_edges() == 0


def test_add_graphs_with_reversed_duplicate_edge():
    G = GainGraph([(0, 1, (1, 2))])
    H = GainGraph([(1, 0, (-1, -2))])

    K = G + H

    # Reversed orientations with inverse gains represent the same edge.
    assert K.number_of_edges() == 1
    assert K.gains_between(0, 1) == [(1, 2)]


GAIN_GRAPH_CASES = [
    pytest.param(
        [(0, 1, (2,)), (1, 2, (-3,))],
        1,
        [0, 1, 2],
        id="1D",
    ),
    pytest.param(
        [
            (0, 1, (1, 0)),
            (0, 1, (0, 1)),
            (1, 2, (-1, 1)),
        ],
        2,
        [0, 1, 2],
        id="2D-parallel-edges",
    ),
    pytest.param(
        [
            ("a", "b", (1, -2, 3)),
            ("b", "c", (0, 1, 0)),
            ("c", "c", (2, 0, -1)),
        ],
        3,
        ["a", "b", "c"],
        id="3D-with-loop",
    ),
]


@pytest.mark.parametrize(
    ("edges", "expected_dim", "expected_vertices"),
    GAIN_GRAPH_CASES,
)
def test_gain_graph_construction(edges, expected_dim, expected_vertices):
    G = GainGraph(edges)

    assert isinstance(G, GainGraphZd)
    assert G.dim == expected_dim
    assert G.vertex_list() == expected_vertices
    assert G.edge_list() == sorted(edges)


@pytest.mark.parametrize(
    ("edges", "expected_dim", "expected_vertices"),
    GAIN_GRAPH_CASES,
)
def test_gain_graph_orientation(edges, expected_dim, expected_vertices):
    G = GainGraph(edges)

    assert G.dim == expected_dim
    assert G.vertex_list() == expected_vertices
    for u, v, gain in edges:
        assert gain in G.gains_between(u, v)

        if u != v:
            inverse_gain = tuple(-x for x in gain)
            assert inverse_gain in G.gains_between(v, u)


@pytest.mark.parametrize(
    ("edges", "expected_dim", "expected_vertices"),
    GAIN_GRAPH_CASES,
)
def test_gain_graph_copy(edges, expected_dim, expected_vertices):
    G = GainGraph(edges)
    H = G.copy()

    assert H == G
    assert H is not G
    assert H.dim == G.dim == expected_dim
    assert H.vertex_list() == G.vertex_list() == expected_vertices
    assert H.edge_list() == G.edge_list()


@pytest.mark.parametrize(
    ("edges", "edge_to_delete"),
    [
        pytest.param(
            [(0, 1, (1, 0)), (0, 1, (0, 1))],
            (0, 1, (1, 0)),
            id="2D-parallel-edges",
        ),
        pytest.param(
            [
                ("left", "right", (1, -2, 3)),
                ("right", "top", (0, 1, 0)),
            ],
            ("left", "right", (1, -2, 3)),
            id="3D",
        ),
    ],
)
def test_delete_edge_reverse_orientation(edges, edge_to_delete):
    G = GainGraph(edges)
    H = G.copy()

    u, v, gain = edge_to_delete
    inverse_gain = tuple(-x for x in gain)

    H.delete_edge((v, u, inverse_gain))

    assert H.number_of_edges() == G.number_of_edges() - 1
    assert gain in G.gains_between(u, v)
    assert gain not in H.gains_between(u, v)


# ---------------------------------------------------------------------------
# Dimension handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("initial_gain", "new_dim", "expected_gain"),
    [
        ((1,), 3, (1, 0, 0)),
        ((1, 2), 4, (1, 2, 0, 0)),
        ((1, 2, 3), 5, (1, 2, 3, 0, 0)),
    ],
)
def test_set_dim_pads_gains_with_zeros(initial_gain, new_dim, expected_gain):
    G = GainGraph([(0, 1, initial_gain)])

    G.set_dim(new_dim)

    assert G.dim == new_dim
    assert G.edge_list() == [(0, 1, expected_gain)]


def test_set_dim_for_empty_graph():
    G = GainGraph(dim=2)

    G.set_dim(4)

    assert G.dim == 4
    assert G.number_of_edges() == 0


@pytest.mark.parametrize(
    ("invalid_dim", "exception"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_set_dim_invalid_value(invalid_dim, exception):
    G = GainGraph(dim=2)

    with pytest.raises(exception):
        G.set_dim(invalid_dim)

    assert G.dim == 2


# ---------------------------------------------------------------------------
# Additional NetworkX compatibility tests
# ---------------------------------------------------------------------------


def test_init_from_networkx_multigraph_preserves_attributes():
    H = nx.MultiGraph()

    H.graph["name"] = "test_graph"
    H.add_node(0, label="first")
    H.add_node(1, label="second")
    H.add_edge(0, 1, gain=(1, 2))

    G = GainGraph(H)

    assert G.dim == 2
    assert G.graph["name"] == "test_graph"
    assert G.nodes[0]["label"] == "first"
    assert G.nodes[1]["label"] == "second"
    assert G.gains_between(0, 1) == [(1, 2)]


def test_init_from_empty_networkx_multigraph_with_dim():
    H = nx.MultiGraph()
    H.dim = 3

    G = GainGraph(H)

    assert G.dim == 3
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


def test_init_from_networkx_multigraph_without_gain():
    H = nx.MultiGraph()
    H.add_edge(0, 1)

    with pytest.raises(TypeError):
        GainGraph(H)


def test_init_from_networkx_multigraph_wrong_gain_dimension():
    H = nx.MultiGraph()
    H.dim = 2
    H.add_edge(0, 1, gain=(1, 2, 3))

    with pytest.raises(TypeError):
        GainGraph(H)


def test_vertex_list_with_incomparable_vertices():
    G = GainGraph(dim=2)
    G.add_vertices([1, "a", 2])

    # Sorting incomparable types is not possible.
    # The insertion order should be preserved.
    assert G.vertex_list() == [1, "a", 2]


def test_edge_list_with_incomparable_vertices():
    G = GainGraph(dim=1)
    G.add_edge(1, 2, gain=(1,))
    G.add_edge("a", "b", gain=(2,))

    assert G.edge_list() == [
        (1, 2, (1,)),
        ("a", "b", (2,)),
    ]
