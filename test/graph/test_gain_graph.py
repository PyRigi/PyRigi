"""Tests for the GainGraph class."""

import networkx as nx
import pytest

from pyrigi.graph import GainGraph

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


def test_init_from_edges_infer_dimension():
    G = GainGraph([(0, 1, (1, 2, 3))])

    assert G.dim == 3


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
    G = GainGraph.from_vertices([0, 1, 2], dim=2)

    assert set(G.vertex_list()) == {0, 1, 2}
    assert G.number_of_edges() == 0
    assert G.dim == 2


def test_from_vertices_and_edges():
    G = GainGraph.from_vertices_and_edges(
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
        GainGraph.from_vertices_and_edges(
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


def test_copy_is_independent():
    G = GainGraph([(0, 1, (1, 0))])
    H = G.copy()

    H.add_edge(1, 2, gain=(0, 1))

    assert G.number_of_edges() == 1
    assert H.number_of_edges() == 2


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

    H = eval(repr(G), {"GainGraph": GainGraph})

    assert H == G


# ---------------------------------------------------------------------------
# Graph addition
# ---------------------------------------------------------------------------


def test_add_graphs():
    G = GainGraph([(0, 1, (1, 0))])
    H = GainGraph([(1, 2, (0, 1))])

    K = G + H

    assert K.dim == 2
    assert set(K.vertex_list()) == {0, 1, 2}
    assert K.number_of_edges() == 2


def test_add_graphs_preserves_parallel_edges():
    G = GainGraph([(0, 1, (1, 0))])
    H = GainGraph([(0, 1, (0, 1))])

    K = G + H

    assert K.number_of_edges() == 2
    assert set(K.gains_between(0, 1)) == {(1, 0), (0, 1)}


def test_add_graph_to_itself():
    G = GainGraph([(0, 1, (1, 0))])

    H = G + G

    assert H.number_of_edges() == 1
    assert H.gains_between(0, 1) == [(1, 0)]


def test_add_graphs_different_dimensions():
    G = GainGraph([(0, 1, (1, 0))])
    H = GainGraph([(0, 1, (1, 0, 0))])

    with pytest.raises(ValueError):
        G + H
