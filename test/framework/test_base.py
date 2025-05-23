import pytest
from sympy import sympify

import pyrigi.frameworkDB as fws
from pyrigi._utils._zero_check import is_zero, is_zero_vector
from pyrigi.framework.base import FrameworkBase
from pyrigi.graph import Graph


def test__str__():
    assert (
        str(FrameworkBase(Graph([(0, 1)]), {0: (0, 0), 1: (1, 0)}))
        == """FrameworkBase in 2-dimensional space consisting of:
Graph with vertices [0, 1] and edges [[0, 1]]
Realization {0:(0, 0), 1:(1, 0)}"""
    )


def test__repr__():
    assert (
        repr(FrameworkBase(Graph([(0, 1)]), {0: (0, 0), 1: (1, 0)}))
        == "FrameworkBase(Graph.from_vertices_and_edges"
        "([0, 1], [(0, 1)]), {0: ['0', '0'], 1: ['1', '0']})"
    )
    F1 = FrameworkBase(Graph([(0, 1)]), {0: ["1/2"], 1: ["sqrt(2)"]})
    F2 = eval(repr(F1))
    assert F1[0] == F2[0] and F1[1] == F2[1]


def test_dimension():
    assert fws.Complete(2, 2).dim == 2
    assert FrameworkBase.Empty(dim=3).dim == 3


def test_vertex_addition():
    F = FrameworkBase.Empty()
    F.add_vertices([[1.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
    F_ = FrameworkBase.Empty()
    F_.add_vertices([[1.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
    F_.set_realization(F.realization())
    assert (
        F.realization() == F_.realization()
        and F.graph.vertex_list() == F_.graph.vertex_list()
        and F.dim == F_.dim
    )
    assert F.graph.vertex_list() == [0, 1, 2] and len(F.graph.edges()) == 0
    F.set_vertex_positions_from_lists([0, 2], [[3.0, 0.0], [0.0, 3.0]])
    F_.set_vertex_pos(1, [2.0, 2.0])
    array = F_.realization()
    array[0] = (3, 0)
    assert F[0] != F_[0] and F[1] != F_[1] and F[2] != F_[2]


def test_edge_lengths():
    G = Graph([(0, 1), (1, 2), (2, 3), (0, 3)])
    F = FrameworkBase(
        G, {0: [0, 0], 1: [1, 0], 2: [1, "1/2 * sqrt(5)"], 3: ["1/2", "4/3"]}
    )
    l_dict = F.edge_lengths(numerical=True)

    expected_result = {
        (0, 1): 1.0,
        (0, 3): 1.4240006242195884,
        (1, 2): 1.118033988749895,
        (2, 3): 0.5443838790578374,
    }

    for edge, length in l_dict.items():
        assert abs(length - expected_result[edge]) < 1e-10

    l_dict = F.edge_lengths(numerical=False)

    expected_result = {
        (0, 1): 1,
        (0, 3): "sqrt(1/4 + 16/9)",
        (1, 2): "1/2 * sqrt(5)",
        (2, 3): "sqrt(1/4 + (1/2 * sqrt(5) - 4/3)**2)",
    }

    for edge, length in l_dict.items():
        assert is_zero(sympify(expected_result[edge]) - length)

    F = fws.Cycle(6)
    assert is_zero_vector([v - 1 for v in F.edge_lengths(numerical=False).values()])


@pytest.mark.parametrize(
    "framework1, framework2",
    [
        [
            fws.Complete(3, dim=2),
            FrameworkBase(Graph.from_int(7), {0: [0, 0], 1: [1, 0], 2: [1, 1]}),
        ],
        [
            fws.Complete(4, dim=2),
            fws.Complete(4, dim=2),
        ],
    ],
)
def test__input_check_underlying_graphs(framework1, framework2):
    assert framework1._input_check_underlying_graphs(framework2) is None
    assert framework2._input_check_underlying_graphs(framework1) is None


@pytest.mark.parametrize(
    "framework1, framework2",
    [
        [
            fws.Complete(3, dim=2),
            FrameworkBase(
                Graph.from_int(31), {0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [2, 2]}
            ),
        ],
        [
            FrameworkBase(Graph([[1, 2], [2, 3]]), {1: [1, 0], 2: [1, 1], 3: [2, 2]}),
            FrameworkBase(Graph([[0, 1], [1, 2]]), {0: [0, 0], 1: [1, 0], 2: [1, 1]}),
        ],
    ],
)
def test__input_check_underlying_graphs_error(framework1, framework2):
    with pytest.raises(ValueError):
        framework1._input_check_underlying_graphs(framework2)


@pytest.mark.parametrize(
    "framework, realization, v",
    [
        [
            FrameworkBase(Graph([[1, 2], [2, 3]]), {1: [1, 0], 2: [1, 1], 3: [2, 2]}),
            None,
            1,
        ],
        [
            Graph([[1, 2], [2, 3]]).random_framework(),
            {1: [1, 0], 2: [1, 1], 3: [2, 2]},
            2,
        ],
        [
            Graph([["a", 2], [2, -3]]).random_framework(),
            {2: [1, 0], -3: [1, 1], "a": [2, 2]},
            2,
        ],
    ],
)
def test__input_check_vertex_key(framework, realization, v):
    assert framework._input_check_vertex_key(v, realization) is None


@pytest.mark.parametrize(
    "framework, realization, v",
    [
        [
            FrameworkBase(Graph([[1, 2], [2, 3]]), {1: [1, 0], 2: [1, 1], 3: [2, 2]}),
            None,
            4,
        ],
        [Graph([[1, 2], [2, 3]]).random_framework(), {1: [1, 0], 2: [1, 1]}, 3],
        [
            Graph([["a", 2], [2, -3]]).random_framework(),
            {2: [1, 0], -3: [1, 1], "a": [2, 2]},
            "b",
        ],
    ],
)
def test__input_check_vertex_key_error(framework, realization, v):
    with pytest.raises(KeyError):
        framework._input_check_vertex_key(v, realization)


@pytest.mark.parametrize(
    "framework, point",
    [
        [
            FrameworkBase(Graph([[1, 2], [2, 3]]), {1: [1, 0], 2: [1, 1], 3: [2, 2]}),
            [2, 3],
        ],
        [Graph([[1, 2], [2, 3]]).random_framework(dim=3), [2, 3, 4]],
        [Graph([["a", 2], [2, -3]]).random_framework(dim=1), [2]],
    ],
)
def test__input_check_point_dimension(framework, point):
    assert framework._input_check_point_dimension(point) is None


@pytest.mark.parametrize(
    "framework, point",
    [
        [
            FrameworkBase(Graph([[1, 2], [2, 3]]), {1: [1, 0], 2: [1, 1], 3: [2, 2]}),
            [2, 3, 3],
        ],
        [Graph([[1, 2], [2, 3]]).random_framework(dim=3), [2, 3]],
        [Graph([["a", 2], [2, -3]]).random_framework(dim=1), []],
    ],
)
def test__input_check_point_dimension_error(framework, point):
    with pytest.raises(ValueError):
        framework._input_check_point_dimension(point)
