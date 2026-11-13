import warnings

import pytest
from sympy import Matrix, sqrt, simplify

from pyrigi import GainGraph, PeriodicFramework
from pyrigi.warning import OutsideFundamentalCellWarning


def example_framework():
    graph = GainGraph(
        [
            (0, 1, (1, 0)),
            (0, 1, (0, 1)),
            (1, 1, (1, -1)),
        ]
    )
    return PeriodicFramework(
        graph,
        {0: (0, 0), 1: (1, 0)},
        [(3, 0), (0, 3)],
    )


def test_initialization_and_getters():
    framework = example_framework()

    assert framework.dim == 2
    assert framework.lattice_dim == 2
    assert framework.lattice_basis == Matrix([[3, 0], [0, 3]])
    assert isinstance(framework.graph, GainGraph)
    assert framework.realization(as_points=True) == {
        0: [0, 0],
        1: [1, 0],
    }
    assert framework[0] == Matrix([0, 0])


@pytest.mark.parametrize(
    "realization, lattice, target_realization",
    [
        [
            {0: ["1/2", "1/2"], 1: ["3/4", 0]},
            [("1/2", 0), (0, "1/2")],
            {0: [0, 0], 1: ["1/4", 0]},
        ],
        [{0: [-1, 0], 1: [0, 1]}, [(2, 0), (0, 2)], {0: [1, 0], 1: [0, 1]}],
        [
            {0: [1, 2], 1: ["12/7", "7/3"]},
            [(2, 3), (0, 2)],
            {0: [1, 2], 1: ["12/7", "13/3"]},
        ],
        [
            {0: [sqrt(2), 2], 1: [sqrt(5), -sqrt(3)]},
            [(-1, 1), (1, 1)],
            {0: [-1 + sqrt(2), 1], 1: ["sqrt(5)-2", "-sqrt(3)+2"]},
        ],
    ],
)
def test__normalization_mod_lattice(realization, lattice, target_realization):
    framework = PeriodicFramework(
        GainGraph.from_vertices([0, 1], dim=2),
        realization,
        lattice,
    )

    assert all(
        coord.is_zero
        for key, point in framework.realization(as_points=True).items()
        for coord in [
            simplify(u) - simplify(v) for u, v in zip(point, target_realization[key])
        ]
    )


def test_edge_vectors_and_lengths():
    framework = example_framework()

    assert framework.edge_vector((0, 1, (1, 0))) == Matrix([4, 0])
    assert framework.edge_vector((0, 1, (0, 1))) == Matrix([1, 3])
    assert framework.edge_vector((1, 1, (1, -1))) == Matrix([3, -3])

    lengths = framework.edge_lengths()
    assert lengths[(0, 1, (1, 0))] == 4
    assert lengths[(0, 1, (0, 1))] == sqrt(10)
    assert lengths[(1, 1, (1, -1))] == 3 * sqrt(2)
    assert framework.edge_lengths(numerical=True)[(0, 1, (0, 1))] == pytest.approx(
        10 ** (0.5)
    )


def test_realization_setters_are_inherited():
    framework = example_framework()
    framework.set_vertex_pos(0, (2, 0))
    framework.set_vertex_positions({1: (1, 2)})
    assert framework.realization(as_points=True) == {
        0: [2, 0],
        1: [1, 2],
    }

    framework.set_realization({0: (1, 1), 1: (2, 2)})
    assert framework.realization(as_points=True) == {
        0: [1, 1],
        1: [2, 2],
    }


def test_lattice_basis_setter_and_defensive_getter():
    framework = example_framework()
    basis = framework.lattice_basis
    basis[0, 0] = 100
    assert framework.lattice_basis == Matrix([[3, 0], [0, 3]])

    framework.set_lattice_basis([(3, 0), (0, 4)])
    assert framework.lattice_basis == Matrix([[3, 0], [0, 4]])
    framework.lattice_basis = Matrix([[5, 0], [0, 6]])
    assert framework.lattice_basis == Matrix([[5, 0], [0, 6]])


def test_add_and_delete_gain_edges_and_vertices():
    framework = example_framework()
    framework.add_vertex((2, 2), "a")
    framework.add_edge((1, "a", (0, 0)))
    framework.add_edge((0, "a", (1, 0)))
    assert (1, "a", (0, 0)) in framework.graph.edge_list()
    assert (0, "a", (1, 0)) in framework.graph.edge_list()

    framework.delete_edge((1, "a", (0, 0)))
    assert (1, "a", (0, 0)) not in framework.graph.edge_list()
    framework.delete_vertex("a")
    assert "a" not in framework.graph


def test_edgeless_graph_infers_gain_dimension_from_basis():
    graph = GainGraph.from_vertices(["v"])
    framework = PeriodicFramework(graph, {"v": (1, 2)}, [(2, 0), (0, 2)])
    assert framework.lattice_dim == 2
    assert framework.graph.dim == 2


def test_lower_dim_basis_is_supported():
    graph = GainGraph([(0, 1, (2,))])
    framework = PeriodicFramework(graph, {0: (0, 0, 0), 1: (1, 1, 1)}, [(2, 0, 0)])
    assert framework.dim == 3
    assert framework.lattice_dim == 1
    assert framework.edge_vector((0, 1, (2,))) == Matrix([5, 1, 1])


def test_invalid_initialization_and_setters():
    with pytest.raises(TypeError):
        PeriodicFramework(object(), {}, [(1, 0), (0, 1)])

    graph = GainGraph([(0, 1, (1, 0))])
    with pytest.raises(ValueError, match="number of lattice basis vectors"):
        PeriodicFramework(graph, {0: (0, 0), 1: (1, 0)}, [(1, 0)])

    with pytest.raises(ValueError, match="linearly independent"):
        PeriodicFramework(
            graph,
            {0: (0, 0), 1: (1, 0)},
            [(1, 0), (2, 0)],
        )

    framework = example_framework()
    with pytest.raises(ValueError, match="shape"):
        framework.set_lattice_basis([(1, 0)])
    with pytest.raises(ValueError, match="dimension"):
        framework.set_vertex_pos(0, (1, 2, 3))
    with pytest.raises(ValueError, match="lattice dimension"):
        framework.lattice_vector((1,))


def test_empty_and_copy():
    framework = PeriodicFramework.Empty([(2, 0), (0, 2)])
    assert framework.dim == 2
    assert framework.lattice_dim == 2
    assert framework.realization(as_points=True) == {}

    original = example_framework()
    copied = original.copy()
    assert isinstance(copied, PeriodicFramework)
    assert copied.graph == original.graph
    assert copied.realization(as_points=True) == original.realization(as_points=True)
    assert copied.lattice_basis == original.lattice_basis
    copied.set_vertex_pos(0, (10, 10))
    print(original)
    assert original[0] == Matrix([0, 0])
    print(copied)
    assert copied[0] == Matrix([1, 1])


def test_getters_return_independent_graph_and_basis():
    framework = example_framework()
    copied_graph = framework.graph
    copied_graph.add_edge(0, 1, (2, 0))
    assert (0, 1, (2, 0)) not in framework.graph.edge_list()

    copied_basis = framework.lattice_basis
    copied_basis[0, 0] = 10
    assert framework.lattice_basis == Matrix([[3, 0], [0, 3]])


def test_edge_vector_requires_an_edge_of_the_gain_graph():
    framework = example_framework()
    with pytest.raises(ValueError, match="not contained in the graph"):
        framework.edge_vector((0, 1, (2, 0)))


def test_zero_gain_loops_are_rejected():
    with pytest.raises(ValueError, match="loop must have nonzero gain"):
        PeriodicFramework(
            GainGraph([(0, 0, (0, 0))]),
            {0: (0, 0)},
            [(1, 0), (0, 1)],
        )

    framework = PeriodicFramework(
        GainGraph.from_vertices([0, 1], dim=2),
        {0: (0, 0), 1: (1, 0)},
        [(1, 0), (0, 1)],
    )
    with pytest.raises(ValueError, match="loop must have nonzero gain"):
        framework.add_edge((0, 0))


def test_repr_round_trip():
    framework = example_framework()
    clone = eval(
        repr(framework),
        {"GainGraph": GainGraph, "PeriodicFramework": PeriodicFramework},
    )
    assert clone.graph == framework.graph
    assert clone.realization(as_points=True) == framework.realization(as_points=True)
    assert clone.lattice_basis == framework.lattice_basis


@pytest.mark.parametrize(
    "realization, lattice",
    [
        [{0: ["1/2", "1/2"], 1: ["3/4", 0]}, [(1, 0), (0, 1)]],
        [{0: [1, 0], 1: [0, 1]}, [(2, 0), (0, 2)]],
        [{0: [1, 1], 1: ["12/7", "7/3"]}, [(2, 2), (0, 2)]],
    ],
)
def test_not__warn_outside_fundamental_cell(realization, lattice):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # ensure warnings are captured

        PeriodicFramework(
            GainGraph.from_vertices([0, 1], dim=2),
            realization,
            lattice,
        )

        assert not any(
            issubclass(wi.category, OutsideFundamentalCellWarning) for wi in w
        )


@pytest.mark.parametrize(
    "realization, lattice",
    [
        [{0: ["1/2", "1/2"], 1: ["3/4", 0]}, [("1/2", 0), (0, "1/2")]],
        [{0: [-1, 0], 1: [0, 1]}, [(2, 0), (0, 2)]],
        [{0: [1, 2], 1: ["12/7", "7/3"]}, [(2, 3), (0, 2)]],
    ],
)
def test__warn_outside_fundamental_cell(realization, lattice):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # ensure warnings are captured

        PeriodicFramework(
            GainGraph.from_vertices([0, 1], dim=2),
            realization,
            lattice,
        )

        assert any(issubclass(wi.category, OutsideFundamentalCellWarning) for wi in w)
