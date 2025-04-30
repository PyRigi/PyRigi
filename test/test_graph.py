import math
from itertools import product
from random import randint

import matplotlib.pyplot as plt
import networkx as nx
import pytest
from sympy import Matrix

import pyrigi.graphDB as graphs
from pyrigi.exception import LoopError
from pyrigi.graph import Graph

is_rigid_algorithms_all_d = ["default", "randomized", "numerical"]
is_rigid_algorithms_d1 = is_rigid_algorithms_all_d + ["graphic"]
is_rigid_algorithms_d2 = is_rigid_algorithms_all_d + ["sparsity"]


TEST_WRAPPED_FUNCTIONS = True


def relabeled_inc(graph: Graph, increment: int = None) -> Graph:
    """
    Return the graph with each vertex label incremented by a given number.

    Note that ``graph`` must have integer vertex labels.
    """
    if increment is None:
        increment = graph.number_of_nodes()
    return nx.relabel_nodes(graph, {i: i + increment for i in graph.nodes()}, copy=True)


def read_sparsity(filename):
    return Graph(nx.read_sparse6("test/input_graphs/sparsity/" + filename + ".s6"))


def read_globally(d_v_):
    return read_random_from_graph6("test/input_graphs/globally_rigid/" + d_v_ + ".g6")


def read_redundantly(d_v_):
    return read_random_from_graph6(
        "test/input_graphs/redundantly_rigid/" + d_v_ + ".g6"
    )


def test__add__():
    G = Graph([[0, 1], [1, 2], [2, 0]])
    H = Graph([[0, 1], [1, 3], [3, 0]])
    assert G + H == Graph([[0, 1], [1, 2], [2, 0], [1, 3], [3, 0]])
    G = Graph([[0, 1], [1, 2], [2, 0]])
    H = Graph([[3, 4], [4, 5], [5, 3]])
    assert G + H == Graph([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]])
    G = Graph.from_vertices_and_edges([0, 1, 2, 3], [[0, 1], [1, 2]])
    H = Graph.from_vertices_and_edges([0, 1, 2, 4], [[0, 1]])
    assert G + H == Graph.from_vertices_and_edges([0, 1, 2, 3, 4], [[0, 1], [1, 2]])


def read_random_from_graph6(filename):
    file_ = nx.read_graph6(filename)
    if isinstance(file_, list):
        return Graph(file_[randint(0, len(file_) - 1)])
    else:
        return Graph(file_)


def test__str__():
    G = Graph([[2, 1], [2, 3]])
    assert str(G) == "Graph with vertices [1, 2, 3] and edges [[1, 2], [2, 3]]"
    G = Graph([(chr(i + 67), i + 1) for i in range(3)] + [(i, i + 1) for i in range(3)])
    assert str(G) == (
        "Graph with vertices ['C', 1, 'D', 2, 'E', 3, 0] "
        "and edges [('C', 1), (1, 0), (1, 2), ('D', 2), (2, 3), ('E', 3)]"
    )
    G = Graph.from_vertices(["C", 1, "D", 2, "E", 3, 0])
    assert str(G) == "Graph with vertices ['C', 1, 'D', 2, 'E', 3, 0] and edges []"


def test__repr__():
    assert (
        repr(Graph([[2, 1], [2, 3]]))
        == "Graph.from_vertices_and_edges([1, 2, 3], [(1, 2), (2, 3)])"
    )
    assert (
        repr(Graph.from_vertices_and_edges([1, 2, 3], [(1, 2)]))
        == "Graph.from_vertices_and_edges([1, 2, 3], [(1, 2)])"
    )


def test_vertex_and_edge_lists():
    G = Graph([[2, 1], [2, 3]])
    assert G.vertex_list() == [1, 2, 3]
    assert G.edge_list() == [[1, 2], [2, 3]]
    G = Graph([(chr(i + 67), i + 1) for i in range(3)] + [(i, i + 1) for i in range(3)])
    assert set(G.vertex_list()) == {"C", 1, "D", 2, "E", 3, 0}
    assert set(G.edge_list()) == {("C", 1), (1, 0), (1, 2), ("D", 2), (2, 3), ("E", 3)}
    G = Graph.from_vertices(["C", 1, "D", 2, "E", 3, 0])
    assert set(G.vertex_list()) == {"C", 2, "E", 1, "D", 3, 0}
    assert G.edge_list() == []


def test_adjacency_matrix():
    G = Graph()
    assert G.adjacency_matrix() == Matrix([])
    G = Graph([[2, 1], [2, 3]])
    assert G.adjacency_matrix() == Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert G.adjacency_matrix(vertex_order=[2, 3, 1]) == Matrix(
        [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
    )
    assert graphs.Complete(4).adjacency_matrix() == Matrix.ones(4) - Matrix.diag(
        [1, 1, 1, 1]
    )
    G = Graph.from_vertices(["C", 1, "D"])
    assert G.adjacency_matrix() == Matrix.zeros(3)
    G = Graph.from_vertices_and_edges(["C", 1, "D"], [[1, "D"], ["C", "D"]])
    assert G.adjacency_matrix(vertex_order=["C", 1, "D"]) == Matrix(
        [[0, 0, 1], [0, 0, 1], [1, 1, 0]]
    )
    M = Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert G.from_adjacency_matrix(M).adjacency_matrix() == M
    M = Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert G.from_adjacency_matrix(M).adjacency_matrix() == M


@pytest.mark.parametrize(
    "graph, gint",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 7],
        [graphs.Complete(4), 63],
        [graphs.CompleteBipartite(3, 4), 507840],
        [graphs.CompleteBipartite(4, 4), 31965120],
        [graphs.ThreePrism(), 29327],
    ],
)
def test_integer_representation(graph, gint):
    assert graph.to_int() == gint
    assert Graph.from_int(gint).is_isomorphic(graph)
    assert Graph.from_int(gint).to_int() == gint
    assert Graph.from_int(graph.to_int()).is_isomorphic(graph)


def test_integer_representation_error():
    with pytest.raises(ValueError):
        Graph([]).to_int()
    with pytest.raises(ValueError):
        M = Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        G = Graph.from_adjacency_matrix(M)
        G.to_int()
    with pytest.raises(ValueError):
        Graph.from_int(0)
    with pytest.raises(TypeError):
        Graph.from_int(1 / 2)
    with pytest.raises(TypeError):
        Graph.from_int(1.2)
    with pytest.raises(ValueError):
        Graph.from_int(-1)


def test_from_vertices_and_edges():
    G = Graph.from_vertices_and_edges([], [])
    assert G.vertex_list() == [] and G.edge_list() == []
    G = Graph.from_vertices_and_edges([0], [])
    assert G.vertex_list() == [0] and G.edge_list() == []
    G = Graph.from_vertices_and_edges([0, 1, 2, 3, 4, 5], [[0, 1]])
    assert G.vertex_list() == [0, 1, 2, 3, 4, 5] and G.edge_list() == [[0, 1]]
    G = Graph.from_vertices_and_edges([0, 1, 2], [[0, 1], [0, 2], [1, 2]])
    assert G.vertex_list() == [0, 1, 2] and G.edge_list() == [[0, 1], [0, 2], [1, 2]]
    G = Graph.from_vertices_and_edges(["a", "b", "c", "d"], [["a", "c"], ["a", "d"]])
    assert G.vertex_list() == ["a", "b", "c", "d"] and G.edge_list() == [
        ["a", "c"],
        ["a", "d"],
    ]
    with pytest.raises(ValueError):
        Graph.from_vertices_and_edges([1, 2, 3], [[1, 2], [2, 4]])


@pytest.mark.parametrize(
    "method, params",
    [
        ["is_rigid", []],
        ["is_min_rigid", []],
        ["is_redundantly_rigid", []],
        ["is_vertex_redundantly_rigid", []],
        ["is_k_vertex_redundantly_rigid", [2]],
        ["is_k_redundantly_rigid", [2]],
        ["is_globally_rigid", []],
        ["is_Rd_dependent", []],
        ["is_Rd_independent", []],
        ["is_Rd_circuit", []],
        ["is_Rd_closed", []],
        ["rigid_components", []],
        ["k_extension", [0, [1, 2], []]],
        ["zero_extension", [[1, 2], []]],
        ["one_extension", [[1, 2, 3], [1, 2]]],
    ],
)
def test_loop_error(method, params):
    with pytest.raises(LoopError):
        G = Graph([[1, 2], [1, 1], [2, 3], [1, 3]])
        func = getattr(G, method)
        func(*params)
    with pytest.raises(LoopError):
        G = Graph([[1, 1]])
        func = getattr(G, method)
        func(*params)


@pytest.mark.parametrize(
    "method, params",
    [
        ["all_k_extensions", [1]],
    ],
)
def test_iterator_loop_error(method, params):
    with pytest.raises(LoopError):
        G = Graph([[1, 2], [1, 1], [2, 3], [1, 3]])
        func = getattr(G, method)
        next(func(*params))
    with pytest.raises(LoopError):
        G = Graph([[1, 1]])
        func = getattr(G, method)
        next(func(*params))


@pytest.mark.parametrize(
    "method, params",
    [
        ["all_k_extensions", [1.1, 2]],
    ],
)
def test_iterator_parameter_type_error(method, params):
    with pytest.raises(TypeError):
        G = Graph([[1, 2], [1, 3], [2, 3]])
        func = getattr(G, method)
        next(func(*params))


@pytest.mark.parametrize(
    "method, params",
    [
        ["all_k_extensions", [-1, 2]],
        ["all_k_extensions", [-2, 1]],
    ],
)
def test_iterator_parameter_value_error(method, params):
    with pytest.raises(ValueError):
        G = Graph([[1, 2], [1, 3], [2, 3]])
        func = getattr(G, method)
        next(func(*params))


@pytest.mark.parametrize(
    "method, params",
    [
        ["extension_sequence", [1.1]],
        ["is_Rd_circuit", [2.1]],
        ["is_Rd_closed", [3.2]],
        ["is_Rd_dependent", [3 / 2]],
        ["is_Rd_independent", [1.2]],
        ["is_globally_rigid", [3.1]],
        ["is_k_redundantly_rigid", [2, 3.7]],
        ["is_k_vertex_redundantly_rigid", [2, 2.3]],
        ["is_min_k_redundantly_rigid", [2, 3.7]],
        ["is_min_k_vertex_redundantly_rigid", [2, 2.3]],
        ["is_min_redundantly_rigid", [2.6]],
        ["is_min_vertex_redundantly_rigid", [3.2]],
        ["is_min_rigid", [1.2]],
        ["is_rigid", [1.1]],
        ["is_redundantly_rigid", [math.log(2)]],
        ["is_vertex_redundantly_rigid", [4.8]],
        ["k_extension", [0, [1, 2], [], 4, 2.6]],
        ["one_extension", [[1, 2, 3], [1, 2], 4, 2.6]],
        ["random_framework", [1.1]],
        ["rigid_components", [3.7]],
        ["zero_extension", [[1, 2], 4, 2.6]],
    ],
)
def test_dimension_type_error(method, params):
    with pytest.raises(TypeError):
        G = Graph([[1, 2], [1, 3], [2, 3]])
        func = getattr(G, method)
        func(*params)


@pytest.mark.parametrize(
    "method, params",
    [
        ["extension_sequence", [0]],
        ["extension_sequence", [-2]],
        ["is_Rd_circuit", [0]],
        ["is_Rd_circuit", [-1]],
        ["is_Rd_closed", [0]],
        ["is_Rd_closed", [-2]],
        ["is_Rd_dependent", [0]],
        ["is_Rd_dependent", [-2]],
        ["is_Rd_independent", [0]],
        ["is_Rd_independent", [-1]],
        ["is_globally_rigid", [0]],
        ["is_globally_rigid", [-2]],
        ["is_k_redundantly_rigid", [2, 0]],
        ["is_k_redundantly_rigid", [2, -4]],
        ["is_k_vertex_redundantly_rigid", [2, 0]],
        ["is_k_vertex_redundantly_rigid", [2, -7]],
        ["is_min_k_redundantly_rigid", [2, 0]],
        ["is_min_k_redundantly_rigid", [2, -4]],
        ["is_min_k_vertex_redundantly_rigid", [2, 0]],
        ["is_min_k_vertex_redundantly_rigid", [2, -7]],
        ["is_min_redundantly_rigid", [0]],
        ["is_min_redundantly_rigid", [-2]],
        ["is_min_vertex_redundantly_rigid", [0]],
        ["is_min_vertex_redundantly_rigid", [-4]],
        ["is_min_rigid", [0]],
        ["is_min_rigid", [-3]],
        ["is_rigid", [0]],
        ["is_rigid", [-2]],
        ["is_redundantly_rigid", [0]],
        ["is_redundantly_rigid", [-2]],
        ["is_vertex_redundantly_rigid", [0]],
        ["is_vertex_redundantly_rigid", [-3]],
        ["k_extension", [0, [1, 2], [], 4, 0]],
        ["k_extension", [0, [1, 2], [], 4, -3]],
        ["one_extension", [[1, 2, 3], [1, 2], 4, 0]],
        ["one_extension", [[1, 2, 3], [1, 2], 4, -3]],
        ["random_framework", [0]],
        ["random_framework", [-2]],
        ["rigid_components", [0]],
        ["rigid_components", [-4]],
        ["zero_extension", [[1, 2], 4, 0]],
        ["zero_extension", [[1, 2], 4, -3]],
    ],
)
def test_dimension_value_error(method, params):
    with pytest.raises(ValueError):
        G = Graph([[1, 2], [1, 3], [2, 3]])
        func = getattr(G, method)
        func(*params)


@pytest.mark.parametrize(
    "method, params",
    [
        ["all_k_extensions", [1, 2.1]],
    ],
)
def test_iterator_dimension_type_error(method, params):
    with pytest.raises(TypeError):
        G = Graph([[1, 2], [1, 3], [2, 3]])
        func = getattr(G, method)
        next(func(*params))


@pytest.mark.parametrize(
    "method, params",
    [
        ["all_k_extensions", [1, 0]],
        ["all_k_extensions", [2, -1]],
    ],
)
def test_iterator_dimension_value_error(method, params):
    with pytest.raises(ValueError):
        G = Graph([[1, 2], [1, 3], [2, 3]])
        func = getattr(G, method)
        next(func(*params))


@pytest.mark.parametrize(
    "method, params",
    [
        ["is_k_redundantly_rigid", [2.4, 3]],
        ["is_k_vertex_redundantly_rigid", [3.7, 2]],
        ["is_min_k_redundantly_rigid", [2.5, 3]],
        ["is_min_k_vertex_redundantly_rigid", [2 / 3, 2]],
        ["k_extension", [0.3, [1, 2], [], 4, 2]],
    ],
)
def test_parameter_type_error(method, params):
    with pytest.raises(TypeError):
        G = Graph([[1, 2], [1, 3], [2, 3]])
        func = getattr(G, method)
        func(*params)


@pytest.mark.parametrize(
    "method, params",
    [
        ["is_k_redundantly_rigid", [-1, 3]],
        ["is_k_redundantly_rigid", [-2, 4]],
        ["is_k_vertex_redundantly_rigid", [-1, 2]],
        ["is_k_vertex_redundantly_rigid", [-3, 7]],
        ["is_min_k_redundantly_rigid", [-1, 3]],
        ["is_min_k_redundantly_rigid", [-2, 4]],
        ["is_min_k_vertex_redundantly_rigid", [-1, 2]],
        ["is_min_k_vertex_redundantly_rigid", [-3, 7]],
        ["k_extension", [-1, [1, 2], [], 4, 2]],
        ["k_extension", [-2, [1, 2], [], 4, 3]],
    ],
)
def test_parameter_value_error(method, params):
    with pytest.raises(ValueError):
        G = Graph([[1, 2], [1, 3], [2, 3]])
        func = getattr(G, method)
        func(*params)


def test_plot():
    G = graphs.DoubleBanana()
    G.plot(layout="random")
    plt.close("all")


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 2],
        [graphs.CompleteBipartite(3, 3), 16],
        [graphs.Diamond(), 4],
        [graphs.ThreePrism(), 24],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_count_reflection(graph, num_of_realizations):
    assert graph.number_of_realizations(count_reflection=True) == num_of_realizations


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 1],
        [graphs.CompleteBipartite(3, 3), 8],
        [graphs.Diamond(), 2],
        [graphs.ThreePrism(), 12],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations(graph, num_of_realizations):
    assert graph.number_of_realizations() == num_of_realizations


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 1],
        [graphs.CompleteBipartite(3, 3), 8],
        [graphs.Diamond(), 2],
        [graphs.ThreePrism(), 16],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_sphere(graph, num_of_realizations):
    assert graph.number_of_realizations(spherical=True) == num_of_realizations


@pytest.mark.parametrize(
    "graph, num_of_realizations",
    [
        [graphs.Complete(2), 1],
        [graphs.Complete(3), 2],
        [graphs.CompleteBipartite(3, 3), 16],
        [graphs.Diamond(), 4],
        [graphs.ThreePrism(), 32],
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_sphere_count_reflection(graph, num_of_realizations):
    assert (
        graph.number_of_realizations(spherical=True, count_reflection=True)
        == num_of_realizations
    )


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(4),
        graphs.K33plusEdge(),
        graphs.ThreePrismPlusEdge(),
        graphs.CompleteBipartite(1, 3),
        graphs.CompleteBipartite(2, 3),
        graphs.Path(3),
    ],
)
@pytest.mark.realization_counting
def test_number_of_realizations_error(graph):
    with pytest.raises(ValueError):
        graph.number_of_realizations()


def test_cone():
    G = graphs.Complete(5).cone()
    assert set(G.nodes) == set([0, 1, 2, 3, 4, 5]) and len(G.nodes) == 6
    G = graphs.Complete(4).cone(vertex="a")
    assert "a" in G.nodes
    G = graphs.Cycle(4).cone()
    assert G.number_of_nodes() == G.max_degree() + 1


@pytest.mark.long_local
def test_randomized_rigidity_properties():  # noqa: C901
    search_space = [range(1, 4), range(1, 7), range(10)]
    for dim, n, _ in product(*search_space):
        for m in range(1, math.comb(n, 2) + 1):
            G = Graph(nx.gnm_random_graph(n, m))
            assert G.number_of_nodes() == n
            assert G.number_of_edges() == m

            prop_rigid = G.is_rigid(dim)
            prop_min_rigid = G.is_min_rigid(dim)
            prop_glob_rigid = G.is_globally_rigid(dim)
            prop_red_rigid = G.is_redundantly_rigid(dim)
            prop_2_red_rigid = G.is_k_redundantly_rigid(2, dim)
            prop_3_red_rigid = G.is_k_redundantly_rigid(3, dim)
            prop_vred_rigid = G.is_vertex_redundantly_rigid(dim)
            prop_2_vred_rigid = G.is_k_vertex_redundantly_rigid(2, dim)
            prop_3_vred_rigid = G.is_k_vertex_redundantly_rigid(3, dim)
            prop_min_red_rigid = G.is_min_redundantly_rigid(dim)
            prop_min_2_red_rigid = G.is_min_k_redundantly_rigid(2, dim)
            prop_min_3_red_rigid = G.is_min_k_redundantly_rigid(3, dim)
            prop_min_vred_rigid = G.is_min_vertex_redundantly_rigid(dim)
            prop_min_2_vred_rigid = G.is_min_k_vertex_redundantly_rigid(2, dim)
            prop_min_3_vred_rigid = G.is_min_k_vertex_redundantly_rigid(3, dim)
            prop_sparse = G.is_kl_sparse(dim, math.comb(dim + 1, 2))
            prop_tight = G.is_kl_tight(dim, math.comb(dim + 1, 2))
            prop_seq = G.has_extension_sequence(dim)
            prop_dep = G.is_Rd_dependent(dim)
            prop_indep = G.is_Rd_independent(dim)
            prop_circ = G.is_Rd_circuit(dim)

            # randomized algorithm
            rprop_rigid = G.is_rigid(dim, algorithm="randomized")
            rprop_min_rigid = G.is_min_rigid(dim, algorithm="randomized")
            rprop_glob_rigid = G.is_globally_rigid(dim, algorithm="randomized")
            rprop_red_rigid = G.is_redundantly_rigid(dim, algorithm="randomized")
            rprop_2_red_rigid = G.is_k_redundantly_rigid(2, dim, algorithm="randomized")
            rprop_3_red_rigid = G.is_k_redundantly_rigid(3, dim, algorithm="randomized")
            rprop_vred_rigid = G.is_vertex_redundantly_rigid(
                dim, algorithm="randomized"
            )
            rprop_2_vred_rigid = G.is_k_vertex_redundantly_rigid(
                2, dim, algorithm="randomized"
            )
            rprop_3_vred_rigid = G.is_k_vertex_redundantly_rigid(
                3, dim, algorithm="randomized"
            )
            rprop_min_red_rigid = G.is_min_redundantly_rigid(
                dim, algorithm="randomized"
            )
            rprop_min_2_red_rigid = G.is_min_k_redundantly_rigid(
                2, dim, algorithm="randomized"
            )
            rprop_min_3_red_rigid = G.is_min_k_redundantly_rigid(
                3, dim, algorithm="randomized"
            )
            rprop_min_vred_rigid = G.is_min_vertex_redundantly_rigid(
                dim, algorithm="randomized"
            )
            rprop_min_2_vred_rigid = G.is_min_k_vertex_redundantly_rigid(
                2, dim, algorithm="randomized"
            )
            rprop_min_3_vred_rigid = G.is_min_k_vertex_redundantly_rigid(
                3, dim, algorithm="randomized"
            )
            rprop_dep = G.is_Rd_dependent(dim, algorithm="randomized")
            rprop_indep = G.is_Rd_independent(dim, algorithm="randomized")
            rprop_circ = G.is_Rd_circuit(dim, algorithm="randomized")

            # subgraph algorithm
            sprop_sparse = G.is_kl_sparse(
                dim, math.comb(dim + 1, 2), algorithm="subgraph"
            )
            sprop_tight = G.is_kl_tight(
                dim, math.comb(dim + 1, 2), algorithm="subgraph"
            )

            # cones
            res_cone = G.cone()
            cprop_rigid = res_cone.is_rigid(dim + 1)
            cprop_min_rigid = res_cone.is_min_rigid(dim + 1)
            cprop_glob_rigid = res_cone.is_globally_rigid(dim + 1)

            # extensions
            if n > dim:
                res_ext0 = G.all_k_extensions(0, dim)
            else:
                res_ext0 = []
            if m > 1 and n > dim + 1:
                res_ext1 = G.all_k_extensions(1, dim)
            else:
                res_ext1 = []

            # framework
            F = G.random_framework(dim)
            fprop_inf_rigid = F.is_inf_rigid()
            fprop_inf_flex = F.is_inf_flexible()
            fprop_min_inf_rigid = F.is_min_inf_rigid()
            fprop_red_rigid = F.is_redundantly_inf_rigid()
            fprop_dep = F.is_dependent()
            fprop_indep = F.is_independent()

            # (min) rigidity
            if prop_min_rigid:
                assert rprop_min_rigid
                assert cprop_min_rigid
                assert prop_rigid
                assert fprop_min_inf_rigid
                assert prop_indep
                if n > dim:
                    assert m == n * dim - math.comb(dim + 1, 2)
                    assert F.rigidity_matrix_rank() == n * dim - math.comb(dim + 1, 2)
                    assert G.min_degree() >= dim
                    assert G.min_degree() <= 2 * dim - 1
                    assert prop_sparse
                    assert prop_tight
                    assert prop_seq
                else:
                    assert m == math.comb(n, 2)
                for graph in res_ext0:
                    assert graph.is_min_rigid(dim)
                for graph in res_ext1:
                    assert graph.is_min_rigid(dim)
            if rprop_min_rigid:
                assert prop_min_rigid
            if prop_rigid:
                assert rprop_rigid
                assert cprop_rigid
                assert fprop_inf_rigid
                if n > dim:
                    assert m >= n * dim - math.comb(dim + 1, 2)
                    assert F.rigidity_matrix_rank() == n * dim - math.comb(dim + 1, 2)
                    assert G.min_degree() >= dim
                    if m > n * dim - math.comb(dim + 1, 2):
                        assert prop_dep
                    else:
                        assert prop_indep
                else:
                    assert m == math.comb(n, 2)
                    assert prop_indep
                if prop_circ:
                    assert m == n * dim - math.comb(dim + 1, 2) + 1
            if rprop_rigid:
                assert prop_rigid

            # sparsity
            if prop_sparse:
                assert sprop_sparse
            if sprop_sparse:
                assert prop_sparse
            if prop_tight:
                assert sprop_tight
                if dim == 2 or dim == 1:
                    assert prop_min_rigid
            if sprop_tight:
                assert prop_tight

            # redundancy
            if prop_red_rigid:
                assert rprop_red_rigid
                assert prop_rigid
                assert fprop_red_rigid
                assert m >= n * dim - math.comb(dim + 1, 2) + 1
                if G.number_of_nodes() >= dim + 1 + 1:
                    assert G.min_degree() >= dim + 1  # thm-vertex-red-min-deg
            if rprop_red_rigid:
                assert prop_red_rigid
            if prop_2_red_rigid:
                assert rprop_2_red_rigid
                assert prop_rigid
                assert prop_red_rigid
                assert m >= n * dim - math.comb(dim + 1, 2) + 2
                if G.number_of_nodes() >= dim + 2 + 1:
                    assert G.min_degree() >= dim + 2  # thm-vertex-red-min-deg
            if rprop_2_red_rigid:
                assert prop_2_red_rigid
            if prop_3_red_rigid:
                assert rprop_3_red_rigid
                assert prop_rigid
                assert prop_2_red_rigid
                assert prop_red_rigid
                assert m >= n * dim - math.comb(dim + 1, 2) + 2
                if G.number_of_nodes() >= dim + 3 + 1:
                    assert G.min_degree() >= dim + 3  # thm-vertex-red-min-deg
            if rprop_3_red_rigid:
                assert prop_3_red_rigid
            if prop_vred_rigid:
                assert rprop_vred_rigid
                assert prop_rigid
                if G.number_of_nodes() >= dim + 1 + 1:
                    assert prop_red_rigid  # thm-vertex-implies_edge
                    assert G.min_degree() >= dim + 1  # thm-vertex-red-min-deg
            if rprop_vred_rigid:
                assert prop_vred_rigid
            if prop_2_vred_rigid:
                assert rprop_2_vred_rigid
                assert prop_rigid
                assert prop_vred_rigid
                if G.number_of_nodes() >= dim + 2 + 1:
                    assert prop_2_red_rigid  # thm-vertex-implies_edge
                    assert G.min_degree() >= dim + 2  # thm-vertex-red-min-deg
            if rprop_2_vred_rigid:
                assert prop_2_vred_rigid
            if prop_3_vred_rigid:
                assert rprop_3_vred_rigid
                assert prop_rigid
                assert prop_2_vred_rigid
                assert prop_vred_rigid
                if G.number_of_nodes() >= dim + 3 + 1:
                    assert prop_3_red_rigid  # thm-vertex-implies_edge
                    assert G.min_degree() >= dim + 3  # thm-vertex-red-min-deg
            if rprop_3_vred_rigid:
                assert prop_3_vred_rigid

            # minimal redundancy
            if prop_min_red_rigid:
                assert rprop_min_red_rigid
                assert prop_rigid
                assert prop_red_rigid
                assert m >= n * dim - math.comb(dim + 1, 2) + 1
                if G.number_of_nodes() >= dim + 1 + 1:
                    assert G.min_degree() >= dim + 1  # thm-vertex-red-min-deg
            if rprop_min_red_rigid:
                assert prop_min_red_rigid
            if prop_min_2_red_rigid:
                assert rprop_min_2_red_rigid
                assert prop_rigid
                assert prop_red_rigid
                assert prop_2_red_rigid
                assert m >= n * dim - math.comb(dim + 1, 2) + 2
                if G.number_of_nodes() >= dim + 2 + 1:
                    assert G.min_degree() >= dim + 2  # thm-vertex-red-min-deg
            if rprop_min_2_red_rigid:
                assert prop_min_2_red_rigid
            if prop_min_3_red_rigid:
                assert rprop_min_3_red_rigid
                assert prop_rigid
                assert prop_2_red_rigid
                assert prop_red_rigid
                assert prop_3_red_rigid
                assert m >= n * dim - math.comb(dim + 1, 2) + 3
                if G.number_of_nodes() >= dim + 3 + 1:
                    assert G.min_degree() >= dim + 3  # thm-vertex-red-min-deg
            if rprop_min_3_red_rigid:
                assert prop_min_3_red_rigid
            if prop_min_vred_rigid:
                assert rprop_min_vred_rigid
                assert prop_rigid
                assert prop_vred_rigid
                if G.number_of_nodes() >= dim + 1 + 1:
                    assert prop_red_rigid  # thm-vertex-implies_edge
                    assert G.min_degree() >= dim + 1  # thm-vertex-red-min-deg
            if rprop_min_vred_rigid:
                assert prop_min_vred_rigid
            if prop_min_2_vred_rigid:
                assert rprop_min_2_vred_rigid
                assert prop_rigid
                assert prop_vred_rigid
                assert prop_2_vred_rigid
                if G.number_of_nodes() >= dim + 2 + 1:
                    assert prop_2_red_rigid  # thm-vertex-implies_edge
                    assert G.min_degree() >= dim + 2  # thm-vertex-red-min-deg
            if rprop_min_2_vred_rigid:
                assert prop_min_2_vred_rigid
            if prop_min_3_vred_rigid:
                assert rprop_min_3_vred_rigid
                assert prop_rigid
                assert prop_2_vred_rigid
                assert prop_vred_rigid
                assert prop_3_vred_rigid
                if G.number_of_nodes() >= dim + 3 + 1:
                    assert prop_3_red_rigid  # thm-vertex-implies_edge
                    assert G.min_degree() >= dim + 3  # thm-vertex-red-min-deg
            if rprop_min_3_vred_rigid:
                assert prop_min_3_vred_rigid

            # global rigidity
            if prop_glob_rigid:
                assert rprop_glob_rigid
                assert prop_rigid
                assert cprop_glob_rigid
                if n > dim + 1:
                    assert m >= n * dim - math.comb(dim + 1, 2)
                    assert prop_red_rigid
                    assert G.vertex_connectivity() >= dim + 1
                else:
                    assert m == math.comb(n, 2)
                if prop_min_rigid:
                    assert m == math.comb(n, 2)
            if rprop_glob_rigid:
                assert prop_glob_rigid

            # cones
            if cprop_min_rigid:
                assert prop_min_rigid
            if cprop_rigid:
                assert prop_rigid
            if cprop_glob_rigid:
                assert prop_glob_rigid

            if not prop_rigid:
                assert not prop_min_rigid
                assert not prop_glob_rigid
                assert not prop_red_rigid
                assert not prop_2_red_rigid
                assert not prop_3_red_rigid
                assert not prop_vred_rigid
                assert not prop_2_vred_rigid
                assert not prop_3_vred_rigid
                assert not prop_min_red_rigid
                assert not prop_min_2_red_rigid
                assert not prop_min_3_red_rigid
                assert not prop_min_vred_rigid
                assert not prop_min_2_vred_rigid
                assert not prop_min_3_vred_rigid

            # dependence
            if prop_circ:
                assert rprop_circ
                assert prop_dep
                assert not prop_indep
            if rprop_circ:
                assert prop_circ
            if prop_indep:
                assert rprop_indep
                assert not prop_circ
                assert not prop_dep
                assert fprop_indep
                if n > dim:
                    assert m <= n * dim - math.comb(dim + 1, 2)
            if rprop_indep:
                assert prop_indep
            if prop_dep:
                assert rprop_dep
                assert fprop_dep
            if rprop_dep:
                assert prop_dep

            # closure
            res_close = Graph(G.Rd_closure())
            assert res_close.is_Rd_closed()
            res_close = Graph(G.Rd_closure(), algorithm="randomized")
            assert res_close.is_Rd_closed()

            # frameworks
            if fprop_inf_rigid:
                assert prop_rigid
                assert not fprop_inf_flex
            if fprop_min_inf_rigid:
                assert prop_min_rigid
            if fprop_red_rigid:
                assert prop_red_rigid
            if fprop_indep:
                assert prop_indep
            if fprop_dep:
                assert prop_dep
            if fprop_inf_flex:
                assert not fprop_inf_rigid
