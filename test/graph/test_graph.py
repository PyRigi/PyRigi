import math
from itertools import product
from random import randint

import matplotlib.pyplot as plt
import networkx as nx
import pytest

import pyrigi.graph._constructions.constructions as constructions
import pyrigi.graph._constructions.extensions as extensions
import pyrigi.graph._general as general
import pyrigi.graph._rigidity.generic as generic
import pyrigi.graph._rigidity.global_ as g_global
import pyrigi.graph._rigidity.matroidal as g_matroidal
import pyrigi.graph._rigidity.redundant as g_redundant
import pyrigi.graph._sparsity.sparsity as sparsity
import pyrigi.framework._rigidity.infinitesimal as infinitesimal
import pyrigi.framework._rigidity.matroidal as fw_matroidal
import pyrigi.framework._rigidity.redundant as fw_redundant
import pyrigi.graphDB as graphs
from pyrigi.exception import LoopError
from pyrigi.framework import Framework
from pyrigi.graph import Graph

is_rigid_algorithms_all_d = ["default", "randomized", "numerical"]
is_rigid_algorithms_d1 = is_rigid_algorithms_all_d + ["graphic"]
is_rigid_algorithms_d2 = is_rigid_algorithms_all_d + ["sparsity"]


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


@pytest.mark.long_local
def test_randomized_rigidity_properties():  # noqa: C901
    search_space = [range(1, 4), range(1, 7), range(10)]
    for dim, n, _ in product(*search_space):
        for m in range(1, math.comb(n, 2) + 1):
            G = Graph(nx.gnm_random_graph(n, m))
            assert G.number_of_nodes() == n
            assert G.number_of_edges() == m

            prop_rigid = generic.is_rigid(G, dim)
            prop_min_rigid = generic.is_min_rigid(G, dim)
            prop_glob_rigid = g_global.is_globally_rigid(G, dim)
            prop_red_rigid = g_redundant.is_redundantly_rigid(G, dim)
            prop_2_red_rigid = g_redundant.is_k_redundantly_rigid(G, 2, dim)
            prop_3_red_rigid = g_redundant.is_k_redundantly_rigid(G, 3, dim)
            prop_vred_rigid = g_redundant.is_vertex_redundantly_rigid(G, dim)
            prop_2_vred_rigid = g_redundant.is_k_vertex_redundantly_rigid(G, 2, dim)
            prop_3_vred_rigid = g_redundant.is_k_vertex_redundantly_rigid(G, 3, dim)
            prop_min_red_rigid = g_redundant.is_min_redundantly_rigid(G, dim)
            prop_min_2_red_rigid = g_redundant.is_min_k_redundantly_rigid(G, 2, dim)
            prop_min_3_red_rigid = g_redundant.is_min_k_redundantly_rigid(G, 3, dim)
            prop_min_vred_rigid = g_redundant.is_min_vertex_redundantly_rigid(G, dim)
            prop_min_2_vred_rigid = g_redundant.is_min_k_vertex_redundantly_rigid(
                G, 2, dim
            )
            prop_min_3_vred_rigid = g_redundant.is_min_k_vertex_redundantly_rigid(
                G, 3, dim
            )
            prop_sparse = sparsity.is_kl_sparse(G, dim, math.comb(dim + 1, 2))
            prop_tight = sparsity.is_kl_tight(G, dim, math.comb(dim + 1, 2))
            prop_seq = extensions.has_extension_sequence(G, dim)
            prop_dep = g_matroidal.is_Rd_dependent(G, dim)
            prop_indep = g_matroidal.is_Rd_independent(G, dim)
            prop_circ = g_matroidal.is_Rd_circuit(G, dim)

            # randomized algorithm
            # fmt: off
            rprop_rigid = generic.is_rigid(G, dim, algorithm="randomized")
            rprop_min_rigid = generic.is_min_rigid(G, dim, algorithm="randomized")
            rprop_glob_rigid = g_global.is_globally_rigid(G, dim, algorithm="randomized")
            rprop_red_rigid = g_redundant.is_redundantly_rigid(G, dim, algorithm="randomized")  # noqa: E501
            rprop_2_red_rigid = g_redundant.is_k_redundantly_rigid(G, 2, dim, algorithm="randomized")  # noqa: E501
            rprop_3_red_rigid = g_redundant.is_k_redundantly_rigid(G, 3, dim, algorithm="randomized")  # noqa: E501
            rprop_vred_rigid = g_redundant.is_vertex_redundantly_rigid(
                G, dim, algorithm="randomized"
            )
            rprop_2_vred_rigid = g_redundant.is_k_vertex_redundantly_rigid(
                G, 2, dim, algorithm="randomized"
            )
            rprop_3_vred_rigid = g_redundant.is_k_vertex_redundantly_rigid(
                G, 3, dim, algorithm="randomized"
            )
            rprop_min_red_rigid = g_redundant.is_min_redundantly_rigid(
                G, dim, algorithm="randomized"
            )
            rprop_min_2_red_rigid = g_redundant.is_min_k_redundantly_rigid(
                G, 2, dim, algorithm="randomized"
            )
            rprop_min_3_red_rigid = g_redundant.is_min_k_redundantly_rigid(
                G, 3, dim, algorithm="randomized"
            )
            rprop_min_vred_rigid = g_redundant.is_min_vertex_redundantly_rigid(
                G, dim, algorithm="randomized"
            )
            rprop_min_2_vred_rigid = g_redundant.is_min_k_vertex_redundantly_rigid(
                G, 2, dim, algorithm="randomized"
            )
            rprop_min_3_vred_rigid = g_redundant.is_min_k_vertex_redundantly_rigid(
                G, 3, dim, algorithm="randomized"
            )
            rprop_dep = g_matroidal.is_Rd_dependent(G, dim, algorithm="randomized")
            rprop_indep = g_matroidal.is_Rd_independent(G, dim, algorithm="randomized")
            rprop_circ = g_matroidal.is_Rd_circuit(G, dim, algorithm="randomized")
            # fmt: on
            # black formatting is skipped to enable nice diff for code review
            # It can be removed in the future, together with  # noqa: E501.

            # subgraph algorithm
            sprop_sparse = sparsity.is_kl_sparse(
                G, dim, math.comb(dim + 1, 2), algorithm="subgraph"
            )
            sprop_tight = sparsity.is_kl_tight(
                G, dim, math.comb(dim + 1, 2), algorithm="subgraph"
            )

            # cones
            res_cone = constructions.cone(G)
            cprop_rigid = generic.is_rigid(res_cone, dim + 1)
            cprop_min_rigid = generic.is_min_rigid(res_cone, dim + 1)
            cprop_glob_rigid = g_global.is_globally_rigid(res_cone, dim + 1)

            # extensions
            if n > dim:
                res_ext0 = list(extensions.all_k_extensions(G, 0, dim))
            else:
                res_ext0 = []
            if m > 1 and n > dim + 1:
                res_ext1 = list(extensions.all_k_extensions(G, 1, dim))
            else:
                res_ext1 = []

            # framework
            F = Framework.Random(G, dim)
            fprop_inf_rigid = infinitesimal.is_inf_rigid(F)
            fprop_inf_flex = infinitesimal.is_inf_flexible(F)
            fprop_min_inf_rigid = infinitesimal.is_min_inf_rigid(F)
            fprop_red_rigid = fw_redundant.is_redundantly_inf_rigid(F)
            fprop_dep = fw_matroidal.is_dependent(F)
            fprop_indep = fw_matroidal.is_independent(F)

            # (min) rigidity
            if prop_min_rigid:
                assert rprop_min_rigid
                assert cprop_min_rigid
                assert prop_rigid
                assert fprop_min_inf_rigid
                assert prop_indep
                if n > dim:
                    assert m == n * dim - math.comb(dim + 1, 2)
                    assert infinitesimal.rigidity_matrix_rank(F) == n * dim - math.comb(
                        dim + 1, 2
                    )
                    assert general.min_degree(G) >= dim
                    assert general.min_degree(G) <= 2 * dim - 1
                    assert prop_sparse
                    assert prop_tight
                    assert prop_seq
                else:
                    assert m == math.comb(n, 2)
                for graph in res_ext0:
                    assert generic.is_min_rigid(graph, dim)
                for graph in res_ext1:
                    assert generic.is_min_rigid(graph, dim)
            if rprop_min_rigid:
                assert prop_min_rigid
            if prop_rigid:
                assert rprop_rigid
                assert cprop_rigid
                assert fprop_inf_rigid
                if n > dim:
                    assert m >= n * dim - math.comb(dim + 1, 2)
                    assert infinitesimal.rigidity_matrix_rank(F) == n * dim - math.comb(
                        dim + 1, 2
                    )
                    assert general.min_degree(G) >= dim
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
                    assert general.min_degree(G) >= dim + 1  # thm-vertex-red-min-deg
            if rprop_red_rigid:
                assert prop_red_rigid
            if prop_2_red_rigid:
                assert rprop_2_red_rigid
                assert prop_rigid
                assert prop_red_rigid
                assert m >= n * dim - math.comb(dim + 1, 2) + 2
                if G.number_of_nodes() >= dim + 2 + 1:
                    assert general.min_degree(G) >= dim + 2  # thm-vertex-red-min-deg
            if rprop_2_red_rigid:
                assert prop_2_red_rigid
            if prop_3_red_rigid:
                assert rprop_3_red_rigid
                assert prop_rigid
                assert prop_2_red_rigid
                assert prop_red_rigid
                assert m >= n * dim - math.comb(dim + 1, 2) + 2
                if G.number_of_nodes() >= dim + 3 + 1:
                    assert general.min_degree(G) >= dim + 3  # thm-vertex-red-min-deg
            if rprop_3_red_rigid:
                assert prop_3_red_rigid
            if prop_vred_rigid:
                assert rprop_vred_rigid
                assert prop_rigid
                if G.number_of_nodes() >= dim + 1 + 1:
                    assert prop_red_rigid  # thm-vertex-implies_edge
                    assert general.min_degree(G) >= dim + 1  # thm-vertex-red-min-deg
            if rprop_vred_rigid:
                assert prop_vred_rigid
            if prop_2_vred_rigid:
                assert rprop_2_vred_rigid
                assert prop_rigid
                assert prop_vred_rigid
                if G.number_of_nodes() >= dim + 2 + 1:
                    assert prop_2_red_rigid  # thm-vertex-implies_edge
                    assert general.min_degree(G) >= dim + 2  # thm-vertex-red-min-deg
            if rprop_2_vred_rigid:
                assert prop_2_vred_rigid
            if prop_3_vred_rigid:
                assert rprop_3_vred_rigid
                assert prop_rigid
                assert prop_2_vred_rigid
                assert prop_vred_rigid
                if G.number_of_nodes() >= dim + 3 + 1:
                    assert prop_3_red_rigid  # thm-vertex-implies_edge
                    assert general.min_degree(G) >= dim + 3  # thm-vertex-red-min-deg
            if rprop_3_vred_rigid:
                assert prop_3_vred_rigid

            # minimal redundancy
            if prop_min_red_rigid:
                assert rprop_min_red_rigid
                assert prop_rigid
                assert prop_red_rigid
                assert m >= n * dim - math.comb(dim + 1, 2) + 1
                if G.number_of_nodes() >= dim + 1 + 1:
                    assert general.min_degree(G) >= dim + 1  # thm-vertex-red-min-deg
            if rprop_min_red_rigid:
                assert prop_min_red_rigid
            if prop_min_2_red_rigid:
                assert rprop_min_2_red_rigid
                assert prop_rigid
                assert prop_red_rigid
                assert prop_2_red_rigid
                assert m >= n * dim - math.comb(dim + 1, 2) + 2
                if G.number_of_nodes() >= dim + 2 + 1:
                    assert general.min_degree(G) >= dim + 2  # thm-vertex-red-min-deg
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
                    assert general.min_degree(G) >= dim + 3  # thm-vertex-red-min-deg
            if rprop_min_3_red_rigid:
                assert prop_min_3_red_rigid
            if prop_min_vred_rigid:
                assert rprop_min_vred_rigid
                assert prop_rigid
                assert prop_vred_rigid
                if G.number_of_nodes() >= dim + 1 + 1:
                    assert prop_red_rigid  # thm-vertex-implies_edge
                    assert general.min_degree(G) >= dim + 1  # thm-vertex-red-min-deg
            if rprop_min_vred_rigid:
                assert prop_min_vred_rigid
            if prop_min_2_vred_rigid:
                assert rprop_min_2_vred_rigid
                assert prop_rigid
                assert prop_vred_rigid
                assert prop_2_vred_rigid
                if G.number_of_nodes() >= dim + 2 + 1:
                    assert prop_2_red_rigid  # thm-vertex-implies_edge
                    assert general.min_degree(G) >= dim + 2  # thm-vertex-red-min-deg
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
                    assert general.min_degree(G) >= dim + 3  # thm-vertex-red-min-deg
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
                    assert nx.node_connectivity(G) >= dim + 1
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
            res_close = Graph(g_matroidal.Rd_closure(G))
            assert g_matroidal.is_Rd_closed(res_close)
            res_close = Graph(g_matroidal.Rd_closure(G, algorithm="randomized"))
            assert g_matroidal.is_Rd_closed(res_close)

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
