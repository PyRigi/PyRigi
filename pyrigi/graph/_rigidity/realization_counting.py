"""
This module provides algorithms related to rigid realization counting.
"""

import importlib.util
import math
from copy import copy, deepcopy

import more_itertools
import networkx as nx

import pyrigi.graph._rigidity.generic as generic_rigidity
import pyrigi.graph._rigidity.global_ as global_rigidity
from pyrigi._utils import _input_check
from pyrigi.graph._export import export
from pyrigi.graph._sparsity import sparsity


def number_of_realizations(  # noqa: C901
    graph: nx.Graph,
    dim: int = 2,
    algorithm: str = "default",
    spherical: bool = False,
    count_reflection: bool = False,
) -> int | float:
    """
    Count the number of complex realizations of a ``dim``-rigid graph.

    Realizations in ``dim``-dimensional sphere
    can be counted using ``spherical=True``.

    For minimally rigid graphs algorithms of :cite:p:`CapcoGalletEtAl2018` and
    :cite:p:`GalletGraseggerSchicho2020` are used.
    Note, however, that here the result from these algorithms
    is by default divided by two.
    This behaviour accounts better for global rigidity,
    but it can be changed using the parameter ``count_reflection``.

    For 2-rigid graphs which are not minimal,
    the algorithm of :cite:p:`DewarGraseggerEtAl2025` is used.

    Caution: PyRigi can compute realizations counts directly but this might be slow.
    Faster computation works only if the python package ``lnumber``
    is installed :cite:p:`Capco2024`.
    See :ref:`installation-guide` for details on installing.

    Definitions
    -----------
    * :prf:ref:`Number of complex realizations<def-number-of-realizations>`
    * :prf:ref:`Number of complex spherical realizations<def-number-of-spherical-realizations>`

    Parameters
    ----------
    dim:
        The dimension in which the realizations are counted.
        Currently, only ``dim=1`` and ``dim=2`` are supported.
    algorithm:
        If "default", PyRigi checks which algorithm is available for the parameters and choses this one.
        If "pyrigi", a pure PyRigi implementation is used.
        If "lnumber", the ``lnumber`` package is used.
        This needs to be installed separately
        but is much faster than the "pyrigi" implementation.
    spherical:
        If ``True``, the number of spherical realizations of the graph is returned.
        If ``False`` (default), the number of planar realizations is returned.
    count_reflection:
        If ``True``, the number of realizations is computed only modulo direct isometries.
        (so reflected realizations are counted to be non-congruent as in
        :cite:p:`CapcoGalletEtAl2018` and
        :cite:p:`GalletGraseggerSchicho2020`).
        If ``False`` (default), reflection is not counted.

    Examples
    --------
    >>> from pyrigi import Graph
    >>> import pyrigi.graphDB as graphs
    >>> G = Graph([(0,1),(1,2),(2,0)])
    >>> G.number_of_realizations() # number of planar realizations
    1
    >>> G.number_of_realizations(spherical=True)
    1
    >>> G = graphs.ThreePrism()
    >>> G.number_of_realizations() # number of planar realizations
    12
    """  # noqa: E501

    algorithm_in = algorithm

    if algorithm == "default":
        if dim == 1:
            algorithm = "pyrigi"
        elif dim == 2:
            if graph.number_of_edges() > 2 * graph.number_of_nodes() - 3:
                algorithm = "pyrigi"
            elif importlib.util.find_spec("lnumber") is not None:
                algorithm = "lnumber"
            else:
                algorithm = "pyrigi"
        else:
            algorithm = "checktrivial"
    if algorithm == "pyrigi":
        if (
            graph.number_of_edges() >= 2 * graph.number_of_nodes() - 3
            or graph.number_of_edges() == math.comb(graph.number_of_nodes(), 2)
        ):
            _input_check.dimension_for_algorithm(
                dim, [1, 2], "number_of_realizations with algorithm pyrigi"
            )
        else:
            _input_check.dimension_for_algorithm(dim, [1], "number_of_realizations")
    elif algorithm == "lnumber":
        _input_check.dimension_for_algorithm(dim, [2], "number_of_realizations")

    if graph.number_of_nodes() == 1:
        return 1

    if not generic_rigidity.is_rigid(graph, dim):
        return math.inf

    fac = 1 if count_reflection else 2

    # Check trivial cases for higher dimensions
    if algorithm == "checktrivial":
        if global_rigidity.is_globally_rigid(graph, dim):
            return 2 // fac
        raise NotImplementedError(
            "There is no combinatorial algorithm for 'dim'>2 available,"
            + "except for trivial cases."
        )

    if dim == 1:
        if global_rigidity.is_globally_rigid(graph, dim):
            return 2 // fac
        if generic_rigidity.is_min_rigid(graph, dim):
            G = deepcopy(graph)
            deg_1 = 0
            while G.number_of_nodes() > 2 and G.min_degree() == 1:
                min_v = [v for v in G.nodes if G.degree(v) == 1]
                G.delete_vertices(min_v)
                deg_1 += len(min_v)
            if G.number_of_nodes() == 1:
                return (2**deg_1) // fac
            return (2 // fac) * (2**deg_1)
        # not 2-connected
        G = deepcopy(graph)
        cut = next(iter(nx.all_node_cuts(G)))
        G.delete_vertices(cut)
        con = nx.connected_components(G)
        sub = [graph.subgraph(c.union(cut)).copy() for c in con]
        return (fac ** (len(sub) - 1)) * math.prod(
            [
                number_of_realizations(
                    g, dim, algorithm_in, spherical, count_reflection
                )
                for g in sub
            ]
        )

    # dim == 2 from now on
    if graph.number_of_nodes() == 2 and graph.number_of_edges() == 1:
        return 1
    if graph.number_of_nodes() == 3 and graph.number_of_edges() == 3:
        return 2 // fac

    if graph.number_of_edges() == 2 * graph.number_of_nodes() - 3:
        if algorithm == "lnumber":
            try:
                import lnumber
            except ImportError:
                raise ImportError(
                    "For counting the number of realizations with 'lnumber', "
                    "the optional package 'lnumber' is used, "
                    "run `pip install pyrigi[realization-counting]`!"
                )

            graph_int = export.to_int(graph)

            if spherical:
                return lnumber.lnumbers(graph_int) // fac
            return lnumber.lnumber(graph_int) // fac
        if algorithm == "pyrigi":
            if spherical:
                return _number_of_sphere_realizations_min_rigid_dim_2(graph) // fac
            return _number_of_plane_realizations_min_rigid_dim_2(graph) // fac
    else:  # not minimally rigid (but rigid)
        if algorithm == "lnumber":
            raise ValueError(
                "The algorithm `lnumber` is only available for minimally rigid graphs "
                + "but the input graph is not minimally rigid."
            )
        if global_rigidity.is_globally_rigid(graph, dim):
            return 2 // fac
        return _number_of_realizations_rigid_not_globally_rigid_dim_2(
            graph, algorithm_in, spherical, count_reflection
        )


def _number_of_realizations_rigid_not_globally_rigid_dim_2(
    graph: nx.Graph,
    algorithm: str = "default",
    spherical: bool = False,
    count_reflection: bool = False,
) -> int:
    """
    Compute the number of realizations in dimension 2
    for a graph that is rigid but
    neither minimally nor globally rigid.

    Parameters
    ----------
    graph:
        A rigid graph
    algorithm:
        See number_of_realizations()
    spherical:
        See number_of_realizations()
    count_reflection:
        See number_of_realizations()
    """
    fac = 1 if count_reflection else 2
    G = deepcopy(graph)
    cut = next(iter(nx.all_node_cuts(G)))
    # Case where the graph is not 3-connected
    if len(cut) == 2:
        G.delete_vertices(cut)
        con = nx.connected_components(G)
        sub = [graph.subgraph(c.union(cut)).copy() for c in con]
        # Case where the vertices of the cut are adjacent
        if graph.has_edge(*cut):
            return fac ** (len(sub) - 1) * math.prod(
                [
                    number_of_realizations(g, 2, algorithm, spherical, count_reflection)
                    for g in sub
                ]
            )
        rig = [generic_rigidity.is_rigid(g, 2) for g in sub]
        count_rig = rig.count(True)
        if count_rig > 1:
            [g.add_edge(*cut) for g in sub]
            return fac ** (len(sub) - 1) * math.prod(
                [
                    number_of_realizations(g, 2, algorithm, spherical, count_reflection)
                    for g in sub
                ]
            )
        pos = rig.index(True)
        res = fac ** (len(sub) - 1) * number_of_realizations(
            sub[pos], 2, algorithm, spherical, count_reflection
        )
        [g.add_edge(*cut) for g in (sub[0:pos] + sub[pos + 1 :])]
        return res * math.prod(
            [
                number_of_realizations(g, 2, algorithm, spherical, count_reflection)
                for g in (sub[0:pos] + sub[pos + 1 :])
            ]
        )
    # Case where the graph is 3-connected but not redundantly rigid
    # Find edge e sucht that graph - e is not rigid
    G = deepcopy(graph)
    edges = G.edge_list()
    found = False
    while len(edges) > 0 and not found:
        e = edges.pop()
        G.remove_edge(*e)
        if generic_rigidity.is_rigid(G, 2):
            G.add_edge(*e)
        else:
            found = True
    # Get maximal rigid subgraphs
    comp = G.rigid_components(2)
    max_sub = [G.subgraph(c).copy() for c in comp]
    # Get minimally rigid spanning subgraphs
    span = [sparsity.spanning_kl_sparse_subgraph(g, 2, 3) for g in max_sub]
    # Compute result
    prod_g = math.prod(
        [
            number_of_realizations(g, 2, algorithm, spherical, count_reflection)
            for g in max_sub
        ]
    )
    prod_h = math.prod(
        [
            number_of_realizations(h, 2, algorithm, spherical, count_reflection)
            for h in span
        ]
    )
    H = nx.compose(span[0], span[1])
    for h in span[2:]:
        H = nx.compose(H, h)
    H.add_edge(*e)
    return (
        number_of_realizations(H, 2, algorithm, spherical, count_reflection)
        * prod_g
        // prod_h
    )


def _number_of_sphere_realizations_min_rigid_dim_2(graph: nx.Graph) -> int:
    """
    Compute the number of spherical realizations in dimension 2
    combinatorially within PyRigi.

    Parameters
    ----------
    graph:
        A minimally rigid graph

    Notes
    -----
    The algorithm from :cite:p:`GalletGraseggerSchicho2020` is used.
    """
    G = deepcopy(graph)
    deg_2 = 0
    while G.number_of_nodes() > 2 and G.min_degree() == 2:
        min_v = [v for v in G.nodes if G.degree(v) == 2]
        G.delete_vertices(min_v)
        deg_2 += len(min_v)

    if G.number_of_nodes() == 0:
        return 2 ** (deg_2 - 2)
    if G.number_of_nodes() == 2:
        return 2**deg_2
    return (2**deg_2) * _number_of_sphere_realizations_min_rigid_dim_2_rec(
        _graph_to_quadrograph(G)
    )


def _number_of_sphere_realizations_min_rigid_dim_2_rec(quadrograph: list) -> int:
    """
    Compute the number of spherical realizations of a quadrograph recursively.

    Parameters
    ----------
    quadrograph:
        A pair [N,Q] representing vertices and edges of a quadrograph.
        This pair comes initially from `_graph_to_quadrograph`
    """
    quad_N = quadrograph[0]
    quad_Q = quadrograph[1]
    if len(quad_N) in [3, 4]:
        return 1
    if len(quad_N) == 2:
        return 0
    q0 = quad_Q[0]
    a, b, c, d = tuple(q0)
    Qprime = quad_Q[1:]
    tempN = list(set(quad_N).difference(q0))
    allLists = more_itertools.powerset(tempN)
    tot_sum = 0
    counter = max(quad_N) + 1
    for subset in allLists:
        set_I = [a, b, *list(subset)]
        Q22 = [q for q in Qprime if len(set(q).intersection(set_I)) == 2]
        if len(Q22) > 0:
            continue
        Q40 = [q for q in Qprime if len(set(q).intersection(set_I)) == 4]
        Q31 = [q for q in Qprime if len(set(q).intersection(set_I)) == 3]

        if len(set_I) - len(Q40 + Q31) == 2:
            set_J = [j for j in quad_N if j not in set_I]
            Q13new = [
                [x if (x in set_J) else counter for x in q]
                for q in Qprime
                if len(set(q).intersection(set_I)) == 1
            ]
            Q04 = [q for q in Qprime if len(set(q).intersection(set_I)) == 0]
            Q31new = [[x if (x in set_I) else counter for x in q] for q in Q31]
            tot_sum += _number_of_sphere_realizations_min_rigid_dim_2_rec(
                [[*set_I, counter], Q40 + Q31new]
            ) * _number_of_sphere_realizations_min_rigid_dim_2_rec(
                [[*set_J, counter], Q13new + Q04]
            )
        else:
            continue
    return tot_sum


def _graph_to_quadrograph(graph: nx.Graph) -> list:
    """
    Generate a quadrograph from a graph.
    A quadrograph here is a pair (N,Q),
    where N represents the set of vertices
    and Q represents the set of biedges.

    Parameters
    ----------
    graph:
        A minimally rigid graph
    """
    n = graph.number_of_nodes()
    if n < 2:
        raise ValueError("Graph is to small")
    quad_N = range(1, 2 * n + 1)
    quad_Q = []
    edges = graph.edge_list()
    vertices = graph.vertex_list()
    mapping = {vertices[i]: i + 1 for i in range(n)}
    for edge in edges:
        quad_Q.append(
            [
                mapping[edge[0]],
                mapping[edge[1]],
                mapping[edge[0]] + n,
                mapping[edge[1]] + n,
            ]
        )
    return [quad_N, quad_Q]


def _number_of_plane_realizations_min_rigid_dim_2(graph: nx.Graph) -> int:
    """
    Compute the number of realizations in dimension 2
    combinatorially within PyRigi.

    Parameters
    ----------
    graph:
        A minimally rigid graph

    Notes
    -----
    The algorithm from :cite:p:`CapcoGalletEtAl2018` is used.
    """
    G = deepcopy(graph)
    deg_2 = 0
    while G.number_of_nodes() > 2 and G.min_degree() == 2:
        min_v = [v for v in G.nodes if G.degree(v) == 2]
        G.delete_vertices(min_v)
        deg_2 += len(min_v)

    if G.number_of_nodes() == 0:
        return 2 ** (deg_2 - 2)
    if G.number_of_nodes() == 2:
        return 2**deg_2
    return (2**deg_2) * _number_of_plane_realizations_min_rigid_dim_2_rec(
        _graph_to_bigraph(G), first=True
    )


def _number_of_plane_realizations_min_rigid_dim_2_rec(
    bigraph: list, first: bool = False
) -> int:
    """
    Compute the number of realizations in dimension 2 of a bigraph recursively.

    Parameters
    ----------
    bigraph:
        A set of biedges.
        This set comes initially from `_graph_to_bigraph`
    first:
        If the input is a bigraph where the left and right component are identical,
        then computation time can be saved by setting `first` to `True`
    """
    if _biedges_have_loop(bigraph):
        return 0
    if len(bigraph) == 1:
        return 1
    selected_edge = bigraph[0]
    result = _number_of_plane_realizations_min_rigid_dim_2_rec(
        _bigraph_contract_delete(bigraph, [selected_edge])
    )
    if first:
        result *= 2
    else:
        result += _number_of_plane_realizations_min_rigid_dim_2_rec(
            _bigraph_delete_contract(bigraph, [selected_edge])
        )

    subsets = more_itertools.powerset(bigraph[1:])
    for sub in subsets:
        if len(sub) == 0 or len(sub) == len(bigraph) - 1:
            continue
        sub_M = list(sub)
        sub_M.append(selected_edge)
        sub_N = []
        for be in bigraph:
            if be not in sub_M:
                sub_N.append(be)
        sub_N.append(selected_edge)

        sub_big_M = _bigraph_contract_delete(bigraph, sub_M)
        sub_big_N = _bigraph_delete_contract(bigraph, sub_N)
        if _bigraph_is_pseudo_laman(sub_big_N) and _bigraph_is_pseudo_laman(sub_big_M):
            sub_result = _number_of_plane_realizations_min_rigid_dim_2_rec(sub_big_M)
            if sub_result != 0:
                sub_result *= _number_of_plane_realizations_min_rigid_dim_2_rec(
                    sub_big_N
                )
            result += sub_result

    return result


def _bigraph_is_pseudo_laman(bigraph: list) -> bool:
    """
    Check whether a bigraph given by a list of biedges is pseudo laman.
    """
    graph1 = nx.Graph([be[0] for be in bigraph])
    graph2 = nx.Graph([be[1] for be in bigraph])
    return (
        graph1.number_of_nodes()
        - len(list(nx.connected_components(graph1)))
        + graph2.number_of_nodes()
        - len(list(nx.connected_components(graph2)))
        == len(bigraph) + 1
    )


def _biedges_have_loop(biedges: list) -> bool:
    """
    Check whether a list of biedges contains a loop.
    """
    return any(be[0][0] == be[0][1] or be[1][0] == be[1][1] for be in biedges)


def _graph_to_bigraph(graph: nx.Graph) -> list:
    """
    Generate a bigraph from a graph.
    A bigraph here is given by a set of biedges,
    i.e. pairs of edges of the two sides of the graph
    correspdonding to each other.

    Parameters
    ----------
    graph:
        A minimally rigid graph
    """
    n = graph.number_of_nodes()
    if n < 2:
        raise ValueError("Graph is to small")
    biedges = []
    edges = graph.edge_list()
    vertices = graph.vertex_list()
    mapping = {vertices[i]: i + 1 for i in range(n)}
    for edge in edges:
        biedges.append(
            [
                [mapping[edge[0]], mapping[edge[1]]],
                [mapping[edge[0]], mapping[edge[1]]],
            ]
        )
    return biedges


def _bigraph_contract_delete(biedges: list, select: list) -> list:
    """
    Contract edges of a bigraph on the left
    and delete on the right.
    """
    new_biedges = copy(biedges)
    for be in select:
        new_biedges.remove(be)
    n = max([max(*be[0], *be[1]) for be in biedges])
    mapping = {i: i for i in range(n + 1)}
    contract_graph = nx.Graph([be[0] for be in select])
    contract_sets = list(nx.connected_components(contract_graph))
    for i in range(len(contract_sets)):
        for v in contract_sets[i]:
            mapping[v] = n + 1 + i
    new_biedges2 = []
    for be in new_biedges:
        new_biedges2.append(
            [
                [mapping[be[0][0]], mapping[be[0][1]]],
                be[1],
            ]
        )
    return new_biedges2


def _bigraph_delete_contract(biedges: list, select: list) -> list:
    """
    Contract edges of a bigraph on the right
    and delete on the left.
    """
    new_biedges = copy(biedges)
    for be in select:
        new_biedges.remove(be)
    n = max([max(*be[0], *be[1]) for be in biedges])
    mapping = {i: i for i in range(n + 1)}
    contract_graph = nx.Graph([be[1] for be in select])
    contract_sets = list(nx.connected_components(contract_graph))
    for i in range(len(contract_sets)):
        for v in contract_sets[i]:
            mapping[v] = n + 1 + i
    new_biedges2 = []
    for be in new_biedges:
        new_biedges2.append(
            [
                be[0],
                [mapping[be[1][0]], mapping[be[1][1]]],
            ]
        )
    return new_biedges2
