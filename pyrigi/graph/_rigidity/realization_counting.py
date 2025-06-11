"""
This module provides algorithms related to rigid realization counting.
"""

import math
from copy import deepcopy
import more_itertools
import importlib.util

import networkx as nx

import pyrigi._utils._input_check as _input_check
import pyrigi.graph._rigidity.generic as generic_rigidity
import pyrigi.graph._rigidity.global_ as global_rigidity
import pyrigi.graph._sparsity.sparsity as sparsity
import pyrigi.graph._export.export as export


def number_of_realizations(
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

    For rigid graphs, TODO

    Caution: Currently the method only works if the python package ``lnumber``
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
        Currently, only ``dim=2`` is supported.
    algorithm:
        If "default" pyrigi checks which algorithm is available for the parameters and choses this one.
        If "pyrigi" a pure pyrigi implementation is used.
        If "lnumber" uses the ``lnumber`` package is used. This needs to be installed separately.
    spherical:
        If ``True``, the number of spherical realizations of the graph is returned.
        If ``False`` (default), the number of planar realizations is returned.
    count_reflection:
        If ``True``, the number of realizations is computed modulo direct isometries.
        But reflection is counted to be non-congruent as used in
        :cite:p:`CapcoGalletEtAl2018` and
        :cite:p:`GalletGraseggerSchicho2020`.
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

    if not generic_rigidity.is_rigid(graph, dim):
        return math.inf

    if algorithm == "default":
        if dim == 1:
            algorithm = "pyrigi"
        elif dim == 2:
            if graph.number_of_edges() > 2 * graph.number_of_nodes() - 3:
                algorithm = "pyrigi"
            elif importlib.util.find_spec('lnumber') is not None:
                algorithm = "lnumber"
            else:
                if spherical:
                    algorithm = "pyrigi"
                else:
                    raise ImportError(
                        "For counting the number of plane realizations, "
                        "the optional package 'lnumber' is used, "
                        "run `pip install pyrigi[realization-counting]`!"
                    )
    if algorithm == "pyrigi":
        if spherical or graph.number_of_edges() > 2 * graph.number_of_nodes() - 3:
            _input_check.dimension_for_algorithm(
                dim, [1, 2], "number_of_realizations"
            )
        else:
            _input_check.dimension_for_algorithm(
                dim, [1], "number_of_realizations"
            )
    elif algorithm == "lnumber":
        _input_check.dimension_for_algorithm(
            dim, [2], "number_of_realizations"
        )

    if graph.number_of_nodes() == 1:
        return 1

    if count_reflection:
        fac = 1
    else:
        fac = 2

    if dim == 1:
        if global_rigidity.is_globally_rigid(graph, dim):
            return 2 // fac
        elif generic_rigidity.is_min_rigid(graph, dim):
            G = deepcopy(graph)
            deg_1 = 0
            while len(G.vertex_list()) > 2 and G.min_degree() == 1:
                min_v = [v for v in G.vertex_list() if G.degree(v) == 1]
                G.delete_vertices(min_v)
                deg_1 += len(min_v)
            if G.number_of_nodes() == 1:
                return (2**deg_1) // fac
            else:
                return (2 // fac) * (2**deg_1)
        else:
            # not 2-connected
            G = deepcopy(graph)
            cut = list(nx.all_node_cuts(G))[0]
            G.delete_vertices(cut)
            con = nx.connected_components(G)
            sub = [graph.subgraph(c.union(cut)).copy() for c in con]
            return fac * math.prod(
                [
                    number_of_realizations(
                        g, dim, algorithm, spherical, count_reflection
                    )
                    for g in sub
                ]
            )

    # dim == 2 from now on
    if graph.number_of_nodes() == 2 and graph.number_of_edges() == 1:
        return 1

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
            else:
                return lnumber.lnumber(graph_int) // fac
        elif algorithm == "pyrigi":
            if spherical:
                return _number_of_spherical_realizations_min_rigid_dim_2(graph) // fac
            else:
                raise NotImplementedError()
    else:  # not minimally rigid
        if algorithm == "lnumber":
            raise ValueError(
                "The algorithm `lnumber` is only available for minimally rigid graphs " +
                "but the input graph is not minimally rigid."
            )
        if global_rigidity.is_globally_rigid(graph, dim):
            return 2 // fac
        else:
            G = deepcopy(graph)
            cut = list(nx.all_node_cuts(G))[0]
            # Case where the graph is not 3-connected
            if len(cut) == 2:
                G.delete_vertices(cut)
                con = nx.connected_components(G)
                sub = [graph.subgraph(c.union(cut)).copy() for c in con]
                # Case where the vertices of the cut are adjacent
                if graph.has_edge(*cut):
                    return fac ** (len(sub) - 1) * math.prod(
                        [
                            number_of_realizations(
                                g, dim, algorithm, spherical, count_reflection
                            )
                            for g in sub
                        ]
                    )
                else:
                    rig = [generic_rigidity.is_rigid(g, dim) for g in sub]
                    count_rig = rig.count(True)
                    if count_rig > 1:
                        [g.add_edge(*cut) for g in sub]
                        return fac ** (len(sub) - 1) * math.prod(
                            [
                                number_of_realizations(
                                    g, dim, algorithm, spherical, count_reflection
                                )
                                for g in sub
                            ]
                        )
                    else:
                        pos = rig.index(True)
                        res = fac ** (len(sub) - 1) * number_of_realizations(
                            sub[pos], dim, algorithm, spherical, count_reflection
                        )
                        [g.add_edge(*cut) for g in (sub[0:pos] + sub[pos + 1 :])]
                        return res * math.prod(
                            [
                                number_of_realizations(
                                    g, dim, algorithm, spherical, count_reflection
                                )
                                for g in (sub[0:pos] + sub[pos + 1 :])
                            ]
                        )
            # Case where the graph is 3-connected but not redundantly rigid
            else:
                # Find edge e sucht that graph - e is not rigid
                G = deepcopy(graph)
                edges = G.edge_list()
                found = False
                while len(edges) > 0 and not found:
                    e = edges.pop()
                    G.remove_edge(*e)
                    if generic_rigidity.is_rigid(G, dim):
                        G.add_edge(*e)
                    else:
                        found = True
                # Get maximal rigid subgraphs
                comp = G.rigid_components(dim)
                max_sub = [G.subgraph(c).copy() for c in comp]
                # Get minimally rigid spanning subgraphs
                span = [sparsity.spanning_kl_sparse_subgraph(g, 2, 3) for g in max_sub]
                # Compute result
                prod_g = math.prod(
                    [
                        number_of_realizations(
                            g, dim, algorithm, spherical, count_reflection
                        )
                        for g in max_sub
                    ]
                )
                prod_h = math.prod(
                    [
                        number_of_realizations(
                            h, dim, algorithm, spherical, count_reflection
                        )
                        for h in span
                    ]
                )
                H = nx.compose(span[0], span[1])
                for h in span[2:]:
                    H = nx.compose(H, h)
                H.add_edge(*e)
                return (
                    number_of_realizations(
                        H, dim, algorithm, spherical, count_reflection
                    )
                    * prod_g
                    // prod_h
                )


def _number_of_spherical_realizations_min_rigid_dim_2(graph: nx.Graph) -> int | str:
    G = deepcopy(graph)
    deg_2 = 0
    while len(G.vertex_list()) > 2 and G.min_degree() == 2:
        min_v = [v for v in G.vertex_list() if G.degree(v) == 2]
        G.delete_vertices(min_v)
        deg_2 += len(min_v)

    if G.number_of_nodes() == 0:
        return 2 ** (deg_2 - 2)
    elif G.number_of_nodes() == 2:
        return 2**deg_2
    else:
        return (2**deg_2) * _number_of_spherical_realizations_min_rigid_dim_2_rec(
            _graph_to_quadrograph(G)
        )


def _number_of_spherical_realizations_min_rigid_dim_2_rec(quadrograph):
    """Computes the number of spherical realizations recursively"""
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
    sum = 0
    counter = max(quad_N) + 1
    for subset in allLists:
        set_I = [a, b] + list(subset)
        Q22 = [q for q in Qprime if len(set(q).intersection(set_I)) == 2]
        if len(Q22) > 0:
            continue
        Q40 = [q for q in Qprime if len(set(q).intersection(set_I)) == 4]
        Q31 = [q for q in Qprime if len(set(q).intersection(set_I)) == 3]

        if len(set_I) - len(Q40 + Q31) == 2:
            set_J = [j for j in quad_N if not (j in set_I)]
            Q13new = [
                [x if (x in set_J) else counter for x in q]
                for q in Qprime
                if len(set(q).intersection(set_I)) == 1
            ]
            Q04 = [q for q in Qprime if len(set(q).intersection(set_I)) == 0]
            Q31new = [[x if (x in set_I) else counter for x in q] for q in Q31]
            sum = sum + _number_of_spherical_realizations_min_rigid_dim_2_rec(
                [set_I + [counter], Q40 + Q31new]
            ) * _number_of_spherical_realizations_min_rigid_dim_2_rec(
                [set_J + [counter], Q13new + Q04]
            )
        else:
            continue
    return sum


def _graph_to_quadrograph(graph):
    """Prepares a graph in edge representation for usage in SphereRealizationCountRec"""
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
