import networkx as nx
from networkx import maximum_flow, maximum_flow_value


from pyrigi.graph._sparsity._pebble_digraph import PebbleDiGraph

from typing import Hashable

# SUBROUTINE 1: MAX-FLOW


def _max_flow(D: nx.DiGraph, s: str, t: str, g: str):
    """
    Return the maximum flow and the source side of a minimum s-t cut.

    The source ``s`` itself is not part of the returned cut side.

    Definitions
    -----------
    :prf:ref:`Maximum flow and minimum cut <def-max-flow>`

    Parameters
    ----------
    D:
        A directed graph representing the flow network.
    s:
        The source vertex node identifier.
    t:
        The sink vertex node identifier.
    g:
        The name of the capacity attribute.
    """
    x = maximum_flow(D, s, t, capacity=g)[1]
    _, (S, _) = nx.minimum_cut(D, s, t, capacity=g)
    return (x, S - {s})


# SUBROUTINE 2: INDEGREE BOUNDED ORIENTATIONS


def _build_flow_network(G: nx.Graph, k: int) -> nx.DiGraph:
    """
    Return the flow network for the ``k``-indegree bounded orientation of ``G``.

    Parameters
    ----------
    G:
        An undirected graph.
    k:
        The maximum allowed indegree for each vertex.
    """
    flow_network = nx.DiGraph()

    source = "s"
    sink = "t"
    flow_network.add_node(source)
    flow_network.add_node(sink)
    flow_network.add_nodes_from(G.nodes())

    for e in G.edges():
        flow_network.add_node(f"z_{e}")
        flow_network.add_edge(source, f"z_{e}", capacity=1)
        flow_network.add_edge(f"z_{e}", e[0], capacity=1)
        flow_network.add_edge(f"z_{e}", e[1], capacity=1)

    for v in G.nodes():
        flow_network.add_edge(v, sink, capacity=k)

    return flow_network


def _k_indegree_bounded_orientation(G: nx.Graph, k: int) -> tuple[set, nx.DiGraph]:
    """
    Return an orientation of ``G`` with indegree at most ``k`` at every vertex.

    A flow network is constructed to determine whether the edges of the undirected
    graph can be oriented without exceeding the maximum indegree ``k`` at any vertex.
    A pair ``(X, D)`` is returned.
    If a valid orientation exists, ``X`` is ``None`` and ``D`` is the
    ``k``-indegree bounded orientation of the graph.
    If no such orientation exists, ``X`` is the violating set of vertices
    and ``D`` is ``None``.

    Parameters
    ----------
    G:
        An undirected graph to be oriented.
    k:
        The maximum allowed indegree for each vertex.
    """
    if len(G.edges()) > k * len(G.nodes()):
        return (set(G.nodes()), None)

    flow_network = _build_flow_network(G, k)
    f, S = _max_flow(flow_network, "s", "t", "capacity")
    x = maximum_flow_value(flow_network, "s", "t", "capacity")

    if x == len(G.edges()):
        D = nx.DiGraph()
        D.add_nodes_from(G.nodes())
        D.add_edges_from(
            (e[1], e[0]) if f[f"z_{e}"][e[0]] == 1 else (e[0], e[1]) for e in G.edges()
        )
        return (None, D)
    else:
        # we build the violating set of vertices
        violating_set = {v for v in G.nodes if v in S}
        return (violating_set, None)


# SUBROUTINE 3 & 4: FOREST DECOMPOSITION & BOUNDED FOREST DECOMPOSITION
# We will treat the forest F as a dictionary of the edges,
# where the keys are the indices of the trees in the forest.


def _indegree_F(v: int, F: dict):
    """
    Return the indegree of the vertex ``v`` in the labelled edges of ``F``.

    Parameters
    ----------
    v:
        A vertex in the graph.
    F:
        A dictionary of edges representing a partition of the edges
        into subgraphs (trees).
    """
    return sum(1 for e, label in F.items() if label is not None and e[1] == v)


def _is_arc_joining_F(e: tuple, F: dict, index: int):
    """
    Return whether the edge ``e`` joins two components of the forest ``F[index]``.

    Parameters
    ----------
    e:
        An edge represented as a tuple ``(u, v)``.
    F:
        A dictionary of edges representing a partition of the edges
        into subgraphs (trees).
    index:
        The index of the tree in ``F`` to check.
    """
    f_i_edges = {edge for edge, label in F.items() if label == index}
    H = nx.Graph(f_i_edges)
    u, v = e

    if u in H.nodes() and v in H.nodes():
        return not nx.has_path(H, u, v)
    return True


def _build_auxiliary_graph(G: nx.DiGraph, F: dict, tau: dict, k: int):
    """
    Return the auxiliary graph ``D(F)`` for the forest decomposition problem.

    Parameters
    ----------
    G:
        The directed graph whose edges are decomposed.
    F:
        A dictionary mapping edges to their forest index or ``None``.
    tau:
        A dictionary mapping vertices to their indegree bounds.
    k:
        The number of forests.
    """
    D = nx.DiGraph()
    D.add_nodes_from(G.nodes())
    D.add_nodes_from(G.edges())
    D.add_node("t")

    _add_vertex_to_edge_arcs(D, G, F, tau, k)
    _add_edge_to_sink_arcs(D, G, F, k)
    _add_edge_to_edge_arcs_same_head(D, G, F)
    _add_edge_to_edge_arcs_swap(D, G, F)

    return D


def _add_vertex_to_edge_arcs(D: nx.DiGraph, G: nx.DiGraph, F: dict, tau: dict, k: int):
    """Add arcs (v, e) for unlabelled edges e with head v and spare indegree."""
    D.add_edges_from(
        (e[1], e)
        for e in G.edges()
        if F[e] is None and _indegree_F(e[1], F) < k - tau.get(e[1], 0)
    )


def _add_edge_to_sink_arcs(D: nx.DiGraph, G: nx.DiGraph, F: dict, k: int):
    """Condition 2: arcs (e, t) for edges e joining some tree F[i]."""
    D.add_edges_from(
        (e, "t") for e in G.edges() if any(_is_arc_joining_F(e, F, i) for i in range(k))
    )


def _add_edge_to_edge_arcs_same_head(D: nx.DiGraph, G: nx.DiGraph, F: dict):
    """Condition 3: arcs (e, f) with e labelled, f unlabelled and the same head."""
    D.add_edges_from(
        (e, f)
        for e in G.edges()
        if F[e] is not None
        for f in G.edges()
        if F[f] is None and e[1] == f[1]
    )


def _add_edge_to_edge_arcs_swap(D: nx.DiGraph, G: nx.DiGraph, F: dict):
    """Add arcs (f, e) with f unlabelled and joining F[i] once e is removed."""
    for f in G.edges():
        if F[f] is not None:
            continue
        for e in G.edges():
            i = F[e]
            if i is None:
                continue
            F_temp = {**F, e: None}
            if _is_arc_joining_F(f, F_temp, i):
                D.add_edge(f, e)


def _augment_F(P: list, F: dict, k: int):
    """
    Update the labelling ``F`` along the augmenting path ``P`` and return it.

    Parameters
    ----------
    P:
        A path in the auxiliary graph, from a vertex to the sink ``"t"``.
    F:
        A dictionary mapping edges to their forest index or ``None``.
    k:
        The number of forests.
    """
    edges_path = P[1:-1]
    e_r = edges_path[-1]

    p = min(j for j in range(k) if _is_arc_joining_F(e_r, F, j))

    labels = [F[e] for e in edges_path]
    new_labels = labels[1:] + [p]

    for e, new_label in zip(edges_path, new_labels):
        F[e] = new_label

    return F


def _search_augmenting_path(
    C: set, G: nx.DiGraph, F: dict, tau: dict, k: int, deficits: dict
) -> bool:
    """
    Search for an augmenting path starting from a deficient vertex in ``C``.

    If a path is found, the forest decomposition ``F`` is augmented along it
    and ``True`` is returned.

    Parameters
    ----------
    C:
        A set of vertices representing a connected component of the k-th forest.
    G:
        The original directed graph.
    F:
        A dictionary mapping edges to their forest index (0 to k-1) or ``None``.
    tau:
        A dictionary mapping vertices to their indegree bounds.
    k:
        The number of forests.
    deficits:
        A dictionary mapping vertices to their current deficit values.
    """
    v = next((node for node in C if deficits.get(node, 0) > 0), None)

    if v is None:
        return False

    aux_graph = _build_auxiliary_graph(G, F, tau, k)

    try:
        P = nx.shortest_path(aux_graph, source=v, target="t")
        _augment_F(P, F, k)
        deficits[v] -= 1
        return True

    except nx.NetworkXNoPath:
        deficits[v] = 0
        return False


def _k_forest_packing(G: nx.DiGraph, F: dict, k: int, tau: dict):
    """
    Extend the labelling ``F`` of the edges of ``G`` to a packing of ``k`` forests.

    The labelling is modified in place and returned.
    The result is a packing, not necessarily a decomposition:
    edges that cannot be added to any of the ``k`` forests keep the label ``None``.

    Parameters
    ----------
    G:
        The directed graph whose edges are decomposed.
    F:
        A dictionary mapping edges to their forest index (0 to k-1) or ``None``.
    k:
        The number of forests.
    tau:
        A dictionary mapping vertices to their indegree bounds.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    >>> F = {(0, 1): None, (1, 2): None, (0, 2): None}
    >>> tau = {0: 0, 1: 0, 2: 0}
    >>> _k_forest_packing(G, F, 2, tau)
    {(0, 1): 0, (1, 2): 1, (0, 2): 0}
    """
    if k > 1:
        _k_forest_packing(G, F, k - 1, tau)

    deficits = {v: k - tau[v] - _indegree_F(v, F) for v in G.nodes}

    while sum(deficits.values()) > 0:
        progress = False
        F_k_edges = [e for e, label in F.items() if F[e] == k - 1]
        H_k = nx.Graph()
        H_k.add_nodes_from(G.nodes())
        H_k.add_edges_from(F_k_edges)

        if (
            H_k.number_of_edges()
            == G.number_of_nodes() - nx.number_connected_components(G.to_undirected())
        ):
            break

        components = nx.connected_components(H_k)
        active_components = [
            C for C in components if sum(deficits.get(v) for v in C) > 0
        ]

        if not active_components:
            break

        for C in active_components:
            F_k_edges = [e for e, label in F.items() if F[e] == k - 1]
            H_k = nx.Graph()
            H_k.add_nodes_from(G.nodes())
            H_k.add_edges_from(F_k_edges)

            current_components = list(nx.connected_components(H_k))
            target_C = None
            for comp in current_components:
                if (
                    any(v in C for v in comp)
                    and sum(deficits.get(v, 0) for v in comp) > 0
                ):
                    target_C = comp
                    break

            if target_C is None:
                continue

            success = _search_augmenting_path(target_C, G, F, tau, k, deficits)
            if success:
                progress = True

        if not progress:
            break

    return F


def _k_forest_decomposition(G: nx.Graph, k: int) -> tuple[set, list]:
    """
    Return a decomposition of ``G`` into ``k`` edge-disjoint forests.

    A tuple ``(X, F_1, ..., F_k)`` is returned.
    If the edges of ``G`` can be partitioned into ``k`` forests,
    ``X`` is ``None`` and ``F_1, ..., F_k`` are these forests, each given as a set
    of edges.
    Otherwise, ``X`` is a vertex set violating (``k``, ``k``)-sparsity
    and ``F_1, ..., F_k`` are empty.

    Parameters
    ----------
    G:
        An undirected graph to be decomposed.
    k:
        The number of forests to decompose the graph into.
    """
    X, D = _k_indegree_bounded_orientation(G, k)

    if D is None:
        return (X, *[set() for _ in range(k)])
    return _k_bounded_forest_decomposition(D, k)


def _violating_set_of_unlabelled_edge(D: nx.DiGraph, F: dict, e: tuple, k: int):
    """
    Return a vertex set violating (``k``, ``k``)-sparsity once ``e`` is added.

    The labelled edges of ``F`` are put into a :class:`.PebbleDiGraph`
    (with reversed orientation, since the pebble game bounds outdegrees)
    and the pebble game is used to find the fundamental circuit of ``e``.

    Parameters
    ----------
    D:
        A ``k``-indegree bounded digraph.
    F:
        A dictionary mapping the edges of ``D`` to their labels
        (``None`` if unlabelled).
    e:
        An unlabelled edge of ``D``.
    k:
        The sparsity parameter.
    """
    P = PebbleDiGraph(k, k)
    P.add_nodes_from(D.nodes())
    P.add_edges_from((b, a) for (a, b), label in F.items() if label is not None)

    circuit = P.fundamental_circuit(*e)
    if circuit is None:
        raise ValueError("The edge can be added, so there is no violating set.")
    return set(circuit)


def _k_bounded_forest_decomposition(D: nx.DiGraph, k: int):
    """
    Return a decomposition of the ``k``-indegree bounded digraph ``D`` into forests.

    A tuple ``(X, F_1, ..., F_k)`` is returned.
    If the edges of ``D`` can be partitioned into ``k`` forests,
    ``X`` is ``None`` and ``F_1, ..., F_k`` are these forests, each given as a set
    of edges.
    Otherwise, ``X`` is a vertex set violating (``k``, ``k``)-sparsity
    and ``F_1, ..., F_k`` are empty.

    Parameters
    ----------
    D:
        A ``k``-indegree bounded digraph.
    k:
        The number of forests to decompose ``D`` into.
    """
    F = {e: None for e in D.edges()}
    _k_forest_packing(D, F, k, {v: 0 for v in D.nodes()})

    unlabelled = next((e for e, label in F.items() if label is None), None)
    if unlabelled is not None:
        X = _violating_set_of_unlabelled_edge(D, F, unlabelled, k)
        return (X, *[set() for _ in range(k)])

    return (None, *[{e for e, label in F.items() if label == i} for i in range(k)])


def _unit_capacity_digraph(D: nx.DiGraph) -> nx.DiGraph:
    """
    Return a simple digraph whose capacities count the parallel arcs of ``D``.
    """
    H = nx.DiGraph()
    H.add_nodes_from(D.nodes())
    for u, v in D.edges():
        if H.has_edge(u, v):
            H[u][v]["capacity"] += 1
        else:
            H.add_edge(u, v, capacity=1)
    return H


def _bridge_separating_terminal(D: nx.DiGraph, s, Z: set):
    """
    Return the vertices cut off from ``s`` by a single arc, if one contains a terminal.

    All vertices of ``D`` must be reachable from ``s``.
    Every arc is subdivided and a dominator tree is computed:
    an arc is a bridge exactly when its subdivision vertex dominates its head.
    The vertices of the dominator subtree of such a head are the ones
    that are only reachable through that arc.
    """
    S = nx.DiGraph()
    S.add_nodes_from(D.nodes())
    arcs = {}
    for u, v in D.edges():
        w = object()
        arcs[w] = v
        S.add_edge(u, w)
        S.add_edge(w, v)

    idom = nx.immediate_dominators(S, s)
    children = {}
    for x, d in idom.items():
        if x != s:
            children.setdefault(d, []).append(x)

    order = [s]
    for x in order:
        order.extend(children.get(x, []))
    has_terminal = {x: x in Z for x in order}
    for x in reversed(order):
        if x != s and has_terminal[x]:
            has_terminal[idom[x]] = True

    for w, v in arcs.items():
        if idom.get(v) is w and has_terminal[v]:
            X, stack = set(), [v]
            while stack:
                x = stack.pop()
                if x not in arcs:
                    X.add(x)
                stack.extend(children.get(x, []))
            return X
    return None


def _eta_rooted_term_connectivity(D: nx.DiGraph, eta: int, s, Z: set):
    """
    Return a vertex set containing a terminal with fewer than ``eta`` entering arcs.

    A nonempty set ``X`` not containing ``s`` is returned such that ``X`` intersects
    ``Z`` and the number of arcs of ``D`` entering ``X`` is less than ``eta``.
    If no such set exists, ``None`` is returned.
    Parallel arcs are allowed if ``D`` is a :class:`networkx.MultiDiGraph`;
    they are counted with multiplicity.

    Parameters
    ----------
    D:
        A directed graph.
    eta:
        The bound on the number of entering arcs.
    s:
        The root vertex.
    Z:
        A set of terminal vertices. The root is ignored if it lies in ``Z``.

    Examples
    --------
    >>> import networkx as nx
    >>> D = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    >>> _eta_rooted_term_connectivity(D, 2, 0, {1, 2})
    {1}
    >>> _eta_rooted_term_connectivity(D, 1, 0, {1, 2}) is None
    True
    """
    Z = set(Z) - {s}
    if eta <= 0 or not Z:
        return None

    reachable = nx.descendants(D, s) | {s}
    if not Z <= reachable:
        return set(D.nodes()) - reachable
    if eta == 1:
        return None

    if eta == 2:
        X = _bridge_separating_terminal(D.subgraph(reachable).copy(), s, Z)
        return None if X is None else X | (set(D.nodes()) - reachable)

    H = _unit_capacity_digraph(D)

    if eta == 3:
        H_prime = H.copy()
        for v in Z:
            if H_prime.has_edge(v, s):
                H_prime[v][s]["capacity"] += 3
            else:
                H_prime.add_edge(v, s, capacity=3)
        component_of_s = _three_edge_connected_component(H_prime, s)
        terminals = [v for v in Z if v not in component_of_s]
    else:
        terminals = Z

    for v in terminals:
        value, T = _flow_value_and_sink_side(H, s, v)
        if value < eta:
            return T
    return None


def _flow_value_and_sink_side(H: nx.DiGraph, s, t):
    """
    Return the value of a maximum ``s``-``t`` flow and the sink side of a minimum cut.

    The :func:`max_flow` subroutine is used, with the arc attribute ``capacity``.

    Parameters
    ----------
    H:
        A digraph with ``capacity`` attributes on the arcs.
    s:
        The source.
    t:
        The sink.
    """
    _, S = _max_flow(H, s, t, "capacity")
    value = maximum_flow_value(H, s, t, "capacity")
    return value, set(H.nodes()) - S - {s}


def _three_edge_connected_component(H: nx.DiGraph, s) -> set:
    """
    Return the vertices that are 3-arc-connected to ``s`` in both directions.

    Two vertices are 3-edge-connected if there are three arc-disjoint paths
    between them in each direction.


    Parameters
    ----------
    H:
        A digraph whose arc capacities are the numbers of parallel arcs.
    s:
        The vertex whose component is returned.
    """

    def reaches_three(a, b):
        return _flow_value_and_sink_side(H, a, b)[0] >= 3

    return {s} | {
        v for v in H.nodes() if v != s and reaches_three(s, v) and reaches_three(v, s)
    }


# SUBROUTINE 6: ROOTED ARC-CONNECTIVITY


def _eta_rooted_arc_connectivity(D: nx.DiGraph, eta: int, s):
    """
    Return a vertex set not containing ``s`` with fewer than ``eta`` entering arcs.

    A nonempty set ``X`` not containing ``s`` is returned such that
    the number of arcs of ``D`` entering ``X`` is less than ``eta``.
    If no such set exists, that is, if ``D`` is rooted ``eta``-arc-connected
    with root ``s``, ``None`` is returned.
    Parallel arcs are allowed if ``D`` is a :class:`networkx.MultiDiGraph`.

    For ``eta <= 3`` this is :func:`eta_rooted_term_connectivity` with all
    vertices other than ``s`` as terminals.
    For ``eta >= 4`` the paper adds ``eta`` parallel arcs from every vertex to
    ``s`` and applies Gabow's algorithm.
    Here, one maximum flow from ``s`` to every other vertex is used instead,
    which gives the same output but is not as fast.

    Parameters
    ----------
    D:
        A directed graph.
    eta:
        The required rooted arc-connectivity.
    s:
        The root vertex.

    Examples
    --------
    >>> import networkx as nx
    >>> D = nx.DiGraph([(0, 1), (1, 2), (2, 1)])
    >>> _eta_rooted_arc_connectivity(D, 2, 0)
    {1, 2}
    >>> _eta_rooted_arc_connectivity(D, 1, 0) is None
    True
    """
    if eta <= 3:
        return _eta_rooted_term_connectivity(D, eta, s, set(D.nodes()) - {s})

    # D' is D with eta parallel arcs from every vertex to s (encoded as capacities).
    # Its cuts of value below eta correspond exactly to the nonempty sets X
    # not containing s with fewer than eta entering arcs in D.
    D_prime = _unit_capacity_digraph(D)
    for v in D.nodes():
        if v == s:
            continue
        if D_prime.has_edge(v, s):
            D_prime[v][s]["capacity"] += eta
        else:
            D_prime.add_edge(v, s, capacity=eta)

    for v in D.nodes():
        if v == s:
            continue
        value, X = _flow_value_and_sink_side(D_prime, s, v)
        if value < eta:
            return X
    return None


def _build_rooted_digraph(D0: nx.DiGraph, U0: set, k: int):
    """
    Return an auxiliary digraph for check_superset_sparsity.

    The vertices of ``U0`` are deleted from ``D0``, a new root ``s`` is added and,
    for every remaining vertex ``v``, ``k - indegree(v)`` parallel arcs
    from ``s`` to ``v`` are inserted.
    For a strict superset ``X`` of ``U0``, the set ``X - U0`` has fewer than
    ``l - t*k`` entering arcs in ``D'_0`` if and only if ``X`` violates
    (``k``, ``l``)-sparsity of the underlying graph of ``D0``, where ``t = |U0|``.

    A pair ``(D_prime, s)`` is returned, where ``D_prime`` is a
    :class:`networkx.MultiDiGraph` and ``s`` is its root.

    Parameters
    ----------
    D0:
        A ``k``-indegree bounded digraph that is ``U0``-source.
    U0:
        An independent vertex set.
    k:
        The maximum allowed indegree.

    Examples
    --------
    >>> import networkx as nx
    >>> D0 = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    >>> D_prime, s = _build_rooted_digraph(D0, {0}, 2)
    >>> sorted(D_prime.nodes() - {s})
    [1, 2]
    >>> D_prime.number_of_edges(s, 1), D_prime.number_of_edges(s, 2)
    (1, 0)
    """
    U0 = set(U0)
    if any(D0.in_degree(u) > 0 for u in U0):
        raise ValueError("D0 must be U0-source.")
    if any(D0.in_degree(v) > k for v in D0.nodes()):
        raise ValueError("D0 must be k-indegree bounded.")

    s = object()
    D_prime = nx.MultiDiGraph()
    D_prime.add_nodes_from(v for v in D0.nodes() if v not in U0)
    D_prime.add_node(s)
    D_prime.add_edges_from((u, v) for u, v in D0.edges() if u not in U0)

    for v in D0.nodes():
        if v not in U0:
            D_prime.add_edges_from((s, v) for _ in range(k - D0.in_degree(v)))

    return D_prime, s


def _check_superset_sparsity(D: nx.DiGraph, U: set, k: int, ell: int) -> set:
    """
    Return a vertex set that violates (k, l)-sparsity of the graph.

    The violating set strictly contains the given set U,
    if such a set exists. Otherwise, return None.
    It is required the inequality tk <= l <= (t+1)k and that D is U-source.

    Parameters
    ----------
    D:
        A k-indegree bounded digraph.
    U:
        An indipendent vertex set with t elements.
    k:
        First parameter of the (k,l)-sparsity condition.
    l:
        Second parameter of the (k,l)-sparsity condition.
    """
    U = set() if U is None else set(U)
    if not (k * len(U) <= ell <= k * (len(U) + 1)):
        raise ValueError(
            "The parameters k and l do not satisfy the required inequality."
        )

    D_prime, s = _build_rooted_digraph(D, U, k)

    X = _eta_rooted_arc_connectivity(D_prime, ell - k * len(U), s)
    if X is None:
        return None
    return X | U


def _reorient(D: nx.DiGraph, U: set, k: int) -> tuple[set, nx.DiGraph]:
    """
    Return a k-indegree bounded U-source reorientation of D,
    or a vertex set violating (k,k)-sparsity if no such reorientation exists.

    Parameters
    ----------
    D:
        A k-indegree bounded digraph.
    U:
        An independent vertex set with t elements.
    k:
        The maximum allowed indegree for each vertex.
    """
    D = D.copy()
    U = set(U)

    if any(D.in_degree(v) > k for v in D.nodes()):
        raise ValueError("D must be k-indegree bounded.")

    while any(D.in_degree(u) > 0 for u in U):
        # vertices outside U with spare indegree
        S = {v for v in D.nodes() if v not in U and D.in_degree(v) < k}
        P = None

        if S:
            paths = nx.multi_source_dijkstra_path(D, S)
            for target, path in paths.items():
                if target in U:
                    P = path
                    break

        if P is None:
            T = set(U)
            for u in U:
                T.update(nx.ancestors(D, u))
            return (T, None)

        for u, v in zip(P[:-1], P[1:]):
            D.remove_edge(u, v)
            D.add_edge(v, u)

    return (None, D)


def _centroid(T: nx.Graph) -> Hashable:
    """
    Return a centroid of the tree ``T``.

    A centroid is a vertex whose removal leaves components with
    at most ``|V(T)| / 2`` vertices each.
    It is found with two (iterative) depth-first searches:
    the first computes the subtree sizes, the second walks from the root
    towards the heavy subtree until no subtree has more than half of the vertices.
    The running time is linear.

    Parameters
    ----------
    T:
        A tree (connected and acyclic undirected graph).

    Examples
    --------
    >>> import networkx as nx
    >>> _centroid(nx.path_graph(5))
    2
    >>> _centroid(nx.star_graph(4))
    0
    """
    n = T.number_of_nodes()
    if n == 0:
        raise ValueError("The tree must have at least one vertex.")
    if T.number_of_edges() != n - 1 or not nx.is_connected(T):
        raise ValueError("T must be a tree.")

    root = next(iter(T.nodes()))

    # First DFS: parents and preorder, then subtree sizes bottom-up.
    parent = {root: None}
    order = []
    stack = [root]
    while stack:
        v = stack.pop()
        order.append(v)
        for w in T.neighbors(v):
            if w != parent[v]:
                parent[w] = v
                stack.append(w)

    size = {v: 1 for v in order}
    for v in reversed(order):
        if parent[v] is not None:
            size[parent[v]] += size[v]

    # Second DFS: move to a child with more than n/2 vertices while one exists.
    c = root
    while True:
        heavy = next(
            (w for w in T.neighbors(c) if w != parent[c] and 2 * size[w] > n),
            None,
        )
        if heavy is None:
            return c
        c = heavy


def _check_sparsity_for_l_less_than_k(G: nx.Graph, K: int, L: int) -> set:
    """
    Returns a vertex set that violates the (k,l)-sparsity of G if it exists,
    otherwise returns None.

    Parameters:
    G:
        An undirected Graph
    k:
        First parameter of the sparsity condition.
    l:
        Second paramater of the sparsity condition."""
    X, D = _k_indegree_bounded_orientation(G, K)
    if X is not None:
        # X violates (k,0)-sparsity
        return X
    return _check_superset_sparsity(D, set(), K, L)


def _saturated_violation(D: nx.DiGraph, T: nx.Graph, k: int, ell: int) -> set:
    """
    Return either None or a set violating (k, l)-sparsity in the underlying graph of D.

    Whenever the underlying graph of D contains a violating set that is saturated
    by T, the procedure returns it.
    We say that a tree saturates a nonempty vertex set X
    if the induced graph T[X] is connected.

    Parameters
    ----------
    D:
        A k-indegree bounded digraph.
    T:
        A spanning tree of the underlying graph of D.
    k:
        First parameter of the sparsity condition.
    l:
        Second parameter of the sparsity condition, with k < l <= 2k.
    """
    n = T.number_of_nodes()
    if n == 0:
        raise ValueError("The tree must have at least one vertex.")
    if T.number_of_edges() != n - 1 or not nx.is_connected(T):
        raise ValueError("T must be a tree.")

    c = _centroid(T)

    X, D0 = _reorient(D, {c}, k)
    if X is not None:
        return X

    X = _check_superset_sparsity(D0, {c}, k, ell)
    if X is not None:
        return X

    T_without_c = T.copy()
    T_without_c.remove_node(c)
    for component in nx.connected_components(T_without_c):
        X = _saturated_violation(
            D.subgraph(component).copy(), T.subgraph(component).copy(), k, ell
        )
        if X is not None:
            return X

    return None


def _check_sparsity_for_l_between_k_and_2k(G: nx.Graph, K: int, L: int) -> set:
    """
    Check the (k,l)-sparsity of ``G`` for k < l < 2k.

    Return a violating vertex set if it exists, otherwise None.

    Parameters
    ----------
    G:
        An undirected Graph.
    k:
        First Parameter in the sparsity condition.
    l:
        Second Parameter in the sparsity condition.
    """
    X, *F = _k_forest_decomposition(G, K)
    if X is not None:
        return X
    return _check_sparsity_with_decomp(G, F, K, L)


def _check_sparsity_with_decomp(G: nx.Graph, F: list, K: int, L: int) -> set:
    """
    Return a vertex set that violates the (k, l)-sparsity of G if it exists,
    None otherwise.

    Parameters
    ----------
    G:
        An undirected Graph.
    F:
        A list of k sets of edges representing a forest decomposition of G.
    k:
        First Parameter in the sparsity condition.
    l:
        Second Parameter in the sparsity condition.
    """
    _, D = _k_indegree_bounded_orientation(G, K)
    for i in range(L - K):
        forest = nx.Graph()
        forest.add_nodes_from(G.nodes())
        forest.add_edges_from(F[i])
        for component in nx.connected_components(forest):
            if len(component) < 2:
                continue
            X = _saturated_violation(
                D.subgraph(component).copy(), forest.subgraph(component).copy(), K, L
            )
            if X is not None:
                return X

    return None


def _check_terminal_superset_sparsity(D: nx.DiGraph, c: str, Z: set, K: int) -> set:
    """
    Returns a violating set of the underlying graph of D such that in X,
    X intersection Z is not None and X different from {c}.
    If such set doesn't exist returns None.

    Parameters
    ----------
    D:
        k-indegree bounded digraph.
    c:
        a vertex, source in D.
    Z:
        a set of vertices not containg c.
    k:
    """

    D_prime, s = _build_rooted_digraph(D, {c}, K)

    X = _eta_rooted_term_connectivity(D_prime, K, s, Z)

    if not X:
        return None
    else:
        return X | {c}


def _saturated_violation_ell_equal_2k(D: nx.DiGraph, T: nx.Graph, K: int) -> set:
    """
    Returns a set violating (k,2k)-sparsity if it exists, None otherwise.

    The procedure returns a violating set whenever the underlying graph of D contains one
    that is saturated by T.

    Parameters
    ----------
    D:
        a k-indegree-bounded digraph.
    T:
        a spanning tree of the underlying graph of D.
    k:
        first paramater of sparsity condition.
    """

    G = D.to_undirected()
    c = _centroid(T)
    X, D0 = _reorient(D, {c}, K)
    if X is not None:
        # X violates (k,k)-sparsity
        return X
    neighbors_c = set(G.neighbors(c))
    Z = set(G.nodes()) - {c} - neighbors_c
    X = _check_terminal_superset_sparsity(D0, c, Z, K)
    if X is not None:
        # X violates (k,2k)-sparsity
        return X
    if K == 1 and len(neighbors_c) > 1:
        return neighbors_c | {c}

    induced_graph = G.subgraph(neighbors_c)
    num_induced_edges = induced_graph.number_of_edges()
    if K == 2 and num_induced_edges > 0:
        u, v = next(iter(induced_graph.edges()))
        return {u, v, c}

    D0 = induced_graph.to_directed()
    if K > 2:
        X, *F = _k_bounded_forest_decomposition(D0, K)
        if X is not None:
            return X | {c}
        X = _check_sparsity_with_decomp(induced_graph, F, K - 1, K)
        if X is not None:
            return X | {c}

    T_without_c = T.copy()
    T_without_c.remove_node(c)
    for component in nx.connected_components(T_without_c):
        X = _saturated_violation_ell_equal_2k(
            D.subgraph(component), T.subgraph(component), K
        )
        if X is not None:
            return X

    return None


def _check_sparsity_for_l_equal_2k(G: nx.Graph, K: int, L: int) -> set:
    """
    Returns a vertex set that violates the (k, 2k)-sparsity of G if it exists,
    None otherwise.

    Parameters
    ----------
    G:
        A graph.
    k:
        First parameter in the sparsity condition.
    L:
        Second Parameter in the sparsity condition.
    """
    assert L == 2 * K
    X, *F = _k_forest_decomposition(G, K)
    if X is not None:
        return X
    _, D = _k_indegree_bounded_orientation(G, K)
    for i in range(K):
        forest = nx.Graph()
        forest.add_nodes_from(G.nodes())
        forest.add_edges_from(F[i])
        for component in nx.connected_components(forest):
            X = _saturated_violation_ell_equal_2k(
                D.subgraph(component), forest.subgraph(component), K
            )
            if X is not None:
                return X

    return None


def _check_sparsity_for_l_between_2k_and_3k(G: nx.Graph, K: int, L: int) -> set:
    """
    Return a vertex set that violates the (k,l)-sparsity of G if it exists;
    None otherwise.

    Parameters
    ----------
    G:
        A graph.
    k:
        First Parameter in the sparsity condition.
    ell:
        Second Parameter in the sparsity condition.
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    D = nx.DiGraph()
    D.add_nodes_from(G.nodes())

    for u, v in G.edges():
        X, D = _reorient(D, {u, v}, K)
        X = _check_superset_sparsity(D, {u, v}, K, L + 1)
        if X is not None:
            return X

        H.add_edge(u, v)
        D.add_edge(u, v)

    return None


def _is_sparse_for_l_less_than_k(G, K, L):
    X = _check_sparsity_for_l_less_than_k(G, K, L)
    if X is None:
        return True
    return False


def _is_sparse_for_l_between_k_and_2k(G, K, L):
    X = _check_sparsity_for_l_between_k_and_2k(G, K, L)
    if X is None:
        return True
    return False


def _is_sparse_for_l_equal_2k(G, K, L):
    X = _check_sparsity_for_l_equal_2k(G, K, L)
    if X is None:
        return True
    return False


def _is_sparse_for_l_between_2k_and_3k(G, K, L):
    X = _check_sparsity_for_l_between_2k_and_3k(G, K, L)
    if X is None:
        return True
    return False
