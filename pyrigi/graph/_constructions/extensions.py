"""
This module provides algorithms related to k-extensions of graphs.
"""

import math
from copy import deepcopy
from itertools import combinations
from typing import Iterable

import networkx as nx

import pyrigi._utils._input_check as _input_check
import pyrigi.graph._utils._input_check as _graph_input_check
from pyrigi.data_type import Edge, Sequence, Vertex
from pyrigi.exception import NotSupportedValueError


def zero_extension(
    graph: nx.Graph,
    vertices: Sequence[Vertex],
    new_vertex: Vertex = None,
    dim: int = 2,
    inplace: bool = False,
) -> nx.Graph:
    """
    Return a ``dim``-dimensional 0-extension.

    Definitions
    -----------
    :prf:ref:`0-extension <def-k-extension>`

    Parameters
    ----------
    vertices:
        A new vertex is connected to these vertices.
        All the vertices must be contained in the graph
        and there must be ``dim`` of them.
    new_vertex:
        Newly added vertex is named according to this parameter.
        If ``None``, the name is set as the lowest possible integer value
        greater or equal than the number of nodes.
    dim:
        The dimension in which the 0-extension is created.
    inplace:
        If ``True``, the graph is modified,
        otherwise a new modified graph is created,
        while the original graph remains unchanged (default).

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.Complete(3)
    >>> print(G)
    Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
    >>> H = G.zero_extension([0, 2])
    >>> print(H)
    Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]]
    >>> H = G.zero_extension([0, 2], 5)
    >>> print(H)
    Graph with vertices [0, 1, 2, 5] and edges [[0, 1], [0, 2], [0, 5], [1, 2], [2, 5]]
    >>> print(G)
    Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
    >>> H = G.zero_extension([0, 1, 2], 5, dim=3, inplace=True)
    >>> print(H)
    Graph with vertices [0, 1, 2, 5] and edges [[0, 1], [0, 2], [0, 5], [1, 2], [1, 5], [2, 5]]
    >>> print(G)
    Graph with vertices [0, 1, 2, 5] and edges [[0, 1], [0, 2], [0, 5], [1, 2], [1, 5], [2, 5]]
    """  # noqa: E501
    return k_extension(graph, 0, vertices, [], new_vertex, dim, inplace)


def one_extension(
    graph: nx.Graph,
    vertices: Sequence[Vertex],
    edge: Edge,
    new_vertex: Vertex = None,
    dim: int = 2,
    inplace: bool = False,
) -> nx.Graph:
    """
    Return a ``dim``-dimensional 1-extension.

    Definitions
    -----------
    :prf:ref:`1-extension <def-k-extension>`

    Parameters
    ----------
    vertices:
        A new vertex is connected to these vertices.
        All the vertices must be contained in the graph
        and there must be ``dim + 1`` of them.
    edge:
        An edge with endvertices from the list ``vertices`` that is deleted.
        The edge must be contained in the graph.
    new_vertex:
        Newly added vertex is named according to this parameter.
        If ``None``, the name is set as the lowest possible integer value
        greater or equal than the number of nodes.
    dim:
        The dimension in which the 1-extension is created.
    inplace:
        If ``True``, the graph is modified,
        otherwise a new modified graph is created,
        while the original graph remains unchanged (default).

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.Complete(3)
    >>> print(G)
    Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
    >>> H = G.one_extension([0, 1, 2], [0, 1])
    >>> print(H)
    Graph with vertices [0, 1, 2, 3] and edges [[0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    >>> print(G)
    Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
    >>> G = graphs.ThreePrism()
    >>> print(G)
    Graph with vertices [0, 1, 2, 3, 4, 5] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]]
    >>> H = G.one_extension([0, 1], [0, 1], dim=1)
    >>> print(H)
    Graph with vertices [0, 1, 2, 3, 4, 5, 6] and edges [[0, 2], [0, 3], [0, 6], [1, 2], [1, 4], [1, 6], [2, 5], [3, 4], [3, 5], [4, 5]]
    >>> G = graphs.CompleteBipartite(3, 2)
    >>> print(G)
    Graph with vertices [0, 1, 2, 3, 4] and edges [[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]]
    >>> H = G.one_extension([0, 1, 2, 3, 4], [0, 3], dim=4, inplace = True)
    >>> print(H)
    Graph with vertices [0, 1, 2, 3, 4, 5] and edges [[0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 5], [4, 5]]
    >>> print(G)
    Graph with vertices [0, 1, 2, 3, 4, 5] and edges [[0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 5], [4, 5]]
    """  # noqa: E501
    return k_extension(graph, 1, vertices, [edge], new_vertex, dim, inplace)


def k_extension(
    graph: nx.Graph,
    k: int,
    vertices: Sequence[Vertex],
    edges: Sequence[Edge],
    new_vertex: Vertex = None,
    dim: int = 2,
    inplace: bool = False,
) -> nx.Graph:
    """
    Return a ``dim``-dimensional ``k``-extension.

    See also :meth:`.zero_extension` and :meth:`.one_extension`.

    Definitions
    -----------
    :prf:ref:`k-extension <def-k-extension>`

    Parameters
    ----------
    k
    vertices:
        A new vertex is connected to these vertices.
        All the vertices must be contained in the graph
        and there must be ``dim + k`` of them.
    edges:
        A list of edges that are deleted.
        The endvertices of all the edges must be contained
        in the list ``vertices``.
        The edges must be contained in the graph and there must be ``k`` of them.
    new_vertex:
        Newly added vertex is named according to this parameter.
        If ``None``, the name is set as the lowest possible integer value
        greater or equal than the number of nodes.
    dim:
        The dimension in which the ``k``-extension is created.
    inplace:
        If ``True``, the graph is modified,
        otherwise a new modified graph is created,
        while the original graph remains unchanged (default).

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.Complete(5)
    >>> print(G)
    Graph with vertices [0, 1, 2, 3, 4] and edges [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    >>> H = G.k_extension(2, [0, 1, 2, 3], [[0, 1], [0,2]])
    >>> print(H)
    Graph with vertices [0, 1, 2, 3, 4, 5] and edges [[0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5]]
    >>> G = graphs.Complete(5)
    >>> print(G)
    Graph with vertices [0, 1, 2, 3, 4] and edges [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    >>> H = G.k_extension(2, [0, 1, 2, 3, 4], [[0, 1], [0,2]], dim = 3)
    >>> print(H)
    Graph with vertices [0, 1, 2, 3, 4, 5] and edges [[0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]]
    >>> G = graphs.Path(6)
    >>> print(G)
    Graph with vertices [0, 1, 2, 3, 4, 5] and edges [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
    >>> H = G.k_extension(2, [0, 1, 2], [[0, 1], [1,2]], dim = 1, inplace = True);
    >>> print(H)
    Graph with vertices [0, 1, 2, 3, 4, 5, 6] and edges [[0, 6], [1, 6], [2, 3], [2, 6], [3, 4], [4, 5]]
    >>> print(G)
    Graph with vertices [0, 1, 2, 3, 4, 5, 6] and edges [[0, 6], [1, 6], [2, 3], [2, 6], [3, 4], [4, 5]]
    """  # noqa: E501
    _input_check.dimension(dim)
    _input_check.integrality_and_range(k, "k", min_val=0)
    _graph_input_check.no_loop(graph)
    _graph_input_check.vertex_members(graph, vertices, "'the vertices'")
    if len(set(vertices)) != dim + k:
        raise ValueError(f"List of vertices must contain {dim + k} distinct vertices!")
    _graph_input_check.is_edge_list(graph, edges, vertices)
    if len(edges) != k:
        raise ValueError(f"List of edges must contain {k} distinct edges!")
    for edge in edges:
        count = edges.count(list(edge)) + edges.count(list(edge)[::-1])
        count += edges.count(tuple(edge)) + edges.count(tuple(edge)[::-1])
        if count > 1:
            raise ValueError(
                "List of edges must contain distinct edges, "
                f"but {edge} appears {count} times!"
            )
    if new_vertex is None:
        candidate = graph.number_of_nodes()
        while graph.has_node(candidate):
            candidate += 1
        new_vertex = candidate
    if graph.has_node(new_vertex):
        raise ValueError(f"Vertex {new_vertex} is already a vertex of the graph!")
    G = graph
    if not inplace:
        G = deepcopy(graph)
    G.remove_edges_from(edges)
    for vertex in vertices:
        G.add_edge(vertex, new_vertex)
    return G


def all_k_extensions(
    graph: nx.Graph,
    k: int,
    dim: int = 2,
    only_non_isomorphic: bool = False,
) -> Iterable[nx.Graph]:
    """
    Return an iterator over all possible ``dim``-dimensional ``k``-extensions.

    Definitions
    -----------
    :prf:ref:`k-extension <def-k-extension>`

    Parameters
    ----------
    k:
    dim:
    only_non_isomorphic:
        If ``True``, only one graph per isomorphism class is included.

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.Complete(3)
    >>> type(G.all_k_extensions(0))
    <class 'generator'>
    >>> len(list(G.all_k_extensions(0)))
    3
    >>> len(list(G.all_k_extensions(0, only_non_isomorphic=True)))
    1

    >>> len(list(graphs.Diamond().all_k_extensions(1, 2, only_non_isomorphic=True)))
    2

    Notes
    -----
    It turns out that possible errors on bad input parameters are only raised,
    when the output iterator is actually used,
    not when it is created.
    """
    _input_check.dimension(dim)
    _graph_input_check.no_loop(graph)
    _input_check.integrality_and_range(k, "k", min_val=0)
    _input_check.greater_equal(
        graph.number_of_nodes(),
        dim + k,
        "number of vertices in the graph",
        "dim + k",
    )
    _input_check.greater_equal(
        graph.number_of_edges(), k, "number of edges in the graph", "k"
    )

    solutions = []
    for edges in combinations(graph.edges, k):
        s = set(graph.nodes)
        w = set()
        for edge in edges:
            s.discard(edge[0])
            s.discard(edge[1])
            w.add(edge[0])
            w.add(edge[1])
        if len(w) > (dim + k):
            break
        w = list(w)
        for vertices in combinations(s, dim + k - len(w)):
            current = k_extension(graph, k, list(vertices) + w, edges, dim=dim)
            if only_non_isomorphic:
                for other in solutions:
                    if nx.is_isomorphic(current, other):
                        break
                else:
                    solutions.append(current)
                    yield current
            else:
                yield current


def all_extensions(
    graph: nx.Graph,
    dim: int = 2,
    only_non_isomorphic: bool = False,
    k_min: int = 0,
    k_max: int | None = None,
) -> Iterable[nx.Graph]:
    """
    Return an iterator over all ``dim``-dimensional extensions.

    All possible ``k``-extensions for ``k`` such that
    ``k_min <= k <= k_max`` are considered.

    Definitions
    -----------
    :prf:ref:`k-extension <def-k-extension>`

    Parameters
    ----------
    dim:
    only_non_isomorphic:
        If ``True``, only one graph per isomorphism class is included.
    k_min:
        Minimal value of ``k`` for the ``k``-extensions (default 0).
    k_max:
        Maximal value of ``k`` for the ``k``-extensions (default ``dim - 1``).

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.Complete(3)
    >>> type(G.all_extensions())
    <class 'generator'>
    >>> len(list(G.all_extensions()))
    6
    >>> len(list(G.all_extensions(only_non_isomorphic=True)))
    1

    >>> list(graphs.Diamond().all_extensions(2, only_non_isomorphic=True, k_min=1, k_max=1)
    ... ) == list(graphs.Diamond().all_k_extensions(1, 2, only_non_isomorphic=True))
    True

    Notes
    -----
    It turns out that possible errors on bad input paramters are only raised,
    when the output iterator is actually used,
    not when it is created.
    """  # noqa: E501
    _input_check.dimension(dim)
    _graph_input_check.no_loop(graph)
    _input_check.integrality_and_range(k_min, "k_min", min_val=0)
    if k_max is None:
        k_max = dim - 1
    _input_check.integrality_and_range(k_max, "k_max", min_val=0)
    _input_check.greater_equal(k_max, k_min, "k_max", "k_min")

    extensions = []
    for k in range(k_min, k_max + 1):
        if graph.number_of_nodes() >= dim + k and graph.number_of_edges() >= k:
            extensions.extend(all_k_extensions(graph, k, dim, only_non_isomorphic))

    solutions = []
    for current in extensions:
        if only_non_isomorphic:
            for other in solutions:
                if nx.is_isomorphic(current, other):
                    break
            else:
                solutions.append(current)
                yield current
        else:
            yield current


def extension_sequence(  # noqa: C901
    graph: nx.Graph, dim: int = 2, return_type: str = "extensions"
) -> list[nx.Graph] | list | None:
    """
    Compute a sequence of ``dim``-dimensional extensions.

    The ``k``-extensions for ``k`` from 0 to ``2 * dim - 1``
    are considered.
    The sequence then starts from a complete graph on ``dim`` vertices.
    If no such sequence exists, ``None`` is returned.

    The method returns either a sequence of graphs,
    data on the extension, or both.

    Note that for dimensions larger than two, the
    extensions are not always preserving rigidity.

    Definitions
    -----------
    :prf:ref:`k-extension <def-k-extension>`

    Parameters
    ----------
    dim:
        The dimension in which the extensions are created.
    return_type:
        Can have values ``"graphs"``, ``"extensions"`` or ``"both"``.

        If ``"graphs"``, then the sequence of graphs obtained from the extensions
        is returned.

        If ``"extensions"``, then an initial graph and a sequence of extensions
        of the form ``[k, vertices, edges, new_vertex]`` as needed
        for the input of :meth:`.k_extension` is returned.

        If ``"both"``, then an initial graph and a sequence of pairs
        ``[graph, extension]``, where the latter has the form from above,
        is returned.

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.Complete(3)
    >>> print(G)
    Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
    >>> G.extension_sequence(return_type="graphs")
    [Graph.from_vertices_and_edges([1, 2], [(1, 2)]), Graph.from_vertices_and_edges([0, 1, 2], [(0, 1), (0, 2), (1, 2)])]
    >>> G = graphs.Diamond()
    >>> print(G)
    Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]]
    >>> G.extension_sequence(return_type="graphs")
    [Graph.from_vertices_and_edges([2, 3], [(2, 3)]), Graph.from_vertices_and_edges([0, 2, 3], [(0, 2), (0, 3), (2, 3)]), Graph.from_vertices_and_edges([0, 1, 2, 3], [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)])]
    >>> G.extension_sequence(return_type="extensions")
    [Graph.from_vertices_and_edges([2, 3], [(2, 3)]), [0, [3, 2], [], 0], [0, [0, 2], [], 1]]
    """  # noqa: E501
    _input_check.dimension(dim)
    _graph_input_check.no_loop(graph)

    if not graph.number_of_edges() == dim * graph.number_of_nodes() - math.comb(
        dim + 1, 2
    ):
        return None
    if graph.number_of_nodes() == dim:
        return [graph]
    degrees = sorted(graph.degree, key=lambda node: node[1])
    degrees = [deg for deg in degrees if deg[1] >= dim and deg[1] <= 2 * dim - 1]
    if len(degrees) == 0:
        return None

    for deg in degrees:
        if deg[1] == dim:
            G = deepcopy(graph)
            neighbors = list(graph.neighbors(deg[0]))
            G.remove_node(deg[0])
            branch = extension_sequence(G, dim, return_type)
            extension = [0, neighbors, [], deg[0]]
            if branch is not None:
                if return_type == "extensions":
                    return branch + [extension]
                elif return_type == "graphs":
                    return branch + [graph]
                elif return_type == "both":
                    return branch + [[graph, extension]]
                else:
                    raise NotSupportedValueError(
                        return_type, "return_type", extension_sequence
                    )
            return branch
        else:
            neighbors = list(graph.neighbors(deg[0]))
            G = deepcopy(graph)
            G.remove_node(deg[0])
            for k_possible_edges in combinations(
                combinations(neighbors, 2), deg[1] - dim
            ):
                if all([not G.has_edge(*edge) for edge in k_possible_edges]):
                    for edge in k_possible_edges:
                        G.add_edge(*edge)
                    branch = extension_sequence(G, dim, return_type)
                    if branch is not None:
                        extension = [
                            deg[1] - dim,
                            neighbors,
                            k_possible_edges,
                            deg[0],
                        ]
                        if return_type == "extensions":
                            return branch + [extension]
                        elif return_type == "graphs":
                            return branch + [graph]
                        elif return_type == "both":
                            return branch + [[graph, extension]]
                        else:
                            raise NotSupportedValueError(
                                return_type, "return_type", extension_sequence
                            )
                    for edge in k_possible_edges:
                        G.remove_edge(*edge)
    return None


def has_extension_sequence(
    graph: nx.Graph,
    dim: int = 2,
) -> bool:
    """
    Return if there exists a sequence of ``dim``-dimensional extensions.

    The method returns whether there exists a sequence of extensions
    as described in :meth:`extension_sequence`.

    Definitions
    -----------
    :prf:ref:`k-extension <def-k-extension>`

    Parameters
    ----------
    dim:
        The dimension in which the extensions are created.

    Examples
    --------
    >>> import pyrigi.graphDB as graphs
    >>> G = graphs.ThreePrism()
    >>> print(G)
    Graph with vertices [0, 1, 2, 3, 4, 5] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]]
    >>> G.has_extension_sequence()
    True
    >>> G = graphs.CompleteBipartite(1, 2)
    >>> print(G)
    Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2]]
    >>> G.has_extension_sequence()
    False
    """  # noqa: E501
    return extension_sequence(graph, dim) is not None
