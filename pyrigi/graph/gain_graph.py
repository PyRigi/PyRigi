"""
Module for gain graphs with gains in Z^d.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from operator import index
from typing import TypeAlias

import networkx as nx

from pyrigi._utils._doc import doc_category
from pyrigi.data_type import Vertex

Gain: TypeAlias = tuple[int, ...]
GainEdge: TypeAlias = tuple[Vertex, Vertex, Gain]


def _is_Zd_gain_graph_data(
    incoming_graph_data: Iterable[GainEdge] | nx.Graph | None,
) -> bool:
    """
    Return whether the input has the structure of gain-graph data.

    Parameters
    ----------
    incoming_graph_data:
        The input data to be checked. This can either be a list of
        gain edges, a networkx graph, or ``None``.
    """
    if incoming_graph_data is None:
        return True

    if isinstance(incoming_graph_data, nx.Graph):
        edges = list(incoming_graph_data.edges(data=True))
        return (
            all("gain" in data for _, _, data in edges)
            if edges
            else hasattr(incoming_graph_data, "dim")
        )

    return all(
        isinstance(edge, (tuple, list))
        and ((len(edge) == 3 and isinstance(edge[2], (tuple, list))) or len(edge) == 2)
        for edge in incoming_graph_data
    )


def _negate_gain(gain: Gain) -> Gain:
    """
    Return the inverse of a gain.
    """
    if isinstance(gain, tuple):
        return tuple(-x for x in gain)
    else:
        raise TypeError("A gain must be a tuple of integers.")


class GainGraph(nx.MultiGraph):
    r"""
    Class representing an undirected gain graph.

    Each edge is represented by an oriented triple ``(u, v, gain)``.

    Definitions
    -----------
    :prf:ref:`:math:`gain graph <def-gain-graph>`

    This class inherits :class:`networkx.MultiGraph`.
    """

    def __new__(
        cls,
        incoming_graph_data: Iterable[GainEdge] | nx.Graph | None = None,
        **kwargs,  # noqa: U100
    ):
        """
        A distributor method that returns a GainGraphZd if the incoming graph
        data has the appropriate format.
        """
        if issubclass(cls, GainGraph) and _is_Zd_gain_graph_data(incoming_graph_data):
            return super().__new__(GainGraphZd)
        else:
            raise TypeError(
                "The incoming graph data is not in the correct format. "
                + "Currently, only Z^d-gain graphs are supported."
            )

    def __init__(
        self, incoming_graph_data: Iterable[GainEdge] | nx.Graph | None = None
    ) -> None:
        """
        Initialize a generic gain graph.
        """
        nx.MultiGraph.__init__(self)

        # Initialize common graph data.
        if incoming_graph_data is not None:
            self.add_edges(incoming_graph_data)

    @classmethod
    @doc_category("Class methods")
    def from_vertices(
        cls,
        vertices: Sequence[Vertex],
    ) -> GainGraph:
        """
        Create a gain graph with no edges.

        Parameters
        ----------
        vertices:
            A sequence of vertices.
        """
        graph = cls()
        graph.add_vertices(vertices)
        return graph

    @classmethod
    @doc_category("Class methods")
    def from_vertices_and_edges(
        cls,
        vertices: Sequence[Vertex],
        edges: Iterable[GainEdge],
    ) -> GainGraph:
        """
        Create a gain graph from a list of vertices and gain edges.

        All endpoints must belong to the specified vertex set.

        Parameters
        ----------
        vertices:
            A sequence of vertices.
        edges:
            An iterable of gain edges.
        """
        graph = cls()
        graph.add_vertices(vertices)

        for edge in edges:
            if not isinstance(edge, (tuple, list)) or len(edge) != 3:
                raise TypeError("Each gain edge must be a triple (u, v, gain).")

            u, v, gain = edge

            if not graph.has_node(u) or not graph.has_node(v):
                raise ValueError(f"Both endpoints of edge {edge!r} must be vertices.")

            graph.add_edge(u, v, gain=gain)

        return graph

    @doc_category("Graph manipulation")
    def add_vertex(self, vertex: Vertex) -> None:
        """
        Add a single vertex to the gain graph.

        Parameters
        ----------
        vertex:
            The vertex to be added.
        """
        self.add_node(vertex)

    @doc_category("Graph manipulation")
    def add_vertices(self, vertices: Sequence[Vertex]) -> None:
        """
        Add multiple vertices to the gain graph.

        Parameters
        ----------
        vertices:
            A sequence of vertices to be added.
        """
        self.add_nodes_from(vertices)

    @doc_category("Graph manipulation")
    def delete_vertex(self, vertex: Vertex) -> None:
        """
        Delete a single vertex from the gain graph.

        Parameters
        ----------
        vertex:
            The vertex to be removed.
        """
        self.remove_node(vertex)

    @doc_category("Graph manipulation")
    def delete_vertices(self, vertices: Sequence[Vertex]) -> None:
        """
        Delete multiple vertices from the gain graph.

        Parameters
        ----------
        vertices:
            A sequence of vertices to be removed.
        """
        self.remove_nodes_from(vertices)

    @doc_category("Attribute getters")
    def vertex_list(self) -> list[Vertex]:
        """
        Return the list of vertices in the gain graph.
        """
        try:
            return sorted(self.nodes)
        except BaseException:
            return list(self.nodes)

    @doc_category("Graph manipulation")
    def delete_edge(self, edge: tuple) -> None:
        """
        Delete an edge.

        Parameter
        ---------
        edge:
            This can be either an endpoint pair ``(u, v)`` or a gain
            edge ``(u, v, gain)``. For a pair, one such edge is deleted.
            For a triple, the first matching gain edge is removed unless
            a key is specified.
        """
        if not isinstance(edge, (tuple, list)):
            raise TypeError("An edge must be a pair or a gain-edge triple.")

        if len(edge) == 2:
            self.remove_edge(edge[0], edge[1])
            return

        if len(edge) != 3:
            raise TypeError("An edge must be a pair or a gain-edge triple.")

        u, v, gain = edge
        target_gain = gain
        edge_data = self.get_edge_data(u, v, default={})

        for edge_key, data in edge_data.items():
            if data["tail"] == u and data["head"] == v:
                candidate_gain = data["gain"]
            else:
                candidate_gain = _negate_gain(data["gain"])

            if u == v:
                matches = min(candidate_gain, _negate_gain(candidate_gain)) == min(
                    target_gain, _negate_gain(target_gain)
                )
            else:
                matches = candidate_gain == target_gain

            if matches:
                self.remove_edge(u, v, key=edge_key)
                return

        raise ValueError(f"Gain edge {edge!r} is not contained in the graph.")

    @doc_category("Graph manipulation")
    def delete_edges(self, edges: Iterable[tuple]) -> None:
        """
        Delete multiple gain edges.
        """
        for edge in edges:
            self.delete_edge(edge)

    @doc_category("Graph manipulation")
    def delete_loops(self) -> None:
        """
        Remove all loops.
        """
        for u, v in list(self.edges()):
            if u == v:
                self.remove_edge(u, v)

    @doc_category("Attribute getters")
    def edge_list(self) -> list[tuple]:
        """
        Return the list of gain edges.

        Each edge is returned as ``(u, v, gain)``.
        """
        edges = []

        for _, _, data in self.edges(data=True):
            edge = (data["tail"], data["head"], data["gain"])
            edges.append(edge)

        try:
            return sorted(edges)
        except TypeError:
            return edges

    @doc_category("Attribute getters")
    def gains_between(self, u: Vertex, v: Vertex) -> list[Gain]:
        """
        Return all gains of edges oriented from ``u`` to ``v``.

        If the orientation in the GainGraph is reversed, return the
        negated gain instead.
        """
        edge_data = self.get_edge_data(u, v, default={})
        result = []

        for data in edge_data.values():
            if data["tail"] == u and data["head"] == v:
                result.append(data["gain"])
            else:
                result.append(_negate_gain(data["gain"]))

        return result

    def _is_gain_edge(self, gain: Gain) -> bool:
        """
        Return whether a gain is a gain edge of the correct format.

        Notes
        -----
        This currently only checks whether the gain is a tuple of integers.
        It does not check whether the gain has the correct dimension. #
        This is done in :meth:`GainGraphZd._validate_gain`.
        """
        return not isinstance(gain, tuple) or not all(isinstance(x, int) for x in gain)

    def _validate_gain(self, gain: Gain) -> Gain:
        """
        Validate a gain.
        """
        if self._is_gain_edge(gain):
            raise TypeError(f"Gain {gain!r} is not of type {Gain!r}.")
        return gain

    def add_edge(self, u_for_edge: Vertex, v_for_edge: Vertex, gain: Gain = None):
        r"""
        Add an edge with a specified gain.

        If the edge is already contained in the GainGraph, nothing happens.

        Parameters
        ----------
        u_for_edge, v_for_edge:
            The endpoints, in the chosen orientation.
        gain:
            The gain vector in :math:`\mathbb{Z}^d`.
        """
        gain = self._validate_gain(gain)
        edge_data = self.get_edge_data(u_for_edge, v_for_edge, default={})

        if (u_for_edge, v_for_edge, gain) in self.edge_list() or (
            v_for_edge,
            u_for_edge,
            _negate_gain(gain),
        ) in self.edge_list():
            return

        edge_data = dict()
        edge_data.update(gain=gain, tail=u_for_edge, head=v_for_edge)

        super().add_edge(u_for_edge, v_for_edge, **edge_data)

    @doc_category("Graph manipulation")
    def add_edges(self, edges: Iterable[GainEdge]) -> None:
        """
        Add multiple edges.

        Accepts edges triples ``(u, v, gain)``.

        Parameters
        ----------
        edges:
            An iterable of gain edges.
        """
        for edge in edges:
            if not isinstance(edge, (tuple, list)):
                raise nx.NetworkXError(f"Edge {edge!r} must be a tuple or list.")

            if len(edge) == 3:
                u, v, gain = edge
            else:
                raise nx.NetworkXError(f"Edge {edge!r} must have 3 elements.")

            self.add_edge(u, v, gain)

    def __str__(self) -> str:
        return (
            f"GainGraph with vertex orbits "
            f"{self.vertex_list()} "
            f"and gain edges {self.edge_list()}"
        )

    def __repr__(self) -> str:
        return (
            f"GainGraph.from_vertices_and_edges("
            f"{self.vertex_list()!r}, {self.edge_list()!r}"
            f")"
        )

    def __eq__(self, other: object) -> bool:
        """
        Return whether two gain graphs are equal.

        Edge attributes are ignored. Reversing an edge
        while negating its gain does not change the gain graph.
        """
        if not isinstance(other, GainGraph):
            return False

        vertex_order = {vertex: i for i, vertex in enumerate(self.vertex_list())}

        def edge_multiset(graph: GainGraph) -> Counter:
            result = Counter()

            for tail, head, gain in graph.edge_list():
                if tail == head:
                    normalized_gain = min(gain, _negate_gain(gain))
                elif vertex_order[tail] < vertex_order[head]:
                    normalized_gain = gain
                else:
                    normalized_gain = _negate_gain(gain)

                result[(frozenset((tail, head)), normalized_gain)] += 1

            return result

        return edge_multiset(self) == edge_multiset(other)

    def __add__(self, other: GainGraph) -> GainGraph:
        """
        Return the multiset union of two gain graphs.
        """
        if not isinstance(other, GainGraph):
            raise AttributeError("The other graph must be a gain graph.")

        graph = GainGraph()
        graph.graph.update(self.graph)
        graph.graph.update(other.graph)

        graph.add_nodes_from(self.nodes(data=True))
        graph.add_nodes_from(other.nodes(data=True))

        for u, v, gain in self.edge_list():
            graph.add_edge(u, v, gain)

        for u, v, gain in other.edge_list():
            graph.add_edge(u, v, gain)

        return graph

    def copy(self, as_view: bool = False) -> GainGraph:
        """
        Create a copy of the gain graph.
        """
        if as_view:
            return nx.graphviews.generic_graph_view(self)

        graph = GainGraph(None)
        graph.graph.update(self.graph)

        graph.add_nodes_from((v, dict(data)) for v, data in self.nodes(data=True))

        for u, v, gain in self.edge_list():
            graph.add_edge(u, v, gain)

        return graph


class GainGraphZd(GainGraph):
    r"""
    Class representing an undirected gain graph with gains in :math:`\mathbb{Z}^d`.

    Each edge is represented by an oriented triple ``(u, v, gain)``,
    where ``gain`` is a vector of integers. Reversing the orientation
    changes the sign of the gain.

    The dimension can be specified explicitly using ``dim`` or inferred
    from the first edge that is added. If there are no edges and the
    ``dim`` is ``None``, the dimension defaults to 2.

    Definitions
    -----------
    :prf:ref:`:math:`\mathbb{Z}^d`-gain graph <def-gain-graph>`

    Examples
    --------
    >>> from pyrigi import GainGraph

    A gain graph with two parallel edges and one loop:

    >>> G = GainGraph([
    ...     (0, 1, (1, 0)),
    ...     (1, 1, (1, -1))
    ... ])
    >>> print(G)
    Z^2-GainGraph with vertex orbits [0, 1] and gain edges
    [(0, 1, (1, 0)), (1, 1, (1, -1))]

    Gain vectors must have the same dimension:

    >>> G.dim
    2

    The reverse orientation has the opposite gain:

    >>> G.gains_between(1, 0)
    [(-1, 0)]

    Notes
    -----
    This class inherits :class:`GainGraph`.
    """

    def __init__(
        self,
        incoming_graph_data: Iterable[GainEdge] | nx.Graph | None = None,
        dim: int | None = None,
    ) -> None:
        nx.MultiGraph.__init__(self)

        self._dim = self._validate_dim(dim)

        if isinstance(incoming_graph_data, nx.Graph):
            if self._dim is None:
                self._dim = self._validate_dim(
                    getattr(incoming_graph_data, "dim", None)
                )

            self.graph.update(incoming_graph_data.graph)
            self.add_nodes_from(incoming_graph_data.nodes(data=True))
            for u, v, data in incoming_graph_data.edges(data=True):
                if (not isinstance(data, dict)) or ("gain" not in data):
                    raise TypeError(
                        "The incoming networkx edge data is not in the correct format. "
                        + "It is required to be a `dict` and every edge needs to contain "
                        + "the label 'gain'."
                    )
                self.add_edge(u, v, data["gain"])
        else:
            if incoming_graph_data is not None:
                self.add_edges(incoming_graph_data)

    @staticmethod
    def _validate_dim(dim: int | None) -> int | None:
        """
        Validate the dimension. The default value is 2."""
        if dim is None:
            return None

        if isinstance(dim, bool):
            raise TypeError("The dimension must be a positive integer.")

        try:
            dim = index(dim)
        except TypeError as error:
            raise TypeError("The dimension must be a positive integer.") from error

        if dim <= 0:
            raise ValueError("The dimension must be positive.")

        return dim

    @property
    def dim(self) -> int | None:
        """
        Return the dimension of the gain vectors.
        """
        return self._dim

    def set_dim(self, dim: int | None) -> None:
        """
        Set the dimension of the gain graph.

        If the gain edges don't have the correct dimension,
        they are padded with zeros.
        """
        self._dim = self._validate_dim(dim)
        for edge in self.edge_list():
            if self._dim is None:
                self._dim = len(edge[2])
            if len(edge[2]) != self._dim:
                _edge = tuple(
                    [
                        edge[0],
                        edge[1],
                        tuple(list(edge[2]) + [0] * (self._dim - len(edge[2]))),
                    ]
                )
                self.delete_edge(edge)
                self.add_edge(*_edge)

    def _validate_gain(self, gain: Sequence[int]) -> Gain:
        """
        Validate and normalize a gain vector.

        Fixes the correct orientation and data types
        of the gain vector and checks that it has the correct dimension.

        Notes
        -----
        See also :meth:`GainGraph._validate_gain` for details.
        """
        if isinstance(gain, (str, bytes)):
            raise TypeError("A gain must be a sequence of integers.")

        try:
            raw_gain = tuple(gain)
        except TypeError as error:
            raise TypeError("A gain must be a sequence of integers.") from error

        if not raw_gain:
            raise ValueError("A gain vector must have positive length.")

        try:
            if any(isinstance(x, bool) for x in raw_gain):
                raise TypeError
            normalized_gain = tuple(index(x) for x in raw_gain)
        except TypeError as error:
            raise TypeError("Every gain coordinate must be an integer.") from error

        if self._dim is None:
            self._dim = len(normalized_gain)
        elif len(normalized_gain) != self._dim:
            raise ValueError(
                f"The gain {normalized_gain} has dimension "
                f"{len(normalized_gain)}, but the graph has "
                f"dimension {self._dim}."
            )

        return super()._validate_gain(normalized_gain)

    # ------------------------------------------------------------------
    # Graph construction and manipulation
    # ------------------------------------------------------------------

    def add_edge(
        self, u_for_edge: Vertex, v_for_edge: Vertex, gain: Sequence[int] = None
    ):
        r"""
        Add an edge with a specified gain.

        If the edge is already contained in the GainGraph, nothing happens.

        Parameters
        ----------
        u_for_edge, v_for_edge:
            The endpoints, in the chosen orientation.
        gain:
            The gain vector in :math:`\mathbb{Z}^d`.

        Examples
        --------
        >>> G = GainGraph(dim=2)
        >>> G.add_edge(0, 1, gain=(1, 0))
        >>> G.add_edge(0, 1, gain=(0, 1))
        >>> G.number_of_edges()
        2

        Notes
        -----
        See also :meth:`GainGraph.add_edge` for details.
        """
        if gain is None and self.dim is None:
            raise AttributeError(
                "The gain and the dimension cannot simultaneously be None."
            )
        if gain is None:
            gain = [0 for _ in range(self.dim)] if self.dim else [0 for _ in range(2)]
        elif self.dim is not None and len(gain) != self.dim:
            raise TypeError("Every edge must have a gain of the correct length.")

        gain = self._validate_gain(gain)
        if self.dim is None:
            self.dim = len(gain)

        super().add_edge(u_for_edge, v_for_edge, gain)

    @doc_category("Graph manipulation")
    def add_edges(self, edges: Iterable[GainEdge]) -> None:
        """
        Add multiple edges.

        Accepts edges ``(u,v)`` as well as triples ``(u, v, gain)``.
        In the former case, the gain is set to the zero vector.

        Parameters
        ----------
        edges:
            An iterable of edges.
        """
        for edge in edges:
            if not isinstance(edge, (tuple, list)):
                raise nx.NetworkXError(f"Edge {edge!r} must be a tuple or list.")

            if len(edge) == 2:
                u, v = edge
                gain = (
                    [0 for _ in range(self.dim)] if self.dim else [0 for _ in range(2)]
                )

            elif len(edge) == 3:
                u, v, gain = edge

            else:
                raise nx.NetworkXError(f"Edge {edge!r} must have 2 or 3 elements.")

            self.add_edge(u, v, gain)

    @classmethod
    @doc_category("Class methods")
    def from_vertices_and_edges(
        cls,
        vertices: Sequence[Vertex],
        edges: Iterable[GainEdge],
        dim: int | None = None,
    ) -> GainGraphZd:
        """
        Create a Z^d-gain graph from a list of vertices and gain edges.

        All endpoints must belong to the specified vertex set.

        Parameters
        ----------
        vertices:
            A sequence of vertices.
        edges:
            An iterable of gain edges.
        dim:
            The dimension of the lattice. Can be None.

        Notes
        -----
        See also :meth:`GainGraph.from_vertices_and_edges` for details.
        """
        graph = super().from_vertices_and_edges(vertices, edges)
        graph.set_dim(dim)
        return graph

    @classmethod
    @doc_category("Class methods")
    def from_vertices(
        cls,
        vertices: Sequence[Vertex],
        dim: int | None = None,
    ) -> GainGraphZd:
        """
        Create a Z^d-gain graph with no edges.

        Parameters
        ----------
        vertices:
            A sequence of vertices.
        dim:
            The dimension of the lattice. Can be None.

        Notes
        -----
        See also :meth:`GainGraph.from_vertices` for details.
        """
        graph = super().from_vertices(vertices)
        graph.set_dim(dim)
        return graph

    # ------------------------------------------------------------------
    # General functionality
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        d = self.dim if self.dim is not None else "d"
        return (
            f"Z^{d}-GainGraph with vertex orbits "
            f"{self.vertex_list()} "
            f"and gain edges {self.edge_list()}"
        )

    def __repr__(self) -> str:
        return (
            f"GainGraphZd.from_vertices_and_edges("
            f"{self.vertex_list()!r}, {self.edge_list()!r}, "
            f"dim={self.dim!r})"
        )

    def __eq__(self, other: object) -> bool:
        """
        Return whether two Z^d-gain graphs are equal.

        See also :meth:`GainGraph.__eq__` for details.
        """
        if not isinstance(other, GainGraphZd):
            return False

        if self.dim != other.dim or set(self.nodes) != set(other.nodes):
            return False

        return super().__eq__(other)

    def __add__(self, other: GainGraphZd) -> GainGraphZd:
        """
        Return the multiset union of two gain graphs.

        Notes
        -----
        See also :meth:`GainGraph.__add__` for details.
        """
        if not isinstance(other, GainGraphZd):
            raise AttributeError("The other graph must be a Z^d-gain graph.")

        if self.dim is not None and other.dim is not None and self.dim != other.dim:
            raise ValueError("Gain graphs must have the same dimension.")

        graph = super().__add__(other)
        graph.set_dim(self.dim or other.dim)

        return graph

    def copy(self, as_view: bool = False) -> GainGraphZd:
        """
        Create a copy of the Z^d-gain graph.

        Notes
        -----
        See also :meth:`GainGraph.copy` for details.
        """
        graph = super().copy(as_view=as_view)

        if not as_view:
            graph.set_dim(self.dim)

        return graph
