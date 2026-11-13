"""
Module for periodic frameworks represented by gain graphs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence, Callable
from copy import deepcopy
from operator import index
import warnings

import numpy as np
import sympy as sp
from sympy import Matrix
from sympy.matrices.matrixbase import MatrixBase

from pyrigi._utils._doc import doc_category
from pyrigi.data_type import Number, Point, Vertex
from pyrigi.framework.base import FrameworkBase
from pyrigi.graph import GainGraph
from pyrigi.warning import OutsideFundamentalCellWarning


class PeriodicFramework(FrameworkBase):
    r"""
    A periodic bar-and-joint framework represented by a gain graph.

    A periodic framework consists of a :class:`~pyrigi.graph.GainGraph`, a
    realization of its vertex orbits in :math:`\\mathbb R^d`, and a lattice
    basis. A gain edge ``(u, v, gain)`` represents the bar joining ``p(u)`` to
    ``p(v) + B * gain``.

    The lattice basis is stored as a matrix whose columns are the lattice
    basis vectors. For convenience, it can also be supplied as a sequence of
    basis vectors; in that case each item of the sequence is interpreted as
    one basis vector.

    Parameters
    ----------
    graph:
        The underlying :class:`~pyrigi.graph.GainGraph`. Its gain dimension
        must equal the number of lattice basis vectors.
    realization:
        A dictionary mapping every vertex orbit to a point in the ambient
        Euclidean space. The ambient dimension is determined by these points.
        The realization is normalized to the fundamental parallelipiped.
    lattice_basis:
        A matrix with lattice vectors as columns, or a sequence of lattice
        basis vectors. The number of rows is the ambient dimension. The
        columns must be linearly independent.

    Notes
    -----
    The ambient dimension is available through :attr:`dim`; the rank of the
    period lattice is available through :attr:`lattice_dim`. These dimensions
    need not be equal.

    Examples
    --------
    >>> from pyrigi import GainGraph, PeriodicFramework
    >>> G = GainGraph([(0, 1, (1, 0)), (0, 1, (0, 1))])
    >>> F = PeriodicFramework(
    ...     G, {0: (0, 0), 1: (1, 0)}, [(2, 0), (0, 2)]
    ... )
    >>> F.dim, F.lattice_dim
    (2, 2)
    >>> F.edge_lengths()
    {(0, 1, (0, 1)): sqrt(5), (0, 1, (1, 0)): 3}
    """

    silence_fundamental_cell_warns = False

    def __init__(
        self,
        graph: GainGraph,
        realization: dict[Vertex, Point],
        lattice_basis: Sequence[Point] | MatrixBase,
    ) -> None:
        if not isinstance(graph, GainGraph):
            raise TypeError("The graph has to be an instance of class GainGraph.")
        if not isinstance(realization, dict):
            raise TypeError("The realization has to be a dictionary.")

        basis = self._lattice_basis_to_matrix(lattice_basis)
        if basis.rows <= 0 or basis.cols <= 0:
            raise ValueError("The lattice basis must have positive dimensions.")
        if basis.rank() != basis.cols:
            raise ValueError("The lattice basis vectors must be linearly independent.")

        if graph.dim is not None and graph.dim != basis.cols:
            raise ValueError(
                "The number of lattice basis vectors must equal the "
                f"dimension of the graph's periodicity ({graph.dim})."
            )

        for u, v, gain in graph.edge_list():
            if u == v and not any(gain):
                raise ValueError("A gain-graph loop must have nonzero gain.")

        # An edgeless GainGraph may not have an inferred gain dimension yet.
        # The lattice basis supplies that missing dimension.
        if graph.dim is None:
            graph_copy = GainGraph(dim=basis.cols)
            graph_copy.graph.update(deepcopy(graph.graph))
            graph_copy.add_nodes_from(deepcopy(list(graph.nodes(data=True))))
            self._graph = graph_copy
        else:
            self._graph = graph.copy()

        ambient_dim = basis.rows
        if realization:
            try:
                ambient_dim = len(next(iter(realization.values())))
            except TypeError as error:
                raise TypeError("Every realization value must be a point.") from error
            if ambient_dim != basis.rows:
                raise ValueError(
                    "The lattice basis must have one row for each ambient "
                    f"coordinate ({ambient_dim})."
                )

        self._dim = ambient_dim
        self._lattice_basis = basis
        self._realization: dict[Vertex, Matrix] = {}
        self.set_realization(realization)

    @staticmethod
    def _lattice_basis_to_matrix(
        lattice_basis: Sequence[Point] | MatrixBase,
    ) -> Matrix:
        """Return a matrix with the lattice vectors as columns."""
        if isinstance(lattice_basis, MatrixBase):
            return Matrix(lattice_basis)

        try:
            basis_vectors = list(lattice_basis)
        except TypeError as error:
            raise TypeError(
                "The lattice basis must be a matrix or a sequence of vectors."
            ) from error

        if not basis_vectors:
            raise ValueError("The lattice basis must contain at least one vector.")

        columns = []
        for vector in basis_vectors:
            try:
                column = Matrix(vector)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "Every lattice basis element must be a vector."
                ) from error
            if column.cols != 1:
                if column.rows == 1:
                    column = column.T
                else:
                    raise ValueError(
                        "Every lattice basis element must be one-dimensional."
                    )
            columns.append(column)

        try:
            return Matrix.hstack(*columns)
        except (ValueError, TypeError) as error:
            raise ValueError(
                "All lattice basis vectors must have the same dimension."
            ) from error

    @staticmethod
    def _normalize_gain(gain: Sequence[int], lattice_dim: int) -> tuple[int, ...]:
        """
        Validate and normalize a gain vector according to the given `lattice_dim`.

        Parameters
        ----------
        gain:
            A Z^d-gain vector given as a sequence of integers.
        lattice_dim:
            Rank of the underlying lattice.
        """
        if isinstance(gain, (str, bytes)):
            raise TypeError("A gain must be a sequence of integers.")
        try:
            raw_gain = tuple(gain)
        except TypeError as error:
            raise TypeError("A gain must be a sequence of integers.") from error

        try:
            if any(isinstance(coordinate, bool) for coordinate in raw_gain):
                raise TypeError
            normalized_gain = tuple(index(coordinate) for coordinate in raw_gain)
        except TypeError as error:
            raise TypeError("Every gain coordinate must be an integer.") from error

        if len(normalized_gain) != lattice_dim:
            raise ValueError(
                f"The gain has dimension {len(normalized_gain)}, but the "
                f"lattice dimension is {lattice_dim}."
            )
        return normalized_gain

    # ------------------------------------------------------------------
    # String representations
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Return the string representation."""
        realization = ", ".join(
            f"{vertex}:{tuple(self._realization[vertex])}"
            for vertex in self._graph.nodes
        )
        return (
            f"PeriodicFramework in {self.dim}-dimensional space with a "
            f"rank-{self.lattice_dim} lattice consisting of:\n"
            f"{self._graph}\nRealization {{{realization}}}\n"
            f"Lattice basis (columns): {self._lattice_basis}"
        )

    def __repr__(self) -> str:
        """Return a representation of the periodic framework."""
        realization = {
            vertex: [str(coordinate) for coordinate in point]
            for vertex, point in self.realization(as_points=True).items()
        }
        basis_vectors = [
            [str(self._lattice_basis[row, column]) for row in range(self.dim)]
            for column in range(self.lattice_dim)
        ]
        return (
            f"PeriodicFramework({self.graph!r}, {realization!r}, " f"{basis_vectors!r})"
        )

    # ------------------------------------------------------------------
    # Attribute getters
    # ------------------------------------------------------------------

    @property
    def graph(self) -> GainGraph:
        """Return a copy of the underlying gain graph."""
        return deepcopy(self._graph)

    @property
    def lattice_dim(self) -> int:
        """Return the rank of the period lattice (the gain dimension)."""
        return self._graph.dim

    @property
    def lattice_basis(self) -> Matrix:
        """Return a copy of the lattice basis matrix (basis vectors in columns)."""
        return self._lattice_basis.copy()

    @lattice_basis.setter
    def lattice_basis(self, lattice_basis: Sequence[Point] | MatrixBase) -> None:
        self.set_lattice_basis(lattice_basis)

    # ------------------------------------------------------------------
    # Attribute setters
    # ------------------------------------------------------------------

    @doc_category("Framework manipulation")
    def set_lattice_basis(self, lattice_basis: Sequence[Point] | MatrixBase) -> None:
        r"""Change the lattice basis of the periodic framework.

        The new basis must have the same shape as the existing basis and its
        columns must be linearly independent.

        Parameters
        ----------
        lattice_basis:
            Basis of the new lattice.
        """
        basis = self._lattice_basis_to_matrix(lattice_basis)
        expected_shape = (self.dim, self.lattice_dim)
        if basis.shape != expected_shape:
            raise ValueError(
                f"The lattice basis must have shape {expected_shape}, "
                f"but the given basis has shape {basis.shape}."
            )
        if basis.rank() != basis.cols:
            raise ValueError("The lattice basis vectors must be linearly independent.")
        self._lattice_basis = basis

    @classmethod
    @doc_category("Class methods")
    def Empty(cls, lattice_basis: Sequence[Point] | MatrixBase) -> PeriodicFramework:
        """Generate an empty periodic framework with a given lattice."""
        basis = cls._lattice_basis_to_matrix(lattice_basis)
        graph = GainGraph(dim=basis.cols)
        return cls(graph, {}, basis)

    def copy(self) -> PeriodicFramework:
        """Return an independent copy of the periodic framework."""
        return type(self)(
            self.graph,
            self.realization(as_points=True),
            self.lattice_basis,
        )

    # ------------------------------------------------------------------
    # Framework manipulation
    # ------------------------------------------------------------------

    @doc_category("Framework manipulation")
    def set_realization(self, realization: dict[Vertex, Point]) -> None:
        r"""Change the realization of the periodic framework.

        Every point is normalized modulo the period lattice before being
        stored. The realization must contain exactly the vertices of the
        underlying gain graph, and all points must have the correct
        ambient dimension.

        Parameters
        ----------
        realization:
            New realization for the periodic framework.
        """
        if not isinstance(realization, dict):
            raise TypeError("The realization has to be a dictionary.")

        if len(realization) != self._graph.number_of_nodes():
            raise IndexError(
                "The realization must contain exactly the vertices of the graph."
            )

        self._warn_outside_fundamental_cell(realization, self.set_realization)

        new_realization = {}
        for vertex in self._graph.nodes:
            self._input_check_vertex_key(vertex, realization)
            new_realization[vertex] = self._normalize_point_mod_lattice(
                realization[vertex]
            )
            self._input_check_point_dimension(new_realization[vertex])

        self._realization = new_realization

    def add_vertex(self, point: Point, vertex: Vertex = None) -> None:
        r"""
        Add a vertex orbit with its representative point.

        A vertex is automatically normalized to the lie in the
        fundamental parallelepiped.

        Parameters
        ----------
        point:
            The position of the new vertex.
        vertex:
            The label of the new vertex.
            Default value: `None`.
        """
        if vertex is None:
            candidate = self._graph.number_of_nodes()
            while self._graph.has_node(candidate):
                candidate += 1
            vertex = candidate

        if self._graph.has_node(vertex):
            raise KeyError(f"Vertex {vertex} is already a vertex of the graph!")

        self._input_check_point_dimension(point)
        self._warn_outside_fundamental_cell({vertex: point}, self.add_vertex)

        normalized_point = self._normalize_point_mod_lattice(point)
        super().add_vertex(normalized_point, vertex)

    def set_vertex_pos(self, vertex: Vertex, point: Point) -> None:
        r"""Set the position of a vertex orbit.

        The point is normalized modulo the period lattice before it is
        stored, using the half-open fundamental parallelepiped determined
        by ``lattice_basis``.

        Parameters
        ----------
        vertex:
            The label of the vertex whose position is modified.
        point:
            The new position of this vertex.
        """
        self._input_check_vertex_key(vertex)
        self._input_check_point_dimension(point)
        self._warn_outside_fundamental_cell({vertex: point}, self.set_vertex_pos)
        normalized_point = self._normalize_point_mod_lattice(point)
        super().set_vertex_pos(vertex, normalized_point)

    def _normalize_point_mod_lattice(self, point: Point) -> Matrix:
        r"""Return the representative of ``point`` in a fundamental domain.

        The lattice basis may be a SymPy Matrix, NumPy array, list, or tuple.
        The basis vectors are assumed to be stored in the columns.

        The lattice coordinates are computed as

            c = (B.T * B)**(-1) * B.T * point.

        Subtracting ``B * floor(c)`` places the lattice coordinates
        in ``[0, 1)``. The component orthogonal to the lattice span
        remains unchanged.

        Parameters
        ----------
        point:
            Position of the vertex which is translated to the
            unit cell via the underlying lattice.
        """
        lattice_basis = sp.Matrix(self._lattice_basis)

        point_vector = sp.Matrix(point)
        if point_vector.cols != 1:
            if point_vector.rows == 1:
                point_vector = point_vector.T
            else:
                raise ValueError("A point must be a one-dimensional vector.")

        self._input_check_point_dimension(point_vector)

        gram_matrix = lattice_basis.T * lattice_basis
        lattice_coordinates = gram_matrix.inv() * lattice_basis.T * point_vector

        lattice_shift = sp.Matrix(
            [sp.floor(coordinate) for coordinate in lattice_coordinates]
        )

        return point_vector - lattice_basis * lattice_shift

    def _is_point_outside_lattice(self, point: Point) -> bool:
        r"""Check whether a point lies outside the fundamental domain.

        The fundamental domain is defined by lattice coordinates in
        the half-open interval [0, 1).

        The lattice basis may be a SymPy Matrix, NumPy array, list, or tuple.

        Parameters
        ----------
        point:
            Point for which the containment in the periodic unit cell is checked.
        """
        lattice_basis = sp.Matrix(self._lattice_basis)
        point_vector = sp.Matrix(point)

        if point_vector.cols != 1:
            if point_vector.rows == 1:
                point_vector = point_vector.T
            else:
                raise ValueError("A point must be a one-dimensional vector.")

        self._input_check_point_dimension(point_vector)

        gram_matrix = lattice_basis.T * lattice_basis
        lattice_coordinates = gram_matrix.inv() * lattice_basis.T * point_vector

        for coordinate in lattice_coordinates:
            lower_bound = sp.simplify(coordinate >= 0)
            upper_bound = sp.simplify(coordinate < 1)

            if lower_bound is False or lower_bound is sp.false:
                return True
            if upper_bound is False or upper_bound is sp.false:
                return True

            if lower_bound is not True and lower_bound is not sp.true:
                raise ValueError(
                    "Could not determine whether the point is inside the lattice."
                )
            if upper_bound is not True and upper_bound is not sp.true:
                raise ValueError(
                    "Could not determine whether the point is inside the lattice."
                )

        return False

    @doc_category("Framework manipulation")
    def add_edge(self, edge: tuple) -> None:
        r"""Add a gain edge to the framework.

        Parameters
        ----------
        edge:
            New edge. Can be an endpoint pair ``(u, v)`` (zero gain) or a gain
            edge triple ``(u, v, gain)`` with a gain in $Z^d$. The endpoints
            must already be vertices of the gain graph.
        """
        if not isinstance(edge, (tuple, list)) or len(edge) not in (2, 3):
            raise TypeError("An edge must be an endpoint pair or a gain-edge triple.")
        u, v = edge[:2]
        if not self._graph.has_node(u) or not self._graph.has_node(v):
            raise KeyError("Both endpoints of an edge must be vertices of the graph.")
        if len(edge) == 3:
            gain = self._normalize_gain(edge[2], self.lattice_dim)
        else:
            gain = (0,) * self.lattice_dim
        if u == v and not any(gain):
            raise ValueError("A gain-graph loop must have nonzero gain.")
        self._graph.add_edge(u, v, gain)

    @doc_category("Framework manipulation")
    def add_edges(self, edges: Iterable[tuple]) -> None:
        """Add multiple endpoint pairs or gain-edge triples."""
        for edge in edges:
            self.add_edge(edge)

    # ------------------------------------------------------------------
    # Periodic geometry
    # ------------------------------------------------------------------

    @doc_category("Other")
    def lattice_vector(self, gain: Sequence[int], numerical: bool = False) -> Matrix:
        r"""Return the Euclidean lattice translation represented by ``gain``.

        The returned vector is ``lattice_basis * gain``.

        Parameters
        ----------
        gain:
            Z^d-linear combination of the lattice basis.
        numerical:
            A boolean determining whether the output is
            converted to a floating point number or not.
        """
        normalized_gain = self._normalize_gain(gain, self.lattice_dim)
        vector = self._lattice_basis * Matrix(normalized_gain)
        return vector.evalf() if numerical else vector

    @doc_category("Other")
    def edge_vector(
        self, edge: tuple[Vertex, Vertex, Sequence[int]], numerical: bool = False
    ) -> Matrix:
        r"""Return the oriented bar vector of a gain edge.

        Parameters
        ----------
        edge:
            For an oriented gain edge ``(u, v, gain)``, this is
            ``p(v) + lattice_basis * gain - p(u)``. The reverse orientation gives
            the negative vector.
        numerical:
            A boolean determining whether the output is
            converted to a floating point number or not.
        """
        if not isinstance(edge, (tuple, list)) or len(edge) != 3:
            raise TypeError("An edge vector requires a gain-edge triple (u, v, gain).")
        u, v, gain = edge
        self._input_check_vertex_key(u)
        self._input_check_vertex_key(v)
        normalized_gain = self._normalize_gain(gain, self.lattice_dim)
        available_gains = self._graph.gains_between(u, v)
        is_present = normalized_gain in available_gains
        if u == v:
            reverse_gain = tuple(-coordinate for coordinate in normalized_gain)
            is_present = is_present or reverse_gain in available_gains
        if not is_present:
            raise ValueError(f"Gain edge {edge!r} is not contained in the graph.")
        vector = (
            self._realization[v]
            + self._lattice_basis * Matrix(normalized_gain)
            - self._realization[u]
        )
        return vector.evalf() if numerical else vector

    @doc_category("Other")
    def edge_vectors(self, numerical: bool = False) -> dict[tuple, Matrix]:
        """
        Return the oriented bar vectors of all gain edges.

        Parameters
        ----------
        numerical:
            A boolean determining whether the output is
            converted to a floating point number or not.
        """
        return {
            edge: self.edge_vector(edge, numerical=numerical)
            for edge in self._graph.edge_list()
        }

    @doc_category("Other")
    def edge_lengths(self, numerical: bool = False) -> dict[tuple, Number]:
        r"""Return the lengths of all bars, indexed by gain-edge triples.

        Parallel edges with different gains are distinct bars, so the keys of
        the returned dictionary are triples ``(u, v, gain)``.

        Parameters
        ----------
        numerical:
            A boolean determining whether the output is
            converted to a floating point number or not.
        """
        lengths = {}
        for edge, vector in self.edge_vectors(numerical=numerical).items():
            if numerical:
                lengths[edge] = float(
                    np.linalg.norm(np.array(vector, dtype=float).reshape(-1))
                )
            else:
                squared_length = sum(coordinate**2 for coordinate in vector)
                lengths[edge] = sp.sqrt(sp.simplify(squared_length))
        return lengths

    def _warn_outside_fundamental_cell(
        self, points: dict[Vertex, Point], method: Callable
    ) -> None:
        """
        Raise a warning if the PeriodicFramework contains vertex positions,
        which lie outside the fundamental parallelipiped.

        Parameters
        ----------
        points:
            A new realization for which the containment in the fundamental
            cell is checked.
        method:
            Reference to the method from which this warning is called.
        """
        cls = type(self)
        if not cls.silence_fundamental_cell_warns:
            affected_points = {}
            normalized_points = {}
            for v, pos in points.items():
                if self._is_point_outside_lattice(pos):
                    affected_points |= {v: pos}
                    normalized_points = {v: self._normalize_point_mod_lattice(pos)}
            if not len(affected_points.values()) == 0:
                warnings.warn(
                    OutsideFundamentalCellWarning(
                        affected_points,
                        normalized_points,
                        self.lattice_basis,
                        method,
                        class_off=cls,
                    ),
                    stacklevel=2,
                )
