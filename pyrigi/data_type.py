"""

Module for defining data type used for type hinting.

"""

from collections.abc import Sequence
from typing import Hashable, NamedTuple, TypeAlias

import sympy as sp
import networkx as nx


Vertex: TypeAlias = Hashable
"""
Any hashable type can be used for a :obj:`pyrigi.data_type.Vertex`.
"""

Edge: TypeAlias = set[Vertex] | tuple[Vertex, Vertex] | list[Vertex]
"""
An unordered pair of :obj:`Vertices <pyrigi.data_type.Vertex>`.
"""

DirectedEdge: TypeAlias = tuple[Vertex, Vertex] | list[Vertex]
"""
An ordered pair of :obj:`Vertices <pyrigi.data_type.Vertex>`.
"""

Number: TypeAlias = int | float | str
"""
An integer, float or a string interpretable by :func:`~sympy.core.sympify.sympify`.
"""

Point: TypeAlias = Sequence[Number]
"""
A :obj:`~collections.abc.Sequence` of :obj:`Numbers <pyrigi.data_type.Number>`
whose length is the dimension of its affine space.
"""

InfFlex: TypeAlias = Sequence[Number] | dict[Vertex, Sequence[Number]]
"""
Given a framework in dimension ``dim`` with ``n`` vertices,
an infinitesimal flex is either given by a :obj:`~collections.abc.Sequence`
of :obj:`Numbers <pyrigi.data_type.Number>`
whose length is ``dim*n`` or by a dictionary from the set of vertices
to a :obj:`~collections.abc.Sequence` of :obj:`Numbers <pyrigi.data_type.Number>`
of length ``dim``.
"""

Stress: TypeAlias = Sequence[Number] | dict[Edge, Number]
"""
Given a framework with ``m`` edges, an equilibrium stress is either
given by a :obj:`~collections.abc.Sequence` of :obj:`Numbers <pyrigi.data_type.Number>`
whose length is ``m`` or by a dictionary from the set of edges
to a :obj:`~collections.abc.Sequence` of :obj:`Numbers <pyrigi.data_type.Number>`.
"""

Inf: TypeAlias = sp.core.numbers.Infinity
"""
Provides a data type that can become infinite.
"""


class SeparatingCut(NamedTuple):
    """
    Represents a separating cut in a graph.

    Definitions
    -----------
    :prf:ref:`Separating set <def-separating-set>`

    Parameters
    ----------
    a:
        vertices in the first component **excluding** the cut vertices
    b:
        vertices in the second component **excluding** the cut vertices
    cut:
        vertices of the cut
    """

    a: set[Vertex]
    b: set[Vertex]
    cut: set[Vertex]

    def __repr__(self) -> str:
        return f"SeparatingCut({self.a}, {self.b} - {self.cut})"

    def __eq__(self, other) -> bool:
        if self.cut != other.cut:
            return False
        if self.a == other.a:
            return self.b == other.b
        return self.a == other.b and self.b == other.a


class StableSeparatingCut(SeparatingCut):
    """
    Represents a stable separating set in a graph.

    Definitions
    -----------
    :prf:ref:`Stable separating set <def-stable-separating-set>`

    Parameters
    ----------
    a:
        vertices in the first component **excluding** the cut vertices
    b:
        vertices in the second component **excluding** the cut vertices
    cut:
        vertices of the cut
    """

    def __repr__(self) -> str:
        return f"StableSeparatingCut({self.a}, {self.b} - {self.cut})"

    def validate(self, graph: nx.Graph) -> bool:
        """
        Checks if the this cut is a stable cut of the given graph

        Parameters
        ----------
        graph:
            The graph on which we check if the set is separating and stable.
        """
        from pyrigi._cuts import is_stable_separating_set

        return is_stable_separating_set(graph, self)
