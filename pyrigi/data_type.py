"""

Module for defining data type used for type hinting.

"""

import sympy as sp
from typing import Hashable
from collections.abc import Sequence


Vertex = Hashable
"""
Any hashable type can be used for a :obj:`pyrigi.data_type.Vertex`.
"""

Edge = set[Vertex] | tuple[Vertex, Vertex] | list[Vertex]
"""
An unordered pair of :obj:`Vertices <pyrigi.data_type.Vertex>`.
"""

DirectedEdge = tuple[Vertex, Vertex] | list[Vertex]
"""
An ordered pair of :obj:`Vertices <pyrigi.data_type.Vertex>`.
"""

Number = int | float | str
"""
An integer, float or a string interpretable by :func:`~sympy.core.sympify.sympify`.
"""

Point = Sequence[Number]
"""
A :obj:`~collections.abc.Sequence` of :obj:`Numbers <pyrigi.data_type.Number>`
whose length is the dimension of its affine space.
"""

InfFlex = Sequence[Number] | dict[Vertex, Sequence[Number]]
"""
Given a framework in dimension `dim` with `n` vertices, an infinitesimal flex is either
given by a :obj:`~collections.abc.Sequence` of :obj:`Numbers <pyrigi.data_type.Number>`
whose length is `dim*n` or by a dictionary from the set of vertices
to a :obj:`~collections.abc.Sequence` of :obj:`Numbers <pyrigi.data_type.Number>`
of length `dim`.
"""

Stress = Sequence[Number] | dict[Edge, Number]
"""
Given a framework in dimension with `m` edges, an equilibrium stress is either
given by a :obj:`~collections.abc.Sequence` of :obj:`Numbers <pyrigi.data_type.Number>`
whose length is `m` or by a dictionary from the set of edges
to a :obj:`~collections.abc.Sequence` of :obj:`Numbers <pyrigi.data_type.Number>`.
"""

Inf = sp.core.numbers.Infinity
"""
Provides a data type that can become infinite.
"""
