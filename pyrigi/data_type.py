"""

Module for defining data type used for type hinting.

"""

from sympy import Matrix, MatrixBase
import sympy as sp
import numpy as np
from typing import Hashable
from collections.abc import Sequence


Vertex = Hashable
"""
Any hashable type can be used for a Vertex.
"""

Edge = set[Vertex] | tuple[Vertex, Vertex] | list[Vertex]
"""
An Edge is an unordered pair of :obj:`Vertices <pyrigi.data_type.Vertex>`.
"""

DirectedEdge = tuple[Vertex, Vertex] | list[Vertex]
"""
A DirectedEdge is an ordered pair of :obj:`Vertices <pyrigi.data_type.Vertex>`.
"""

Number = int | float | str
"""
An integer, float or a string interpretable by :func:`~sympy.core.sympify.sympify`.
"""

Point = Sequence[Number]
"""
A Point is a Sequence of Coordinates whose length is the dimension of its affine space.
"""

Inf_Flex = Sequence[Number] | dict[Vertex, Sequence[Number]]
"""
Given a framework in dimension `dim` with `n` vertices. An infinitesimal flex is either
given by a Sequence of `Number` whose length is `dim*n` or by a dictionary from the set
of vertices to a `Sequence` of length `dim`.
"""

Stress = Sequence[Number] | dict[Edge, Number]
"""
Given a framework in dimension with `m` edges. An equilibrium stress is either
given by a Sequence of `Number` whose length is `m` or by a dictionary from the set
of edges to a `Sequence` of `Number`.
"""

Inf = sp.core.numbers.Infinity | np.float32
"""
Provides a data type that can become infinite.
"""


def point_to_vector(point: Point) -> Matrix:
    """
    Return point as single column sympy Matrix.
    """
    if isinstance(point, MatrixBase) or isinstance(point, np.ndarray):
        if (
            len(point.shape) > 1 and point.shape[0] != 1 and point.shape[1] != 1
        ) or len(point.shape) > 2:
            raise ValueError("Point could not be interpreted as column vector.")
        if isinstance(point, np.ndarray):
            point = np.array([point]) if len(point.shape) == 1 else point
            point = Matrix(
                [
                    [float(point[i, j]) for i in range(point.shape[0])]
                    for j in range(point.shape[1])
                ]
            )
        return point if (point.shape[1] == 1) else point.transpose()

    if not isinstance(point, Sequence) or isinstance(point, str):
        raise TypeError("The point must be a Sequence of Coordinates.")

    try:
        res = Matrix(point)
    except Exception as e:
        raise ValueError("A coordinate could not be interpreted by sympify:\n" + str(e))

    if res.shape[0] != 1 and res.shape[1] != 1:
        raise ValueError("Point could not be interpreted as column vector.")
    return res if (res.shape[1] == 1) else point.transpose()
