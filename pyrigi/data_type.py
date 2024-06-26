"""

Module for defining data type used for type hinting.

"""

from sympy import Matrix
from typing import TypeVar, Tuple, Hashable
from collections.abc import Sequence


Vertex = Hashable
"""
Any hashable type can be used for a Vertex.
"""

Edge = Tuple[Vertex, Vertex]
"""
An Edge is a pair of :obj:`Vertices <pyrigi.data_type.Vertex>`.
"""

Coordinate = int | float | str
"""
An integer, float or a string interpretable by :func:`sympy.core.sympify.sympify`.
"""

Point = Sequence[Coordinate]
"""
A Point is a Sequence of Coordinates whose length is the dimension of its affine space.
"""


Vector = TypeVar("Matrix")
GraphType = TypeVar("Graph")
FrameworkType = TypeVar("Framework")
MatroidType = TypeVar("Matroid")


def point_to_vector(point: Point) -> Vector:
    """
    Return point as sympy Matrix.
    """

    res = Matrix(point)
    if res.shape[1] != 1:
        raise ValueError("Point could not be interpreted as column vector.")
    return res
