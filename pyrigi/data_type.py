"""

Module for defining data type used for type hinting.

"""
from typing import TypeVar, List, Tuple, Hashable, Any, Dict


Vertex = Hashable
"""
Any hashable type can be used for a Vertex.
"""

Edge = Tuple[Vertex, Vertex]
"""
An Edge is a pair of :obj:`Vertices <pyrigi.data_type.Vertex>`.
"""

Point = List[float]
"""
A Point is a list of coordinates whose length is the dimension of its affine space.
"""

GraphType = TypeVar("Graph")
MatroidType = TypeVar("Matroid")
