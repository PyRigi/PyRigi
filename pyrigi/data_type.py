"""

Module for defining data type used for type hinting.

"""
from typing import TypeVar, List, Tuple, Hashable, Any, Dict


Vertex = Hashable
Edge = Tuple[Vertex, Vertex]
Point = List[float]

GraphType = TypeVar("Graph")
MatroidType = TypeVar("Matroid")
Vertex = Hashable
Edge = Tuple[Vertex, Vertex]
