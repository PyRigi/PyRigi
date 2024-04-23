"""

Module for defining data type used for type hinting.

"""
from typing import TypeVar, List, Tuple, Hashable, Any, Dict


Vertex = Hashable
Edge = Tuple[Vertex, Vertex]
Point = List[float]

GraphType = TypeVar("Graph")
Vertex = Hashable
Edge = Tuple[Vertex, Vertex]
