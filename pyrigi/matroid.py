"""
This is the module for matroid functionality.
"""

from copy import deepcopy

from sympy import Matrix
from networkx import minimum_spanning_tree

from pyrigi.graph import Graph
from pyrigi.data_type import MatroidType


class Matroid(object):

    def __init__(self, _ground_set) -> MatroidType:
        """Initialize the matroid object."""
        raise NotImplementedError()
    
    def __str__(self) -> str:
        return 'Matroid:\t'+str(self._ground_set)

    def ground_set(self):
        """Return the ground set of the matroid."""

        return self._ground_set

    def rank(self, F=None):
        """Compute the rank of a subset of the ground set."""

        if F is None:
            F = self.ground_set()
        raise NotImplementedError()

    def is_independent(self, F):
        """Check whether a given subset of the ground set is independent."""

        return self.rank(F) == len(F)

    def is_dependent(self, F):
        """Check whether a given subset of the ground set is dependent."""

        return not (self.is_independent(F))

    def is_circuit(self, F):
        """Check whether a given subset of the ground set is a circuit."""

        if self.is_independent(F):
            return False
        for a in F:
            FF = deepcopy(F)
            FF.remove(a)
            if self.is_dependent(FF):
                return False
        return True

    def is_basis(self, F):
        """Check whether a given subset of the ground set is a basis."""

        return self.rank(F) == self.rank()

    def is_closed(self):
        raise NotImplementedError()


class LinearMatroid(Matroid):

    def __init__(self, matrix):
        self._matrix = matrix
        self._ground_set = list(range(matrix.rows))

    def rank(self, F=None):

        if F is None:
            F = self.ground_set()
        return self._matrix.row(F).rank()


class GraphicalMatroid(Matroid):

    def __init__(self, graph):
        self._graph = graph
        self._ground_set = graph.edges()

    def rank(self, F=None):

        if F is None:
            F = self.ground_set()
        G = Graph(F)
        return len(minimum_spanning_tree(G).edges())
