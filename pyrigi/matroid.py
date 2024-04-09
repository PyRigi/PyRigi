"""
This is the module for matroid functionality.
"""

from sympy import Matrix
from networkx import minimum_spanning_tree

class Matroid(object):

    def __init__(self, family: str, aux):
        """Initialize the matroid object."""

        if family == "linear":
            self.family = family
            self._matrix = aux
            self._ground_set = list(range(aux.rows))
        elif family == "graphical":
            self.family = family
            self._graph = aux
            self._ground_set = aux.edges()
        else:
            raise NotImplementedError()

    def ground_set(self):
        """Return the ground set of the matroid."""

        return self._ground_set

    def rank(self, F=None):
        """Compute the rank of a subset of the ground set."""

        if F is None:
            F = self.ground_set()
        if self.family == "linear":
            return self._matrix.row(F).rank()
        elif self.family == "graphical":
            G = Graph(F)
            return len(minimum_spanning_tree(G).edges())
        raise NotImplementedError()

    def is_independent(self, F):
        """Check whether a given subset of the ground set is independent."""

        if self.family == "linear" or self.family == "graphical":
            return self.rank(F) == len(F)
        raise NotImplementedError()

    def is_dependent(self, F):
        """Check whether a given subset of the ground set is dependent."""

        if self.family == "linear" or self.family == "graphical":
            return not(self.is_independent(F))
        raise NotImplementedError()

    def is_circuit(self, F):
        """Check whether a given subset of the ground set is a circuit."""

        if self.family == "linear" or self.family == "graphical":
            return self.rank(F) == len(F) - 1
        raise NotImplementedError()

    def is_basis(self, F):
        """Check whether a given subset of the ground set is a basis."""

        if self.family == "linear" or self.family == "graphical":
            return self.rank(F) == self.rank()

    def is_closed(self):
        raise NotImplementedError()
