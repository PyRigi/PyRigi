from collections.abc import Hashable, Iterator
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


class UnionFind:
    """
    Union find data structure implementation

    Note
    ----
    For integers, this can be implemented more efficiently with lists and indexing.
    Uses tree collapsing internally to improve performance.

    Suggested Improvements
    ----------------------
    Use class generics in Python 3.12.
    """

    def __init__(self):
        # Maps used type into ID used for list indexing
        self._data: dict[T, T] = {}

    def __repr__(self) -> str:
        return self._data.__repr__()

    def __str__(self) -> str:
        return self._data.__str__()

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def same_set(self, a: T, b: T) -> bool:
        return self.find(a) == self.find(b)

    def find(self, a: T) -> T:
        """
        Find class for the given item
        """
        # initial recursion end
        if a not in self._data:
            self._data[a] = a
            return a

        while self._data[a] != a:
            self._data[a] = self._data[self._data[a]]
            a = self._data[a]
        return a

    def join(self, a: T, b: T) -> bool:
        """
        Join classes of two given items
        """
        ca, cb = self.find(a), self.find(b)
        if ca == cb:
            return False
        self._data[cb] = ca
        return True

    def root_cnt(self, total: int) -> int:
        """
        Return no. of root nodes

        Parameters
        ----------
        total:
            Total number of the nodes handled by this data structure
        """
        return total - len(self._data)
