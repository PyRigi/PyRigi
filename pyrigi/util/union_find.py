from typing import Dict, Hashable


class UnionFind[T: Hashable]:
    """
    Union find data structure implementation

    Use only with types other then int,
    use implementation without dicts in that case.

    Uses tree collapsing internally to improve performance.
    """

    def __init__(self):
        # Maps used type into id used for list indexing
        self._data: Dict[T, T] = {}

    def __repr__(self) -> str:
        return self._data.__repr__()

    def __str__(self) -> str:
        return self._data.__str__()

    def same_set(self, a: T, b: T) -> bool:
        return self.find(a) == self.find(b)

    def find(self, a: T) -> T:
        # initial recursion end
        if a not in self._data:
            self._data[a] = a
            return a

        # recursion end
        val = self._data[a]
        if val == a:
            return a

        res = self.find(val)
        # used to collapse union find trees
        self._data[a] = res
        return res

    def join(self, a: T, b: T) -> bool:
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
