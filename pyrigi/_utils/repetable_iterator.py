from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


class RepeatableIterator(Iterator[T]):
    """
    Wrapper for an iterator that caches all the items yielded by the iterator
    in the first pass for future passes.
    The iterator must be exhausted the first time it is iterated.

    Suggested Improvements
    ---------------------
    Use class generics in Python 3.12.
    """

    def __init__(self, iterable: Iterable[T]):
        if isinstance(iterable, list):
            self._is_first = False
            self._cache: list[T] = iterable
            return

        self._iterable = iter(iterable)
        self._is_first = True
        self._cache: list[T] = []

    def __iter__(self) -> Iterator[T]:
        if self._is_first:
            self._is_first = False
            return self
        else:
            return iter(self._cache)

    def __next__(self) -> T:
        item = next(self._iterable)
        self._cache.append(item)
        return item
