from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


class RepeatableIterator:
    """
    Wrapper for an iterator that caches all the items yielded
    by the iterator for future iterations.

    Suggested Improvements
    ---------------------
    Use class generics in Python 3.12.
    """

    __slots__ = ("_source", "_cache")

    def __init__(self, iterable: Iterable[T]):
        self._source: Iterator[T] = iter(iterable)
        self._cache: list[T] = []

    def __iter__(self) -> Iterator[T]:
        return self._Cursor(self)

    class _Cursor(Iterator[T]):
        __slots__ = ("_parent", "_index")

        def __init__(self, parent: "RepeatableIterator"):
            self._parent = parent
            self._index = 0

        def __next__(self) -> T:
            parent = self._parent
            cache = parent._cache

            if self._index < len(cache):
                item = cache[self._index]
            else:
                try:
                    item = next(parent._source)
                    cache.append(item)
                except StopIteration:
                    raise StopIteration from None

            self._index += 1
            return item
