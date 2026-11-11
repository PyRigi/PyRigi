from typing import Iterable, Iterator


class RepeatableIterator[T]:
    """
    Wrapper for an iterator that caches all the items yielded
    by the iterator for future iterations.
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
