from typing import Callable, ParamSpec, TypeVar, TypeVarTuple

P = ParamSpec("P")
T = TypeVar("T")
R = TypeVarTuple("R")


def copy_doc(
    proxy_func: Callable[P, T],
) -> Callable[[Callable[..., T]], Callable[P, T]]:
    """
    Copy the docstring from the provided function.

    In tests, it also ensures that the signatures match.
    """

    def wrapped(method: Callable[..., T]) -> Callable[P, T]:
        method.__doc__ = proxy_func.__doc__
        method._wrapped_func = proxy_func
        return method

    return wrapped
