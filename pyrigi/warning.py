"""

Module for defining warnings.

"""

import warnings
from collections.abc import Callable

import networkx as nx

from pyrigi.data_type import (
    Number,
    Sequence,
    Vertex,
)


class RandomizedAlgorithmWarning(UserWarning):
    """
    Warning raised when randomized algorithm is used without explicit call.
    """

    def __init__(
        self,
        method: Callable,
        msg: str = None,
        explicit_call: str = None,
        class_off: object = None,
        *args,
    ):
        if msg is not None:
            super().__init__(msg, *args)
        else:
            msg_str = (
                f"The method {method.__qualname__} uses a randomized algorithm,"
                + "see its docstring for more information!"
            )
            if explicit_call is not None:
                msg_str += (
                    f"\nIf the parameter `{explicit_call}` is used explicitly,"
                    + "this warning is not displayed."
                )
            if class_off is not None:
                msg_str += (
                    "\nTo switch off all RandomizedAlgorithmWarnings"
                    + f"for the class {class_off.__name__} and all its subclasses,"
                    + f" use `{class_off.__name__}.silence_rand_alg_warns=True`."
                )
            msg_str += "\n"
            super().__init__(msg_str, *args)


class NumericalAlgorithmWarning(UserWarning):
    """
    Warning raised when a numerical algorithm is called, whose output cannot
    be guaranteed to be correct.
    """

    def __init__(
        self,
        method: Callable,
        msg: str = None,
        class_off: object = None,
        *args,
    ):
        if msg is not None:
            super().__init__(msg, *args)
        else:
            msg_str = (
                f"The method {method.__qualname__} uses a numerical algorithm, "
                + "that is not guaranteed to work every time!"
            )
            if class_off is not None:
                msg_str += (
                    "\n\nTo switch off all NumericalAlgorithmWarning"
                    + f"for the class {class_off.__name__} and all its subclasses,"
                    + f" use `{class_off.__name__}.silence_numerical_alg_warns=True`."
                )
            msg_str += "\n"
            super().__init__(msg_str, *args)


def _warn_randomized_alg(
    graph: nx.Graph, method: Callable, explicit_call: str = None
) -> None:
    """
    Raise a warning if a randomized algorithm is silently called.

    Parameters
    ----------
    graph:
        Instance from which the warning is raised.
    method:
        Reference to the method that is called.
    explicit_call:
        Parameter and its value specifying
        when the warning is not raised (e.g. ``algorithm="randomized"``).
    """
    cls = type(graph)

    from pyrigi import Graph

    if isinstance(graph, Graph):
        if not cls.silence_rand_alg_warns:
            warnings.warn(
                RandomizedAlgorithmWarning(
                    method, explicit_call=explicit_call, class_off=cls
                ),
                stacklevel=2,
            )
    else:
        warnings.warn(
            RandomizedAlgorithmWarning(method, explicit_call=explicit_call),
            stacklevel=2,
        )


class NumericalCoordinateWarning(UserWarning):
    """
    Warning raised when numerical coordinates are used in symbolic computation.
    """

    def __init__(
        self,
        realization: dict[Vertex, Sequence[Number]],
        method: Callable,
        msg: str = None,
        class_off: object = None,
        *args,
    ):
        if msg is not None:
            super().__init__(msg, *args)
        else:
            msg_str = (
                "\nNumerical coordinates were detected in the Framework's realization. "
                + "The following points contain numerical coordinates: "
                + f"\n{realization}.\nHowever, the method {method.__qualname__} has "
                + " been set to symbolic computations which are performed via sympy."
                + " The result is therefore not guaranteed to be correct."
                + " Consider using exact coordinates,"
                + " or setting the keyword `numerical=True`, which ensures"
                + " that numpy-based computations are performed instead."
            )
            if class_off is not None:
                msg_str += (
                    "\nTo switch off all NumericalCoordinateWarnings"
                    + f" for the class {class_off.__name__},"
                    + f" use `{class_off.__name__}.silence_numerical_coord_warns=True`."
                )
            msg_str += "\n"
            super().__init__(msg_str, *args)


warnings.filterwarnings("always", category=NumericalCoordinateWarning)
