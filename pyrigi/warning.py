"""

Module for defining warnings.

"""

from collections.abc import Callable


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
                + "that is not guaranteed to work everytime!"
            )
            if class_off is not None:
                msg_str += (
                    "\nTo switch off all NumericalAlgorithmWarning"
                    + f"for the class {class_off.__name__} and all its subclasses,"
                    + f" use `{class_off.__name__}.silence_numerical_alg_warns=True`."
                )
            msg_str += "\n"
            super().__init__(msg_str, *args)
