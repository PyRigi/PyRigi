"""

Module for defining warnings.

"""

from collections.abc import Callable


class RandomizedAlgorithmWarning(UserWarning):
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
                    + f"for the class {class_off.__name__},"
                    + f" use `{class_off.__name__}.silence_rand_alg_warns=True`."
                )
            msg_str += "\n"
            super().__init__(msg_str, *args)
