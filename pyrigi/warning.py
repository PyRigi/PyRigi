"""

Module for defining exceptions and warnings.

"""


class RandomizedAlgorithmWarning(UserWarning):
    def __init__(self, msg: str = None, suppression: str = None, *args, **kwargs):
        if msg is not None:
            super().__init__(msg, *args, **kwargs)
        else:
            msg_str = "A randomized algorithm is used!"
            if suppression is not None:
                msg_str += f" Use `{suppression}` to suppress this warning."
            super().__init__(msg_str, *args, **kwargs)
