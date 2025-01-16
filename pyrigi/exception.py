class LoopError(ValueError):
    def __init__(self, msg: str = "The graph needs to be loop-free.", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class DimensionValueError(ValueError):
    """
    Error when the dimension is not a positive integer.
    """
    def __init__(self, dim, *args, **kwargs):
        super().__init__("The dimension needs to be a positive integer, but is "+str(dim)+"!", *args, **kwargs)
