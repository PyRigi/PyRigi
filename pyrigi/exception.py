class LoopError(ValueError):
    def __init__(self, msg: str = "The graph needs to be loop-free.", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class DimensionValueError(ValueError):
    """
    Error when the dimension is not a positive integer.
    """
    def __init__(self, dim, *args, **kwargs):
        super().__init__("The dimension needs to be a positive integer, but is "+str(dim)+"!", *args, **kwargs)


class DimensionCombinatorialValueError(ValueError):
    """
    Error when the dimension is not in the range of a combinatorial computation.
    """
    def __init__(self, dim_expected, dim, *args, **kwargs):
        if isinstance(dim_expected, int):
            str_expected = str(dim_expected)
        elif isinstance(dim_expected, list):
            str_expected = "in "+str(dim_expected)
        super().__init__("The combinatorial computation is only available for dimension "+str_expected+", but is "+str(dim)+"!", *args, **kwargs)
