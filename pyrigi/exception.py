class LoopError(ValueError):
    def __init__(self, msg: str = "The graph needs to be loop-free.", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)


class DimensionCombinatorialValueError(ValueError):
    """
    Error when the dimension is not in the range of a combinatorial computation.
    """

    def __init__(self, dim_expected, dim, *args, **kwargs):
        if isinstance(dim_expected, int):
            str_expected = str(dim_expected)
        elif isinstance(dim_expected, list):
            str_expected = "in " + str(dim_expected)
        super().__init__(
            "The combinatorial computation is only available for dimension "
            + str_expected
            + ", but is "
            + str(dim)
            + "!",
            *args,
            **kwargs,
        )


class NonNegativeIntParameterValueError(ValueError):
    """
    Error when a parameter is not a nonnegative integer.
    """

    def __init__(self, param, name: str = "k ", *args, **kwargs):
        super().__init__(
            name + f" needs to be a nonnegative integer, but is {param}!",
            *args,
            **kwargs,
        )
