class LoopError(ValueError):
    def __init__(self, msg: str = "The graph needs to be loop-free.", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)


class NonSupportedParameterError(ValueError):
    def __init__(
        self,
        wrong_param,
        parameter_name: str,
        method_name: str,
        msg: str = None,
        *args,
        **kwargs,
    ):
        if msg is not None:
            super().__init__(msg, *args, **kwargs)
        else:
            msg_str = (
                f"The specified value of the `{parameter_name}` parameter is '{wrong_param}', "
                + "which is not supported!\n"
            )
            if method_name is not None:
                msg_str += f"Call `help({method_name})` for further information"
        super().__init__(msg_str, *args, **kwargs)
